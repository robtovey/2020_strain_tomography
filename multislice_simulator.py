'''
Created on 27 Apr 2020

@author: Rob Tovey
'''
from numpy import (unique, array, empty, ones, zeros, exp, arange, sqrt,
    ascontiguousarray, logical_and, isscalar, linspace)
from crystal import (FTYPE, CTYPE, _pointwise_exp, c_exp, c_erf, ceil, pi, LOG0,
    getA, floor, __CPUatom, __atom)
from utils import (rebin, toc, progressBar, rot3d, isLoud, SILENT, USE_CPU,
    plan_fft, plan_ifft, fftshift)
import numba
from numba import cuda
from FFT_simulator import getBessel, precess


def doPrecomp(x, d, z, dim):
    A, precomp, pms, Rmax_out = [], [], [], []

    for zz in z:
        a, b = getA(zz, False)
        A.append(((a * (4 * pi * b) ** (-dim / 2)).astype(FTYPE),
              (1 / (4 * b)).astype(FTYPE)))
        a, b = A[-1]

        if _pointwise_exp:
            f = lambda x: sum(a[i] * c_exp(-b[i] * x ** 2) for i in range(a.size))
            Rmax = 1
            while f(Rmax) > exp(LOG0) * f(0):
                Rmax *= 1.1
            h = max(Rmax / 200, max(d[1:]) / 10)
            pms.append(array([Rmax ** 2, 1 / h], dtype=FTYPE))
            precomp.append(array([f(x) for x in arange(0, Rmax + 2 * h, h)], dtype=FTYPE))
            Rmax_out.append(Rmax)

        else:

            def f(i, j, x):
                A = a[i] ** (1 / 3)  # factor spread evenly over 3 dimensions
                B = b[i] ** .5
                if d[j] == 0:
                    return A * 2 / B
                return A * (c_erf(B * (x + d[j] / 2))
                            -c_erf(B * (x - d[j] / 2))) / (2 * d[j] * B) * pi ** .5

            h = [D / 10 for D in d]
            Rmax = ones([a.size, 3], dtype=FTYPE)
            L = 1
            for i in range(a.size):
                for j in range(3):
                    if d[j] == 0:
                        Rmax[i, j] = 1e5
                        continue
                    while f(i, j, Rmax[i, j]) > exp(LOG0) * f(i, j, 0):
                        Rmax[i, j] *= 1.1
                        L = max(L, Rmax[i, j] / h[j] + 2)
            L = min(200, int(ceil(L)))
            grid = arange(L)
            precomp.append(zeros([a.size, 3, L], dtype=FTYPE))
            for i in range(a.size):
                for j in range(3):
                    h[j] = Rmax[:, j].max() / (L - 2)
                    precomp[-1][i, j] = array([f(i, j, x * h[j]) for x in grid], dtype=FTYPE)
            pms.append(array([Rmax.max(0), 1 / array(h)], dtype=FTYPE))
            Rmax_out.append(Rmax.max(0).min())

        # return thickness integral rather than average
        precomp[-1] *= d[2] if _pointwise_exp else d[2] ** (1 / 3)
    return A, precomp, pms, Rmax_out


@numba.jit(cache=True)
def __countbins(x0, x1, loc, r, s, Len):
    for j0 in range(loc.shape[0]):
        bin0 = int((loc[j0, 0] - x0) / r[0])
        bin1 = int((loc[j0, 1] - x1) / r[1])
        for i in range(max(0, bin0 - s), min(Len.shape[0], bin0 + s + 1)):
            for j in range(max(0, bin1 - s), min(Len.shape[1], bin1 + s + 1)):
                Len[i, j] += 1


@numba.jit(cache=True)
def __rebin(x0, x1, loc, sublist, r, s, Len):
    for j0 in range(loc.shape[0]):
        bin0 = int((loc[j0, 0] - x0) / r[0])
        bin1 = int((loc[j0, 1] - x1) / r[1])
        for i in range(max(0, bin0 - s), min(Len.shape[0], bin0 + s + 1)):
            for j in range(max(0, bin1 - s), min(Len.shape[1], bin1 + s + 1)):
                sublist[i, j, Len[i, j]] = j0
                Len[i, j] += 1

    for b0 in range(sublist.shape[0]):
        for b1 in range(sublist.shape[1]):
            j0 = Len[b0, b1]
            if j0 < sublist.shape[2]:
                sublist[b0, b1, j0] = -1


def rebin2D(x, loc, r, k):
    if isscalar(r):
        r = array([r, r, r], dtype='f4')
    else:
        r = array(r).copy()
    xmin = array([X.item(0) if X.size > 1 else -1e5 for X in x], dtype=x[0].dtype)
    nbins = [int(ceil((x[i].item(-1) - x[i].item(0)) / r[i])) + 1
             for i in range(2)]
    Len = zeros(nbins, dtype='i4')
    __countbins(xmin[0], xmin[1], loc, r, k, Len)

    L = Len.max()
    subList = zeros(nbins + [L], dtype='i4')
    Len.fill(0)
    __rebin(xmin[0], xmin[1], loc, subList, r, k, Len)

    return subList


def doBinning(x, loc, z, Rmax, d):
    '''
    For each atom type:
        First pass: bin into slices
        Second pass: bin into pixels

    # TODO: should block in z so we don't duplicate too many atoms
    # TODO: should be much more memory checks surrounding this function...
    '''

    if len(z) == 1:
        loc = [loc]
    else:
        loc = [ascontiguousarray(loc[c.Z == zz, :]) for zz in z]  # split by atom type

    newLoc, subList = [], []

    for i, atom in enumerate(loc):
        newLoc.append([])
        subList.append([])
        for z in x[2]:  # Find each atom in slice
            ind = logical_and(atom[:, 2] > z - Rmax[i],
                              atom[:, 2] < z + d[2] + Rmax[i])
            newLoc[-1].append(atom[ind])

            # Bin the atoms
            k = 3  # each atom in ~9 sub-pixels
            r = array([2e5 if D == 0 else max(Rmax[i] / k, D) for D in d], dtype='f4')
            subList[-1].append(rebin2D(x[:2], newLoc[-1][-1], r, k))

    return newLoc, subList, r


def doGrid(sz, tpb=None):
    dim = len(sz)

    tpb = 16 if tpb is None else tpb
    tpb = [tpb] * len(sz)
    grid = [0] * len(sz)

    for i in range(dim):
        if tpb[i] > sz[i]:
            tpb[i] = sz[i]
            grid[i] = 1
        else:
            while tpb[i] * (sz[i] // tpb[i]) != sz[i]:
                tpb[i] -= 1
            grid[i] = sz[i] // tpb[i]

    return grid, tpb


@numba.jit(parallel=True, fastmath=True, nopython=True)
def __CPU_density(x0, x1, x2, xmin,
                  loc, sublist, r, a, d, B, precomp, h, out):
    X2 = x2
    for i0 in numba.prange(x0.size):
        X0 = x0[i0]
        bin0 = int(floor((X0 - xmin[0]) / r[0]))
        if bin0 < 0 or bin0 >= sublist.shape[0]:
            continue
        for i1 in range(x1.size):
            X1 = x1[i1]
            bin1 = int(floor((X1 - xmin[1]) / r[1]))
            if bin1 < 0 or bin1 >= sublist.shape[1]:
                continue

            Sum = 0
            for bb in range(sublist.shape[2]):
                j0 = sublist[bin0, bin1, bb]
                if j0 < 0:
                    break

                Y0 = loc[j0, 0] - X0
                Y1 = loc[j0, 1] - X1
                Y2 = loc[j0, 2] - X2
                Sum += __CPUatom(a, B, Y0, Y1, Y2, d, precomp, h)
            out[i0, i1] += Sum


@cuda.jit
def __GPU_density(x0, x1, x2, xmin,
                  loc, sublist, r, a, d, B, precomp, h, out):
    i0, i1 = cuda.grid(2)
    if i0 >= x0.size or i1 >= x1.size:
        return
    X0, X1, X2 = x0[i0], x1[i1], x2
    Y0, Y1, Y2 = 0, 0, 0

    bin0 = int((X0 - xmin[0]) / r[0])
    bin1 = int((X1 - xmin[1]) / r[1])
    if bin0 >= sublist.shape[0] or bin1 >= sublist.shape[1]:
        return

    Sum = 0
    for bb in range(sublist.shape[2]):
        j0 = sublist[bin0, bin1, bb]
        if j0 < 0:
            break
        Y0 = loc[j0, 0] - X0
        Y1 = loc[j0, 1] - X1
        Y2 = loc[j0, 2] - X2
        Sum += __atom(a, B, Y0, Y1, Y2, d, precomp, h)
    out[i0, i1] += Sum


def compute_slice_density(i, x, density, xmin, loc, subList,
                          r, A, d, precomp, pms, GPU, **_):
    density.fill(0)
    notComputed = True
    if GPU and not USE_CPU:
        grid, tpb = doGrid(density.shape, 16)

        try:
            D = cuda.to_device(density)
            for a in range(len(loc)):
                __GPU_density[grid, tpb](
                    x[0], x[1], x[2][i], xmin,
                    loc[a][i], subList[a][i], r, A[a][0], d, sqrt(A[a][1]),
                    precomp[a], pms[a], D)
            D.copy_to_host(density)
            notComputed = False
        except Exception:
            USE_CPU(True)

    if notComputed:
        raise
        for a in range(len(loc)):
            # Sum contributions from each atom
            # density = \int_{x0}^{x0+d[0]} [electrostatic potential]
            __CPU_density(x[0], x[1], x[2][i], xmin,
                          loc[a][i], subList[a][i], r, A[a][0], d, sqrt(A[a][1]),
                          precomp[a], pms[a], density)


def getDiscretisation(c, x, GPU=True):
    loc, z = c.loc, unique(c.Z).astype(int)
    dim = loc.shape[-1]

    # Precompute extra variables
    d = array([abs(X.item(1) - X.item(0)) if X.size > 1 else 0 for X in x])
    xmin = array([X.item(0) if X.size > 1 else -1e5 for X in x], dtype=FTYPE)
    A, precomp, pms, Rmax = doPrecomp(x, d, z, dim)
    loc, subList, r = doBinning(x, loc, z, Rmax, d)

    density = empty([X.size for X in x], dtype=FTYPE)

    if isLoud():
        print('Starting discretisation ...', flush=True)

    tic = toc()
    for i in range(x[2].size):
        compute_slice_density(i, x, density[..., i], xmin, loc, subList,
                          r, A, d, precomp, pms, GPU)
        progressBar(i + 1, x[2].size, tic)
    if isLoud():
        print('finished.')

    return density


def Multislice_farfieldwave(x, DP, density, buf, ft, ift, propagator, **extra):
    # Kirkland suggests band-limiting the data:
#     from FourierTransform import toFreq
#     y = toFreq(x)
#     BW = fftshift((y[0].reshape(-1, 1) ** 2 + y[1].reshape(1, -1) ** 2 < 2 * y[0].max() ** 2 * (2 / 3) ** 2))
#     propagator *= BW  # <-- band-width suppression
#     from utils import fftn, ifftn

    def inplace_op(F, A):
        if A is F.output_array:
            F()
        else:
            A[...] = F()

    if isLoud():
        print('Starting multislice ...', flush=True)

    tic = toc()
    for i in range(x[2].size - 1):
        if density.ndim == 2:
            compute_slice_density(i, x, density, **extra)
#             DP *= exp((1j * extra['sigma']) * ifftn(fftn(density) * BW).real)  # <-- band-width suppression
            DP *= exp((1j * extra['sigma']) * density)
        else:
            DP *= exp((1j * extra['sigma']) * density[i])
        inplace_op(ft, buf)  # buf = FT(DP)
        buf *= propagator
        inplace_op(ift, DP)  # DP = IFT(buf)

        progressBar(i + 1, x[2].size - 1, tic)
    if isLoud():
        print('finished.')

    inplace_op(ft, buf)  # buf = FT(DP)
    return fftshift(buf)


def getDiffractionPattern(c, x, y, probewidth, E, GPU=True):
    loc, z = c.loc, unique(c.Z).astype(int)
    dim = loc.shape[-1]

    # Precompute extra variables
    extra = {'GPU':GPU}
    extra['d'] = array([abs(X.item(1) - X.item(0)) if X.size > 1 else 0 for X in x])
    extra['xmin'] = array([X.item(0) if X.size > 1 else -1e5 for X in x], dtype=FTYPE)
    extra.update({k:v for k, v in zip(('A', 'precomp', 'pms', 'Rmax'),
                                      doPrecomp(x, extra['d'], z, dim))})
    extra.update({k:v for k, v in zip(('loc', 'subList', 'r'),
                                      doBinning(x, loc, z, extra['Rmax'], extra['d']))})

    wavelength = 12.3984244 / sqrt(E * (2 * 510.99906 + E))
    extra['sigma'] = 2 * pi * (510.99906 + E) / (2 * 510.99906 + E) / (wavelength * E)
    propagator = exp(-(1j * wavelength / (4 * pi))  # scaling
                    * (y[0].reshape(-1, 1) ** 2 + y[1].reshape(1, -1) ** 2)  # Laplace operator
                     * extra['d'][2])  # thickness
    propagator = fftshift(propagator)

    density = zeros([X.size for X in x[:2]], dtype=FTYPE)
    DP = zeros([X.size for X in x[:2]], dtype=CTYPE)

    ft, DP = plan_fft(DP, overwrite=True, planner=1)
    ift, buf = plan_ifft(ft.output_array, overwrite=True, planner=1)

    b = getBessel(probewidth)[0]
    b(x[:2], out=DP)  # initialise wave

    return abs(Multislice_farfieldwave(x, DP, density, buf, ft, ift, propagator, **extra)) ** 2


def getPrecessedDiffractionPattern(c, x, y, probewidth, E, alpha, ntheta, GPU=True):
    z = unique(c.Z).astype(int)
    dim = c.loc.shape[-1]

    # Precompute extra variables
    extra = {'GPU':GPU}
    extra['d'] = array([abs(X.item(1) - X.item(0)) if X.size > 1 else 0 for X in x])
    extra['xmin'] = array([X.item(0) if X.size > 1 else -1e5 for X in x], dtype=FTYPE)
    extra.update({k:v for k, v in zip(('A', 'precomp', 'pms', 'Rmax'),
                                      doPrecomp(x, extra['d'], z, dim))})

    wavelength = 12.3984244 / sqrt(E * (2 * 510.99906 + E))
    extra['sigma'] = 2 * pi * (510.99906 + E) / (2 * 510.99906 + E) / (wavelength * E)
    extra['propagator'] = exp(-(1j * wavelength / (4 * pi))  # scaling
                    * (y[0].reshape(-1, 1) ** 2 + y[1].reshape(1, -1) ** 2)  # Laplace operator
                     * extra['d'][2])  # thickness
    extra['propagator'] = fftshift(extra['propagator'])
    # TODO: should be able to modify propagator to mimick tilt

    density = zeros([X.size for X in x[:2]], dtype=FTYPE)
    DP = zeros([X.size for X in x[:2]], dtype=CTYPE)

    extra['ft'], DP = plan_fft(DP, overwrite=True, planner=1)
    extra['ift'], buf = plan_ifft(extra['ft'].output_array, overwrite=True, planner=1)

    b = getBessel(probewidth)[0]

    if alpha == 0 or ntheta == 1:
        b(x[:2], out=DP)  # initialise wave
        extra.update({k:v for k, v in zip(('loc', 'subList', 'r'),
                                          doBinning(x, c.loc, z, extra['Rmax'], extra['d']))})
        return abs(Multislice_farfieldwave(x, DP, density, buf, **extra)) ** 2

    else:
        out = zeros([X.size for X in x[:2]], dtype=FTYPE)
        theta = linspace(0, 360, ntheta, endpoint=False)
        if isLoud():
            print('Starting precession ...', flush=True)
        with SILENT() as CC:
            tic = toc()
            for i in range(ntheta):
                R = precess(None, alpha, theta[i])
                b(x[:2], out=DP)  # initialise wave
                extra.update({k:v for k, v in zip(('loc', 'subList', 'r'),
                                                  doBinning(x, c.loc.dot(R.T), z, extra['Rmax'], extra['d']))})
                out += abs(Multislice_farfieldwave(x, DP, density, buf, **extra)) ** 2

                progressBar(i + 1, ntheta, tic, context=CC)
        if isLoud():
            print('finished.', flush=True)

        return out / ntheta


from os.path import join
file_prefix = join('sim', 'multislice_test')

if __name__ == '__main__':

    TEST = False

    if TEST:
        from matplotlib import pyplot as plt
        from FFT_simulator import getFTpoints, getSiBlock, crystal, doPrecession, getDFT
        from numpy import log

        rep, alpha, ntheta = 30, 0, 1
        spacing, probewidth, E = 5.4, 3, 100
        wavelength = 12.3984244 / sqrt(E * (2 * 510.99906 + E))

        Rad = 2 * pi / wavelength  # units of keV
        n, rX = 512, [10 * spacing] * 2 + [300]

        sz = spacing * getSiBlock(1, rotate=111, justwidth=True)
        c = crystal(spacing * getSiBlock(1, rotate=111), 14, sz)
        c = c.repeat(rep)
        c = c - c.box[:, 1] / 2

        x, y = getFTpoints(3, n=n, rX=rX)
        dft3, idft3 = getDFT(x, y);

        multislice = getDiffractionPattern(c, x, y, probewidth, E, GPU=True)
        multislice /= multislice.max()

        DP, _, flatDP = doPrecession(dft3(c(x)), probewidth, x, y, Rad, alpha, ntheta, False)
        DP /= DP.max(); flatDP /= flatDP.max()

        scale = lambda A:log(1e-20 + A)

        plt.subplot(121)
        plt.imshow(scale(multislice), cmap='gray', vmin=log(1e-3),
                   extent=[x[0][0], x[0][-1], x[1][0], x[1][-1]])
        plt.title('Multislice')
        plt.subplot(122)
        plt.imshow(scale(DP), cmap='gray', vmin=log(1e-5),
                   extent=[y[0][0], y[0][-1], y[1][0], y[1][-1]])
        plt.title('Precessed Ewald sphere')
        plt.show()
        exit()
    else:

        from FourierTransform import getDFT, getFTpoints
        from accuracy_test import x, y, rX, probe, rad, nTheta, params2crystal, random, save
        from scipy.io import savemat
        from numpy import concatenate
        E = 300 if rad == 320 else 100  # Map from Ewald sphere radius to energy
        n = [X.size for X in x]
        filename = file_prefix + '_crystal'

        dimList, stackList, duplicates, max_strain = [1, 2, 3], [1, 3, 10], 1, .01

        for rotate in [True, False]:

            def params2arrs(dim, layers, strain):
                c, s = params2crystal(layers, rX, strain, dim, rotate)
                loc = concatenate([cc.loc for cc in c._sum], axis=0)
                ind = (abs(loc[:, 0]) <= box[0] / 2) * (abs(loc[:, 1]) <= box[1] / 2) * (abs(loc[:, 2]) <= box[2] / 2)
                loc = loc[ind]

                loc = loc[loc[:, 2].argsort(kind='mergsort')]  # sort by z-value
                loc += box / 2
                arr = zeros((loc.shape[0], 8), dtype=float)
                arr[:, 0] = 14; arr[:, 1:4] = loc; arr[:, 5] = 1

                return arr, s

            rX = [180, 180, 2 * 555]
            box = array([200, 200, 1000], dtype=float)
            a, s = params2arrs(1, 1, 0)

            store = {'arr0':a, 'box':box}
            print('finished dimension 0')
            for i in (1, 2, 3):
                for j in (1, 3, 10):
                    a, s = params2arrs(i, j, max_strain)
                    store['arr' + str(i) + str(j)] = a
                    store['strain' + str(i) + str(j)] = s
                print('finished dimension ' + str(i))

            savemat(filename + ('_flipped' if rotate else '') + '.mat', mdict=store, do_compression=True)
            print('finished saving')
        exit()

########################################

        alpha = 1
        ft3, ift3 = getDFT(x, y)
        n = [X.size for X in x]
        filename = file_prefix + '_' + str(n[0]) + '_' + str(alpha)

        dimList, stackList, duplicates, max_strain = [1], [1, 3, 10], 5, .01
        rotate = True
#         dimList = 0

        if dimList == 0 or dimList == [0]:
            c, s = params2crystal(1, rX[:2] + [rX[2] - 20], 0, 1, rotate)

            flat = getPrecessedDiffractionPattern(c, x, y, probe, E, 0, 1)
            flat /= flat.max()
            precessed = getPrecessedDiffractionPattern(c, x, y, probe, E, alpha, nTheta)
            precessed /= precessed.max()

            save(filename + '_0' + ('_flipped' if rotate else ''), dict(precessed=precessed, flat=flat))
            print('\n\nComplete\n'); exit()

        count = [0, len(stackList) * duplicates]
        with SILENT() as CC:
            for i, I in enumerate(dimList):
                random.seed(0)
                count[0], sList, tic = 0, [], toc()
                precessed = empty((1, len(stackList), duplicates, n[0], n[1]), dtype='float32')
                flat = empty(precessed.shape, dtype='float32')

                for j, J in enumerate(stackList):
                    for k in range(duplicates):
                        c, s = params2crystal(J,
                                             # extends beyond in [x,y] but zeropadded in z
                                             [1.2 * rX[0], 1.2 * rX[1], rX[2] - 20],
                                             max_strain, I, rotate)
                        sList.append(s)
                        flat[0, j, k] = getPrecessedDiffractionPattern(c, x, y, probe, E, 0, 1)
                        flat[0, j, k] /= flat[0, j, k].max()
                        precessed[0, j, k] = getPrecessedDiffractionPattern(c, x, y, probe, E, alpha, nTheta)
                        precessed[0, j, k] /= precessed[0, j, k].max()
                        count[0] += 1
                        progressBar(count[0], count[1], tic, context=CC)

                save(filename + '_' + str(I) + ('_flipped' if rotate else ''),
                    dict(precessed=precessed, flat=flat, strain=sList, stackList=stackList))

        print('\n\nComplete\n')

