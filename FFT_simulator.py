'''
Created on, 5 Oct, 2018

@author: Rob Tovey

define u = \sum_i a_i\star \delta_{x_i} is the atomic density
map where a_i is the 'shape' of atom iand x_i is its location.
Assume a_i is given by the paper 'Robust Parameterization of
Elastic and Absorptive Electron Atomic Scattering Factors' by
Peng, Ren, Dudarev and Whelen, basically a sum of, 5 Gaussians.

Forward model is then u \mapsto |\psi \star P[F[u]]|^2
where P projects the, 3D Fourier transform to a, 2D hyperplane.
\psi is dictated by the probe shape, \psi(k|x_0) = F[\Psi(x-x_0)]
and \Psi(r) = Bessel_1(r)/r.
'''
from numpy import (pi, sqrt, zeros, linspace, empty, array, cos,
                   sin, concatenate, logical_and, log10, maximum, minimum, log,
                   cross, abs, arange, multiply, ones)
from scipy.special import jv
from math import sqrt as c_sqrt, inf
from scipy.ndimage.interpolation import rotate, affine_transform
from scipy.interpolate import interpn
import numba
from numba import cuda
from utils import toMesh, toc, progressBar, convolve, SILENT, USE_CPU, savemat, loadmat, \
    fast_abs, plan_ifft, isLoud
from utils import fftn, ifftn, fftshift, ifftshift, plan_fft, fftshift_phase
from FourierTransform import getFTpoints, getDFT, translateFT, fromFreq
from crystal import getDiscretisation, getSiBlock, crystal
FTYPE, CTYPE = 'f8', 'c16'

pass
##################################################
# Low level functions
##################################################

pass
##################################################
# Other helper functions
##################################################


def getBessel(r):
    '''
    bess(x) = J_1(|x[:2]|)/|x[:2]|

    FTbess(y) = \delta(y_3==0) \chi(|y[:2]|<=1)
    '''
    r = r / 3.83170597020751

    @numba.jit(nopython=True, parallel=True, fastmath=True)
    def __bess(X, R, H, J, scale, out):
        if scale.size == 1:
            for i in numba.prange(X.shape[0]):
                rad = c_sqrt(X[i, 0] * X[i, 0] + X[i, 1] * X[i, 1]) * R
                ind = int(rad * H)
                if ind < J.size:
                    out[i] = J[ind]
                else:
                    out[i] = 0
        else:
            for i in numba.prange(X.shape[0]):
                rad = c_sqrt(X[i, 0] * X[i, 0] + X[i, 1] * X[i, 1]) * R
                ind = int(rad * H)
                if ind < J.size:
                    out[i] = scale[i] * J[ind]
                else:
                    out[i] = 0

    def bess(x, out=None, scale=None):
        if not(hasattr(x, 'shape')):
            x = toMesh(x)
        scale = ones(1, dtype=x.dtype) if scale is None else scale
        if out is None:
            out = empty(x.shape[:-1], dtype=scale.dtype)
        if x.shape[-1] == 1 or x.ndim == 1:
            x = maximum(1e-16, abs(x))
            out[...] = jv(1, x) / x * scale
        elif x.shape[-1] == 2:
            x = x[..., :2] / r
            x = maximum(1e-16, sqrt(abs(x * x).sum(-1)))
            out[...] = jv(1, x) / x
        else:
            d = abs(x[1, 1, 0, :2] - x[0, 0, 0, :2])
            h = d.min() / 10
            s = ((d[0] * x.shape[0]) ** 2 + (d[1] * x.shape[1]) ** 2) ** .5

            fine_grid = arange(h / 2, s / r + h, h)
            j = jv(1, fine_grid) / fine_grid

            __bess(x.reshape(-1, 3), 1 / r, 1 / h, j, scale.reshape(-1), out.reshape(-1))
        return out

    @numba.jit(nopython=True, parallel=True, fastmath=True)
    def __bessFT(X, R, s, eps, out):
        for i in numba.prange(X.shape[0]):
            rad = X[i, 0] * X[i, 0] + X[i, 1] * X[i, 1]
            if rad > R or abs(X[i, 2]) > eps:
                out[i] = 0
            else:
                out[i] = s

    def bessFT(x, out=None):
        if not(hasattr(x, 'shape')):
            x = toMesh(x)
        if x.shape[-1] == 1 or x.ndim == 1:
            x = x * r
            x[abs(x) > 1] = 1
            if out is None:
                out = (2 * r) * sqrt(1 - x * x)
            else:
                out[...] = (2 * r) * sqrt(1 - x * x)
        else:
            if x.shape[-1] == 3:
                dx2 = []
                for i in range(x.ndim - 1):
                    tmp = tuple(0 if j != i else 1 for j in range(x.ndim - 1)) + (2,)
                    try:
                        dx2.append(abs(x[tmp] - x[..., 2].item(0)))
                    except Exception:
                        dx2.append(1)
                eps = max(1e-16, max(dx2) * .5)
                if out is None:
                    out = empty(x.shape[:3], dtype=x.dtype)

                __bessFT(
                    x.reshape(-1, 3), 1 / r ** 2, 2 * pi * r ** 2, eps, out.reshape(-1))

            else:
                if out is None:
                    out = (2 * pi * r ** 2) * (abs(x * x).sum(-1) <= 1 / r ** 2)
                else:
                    out[...] = (2 * pi * r ** 2) * (abs(x * x).sum(-1) <= 1 / r ** 2)
        return out

    return bess, bessFT


def precess(baseFT, alpha, theta):
    #     def crotate(arr, *args, **kwargs):
    #         arr.real = rotate(arr.real, *args, **kwargs)
    #         arr.imag = rotate(arr.imag, *args, **kwargs)
    #         return arr
    #     newFT = baseFT.copy()
    #     newFT = crotate(newFT, theta, axes=(1, 0), reshape=False, order=3)
    #     newFT = crotate(newFT, alpha, axes=(1, 2), reshape=False, order=3)
    #     newFT = crotate(newFT, -theta, axes=(1, 0), reshape=False, order=3)
    #     return newFT

    if alpha == 0:
        return array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) if baseFT is None else baseFT

    alpha, theta = alpha * pi / 180, theta * pi / 180
    R_a = array([[1, 0, 0], [0, cos(alpha), -sin(alpha)],
                 [0, sin(alpha), cos(alpha)]])
    R_t = array([[cos(theta), -sin(theta), 0],
                 [sin(theta), cos(theta), 0], [0, 0, 1]])
    R = (R_t.T.dot(R_a.dot(R_t)))
    if baseFT is None:
        return R

    c = array(baseFT.shape) / 2
    newFT = affine_transform(
        baseFT.real, R.T, offset=c - R.T.dot(c), order=3)
    newFT = newFT.astype(CTYPE)
    newFT.imag = affine_transform(
        baseFT.imag, R.T, offset=c - R.T.dot(c), order=3)
    return newFT


def grid2sphere(arr, x, dx, C):
    '''
    dx is an orthonomal basis
    '''
    if C is None or x[2].size == 1:
        if arr.ndim == 2:
            return arr
        elif arr.shape[2] == 1:
            return arr[:, :, 0]

    y = toMesh((x[0], x[1], array([0])), dx).reshape(-1, 3)
#     if C is not None:
#         w = C - sqrt(maximum(0, C ** 2 - (y ** 2).sum(-1)))
#         if dx is None:
#             y[:, 2] = w.reshape(-1)
#         else:
#             y += w.reshape(y.shape[0], 1) * dx[2].reshape(1, 3)

    if C is not None:
        w = 1 / (1 + (y ** 2).sum(-1, keepdims=True) / C ** 2)
        y *= w
        if dx is None:
            y[:, 2] = C * (1 - w)
        else:
            y += C * (1 - w) * dx[2]

    out = interpn(x, arr, y, method='linear', bounds_error=False, fill_value=0)

    return out.reshape(x[0].size, x[1].size)


def subSphere(y, alpha, Rad):
    # Rad is the radius of the Ewald sphere
    r = sqrt(y[0][:, None] ** 2 + y[1][None, :] ** 2)
    if Rad is None:
        Y = 0 * r
    else:
        Y = Rad ** 2 - r ** 2
        Y[Y < 0] = 0
        # Y = points on sphere
        Y = Rad - sqrt(Y)
    # Y = points on sphere at maximum precession
    Y = sin(alpha * pi / 180) * r + cos(alpha * pi / 180) * Y
    # Y = maximum Fourier frequency needed for precession
    Y = Y.max() + abs(y[2].item(min(1, y[2].size - 1)) - y[2].item(0)) / 2

    return y[0], y[1], y[2].copy()[abs(y[2]) <= Y]


def planes2basis(planes, points=None):

    def normalise(X): return X / maximum(1e-8, sqrt((X ** 2).sum()))

    isList = hasattr(planes[0], '__len__')
    if not isList:
        planes = [planes]

    if points is None:

        def basis(p):
            if abs(p[2]) > .9:
                e1 = array([0, 1, 0])
            else:
                e1 = array([0, 0, 1])
            e1 = normalise(cross(p, e1))
            e2 = normalise(cross(p, e1))
            return e1, e2, normalise(p)

        R = [basis(p) for p in planes]
    else:
        if not hasattr(points[0], '__len__'):
            points = [points]

        def basis(p, q):
            e1 = normalise(cross(p, q))
            e2 = normalise(cross(p, e1))
            return e1, e2, normalise(p)

        R = [basis(planes[i], points[i]) for i in range(len(planes))]

    return (R if isList else R[0])


pass
##################################################
# Visualisation tools
##################################################


def debias_FT(FT, y, Z, twice=False):
    _, FTa = getDiscretisation(
        array([[0] * len(y)], dtype=FTYPE), Z, GPU=True)
    atom = abs(FTa(y))
    atom = atom / atom.max()
    atom = maximum(atom, 1e-16)
    if twice:
        atom *= atom

    FT = minimum(FT / atom, abs(FT).max())
    return FT


def _plot(a, FTa, FTb, DP, x, y, titles=None, block=True, Z=None):
    from matplotlib import pyplot as plt

    def squash(X):
        from numpy import percentile

        X = X.copy()
        X /= X.max()

        peaks = percentile(X, [100, 99.9, 90, 80, 50, 0])

        if Z is not None:
            XX = debias_FT(X, y[:X.ndim], Z, True)
            ind = (XX < 1e-6 * XX.max())
            X = X.copy()
            X[ind] = 0
            return X ** .25

#         return X ** 1
        return minimum(X, .1 * peaks[0])
#         return log10(X + max(1e-32, 1e-0 * peaks[2]))

        if peaks[1] != peaks[-2]:
            X[X > peaks[1]] = peaks[1]
            X[X < peaks[-2]] = peaks[-2]

        # Small value should be fraction of the intensity
        n = log10(20) / log10(peaks[0] / (peaks[3] + 1e-10))
        return X ** n

    if a is not None:
        a = a if a.ndim == 2 else a[..., int(a.shape[2] / 2)]
    if FTa is not None:
        FTa = FTa if FTa.ndim == 2 else FTa[..., int(FTa.shape[2] / 2)]
    if titles is None:
        titles = ('Slice of Crystal', 'FT of Crystal on Ewald Sphere',
                  'Fourier Probe function', 'Diffraction pattern')

    if Z is None:
        plt.figure('Constant threshold, non-linear scale')
    else:
        plt.figure('Radial threshold, linear scale')
    if a is not None:
        plt.subplot(221)
        plt.title(titles[0])
        plt.imshow(abs(a), aspect='auto',
                   extent=[x[0][0], x[0][-1], x[1][-1], x[1][0]])
    if FTa is not None:
        plt.subplot(222)
        plt.title(titles[1])
        plt.imshow(squash(abs(FTa) ** 2), aspect='auto',
                   extent=[y[0][0], y[0][-1], y[1][-1], y[1][0]])
    if FTb is not None:
        plt.subplot(223)
        plt.title(titles[2])
        plt.imshow(squash(abs(FTb) ** 2), aspect='auto',
                   extent=[y[0][0], y[0][-1], y[1][-1], y[1][0]])
    #     plt.imshow(abs(FTb.real), aspect='auto',
    #                extent=[X2[0][0], X2[0][-1], X2[1][-1], X2[1][0]])
    if DP is not None:
        plt.subplot(224)
        plt.title(titles[3])
        plt.imshow(squash(DP), aspect='auto',
                   extent=[y[0][0], y[0][-1], y[1][-1], y[1][0]])
    plt.show(block=block)


pass
##################################################
# Top level functions
##################################################


def _test(spacing, mesh, n=None, dX=inf, rX=0, dY=inf, rY=1e-16, Rad=None, at45=False):
    if Rad is None:
        alpha = 0
    else:
        Rad = array(Rad, dtype=FTYPE)
        alpha = 3

    X3, Y3 = getFTpoints(3, n, dX, rX, dY, rY)
    print(X3[0].size, X3[0][1] - X3[0][0], X3[0].ptp())
    print(Y3[0].size, Y3[0][1] - Y3[0][0], Y3[0].ptp())
#     exit()
    X2, Y2 = getFTpoints(2, n, dX, rX, dY, rY)
    dft3, idft3 = getDFT(X3, Y3)
    dy = array((array([1, 0, 0]), array([0, 1, 0]), array([0, 0, 1])))

    loc = spacing * getSiBlock(1, at45).astype(FTYPE)
    c = crystal(loc, 14, spacing * getSiBlock(1, at45, True), GPU=True)
    c = c.repeat(mesh)
    c = c - array(c.box)[:, 1] / 2
#     c = c * dy.T

#     loc = spacing * getSiBlock(mesh, at45).astype(FTYPE)
#     c = crystal(loc, 14, spacing * getSiBlock(mesh, at45, True), GPU=True)
#     c = c - array(c.box)[:, 1] / 2

    b, FTb = getBessel(30)

    tic = toc()
    a, FTa, x, y = getDiscreteCrystal(c, X3, Y3, Rad, alpha)

    tic = toc('Discretise Crystal', tic)
    realSlice = a[:, :, abs(x[2]).argmin()].squeeze()
    FourierSlice = grid2sphere(FTa, y, dy, Rad)
    tic = toc('Extract slice of Crystal', tic)
    realProbe = b(X2)
    tic = toc('Calculate of real probe', tic)
    FourierProbe = FTb(Y2)
    tic = toc('Calculate of Fourier probe', tic)

    if True:  # nY >= nX:
        DP = convolve(FTa, FTb(toMesh(y, dy)))
#         DP = convolve(FTa, FourierProbe, [abs(yy.item(1) - yy.item(0)) for yy in Y2], axes=(0, 1))
    else:
        DP = dft3(a * realProbe[:, :, None])
    DP = grid2sphere(DP, y, dy, Rad)
    DP = abs(DP) ** 2
    tic = toc('Convolve', tic)

    _plot(realSlice, FourierSlice, FourierProbe, DP, x, y, Z=None)


def getDiscreteCrystal(c, x, y, Rad=None, alpha=0, filename=None):
    # Real Space representation
    realCrystal = c(x)

    # Fourier Space representation
    if Rad is None:
        FourierCrystal = c.FT(y[:2])
#         dft3, idft3 = getDFT(x, y)
#         tmp = dft3(realCrystal)[..., abs(y[2]).argmin()]
#         plt.subplot(131);plt.imshow(abs(FourierCrystal))
#         plt.colorbar()
#         plt.subplot(132);plt.imshow(abs(tmp))
#         plt.colorbar()
#         plt.subplot(133);plt.imshow(abs(FourierCrystal) - abs(tmp))
#         plt.colorbar()
    else:
        y = list(subSphere(y, alpha, Rad))
        y[2] = y[2][y[2] >= -1e-10]

        FourierCrystal = c.FT(y)
        tmp = FourierCrystal[:, :, -1:0:-1]
        # Crystal is real so negative frequencies are just conjugates
        FourierCrystal = concatenate((tmp.conj(), FourierCrystal), axis=2)
        y[2] = concatenate((-y[2][-1:0:-1], y[2]), axis=0)

    if filename is not None:
        myDict = {'c': realCrystal, 'FTc': FourierCrystal}
        for i in range(3):
            myDict['x' + str(i)] = x[i]
            myDict['y' + str(i)] = y[i]
        savemat(filename, **myDict)

    return realCrystal, FourierCrystal, x, y


def doPrecession(FT, probewidth, x, y, Rad, alpha, nTheta, keeppath=True, filename=None):
    x = fromFreq(y)
    b, FTb = getBessel(probewidth)
    DTYPE = FT.dtype if hasattr(FT, 'dtype') else 'complex128'

    dft3, idft3 = getDFT(x, y)
    inFT = False
    if inFT:
        FourierProbe = FTb(y[:2])
    else:
        dft2 = getDFT(x[:2], y[:2])[0]
        FourierProbe = dft2(b(x[:2] + [array([0])])[..., 0])

    d = [abs(yy.item(min(1, yy.size - 1)) - yy.item(0)) for yy in y]
    if d[2] == 0:
        d[2] = (d[0] * d[1]) ** .5

    if nTheta <= 1:
        theta, nTheta = [0], 1
    else:
        theta = linspace(0, 360, nTheta, endpoint=False)

    if isLoud():
        print('Starting precession ...', flush=True)

    DP, DPs = None, None
    with SILENT() as c:
        if hasattr(FT, '__call__'):
            dy = (array([1, 0, 0]), array([0, 1, 0]), array([0, 0, 1]))
            tic = toc()
            for i in range(nTheta):
                # Rotate the Fourier Transform
                R = precess(None, alpha, theta[i])
                newFT = (FT * R).FT(y)
                # Do convolution
                newFT = convolve(newFT, FourierProbe, d, (0, 1))
                # Project to sphere
                newFT = fast_abs(newFT, newFT).real
                newFT *= newFT
#                 newFT = abs(newFT)**2
                newFT = grid2sphere(newFT, y, dy, Rad)

                if DP is None:
                    DP = newFT
                    if keeppath:
                        DPs = zeros((nTheta, DP.shape[0], DP.shape[1]), dtype=FTYPE)
                        DPs[0] = DP
                elif keeppath:
                    DPs[i] = newFT
                    DP += DPs[i]
                else:
                    DP += newFT
                progressBar(i + 1, nTheta, tic, context=c)

            flatDP = abs(convolve(FT.FT(y[:2]), FourierProbe, d)) ** 2
        elif inFT:
            flatDP = abs(
                convolve(FT[..., abs(y[2]).argmin()], FourierProbe, d[:2])) ** 2
            real = fftn(ifftshift(FT)) * array(d).prod().astype(DTYPE)

            buf = empty(real.shape, dtype=real.dtype)
            ft, buf = plan_fft(buf, overwrite=True, planner=1)
            ift, buf1 = plan_ifft(ft.output_array, overwrite=True, planner=1)

            tic = toc()
            for i in range(nTheta):
                # Rotate the Probe
                R = precess(None, alpha, theta[i])
                dy = list(R)  # Don't have to worry about R/R.T
                FTb(toMesh(y, dy), out=buf)
                if buf1 is ft.output_array:
                    ft()
                else:
                    # copy output if necessary (it shouldn't be)
                    buf1[...] = ft()
#                 buf1 = fftn(FTb(toMesh(y, dy)).astype(DTYPE, copy=False))

                # Do convolution
                buf1 *= real
                newFT = ift()
#                 newFT = ifftn(real * buf1)
                newFT = fast_abs(newFT, buf).real
                newFT *= newFT
#                 newFT = abs(newFT)**2
                newFT = grid2sphere(newFT, y, dy, Rad)

                if DP is None:
                    DP = newFT
                    if keeppath:
                        DPs = zeros((nTheta, DP.shape[0], DP.shape[1]), dtype=FTYPE)
                        DPs[0] = DP
                elif keeppath:
                    DPs[i] = newFT
                    DP += DPs[i]
                else:
                    DP += newFT
                progressBar(i + 2, nTheta + 1, tic, context=c)
        else:
            flatDP = abs(
                convolve(FT[..., abs(y[2]).argmin()], FourierProbe, d[:2])) ** 2
            real = idft3(FT) * array(d).prod().astype(DTYPE)
            fftshift_phase(real)  # removes need for fftshift after fft
            buf = empty(real.shape, dtype=real.dtype)
            ft, buf = plan_fft(buf, overwrite=True, planner=1)
            tic = toc()
            for i in range(nTheta):
                # Rotate the Probe
                R = precess(None, alpha, theta[i])
                dy = list(R)
                b(toMesh(x, R.T), out=buf, scale=real)  # = bess*real

                # Do convolution
                newFT = ft()
                newFT = fast_abs(newFT, buf).real
                newFT *= newFT
#                 newFT = abs(newFT)**2
                newFT = grid2sphere(newFT.real, y, dy, Rad)

                if DP is None:
                    DP = newFT
                    if keeppath:
                        DPs = zeros((nTheta, DP.shape[0], DP.shape[1]), dtype=FTYPE)
                        DPs[0] = DP
                elif keeppath:
                    DPs[i] = newFT
                    DP += DPs[i]
                else:
                    DP += newFT
                progressBar(i + 2, nTheta + 1, tic, context=c)

    if isLoud():
        print('finished.', flush=True)

    DP /= nTheta

    if filename is not None:
        savemat(filename, DP=DP, DPs=DPs, flatDP=flatDP)

    return DP, DPs, flatDP


def doScanFFT(C, probewidth, x, y, Rad, alpha, nTheta, R, out0, out1):
    '''
    C is the crystal
    probewidth is the width of the probe in Angstroms, ~30
    x is the scan grid
    y is the Fourier grid
    Rad is the radius of Ewald sphere
    alpha is precession angle
    nTheta is the number of precession angles
    R is the vector of rotations in tilt series
    out0/out1 are the pre-buffered output storage
    '''
    sz = out0.shape
    y = subSphere(y, alpha, Rad)
    # Start computations
    with SILENT() as c:
        tic = toc()
        for i0 in range(len(R)):
            # Rotate to new tilt
            tempC = C * array(R[i0])
            FT = tempC.FT(y)

            for j0 in range(x[0].size):
                for j1 in range(x[1].size):
                    if out0[i0, j0, j1].min() >= 0:
                        # This projection has already been computed
                        continue
#                     j0, j1 = 1, 1
                    # Physically move the probe by x -> move crystal by -x
                    #                                -> translateFT by x
                    DP = doPrecession(translateFT(FT, y, [x[0][j0], x[1][j1], 0]),
                                                   probewidth, x, y, Rad, alpha, nTheta)

                    DP = DP[0] / DP[0].max(), DP[2] / DP[2].max()
                    from matplotlib import pyplot as plt
                    plt.subplot(121)
                    plt.imshow((1e-6 + DP[0]))
                    plt.subplot(122)
                    plt.imshow((1e-6 + DP[1]))
#                     plt.plot((1e-4 + DP[1][:, DP[1].shape[1] // 2]))
                    plt.title(str(i0) + ', ' + str(j0) + ', ' + str(j1))
                    plt.show()
                    exit()
                    out0[i0, j0, j1], out1[i0, j0, j1] = DP

                    progressBar(j1 + sz[2] * (j0 + sz[1] * i0) + 1, sz[0] * sz[1] * sz[2], tic, context=c)


def doScanReal(C, probewidth, x, y, Rad, alpha, nTheta, R, out0, out1):
    '''
    C is the crystal
    probewidth is the width of the probe in Angstroms, ~30
    x is the scan grid
    y is the Fourier grid
    Rad is the radius of Ewald sphere
    alpha is precession angle
    nTheta is the number of precession angles
    R is the vector of rotations in tilt series
    out0/out1 are the pre-buffered output storage
    '''
    sz = out0.shape
    y = subSphere(y, alpha, Rad)
    xx = fromFreq(y)
    dx = [X[1] - X[0] if X.size > 1 else 0 for X in xx]
    fullx = [arange(xx[i].min() + x[i].min() - dx[i], xx[i].max() + x[i].max() + dx[i],
                    dx[i], dtype=xx[i].dtype) if dx[i] > 0 else xx[i] for i in range(3)]
    dft3, _ = getDFT(xx, y)
    # Start computations
    with SILENT() as c:
        tic = toc()
        for i0 in range(len(R)):
            # Rotate to new tilt
            tempC = C * array(R[i0])
            vol = tempC(fullx)

            for j0 in range(x[0].size):
                for j1 in range(x[1].size):
                    if out0[i0, j0, j1].min() >= 0:
                        # This projection has already been computed
                        continue
#                     j0, j1 = 1, 1
                    # fullx[slice][i] = x + xx[i] for all i
                    # fullx[slice][0] = x + xx[0]
                    # slice = slice(a,b) s.t. fullx[a] = x+xx[0], b-a = xx.size
                    Slice = (abs(fullx[0] - x[0][j0] - xx[0][0]).argmin(),
                              abs(fullx[1] - x[1][j1] - xx[1][0]).argmin(),
                              0)
                    Slice = tuple(slice(Slice[i], Slice[i] + xx[i].size) for i in range(3))
                    FT = dft3(vol[tuple(Slice)])
                    if vol[tuple(Slice)].max() < 1e-6:
                        DP = 0 * FT.real, 0 * FT.real
                    else:
                        DP = doPrecession(FT, probewidth, xx, y, Rad, alpha, nTheta)
                        DP = DP[0] / DP[0].max(), DP[2] / DP[2].max()

    #                     from matplotlib import pyplot as plt
    #                     plt.subplot(121)
    #                     plt.imshow((1e-6 + DP[0]))
    #                     plt.subplot(122)
    #                     plt.imshow((1e-6 + DP[1]))
    # #                     plt.plot((1e-4 + DP[1][:, DP[1].shape[1] // 2]))
    #                     plt.title(str(i0) + ', ' + str(j0) + ', ' + str(j1))
    #                     plt.show()
    # #                     exit()
                    out0[i0, j0, j1], out1[i0, j0, j1] = DP

                    progressBar(j1 + sz[2] * (j0 + sz[1] * i0) + 1, sz[0] * sz[1] * sz[2], tic, context=c)


def doScan(C, probewidth, x, y, planes, Rad, alpha, nTheta, filename=None, points=None, FFT=False):
    planes = planes.reshape(-1, 3)
    sz = (planes.shape[0], x[0].size, x[1].size, y[0].size, y[1].size)
    try:
        out0, out1 = loadmat(filename, 'precessed', 'flat')
        if any(out0.shape[i] != sz[i] for i in range(out0.ndim)) \
            or any(out1.shape[i] != sz[i] for i in range(out1.ndim)):
            raise Exception('Sizes do not match')
    except Exception:
        out0 = zeros(sz, dtype='f8')
        out1 = zeros(sz, dtype='f8')

    points = points if points is None else points.reshape(-1, 3)
    R = planes2basis(planes, points)

    # Determine flags for restarting computations
    # Any negative components in a projection means not computed
    if out0.min() >= 0:
        out0.fill(-1)

    # If any interupt then save the result
    if filename is not None:

        def saveall(*_, **__):
            savemat(filename, precessed=out0, flat=out1)
            print('\nInterrupt detected, saving partial computation')
            exit()

        import signal
        __orig = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGTERM, saveall)
        signal.signal(signal.SIGINT, saveall)

    if FFT:
        doScanFFT(C, probewidth, x, y, Rad, alpha, nTheta, R, out0, out1)
    else:
        doScanReal(C, probewidth, x, y, Rad, alpha, nTheta, R, out0, out1)

    if filename is not None:
        signal.signal(signal.SIGTERM, __orig[0])
        signal.signal(signal.SIGINT, __orig[1])
        savemat(filename, precessed=out0, flat=out1)
    return out0, out1


def getOrientations(c, y, Z, cutoff, rad=None, angle=None, filename=None):
    with SILENT():
        print('Starting full volume FT...')
        FT = c.FT(y)
        print('complete.')
        FT = abs(FT / abs(FT).max()) ** 2
        debFT = debias_FT(FT, y, Z, twice=True)
    from numpy import savetxt, where, vstack
    from os.path import join

    if rad is None:
        rad = 3 * abs(y[0].item(1) - y[0].item(0))
    if angle is None:
        angle = 4

    # filters points within a radius of eps
    @numba.jit
    def filter1(arr, ind, eps):
        for i in range(arr.shape[0]):
            curr = arr[i]
            for j in range(i + 1, arr.shape[0]):
                FLAG = False
                for k in range(arr.shape[1]):
                    if abs(arr[j, k] - curr[k]) > eps:
                        FLAG = True
                        break
                if ind[j]:
                    ind[j] = FLAG

    minthresh = 1e-2
    print('Finding second peak...')
    while True:
        # Find second biggest peak ~10^-10
        thresh = where(logical_and(FT > minthresh, debFT > cutoff[1]))
        thresh = vstack(tuple(y[i][thresh[i]]
                              for i in range(len(thresh))) + (FT[thresh],)).T
        thresh = list(thresh)
        thresh.sort(key=lambda a:-a[-1] -
                    (1 if abs(a[:-1]).max() < 1e-10 else 0))
        thresh = array(thresh)
        ind = zeros(thresh.shape[0], dtype='bool') + True
        filter1(thresh[:, :3], ind, rad)
        thresh = -thresh[ind, -1]
        if thresh.size > 1:
            thresh.sort()
            peak = -thresh[1]
            break
        else:
            print('threshold dropped')
            minthresh /= 10
            if minthresh < 1e-10:
                from matplotlib import pyplot as plt
                plt.subplot(121); plt.imshow(log10(maximum(FT.max(0), 1e-10)))
                plt.colorbar()
                plt.subplot(122); plt.imshow(log10(maximum(debFT.max(0), 1e-10)))
                plt.colorbar()
                plt.title('log_{10} to give cutoff 0/1 respectively')
                plt.show()
                raise ValueError('thresholds are too large')
    print('Second peak found.')

    # Select points
    thresh = where(logical_and(FT > cutoff[0] * peak, debFT > cutoff[1]))
    thresh = vstack(tuple(y[i][thresh[i]]
                          for i in range(len(thresh))) + (FT[thresh],)).T

    # Sort by magnitude, guarrantee 0 is at the top
    thresh = list(thresh)
    thresh.sort(key=lambda a:-a[-1] - (1 if abs(a[:-1]).max() < 1e-10 else 0))
    thresh = array(thresh)

    # Remove duplicate points
    print('\nTotal points', thresh.shape[0],
          '(%d)' % (abs(thresh[:, 2]) < 1e-10).sum())
    ind = zeros(thresh.shape[0], dtype='bool') + True
    filter1(thresh[:, :3], ind, rad)
    thresh = thresh[ind]
    print('Unique points', thresh.shape[0],
          '(%d)' % (abs(thresh[:, 2]) < 1e-10).sum())

    if thresh.shape[0] == 1:
        raise ValueError(
            '\nOnly direct beam satisfies cutoff conditions. Please lower cutoff thresholds.')
    print('\nMagnitude of direct beam is 1\n'
          +'Magnitude of second peak is %1.1e\n' % thresh[1, -1])

    def normalise(X):
        return X / maximum(1e-8, sqrt((X ** 2).sum(-1, keepdims=True)))

    def vecprod(X):
        return cross(X.reshape(-1, 1, 3), X.reshape(1, -1, 3), axis=-1)

    # Compute planes
    @numba.jit
    def filter2(arr, ind, eps):
        for i in range(arr.shape[0]):
            curr = arr[i]
            for j in range(i + 1, arr.shape[0]):
                if abs(arr[j, 0] * curr[0] + arr[j, 1] * curr[1] + arr[j, 2] * curr[2]) > eps:
                    ind[j] = False

    plane = normalise(thresh[1:, :3])  # cuts out direct beam then normalises
    ind = zeros(plane.shape[0], dtype='bool') + True
    filter2(plane, ind, cos(2 * pi / 180))  # filters directions
    print('Unique directions', ind.sum(),
          '(%d in-plane)' % (abs(plane[ind, 2]) < 1e-10).sum())
    plane = normalise(vecprod(plane[ind]).reshape(-1, 3))  # cross product every pair of vectors
    plane = plane[abs(plane).max(1) > 1e-10]  # remove null vectors
    ind = zeros(plane.shape[0], dtype='bool') + True
    filter2(plane, ind, cos(2 * pi / 180))  # Filter duplicated hyperplanes
    plane = plane[ind]
    print('Unique planes', plane.shape[0],
          '(%d in single tilt series)' % max((abs(plane) < 1e-10).sum(0)))
    plane = plane[plane[:, 0] > cos(70 * pi / 180)]
    print('Unique planes with angle cutoff', plane.shape[0],
          '(%d in a single tilt-series)' % max((abs(plane) < 1e-10).sum(0)))

    if filename is not None:
        savetxt(join('sim', filename + '.csv'),
                thresh.astype('f4'), delimiter=',')

    return FT, debFT, thresh, plane


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    TEST, PURE = True, True
########################
    if TEST:
        # Testing code
        n, rX, dY, rY = 256, 0 * 350, inf, 12
        spacing, mesh = 5.4, (64,) * 3
        Rad = 50 * 2 * pi  # Units Ang^{-1}
        Rad = None
        _test(spacing, mesh, n=n, rX=rX, dY=dY, rY=rY, Rad=Rad, at45=False)

########################

    elif PURE:
        rep, Rad, alpha, ntheta = 32, 21, 0, 1
        spacing = 5.4
        E = 100
        Rad = sqrt(E * (2 * 511 + E)) / 12.398 * 2 * pi  # units of keV
        n, dX, rX = 512, [.1] * 2 + [None], [None] * 2 + [min(500, 1 * rep * spacing)]

        sz = spacing * getSiBlock(1, rotate=False, justwidth=True)
        c = crystal(spacing * getSiBlock(1, rotate=False), 14, sz)
        c = c.repeat(rep)
#         a = 1 / sqrt(3); c = c.scale([[(1 + a) / 2, -(1 - a) / 2, -a], [-(1 - a) / 2, (1 + a) / 2, -a], [a, a, a]])
        c = c - c.box[:, 1] / 2
#         c = c - c._sum[0].loc.mean(0)
#         c = c.scale([[2 / 5 ** .5, 1 / 5 ** .5, 0], [-1 / 5 ** .5, 2 / 5 ** .5, 0], [0, 0, 1]])
#         c = c.scale([[1, 0, 0], [0, 2, 0], [0, 0, .5]])
#         c = c.scale([[1, 0, 0], [0, cos(.05), -sin(.05)], [0, sin(.05), cos(.05)]])

#         rep, alpha, ntheta, Rad = [50, 35, 18], 0, 30, 280
#         A = array([
#                [[ 0.99599326, -0.00534346, 0.        ],
#                 [-0.00661995, 0.9957806 , 0.        ],
#                 [ 0.        , 0.        , 1.        ]],
#                [[ 0.99671185, -0.00745167, 0.        ],
#                 [-0.00720214, 0.99726   , 0.        ],
#                 [ 0.        , 0.        , 1.        ]],
#                [[ 0.9930332 , 0.00286672, 0.        ],
#                 [-0.00782284, 1.0048785 , 0.        ],
#                 [ 0.        , 0.        , 1.        ]],
#                [[ 1.0037364 , -0.00919196, 0.        ],
#                 [ 0.0068842 , 0.999102  , 0.        ],
#                 [ 0.        , 0.        , 1.        ]],
#                [[ 1.0092281 , 0.00706319, 0.        ],
#                 [ 0.00891825, 0.997456  , 0.        ],
#                 [ 0.        , 0.        , 1.        ]],
#                [[ 0.996681  , 0.00391212, 0.        ],
#                 [-0.00759633, 0.99793637, 0.        ],
#                 [ 0.        , 0.        , 1.        ]],
#                [[ 0.994892  , 0.00178384, 0.        ],
#                 [-0.00331492, 1.0038745 , 0.        ],
#                 [ 0.        , 0.        , 1.        ]],
#                [[ 0.99072593, 0.00302923, 0.        ],
#                 [-0.00727155, 1.0083034 , 0.        ],
#                 [ 0.        , 0.        , 1.        ]],
#                [[ 0.9991862 , 0.00703546, 0.        ],
#                 [ 0.00666022, 0.9947785 , 0.        ],
#                 [ 0.        , 0.        , 1.        ]],
#                [[ 0.99808955, -0.0040341 , 0.        ],
#                 [-0.0029109 , 0.99245864, 0.        ],
#                 [ 0.        , 0.        , 1.        ]],
#                [[ 1.0060972 , 0.00828733, 0.        ],
#                 [ 0.00421763, 0.9963279 , 0.        ],
#                 [ 0.        , 0.        , 1.        ]],
#                [[ 1.0069501 , 0.00559179, 0.        ],
#                 [-0.00325314, 1.0062559 , 0.        ],
#                 [ 0.        , 0.        , 1.        ]],
#                [[ 0.99109674, -0.0081964 , 0.        ],
#                 [-0.00197862, 1.0025555 , 0.        ],
#                 [ 0.        , 0.        , 1.        ]],
#                [[ 1.0054597 , 0.00903043, 0.        ],
#                 [ 0.00452853, 0.99276143, 0.        ],
#                 [ 0.        , 0.        , 1.        ]],
#                [[ 1.0021302 , 0.00375169, 0.        ],
#                 [-0.00244737, 1.0083808 , 0.        ],
#                 [ 0.        , 0.        , 1.        ]]], dtype='float32')
#         c = [c.scale(a) for a in A]
#         for i in range(len(c)):
#             c[i] = c[i].translate([ -(c[i].box[0][0] + c[i].box[0][1]) / 2,
#                                 -(c[i].box[1][0] + c[i].box[1][1]) / 2,
#                                 -.9 * 75 if i == 0 else c[i - 1].box[2, 1] - c[i].box[2, 0]])
#         c = sum(c)

#         c.tofile('somefile.xyz')
#         print('done'); exit()

        x, y = getFTpoints(3, n=n, dX=dX, rX=rX)
        y = subSphere(y, alpha, Rad)
        x = fromFreq(y)

        dft3, idft3 = getDFT(x, y)
        C = c(x)
        FT = dft3(C)

#         FT = c.FT(y)
#         plt.subplot(211)
#         plt.imshow(abs(FT).max(0).T, extent=[y[1].min(), y[1].max(), y[2].min(), y[2].max()], origin='lower', aspect='auto')
#         plt.plot(y[1], Rad - sqrt(Rad ** 2 - y[1] ** 2), 'r')
#         b, FTb = getBessel(5)
#         FT = convolve(FT, FTb(y[:2]))
#         plt.subplot(212)
#         plt.imshow(abs(FT).max(0).T, extent=[y[1].min(), y[1].max(), y[2].min(), y[2].max()], origin='lower', aspect='auto')
#         plt.plot(y[1], Rad - sqrt(Rad ** 2 - y[1] ** 2), 'r')
#         plt.show()
#         exit()

        DP, _, flatDP = doPrecession(FT, 3, x, y, Rad, alpha, ntheta, False)

#         r = (y[0].reshape(-1, 1) ** 2 + y[1].reshape(1, -1) ** 2) ** .5
#         DP *= maximum(1, r ** 4.5)
        DP /= DP.max()
        flatDP /= flatDP.max()

#         DP, flatDP = DP ** .3, flatDP ** .3
#         DP, flatDP = minimum(0.1, DP), minimum(0.1, flatDP)
#         DP, flatDP = minimum(0.0075, DP), minimum(0.1, flatDP)

#         plt.imshow(C.max(-1).T, cmap='gray', extent=[x[0][0], x[0][-1], x[1][-1], x[1][0]])
#         plt.figure()
#         plt.subplot(121)
#         plt.imshow(DP.T, cmap='gray', extent=[y[0][0], y[0][-1], y[1][-1], y[1][0]])
#         plt.subplot(122)
#         plt.imshow(flatDP.T, cmap='gray', extent=[y[0][0], y[0][-1], y[1][-1], y[1][0]])
#         plt.show()
#         exit()

        plt.subplot(121)
        plt.imshow(-log(1e-6 + DP), cmap='gray', extent=[y[0][0], y[0][-1], y[1][0], y[1][-1]])
        plt.title('Precessed')
        plt.subplot(122)
#         plt.figure()
        plt.imshow(-log(1e-6 + flatDP), cmap='gray', extent=[y[0][0], y[0][-1], y[1][0], y[1][-1]])
        plt.title('Flat')
        plt.show()
        exit()

#         for __ in range(2):
#             for i in range(_.shape[0]):
#                 plt.cla()
#                 plt.imshow(_[i], extent=[y[0][0], y[0][-1], y[1][0], y[1][-1]])
#                 plt.draw()
#                 plt.pause(.3)
        plt.figure()
#         exit()

        mid = DP.shape[0] // 2
        plt.subplot(131)
        plt.plot(y[0], DP[mid])
        plt.title('Precessed')
        plt.subplot(132)
        plt.plot(y[0], flatDP[mid])
        plt.title('Flat')
        plt.subplot(133)
        plt.plot(y[0], DP[mid] - flatDP[mid])
#         plt.yscale('symlog', linthreshy=1e-10)
        plt.title('Difference')

        plt.show()

########################
# Running code
    else:
        from os.path import join
        n, rX, dY, rY = 128, 200, 1, 5
    #     ranY = int(2 * resX * ranX / pi)
        spacing, mesh = 5, (40, 40, 20)
        Rad = 50  # Units Ang^{-1}
        alpha, nTheta = 1, 50  # Units degrees

        # Generate other parameters:
        x, y = getFTpoints(3, n=n, rX=rX, dY=dY, rY=rY)

        def toCrystal(Mesh, Spacing, theta=0):
            loc = getSiBlock(1, rotate=True).astype(FTYPE)
            c = crystal(loc, 14, Spacing * getSiBlock(1, True, True), GPU=True)
            c = c.repeat(Mesh)
            return c

        A = toCrystal(mesh, spacing / 1.02)
        B = toCrystal(mesh, spacing)
        C = toCrystal(mesh, spacing * 1.02)
        h = A.box[2][1] + B.box[2][1] + C.box[2][1]
        A = A - [A.box[0][1] / 2, A.box[1][1] / 2, h / 2]
        B = B - [B.box[0][1] / 2, B.box[1][1] / 2, -A.box[2][1]]
        C = C - [C.box[0][1] / 2, C.box[1][1] / 2, -B.box[2][1]]
        ABC = A + B + C

        pure = toCrystal((mesh[0], mesh[1], 3 * mesh[2]), spacing)
        pure = pure + [b[0] for b in B.box]

        myCrystals = (pure, A, B, C, ABC)
        filename = ('pure', 'A', 'B', 'C', 'ABC')

        # Generate simulation:
        for i in [0]:
            print('Starting on crystal ' + filename[i])
            myFiles = [join('sim', 'lattice_' + filename[i]),
                       join('sim', 'DP_' + filename[i])]

            tic = toc()
            if filename[i] != 'ABC':
                getDiscreteCrystal(myCrystals[i], x, y, Rad, 3, myFiles[0])
            else:
                # ABC = A+B+C, FT(ABC) = FT(A)+FT(B)+FT(C)
                tmp = [0, 0]
                for c in 'ABC':
                    precomp = loadmat(myFiles[0].replace('ABC', c), 'c', 'FTc')
                    tmp[0] += precomp[0]
                    tmp[1] += precomp[1]
                savemat(myFiles[0], c=tmp[0], FTc=tmp[1])
            a, FTa = loadmat(myFiles[0], 'c', 'FTc')
            x = [X.reshape(-1)
                 for X in (precomp['x0'], precomp['x1'], precomp['x2'])]
            y = [Y.reshape(-1)
                 for Y in (precomp['y0'], precomp['y1'], precomp['y2'])]
            tic = toc('Crystal ' + filename[i] + ' simulated', tic)

    #         doPrecession(FTa, 30, x, y, Rad, alpha, 100, myFiles[1])
            doPrecession(myCrystals[i], 30, x, y,
                         Rad, alpha, 100, myFiles[1])
            DP, DPs, flatDP = loadmat(myFiles[1], 'DP', 'DPs', 'flatDP')
            tic = toc('Crystal ' + filename[i] + ' precessed', tic)

            _plot(a, grid2sphere(FTa, y, Rad), flatDP, DP, x, y, block=False,
                  titles=('Crystal slice', 'FT on Ewald Sphere', 'Hyperplane DP',
                          'Precessed DP'))

        def squash(x):
            from numpy import median, percentile

    #         bins = percentile(x, linspace(0, 100, 1000))
    #         x = bins.searchsorted(x[..., None]).reshape(x.shape)
    #         return x

            peaks = percentile(x, [100, 99.9, 90, 80, 50])
            m = median(x)

            x = x.copy()
            x[x > peaks[1]] = peaks[1]
            x[x < peaks[-1]] = peaks[-1]

            # Small value should be fraction of the intensity
            n = log10(20) / log10(peaks[0] - peaks[3])
            return x ** n

    #         return log10(x)
        fig = plt.figure(figsize=(16, 9))
        plt.subplot(121)
        plt.imshow(squash(DP), extent=[y[0][0], y[0][-1], y[1][-1], y[1][0]])
        plt.subplot(122)
        for i in range(DPs.shape[0]):
            plt.cla()
            plt.imshow(squash(DPs[i]),
                       extent=[y[0][0], y[0][-1], y[1][-1], y[1][0]])
            plt.title('Pattern ' + str(i))
            plt.pause(.2)
        plt.show()
