'''
Created on 13 Nov 2018

@author: Rob Tovey
'''
from numpy import abs, exp, pi, array, sin, log10, arange, sinc, log, empty, eye, \
    ascontiguousarray
from numpy.fft import fftfreq
from math import ceil, floor, inf
from utils import toMesh, fftn, ifftn, fftshift, ifftshift, USE_CPU, fast_fft_len, fftshift_phase


def toFreq(x):
    y = []
    for X in x:
        if X.size > 1:
            y.append(fftfreq(X.size, X.item(1) - X.item(0)) * (2 * pi))
        else:
            y.append(array([0]))
    return [fftshift(Y) for Y in y]


def fromFreq(y):
    x = []
    for Y in y:
        if Y.size > 1:
            x.append(fftfreq(Y.size, (Y.item(1) - Y.item(0)) / (2 * pi)))
        else:
            x.append(array([0]))
    return [fftshift(X) for X in x]


def getFTpoints(ndim, n=None, dX=inf, rX=0, dY=inf, rY=1e-16):
    '''
    Generates real and fourier space meshes based on requested properties.

    X,Y = getFTpoints(ndim, n=None, dX=inf, rX=0, dY=inf, rY=1e-16)

    X = [X[0], X[1], ..., X[ndim-1]]
        X[i] is an array of length at most n[i] with resolution at least dX[i]
        spanning between at least -rX[i]/2 and rX[i]/2.
    Y = [Y[0], Y[1], ..., Y[ndim-1]]
        Y[i] is an array of length at most n[i] with resolution at least dY[i]
        spanning between at least -rY[i]/2 and rY[i]/2.

    If some values are not specified then they will be inferred.
    Ideally:
        rX = n*dX, rY = n*dY
        rX = 2*pi/dY
        dX = 2*pi/rY
    If this is not exact then some intelligent guessing is used.

    '''

    '''
    1D domains have 3 parameters, range, resolution and number of pixels.
    Any 2 define the domain, we see how the Fourier Transform maps these:
        FT:(r, dx, n) -> (R, dX, N) such that r = n*dx, R = N*dX
            (r, dx)   -> (1/dx, 1/r)
            (r, n)    -> (n/r, n)
            (dx, n)   -> (1/(ndx), n)

    Input of (r, dx, R, dX)
        -> r = max(r, 1/dX), dx = min(dx, 1/R)
        n = int(r/dx)

    Note, with our 'no pi' convention we have:
        (r, dx, n) -> (2*pi/dx, 2*pi/r, n)
            r = max(r, 2*pi/dX), min(dx, 2*pi/R)

    '''
    pad = lambda t: list(t) if hasattr(t, '__len__') else [t] * ndim
    n, dX, rX, dY, rY = (pad(t) for t in (n, dX, rX, dY, rY))

    X, Y = [], []
    for i in range(ndim):
        dX[i] = inf if dX[i] is None else dX[i]
        rX[i] = 0 if rX[i] is None else rX[i]
        dY[i] = inf if dY[i] is None else dY[i]
        rY[i] = 1e-16 if rY[i] is None else rY[i]

        r, d = max(rX[i], 2 * pi / dY[i]), min(dX[i], 2 * pi / rY[i])
        if n[i] is None:  # n not specified
            n[i] = r / d
        elif d > 1e15 or n[i] * d < r:  # n and d specified
            # Real range/ Fourier resolution is more important
            d = r / max(n[i], 1)
        elif r > 1e-10:  # all specified
            # If n can be reduced then do
            n[i] = min(n[i], r / d)
        n[i] = fast_fft_len(max(int(ceil(n[i])), 1))
        r = n[i] * d

        X.append(arange(n[i]) * d)
        X[-1] = X[-1] - X[-1].mean()
        Y.append(2 * pi * fftshift(fftfreq(n[i], d)))

    return X, Y


def getDFT(X, Y):
    '''
    DFT, iDFT = getDFT(X, Y)

    X, Y = getFTpoints(...) are the meshgrid of spatial/frequency sampling of the FT.
    DFT and iDFT are exact inverses of each other.

    g = DFT(f):
        f is an array representing a fuction such that f(X[i]) = f[i]
        g is an array representing a function such that g[j] \approx FT(f)(Y[j])
    f = iDFT(g):
        g is an array representing a fuction such that g(Y[j]) = g[j]
        f is an array representing a function such that f[i] \approx FT^{-1}(g)(X[i])
    '''
    ndim = len(X)
    dx = [x.item(min(1, x.size - 1)) - x.item(0) for x in X]

    bigY = Y
    xmin = [x.item(0) for x in X]

    @jit(nopython=True, parallel=True, fastmath=True)
    def apply_phase_3D(x, f0, f1, f2):
        for i0 in prange(x.shape[0]):
            F0 = f0[i0]
            for i1 in range(x.shape[1]):
                F01 = F0 * f1[i1]
                for i2 in range(x.shape[2]):
                    x[i0, i1, i2] *= F01 * f2[i2]

    def DFT(fx, axes=None):
        NDIM = fx.ndim
        if axes is None:
            axes = [NDIM + i for i in range(-ndim, 0)]
        elif not hasattr(axes, '__iter__'):
            axes = (axes,)
        axes = array(axes)
        axes.sort()

        FT = fftshift(fftn(fx, axes=axes), axes=axes)

        if NDIM != 3:
            for i in axes:
                sz = [1] * NDIM
                sz[axes[i]] = -1
                FT *= exp(-xmin[i] * bigY[i].reshape(sz) * 1j) * (dx[i] if dx[i] != 0 else 1)
        else:
            F = [exp(-xmin[i] * bigY[i] * 1j) * (dx[i] if dx[i] != 0 else 1) for i in range(NDIM)]
            apply_phase_3D(FT, *F)

        return FT

    def iDFT(fy, axes=None):
        NDIM = fy.ndim
        if axes is None:
            axes = [NDIM + i for i in range(-ndim, 0)]
        elif not hasattr(axes, '__iter__'):
            axes = (axes,)
        axes = array(axes)
        axes.sort()

        FT = fy.astype(
            'complex' + ('128' if fy.real.dtype.itemsize == 8 else '64'),
            copy=True)

        if NDIM != 3:
            for i in axes:
                sz = [1] * NDIM
                sz[axes[i]] = -1
                FT *= exp(+xmin[i] * bigY[i].reshape(sz) * 1j) / (dx[i] if dx[i] != 0 else 1)
        else:
            F = [exp(+xmin[i] * bigY[i] * 1j) / (dx[i] if dx[i] != 0 else 1) for i in range(FT.ndim)]
            apply_phase_3D(FT, *F)

#         if NDIM != 3:
#             FT = ifftshift(FT, axes=axes)
#         else:
#             fftshift_phase(FT)  # removes need for ifftshift
        FT = ifftshift(FT, axes=axes)
        FT = ifftn(FT, axes=axes, overwrite_input=True)

        return FT

    return DFT, iDFT


from numba import cuda, jit, prange
from cmath import exp as c_exp, log as c_log


@jit(nopython=True, parallel=True, fastmath=True)
def __translate_CPU(y0, y1, y2, s, res):
    for i0 in prange(y0.size):
        for i1 in range(y1.size):
            for i2 in range(y2.size):
                res[i0, i1, i2] *= s * c_exp(1j * (y0[i0] + y1[i1] + y2[i2]))


@cuda.jit
def __translate_GPU(y0, y1, y2, s, res):
    i0, i1, i2 = cuda.grid(3)
    if (i0 >= y0.size) or (i1 >= y1.size) or (i2 >= y2.size):
        return
    res[i0, i1, i2] *= s * c_exp(1j * (y0[i0] + y1[i1] + y2[i2]))


@jit(nopython=True, parallel=True, fastmath=True)
def __periodic_CPU(z, s, r, res):
    for i in prange(z.shape[0]):
        v = 1
        for k in range(1, r):
            v += c_exp(1j * k * z[i]) * s[k]
        res[i] = v


@cuda.jit
def __periodic_GPU(z, s, r, res):
    i = cuda.grid(1)
    if i >= z.shape[0]:
        return
    v = 1
    for k in range(1, r):
        v += c_exp(1j * k * z[i]) * s[k]
    res[i] = v


@jit(nopython=True, parallel=True, fastmath=True)
def __periodic_point_CPU(z, r, res):
    for i in prange(z.shape[0]):
        v = 1 - c_exp(1j * z[i])
        if abs(v) < 1e-8:
            res[i] = r
        else:
            res[i] = (1 - c_exp(1j * z[i] * r)) / v


@cuda.jit
def __periodic_point_GPU(z, r, res):
    i = cuda.grid(1)
    if i >= z.shape[0]:
        return
    v = 1 - c_exp(1j * z[i])
    if abs(v) < 1e-8:
        res[i] = r
    else:
        res[i] = (1 - c_exp(1j * z[i] * r)) / v


@cuda.jit
def __periodic_high_CPU(z, dz, s, r, res):
    for i in prange(z.shape[0]):
        exp_p = 1 - c_exp(1j * (z[i] + dz))
        exp_m = 1 - c_exp(1j * (z[i] - dz))
        if (abs(exp_p.real) < 1e-8) or (abs(exp_m.real) < 1e-8):
            res[i] = 0
        else:
            res[i] = (1 + .5j / dz) * ((1 - c_exp(1j * z[i] * r) * s)
                                       * (c_log(exp_p) - c_log(exp_m)))


@cuda.jit
def __periodic_high_GPU(z, dz, s, r, res):
    i = cuda.grid(1)
    if i >= z.shape[0]:
        return
    exp_p = 1 - c_exp(1j * (z[i] + dz))
    exp_m = 1 - c_exp(1j * (z[i] - dz))
    if (abs(exp_p.real) < 1e-8) or (abs(exp_m.real) < 1e-8):
        res[i] = 0
    else:
        res[i] = (1 + .5j / dz) * ((1 - c_exp(1j * z[i] * r) * s)
                                   * (c_log(exp_p) - c_log(exp_m)))


def translateFT(FT, y, d, dy=None, pointwise=False):
    '''
    Input arr[i] = FT(u)(x_i), output arr[i] = FT(u)(x_i+d)
    '''
    if dy is not None:
        d = array(d).reshape(1, -1)
        d = [d.dot(ddy.reshape(-1, 1)) for ddy in dy]
    dim = FT.ndim

    y = [ascontiguousarray((y[i] * d[i]).reshape(-1)) for i in range(dim)]
    dz = [abs(yy.item(min(1, yy.size - 1)) - yy.item(0)) / 2 for yy in y]
    FT = FT.copy()
    if dim != 3:
        for i in range(dim):
            sz = [1] * dim
            sz[i] = -1

            if y[i].size == 1:
                continue

            if pointwise:
                FT = FT * exp(1j * y[i]).reshape(sz)
            else:
                FT *= (exp(1j * y[i]).reshape(sz) * sinc(dz[i] / pi))
    else:
        try:
            if USE_CPU: raise Exception
            tpb, grid = [8] * 3, [0] * 3
            for i in range(3):
                if tpb[i] > FT.shape[i]:
                    tpb[i] = FT.shape[i]
                    grid[i] = 1
                else:
                    while tpb[i] * (FT.shape[i] // tpb[i]) != FT.shape[i]:
                        tpb[i] -= 1
                    grid[i] = FT.shape[i] // tpb[i]
            __translate_GPU[grid, tpb](*y, sinc(array(dz) / pi).prod(), FT)

        except Exception:
            __translate_CPU(*y, sinc(array(dz) / pi).prod(), FT)

    return FT


def periodicFT(FT, y, reps, d, dy=None, pointwise=False):
    '''
    Assume FT = FourierTransform(u(x))
    This function returns:
        FourierTransform( sum_{j=0}^{reps-1} u(x+j*d) )
    Inputs:
        FT: complex ndarray
        y: mesh for the Fourier frequencies of FT
        reps: integer number of repetitions to do in each axis
        d: spacing between each repeated block in each dimension
        dy: coordinate vectors for mesh, default is dy=[(1,0,0),(0,1,0),(0,0,1)]


    Note:
        FT(u(x+d)) = \int e^{-ix\cdot y}u(x+d)
                   = \int e^{-i(x+d)\cdot y +id\cdot y} u(x+d)
                   = e^{id\cdot y} FT(u)
        FT(\sum u(x+jd)) = \sum FT(u(x+jd))
                         = FT(u) \sum [e^{id\cdot y}]^j
                         = FT[u] (e^{... jmin} - e^{...(jmax+1)})/ (1-e^{...})

sum between 0 and r-1 is (1-exp(irY))/(1-exp(iY)), Y = <d,y>
We want int_{Y-dy}^{Y+dy} (1-exp(irz))/(1-exp(iz))dz
For large r treat two integrands as `independent' to give
leading order term.
int (1-exp(irz))/(1-exp(iz)) = rz at exp(iz)=1
    '''
    dim = FT.ndim
    if not hasattr(reps, '__iter__'):
        reps = (reps,) * dim
    reps = [int(r) for r in reps]
    if not hasattr(d, '__iter__'):
        d = (d,) * dim

    # Account for trivial cases of dy
    if dy is not None:
        dy = [array(dd) for dd in dy]
        ind = [abs(dy[i]).argmax() for i in range(dim)]
#         iind = [ind.index(i) for i in range(dim)]
        if all(abs(dy[i][ind[i]]) >= (1 - 1e-6) * abs(dy[i]).sum() for i in range(dim)):
            y = [y[i] * dy[i][ind[i]] for i in range(dim)]
            d = [d[ind[i]] for i in range(dim)]
            reps = [reps[ind[i]] for i in range(dim)]
            dy = None

#     x = fromFreq(y) # to compute if wrap-around occurs
    sz = FT.shape
    if dy is None:
        for i in range(dim):
            if y[i].size < 2:
                continue

            ##### stop wrap-around error
#             reps[i] = min(reps[i], 2 * x[i].max() / abs(d[i]))
            #####

            sz = [1] * dim
            sz[i] = -1
            z = (d[i] * y[i]).reshape(sz)
            dz = abs(z.item(1) - z.item(0)) / 2

            if pointwise or abs(dz * reps[i]) < 2 * pi / 2:
                dist = abs(z - (z / (2 * pi)).round() * 2 * pi)
                ind = (dist < dz / 1e8)
                z[ind] = 1e-8  # dummy value
                phase = (1 - exp(1j * z * reps[i])) / (1 - exp(1j * z))
                phase[ind] = reps[i]

            elif reps[i] > 100 or (reps[i] % 1 != 0):  # Assymptotic expansion
                dist = abs(z - (z / (2 * pi)).round() * 2 * pi)
                z[abs(dist - dz) < 1e-8] += 1e-8
                phase = (1 - exp(1j * z * reps[i]) * sinc(reps[i] * dz / pi)
                         ) * (1 + (1j / (2 * dz)) * (log(1 - exp(1j * (z + dz)))
                                                     -log(1 - exp(1j * (z - dz)))))
            else:  # Exact sum
                phase = sum(exp(1j * z * j) * sinc(j * dz / pi)
                            for j in range(reps[i]))

            FT = FT * phase
    else:
        d = array(d).reshape(-1)
        dy = array([dd.reshape(-1) * d for dd in dy])
        dz = array([dy[i] * abs(y[i].item(min(1, y[i].size - 1)) - y[i].item(0)) for i in range(dim)])
        dz = [(dz[:, i] ** 2).sum() ** .5 / 2 for i in range(dim)]
        y = toMesh(y, dy)
        z = [ascontiguousarray(y[..., i]) for i in range(dim)]

        phase = empty(sz, dtype='c16')
        for i in range(dim):
            if dz[i] == 0:
                continue

            if pointwise or abs(dz[i] * reps[i]) < 2 * pi / 2:
#                 dist = abs(z[i] - (z[i] / (2 * pi)).round() * 2 * pi) / dz[i]
#                 ind = (dist < 1e-8)
#                 z[i][ind] = 1e-8 / reps[i]  # dummy value
#                 phase = (1 - exp(1j * z[i] * reps[i])) / (1 - exp(1j * z[i]))
# #                 phase[ind] = reps[i]
                try:
                    if USE_CPU: raise Exception
                    __periodic_point_GPU[-(-z[i].size // 128), 128](z[i].reshape(-1),
                                   reps[i], phase.reshape(-1))
                except Exception:
#                     USE_CPU(True)
                    __periodic_point_CPU(z[i].reshape(-1),
                                   reps[i], phase.reshape(-1))

            elif reps[i] > 100:  # Assymptotic expansion
                try:
                    if USE_CPU: raise Exception
                    __periodic_high_GPU[-(-z[i].size // 128), 128](z[i].reshape(-1),
                                   dz[i], sinc(reps[i] * dz[i] / pi),
                                   reps[i], phase.reshape(-1))
                except Exception:
#                     USE_CPU(True)
                    dist = abs(z[i] - (z[i] / (2 * pi)).round() * 2 * pi) / dz[i]
                    z[i][abs(dist - dz[i]) < 1e-8] += 1e-8 / reps[i]
                    phase = (1 - exp(1j * z[i] * reps[i]) * sinc(reps[i] * dz[i] / pi)
                             ) * (1 + (1j / (2 * dz[i])) * (log(1 - exp(1j * (z[i] + dz[i])))
                                                         -log(1 - exp(1j * (z[i] - dz[i])))))
            else:  # Exact sum
#                 phase = sum(exp(1j * z[i] * j) * sinc(j * dz[i] / pi)
#                             for j in range(reps[i]))
                try:
                    if USE_CPU: raise Exception
                    __periodic_GPU[-(-z[i].size // 128), 128](z[i].reshape(-1),
                                   array([sinc(j * dz[i] / pi)
                                          for j in range(reps[i])]),
                                   reps[i], phase.reshape(-1))
                except Exception:
#                     USE_CPU(True)
                    __periodic_CPU(z[i].reshape(-1),
                                   array([sinc(j * dz[i] / pi)
                                          for j in range(reps[i])]),
                                   reps[i], phase.reshape(-1))
            FT = FT * phase
    return FT


def cutoffFT(FT, y, box, dy=None):
# TODO: something smarter with cutoffs?
# If the Fourier transform is too small then convolution kernel does not decay fast enough
# Can cut off in spatial domain so long as x covers right range...
    '''
    Assume FT = FourierTransform(u(x))
    This function returns:
        FourierTransform( 1_{x\in box} u(x) )
    Inputs:
        FT: complex ndarray
        y: mesh for the Fourier frequencies of FT
        box: integer number of repetitions to do in each axis
        dy: coordinate vectors for mesh, default is dy=[(1,0,0),(0,1,0),(0,0,1)]


    All zero-filling takes place in real domain. If the grids do not align then
    we use IFT[FT[u](Ay)](x) = u(A^{-T}x)/|A|, i.e. u(x)/|A| = IFT[FT](A^Tx)
    '''
    dim = FT.ndim

    # Account for trivial cases of dy
    if dy is not None:
        dy = array(dy)
        ind = [abs(dy[i]).argmax() for i in range(dim)]
        if all(abs(dy[i][ind[i]]) >= (1 - 1e-6) * abs(dy[i]).sum() for i in range(dim)):
            y = [y[i] * dy[i][ind[i]] for i in range(dim)]
            newb = [box[i] for i in range(dim)]
            box, dy = newb, None

    from numpy.linalg import inv
    dx = eye(dim) if dy is None else inv(dy).T
    x = fromFreq(y)
    t = sum((x[i][-1] + x[i][0]) * dx[i] for i in range(dim))
    t = (array(box).sum(1) - t) / 2
    t = t if dy is None else dy.dot(t)
    x = [x[i] + t[i] for i in range(dim)]
#     x = [arange(x[i][0] - (x[i][-1] - x[i][0]) / 2,
#                 x[i][-1] + (x[i][-1] - x[i][0]) / 2 + 1.1 * (x[i][1] - x[i][0]),
#                 x[i][1] - x[i][0]) for i in range(dim)]

    ft, ift = getDFT(x, y)
    real = ift(FT)

    sz = FT.shape
    y = toMesh(y, dy, sz)

    if dy is None:
        real[x[0] < box[0][0], ...] = 0
        real[x[0] > box[0][1], ...] = 0
        real[:, x[1] < box[1][0], ...] = 0
        real[:, x[1] > box[1][1], ...] = 0
        if dim == 3:
            real[:, :, x[2] < box[2][0]] = 0
            real[:, :, x[2] > box[2][1]] = 0
        elif dim > 3:
            raise ValueError
    else:
#         X = x
        x = toMesh(x, dx)

#         from matplotlib import pyplot as plt
#         plt.clf()
#         plt.subplot(121)
#         plt.title('Before')
#         print('here')
#         plt.imshow(abs(real.sum(-1)),
#                    extent=[X[0][0], X[0][-1], X[1][0], X[1][-1]])

        for i in range(dim):
            real[x[..., i] < box[i][0]] = 0
            real[x[..., i] > box[i][1]] = 0

#         plt.subplot(122)
#         plt.title('After')
#         plt.imshow(abs(real.sum(-1)),
#                    extent=[X[0][0], X[0][-1], X[1][0], X[1][-1]])
#         plt.draw()
#         plt.pause(.1)
#         plt.show()
#         exit()
    return ft(real)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import numpy as np
    from crystal import getA

    def test_DFT(rX, rY):
        x, y = getFTpoints(len(rX), rX=rX, rY=rY)
        axes = 0 if len(x) == 1 else None

        f, g = getA(0, returnFunc=True)
        f, g = f(toMesh(x) + .3), g(toMesh(y)) * exp(+.3 * 1j * toMesh(y).sum(-1))

        ft, ift = getDFT(x, y)

#         ax = tuple([i for i in range(1, len(x))])
#         plt.subplot(221); plt.plot(f.sum(axis=ax))
#         plt.subplot(222); plt.plot(g.sum(axis=ax).real)
#         plt.plot(g.sum(axis=ax).imag)
#         plt.subplot(223); plt.plot(ift(g).sum(axis=ax).real)
#         plt.plot(ift(g).sum(axis=ax).imag)
#         plt.subplot(224); plt.plot(ft(f).sum(axis=ax).real)
#         plt.plot(ft(f).sum(axis=ax).imag)
#         plt.show()
#         exit()

        np.testing.assert_allclose(g, ft(f, axes=axes), 1e-4, 1e-4)
        np.testing.assert_allclose(f, ift(g, axes=axes), 1e-4, 1e-4)

    for args in [([2], 500), ([2] * 2, 500), ([2] * 3, 500), ]:
        test_DFT(*args)

    print('finished')
    exit()

    from matplotlib import pyplot as plt

    x, y = getFTpoints(1, n=256, rX=10, dY=.2, rY=20)
    dft, idft = getDFT(x, y)
    x, y = x[0], y[0]

    af = exp(-(x - 1) ** 2 / 2)
    aFf = (2 * pi) ** .5 * exp(-y ** 2 / 2 - 1j * y)
    dFf, df = dft(af), idft(aFf)

    print('f: ', abs(af - df).max())
    print('Ff: ', abs(aFf - dFf).max())

    plt.subplot(221)
    plt.plot(x, af)
    plt.title('f')
    plt.subplot(222)
    plt.plot(x, df.real)
    plt.plot(x, df.imag)
    plt.title('idft(FT)')
    plt.subplot(223)
    plt.plot(y, aFf.real)
    plt.plot(y, aFf.imag)
    plt.title('FT')
    plt.subplot(224)
    plt.plot(y, dFf.real)
    plt.plot(y, dFf.imag)
    plt.title('dft(f)')
    plt.show()

    # TODO: write some tests for different values of s
    pass
