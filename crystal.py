'''
Created on 13 Nov 2018

@author: Rob Tovey
'''
c_abs = abs
from numpy import (exp, pi, sqrt, zeros, empty, array, concatenate, log,
                   isscalar, unique, ascontiguousarray, eye, minimum,
                   maximum, sinc, abs, ceil, random, require, linspace, arange,
    ones)
from numpy.linalg import inv, solve, det
import numba
from numba import cuda
from math import (exp as c_exp, cos as c_cos, sin as c_sin,
                  sqrt as c_sqrt, floor, ceil as c_ceil, erf as c_erf)
from utils import rebin, toc, progressBar, toMesh, rot3d, isLoud, convolve, \
    SILENT, USE_CPU, rebin_block
from FourierTransform import periodicFT, toFreq, getDFT, cutoffFT
FTYPE, CTYPE = 'f8', 'c16'
c_FTYPE = numba.float64 if True else numba.float32
LOG0 = -15


def _norm2(x): return (abs(x) ** 2).sum()


def _norm(x): return (abs(x) ** 2).sum() ** .5


_pointwise_exp = False
if _pointwise_exp:

    @cuda.jit(device=True, inline=True)
    def __exp(b, x0, x1, x2, dx0, dx1, dx2, precomp, h):
        i = int(floor(h * b * b * __GPU_norm(x0, x1, x2)))
        if i >= precomp.size:
            return 0
        else:
            return precomp[i]
#         return c_exp(-b * b * __GPU_norm(x0, x1, x2))

    @numba.jit(fastmath=True, nopython=True)
    def __CPUexp(b, x0, x1, x2, dx0, dx1, dx2):
        return c_exp(-b * b * __CPU_norm(x0, x1, x2))

    @cuda.jit(device=True, inline=True)
    def __atom(a, b, x0, x1, x2, dx, precomp, h):
        n = __GPU_norm(x0, x1, x2)
#         s = 0
#         for i in range(a.size):
#             s += a[i] * c_exp(-b[i] * b[i] * n)
#         return s

        if n >= h[0]:
            return 0
        else:
            n = h[1] * c_sqrt(n)
            i = int(n)
            n -= i
            return (1 - n) * precomp[i] + n * precomp[i + 1]

    @numba.jit(fastmath=True, nopython=True)
    def __CPUatom(a, b, x0, x1, x2, dx, precomp, h):
        n = __CPU_norm(x0, x1, x2)
        if n >= h[0]:
            return 0
        else:
            n = h[1] * c_sqrt(n)
            i = int(n)
            n -= i
            return (1 - n) * precomp[i] + n * precomp[i + 1]

else:
    __exp_scale = pi ** 1.5 / 2 ** 6

    @cuda.jit(device=True, inline=True)
    def __erf(x, precomp):
        if x >= 0:
            if x < precomp.size:
                return precomp[int(x)]
            else:
                return 1
        elif x > -precomp.size:
            return -precomp[int(-x)]
        else:
            return -1

    @cuda.jit(device=True, inline=True)
    def __exp(b, x0, x1, x2, dx0, dx1, dx2, pc, h):
        ''' \int_{x-dx}^{x+dx} exp(-b^2|y|^2)
                = sqrt{pi}/(2b)(erf(b(x+dx))-erf(b(x-dx)))
            The 1/dx scale 'averages' over the voxel
            '''
        v = __exp_scale
        if dx0 > 0:
            v *= (__erf(h * b * (x0 + dx0), pc) - __erf(h * b * (x0 - dx0), pc)) / (b * dx0)
        else:
            v *= 2 / b
        if dx1 > 0:
            v *= (__erf(h * b * (x1 + dx1), pc) - __erf(h * b * (x1 - dx1), pc)) / (b * dx1)
        else:
            v *= 2 / b
        if dx2 > 0:
            v *= (__erf(h * b * (x2 + dx2), pc) - __erf(h * b * (x2 - dx2), pc)) / (b * dx0)
        else:
            v *= 2 / b

#         v = __exp_scale
#         if dx0 > 0:
#             v *= (c_erf(b * (x0 + dx0)) - c_erf(b * (x0 - dx0))) / (b * dx0)
#         else:
#             v *= 2 / b
#         if dx1 > 0:
#             v *= (c_erf(b * (x1 + dx1)) - c_erf(b * (x1 - dx1))) / (b * dx1)
#         else:
#             v *= 2 / b
#         if dx2 > 0:
#             v *= (c_erf(b * (x2 + dx2)) - c_erf(b * (x2 - dx2))) / (b * dx2)
#         else:
#             v *= 2 / b

#         if dx0 > 0:
#             v *= (c_erf(b * (x0 + dx0)) - c_erf(b * (x0 - dx0))) / b
#         else:
#             v *= c_exp(-b * b * x0 * x0) * __exp_scale1
#         if dx1 > 0:
#             v *= (c_erf(b * (x1 + dx1)) - c_erf(b * (x1 - dx1))) / b
#         else:
#             v *= c_exp(-b * b * x1 * x1) * __exp_scale1
#         if dx2 > 0:
#             v *= (c_erf(b * (x2 + dx2)) - c_erf(b * (x2 - dx2))) / b
#         else:
#             v *= c_exp(-b * b * x2 * x2) * __exp_scale1

        return v

    @numba.jit(fastmath=True, nopython=True)
    def __CPUexp(b, x0, x1, x2, dx0, dx1, dx2):
        v = __exp_scale
        if dx0 > 0:
            v *= (c_erf(b * (x0 + dx0)) - c_erf(b * (x0 - dx0))) / (b * dx0)
        else:
            v *= 2 / b
        if dx1 > 0:
            v *= (c_erf(b * (x1 + dx1)) - c_erf(b * (x1 - dx1))) / (b * dx1)
        else:
            v *= 2 / b
        if dx2 > 0:
            v *= (c_erf(b * (x2 + dx2)) - c_erf(b * (x2 - dx2))) / (b * dx2)
        else:
            v *= 2 / b
        return v

    @cuda.jit(device=True, inline=True)
    def __atom(a, b, x0, x1, x2, dx, pc, h):
#         x0p, x0m = (x0 + dx[0]), (x0 - dx[0])
#         x1p, x1m = (x1 + dx[1]), (x1 - dx[1])
#         x2p, x2m = (x2 + dx[2]), (x2 - dx[2])
#         s = 0
#         for i in range(a.size):
#             v = __exp_scale
#             if dx[0] > 0:
#                 v *= (c_erf(b[i] * x0p) - c_erf(b[i] * x0m)) / (b[i] * dx[0])
#             else:
#                 v *= 2 / b[i]
#             if dx[1] > 0:
#                 v *= (c_erf(b[i] * x1p) - c_erf(b[i] * x1m)) / (b[i] * dx[1])
#             else:
#                 v *= 2 / b[i]
#             if dx[2] > 0:
#                 v *= (c_erf(b[i] * x2p) - c_erf(b[i] * x2m)) / (b[i] * dx[2])
#             else:
#                 v *= 2 / b[i]
#
#             s += a[i] * v
        x0, x1, x2 = c_abs(x0), c_abs(x1), c_abs(x2)
        if x0 > h[0, 0] or x1 > h[0, 1] or x2 > h[0, 2]:
            return 0

        x0, x1, x2 = h[1, 0] * x0, h[1, 1] * x1, h[1, 2] * x2
        i0, i1, i2 = int(x0), int(x1), int(x2)
        x0, x1, x2 = x0 - i0, x1 - i1, x2 - i2
        X0, X1, X2 = 1 - x0, 1 - x1, 1 - x2
        s = 0
        for i in range(pc.shape[0]):
            v = __exp_scale
            v *= X0 * pc[i, 0, i0] + x0 * pc[i, 0, i0 + 1]
            v *= X1 * pc[i, 1, i1] + x1 * pc[i, 1, i1 + 1]
            v *= X2 * pc[i, 2, i2] + x2 * pc[i, 2, i2 + 1]

            s += v

        return s

    @numba.jit(fastmath=True, nopython=True)
    def __CPUatom(a, b, x0, x1, x2, dx, pc, h):
        x0, x1, x2 = c_abs(x0), c_abs(x1), c_abs(x2)
        if x0 > h[0, 0] or x1 > h[0, 1] or x2 > h[0, 2]:
            return 0

        x0, x1, x2 = h[1, 0] * x0, h[1, 1] * x1, h[1, 2] * x2
        i0, i1, i2 = int(x0), int(x1), int(x2)
        x0, x1, x2 = x0 - i0, x1 - i1, x2 - i2
        X0, X1, X2 = 1 - x0, 1 - x1, 1 - x2
        s = 0
        for i in range(pc.shape[0]):
            v = __exp_scale
            v *= X0 * pc[i, 0, i0] + x0 * pc[i, 0, i0 + 1]
            v *= X1 * pc[i, 1, i1] + x1 * pc[i, 1, i1 + 1]
            v *= X2 * pc[i, 2, i2] + x2 * pc[i, 2, i2 + 1]

            s += v

        return s


@cuda.jit(device=True, inline=True)
def __GPU_norm(x, y, z): return x * x + y * y + z * z


@cuda.jit
def __GPU_density_sublist(x0, x1, x2, xmin,
                  loc, sublist, r, a, d, B, precomp, h, out):

    i0, i1, i2 = cuda.grid(3)
    if i0 >= x0.size or i1 >= x1.size or i2 >= x2.size:
        return
    X0, X1, X2 = x0[i0], x1[i1], x2[i2]
    Y0, Y1, Y2 = 0, 0, 0

    bin0 = int((X0 - xmin[0]) / r[0])
    bin1 = int((X1 - xmin[1]) / r[1])
    bin2 = int((X2 - xmin[2]) / r[2])
    if bin0 >= sublist.shape[0] or bin1 >= sublist.shape[1] or bin2 >= sublist.shape[2]:
        return

    Sum = 0
    for bb in range(sublist.shape[3]):
        j0 = sublist[bin0, bin1, bin2, bb]
        if j0 < 0:
            break
#     for j0 in range(loc.shape[0]):
        Y0 = loc[j0, 0] - X0
        Y1 = loc[j0, 1] - X1
        Y2 = loc[j0, 2] - X2
        Sum += __atom(a, B, Y0, Y1, Y2, d, precomp, h)
    out[i0, i1, i2] = Sum


@cuda.jit
def __GPU_FT(x0, x1, x2, dx0, dx1, dx2, loc, a, b, d, B, precomp, h, out):
    i0, i1, i2 = cuda.grid(3)
    X0 = x0[i0] * dx0[0] + x1[i1] * dx1[0] + x2[i2] * dx2[0]
    X1 = x0[i0] * dx0[1] + x1[i1] * dx1[1] + x2[i2] * dx2[1]
    X2 = x0[i0] * dx0[2] + x1[i1] * dx1[2] + x2[i2] * dx2[2]

    scale = __atom(a, B, X0, X1, X2, d, precomp, h)

    IP, Sum = 0, complex(0, 0)
    for j0 in range(loc.shape[0]):
        IP = loc[j0, 0] * X0 + loc[j0, 1] * X1 + loc[j0, 2] * X2
        Sum += complex(scale * c_cos(IP), -scale * c_sin(IP))

    out[i0, i1, i2] = Sum


@cuda.jit
def __GPU_FT2D(x0, x1, x2, dx0, dx1, dx2, loc, a, b, d, B, precomp, h, out):
    i0, i1 = cuda.grid(2)
    X0 = x0[i0] * dx0[0] + x1[i1] * dx1[0] + x2 * dx2[0]
    X1 = x0[i0] * dx0[1] + x1[i1] * dx1[1] + x2 * dx2[1]
    X2 = x0[i0] * dx0[2] + x1[i1] * dx1[2] + x2 * dx2[2]

    scale = __atom(a, B, X0, X1, X2, d, precomp, h)

    IP, Sum = 0, complex(0, 0)
    for j0 in range(loc.shape[0]):
        IP = loc[j0, 0] * X0 + loc[j0, 1] * X1 + loc[j0, 2] * X2
        Sum += complex(scale * c_cos(IP), -scale * c_sin(IP))

    out[i0, i1] = Sum


@cuda.jit
def __GPU_FT_sphere(x0, x1, C, dx0, dx1, dx2, loc, a, b, d, B, precomp, h, out):
    # dx2 has already been normalised
    i0, i1 = cuda.grid(2)
    X0 = x0[i0] * dx0[0] + x1[i1] * dx1[0]
    X1 = x0[i0] * dx0[1] + x1[i1] * dx1[1]
    X2 = x0[i0] * dx0[2] + x1[i1] * dx1[2]
    x2 = C * C - __GPU_norm(X0, X1, X2)
    if x2 >= 0:
        x2 = C - c_sqrt(x2)
    else:
        out[i0, i1] = complex(0, 0)
        return

    X0 += x2 * dx2[0]
    X1 += x2 * dx2[1]
    X2 += x2 * dx2[2]
    scale = __atom(a, B, X0, X1, X2, d, precomp, h)

    IP, Sum = 0, complex(0, 0)
    for j0 in range(loc.shape[0]):
        IP = loc[j0, 0] * X0 + loc[j0, 1] * X1 + loc[j0, 2] * X2
        Sum += complex(scale * c_cos(IP), -scale * c_sin(IP))

    out[i0, i1] = Sum


@numba.jit(parallel=True, nopython=True)
def __CPU_norm(x, y, z): return x * x + y * y + z * z


@numba.jit(parallel=True, fastmath=True, nopython=True)
def __CPU_density(x0, x1, x2, xmin,
                  loc, sublist, r, a, d, B, precomp, h, out):
    X0 = x0
    bin0 = int(floor((X0 - xmin[0]) / r[0]))
    if bin0 < 0 or bin0 >= sublist.shape[0]:
        return
    for i1 in numba.prange(x1.size):
        X1 = x1[i1]
        bin1 = int(floor((X1 - xmin[1]) / r[1]))
        if bin1 < 0 or bin1 >= sublist.shape[1]:
            continue
        for i2 in range(x2.size):
            X2 = x2[i2]
            bin2 = int(floor((X2 - xmin[2]) / r[2]))
            if bin2 < 0 or bin2 >= sublist.shape[2]:
                continue
#             Y0, Y1, Y2 = 0, 0, 0

            Sum = 0
            for bb in range(sublist.shape[3]):
                j0 = sublist[bin0, bin1, bin2, bb]
                if j0 < 0:
                    break

                Y0 = loc[j0, 0] - X0
                Y1 = loc[j0, 1] - X1
                Y2 = loc[j0, 2] - X2
                Sum += __CPUatom(a, B, Y0, Y1, Y2, d, precomp, h)
            out[i1, i2] = Sum


@numba.jit(parallel=True, fastmath=True, nopython=True)
def __CPU_FT(x0, x1, x2, dx0, dx1, dx2, loc, a, b, d, B, precomp, h, out):
    for i1 in numba.prange(x1.size):
        for i2 in range(x2.size):
            X0 = x0 * dx0[0] + x1[i1] * dx1[0] + x2[i2] * dx2[0]
            X1 = x0 * dx0[1] + x1[i1] * dx1[1] + x2[i2] * dx2[1]
            X2 = x0 * dx0[2] + x1[i1] * dx1[2] + x2[i2] * dx2[2]

            scale = __CPUatom(a, B, X0, X1, X2, d, precomp, h)

            IP, Sum = 0, complex(0, 0)
            for j0 in range(loc.shape[0]):
                IP = loc[j0, 0] * X0 + loc[j0, 1] * X1 + loc[j0, 2] * X2
                Sum += complex(scale * c_cos(IP), -scale * c_sin(IP))

            out[i1, i2] = Sum


pass
##################################################
# Building Crystals
##################################################


def getA(Z, returnFunc=True):
    '''
    This data table is from 'Robust Parameterization of
    Elastic and Absorptive Electron Atomic Scattering
    Factors' by L.-M. Peng, G. Ren, S. L. Dudarev and
    M. J. Whelan, 1996
    '''
    Z = unique(Z).astype(int)
    if Z.size > 1:
        raise ValueError('Only 1 atom can generated at once')
    else:
        Z = Z[0]
    if Z == 0:
        # Mimics a Dirac spike
        a = [1] * 5 + [.1] * 5
    elif Z == 1:
        a = [0.0349, 0.1201, 0.1970, 0.0573, 0.1195,
             0.5347, 3.5867, 12.3471, 18.9525, 38.6269]
    elif Z == 2:
        a = [0.0317, 0.0838, 0.1526, 0.1334, 0.0164,
             0.2507, 1.4751, 4.4938, 12.6646, 31.1653]
    elif Z == 3:
        a = [0.0750, 0.2249, 0.5548, 1.4954, 0.9354,
             0.3864, 2.9383, 15.3829, 53.5545, 138.7337]
    elif Z == 4:
        a = [0.0780, 0.2210, 0.6740, 1.3867, 0.6925,
             0.3131, 2.2381, 10.1517, 30.9061, 78.3273]
    elif Z == 5:
        a = [0.0909, 0.2551, 0.7738, 1.2136, 0.4606,
             0.2995, 2.1155, 8.3816, 24.1292, 63.1314]
    elif Z == 6:
        a = [0.0893, 0.2563, 0.7570, 1.0487, 0.3575,
             0.2465, 1.7100, 6.4094, 18.6113, 50.2523]
    elif Z == 7:
        a = [0.1022, 0.3219, 0.7982, 0.8197, 0.1715,
             0.2451, 1.7481, 6.1925, 17.3894, 48.1431]
    elif Z == 8:
        a = [0.0974, 0.2921, 0.6910, 0.6990, 0.2039,
             0.2067, 1.3815, 4.6943, 12.7105, 32.4726]
    elif Z == 9:
        a = [0.1083, 0.3175, 0.6487, 0.5846, 0.1421,
             0.2057, 1.3439, 4.2788, 11.3932, 28.7881]
    elif Z == 10:
        a = [0.1269, 0.3535, 0.5582, 0.4674, 0.1460,
             0.2200, 1.3779, 4.0203, 9.4934, 23.1278]
    elif Z == 11:
        a = [0.2142, 0.6853, 0.7692, 1.6589, 1.4482,
             0.3334, 2.3446, 10.0830, 48.3037, 138.2700]
    elif Z == 12:
        a = [0.2314, 0.6866, 0.9677, 2.1882, 1.1339,
             0.3278, 2.2720, 10.9241, 39.2898, 101.9748]
    elif Z == 13:
        a = [0.2390, 0.6573, 1.2011, 2.5586, 1.2312,
             0.3138, 2.1063, 10.4163, 34.4552, 98.5344]
    elif Z == 14:
        a = [0.2519, 0.6372, 1.3795, 2.5082, 1.0500,
             0.3075, 2.0174, 9.6746, 29.3744, 80.4732]
    elif Z == 15:
        a = [0.2548, 0.6106, 1.4541, 2.3204, 0.8477,
             0.2908, 1.8740, 8.5176, 24.3434, 63.2996]
    elif Z == 16:
        a = [0.2497, 0.5628, 1.3899, 2.1865, 0.7715,
             0.2681, 1.6711, 7.0267, 19.5377, 50.3888]
    elif Z == 17:
        a = [0.2443, 0.5397, 1.3919, 2.0197, 0.6621,
             0.2468, 1.5242, 6.1537, 16.6687, 42.3086]
    elif Z == 18:
        a = [0.2385, 0.5017, 1.3428, 1.8899, 0.6079,
             0.2289, 1.3694, 5.2561, 14.0928, 35.5361]
    elif Z == 19:
        a = [0.4115, -1.4031, 2.2784, 2.6742, 2.2162,
             0.3703, 3.3874, 13.1029, 68.9592, 194.4329]
    elif Z == 20:
        a = [0.4054, 1.3880, 2.1602, 3.7532, 2.2063,
             0.3499, 3.0991, 11.9608, 53.9353, 142.3892]

    a, b = array(a[:5], dtype=FTYPE), array(a[5:], dtype=FTYPE)
    b /= (4 * pi) ** 2  # Weird scaling in initial paper

    def myAtom(x):
        dim = x.shape[-1]
        x = abs(x * x).sum(-1)
        y = 0
        for i in range(5):
            y += (a[i] / (4 * pi * b[i]) ** (dim / 2)) * exp(-x / (4 * b[i]))

        return y

    def myAtomFT(x):
        x = abs(x * x).sum(-1)
        y = 0
        for i in range(5):
            y += a[i] * exp(-b[i] * x)

        return y

    if returnFunc:
        return myAtom, myAtomFT
    else:
        return a, b


def getDiscretisation(loc, Z, GPU=False):
    '''
    u(x) = sum_j a_Z(x-loc_j)
    FTu(y) = FT(a_Z)(y)sum_j exp(-iloc_j\cdot y)
    '''
    dim = loc.shape[-1]
    z = unique(Z).astype(int)
    if z.size > 1:
        funcs = [getDiscretisation(loc[Z == zz], zz, GPU) for zz in z]

        def density(x, dx=None): return sum(f[0](x, dx) for f in funcs)

        def FTdensity(x, dx=None, C=None): return sum(f[1](x, dx, C) for f in funcs)

    else:
        Z = z[0]
        a, b = getA(Z, False)
        B = (1 / (4 * b)).astype(FTYPE)
        A = (a * (B / pi) ** (dim / 2)).astype(FTYPE)

        loc = require(loc.reshape([-1, dim]), requirements='AC')

        def doPrecomp(x, a, b, d):
            if _pointwise_exp:
                f = lambda x: sum(a[i] * c_exp(-b[i] * x ** 2) for i in range(a.size))
                Rmax = .1
                while f(Rmax) > exp(LOG0) * f(0):
                    Rmax *= 1.1
                h = max(Rmax / 200, max(d) / 10)
                pms = array([Rmax ** 2, 1 / h], dtype=FTYPE)
                precomp = array([f(x) for x in arange(0, Rmax + 2 * h, h)], dtype=FTYPE)

# #                 v0 = k+1/c, c = 1/(v0-k)
# #                 v1 = k+1/(r+c), r = 1/(v1-k)-1/(v0-k) = (v0-v1)/(v0-k)/(v1-k)
# #                 v2 = k+1/(Mr+c-r), Mr+c-r = 1/(v2-k)
# #                 v3 = k+1/(Mr+c), Mr+c = 1/(v3-k), M = 1/(v3-k)/r-c/r
# #                 r = (v0-v1)/(v0-k)/(v1-k) = (v2-v3)/(v2-k)/(v3-k)
# #                 -> v4(k-v0)(k-v1) = (k-v2)(k-v3), v4 = (v2-v3)/(v0-v1)
# #                 -> (v4-1)k^2 + (v2+v3-v4(v0+v1))k + v0v1v4-v2v3 = 0
#                 v = [0, max(1e-3, h ** 2), Rmax ** 2 / 4, Rmax ** 2]
#                 tmp = (v[2] - v[3]) / (v[0] - v[1])
#                 tmp = [tmp - 1, v[2] + v[3] - tmp * (v[0] + v[1]), v[0] * v[1] * tmp - v[2] * v[3]]
#                 k = (-tmp[1] - sqrt(tmp[1] ** 2 - 4 * tmp[0] * tmp[2])) / (2 * tmp[0])
#                 c = 1 / (v[0] - k)
#                 r = (v[0] - v[1]) / (v[0] - k) / (v[1] - k)
#                 M = (1 / (v[3] - k) - c) / r
#                 r, M = (M * r) / ceil(M), ceil(M)
#
#                 g2p = lambda x: k + 1 / (r * x + c)
#                 p2g = lambda x: 1 / (r * (x ** 2 - k)) - c / r
#                 pms = array([Rmax ** 2, k, r, c / r], dtype='float32')

            else:

                def f(i, j, x):
                    A = a[i] ** (1 / 3)  # factor spread evenly over 3 dimensions
                    B = b[i] ** .5
                    if d[j] == 0:
                        return A * 2 / B
                    return A * (c_erf(B * (x + d[j] / 2))
                                -c_erf(B * (x - d[j] / 2))) / (2 * d[j] * B) * pi ** .5

                h = [D / 10 for D in d]
                Rmax = ones([a.size, 3], dtype=FTYPE) / 10
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
                precomp, grid = zeros([a.size, 3, L], dtype=FTYPE), arange(L)
                for i in range(a.size):
                    for j in range(3):
                        h[j] = Rmax[:, j].max() / (L - 2)
                        precomp[i, j] = array([f(i, j, x * h[j]) for x in grid], dtype=FTYPE)
                pms = array([Rmax.max(0), 1 / array(h)], dtype=FTYPE)
                Rmax = Rmax.max(0).min()

#                 pms = 1 / h[0]
#                 precomp = array([c_erf(x * h[0]) for x in grid], dtype=FTYPE)

#                 h, Rmax = min(h), Rmax.max()
#                 f = lambda x: sum(A[i] * (c_erf(BB[i] * (x + d[0] / 2)) - c_erf(BB[i] * (x - d[0] / 2))) / (d[0] * BB[i] * 2) * pi ** .5 for i in range(a.size))
#
#                 g2p = lambda x: x
#                 p2g = lambda x: x / h
#
# #
#                 def interp(x):
#                     I, i = p2g(x), int(p2g(x))
#                     return (i + 1 - I) * precomp[i] + (I - i) * precomp[i + 1] if i < precomp.size - 1 else 0
# #                     return exp((i + 1 - I) * log(precomp[i]) + (I - i) * log(precomp[i + 1])) if i < precomp.size - 1 else 0
#
#                 fine_grid = arange(0, 25, h)
#                 fine_grid = fine_grid[g2p(fine_grid) <= Rmax]
#                 precomp = array([f(g2p(x)) for x in fine_grid] + [f(Rmax)], dtype=FTYPE)
#                 f = lambda x: sum(A[i] * c_exp(-B[i] * x ** 2) for i in range(a.size))
#
#                 from matplotlib import pyplot as plt
#                 grid = arange(1e-32, .5, 0.001)
#                 plt.subplot(121)
# #                 plt.plot(grid, [precomp[min(precomp.size - 1, max(0, int(round(p2g(x)))))] for x in grid], 'r.')
#                 plt.plot(grid, [interp(x) for x in grid], 'r')
#                 plt.plot(grid, [f(x) for x in grid])
#                 plt.title(str(precomp.size))
#
#                 grid = arange(1e-32, Rmax, 0.001)
#                 tmp = array([interp(x) for x in grid]), array([f(x) for x in grid])
#                 plt.subplot(122)
#                 plt.semilogy(grid, [precomp[min(precomp.size - 1, max(0, int(round(p2g(x)))))] for x in grid], 'r.')
#                 plt.semilogy(grid, tmp[0], 'r')
#                 plt.semilogy(grid, tmp[1])
#                 plt.title(str((log(((tmp[0] - tmp[1]) ** 2).sum()) - log((tmp[1] ** 2).sum())) / log(10)))
#                 plt.show()
#                 exit()

            return precomp, pms, Rmax

        def doBinning(x, loc, Rmax, d, GPU):
            # Bin the atoms
            k = int(Rmax / max(d)) + 1
            try:
                if (not GPU) or USE_CPU: raise Exception
                cuda.current_context().deallocations.clear()
                mem = cuda.current_context().get_memory_info()[0] / 2
            except Exception:
                mem = 1e12

            while k > 0:
                r = array([2e5 if D == 0 else max(Rmax / k, D) for D in d], dtype='f4')
                subList = None
                try:
                    subList = rebin(x, loc, r, k, mem=.25 * mem)
                    if subList.size * subList.itemsize > .25 * mem:
                        subList = None  # treat like memory error
                except MemoryError:
                    pass

                if subList is None and k == 1:
#                     raise MemoryError('List of atoms is too large to be stored on device')
                    pass  # Memory error at smallest k
                elif subList is None:
                    k -= 1; continue  # No extra information
                else:
                    break

                return None, r, mem

            return subList, r, mem

        def doGrid(sz, tpb=None):
            dim = len(sz)

            tpb = 8 if tpb is None else tpb
            tpb = [tpb] * len(sz) if isscalar(tpb) else list(tpb)
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

        def density(x, dx=None, GPU=GPU):
            if dx is not None:
                raise NotImplementedError('The rebinning algorithm currently does not support non-None dx')

            out = empty([X.size for X in x], dtype=FTYPE)
            if out.size == 0:
                return 0 * out

            # Precompute extra variables
            d = array([abs(X.item(1) - X.item(0)) if X.size > 1 else 0 for X in x])

            precomp, pms, Rmax = doPrecomp(x, A, B, d)

            xmin = array([X.item(0) if X.size > 1 else -1e5 for X in x], dtype=FTYPE)

            subList, r, mem = doBinning(x, loc, Rmax, d, GPU)

            if subList is None:
                print('splitting crystal into 2')
                Slice = [None] * 3
                f = getDiscretisation(loc, Z, GPU)[0]
                for i in range(2):
                    Slice[0] = slice(out.shape[0] // 2, None) if i else slice(out.shape[0] // 2)
#                     Slice[0] = slice(None) if i else slice(0)
                    for j in range(2):
                        Slice[1] = slice(out.shape[1] // 2, None) if j else slice(out.shape[1] // 2)
#                         Slice[1] = slice(None) if j else slice(0)
                        for k in range(2):
#                             Slice[2] = slice(out.shape[2] // 2, None) if k else slice(out.shape[2] // 2)
                            Slice[2] = slice(None) if k else slice(0)
                            out[tuple(Slice)] = f([x[t][Slice[t]] for t in range(3)], dx)
                return out

            elif subList.size == 0:
                out.fill(0)
                return out

            # Define the parallel grid
            if GPU:
                n = x[0].size / 40 if isLoud() else x[0].size
                n = max(1, int(ceil(min(n, mem / (2 * out[0].size * out.itemsize)) - 1e-5)))
                grid, tpb = doGrid((n,) + out.shape[-2:], 8)

            if isLoud():
                print('Starting discretisation ...', flush=True)

            notComputed = True
            if GPU and not USE_CPU:
                try:
                    ssubList, lloc = cuda.to_device(subList, stream=cuda.stream()), cuda.to_device(loc, stream=cuda.stream())
                    i, tic = 0, toc()
                    for j in random.permutation(int(ceil(x[0].size / n))):
                        bins = j * n, (j + 1) * n
                        __GPU_density_sublist[grid, tpb](
                            x[0][bins[0]:bins[1] + 1], x[1], x[2], xmin, lloc, ssubList,
                            r, A, d, sqrt(B), precomp, pms, out[bins[0]:bins[1] + 1])
                        i += 1
                        progressBar(i, ceil(x[0].size / n), tic)
                    notComputed = False
                except Exception:
                    USE_CPU(True)

            if notComputed:
                tic = toc()
                for i in range(x[0].size):
                    __CPU_density(
                        x[0][i], x[1], x[2], xmin, loc, subList,
                        r, A, d, sqrt(B), precomp, pms, out[i])
                    progressBar(i + 1, x[0].size, tic)
            if isLoud():
                print('finished.')

            return out

        def FTdensity(x, dx=None, C=None, GPU=GPU):
            STRETCH = ((len(x) == 2) and (C is None))
            if STRETCH:
                x = (x[0], x[1], array([0], dtype=FTYPE))
            if dx is None:
                dx = (array((1, 0, 0), dtype=FTYPE),
                      array((0, 1, 0), dtype=FTYPE),
                      array((0, 0, 1), dtype=FTYPE))
            elif len(dx) == 2:
                dx = (dx[0], dx[1], array((0, 0, 1), dtype=FTYPE))
            dx = tuple(ascontiguousarray(d.reshape(-1)) for d in dx)
            dim = 3 if C is None else 2
            out = empty([X.size for X in x], dtype=CTYPE)
            if out.size == 0:
                return 0 * out

            # Precompute extra variables
            d = array([abs(X.item(1) - X.item(0)) if X.size > 1 else 0 for X in x])
            precomp, pms, mem = doPrecomp(x, a, b, d)
            if GPU:
                grid, tpb = doGrid(out.shape[:dim])

            if isLoud():
                print('Starting FT discretisation ...', flush=True)
            if C is not None:
                raise NotImplementedError
                # TODO: need a CPU version if this is left in
                __GPU_FT_sphere[grid, tpb](
                    x[0], x[1], C, dx[0], dx[1], dx[2] / _norm(dx[2]),
                    loc, a, b, d, sqrt(b), out)

            notComputed = True
            if GPU and not USE_CPU:
                try:
                    if (not isLoud()) and out.size * out.itemsize < .8 * mem:
                        __GPU_FT[grid, tpb](x[0], x[1], x[2], *dx, loc, a, b,
                                            d, sqrt(b), precomp, pms, out)
                    else:
                        lloc = cuda.to_device(loc, stream=cuda.stream())
                        n, buf = len(x[2]), ascontiguousarray(out[..., 0])
                        tic = toc()
                        for i in range(n):
                            __GPU_FT2D[grid[:2], tpb[:2]](x[0], x[1], x[2][i],
                                                        *dx, lloc, a, b,
                                                        d, sqrt(b), precomp, pms, buf)
                            out[..., i] = buf
                            progressBar(i + 1, n, tic)
                    notComputed = False
                except Exception:
                    USE_CPU(True)

            if notComputed:
                tic = toc()
                for i in range(x[0].size):
                    __CPU_FT(x[0][i], x[1], x[2], *dx, loc, a, b, d, sqrt(b),
                         precomp, pms, out[i])
                    progressBar(i + 1, x[0].size, tic)

            if isLoud(): print(' finished.')

            return out[..., 0] if STRETCH else out

    return density, FTdensity


def getSiBlock(width=1, rotate=False, justwidth=False):
    width = (width,) * 3 if type(width) is int else width
    width = array(width)
    from pymatgen import Element, Lattice, Structure
    Si = Element('Si')
    lattice = Lattice.cubic(1)
    struct = Structure.from_spacegroup(
        "Fd-3m", lattice, [Si, Si], [[0, 0, 0], [.25, .25, .25, ]])

    scale_width = {
        False: array((1, 1, 1)), True: array((1, 2 ** .5, 2 ** .5)),
        111: array((1 / 2 ** .5, (3 / 2) ** .5, 3 ** .5))
    }
    if justwidth:
        return scale_width[rotate] * width

    if not rotate:
        struct = struct * width
        return unique(struct.cart_coords, axis=0)

    def vecInList(v, L): return any(abs(v - u).max() < 1e-6 for u in L)

    struct = struct * 5
    struct = unique(struct.cart_coords, axis=0) - 2
    block = scale_width[rotate]

    if rotate == 111:
        R = array([[-.5, .5, 0], [-.5, -.5, 1], [1, 1, 1]])

    else:
        R = array([[1, 0, 0], [0, 1, -1], [0, 1, 1]])

    R = R / block.reshape(-1, 1)  # Normalise the matrix
    struct = struct.dot(R.T)

    x = [c for c in struct if (
        (c.min() > -1e-6) and all(c < block + 1e-6))]
    keepList = [i for i in range(len(x))
                if not vecInList(x[i] - array((block[0], 0, 0)), x)]
    x = [x[i] for i in keepList]
    keepList = [i for i in range(len(x))
                if not vecInList(x[i] - array((0, block[1], 0)), x)]
    x = [x[i] for i in keepList]
    keepList = [i for i in range(len(x))
                if not vecInList(x[i] - array((0, 0, block[2])), x)]
    x = [x[i] for i in keepList]

    x = concatenate([x + i * array([[block[0], 0, 0]])
                     for i in range(width[0])], axis=0)
    x = concatenate([x + i * array([[0, block[1], 0]])
                     for i in range(width[1])], axis=0)
    x = concatenate([x + i * array([[0, 0, block[2]]])
                     for i in range(width[2])], axis=0)

    return x


class baseCrystal:

    def __init__(self, box): self.box = box

    def __call__(self, x, dx=None): raise NotImplementedError

    def FT(self, x, dx=None, C=None): raise NotImplementedError

    def __add__(self, other):
        if isinstance(other, baseCrystal):
            return self.sum(other)
        other = array(other)
        if other.size in [1, 3]:
            return self.translate(other)
        raise ValueError(
            '+ is only defined for crystal+crystal or crystal+translation where translation is a length 3 vector.')

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            raise TypeError('crystals cannot be added to by objects of type %s' % repr(type(other)))

    def __sub__(self, other): return self.translate(-array(other))

    def __mul__(self, other):
        other = array(other)
        if other.size in (1, 3):
            return self.repeat(other)
        other.shape = 3, -1
        if other.shape[1] == 3:
            return self.scale(other)

        raise ValueError(
            'Second argument of * must be replication length (size 1 or 3) or a 3x3 array.')

    __rmul__ = __mul__

    def sum(self, other): return compositeCrystal(self, other)

    def tofile(self, filename=None): raise NotImplementedError


class simpleCrystal(baseCrystal):

    def __init__(self, unit, Z, block, reps, A, b, GPU):
        '''
        Assume <unit> is a list of atom locations with atomic
        number <Z>. The unit is periodic in the 3 axes directions
        with length <block> in each direction. The unit is then
        repeated <reps> number of times in each direction.
        Finally, atom locations are skewed and translated by <A>
        and <b>.

        Compressing the repetitions into 1D notation:
        c(x) = sum_{i<reps} sum_{x_j\in unit}
                            atom_Z(x - A(x_j+b+i*block))
        where atom_Z is the scattering potential of an atom with
        atomic number Z.

        Correspondingly, the Fourier Transform:
        c.FT(y) = \int c(x)\exp(-iy\cdot x)dx
                = \sum_{i,j}\int atom_Z(x-...)exp(-iy\cdot x)dx
                = \sum_{i,j}\exp(-i...\cdot y)FT_Z(y)
        where ... = A(x_j+b+i*block)
        so unit -> A(unit+b) then repeat in direction A(block)
        '''

        # self.loc = \{A(x_j+b+i*block) for all (i,j)\}
        # self.loc = \{A(x_j+i*block)+b for all (i,j)\}
        e = [array([[block[0], 0, 0]]), array([[0, block[1], 0]]), array([[0, 0, block[2]]])]
        loc = unit
        for i in range(3):
            loc = concatenate([loc + e[i] * j for j in range(int(reps[i]))], axis=0)
        loc = loc.dot(A.T) + b

        box = []
        for i0 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    box.append(array([i0 * reps[0] * block[0],
                                      i1 * reps[1] * block[1],
                                      i2 * reps[2] * block[2]]))
        offset = unit.min(axis=0)
        box = [A.dot(v + offset) + b[0] for v in box]
        box = concatenate([v.reshape(1, -1) for v in box])

        baseCrystal.__init__(self, concatenate((box.min(axis=0).reshape(-1, 1), box.max(axis=0).reshape(-1, 1)), axis=1))
        self.unit, self.Z, self.block = unit, Z, block
        self.reps, self.A, self.b = reps, A, b
        self.GPU, self.loc, self._box = GPU, loc, box
        self._cutoff = None

    def __call__(self, x, dx=None):
        f = getDiscretisation(self.loc, self.Z, self.GPU)[0]
        arr = f(x, dx)
        if self._cutoff is not None:
            if dx is None:
                arr[x[0] < self._cutoff[0][0], :, :] = 0
                arr[x[0] > self._cutoff[0][1], :, :] = 0
                arr[:, x[1] < self._cutoff[1][0], :] = 0
                arr[:, x[1] > self._cutoff[1][1], :] = 0
                arr[:, :, x[2] < self._cutoff[2][0]] = 0
                arr[:, :, x[2] > self._cutoff[2][1]] = 0
            else:
                xx = toMesh(x, dx)
                for i in range(3):
                    arr[xx[..., i] < self._cutoff[i][0]] = 0
                    arr[xx[..., i] > self._cutoff[i][1]] = 0
        return arr

    def FT(self, x, dx=None, C=None):
        loc = self.unit.dot(self.A.T) + self.b
        f = getDiscretisation(loc, self.Z, self.GPU)[1]
        arr = f(x, dx, C)

        if max(self.reps) > 1:
            arr = periodicFT(arr, x, self.reps, -self.block,
                             self.A if dx is None else array(dx).dot(self.A), pointwise=False)
        if self._cutoff is not None:
            if C is None:
                arr = cutoffFT(arr, x, self._cutoff, dx)
            else:
                # TODO: decide what to do with sphere evaluation and cut-offs
                # Evaluating on the sphere is only a sketch method anyway so
                # it doesn't matter?
                raise NotImplementedError
        return arr

    def scale(self, other):
        '''
        c(x) = sum_{i,j} atom_Z(x - A(x_j+b+i*block))
        ->
        c(x) = sum_{i,j} atom_Z(x - [A*other](x_j+b+i*block))


        Loc = other*loc
            = other*(A*grid + b)
        '''
        other = array(other)
        if other.size == 1:
            newA = self.A * other
            newb = self.b * other
        else:
            other = other.reshape(3, 3)
            newA = other.dot(self.A)
            newb = other.dot(self.b.reshape(-1)).reshape(1, -1)

        return simpleCrystal(self.unit, self.Z, self.block, self.reps, newA, newb, self.GPU)

    def translate(self, other):
        '''
        c(x) = sum_{i,j} atom_Z(x - A(x_j+b+i*block))
        ->
        c(x) = sum_{i,j} atom_Z(x - A(x_j+[b+other]+i*block))
        '''
        other = array(other).reshape(1, -1)
        return simpleCrystal(self.unit, self.Z, self.block, self.reps, self.A, self.b + other, self.GPU)

    def repeat(self, *other):
        '''
        reps -> other*reps
        '''
        other = array(other if len(other) > 1 else other[0]).reshape(-1)
        return simpleCrystal(self.unit, self.Z, self.block, other * self.reps, self.A, self.b, self.GPU)

    def cutoff(self, box):
        raise NotImplementedError
        if len(box) == 2:
            box = [(box[0], box[1])] * 3
        elif isscalar(box[0]):
            box = [(0, box[i]) for i in range(3)]

        return compositeCrystal(self, 'cut-off', box)

    def tofile(self, filename=None, offset=None):
        i = [0, 1, 2]
        if offset is None:
            offset = self.box[:, 0]
        # atomic number, x, y, z, probability, thermal
        mystr = ''
        self.loc = self.loc[self.loc[:, 2].argsort(kind='mergsort')]
        for row in self.loc:
#             mystr += '%3d %11.6f %11.6f %11.6f %2.1f %.6f 0 0\n' % (self.Z,
#                                                 row[i[0]] - offset[i[0]], row[i[1]] - offset[i[1]],
#                                                 row[i[2]] - offset[i[2]], 0.0, 0)
            mystr += '%3d %11.6f %11.6f %11.6f 0 1 0 0\n' % (self.Z,
                                                row[i[0]] - offset[i[0]], row[i[1]] - offset[i[1]],
                                                row[i[2]] - offset[i[2]])
        if filename is None:
            return mystr[:-1]

        with open(filename, 'w') as f:
#             print('Default comment', file=f)
            print('    %11.6f %11.6f %11.6f' % tuple((self.box[:, 1] - self.box[:, 0])[i]), file=f)
            print(mystr, file=f)
#             print('-1', file=f)


class compositeCrystal(baseCrystal):

    def __init__(self, *args):
        '''
        orig and other are always baseCrystal instances
        All attributes inherited from each individual object.
        '''

        # Catch generators:
        if hasattr(args[0], '__iter__'):
            args = [a for a in args[0]]

        box = args[0].box.copy()
        for j in range(1, len(args)):
            bbox = args[j].box
            for i in range(3):
                box[i] = (min(box[i][0], bbox[i][0]), max(box[i][1], bbox[i][1]))

        baseCrystal.__init__(self, box)
        s = []
        for a in args:
            if hasattr(a, '_sum'):
                s.extend(a._sum)
            else:
                s.append(a)
        self._sum = s

    def __call__(self, x, dx=None):
        if len(self._sum) == 1:
            return self._sum[0](x, dx)

        with SILENT() as c:
            tic = toc()
            out = self._sum[0](x, dx)
            progressBar(1, len(self._sum), tic, c)
            for i in range(1, len(self._sum)):
                out += self._sum[i](x, dx)
                progressBar(i + 1, len(self._sum), tic, c)
        return out

    def FT(self, x, dx=None, C=None):
        if len(self._sum) == 1:
            return self._sum[0].FT(x, dx, C)

        with SILENT() as c:
            tic = toc()
            out = self._sum[0].FT(x, dx, C)
            progressBar(1, len(self._sum), tic, c)
            for i in range(1, len(self._sum)):
                out += self._sum[i].FT(x, dx, C)
                progressBar(i + 1, len(self._sum), tic, c)
        return out

    def __getattr__(self, name):
        if name in ['scale', 'translate', 'repeat', 'cutoff']:
            return lambda *other: compositeCrystal(getattr(c, name)(*other) for c in self._sum)
        if name == 'loc':
            return concatenate([c.loc for c in self._sum], axis=0)
        if name == 'Z':
            return concatenate([array(c.Z).reshape(-1) for c in self._sum], axis=0)
        else:
            raise AttributeError("'%s' attribute does not exist.\nCrystal manipulations are: 'scale', 'translate', 'repeat', 'cutoff'" % name)

    def tofile(self, filename=None, offset=None):
        i = [0, 1, 2]
        if offset is None:
            offset = self.box[:, 0]
        mystr = ''
        for c in self._sum:
            mystr += c.tofile(offset=offset)
        if filename is None:
            return mystr

        with open(filename, 'w') as f:
#             print('Default comment', file=f)
            print('    %11.6f %11.6f %11.6f' % tuple((self.box[:, 1] - self.box[:, 0])[i]), file=f)
            print(mystr, file=f)
#             print('-1', file=f)


class crystal(compositeCrystal):

    def __init__(self, loc, Z, block=None, GPU=True):
        reps = array([1, 1, 1], dtype=int)
        A = eye(3, 3, dtype=float)
        b = zeros((1, 3), dtype=float)
        if block is None:
            block = loc.ptp(axis=0)
        elif isscalar(block):
            block = array([block] * 3)
        else:
            block = array(block)
        params = block, reps, A, b, GPU

        if isscalar(Z):
            _sum = (simpleCrystal(loc, Z, *params),)
        else:
            ZZ = unique(Z)
            _sum = tuple(
                simpleCrystal(ascontiguousarray(loc[Z == z, :]), z, *params)
                for z in ZZ
            )
        compositeCrystal.__init__(self, *_sum)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from FourierTransform import getFTpoints
    with SILENT():
#         C1 = crystal(5.4 * getSiBlock(1, False), 14,
#                      5.4 * getSiBlock(1, False, True))
#         C1 = C1.repeat(30)
#         C1 = C1 - array(C1.box)[:, 1] / 2

#     __makeVid()
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# plt.figure(figsize=(20, 10))
# if smallVid is not None:
#     writer = __makeVid(plt.gcf(), smallVid, stage=1, fps=30)
# ax = (plt.gcf().add_subplot(121, projection='3d'),
#       plt.gcf().add_subplot(122, projection='3d'))
# _get3DvolPlot(ax[0], x[::n, ::n, ::n], (-168, -158), .07)
# _get3DvolPlot(ax[1], y[::n, ::n, ::n], (-168, -158), .07)
# ax[0].set_title('TV reconstruction')
# ax[1].set_title('Gaussian reconstruction')
# i, j = 0, 0
# while j < 2:
#     ax[0].view_init(-168, i)
#     ax[1].view_init(-168, i)
#     i += 2.5
#     print(i)
#     if i >= 360:
#         break
#
#     plt.draw()
#     if smallVid is None:
#         plt.show(block=False)
#         plt.pause(.2)
#     else:
#         __makeVid(writer, plt, stage=2)
# if smallVid is not None:
#     __makeVid(writer, plt, stage=3)

#         import matplotlib
#         matplotlib.use('Agg')
#         from matplotlib import pyplot as plt
#         plt.figure(figsize=(10, 10))
#         from matplotlib import animation as mv
#         writer = mv.writers['ffmpeg'](fps=18, metadata={'title': 'crystal_vid'})
#         writer.setup(plt.gcf(), 'big_crystal_vid' + '.mp4', dpi=100)
#         from mpl_toolkits.mplot3d import Axes3D
#         ax = plt.gcf().add_subplot(111, projection='3d')
#         tmp = C1.loc[abs(C1.loc).max(1) < 30, :];
#         ax.scatter(*tmp.T, c='b')
#         plt.tight_layout()
#
#         for i in range(360):
#             ax.view_init(0, i)
#             plt.pause(.01)
# #             plt.draw()
#             writer.grab_frame()
#             print(i)
#
#         writer.finish()
#         print('finished')
#         exit()

#         x, y = getFTpoints(3, n=256, rX=5, rY=3)
#         arr = C1(x)
#
#         plt.imshow(arr.max(-1).real, aspect='auto', cmap='gray',
#                    extent=[x[0][0], x[0][-1], x[1][-1], x[1][0]])
#         plt.show()
#         exit()

        x, y = getFTpoints(3, n=128, rX=6)
        dft, idft = getDFT(x, y)
        C1 = crystal(1 * getSiBlock(1, False), 0,
                     1 * getSiBlock(1, False, True))
        t, s, r = -array(C1.box)[:, 1] / 2, 1 / 3, 2
        s = array([[5 / 3, 1 / 3, 0], [-1 / 3, 5 / 3, 0], [0, 0, 1]])
    #     s = array([[0, 1, 0], [-2, 0, 0], [0, 0, 1]])

        print('Original')
        arr = C1(x), abs(idft(C1.FT(y)))
        plt.subplot(251)
        plt.imshow(arr[0].sum(-1).real, aspect='auto',
                   extent=[x[0][0], x[0][-1], x[1][-1], x[1][0]])
        plt.title('Original')
        plt.subplot(256)
        plt.imshow(arr[1].sum(-1).real, aspect='auto',
                   extent=[x[0][0], x[0][-1], x[1][-1], x[1][0]])

        print('Translated')
        C1 = C1.translate(t)
        arr = C1(x), idft(C1.FT(y))
        plt.subplot(252)
        plt.imshow(arr[0].sum(-1).real, aspect='auto',
                   extent=[x[0][0], x[0][-1], x[1][-1], x[1][0]])
        plt.title('Translated')
        plt.subplot(257)
        plt.imshow(arr[1].sum(-1).real, aspect='auto',
                   extent=[x[0][0], x[0][-1], x[1][-1], x[1][0]])

        print('Repeated')
        C1 = C1.repeat(r)
        arr = C1(x), idft(C1.FT(y))
        plt.subplot(253)
        plt.imshow(arr[0].sum(-1).real, aspect='auto',
                   extent=[x[0][0], x[0][-1], x[1][-1], x[1][0]])
        plt.title('Repeated')
        plt.subplot(258)
        plt.imshow(arr[1].sum(-1).real, aspect='auto',
                   extent=[x[0][0], x[0][-1], x[1][-1], x[1][0]])

        print('Scaled')
        C1 = C1.scale(s)
        arr = C1(x), idft(C1.FT(y))
        plt.subplot(254)
        plt.imshow(arr[0].sum(-1).real, aspect='auto',
                   extent=[x[0][0], x[0][-1], x[1][-1], x[1][0]])
        plt.title('Scaled')
        plt.subplot(259)
        plt.imshow(arr[1].sum(-1).real, aspect='auto',
                   extent=[x[0][0], x[0][-1], x[1][-1], x[1][0]])

        print('Summed')
        C2 = crystal(getSiBlock(1, True), 0, 1, GPU=True)
        C2 = C2 - array(C2.box)[:, 1] / 2
        plt.subplot(255)
        plt.title('Summed')
        plt.imshow((C1 + C2)(x).sum(-1), aspect='auto',
                   extent=[x[0][0], x[0][-1], x[1][-1], x[1][0]])
        plt.subplot(2, 5, 10)
        plt.imshow(C1(x).sum(-1) + C2(x).sum(-1), aspect='auto',
                   extent=[x[0][0], x[0][-1], x[1][-1], x[1][0]])

#         plt.figure()
#         C1 = C1.repeat(100)
#         plt.subplot(121)
#         plt.title('Fourier Transform')
#         plt.imshow(abs(C1.FT(y[:2])))
#         plt.subplot(122)
#         plt.title('Fourier Transform on a Sphere')
#         plt.imshow(abs(C1.FT(y[:2], C=10)))

        plt.show()
