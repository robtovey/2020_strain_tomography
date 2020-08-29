'''
Created on 19 Oct 2018

@author: Rob Tovey
'''
from time import time
from numpy import (zeros, array, sin, cos, pi, empty, pad, random, prod, isscalar, \
    ascontiguousarray)
import h5py
import numba
from math import floor, ceil
from os.path import isfile
from numba import cuda
from shutil import get_terminal_size
DEBUG = False

try:
    import pyfftw
    next_fast_len = pyfftw.next_fast_len
    from pyfftw.interfaces.numpy_fft import fftn, ifftn, ifftshift, fftshift
    pyfftw.config.NUM_THREADS = getattr(numba.config, 'NUMBA_DEFAULT_NUM_THREADS')
    pyfftw.interfaces.cache.enable()  # This can increase memory hugely sometimes
#     pyfftw.interfaces.cache.set_keepalive_time(10)

    def plan_fft(A, n=None, axis=None, overwrite=False, planner=1, threads=None,
            auto_align_input=True, auto_contiguous=True,
            avoid_copy=False, norm=None):

        if threads is None:
            threads = getattr(numba.config, 'NUMBA_DEFAULT_NUM_THREADS')
        planner_effort = 'FFTW_' + ['ESTIMATE', 'MEASURE', 'PATIENT', 'EXHAUSTIVE'][planner]

        plan = pyfftw.builders.fftn(A, n, axis, overwrite,
                                   planner_effort, threads,
                                   auto_align_input, auto_contiguous,
                                   avoid_copy, norm)

        return plan, plan.input_array

#     def plan_fft(A, n=None, axis=None, norm=None, threads=None, **_):
#         if threads is None:
#             threads = getattr(numba.config, 'NUMBA_DEFAULT_NUM_THREADS')
#         return lambda : fftn(A, n, axis, norm=norm, threads=threads), A

    def plan_ifft(A, n=None, axis=None, overwrite=False, planner=1, threads=None,
            auto_align_input=True, auto_contiguous=True,
            avoid_copy=False, norm=None):
        if threads is None:
            threads = getattr(numba.config, 'NUMBA_DEFAULT_NUM_THREADS')
        planner_effort = 'FFTW_' + ['ESTIMATE', 'MEASURE', 'PATIENT', 'EXHAUSTIVE'][planner]

        plan = pyfftw.builders.ifftn(A, n, axis, overwrite,
                                   planner_effort, threads,
                                   auto_align_input, auto_contiguous,
                                   avoid_copy, norm)
        return plan, plan.input_array

except ImportError:
    from scipy.fftpack import fftn, ifftn, ifftshift, fftshift, next_fast_len
    from numpy.fft import fftn, ifftn, ifftshift, fftshift

    old_fftn, old_ifftn = fftn, ifftn

    # Numpy versions:
    def fftn(A, n=None, axis=None, norm=None, **_): return old_fftn(A, n, axis, norm)

    def ifftn(A, n=None, axis=None, norm=None, **_): return old_ifftn(A, n, axis, norm)

    class __plan:

        def __init__(self, fwrd, A, n=None, axis=None, norm=None, **_):
            self.fwrd = fwrd
            self.input_array = A
            self.output_array = A + 1j
            self.params = n, axis, norm

        def __call__(self):
            if self.fwrd:
                self.output_array[...] = old_fftn(self.input_array, *self.params)
            else:
                self.output_array[...] = old_ifftn(self.input_array, *self.params)
            return self.output_array

    def plan_fft(A, n=None, axis=None, norm=None, **_):
        return __plan(True, A, n, axis, norm), A

    def plan_ifft(A, n=None, axis=None, norm=None, **_):
        return __plan(False, A, n, axis, norm), A

try:
    from cupy.scipy.fftpack import fftn as gpu_fftn, ifftn as gpu_ifftn
#     plan = cupyx.scipy.fftpack.get_fft_plan(x)
#     out = plan.get_output_array(a)

    def plan_gpu_fft(A, n=None, axis=None, overwrite=False, planner=1, threads=None,
            auto_align_input=True, auto_contiguous=True,
            avoid_copy=False, norm=None):

        if threads is None:
            threads = getattr(numba.config, 'NUMBA_DEFAULT_NUM_THREADS')
        planner_effort = 'FFTW_' + ['ESTIMATE', 'MEASURE', 'PATIENT', 'EXHAUSTIVE'][planner]

        plan = pyfftw.builders.fftn(A, n, axis, overwrite,
                                   planner_effort, threads,
                                   auto_align_input, auto_contiguous,
                                   avoid_copy, norm)

        return plan, plan.input_array

except ImportError:
    pass


def fast_fft_len(n):
    N = next_fast_len(n)
    return N if N % 2 == 0 else fast_fft_len(N + 1)


@numba.jit(nopython=True, parallel=True, fastmath=True)
def fftshift_phase(x):
    '''
    fft_shift(fft(x)) = fft(P*x)
    where P_i = (-1)^i
    '''
    for i in numba.prange(x.shape[0]):
        for j in range(x.shape[1]):
            start = int((i + j) % 2)
            for k in range(start, x.shape[2], 2):
                x[i, j, k] = -x[i, j, k]


def fast_abs(x, y=None):
    if y is None:
        y = empty(x.shape, dtype=abs(x[(slice(1),) * x.ndim]).dtype)
    __fast_abs(x.reshape(-1), y.reshape(-1))
    return y


@numba.jit(nopython=True, parallel=True, fastmath=True)
def __fast_abs(x, y):
    for i in numba.prange(x.size):
        y[i] = abs(x[i])


def savemat(fname, **data):
    with h5py.File(fname + ".hdf5", "w") as f:
        for thing in data:
            f.create_dataset(thing, data=data[thing],
                             chunks=True, compression="gzip")


def loadmat(fname, *params, dtype=None):
    if isfile(fname + ".hdf5"):
        with h5py.File(fname + ".hdf5", "r") as f:
            if dtype is None:
                return [f[thing][()] for thing in params]
            else:
                return [f[thing][()].astype(dtype) for thing in params]
    else:
        raise IOError('File does not exist or cannot be opened.')


class GLOBAL_BOOL:

    def __init__(self, val): self.val = val

    def __call__(self, val):
        if val != self.val:
#             raise
            print('\nGLOBAL_BOOL changed to %s\n' % str(val))
        self.val = val

    def set(self, val): self.val = val

    def __bool__(self): return self.val

    def __str__(self): return str(self.val)


USE_CPU = GLOBAL_BOOL(not cuda.is_available())
# USE_CPU(True)

_managers = []

if DEBUG:

    def isLoud(c=None): return False

else:

    def isLoud(c=None): return ((len(_managers) == 0) or (c == _managers[0]))


class SILENT:

    def __init__(self): self.__randomid = random.rand()

    def __enter__(self): _managers.append(self); return self

    def __exit__(self, *_): _managers.remove(self)

    def __eq__(self, other):
        try:
            return self.__randomid == other.__randomid
        except Exception:
            return False


def toc(s='', tic=None, context=None):
    if tic is None:
        return time()
    if isLoud(context):
        print(s + ' ' * (max(1, 30 - len(s))) + __timelen(time() - tic))
    return time()


def progressBar(count, total, tic=None, context=None):
    if not isLoud(context):
        return
    line_len = get_terminal_size().columns - 1
    bar_len, frac = 50, max(count / total, 1e-5)
    percent = '%2d' % int(round(100 * frac)) + '%'
    if tic is None:
        tim = ''
    else:
        tim = '... %s left, %s total   ' % \
            (__timelen((time() - tic) * (1 / frac - 1)), __timelen((time() - tic) / frac))

    bar_len = min(bar_len, line_len - 3 - len(percent) - len(tim))
    length = int(round(bar_len * frac))
    if bar_len < 3:
        bar = '[' + '-' * length + ' ' * (bar_len - length) + ']'
    else:
        bar = ''  # ignore the progress bar if too long

    tmp = '\r' + bar + ' ' + percent + tim
    print(tmp + ' ' * max(0, line_len - len(tmp)), flush=True,
          end=('' if count < total else '\n'))


@numba.jit(parallel=True, fastmath=True, cache=False, nopython=True)
def __toMesh2d(x0, x1, dx0, dx1, out):
    for i0 in numba.prange(x0.size):
        X00 = x0[i0] * dx0[0]
        X01 = x0[i0] * dx0[1]
        for i1 in range(x1.size):
            out[i0, i1, 0] = X00 + x1[i1] * dx1[0]
            out[i0, i1, 1] = X01 + x1[i1] * dx1[1]


@numba.jit(parallel=True, fastmath=True, cache=False, nopython=True)
def __toMesh3d(x0, x1, x2, dx0, dx1, dx2, out):
    for i0 in numba.prange(x0.size):
        X00 = x0[i0] * dx0[0]
        X01 = x0[i0] * dx0[1]
        X02 = x0[i0] * dx0[2]
        for i1 in range(x1.size):
            X10 = x1[i1] * dx1[0]
            X11 = x1[i1] * dx1[1]
            X12 = x1[i1] * dx1[2]
            for i2 in range(x2.size):
                out[i0, i1, i2, 0] = X00 + X10 + x2[i2] * dx2[0]
                out[i0, i1, i2, 1] = X01 + X11 + x2[i2] * dx2[1]
                out[i0, i1, i2, 2] = X02 + X12 + x2[i2] * dx2[2]


def toMesh(x, dx=None, shape=None, dtype=None):
    if shape is None:
        shape = [xi.size for xi in x]
    else:
        shape = list(shape)
    if dtype is None:
        dtype = x[0].dtype
    else:
        x = [xi.astype(dtype, copy=False) for xi in x]
    dim = len(shape)
    X = zeros(shape + [dim], dtype=dtype)

    if dim == 2:
        dx = [array([1, 0]), array([0, 1])] if dx is None else list(dx)
        __toMesh2d(*x, *dx, X)
        return X
    elif dim == 3:
        dx = [array([1, 0, 0]), array([0, 1, 0]), array([0, 0, 1])] if dx is None else list(dx)
        __toMesh3d(*x, *dx, X)
        return X

    dim = X.shape[-1]
    for i in range(len(x)):
        sz = [1] * dim
        sz[i] = -1
        X[..., i] += x[i].reshape(sz)
    return X


def rot3d(vec, axis, theta):
    axis = array(axis)
    axis = axis / ((axis * axis).sum()) ** .5

    a = cos(theta / 2.0 * pi / 180)
    b, c, d = -axis * sin(theta / 2.0 * pi / 180)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rotmat = array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                    [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                    [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    if vec is None:
        return rotmat
    else:
        return vec.dot(rotmat.T)


def __timelen(t):
    if t > 3600:
        H = t // 3600
        M = (t - H * 3600) // 60
        return '%d:%02d hours' % (H, M)
    elif t > 60:
        M = t // 60
        T = int(t - M * 60)
        return '%2d:%02d mins' % (M, T)
    else:
        return '%2d secs' % (int(t))


@numba.jit(cache=True)
def __countbins(x0, x1, x2, loc, r, s, Len, MAX):
    for j0 in range(loc.shape[0]):
        bin0 = int((loc[j0, 0] - x0) / r[0])
        bin1 = int((loc[j0, 1] - x1) / r[1])
        bin2 = int((loc[j0, 2] - x2) / r[2])
        for i in range(max(0, bin0 - s), min(Len.shape[0], bin0 + s + 1)):
            for j in range(max(0, bin1 - s), min(Len.shape[1], bin1 + s + 1)):
                for k in range(max(0, bin2 - s), min(Len.shape[2], bin2 + s + 1)):
                    Len[i, j, k] += 1
                    if Len[i, j, k] == MAX:
                        return


@numba.jit(cache=True)
def __rebin(x0, x1, x2, loc, sublist, r, s, Len):
    for j0 in range(loc.shape[0]):
        bin0 = int((loc[j0, 0] - x0) / r[0])
        bin1 = int((loc[j0, 1] - x1) / r[1])
        bin2 = int((loc[j0, 2] - x2) / r[2])
        for i in range(max(0, bin0 - s), min(Len.shape[0], bin0 + s + 1)):
            for j in range(max(0, bin1 - s), min(Len.shape[1], bin1 + s + 1)):
                for k in range(max(0, bin2 - s), min(Len.shape[2], bin2 + s + 1)):
                    sublist[i, j, k, Len[i, j, k]] = j0
                    Len[i, j, k] += 1

    for b0 in range(sublist.shape[0]):
        for b1 in range(sublist.shape[1]):
            for b2 in range(sublist.shape[2]):
                j0 = Len[b0, b1, b2]
                if j0 < sublist.shape[3]:
                    sublist[b0, b1, b2, j0] = -1


@numba.jit(cache=True)
def __ind2loc(loc, subList, Len, out):
    for j0 in range(subList.shape[0]):
        for j1 in range(subList.shape[1]):
            for j2 in range(subList.shape[2]):
                for i in range(Len[j0, j1, j2]):
                    k = subList[j0, j1, j2, i]
                    out[j0, j1, j2, i, :] = loc[k]


def rebin(x, loc, r, k, toLoc=False, mem=None):
    '''
    x is the grid for the discretisation, used for bounding box
    loc is the locations of each Atom
    '''
    assert len(x) == 3, 'x must represent a 3 dimensional grid'
    mem = 1e10 if mem is None else mem

    if isscalar(r):
        r = array([r, r, r], dtype='f4')
    else:
        r = array(r).copy()
    xmin = array([X.item(0) if X.size > 1 else -1e5 for X in x], dtype=x[0].dtype)
    nbins = [int(ceil((x[i].item(-1) - x[i].item(0)) / r[i])) + 1
             for i in range(3)]
    if prod(nbins) * 32 * 10 > mem:
        raise MemoryError
    Len = zeros(nbins, dtype='i4')
    L = int(mem / (Len.size * Len.itemsize)) + 2
    __countbins(xmin[0], xmin[1], xmin[2], loc, r, k, Len, L)

    L = Len.max()
    if Len.size * Len.itemsize * L > mem:
        raise MemoryError
    subList = zeros(nbins + [L], dtype='i4')
    Len.fill(0)
    __rebin(xmin[0], xmin[1], xmin[2], loc, subList, r, k, Len)

    if toLoc:
        out = zeros(nbins + [L, 3], 'f4')
        __ind2loc(loc, subList, Len, out)
        return out, Len
    else:
        return subList


@numba.jit(cache=True)  # 'void(F,F,F[:,:],F,i4[:,:])'
def __countbins_col(x0, x1, loc, r, Len):
    for j0 in range(loc.shape[0]):
        bin0 = 1 + int(floor((loc[j0, 0] - x0) / r))
        bin1 = 1 + int(floor((loc[j0, 1] - x1) / r))
        Len[bin0, bin1] += 1


@numba.jit(cache=True)  # 'void(F,F,F[:,:],i4[:,:,:],F,i4[:,:])'
def __rebin_col(x0, x1, loc, sublist, r, Len):
    for j0 in range(loc.shape[0]):
        bin0 = 1 + int(floor((loc[j0, 0] - x0) / r))
        bin1 = 1 + int(floor((loc[j0, 1] - x1) / r))
        if (bin0 >= 0) and (bin0 < sublist.shape[0]) \
                and (bin1 >= 0) and (bin1 < sublist.shape[1]):
            sublist[bin0, bin1, Len[bin0, bin1]] = j0
            Len[bin0, bin1] += 1

    for b0 in range(sublist.shape[0]):
        for b1 in range(sublist.shape[1]):
            j0 = Len[b0, b1]
            if j0 < sublist.shape[2]:
                sublist[b0, b1, j0] = -1


def rebin_col(x, loc, r):
    assert len(x) == 2, 'x must represent a 2 dimensional grid'

    xmin = array([X.item(0) for X in x], dtype=x[0].dtype)
    nbins = [int(ceil((x[i].item(-1) - x[i].item(0)) / r) + 2)
             for i in range(2)]
    Len = zeros(nbins, dtype='i4')
    __countbins_col(xmin[0], xmin[1], loc, r, Len)

    subList = zeros(nbins + [Len.max()], dtype='i4')
    Len[:] = 0
    __rebin(xmin[0], xmin[1], xmin[2], loc, subList, r, Len)

    return subList


@numba.jit(cache=True)
def __countbins_block(x0, x1, loc, block, r, sgn, Len):
    for j0 in range(loc.shape[0]):
        x0min = int(floor((loc[j0, 0] - x0 - sgn[0] * r[0]) / block[0]))
        x0max = int(ceil((loc[j0, 0] - x0 + sgn[0] * r[0]) / block[0]))
        x1min = int(floor((loc[j0, 1] - x1 - sgn[1] * r[1]) / block[1]))
        x1max = int(ceil((loc[j0, 1] - x1 + sgn[1] * r[1]) / block[1]))

        for i in range(max(0, x0min), min(Len.shape[0], x0max + 1)):
            for j in range(max(0, x1min), min(Len.shape[1], x1max + 1)):
                Len[i, j] += 1


@numba.jit(cache=True)
def __rebin_block(x0, x1, loc, sublist, block, r, sgn, Len):
    for j0 in range(loc.shape[0]):
        x0min = int(floor((loc[j0, 0] - x0 - sgn[0] * r[0]) / block[0]))
        x0max = int(ceil((loc[j0, 0] - x0 + sgn[0] * r[0]) / block[0]))
        x1min = int(floor((loc[j0, 1] - x1 - sgn[1] * r[1]) / block[1]))
        x1max = int(ceil((loc[j0, 1] - x1 + sgn[1] * r[1]) / block[1]))
        for i in range(max(0, x0min), min(Len.shape[0], x0max + 1)):
            for j in range(max(0, x1min), min(Len.shape[1], x1max + 1)):
                    sublist[i, j, Len[i, j]] = j0
                    Len[i, j] += 1

    for b0 in range(sublist.shape[0]):
        for b1 in range(sublist.shape[1]):
            j0 = Len[b0, b1]
            if j0 < sublist.shape[2]:
                sublist[b0, b1, j0] = -1


def rebin_block(x, loc, block, r, toLoc=False, mem=None):
    assert len(x) == 3, 'x must represent a 3 dimensional grid'
    mem = 1e10 if mem is None else mem

    # convert to real length blocks
    block = [x[0][block[0]] - x[0][0], x[1][block[1]] - x[1][0]]
    if isscalar(r):
        r = array([r, r, r], dtype='f4')
    xmin = array([X.item(0) for X in x], dtype=x[0].dtype)
    sgn = [1 if X.size == 1 or X.item(1) > X.item(0) else -1 for X in x]
    nbins = [int(ceil(x[i].ptp() / block[i])) for i in range(2)]
    if prod(nbins) > mem / 10000:
        raise MemoryError
    Len = zeros(nbins, dtype='i4')
    __countbins_block(xmin[0], xmin[1], loc, block, r, sgn, Len)

    L = Len.max()
    if Len.size * Len.itemsize * L > mem:
        raise MemoryError
    subList = zeros(nbins + [L], dtype='i4')
    Len.fill(0)
    __rebin_block(xmin[0], xmin[1], loc, subList, block, r, sgn, Len)

    if toLoc:
        out = zeros(nbins + [1, L, 3], 'f4')
        __ind2loc(loc, subList.reshape(nbins + [1, -1]), Len.reshape(nbins + [1]), out)
        return out.reshape(nbins + [-1, 3]), Len.reshape(nbins)
    else:
        return subList


@numba.jit(parallel=True, fastmath=True, cache=False)
def __upscale1d(arr, out, factor):
    scale = (1 / (factor)) ** .5
    for i in numba.prange(arr.shape[0]):
        I = factor * i
        val = arr[i] * scale
        for ii in range(factor):
            out[I + ii] = val


@numba.jit(parallel=True, fastmath=True, cache=False)
def __upscale2d(arr, out, factor):
    scale = (1 / (factor * factor)) ** .5
    for i in numba.prange(arr.shape[0]):
        I = factor * i
        for j in range(arr.shape[1]):
            J = factor * j
            val = arr[i, j] * scale
            for ii in range(factor):
                for jj in range(factor):
                    out[I + ii, J + jj] = val


@numba.jit(parallel=True, fastmath=True, cache=False)
def __upscale2p5d(arr, out, factor, axes):
    scale = (1 / (factor * factor)) ** .5
    if axes[0] == 0 and axes[1] == 1:
        for i in numba.prange(arr.shape[0]):
            I = factor * i
            for j in range(arr.shape[1]):
                J = factor * j
                for k in range(arr.shape[2]):
                    val = arr[i, j, k] * scale
                    for ii in range(factor):
                        for jj in range(factor):
                            out[I + ii, J + jj, k] = val
    elif axes[0] == 0 and axes[1] == 2:
        for i in numba.prange(arr.shape[0]):
            I = factor * i
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    K = factor * k
                    val = arr[i, j, k] * scale
                    for ii in range(factor):
                        for kk in range(factor):
                            out[I + ii, j, K + kk] = val
    else:
        for i in numba.prange(arr.shape[0]):
            for j in range(arr.shape[1]):
                J = factor * j
                for k in range(arr.shape[2]):
                    K = factor * k
                    val = arr[i, j, k] * scale
                    for jj in range(factor):
                        for kk in range(factor):
                            out[i, J + jj, K + kk] = val


@numba.jit(parallel=True, fastmath=True, cache=False)
def __upscale3d(arr, out, factor):
    scale = (1 / (factor * factor * factor)) ** .5
    for i in numba.prange(arr.shape[0]):
        I = factor * i
        for j in range(arr.shape[1]):
            J = factor * j
            for k in range(arr.shape[2]):
                K = factor * k
                val = arr[i, j, k] * scale
                for ii in range(factor):
                    for jj in range(factor):
                        for kk in range(factor):
                            out[I + ii, J + jj, K + kk] = val


@numba.jit(parallel=True, fastmath=True, cache=False)
def __downscale1d(arr, out, factor):
    scale = (1 / (factor)) ** .5
    for i in numba.prange(out.shape[0]):
        I = factor * i
        val = 0
        for ii in range(factor):
            val += arr[I + ii]
        out[i] = val * scale


@numba.jit(parallel=True, fastmath=True, cache=False)
def __downscale2d(arr, out, factor):
    scale = (1 / (factor * factor)) ** .5
    for i in numba.prange(out.shape[0]):
        I = factor * i
        for j in range(out.shape[1]):
            J = factor * j
            val = 0
            for ii in range(factor):
                for jj in range(factor):
                    val += arr[I + ii, J + jj]
            out[i, j] = val * scale


@numba.jit(parallel=True, fastmath=True, cache=False)
def __downscale2p5d(arr, out, factor, axes):
    scale = (1 / (factor * factor)) ** .5
    if axes[0] == 0 and axes[1] == 1:
        for i in numba.prange(arr.shape[0]):
            I = factor * i
            for j in range(arr.shape[1]):
                J = factor * j
                for k in range(arr.shape[2]):
                    val = 0
                    for ii in range(factor):
                        for jj in range(factor):
                            val += arr[I + ii, J + jj, k]
                    out[i, j, k] = val * scale
    elif axes[0] == 0 and axes[1] == 2:
        for i in numba.prange(arr.shape[0]):
            I = factor * i
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    K = factor * k
                    val = 0
                    for ii in range(factor):
                        for kk in range(factor):
                            val += arr[I + ii, j, K + kk]
                    out[i, j, k] = val * scale
    else:
        for i in numba.prange(arr.shape[0]):
            for j in range(arr.shape[1]):
                J = factor * j
                for k in range(arr.shape[2]):
                    K = factor * k
                    val = 0
                    for jj in range(factor):
                        for kk in range(factor):
                            val += arr[i, J + jj, K + kk]
                    out[i, j, k] = val * scale


@numba.jit(parallel=True, fastmath=True, cache=False)
def __downscale3d(arr, out, factor):
    scale = (1 / (factor * factor * factor)) ** .5
    for i in numba.prange(out.shape[0]):
        I = factor * i
        for j in range(out.shape[1]):
            J = factor * j
            for k in range(out.shape[2]):
                K = factor * k
                val = 0
                for ii in range(factor):
                    for jj in range(factor):
                        for kk in range(factor):
                            val += arr[I + ii, J + jj, K + kk]
                out[i, j, k] = val * scale


def upscale(arr, factor=2, axes=None):
    factor = int(factor)
    out = empty(tuple(factor * s for s in arr.shape), dtype=arr.dtype)
    if arr.ndim == 2:
        __upscale2d(arr, out, factor)
    elif arr.ndim == 3:
        if axes is None or len(axes) == 3:
            __upscale3d(arr, out, factor)
        else:
            __upscale2p5d(arr, out, factor, array(axes, dtype='i4'))

    elif arr.ndim == 1 and (axes is None or len(axes) == 1):
        __upscale1d(arr, out, factor)
    else:
        raise ValueError('arr must be a dimension 1 or 2 array')
    return out


def downscale(arr, factor=2, axes=None):
    # We trust that the sizes actually match up
    factor = int(factor)
    out = empty(tuple(int(round(s / factor))
                      for s in arr.shape), dtype=arr.dtype)
    if arr.ndim == 2:
        __downscale2d(arr, out, factor)
    elif arr.ndim == 3:
        if axes is None or len(axes) == 3:
            __downscale3d(arr, out, factor)
        else:
            __downscale2p5d(arr, out, factor, array(axes, dtype='i4'))
    elif arr.ndim == 1 and (axes is None or len(axes) == 1):
        __downscale1d(arr, out, factor)
    else:
        raise ValueError('arr must be a dimension 1 or 2 array')
    return out


def zeropad(arr, factor=2, central=False, axes=None):
    padsize = [(factor - 1) * s for s in arr.shape]
    if central:
        padsize = [(int(s / 2), s - int(s / 2)) for s in padsize]
    else:
        padsize = [(0, s) for s in padsize]

    if axes is not None:
        axes = tuple(i if i >= 0 else arr.ndim + i for i in axes)
        for i in range(arr.ndim):
            if i not in axes:
                padsize[i] = (0, 0)

    return pad(arr, pad_width=padsize, mode='constant')


def cut(arr, factor=2, central=False, axes=None):
    padsize = [int(round((1 - 1 / factor) * s)) for s in arr.shape]
    if central:
        padsize = [(int(s / 2), s - int(s / 2)) for s in padsize]
    else:
        padsize = [(0, s) for s in padsize]

    if axes is not None:
        axes = tuple(i if i >= 0 else arr.ndim + i for i in axes)
        for i in range(arr.ndim):
            if i not in axes:
                padsize[i] = (0, 0)

    mySlice = tuple(slice(p[0], (-p[1] if p[1] > 0 else None)) for p in padsize)
    return arr[mySlice]


def convolve(arr1, arr2, dx=None, axes=None):
    if arr2.ndim > arr1.ndim:
        arr1, arr2 = arr2, arr1
        if axes is None:
            axes = range(arr2.ndim)
    arr2 = arr2.reshape(arr2.shape + (1,) * (arr1.ndim - arr2.ndim))

    if dx is None:
        dx = 1
    elif isscalar(dx):
        dx = dx ** (len(axes) if axes is not None else arr1.ndim)
    else:
        dx = prod(dx)

    arr1 = fftn(arr1, axes=axes)
    arr2 = fftn(ifftshift(arr2), axes=axes)
    out = ifftn(arr1 * arr2, axes=axes) * dx
    if not out.flags['C_CONTIGUOUS']:
        out = ascontiguousarray(out)
    return out
