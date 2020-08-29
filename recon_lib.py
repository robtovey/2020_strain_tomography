'''
Created on 27 Jun 2019

@author: Rob Tovey
'''

from numpy import array, empty, prod, zeros, inf, maximum, minimum, logical_not, logical_or, random, isscalar, \
    kron, ascontiguousarray, pad
from numpy.linalg import norm
from scipy.sparse.linalg import LinearOperator

###########################################################
# code which writes module for fast compiled functions
###########################################################
import numba, importlib


@numba.jit(nopython=True, parallel=True, fastmath=True)
def shrink(y, scale, out):
    for i in numba.prange(y.shape[0]):
        n = 0
        for j in range(y.shape[1]):
            n += y[i, j] * y[i, j]
        if n < scale * scale:
            for j in range(y.shape[1]):
                out[i, j] = y[i, j]
        else:
            n = min(scale * (n ** -.5), 1e8)
            for j in range(y.shape[1]):
                out[i, j] = n * y[i, j]


@numba.jit(nopython=True, parallel=True, fastmath=True)
def sym(A, B):
    for i in numba.prange(A.shape[0]):
        for j0 in range(A.shape[1]):
            for j1 in range(j0):
                B[i, j0, j1] = .5 * (A[i, j0, j1] + A[i, j1, j0])
                B[i, j1, j0] = B[i, j0, j1]
            B[i, j0, j0] = A[i, j0, j0]


def genbody(lines, var, end, tab, dim, options):
    from itertools import product

    Dim = len(var)
    func = ''
    old = (None,) * dim
    for block in product(*((options,) * dim)):
        stepdown, first = False, True
        for i in range(dim):
            if block[i] == old[i]:
                func += '#'
            if block[i] == 0:
                func += '\t' * tab + var[i] + ' = 0\n'
            elif block[i] == 1:
                if block[i] == old[i]:
                    func += '\t' * tab + var[i] + ' is looping\n'
                else:
                    func += ('\t' * tab + 'for ' + var[i] + ' in ' +
                             ('numba.prange(' if first else 'range(') +
                             ('1' if options[0] == 0 else '0') + ', ' + end[i] + '):\n')
                    tab += 1
                stepdown = all(j == 2 for j in block[i + 1:])
                first = False
            else:
                func += '\t' * tab + \
                    var[i] + ' = ' + end[i] + '\n'

        for i in range(dim, Dim):
            func += ('\t' * tab + 'for ' +
                     var[i] + ' in range(' + end[i] + '):\n')
            tab += 1

        # Print actual computations
        for i in range(dim):
            func += '\t' * tab + lines[block[i]][i] + '\n'

        tab -= (Dim - dim)
        if stepdown:
            tab -= 1
        old = tuple(b for b in block)
        func += '\n'
    return func


def genbody_GPU(lines, var, end, tab, dim, options):
    isPad = len(var) - dim  # either 1 or 0
    func = ''

    func += '\t' * tab + \
        ','.join(var[:dim]) + ' = cuda.grid(' + str(dim) + ')\n'

    # Don't permit out of bound indices
    for i in range(dim):
        func += ('\t' * tab + 'if ' +
                 var[i] + ' >= f.shape[' + str(i) + ']:\n')
        func += ('\t' * (tab + 1) + 'return\n')

    old_options = [i for i in options]
    options = tuple(i for i in (0, 2, 1) if i in old_options)

    # if i == first:
    #    out = in
    # elif i == last:
    #    out = in
    # else:
    #    out = in
    for i in range(dim):
        for j in options:
            if j == 0:
                func += '\t' * tab + 'if ' + var[i] + '== 0:\n'
            elif j == 2:
                func += ('\t' * tab + ('el' if 0 in options else '')
                         +'if ' + var[i] + ' == ' + end[i] + ':\n')
            else:
                func += ('\t' * tab + 'else:\n')
            tab += 1

            if isPad:
                func += ('\t' * tab + 'for ' +
                         var[-1] + ' in range(' + end[-1] + '):\n')
                tab += 1

            func += '\t' * tab + lines[j][i] + '\n'
            tab -= 1 + isPad

    return func


def gen_g(dim, pad=0, GPU=True):
    start = 'def _g_' + str(dim) + '_' + str(pad) + '(f,Df):'

    lines = [[], [], []]
    var = ['i' + str(i) for i in range(dim + pad)]
    end = tuple('f.shape[' + str(i) + ']-1' for i in range(dim))
    # lines[0] = first index lines
    # lines[1] = middle index lines
    for i in range(dim):
        lines[1].append(
            'Df[' + ','.join(var) + ', ' + str(i) + '] = ' +
            'f[' + ','.join(var[:i] + [var[i] + '+1'] + var[(i + 1):]) + ']' +
            ' - f[' + ','.join(var) + ']')
    # lines[2] = end index lines
    for i in range(dim):
        lines[2].append(
            'Df[' + ','.join(var) + ', ' + str(i) + '] = 0')

    for j in range(pad):
        for i in range(3):
            lines[i].append('')
        end += ('f.shape[' + str(dim + j) + ']',)

    tab = 0
    func = '\t' * tab + start + '\n'
    tab += 1
    if GPU:
        return func + genbody_GPU(lines, var, end, tab, dim, (1, 2))
    else:
        return func + genbody(lines, var, end, tab, dim, (1, 2))


def gen_gt(dim, pad=0, GPU=True):
    start = 'def _gt_' + str(dim) + '_' + str(pad) + '(Df,f):'

    lines = [[], [], []]
    var = ['i' + str(i) for i in range(dim + pad)]
    end = tuple('Df.shape[' + str(i) + ']-1' for i in range(dim))
    # lines[0] = first index lines
    lines[0].append(
        'f[' + ','.join(var) + '] = -Df[' + ','.join(var) + ', 0]')
    for i in range(1, dim):
        lines[0].append(
            'f[' + ','.join(var) + '] -= Df[' + ','.join(var) + ', ' + str(i) + ']')
    # lines[1] = middle index lines
    lines[1].append(
        'f[' + ','.join(var) + '] = Df[' + ','.join([var[0] + '-1'] + var[1:]) + ', 0] - Df[' + ','.join(var) + ', 0]')
    for i in range(1, dim):
        lines[1].append(
            'f[' + ','.join(var) + '] += Df[' + ','.join(var[:i] + [var[i] + '-1'] + var[(i + 1):]) + ', ' + str(i) + '] - Df[' + ','.join(var) + ', ' + str(i) + ']')
    # lines[2] = end index lines
    lines[2].append(
        'f[' + ','.join(var) + '] = Df[' + ','.join([var[0] + '-1'] + var[1:]) + ', 0]')
    for i in range(1, dim):
        lines[2].append(
            'f[' + ','.join(var) + '] += Df[' + ','.join(var[:i] + [var[i] + '-1'] + var[(i + 1):]) + ', ' + str(i) + ' ]')

    for j in range(pad):
        for i in range(3):
            lines[i].append('')
        end += ('Df.shape[' + str(dim + j) + ']',)

    tab = 0
    func = '\t' * tab + start + '\n'
    tab += 1
    if GPU:
        return func + genbody_GPU(lines, var, end, tab, dim, (0, 1, 2))
    else:
        return func + genbody(lines, var, end, tab, dim, (0, 1, 2))


def gen_g2(dim, pad=0, GPU=True):
    start = 'def _g2_' + str(dim) + '_' + str(pad) + '(f,Df):'

    lines = [[], [], []]
    var = ['i' + str(i) for i in range(dim + pad)]
    end = tuple('f.shape[' + str(i) + ']' for i in range(dim))
    # lines[0] = first index lines
    for i in range(dim):
        lines[0].append(
            'Df[' + ','.join(var) + ', ' + str(i) + '] = f[' + ','.join(var) + ']')
    # lines[1] = middle index lines
    for i in range(dim):
        lines[1].append(
            'Df[' + ','.join(var) + ', ' + str(i) + '] = ' +
            'f[' + ','.join(var) + ']' +
            ' - f[' + ','.join(var[:i] + [var[i] + '-1'] + var[(i + 1):]) + ']')
    # lines[2] = end index lines

    for j in range(pad):
        for i in range(3):
            lines[i].append('')
        end += ('f.shape[' + str(dim + j) + ']',)

    tab = 0
    func = '\t' * tab + start + '\n'
    tab += 1
    if GPU:
        return func + genbody_GPU(lines, var, end, tab, dim, (0, 1))
    else:
        return func + genbody(lines, var, end, tab, dim, (0, 1))


def gen_g2t(dim, pad=0, GPU=True):
    start = 'def _g2t_' + str(dim) + '_' + str(pad) + '(Df,f):'

    lines = [[], [], []]
    var = ['i' + str(i) for i in range(dim + pad)]
    end = tuple('Df.shape[' + str(i) + ']-1' for i in range(dim))
    # lines[0] = first index lines
    # lines[1] = middle index lines
    lines[1].append(
        'f[' + ','.join(var) + '] = Df[' + ','.join(var) + ', 0] - Df[' + ','.join([var[0] + '+1'] + var[1:]) + ', 0]')
    for i in range(1, dim):
        lines[1].append(
            'f[' + ','.join(var) + '] += Df[' + ','.join(var) + ', ' + str(i) + '] - ' +
            'Df[' + ','.join(var[:i] + [var[i] + '+1'] + var[(i + 1):]) + ', ' + str(i) + ']')
    # lines[2] = end index lines
    lines[2].append(
        'f[' + ','.join(var) + '] = Df[' + ','.join(var) + ', 0]')
    for i in range(1, dim):
        lines[2].append(
            'f[' + ','.join(var) + '] += Df[' + ','.join(var) + ', ' + str(i) + ' ]')

    for j in range(pad):
        for i in range(3):
            lines[i].append('')
        end += ('Df.shape[' + str(dim + j) + ']',)

    tab = 0
    func = '\t' * tab + start + '\n'
    tab += 1
    if GPU:
        return func + genbody_GPU(lines, var, end, tab, dim, (1, 2))
    else:
        return func + genbody(lines, var, end, tab, dim, (1, 2))


GPU = False
if GPU:
    numbastr = ['@cuda.jit(', ')\n']
else:
    numbastr = [
        '@numba.jit(', 'nopython=True, parallel=False, fastmath=True, cache=True)\n']


def tosig(i, j):
    return ''
#     return ('["void( T[' + ','.join(':' * i) + '], T[' + ','.join(':' * j) + '])".replace("T",t) for t in ["f4","f8"]], ')


with open('_bin.py', 'w') as f:
    if GPU:
        print('import numba\nfrom numba import cuda\n', file=f)
    else:
        print('import numba\n', file=f)

    for i in range(3):
        for j in range(2):
            print(numbastr[0] + tosig(i + 1 + j, i + 2 + j) +
                  numbastr[1] + gen_g(i + 1, j, GPU), file=f)
#             print('print("' + str(8 * i + 4 * j + 1) + '/24", end="\\r")', file=f)
            print(numbastr[0] + tosig(i + 2 + j, i + 1 + j) +
                  numbastr[1] + gen_gt(i + 1, j, GPU), file=f)
#             print('print("' + str(8 * i + 4 * j + 2) + '/24", end="\\r")', file=f)

            print(numbastr[0] + tosig(i + 1 + j, i + 2 + j) +
                  numbastr[1] + gen_g2(i + 1, j, GPU), file=f)
#             print('print("' + str(8 * i + 4 * j + 3) + '/24", end="\\r")', file=f)

            print(numbastr[0] + tosig(i + 2 + j, i + 1 + j) +
                  numbastr[1] + gen_g2t(i + 1, j, GPU), file=f)
#             print('print("' + str(8 * i + 4 * j + 4) + '/24", end="\\r")', file=f)
importlib.invalidate_caches()
import _bin

c_diff = {'grad': {}, 'gradT': {}, 'grad2': {}, 'grad2T': {}, }
for i in range(3):
    for j in range(2):
        end = str(i + 1) + '_' + str(j)
        c_diff['grad'][end] = getattr(_bin, '_g_' + end)
        c_diff['gradT'][end] = getattr(_bin, '_gt_' + end)
        c_diff['grad2'][end] = getattr(_bin, '_g2_' + end)
        c_diff['grad2T'][end] = getattr(_bin, '_g2t_' + end)

###########################################################
# Helper functions to perform general convex optimisation
###########################################################


def Tuple(x):
    return tuple(x) if (hasattr(x, '__iter__') and (type(x) is not str)) else (x,)


def Vector(*x):
    # Test for generators:
    if hasattr(x[0], 'gi_yieldfrom'):
        x = tuple(x[0])
#     # Ignore length 1 vectors:
#     if len(x) == 1:
#         return x[0]

    X = array(x, dtype=object)
    if X.ndim != 1:
        X = empty(len(x), dtype=object)
        X[:] = x
    return X


class scalar_mat(LinearOperator):

    def __init__(self, shape, scale=0, dtype='f4'):
        LinearOperator.__init__(self, dtype, shape)
        self.scale = scale
        self._transpose = self._adjoint

    def _matvec(self, x):
        s = self.scale
        if s == 0:
            return zeros(self.shape[0], dtype=x.dtype)
        elif s == 1:
            return x.copy()
        elif s == -1:
            return -x
        else:
            return s * x

    def _rmatvec(self, x):
        s = self.scale
        if s == 0:
            return zeros(self.shape[1], dtype=x.dtype)
        elif s == 1:
            return x.copy()
        elif s == -1:
            return -x
        else:
            return s * x

    def norm(self): return self.scale


class Matrix(LinearOperator):

    def __init__(self, m, shape=None, _adjoint=None):
        # Create block matrix
        if shape is None:
            buf = array(m, dtype=object)
            if buf.ndim < 2:
                buf.shape = [-1, 1]
            elif buf.ndim > 2:
                if type(m) not in (list, tuple):
                    buf = empty((1, 1), dtype=object)
                    buf[0, 0] = m
                elif type(m[0]) not in (list, tuple):
                    buf = empty((len(m), 1), dtype=object)
                    buf[:, 0] = m
                else:
                    buf = empty((len(m), len(m[0])), dtype=object)
                    for i in range(len(m)):
                        buf[i] = m[i]
        else:
            buf = empty(shape, dtype=object)
            for i in range(len(m)):
                buf[i] = m[i]

        # Check shapes of blocks
        h, w = 0 * empty(buf.shape[0], dtype=int), 0 * \
            empty(buf.shape[1], dtype=int)
        for i in range(buf.shape[0]):
            for j in range(buf.shape[1]):
                b = buf[i, j]
                if hasattr(b, 'shape'):
                    h[i], w[j] = b.shape[:2]
        if (h.min() == 0) or (w.min() == 0):
            raise ValueError('Every row and column must have a known shape')
        for i in range(buf.shape[0]):
            for j in range(buf.shape[1]):
                if buf[i, j] is None:
                    buf[i, j] = scalar_mat((h[i], w[j]), 0)
                elif isscalar(buf[i, j]):
                    buf[i, j] = scalar_mat((h[i], w[j]), buf[i, j])
                elif not hasattr(buf[i, j], 'shape'):
                    buf[i, j].shape = (h[i], w[j])

        LinearOperator.__init__(self, object, buf.shape)

        self.m = buf
        self.shapes = (h, w)

        if _adjoint is None:
            # Adjoint operator
            buf = empty((buf.shape[1], buf.shape[0]), dtype=object)
            for i in range(buf.shape[0]):
                for j in range(buf.shape[1]):
                    buf[i, j] = self.m[j, i].H
            self.mH = Matrix(buf, _adjoint=self)
        else:
            self.mH = _adjoint

    def _matvec(self, x): return self.m.dot(x)

    def _rmatvec(self, y): return self.mH.dot(y)

    def _transpose(self):
        return self.H

    def _adjoint(self):
        return self.mH

    def __getitem__(self, *args, **kwargs):
        return self.m.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        self.m.__setitem__(*args, **kwargs)


class diff(LinearOperator):

    def __init__(self, shape, order=1, spectDim=0, dtype='f4', bumpup=False):
        if order not in [0, 1, 2]:
            raise ValueError('order = %s not supported' % str(order))

        if order == 0:
            scalar_mat.__init__(self, shape, 1, dtype)
            return

        dim = len(shape) - spectDim

        if order == 1:
            shape2 = (dim * prod(shape), prod(shape))
            if bumpup:
                shape2 = tuple(dim * s for s in shape2)
        elif order == 2:
            shape2 = (dim ** 2 * prod(shape), prod(shape))
        LinearOperator.__init__(self, dtype, shape2)
        self.dim = dim
        self.vol_shape = tuple(shape[:dim])

        self.order = order
        if order == 1:
            if bumpup:
                self._matvec = self.__grad2
                self._rmatvec = self.__grad2T
            else:
                self._matvec = self.__grad
                self._rmatvec = self.__gradT
        elif order == 2:
            self._matvec = self.__hess
            self._rmatvec = self.__hessT

        self._transpose = self._adjoint

    def __grad(self, f):
        # Forward difference with Neumann boundary:
        # df/dx[i] = f[i+1]-f[i]
        ravel = (f.ndim == 1)
        f = f.reshape(*self.vol_shape, -1)
        dim, spect = self.dim, f.shape[-1]
        Df = empty(f.shape + (dim,), dtype=f.dtype, order='C')

        mystr = str(dim) + '_' + '0'
        if c_diff['grad'].get(mystr, None) is not None:
            func = c_diff['grad'][mystr]
            if hasattr(func, 'targetdescr'):
                if spect == 1:
                    func(f[..., 0], Df.reshape(self.vol_shape + (dim,)))
                else:
                    tmp = empty(self.vol_shape + (dim,), dtype=f.dtype, order='C')
                    for i in range(f.shape[-1]):
                        func(f[..., i], tmp)
                        Df[..., i, :] = tmp
            else:
                if dim == 1:
                    tpb = (256,)
                elif dim == 2:
                    tpb = (16, 16)
                else:
                    tpb = (8,) * 3
                blocks = tuple(-(-f.shape[i] // tpb[i]) for i in range(dim))

                if spect == 1:
                    func[blocks, tpb](f[..., 0], Df.reshape(self.vol_shape + (dim,)))
                else:
                    tmp = empty(self.vol_shape + (dim,), dtype=f.dtype, order='C')
                    for i in range(f.shape[-1]):
                        func[blocks, tpb](ascontiguousarray(f[..., i]), tmp)
                        Df[..., i, :] = tmp

        else:
            for i in range(dim):
                null = [slice(None) for _ in range(dim + 1)]
                x = [slice(None) for _ in range(dim + 1)]
                xp1 = [slice(None) for _ in range(dim + 1)]
                null[i] = -1
                x[i], xp1[i] = slice(-1), slice(1, None)

                Df[tuple(x) + (i,)] = f[tuple(xp1)] - f[tuple(x)]
                Df[tuple(null) + (i,)] = 0

        if ravel:
            return Df.ravel()
        else:
            return Df

    def __gradT(self, Df):
        # Adjoint of forward difference with Neumann boundary
        # is backward difference divergence with Dirichlet boundary.
        # The numerical adjoint assumes Df[...,-1,...] = 0 too.
        ravel = (Df.ndim == 1)
        dim = self.dim
        Df = Df.reshape(*self.vol_shape, -1, dim)
        f = empty(Df.shape[:-1], dtype=Df.dtype, order='C')
        spect = f.shape[-1]

        mystr = str(dim) + '_' + '0'
        if c_diff['gradT'].get(mystr, None) is not None:
            func = c_diff['gradT'][mystr]
            if hasattr(func, 'targetdescr'):
                if spect == 1:
                    func(Df[..., 0, :], f.reshape(self.vol_shape))
                else:
                    tmp = empty(self.vol_shape, dtype=f.dtype, order='C')
                    for i in range(f.shape[-1]):
                        func(Df[..., i, :], tmp)
                        f[..., i] = tmp
            else:
                if dim == 1:
                    tpb = (256,)
                elif dim == 2:
                    tpb = (16, 16)
                else:
                    tpb = (8,) * 3
                blocks = tuple(-(-f.shape[i] // tpb[i]) for i in range(dim))
                if spect == 1:
                    func[blocks, tpb](Df[..., 0, :], f.reshape(self.vol_shape))
                else:
                    tmp = empty(self.vol_shape, dtype=f.dtype, order='C')
                    for i in range(f.shape[-1]):
                        func[blocks, tpb](ascontiguousarray(Df[..., i, :]), tmp)
                        f[..., i] = tmp
        else:
            f *= 0
            # We implement the numerical adjoint
            for i in range(dim):
                x = [slice(None) for _ in range(dim + 2)]
                xm1 = [slice(None) for _ in range(dim + 2)]
                x[i], xm1[i] = slice(1, -1), slice(-2)
                x[-1], xm1[-1] = i, i
                f[tuple(x[:-1])] += Df[tuple(xm1)] - Df[tuple(x)]

                first = [slice(None) for _ in range(dim + 2)]
                first[i], first[-1] = 0, i
                f[tuple(first[:-1])] -= Df[tuple(first)]

                x = [slice(None) for _ in range(dim + 2)]
                xm1 = [slice(None) for _ in range(dim + 2)]
                x[i], xm1[i] = slice(-1, None), slice(-2, -1)
                x[-1], xm1[-1] = i, i
                f[tuple(x[:-1])] += Df[tuple(xm1)]

        if ravel:
            return f.ravel()
        else:
            return f

    def __grad2(self, f):
        raise NotImplementedError('Copy the spectral versions from __grad')
        # Backward difference with Dirichlet boundary:
        # df/dx[i] = f[i+1]-f[i]
        ravel = (f.ndim == 1)
        dim = len(self.vol_shape)

        if f.size == prod(self.vol_shape):
            # compute f -> Df
            f = f.reshape(self.vol_shape)
        else:
            # compute Df -> D^2f
            f = f.reshape(*self.vol_shape, dim)

        Dim = f.ndim
        Df = empty(f.shape + (dim,), dtype=f.dtype, order='C')

        mystr = str(dim) + '_' + ('0' if dim == Dim else '1')
        if c_diff['grad2'].get(mystr, None) is not None:
            func = c_diff['grad2'][mystr]
            if hasattr(func, 'targetdescr'):
                func(f, Df)
            else:
                if dim == 1:
                    tpb = (256,)
                elif dim == 2:
                    tpb = (16, 16)
                else:
                    tpb = (8,) * 3
                blocks = tuple(-(-f.shape[i] // tpb[i]) for i in range(dim))
                func[blocks, tpb](f, Df)
        else:
            for i in range(dim):
                x = [slice(None) for _ in range(Dim)]
                xm1 = [slice(None) for _ in range(Dim)]
                x[i], xm1[i] = slice(1, None), slice(-1)
                Df[tuple(x) + (i,)] = f[tuple(x)] - f[tuple(xm1)]

                null = [slice(None) for _ in range(Dim)]
                null[i] = 0
                Df[tuple(null) + (i,)] = f[tuple(null)]

        if ravel:
            return Df.ravel()
        else:
            return Df

    def __grad2T(self, Df):
        raise NotImplementedError('Copy the spectral versions from __gradT')
        # Adjoint of backward difference with Dirichlet boundary
        # is forward difference divergence with Neumann boundary.
        # The numerical adjoint is also Dirichlet boundary.
        ravel = (Df.ndim == 1)
        dim = len(self.vol_shape)

        if Df.size == prod(self.vol_shape) * dim:
            # compute Df -> div(Df)
            Df = Df.reshape(*self.vol_shape, dim)
        else:
            # compute D^2f -> div(D^2f)
            Df = Df.reshape(*self.vol_shape, dim, dim)

        Dim = Df.ndim - 1
        f = empty(Df.shape[:-1], dtype=Df.dtype, order='C')

        mystr = str(dim) + '_' + ('0' if dim == Dim else '1')
        if c_diff['grad2T'].get(mystr, None) is not None:
            func = c_diff['grad2T'][mystr]
            if hasattr(func, 'targetdescr'):
                func(Df, f)
            else:
                if dim == 1:
                    tpb = (256,)
                elif dim == 2:
                    tpb = (16, 16)
                else:
                    tpb = (8,) * 3
                blocks = tuple(-(-f.shape[i] // tpb[i]) for i in range(dim))
                func[blocks, tpb](Df, f)
        else:
            f *= 0
            # We implement the numeric adjoint
            for i in range(dim):
                x = [slice(None) for _ in range(Dim + 1)]
                xp1 = [slice(None) for _ in range(Dim + 1)]
                x[i], xp1[i] = slice(0, -1), slice(1, None)
                x[-1], xp1[-1] = i, i
                f[tuple(x[:-1])] += Df[tuple(x)] - Df[tuple(xp1)]

                last = [slice(None) for _ in range(Dim + 1)]
                last[i], last[-1] = -1, i
                f[tuple(last[:-1])] += Df[tuple(last)]

        if ravel:
            return f.ravel()
        else:
            return f

    def __hess(self, f):
        raise NotImplementedError('Copy the spectral versions from __grad')
        # Symmetrised Forward-Backward differences:
        # d^2fdxdy = 1/2(f[i+1,j]+f[i,j+1]+f[i-1,j]+f[i,j-1]
        #                -f[i-1,j+1]-f[i+1,j-1]-2f[i,j])
        # which equates to computing forward differences
        # then backwards then symmetrising.
        ravel = (f.ndim == 1)
        f = f.reshape(self.vol_shape)
        dim = f.ndim

        D2f = self.__grad2(self.__grad(f))

        try:
            sym(D2f.reshape(-1, dim, dim), D2f.reshape(-1, dim, dim))
        except NameError:
            for i in range(dim):
                for j in range(i + 1, dim):
                    D2f[..., i, j] = 0.5 * (D2f[..., i, j] + D2f[..., j, i])
                    D2f[..., j, i] = D2f[..., i, j]

        if ravel:
            return D2f.ravel()
        else:
            return D2f

    def __hessT(self, D2f):
        raise NotImplementedError('Copy the spectral versions from __gradT')
        # Adjoint of symmetrisation is symmetrisation
        ravel = (D2f.ndim == 1)
        dim = len(self.vol_shape)

        try:
            D2f = D2f.reshape(-1, dim, dim)
            tmp = empty(D2f.shape, dtype=D2f.dtype)
            sym(D2f, tmp)
            D2f = tmp.reshape(*self.vol_shape, dim, dim)
        except Exception:
            D2f = D2f.copy().reshape(*self.vol_shape, dim, dim)
            for i in range(dim):
                for j in range(i + 1, dim):
                    D2f[..., i, j] = 0.5 * (D2f[..., i, j] + D2f[..., j, i])
                    D2f[..., j, i] = D2f[..., i, j]

        D2f = self.__gradT(self.__grad2T(D2f))

        if ravel:
            return D2f.ravel()
        else:
            return D2f

    def _matvec(self, _):
        # Dummy function for LinearOperator
        return None

    def norm(self, p=2):
        dim = self.dim
        if self.order == 1:
            if p == 1:
                return 2 * dim
            elif p == 2:
                return 2 * dim ** .5
            elif p in ['inf', inf]:
                return 2
        elif self.order == 2:
            if p == 1:
                return 4 * dim ** 2
            elif p == 2:
                return 4 * dim
            elif p in ['inf', inf]:
                return 4
        else:
            if p == 1:
                return (2 * dim) ** self.order
            elif p == 2:
                return (4 * dim) ** (self.order / 2)
            elif p in ['inf', inf]:
                return 2 ** self.order


def getVec(mat, rand=False):
    if len(mat.shape) in [1, 2]:
        if mat.dtype == object:
            Z = empty(mat.shape[-1], dtype=mat.dtype)
            if len(mat.shape) == 1:
                Z = Vector(getVec(m, rand) for m in mat)
            else:
                Z = Vector(getVec(m, rand) for m in mat[0, :])
        else:
            if rand:
                Z = random.rand(mat.shape[-1]).astype(mat.dtype)
            else:
                Z = zeros(mat.shape[-1], dtype=mat.dtype)
    else:
        raise ValueError('input must either be or dimension 1 or 2')
    return Z


def vecNorm(x):
    x = x.reshape(-1)
    if x.dtype == object:
        x = array([vecNorm(xi) for xi in x])
    return norm(x.reshape(-1))


def vecIP(x, y):
    x, y = x.reshape(-1), y.reshape(-1)
    if x.dtype == object:
        return sum(vecIP(x[i], y[i]) for i in range(x.size))
    else:
        return x.T.dot(y)


def poweriter(mat, maxiter=300):
    x = getVec(mat, rand=True)
    for _ in range(maxiter):
        x /= vecNorm(x)
        x = mat.H * (mat * x)
    return vecNorm(x) ** .5


class proxFunc():

    def __init__(self, f, prox=None, setprox=None, violation=None, grad=None):
        self._f = f
        self._prox = prox
        if setprox is None:

            def tmp(_): raise NotImplementedError

            self._setprox = tmp
        else:
            self._setprox = setprox

        if violation is None:
            self._violation = lambda _: 0
        else:
            self._violation = violation

        if grad is None:

            def tmp(_): raise NotImplementedError

            self._grad = tmp
        else:
            self._grad = grad

    def __call__(self, x):
        return self._f(x.ravel())

    def setprox(self, t):
        return self._setprox(t)

    def prox(self, x, t=None):
        if t is not None:
            try:
                self.setprox(t)
            except NotImplementedError:
                return self._prox(x.ravel(), t).ravel()
        return self._prox(x.ravel()).ravel()

    def violation(self, x):
        return self._violation(x.ravel())

    def grad(self, x):
        return self._grad(x.ravel()).ravel()


def dualableFunc(f, fstar):
    f.dual = fstar
    fstar.dual = f
    return f


class stackProxFunc(proxFunc):

    def __init__(self, *args, ignoreDual=False):
        proxFunc.__init__(self, self.__call__)
        self.__funcs = tuple(args)

        if (not ignoreDual) and all(hasattr(f, 'dual') for f in args):
            dualableFunc(self, stackProxFunc(
                *(f.dual for f in args), ignoreDual=True))

    def __call__(self, x):
        return sum(self.__funcs[i](x[i]) for i in range(len(x)))

    def setprox(self, t):
        if isscalar(t):
            for f in self.__funcs:
                f.setprox(t)
        else:
            for i in range(len(t)):
                self.__funcs[i].setprox(t[i])

    def prox(self, x):
        return Vector(*tuple(self.__funcs[i].prox(x[i]) for i in range(len(x))))

    def violation(self, x):
        return sum(self.__funcs[i].violation(x[i]) for i in range(len(x)))

    def grad(self, x):
        return Vector(*tuple(self.__funcs[i].grad(x[i]) for i in range(len(x))))


class ZERO(proxFunc):

    def __init__(self):
        proxFunc.__init__(self,
                          lambda _: 0,
                          prox=lambda x: x,
                          setprox=lambda _: None,
                          violation=lambda _: 0,
                          grad=lambda x: 0 * x)

        dual = proxFunc(lambda _: 0,
                        prox=lambda x: 0 * x,
                        setprox=lambda _: None,
                        violation=lambda x: abs(x).max())
        dualableFunc(self, dual)


class NONNEG(proxFunc):

    def __init__(self):
        proxFunc.__init__(self,
                          lambda _: 0,
                          prox=lambda x: maximum(0, x),
                          setprox=lambda _: None,
                          violation=lambda x: max(0, -x.min()))

        dual = proxFunc(lambda _: 0,
                        prox=lambda x: minimum(0, x),
                        setprox=lambda _: None,
                        violation=lambda x: max(0, x.max()))
        dualableFunc(self, dual)


class L2(proxFunc):

    def __init__(self, scale=1, translation=None):
        '''
        f(x) = scale/2|x-translation|^2
             = s/2|x-d|^2
        df = s(x-d)
        lip = hess = s
        proxf = argmin 1/2|x-X|^2 + st/2|x-d|^2
              = (X+std)/(1+st)

        g(y) = sup <x,y> - f(x)
             = 1/(2s)|y|^2 + <d,y>
        dg = y/s + d
        lip = hess = 1/s
        proxg = argmin 1/2|y-Y|^2 + t/(2s)|y|^2 + t<d,y>
              = (Y-td)/(1+t/s)

        f(X,x) = |X+ix - T-it|^2 = f(X) + f(x)
        g(Y,y) = sup <X,Y>+<x,y> - (X-T)^2 - (x-t)^2
               = g(Y) + g(y)
        proxg(Y,y) = (proxg(Y), proxg(y))
        '''
        if translation is None:

            def f(x): return (scale * .5) * norm(x, 2) ** 2

            def g(y): return (.5 / scale) * norm(y, 2) ** 2

            if scale == 1:

                def df(x): return x

                def dg(y): return y

            else:

                def df(x): return scale * x

                def dg(y): return (1 / scale) * y

            def setproxf(t): self.__proxfparam = 1 / (1 + scale * t)

            def setproxg(t): self.__proxgparam = 1 / (1 + t / scale)

            def proxf(x): return x * self.__proxfparam

            def proxg(y): return y * self.__proxgparam

        else:
            translation = translation.reshape(-1)

            def f(x):
                x = x - translation
                return (scale * .5) * norm(x, 2) ** 2

            def g(y):
                return (.5 / scale) * norm(y, 2) ** 2 + (y.conj() * translation).real.sum()

            if scale == 1:

                def df(x): return x - translation

                def dg(y): return y + translation

            else:

                def df(x): return scale * (x - translation)

                def dg(y): return (1 / scale) * y + translation

            def setproxf(t):
                self.__proxfparam = (
                    1 / (1 + scale * t), translation * (scale * t / (1 + scale * t)))

            def setproxg(t):
                self.__proxgparam = (
                    1 / (1 + t / scale), translation * (t / (1 + t / scale)))

            def proxf(x):
                return x * self.__proxfparam[0] + self.__proxfparam[1]

            def proxg(y):
                return y * self.__proxgparam[0] - self.__proxgparam[1]

        proxFunc.__init__(self, f, prox=proxf, setprox=setproxf, grad=df)
        dual = proxFunc(g, prox=proxg, setprox=setproxg, grad=dg)
        dualableFunc(self, dual)


class L1(proxFunc):

    def __init__(self, scale=1, translation=None):
        '''
        f(x) = scale|x-translation|_1
             = s|x-d|_1
        df(x) = s*sign(x)
        lip = hess = nan
        prox(x,t) = argmin_X 1/2|X-x|^2 + st|X-d|
                  = d + argmin_X 1/2|X-(x-d)|^2 + st|X|
            X + (st)sign(X) = x-d
            X = x-d -st, 0, x-d +st

        g(y) = sup_x <y,x> - s|x-d|
             = <y,d> if |y|_\infty <= s
        dg(y) = d
        lip = hess = nan
        prox(y,t) = argmin 1/2|Y-y|^2 + t<Y,d> s.t. |Y|< s
                  = proj_{|Y|<s}(y-td)


        '''
        if translation is None:

            def f(x): return scale * norm(x, 1)

            def g(_): return 0

            def setproxf(t): self.__proxfparam = scale * t

            def setproxg(_): self.__proxgparam = 0

            def proxf(x):
                t = self.__proxfparam
                X = x.copy()
                indPos, indNeg = x > t, x < -t
                X[indPos] -= t
                X[indNeg] += t
                X[logical_not(logical_or(indPos, indNeg))] = 0
                return X

            def proxg(y):
                Y = y.copy()
                Y[Y > scale] = scale
                Y[Y < -scale] = -scale
                return Y

        else:
            translation = translation.reshape(-1)

            def f(x): return scale * norm(x - translation, 1)

            def g(y): return (y.conj() * translation).sum()

            def setproxf(t): self.__proxfparam = scale * t

            def setproxg(t): self.__proxgparam = translation * t

            def proxf(x):
                t = self.__proxfparam
                X = x - translation
                indPos, indNeg = x > t, x < -t
                X[indPos] -= t
                X[indNeg] += t
                X[logical_not(logical_or(indPos, indNeg))] = 0
                return X + translation

            def proxg(y):
                Y = y - self.__proxgparam
                Y[Y > scale] = scale
                Y[Y < -scale] = -scale
                return Y

        def violation(x): return max(0, norm(x, inf) / scale - 1)

        proxFunc.__init__(self, f, prox=proxf, setprox=setproxf)
        dual = proxFunc(g, prox=proxg,
                        setprox=setproxg, violation=violation)
        dualableFunc(self, dual)


class L1_2(proxFunc):

    def __init__(self, size, scale=1, translation=None):
        '''
        f(x) = scale||x-translation|_2|_1
             = s||x-d|_2|_1
        df(x) = s*x/|x|_2
        lip = hess = nan
        prox(x,t) = argmin_X 1/2|X-x|^2 + st|X-d|_2
                  = d + argmin_X 1/2|X-(x-d)|^2 + st|X|_2
            X-d + (st)sign(X-d) = x-d
            |X-d| = max(0, |x-d| -st)

        g(y) = sup_x <y,x> - s|x-d|_2
             = <y,d> if |y|_2 <= s
        dg(y) = d
        lip = hess = nan
        prox(y,t) = argmin 1/2|Y-y|^2 + t<Y,d> s.t. |Y|_2< s
                  = proj_{|Y|_2<s}(y-td)

        Vector norm is computed along axis 0
        '''

        self.shape = (size, -1)

        if translation is None:

            def f(x):
                x = norm(x.reshape(self.shape), 2, -1, True)
                return scale * x.sum()

            def g(_): return 0

            def setproxf(t): self.__proxfparam = scale * t

            def setproxg(_): self.__proxgparam = 0

            def proxf(x):
                t = self.__proxfparam
                n = self.__vecnorm(x.reshape(self.shape))
                n = maximum(0, 1 - t / n)
                return x * n

            def proxg(y):
                y = y.reshape(self.shape)
                out = empty(y.shape, dtype=y.dtype)
                shrink(y, scale, out)
                return out

        else:
            translation = translation.reshape(self.shape)

            def f(x):
                x = x.reshape(self.shape) - translation
                x = norm(x, 2, -1, True)
                return scale * x.sum()

            def g(y): return (y.conj().reshape(self.shape) * translation).sum()

            def setproxf(t): self.__proxfparam = scale * t

            def setproxg(t): self.__proxgparam = translation * t

            def proxf(x):
                t = self.__proxfparam
                x = x.reshape(self.shape) - translation
                n = self.__vecnorm(x)
                n = maximum(0, 1 - t / n)
                return x * n + translation

            def proxg(y):
                y = y.reshape(self.shape) - self.__proxgparam
                shrink(y, scale, y)
                return y

        def violation(y):
            y = norm(y.reshape(self.shape), 2, axis=-1)
            return max(0, y.max() / scale - 1)

        proxFunc.__init__(self, f, prox=proxf, setprox=setproxf)
        dual = proxFunc(g, prox=proxg, setprox=setproxg, violation=violation)
        dualableFunc(self, dual)

    def __vecnorm(self, x):
        x = norm(x, 2, -1, True)
        return x + 1e-8


class PDHG:

    def __init__(self, A, f, g):
        '''
        Compute saddle points of:
         f(x) + <Ax,y> - g(y)
        '''
        self.A = A
        self.f = f
        self.g = g
        self.x, self.y = None, None
        self.s, self.t = None, None
        self.tol = None

    def setParams(self, A=None, f=None, g=None, x=None, y=None, sigma=None,
                  tau=None, balance=None, normA=None, tol=None,
                  steps=None, stepParams=None, **_):
        # Reset problem
        if f is not None:
            self.f = f
        if g is not None:
            self.g = g
        if A is not None:
            self.A = A

        # Set starting point
        if x is not None:
            self.x = x
            self.xm1 = x
        if y is not None:
            self.y = y
            self.ym1 = y

        self.Ax, self.Ay = self.A * self.x, self.A.T * self.y
        self.Axm1, self.Aym1 = self.Ax, self.Ay

        # Set initial step sizes
        # tau/sigma is primal/dual step size respectively
        if (sigma is not None) and (tau is not None):
            sigma, tau = sigma, tau
        elif (balance is None) and (normA is None) and (self.s is not None):
            sigma, tau = self.s, self.t
        else:
            balance = 1 if balance is None else balance
            if normA is None:
                normA = poweriter(self.A)
                print('norm of matrix computed: ', normA)
            sigma, tau = balance / normA, 1 / (balance * normA)
        self.s, self.t = sigma, tau
        self.f.setprox(self.s)
        self.g.setprox(self.t)

        # Set adaptive step criterion
        stepParams = {} if stepParams is None else stepParams
        if steps in [None, 'None']:
            self._stepsize = lambda _: False
        elif steps[0].lower() == 'a':
            if not (isscalar(self.s) and isscalar(self.t)):
                raise ValueError(
                    'Step sizes must be scalar for adaptive choice')
            self._stepsize = self._adaptive(**stepParams)
        elif steps[0].lower() == 'b':
            if not (isscalar(self.s) and isscalar(self.t)):
                raise ValueError(
                    'Step sizes must be scalar for backtracking choice')
            self._stepsize = self._backtrack(**stepParams)
            # Up to a factor of 2 off on normA estimate
            self.s *= 2
            self.t *= 2
        else:
            raise ValueError('steps must be None, adaptive or backtrack.')

        # Set tolerance
        if tol is not None:
            self.tol = tol
        elif self.tol is None:
            self.tol = 1e-4

    def run(self, maxiter=100, callback=None, callback_freq=10, **kwargs):
        from time import time
        tic = time()

        self.start(**kwargs)

        if callback is None:
            print('Started reconstruction... ', flush=True, end='')
            self.step(0, maxiter)
            print('Finished after ' + str(int(time() - tic)), 's')
        else:
            callback = ('Iter', 'Time',) + Tuple(callback)
            prints = []

            def padstr(x, L=13):
                x = str(x)
                l = max(0, L - len(x))
                return ' ' * int(l / 2) + x + ' ' * (l - int(l / 2))

            def frmt(x):
                if type(x) == int:
                    x = '% 3d' % x
                elif isscalar(x):
                    x = '% 1.3e' % float(x)
                else:
                    x = str(x)
                return padstr(x)

            i = 0
            print(padstr(callback[0], 6), padstr(callback[1], 6),
                  *(padstr(c) for c in callback[2:]))
            prints.append((i, time() - tic,
                           ) + Tuple(self.callback(callback[2:])))
            print(padstr('%3d%%' % (i / maxiter * 100), 6), padstr('%3ds' % prints[-1][1], 6),
                  *(frmt(c) for c in prints[-1][2:]), flush=True)
            while i < maxiter:
                leap = min(callback_freq, maxiter - i)
                self.step(i, leap)
                i += leap
                prints.append((i, time() - tic,) +
                              Tuple(self.callback(callback[2:])))
                print(padstr('%3d%%' % (i / maxiter * 100), 6), padstr('%3ds' % prints[-1][1], 6),
                      *(frmt(c) for c in prints[-1][2:]), flush=True)
            print()

            dtype = [(callback[i], ('S20' if type(prints[0][i]) is str else 'f4'))
                     for i in range(len(callback))]
            prints = array(prints, dtype)
            Q = {callback[j]: prints[callback[j]]
                 for j in range(len(callback))}

        if callback is None:
            return self.getRecon()
        else:
            return self.getRecon(), Q

    def start(self, A=None, x=None, y=None, **kwargs):
        # Set matrix
        if A is None:
            if self.A is None:
                raise ValueError('Matrix A must be provided')
        else:
            self.A = A

        # Choose starting point:
        if (x is None) and (self.x is None):
            x = getVec(self.A, rand=False)
        if (y is None) and (self.y is None):
            y = getVec(self.A.T, rand=False)

        self.setParams(x=x, y=y, **kwargs)

    def step(self, i, niter):
        for _ in range(niter):
            # Primal step:
            tmp = self.x
            self.x = self.f.prox(self.x - self.t * self.Ay)
            self.xm1, self.Axm1 = tmp, self.Ax
            self.Ax = self.A * self.x

            # Dual step:
            tmp = self.y
            self.y = self.g.prox(self.y + self.s * (2 * self.Ax - self.Axm1))
            self.ym1, self.Aym1 = tmp, self.Ay
            self.Ay = self.A.T * self.y

            # Check step size:
            if self._stepsize(self):
                self.f.setprox(self.t)
                self.g.setprox(self.s)

    def _backtrack(self, beta=0.95, gamma=0.8):
        self.stepParams = {'beta': beta, 'gamma': gamma}

        def stepsize(alg):
            dx = (alg.x - alg.xm1)
            dy = (alg.y - alg.ym1)

            b = (2 * alg.t * alg.s * vecIP(dx, alg.Ay - alg.Aym1).real) / (
                alg.s * vecNorm(dx) ** 2 + alg.t * vecNorm(dy) ** 2)

            if b > gamma:
                b *= beta / gamma
                alg.s /= b
                alg.t /= b
                return True
            else:
                return False

        return stepsize

    def _adaptive(self, alpha=0.5, eta=0.95, delta=1.5, s=1):
        self.stepParams = {'alpha': alpha, 'eta': eta, 'delta': delta, 's': s}
        params = (alpha, eta, s * delta, s / delta)
        a = [alpha / eta]

        def stepsize(alg):
            p = vecNorm((alg.x - alg.xm1) / alg.t - alg.Ay + alg.Aym1)
            d = vecNorm((alg.y - alg.ym1) / alg.s + alg.Ax - alg.Axm1)
            r = p / d

            if r > params[2]:
                a[0] *= params[1]
                self.s *= 1 - a[0]
                self.t /= 1 - a[0]
                return True
            elif r < params[3]:
                a[0] *= params[1]
                self.s /= 1 - a[0]
                self.t *= 1 - a[0]
                return True
            else:
                return False

        return stepsize

    def callback(self, names):
        for n in names:
            if n == 'grad':
                yield self._grad
            elif n == 'gap':
                yield self._gap
            elif n == 'primal':
                yield self._prim
            elif n == 'dual':
                yield self._dual
            elif n == 'violation':
                yield self._violation
            elif n == 'step':
                yield self._step

    def getRecon(self): return self.recon

    @property
    def _grad(self):
        p = vecNorm((self.x - self.xm1) / self.t - self.Ay + self.Aym1)
        d = vecNorm((self.y - self.ym1) / self.s + self.Ax - self.Axm1)
        return p / (1e-8 + vecNorm(self.x)) + d / (1e-8 + vecNorm(self.y))

    @property
    def _prim(self):
        return self.f(self.x) + self.g.dual(self.Ax)

    @property
    def _dual(self):
        return self.f.dual(-self.Ay) + self.g(self.y)

    @property
    def _gap(self):
        '''
        f(x) + f^*(-A^Ty) >= -<x,A^Ty>
        g^*(Ax) + g(y) >= <Ax,y>
        '''
        p = self._prim
        d = self._dual
        z = vecIP(self.Ax, self.y) - vecIP(self.x, self.Ay)
        return (p + d - z) / (abs(p) + abs(d))

    @property
    def _step(self):
        x = vecNorm(self.x - self.xm1) / (1e-8 + vecNorm(self.x))
        y = vecNorm(self.y - self.ym1) / (1e-8 + vecNorm(self.y))
        return x + y

    @property
    def _violation(self):
        return (self.f.violation(self.x) + self.f.dual.violation(-self.Ay)
                +self.g.dual.violation(self.Ax) + self.g.violation(self.y))


class TV(PDHG):

    def __init__(self, shape, op=None, order=1, spectDim=0, weight=0.1, pos=False, **kwargs):
        '''
        Minimise the energy:
        1/2|op(u)-data|^2 + weight*|\nabla^{order}u|_1
        where:
            shape: the shape of the output, u
            order=1: integer 0, 1 or 2
            op=None: if provided, the forward operator to use
            weight=0.1: positive scalar value, almost definitely less than 1
            pos=False: If true then a non-negativity constraint is applied
        '''
        PDHG.__init__(self, None, None, None, **kwargs)
        self.shape = shape
        self.op = op
        self.pos = pos
        self.d = diff(shape, order=order, spectDim=spectDim)
        self.order = order
        self.weight = weight
        self.data = None

    def setParams(self, data, op=None, x=None, y=None, **kwargs):
        vol_shape = self.d.vol_shape
        if op is not None:
            self.op = op
        elif self.op is None:
            self.op = scalar_mat([data.size] * 2, 1)

        self.weight = kwargs.get('weight', self.weight)

        normR = self.op.norm()
        normD = self.d.norm()
        dataScale = (normD / normR) ** 2 * abs(self.op.T * data.reshape(-1)).max()
        data = data.reshape(-1) * (normD / normR) / dataScale
        self.dataScale = dataScale

        if self.weight > 0:
            A = Matrix([[(normD / normR) * self.op], [self.d]])
            normA = normD * 2 ** .5
    #         norm = poweriter(A)
    #         print('operator norm = ', norm)
    #         print('estimate = ', normA)
    #         exit()

            gstar = stackProxFunc(
                L2(scale=1, translation=data),
                L1_2(size=prod(vol_shape), scale=self.weight)
            )
        else:
            A = Matrix([[(normD / normR) * self.op]])
            normA = normD
            gstar = stackProxFunc(L2(scale=1, translation=data))

        # Choose starting point:
        if (x is None) and (self.x is None):
            x = getVec(A, rand=False)
        elif x is not None and x.dtype != object:
            x = Vector(x.reshape(-1) / self.dataScale)
        if (y is None) and (self.y is None):
            y = getVec(A.T, rand=False)
        elif y is not None and y.dtype != object:
            y = Vector(y.reshape(-1))

        if 'steps' not in kwargs:
            kwargs['steps'] = 'backtrack'

        self.pos = kwargs.get('pos', self.pos)

        PDHG.setParams(self, A=A,
                       f=stackProxFunc(NONNEG() if self.pos else ZERO()),
                       g=gstar.dual, x=x,
                       y=y, normA=normA, **kwargs)

    def start(self, data=None, x=None, y=None, **kwargs):
        if data is None:
            if self.data is None:
                raise ValueError('data must be provided')
            else:
                data = self.data

        self.setParams(data=data, x=x, y=y, **kwargs)

    def getRecon(self): return self.x[0].reshape(self.shape) * self.dataScale


class LSQR(PDHG):

    def __init__(self, shape, op=None, order=1, spectDim=0, weight=0.1, pos=False, **kwargs):
        '''
        Minimise the energy:
        1/2|op(u)-data|^2 + weight*|u|_2^2
        where:
            shape: the shape of the output, u
            op=None: if provided, the forward operator to use
            weight=0.1: positive scalar value, almost definitely less than 1
        '''
        PDHG.__init__(self, None, None, None, **kwargs)
        self.shape = shape
        self.op = op
        self.pos = pos
        self.d = diff(shape, order=order, spectDim=spectDim)
        self.order = order
        self.weight = weight
        self.data = None

    def setParams(self, data, op=None, x=None, y=None, **kwargs):
        if op is not None:
            self.op = op
        elif self.op is None:
            self.op = scalar_mat([data.size] * 2, 1)

        self.weight = kwargs.get('weight', self.weight)

        normR = self.op.norm()
        dataScale = norm(data.reshape(-1))
        data = data.reshape(-1) / dataScale
        self.dataScale = dataScale

        f = L2(scale=self.weight)
        gstar = L2(scale=1, translation=data)

        # Choose starting point:
        if (x is None) and (self.x is None):
            x = getVec(self.op, rand=False)
        elif x is not None and x.dtype != object:
            x = x.reshape(-1) / self.dataScale
        if (y is None) and (self.y is None):
            y = getVec(self.op.T, rand=False)
        elif y is not None and y.dtype != object:
            y = y.reshape(-1)

        if 'steps' not in kwargs:
            kwargs['steps'] = 'backtrack'

        self.pos = kwargs.get('pos', self.pos)

        PDHG.setParams(self, A=self.op,
                       f=f, g=gstar.dual, x=x,
                       y=y, normA=normR, **kwargs)

    def start(self, data=None, x=None, y=None, **kwargs):
        if data is None:
            if self.data is None:
                raise ValueError('data must be provided')
            else:
                data = self.data

        self.setParams(data=data, x=x, y=y, **kwargs)

    def getRecon(self): return self.x.reshape(self.shape) * self.dataScale


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    x = .1 * random.randn(100, 50)
    x[50:] += 1

    problem = TV(x.shape, order=2, weight=.1, pos=True)
    recon = problem.run(data=x, maxiter=100, steps='adaptive',
                        callback=('gap', 'primal', 'dual', 'violation', 'step'))[0]

    plt.subplot(221)
    plt.imshow(x.T, vmin=0, vmax=1)
    plt.subplot(222)
    plt.imshow(recon.reshape(x.shape).T, vmin=0, vmax=1)
    plt.subplot(223)
    plt.plot(x.mean(1)); plt.ylim(0, 1)
    plt.subplot(224)
    plt.plot(recon.reshape(x.shape).mean(1)); plt.ylim(0, 1)
    plt.show()
