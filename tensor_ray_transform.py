'''
Created on 29 Nov 2018

@author: Rob Tovey
'''
from numpy.linalg import pinv
from numpy import hstack, zeros, array, concatenate, matmul, eye, prod
from scipy.sparse.linalg import LinearOperator
from astra import create_proj_geom, create_vol_geom, create_projector, OpTomo
import atexit
import signal

##################################################
# Astra cleanup


def del_astra(*_, **__):
    try:
        import astra
        astra.data2d.clear()
        astra.data3d.clear()
        astra.projector.clear()
        astra.algorithm.clear()
        astra.matrix.clear()
        astra.functions.clear()
    except Exception:
        pass


atexit.register(del_astra)
signal.signal(signal.SIGTERM, del_astra)
signal.signal(signal.SIGINT, del_astra)
##################################################


def getGeneralProjGeometry(det, beam, e1, e2):
    # beam direction = beam[i]
    # first axis detector is e1[i] (number of pixels = det[0])
    # second axis detector is e2[i] (number of pixels = det[1])
    V = hstack((beam, zeros((len(beam), 3)), e2, e1))
    return create_proj_geom('parallel3d_vec', *det, V)


def getGeneralVolGeometry(sz):
    return create_vol_geom(sz[1], sz[2], sz[0])


def getGeneralProjector(sz, det, beam, e1, e2):
    '''
    sz = (sz_x,sz_y,sz_z)
        number of pixels in reconstruction
    det = (det_x,det_y)
        number of pixels on detector
    beam = array(shape=(n,3), dtype=float)
        beam[i] = (x,y,z) is a unit vector representing the direction
        of the beam relative to the reconstruction
    e1/e2 = array(shape=(n,3), dtype=float)
        e1[i] = (x,y,z) is a vector orthogonal to beam[i] indicating the
        physical direction of the first axis on the detector relative to
        the reconstruction. e2[i] is the equivalent vector with respect
        to the second detector axis.

    returns X-ray transform for scalar valued functions
    '''
    vol = getGeneralVolGeometry(sz)
    proj = getGeneralProjGeometry(det, beam, e1, e2)
    ID = create_projector('cuda3d', proj, vol)
    return OpTomo(ID)


class vectorProjector(LinearOperator):
    '''
    Operator which computes the component-wise X-ray transform of a tensor
    field.

    vectorProjector(proj, dom, ran, vsize=None)

    dom = tuple of ints
        number of pixels in reconstruction plus tensor dimensions
    ran = tuple of ints
        number of pixels in tomogram plus tensor dimensions
    vsize = int (optional)
        total number of tensor components
    proj = getGeneralProjector(...)-like object (scalar X-ray transform)
        Let <dim> be the dimension of the reconstruction (either 2 or 3).
        proj maps arrays of shape <dom[:dim]> to tomograms of shape
        <ran[:dim]>.
    '''

    def __init__(self, proj, dom, ran, vsize=None):
        dim = 3 if proj.appendString[:1] == '3' else 2
        if vsize is None:
            vsize = max(prod(dom[dim:]), prod(ran[dim:]))
        LinearOperator.__init__(self, 'float32', (vsize * prod(ran[:dim]), vsize * prod(dom[:dim])))
        self.proj = proj
        self.dim = dim
        self.dom, self.ran = dom, ran
        self._transpose = self._adjoint

    def FP(self, u):
        u = u.reshape(*self.dom[:self.dim], -1)
        v = [self.proj.FP(u[..., i])[..., None] for i in range(u.shape[-1])]
        return concatenate(v, axis=-1).reshape(self.ran)

    def BP(self, u):
        u = u.reshape(*self.ran[:self.dim], -1)
        v = [self.proj.BP(u[..., i])[..., None] for i in range(u.shape[-1])]
        return concatenate(v, axis=-1).reshape(self.dom)

    def norm(self):
        # known |R|_2^2 <= min(|R^TR|_1, |R^TR|_inf)
        # |R^TR|_1 = max column sum = max(RR^T * 1)
        # |R^TR|_inf = max row sum = max(R^TR * 1)
        R = self.proj
        normR = [R * (zeros(R.shape[1]) + 1), R.T * (zeros(R.shape[0]) + 1)]
        normR += [R.T * normR[0], R * normR[1]]
        normR = [n.max() ** .5 for n in normR]
        normR = min(normR[0] * normR[1], *normR[2:])

        # Full projector is block diagonal [R,R,...] so same norm
        return normR

    _matvec, _rmatvec = FP, BP


class tensorProjector(LinearOperator):
    '''
    3D Transverse ray transform map.

    tensorProjector(proj, dom, ran, beam)

    dom = tuple of ints
        number of pixels in reconstruction plus tensor dimensions
    ran = tuple of ints
        number of pixels in tomogram plus tensor dimensions
    beam = array(shape=(n,3))
        beam direction corresponding to tomogram
    proj = getGeneralProjector(...)-like object (scalar X-ray transform)
        proj maps arrays of shape <dom[:3]> to tomograms of shape
        <ran[:3]>.
    '''

    def __init__(self, proj, dom, ran, beam):
        P = vectorProjector(proj, dom, ran)
        LinearOperator.__init__(self, 'float32', P.shape)
        self.proj = P
        self.beam = beam
        I = eye(3).reshape(1, 1, 1, 3, 3)
        T = beam.reshape(1, -1, 1, 1, 3) * beam.reshape(1, -1, 1, 3, 1)
        self.orthoProj = I - T
        self._transpose = self._adjoint

    def FP(self, u):
        v = self.proj.FP(u)
        return matmul(self.orthoProj, matmul(v, self.orthoProj))

    def BP(self, u):
        '''
        define <u,v> = \int tr(u(i)v(i))di

        \int tr((P(i)I[u](i)P(i))^T v(i))di
            = \int tr(I[u](i)^T P(i)v(i)P(i))di
            = \int tr(u(j)^T I^T[PvP](j))dj
        '''
        u = matmul(self.orthoProj, matmul(u.reshape(self.proj.ran), self.orthoProj))
        return self.proj.BP(u)

    def norm(self): return self.proj.norm()

    _matvec, _rmatvec = FP, BP


def getVecProj(sz, det, beam, e1, e2, size=(1,)):
    '''
    sz = (sz_x,sz_y,sz_z)
        number of pixels in reconstruction
    det = (det_x,det_y)
        number of pixels on detector
    beam = array(shape=(n,3), dtype=float)
        beam[i] = (x,y,z) is a unit vector representing the direction
        of the beam relative to the reconstruction
    e1/e2 = array(shape=(n,3), dtype=float)
        e1[i] = (x,y,z) is a vector orthogonal to beam[i] indicating the
        physical direction of the first axis on the detector relative to
        the reconstruction. e2[i] is the equivalent vector with respect
        to the second detector axis.
    size = int or tuple of ints
        Spectral dimension of tensor field.

    returns component-wise X-ray transform for tensor valued functions
    '''
    size = tuple(size) if hasattr(size, '__iter__') else (size,)
    proj = getGeneralProjector(sz, det, beam, e1, e2)
    dom = tuple(sz) + size
    ran = (det[0], len(beam), det[1]) + size
    return tensorProjector(proj, dom, ran, beam)


def lift2to3(points, e1, e2):
    '''
    out = lift2to3(points, e1, e2)

    Lifts 2D points to 3D.

    points = array(shape=(n,2), dtype=float)
    e1/e2 = array(shape=(n,3), dtype=float)
        e1[i] = (x,y,z) is a vector orthogonal to beam[i] indicating the
        physical direction of the first axis on the detector relative to
        the reconstruction. e2[i] is the equivalent vector with respect
        to the second detector axis.
    out = array(shape=(n,3))
        out[i] = e1[i]*points[i,0] + e2*points[i,1]

    If n does not represent a single index then this is either broadcasted

    '''
    try:
        return points[..., 0] * e1 + points[..., 1] * e2
    except Exception:
        e1, e2 = e1.reshape(*points.shape[:-1], 3), e2.reshape(*points.shape[:-1], 3)
        return points[..., 0] * e1 + points[..., 1] * e2


def points2mats(points, values):
    '''
    A = points2mats(points, values)

    points = array(shape=[...,n,3])
    values = array(shape=[...,n,3])
    A = array(shape=[...,3,3])

    A is chosen such that
    A[...]points[...,i,:] \approx values[...,i,:]
    for each i. The approximation is chosen in a least-squares sense, i.e.
    A[...] minimises
        sum_i |A[...]points[...,i,:] - values[...,i,:]|^2

    '''
    '''
    This can be computed with the pseudo-inverse:
    -> AP^T = V -> A = VP^{\dagger T}
    '''
    V, P = array(values), array(points)
    axis = [i for i in range(values.ndim)]
    axis[-1], axis[-2] = axis[-2], axis[-1]
    return matmul(V.transpose(axis), pinv(P.transpose(axis)))


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from numpy import mgrid, random
    n = 1
    # sz = 'real' dimension of volume in pixels
    # det = number of pixels in detector
    sz, det = (50, 70, 90), (n * 100, n * 120)
    e = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    beam = array([e[0], e[1], e[2]])
    e1 = array([e[1], e[0], e[0]]) / n
    e2 = array([e[2], e[2], e[1]]) / n
    sz2 = det[0], len(beam), det[1]

    TEST = 2

    if TEST == 1:  # Check astra has same x/y/z convention
        x, dx = mgrid[:sz[0], :sz[1], :sz[2]], 2 / (min(sz) - 1)
        x = [(xx - xx.mean()) * dx for xx in x]
        y = mgrid[:det[0], :det[1]]
        y = [(yy - yy.mean()) / n * dx for yy in y]

        u = (x[0] < -.5) * (x[1] < .25) * (x[1] > -.5) * (x[2] > .5)

        R = getGeneralProjector(u.shape, det, beam, e1, e2)

        Ru = R * u.reshape(-1)
        for i in range(3):
            plt.subplot(3, 2, 2 * i + 1)
            extent = [[x[j].min(), x[j].max()] for j in range(3) if j != i]
            plt.imshow(u.sum(i), origin='lower', aspect='equal',
                       extent=extent[1] + extent[0])
            plt.title('Manual sum dimension ' + str(i))

            extent = [[yy.min(), yy.max()] for yy in y]
            plt.axis(extent[1] + extent[0])
            plt.subplot(3, 2, 2 * i + 2)
            plt.imshow(Ru[:, i], origin='lower', aspect='equal',
                       extent=extent[1] + extent[0])
            plt.title('Astra sum')
        plt.show()

    elif TEST == 2:  # Check adjoint of tensor map is correct
        size = (3, 3)
        proj = getVecProj(sz, det, beam, e1, e2, size)
        err = zeros(100)
        for i in range(100):
            u = 2 * random.rand(*sz, *size) - 1
            v = 2 * random.rand(*sz2, *size) - 1
            err[i] = (proj.FP(u) * v).sum() - (u * proj.BP(v)).sum()

        err /= ((u ** 2).sum() * (v ** 2).sum()) ** .5
        print('maximum error: %e\nmean error: %e\nstd: %e' % (abs(err).max(), err.mean(), err.std()))
