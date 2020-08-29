'''
Created on 9 Nov 2019

@author: Rob Tovey
'''
from os.path import join
from numpy import array, empty, pi, linspace, cross, arctan2, cos, log, sin, logical_or, matmul
from utils import toMesh, savemat
from FFT_simulator import getSiBlock, crystal, getFTpoints, doScan
from FourierTransform import getDFT, fromFreq

try:
    from numba import cuda; cuda.select_device(1)
except Exception:
    pass

spacing = 5.43


def to_basis(b, u):
    b, u = array(b), array(u)

    normalise = lambda x: x / (x ** 2).sum() ** .5
    z = u; x = cross(u, b); y = cross(z, x)
    x_hat, y_hat, z_hat = (normalise(t) for t in (x, y, z))
    for vec in (x_hat, y_hat, z_hat, b, u):
        vec.shape = (1, 3)

    return b, u, x_hat, y_hat, z_hat


def strain_map(loc, b, u, tensors=False):
    loc = loc.reshape(-1, 3)
    b, u, x_hat, y_hat, z_hat = to_basis(b, u)
    v = 0.22

    X, Y, Z = (loc * x_hat).sum(-1), (loc * y_hat).sum(-1), (loc * z_hat).sum(-1)

    phi = arctan2(Y, X)  # range [-pi,pi]
    phi[abs(Y) < 1e-5] = 0
    r = (Y ** 2 + X ** 2) ** .5
    ind = (r < 1e-16)
    r[ind] = 1e-16

    if not tensors:
        for vec in (phi, r):
            vec.shape = loc.shape[0], 1
        shift = (b * (phi + sin(2 * phi) / (4 * (1 - v)))
                 +cross(u, b) / (2 * (1 - v)) * ((1 - 2 * v) * log(r) + cos(2 * phi) / 2))
#         shift[ind] = 0
        return loc + shift / (2 * pi)
    else:
        T = empty((loc.shape[0], 3, 3))
        T.fill(0)
        c = cross(u, b) / (2 * (1 - v))
        for i in range(3):
            T[:, i, 0] = (
                b.item(i) * (1 + cos(2 * phi) / (2 * (1 - v))) * (-Y)
                +c.item(i) * ((1 - 2 * v) * X - sin(2 * phi) * (-Y))
                )
            T[:, i, 1] = (
                b.item(i) * (1 + cos(2 * phi) / (2 * (1 - v))) * (X)
                +c.item(i) * ((1 - 2 * v) * Y - sin(2 * phi) * (X))
                )
            T[:, i, 2] = 0
        T /= 2 * pi * r.reshape(-1, 1, 1) ** 2
        T[ind] = 0

        T = matmul(T, array([x_hat, y_hat, z_hat]).reshape(3, 3))
        for i in range(3):
            T[:, i, i] += 1

        return T


def slice_atoms(loc, b, u):
    b, u, x_hat, y_hat, z_hat = to_basis(b, u)
    s = (b ** 2).sum() ** .5 / 2.5

    X, Y, Z = (loc * x_hat).sum(-1), (loc * y_hat).sum(-1), (loc * z_hat).sum(-1)
    ind = logical_or(abs(Y - s) >= (1 + 1e-5) * s, X > 1e-5)
#     ind = abs(Y + s) >= (1 + 1e-5) * s
#     ind = (1 - ind).astype(bool)
#     print(loc + array([.5, .5, 0]).reshape(1, -1) * spacing)
#     print((Y + s).round(1).reshape(-1, 1))
#     exit()
    X, Y, Z, loc = (t[ind] for t in (X, Y, Z, loc))
#     tmp = loc + array([.5, .5, 0]).reshape(1, -1) * spacing
#     tmp2 = (array((.5, 0, .5)) * spacing,
#             array((.5, .5, 0)) * spacing,
#             array((0, .5, .5)) * spacing)
#     for vec in tmp:
#         vec = (vec % spacing + spacing) % spacing
#         if any(abs(vec - x).max() < 1e-5 for x in tmp2):
#             print('error: ', vec)
#         elif any(abs(vec - .25 * spacing - x).max() < 1e-5 for x in tmp2):
#             print('error: ', vec)
#     print(loc + array([.5, .5, 0]).reshape(1, -1) * spacing)
#     print(loc)
#     exit()

    return loc


def test_tensors():
    b, u = [1, 1, 1], [2, -1, -1]
    x = [linspace(-1, 1, 1000)] * 3

    diff = lambda f, dx: (f[1:] - f[:-1]) / dx
    av = lambda f: (f[1:] + f[:-1]) / 2
    for i in range(3):
        X = [array([.3]) for _ in range(3)]
        X[i] = x[i]
        X = toMesh(X)

        f = strain_map(X.reshape(-1, 3), b, u, False).reshape(-1, 3)
        T = strain_map(X.reshape(-1, 3), b, u, True).reshape(-1, 3, 3)

        for j in range(3):
            df = diff(f[:, j], x[i][1] - x[i][0])
            t = av(T[:, j, i])
            ind = abs(df) < min(10, abs(df).max())  # ignore discontinuities
            if ind.sum() > 0:
                assert abs(df - t)[ind].max() <= 1e-4 * abs(t).max()

#         if i == 0:
#             from matplotlib import pyplot as plt
#             plt.subplot(131);plt.plot(diff(f[:, 1], x[i][1] - x[i][0]))
#             plt.subplot(132);plt.plot(av(T[:, 1, i]))
#             plt.subplot(133);plt.plot(abs(diff(f[:, 1], x[i][1] - x[i][0]) - av(T[:, 1, i])))
#             plt.show()
#             exit()

    print('test_tensors complete')


plotting, iso = False, False
if plotting:
    n, dX, rX, rY = None, [.1] * 3, [(40 if iso else 30), (80 if iso else 30), (20 if iso else 30)], None
else:
    n, dX, rX, rY = None, [.4, .4, None], [180, 180, 240], [None, None, 5]
probe, rad = 7, 320  # probe width(A), Ewald radius(100keV=170, 300keV=320)
alpha, nTheta = 2, 64  # precession angle/discretisation
X = [linspace(-1, 1, 11) * .5 * r for r in rX[:2]] + [array((0,))]  # scanning grid
file_prefix = join('sim', 'analytic')
# Geometry for computing tilt series
x, y = getFTpoints(3, dX=dX, rX=rX, rY=rY)
ft3, ift3 = getDFT(x, y)
n = [X.size for X in x]
filename = file_prefix + '_' + str(n[0]) + '_' + str(alpha)

if __name__ == '__main__':

    base_crystal = crystal(spacing * getSiBlock(rotate=False)[slice(1) if iso else slice(None)], 14,
                           spacing * getSiBlock(rotate=False, justwidth=True))
    base_crystal = base_crystal.repeat([int(1.2 * rX[0] / base_crystal.box[0, 1]), int(1.2 * rX[1] / base_crystal.box[1, 1]),
                                        int((rX[2] - (0 if plotting else 40)) / base_crystal.box[2, 1])])
    base_crystal = base_crystal.translate(-(base_crystal.box[:, 1] / 2 // spacing) * spacing)

    if iso:
        b, u = [spacing, 0, 0], [0, 0, 1]
        slice_crystal = crystal(slice_atoms(base_crystal.loc, [spacing, 0, 0],
                                           [0, 0, 1]), 14)
        strained_crystal = crystal(strain_map(slice_crystal.loc, b, u), 14)
    else:
        b, u = [spacing / 6 ** .5] * 3, [1, -1, 0]
        base_crystal = base_crystal.translate([-.5 * spacing, -.5 * spacing, 0])
        slice_crystal = crystal(slice_atoms(base_crystal.loc, b, u), 14)
        strained_crystal = crystal(strain_map(slice_crystal.loc,
                                              b, u), 14)

    if plotting:
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.pyplot import *
        d0 = list(slice_crystal.loc.T)
        d1 = list(strained_crystal.loc.T)

        ticks, lims = (-10, 0, 10), (-20, 15)
        fig = figure()
        ax = fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')
        ax[0].scatter(*d0);  ax[0].view_init(*((75, -90) if iso else (.1, 135)))
        ax[1].scatter(*d1);  ax[1].view_init(*((75, -90) if iso else (.1, 135)))
        for a in ax:
            a.set_xticks(ticks); a.set_yticks(ticks)
            a.set_xlim(lims); a.set_ylim(lims)
            a.set_xlabel('x'); a.set_ylabel('y')
        tight_layout()
        savefig('dislocation_phantom.png', dpi=300)
        show()
        exit()

    nz = 100
    strain = empty((X[0].size, X[1].size, nz, 3, 3))
    for i in range(len(X[0])):
        for j in range(len(X[1])):
            tmp = 3 * spacing  # average over 6x6 units sized square
            mesh = toMesh([linspace(X[0][i] - tmp, X[0][i] + tmp, 10),
                           linspace(X[1][j] - tmp, X[1][j] + tmp, 10),
                           linspace(strained_crystal.box[2, 0], strained_crystal.box[2, 1], nz)])
            strain[i, j] = strain_map(mesh, b, u, tensors=True).reshape(-1, nz, 3, 3).mean(0)

    savemat(filename + '_strain', strain=strain)

#     doScan(base_crystal, probe, [array([0])] * 3, y, array([0, 0, 1]), rad, alpha, nTheta, filename + '_gt')
#     doScan(strained_crystal, probe, X, y, array([0, 0, 1]), rad, alpha, nTheta, filename)

    print('\n\nSimulation Complete')
