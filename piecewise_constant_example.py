'''
Created on 27 Nov 2018

@author: Rob Tovey
'''

from os.path import join
from numpy import array, random, empty, pi, linspace, eye
from utils import loadmat, savemat
from FFT_simulator import getSiBlock, crystal, getOrientations, getFTpoints

# Number of blocks in each direction and total size (nm) in each direction
__shapes = (1, 1, 3), (194.4,) * 3
__spacing = 5.43

__c = empty(__shapes[0], dtype='object')
__strain = empty(__shapes[0] + (3, 3))
__max_strain = 0.02  # Average strain in spectral norm
__iso_strain = False

# Compute strain
random.seed(1)
for i in range(__shapes[0][0]):
    for j in range(__shapes[0][1]):
        for k in range(__shapes[0][2]):
            if __iso_strain:
                tmp = 1 + (2 * random.rand(1) - 1) * __max_strain
                tmp = tmp * eye(3)
            else:
                tmp = random.rand(3, 3) - .5
                tmp *= __max_strain / 0.72
                tmp += eye(3)
            __strain[i, j, k] = tmp.copy()

# Compute Crystals
# tmp = [1 + int(round(__shapes[1][i] / __shapes[0][i] / __spacing
#         * (1 + __max_strain))) for i in range(3)]
tmp = [1 + int(round(__shapes[1][i] / __shapes[0][i] / __spacing))
       for i in range(3)]
for i in range(__shapes[0][0]):
    for j in range(__shapes[0][1]):
        for k in range(__shapes[0][2]):
            tmp1 = crystal(__spacing * getSiBlock(rotate=True), 14,
                           __spacing * getSiBlock(rotate=True, justwidth=True),
                           GPU=True)
            tmp1 = tmp1.repeat(tmp)
            tmp1 = tmp1.scale(__strain[i, j, k])
#             tmp1 = tmp1.translate(
#                 [(__box[i, j, k][l][0] + __box[i, j, k][l][1]) / 2
#                  -(tmp1.box[l][0] + tmp1.box[l][1]) / 2
#                  for l in range(3)]
#             )
            # Just for stack of slabs:
            if k > 0:
                tmp1 = tmp1.translate([-(tmp1.box[0][0] + tmp1.box[0][1]) / 2,
                                       -(tmp1.box[1][0] + tmp1.box[1][1]) / 2,
                                       __c[i, j, k - 1].box[2][1]
                                       -tmp1.box[2][0] + __spacing])
            else:
                tmp1 = tmp1.translate([-(tmp1.box[0][0] + tmp1.box[0][1]) / 2,
                                       -(tmp1.box[1][0] + tmp1.box[1][1]) / 2,
                                       0])
#             tmp1 = tmp1.cutoff(__box[i, j, k])
            __c[i, j, k] = tmp1

myCrystal = __c.sum()
myCrystal = myCrystal.translate(-(myCrystal.box[:, 0] + myCrystal.box[:, 1]) / 2)
pureCrystal = crystal(__spacing * getSiBlock(rotate=True), 14,
                      __spacing * getSiBlock(rotate=True, justwidth=True),
                      GPU=True)
pureCrystal = pureCrystal.repeat([int(round(__shapes[1][i] / pureCrystal.box[i, 1]))
                                  for i in range(3)])
pureCrystal = pureCrystal.translate(-pureCrystal.box[:, 1] / 2)

del i, j, k, tmp, tmp1

nX, rX, nY, rY = 50, __shapes[1], 512, 12  # discretisation
probe, rad = 30 / 2, None  # 50 * 2 * pi  # probe width, Ewald radius
alpha, nTheta = 1, 30  # precession angle/discretisation
file_prefix = join('sim', 'piecewise_constant')
# Geometry for computing tilt series
x = [linspace(-rX[i] / 2, rX[i] / 2, nX, endpoint=True) for i in range(3)]
_, y = getFTpoints(3, nY, rY=rY)
filename = [file_prefix + e + '_' + str(nY) for e in ('_pure', '', 'tomo_data', 'recon')]


def updateParams(nY):
    _, y = getFTpoints(3, nY, rY=rY)
    filename = [file_prefix + e + '_' + str(nY) for e in ('_pure', '')]
    globals().update({'y':y, 'filename':filename})


if __name__ == '__main__':
    from numpy import log10, sqrt, maximum, minimum, linalg
    from matplotlib import pyplot as plt
    from FFT_simulator import doScan, planes2basis
    from tensor_ray_transform import getVecProj, images2coms, points2mats
    from recon_lib import interpStrain, TV, LSQR

    precomputed = [False, False, False]

    # E(|x|) = sqrt(2/pi)*std for normal distributions
    # Noise averaged over 2 spots => /sqrt(2)
    # error=1e-3 or 1e-2
    synthetic, boundary, error = True, .8, 1e-2 / pi ** .5

    ##################################################
    # Define helper functions

    def plot_FT(thing, *args, **kwargs):
        if thing.ndim == 2:
            plt.imshow(thing, *args, aspect='equal',
                       extent=[y[0][0], y[0][-1], y[1][-1], y[1][0]], **kwargs)
        else:
            plt.plot(y[0], thing, *args, **kwargs)

    sqrt = lambda x: x ** .25
    r = sqrt(y[0].reshape(-1, 1) ** 2 + y[1].reshape(1, -1) ** 2)
    corrected = lambda x: x * maximum(1, 2 * r)

    ##################################################
    # Compute tilt angles

    if not precomputed[0]:
        cutoff = 2e-1, 1e-2
        _, y = getFTpoints(3, 400, rY=10)
        c = crystal(__spacing * getSiBlock(rotate=True), 14,
                    __spacing * getSiBlock(rotate=True, justwidth=True),
                    GPU=True)
        c = c.repeat(4)

        FT, debFT, points, plane = getOrientations(
            c, y, 14, cutoff, rad=.5, angle=alpha * 4)
        print('Saving orientations...', end='')
        savemat(file_prefix + '_plane', plane=plane, points=points)
        print(' complete.')

        plt.subplot(131)
        plot_FT(minimum(FT[..., abs(y[2]).argmin()] / points[1, -1], 1))
        plt.title('Exact FT')
        plt.subplot(132)
        plot_FT(debFT[..., abs(y[2]).argmin()])
        plt.title('Debiased FT')
        plt.subplot(133)
        p = points[abs(points[:, 2]) < 1e-10]
        plt.scatter(p[:, 1], p[:, 0], c=p[:, 3], marker='x')
        plt.axis([y[0][0], y[0][-1], y[1][-1], y[1][0]])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title('Extracted points')
        from mpl_toolkits.mplot3d import Axes3D
        plt.figure()
        plt.gcf().add_subplot(111, projection='3d')
        plt.gca().scatter(plane[:, 0], plane[:, 1], plane[:, 2], c='b')
        plt.gca().scatter(-plane[:, 0], -plane[:, 1], -plane[:, 2], c='r')
        for _ in range(0):
            for angle in range(0, 360):
                try:
                    plt.gca().view_init(30, angle)
                    plt.draw()
                    plt.pause(.001)
                except Exception:
                    break
        plt.show()

        plane, points = loadmat(file_prefix + '_plane', 'plane', 'points')

        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
            "font.size":25})
        plt.figure(figsize=(8, 8))
        p = list(plane.T)
        plt.plot(p[1] / (1 + p[0]), p[2] / (1 + p[0]), 'kx', label='Acquisition direction')
        plt.xlabel(r'$\frac{x}{1+z}$');plt.ylabel(r'$\frac{y}{1+z}$');
        tmp = plt.Circle((0, 0), 0.700, color='blue', fill=False)
        tmp.set_label('70 degree cuttoff'); plt.gca().add_artist(tmp)
        tmp = plt.Circle((0, 0), 1, color='red', linestyle=(0, (5, 5)), fill=False)
        tmp.set_label('90 degree cuttoff'); plt.gca().add_artist(tmp)
        plt.gca().axis('equal'); plt.tight_layout(); plt.xlim(-1, 1); plt.ylim(-1, 1)
        plt.savefig('stereographic_projection.png', dpi=300)
        plt.show()
        exit()

#         from FFT_simulator import debias_FT
#
#         y = [y[0], y[1], array([0])]
#         plt.figure()
#         for p in plane:
#             plt.clf()
#             ind = (p.reshape(1, 3) * points[1:, :3].reshape(-1, 3)).sum(-1)
#             ind = (abs(ind) > 1e-8).argmin()
#
#             tmp = abs(c.FT(y, planes2basis(p, points[1 + ind, :3]))) ** 2
#             tmp /= tmp.max()
#             plt.subplot(121)
#             plot_FT(minimum(tmp[..., 0], points[1, -1]))
#             plt.title('Exact')
#             plt.subplot(122)
#             tmp = abs(debias_FT(tmp, y, 14))
#             plot_FT(tmp[..., 0] * (tmp[..., 0] > cutoff[1]))
#             plt.title('Debiased')
#             plt.draw()
#             plt.pause(2)
#
#         exit()
    else:
        plane, points = loadmat(file_prefix + '_plane', 'plane', 'points')
        points = points[1:, :3]
        ind = (plane.reshape(-1, 1, 3) * points.reshape(1, -1, 3)).sum(-1)
        ind = (abs(ind) > 1e-8).argmin(1)

    ##################################################
    # Compute or load tilt series

    E = array(planes2basis(plane, points[ind]))
    gt = interpStrain(__strain, [int(boundary * X.size) for X in x], pad_sz=[X.size for X in x])
    R = getVecProj(gt.shape[:3], [X.size for X in x[:2]], E[:, 2], E[:, 0],
                   E[:, 1], size=(3, 3))

    if not precomputed[1]:
        if synthetic:
            random.seed(100)
            tomo_data = R * gt.reshape(-1)
            tomo_data.shape = x[0].size, len(plane), x[1].size, 3, 3
            for i in range(2):
                for j in range(2):
                    tomo_data += error * (random.randn(*tomo_data.shape[:3], 1, 1)
                                          * E[:, i].reshape(1, -1, 1, 3, 1)
                                          * E[:, j].reshape(1, -1, 1, 1, 3))
            savemat(filename[2], tomo_data=tomo_data)
            print('Simulation saved')
        else:
            for i in range(len(plane)):
                doScan(pureCrystal, probe, x, y, plane[i], rad, alpha, nTheta,
                       filename=join(filename[0], 'plane_%02d' % i),
                       points=points[ind[i]])
                doScan(myCrystal, probe, x, y, plane[i], rad, alpha, nTheta,
                       filename=join(filename[1], 'plane_%02d' % i),
                       points=points[ind[i]])
                print('%.2f' % 100 * i / len(plane))
    else:
        tomo_data, = loadmat(filename[2], 'tomo_data')

    ##################################################
    # do reconstruction

    if not precomputed[2]:
        # iso: noise=1e-3 -> weight=1e-5/2, noise=1e-2 -> weight=1e-4/2
        problem = TV(gt.shape, op=R, order=1, spectDim=2, weight=(1e-4 / 2 if error > 1e-3 else 1e-5 / 2))

        recon = problem.run(data=tomo_data, maxiter=500, steps='backtrack', x=gt,
                            callback=('gap', 'primal', 'dual', 'violation', 'step'))[0]
        print('Recon data error: %e' % (((R.FP(recon) - tomo_data) ** 2).sum() / 2))
        print('GT data error: %e' % (((R.FP(gt) - tomo_data) ** 2).sum() / 2))
        print('Recon TV score: %e' % (((problem.d * recon.reshape(-1)).reshape(-1, 9 * 3) ** 2).sum(-1) ** .5).sum())
        print('GT TV score: %e' % (((problem.d * gt.reshape(-1)).reshape(-1, 9 * 3) ** 2).sum(-1) ** .5).sum())

        savemat(filename[3] + ('_noisy' if error > 1e-3 else ''), recon=recon)
    else:
        recon, = loadmat(filename[3] + ('_noisy' if error > 1e-3 else ''), 'recon')

    x = [linspace(-rX[i] / 2, rX[i] / 2, recon.shape[i], endpoint=True) for i in range(3)]
    cax = 1 - __max_strain, 1 + __max_strain

    def tripPlot(title, arr, s, f, vmin=None, vmax=None):
        plt.figure(title)
        for i in range(3):
            plt.subplot(1, 3, i + 1);
            plt.imshow(f(arr, i).T, extent=[x[0].min(), x[0].max(), x[1].min(), x[1].max()],
                       vmin=vmin, vmax=vmax)
            plt.title(s + ' down %s-axis' % ('xyz'[i]))
            plt.xlabel('x');plt.ylabel('z');
            plt.gca().yaxis.set_label_coords(-0.1, .5)
        plt.tight_layout()

    plt.rcParams.update({'font.size': 22, 'figure.figsize':[20, 10]})

    tripPlot('reconstruction', abs(recon).max((-2, -1)), 'mean',
             lambda a, i: a.mean(i) / boundary, *cax)
    plt.savefig('../Publications/Theory/Images/piecewise_constant' + ('_noisy' if error > 1e-3 else '') + '_recon.png', dpi=300)
    tripPlot('gt', abs(gt).max((-2, -1)), 'mean',
             lambda a, i: a.mean(i) / boundary, *cax)
    plt.savefig('../Publications/Theory/Images/piecewise_constant_gt.png', dpi=300)
    tripPlot('reconstruction error', abs(recon - gt).max((-2, -1)), 'maximum',
             lambda a, i: a.max(i) / boundary, -__max_strain, +__max_strain)

    from numpy import percentile
    from matplotlib.ticker import FormatStrFormatter
    sym = [.5 * ((recon - gt)[..., i, j] + (recon - gt)[..., j, i]) for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))]
    plt.rcParams.update({'font.size': 25, 'figure.figsize':[20, 10]})

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.sans-serif": ["cm10"],
        "font.size":25})
    # percentiles plots
    f, ax = plt.subplots(1, 3, num='Cross sections')
    ax[0].plot(x[2], abs(gt).max((-2, -1)).mean((0, 1)) / boundary ** 2, 'r-', label='ground truth')
    ax[0].plot(x[2], abs(recon).max((-2, -1)).mean((0, 1)) / boundary ** 2, 'b--', label='reconstruction')
    ax[0].set_ylim(cax); ax[0].legend(loc='upper right'); ax[0].set_title('Strain magnitude', y=1.02)
    ax[1].plot(x[2], percentile(abs(recon - gt), 99, axis=(0, 1)).reshape(-1, 9))
    ax[1].set_title('Gradient deformation error', y=1.02)
    ax[1].set_ylim((0, None))
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f' if error > 1e-3 else '%.4f'))
    ax[1].legend([r'$\vec{R}_{%d,%d}$' % (i + 1, j + 1) for i in range(3) for j in range(3)],
                 loc='upper center', ncol=3, handlelength=.5, borderaxespad=.2,
                 columnspacing=.5, labelspacing=.2, handletextpad=.2)
    for s in sym:
        ax[2].plot(x[2], percentile(abs(s), 99, axis=(0, 1)).reshape(-1))
    ax[2].set_title('Symmetric strain error', y=1.02)
    ax[2].set_ylim(ax[1].get_ylim())
    ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.3f' if error > 1e-3 else '%.4f'))
    ax[2].legend([r'$\vec{\varepsilon}_{%d,%d}$' % (i + 1, j + 1) for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))],
                 loc='upper center', ncol=2, handlelength=.5, borderaxespad=.2,
                 columnspacing=.5, labelspacing=.2, handletextpad=.2)
    for a in ax:
        a.set_xlim((-100, 100))
        a.set_xticks([-50, 0, 50]);
        a.set_xlabel('z')
        a.xaxis.set_label_coords(.5, -.05)
    plt.tight_layout()
    plt.savefig('../Publications/Theory/Images/piecewise_constant_percentile' + ('_noisy' if error > 1e-3 else '') + '_slice.png', dpi=300)

    # comparison plots
    f, ax = plt.subplots(1, 3, num='Cross sections comparisons')
    ax[2].set_title('Maximum', y=1.0)
    ax[2].plot(x[2], abs(recon - gt).max((0, 1)).reshape(-1, 9))
    ax[1].set_title('99th percentile', y=1.0)
    ax[1].plot(x[2], percentile(abs(recon - gt), 99, axis=(0, 1)).reshape(-1, 9))
    ax[0].set_title('Middle slice', y=1.0)
    ax[0].plot(x[2], abs(recon - gt)[gt.shape[0] // 2, gt.shape[1] // 2].reshape(-1, 9))
    for a in ax:
        a.set_xlim((-100, 100))
        a.set_xticks([-50, 0, 50]);
        a.set_xlabel('z')
        a.xaxis.set_label_coords(.5, -.05)
        a.set_ylim(0, ax[2].get_ylim()[1])
        a.yaxis.set_major_formatter(FormatStrFormatter('%.3f' if error > 1e-3 else '%.4f'))
    ax[1].legend([r'$\vec{R}_{%d,%d}$' % (i + 1, j + 1) for i in range(3) for j in range(3)],
                 loc='upper center', ncol=3, handlelength=.5, borderaxespad=.2,
                 columnspacing=.5, labelspacing=.2, handletextpad=.2)
    plt.tight_layout()
    plt.savefig('../Publications/Theory/Images/piecewise_constant_max' + ('_noisy' if error > 1e-3 else '') + '_slice.png', dpi=300)
    plt.show()
