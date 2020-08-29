'''
Created on 30 May 2019

@author: Rob Tovey
'''
from os.path import join
from numpy import array, random, empty, pi, linspace, eye, save, load, linalg
from utils import progressBar, toc, SILENT
from FFT_simulator import getSiBlock, crystal, getFTpoints, doPrecession, subSphere
from FourierTransform import getDFT, fromFreq

try:
    from numba import cuda; cuda.select_device(1)
except Exception:
    pass

spacing = 5.43


def params2crystal(stack, box, max_strain, strain_dim, rotate=True):
    c, strain = empty(stack, dtype='object'), empty((stack, 3, 3), dtype='float32')
    box = array(box) / 2

    copy = getSiBlock(rotate=rotate, justwidth=True)
    copy = (box[0] / copy[0] / spacing, box[1] / copy[1] / spacing,
            box[2] / (stack * copy[2] * spacing) * (1 if strain_dim < 3 else 2))
    copy = [int(round(.9 * 2 * i)) for i in copy]
    for i in range(stack):
        if strain_dim == 1:
            tmp = (random.rand(1) - .5) * eye(3)
        else:
            tmp = random.rand(3, 3) - .5
            if strain_dim == 2:
                tmp[:, 2], tmp[2, :] = 0, 0
                tmp /= 0.52  # normalised in average spectral norm
            else:
                tmp /= 0.72
        strain[i] = eye(3) + max_strain * tmp

        tmpC = crystal(spacing * getSiBlock(rotate=rotate), 14,
                       spacing * getSiBlock(rotate=rotate, justwidth=True))
        tmpC = tmpC.repeat(copy)
        tmpC = tmpC.scale(strain[i])

        if strain_dim == 3:
            thickness, middle = .9 * box[2] / stack, tmpC.box[2, 1] / 2
            loc = array([atom for atom in tmpC.loc if abs(atom[2] - middle - spacing / 2) < thickness])
            loc[:, 2] -= loc[:, 2].min()

            tmpC = crystal(loc, 14, loc.max(0))

        tmpC = tmpC.translate([ -(tmpC.box[0][0] + tmpC.box[0][1]) / 2,
                                -(tmpC.box[1][0] + tmpC.box[1][1]) / 2,
                                -.9 * box[2] if i == 0 else c[i - 1].box[2, 1] - tmpC.box[2, 0]])
        c.itemset(i, tmpC)

    return c.sum(), strain


n, dX, rX = None, [.4, .4, 2], [180, 180, 240]
probe, rad = 7, 320  # probe width(A), Ewald radius(100keV=170, 300keV=320)
alpha, nTheta = 2, 32  # precession angle/discretisation
file_prefix = join('sim', 'test')
# Geometry for computing tilt series
x, y = getFTpoints(3, dX=dX, rX=rX)
ft3, ift3 = getDFT(x, y)
n = [X.size for X in x]
filename = file_prefix + '_' + str(n[0]) + '_' + str(alpha).replace('.', 'p')

if __name__ == '__main__':
    from matplotlib.pyplot import *

#     # Initial plotting of crystals
#     C, strain = params2crystal(3, [20] * 3, 0.1, 1, True)
#     vol = C(getFTpoints(3, dX=.2, rX=[40] * 3)[0])
#
#     subplot(131); imshow(abs(vol).max(0))
#     subplot(132); imshow(abs(vol).max(1))
#     subplot(133); imshow(abs(vol).max(2))
#     show()
#     exit()

    dimList, stackList, duplicates, max_strain = [1, 2, 3], [1, 3, 10], 30, .01
    rotate = False
    dimList = [0, 1, 2, 3]

    if dimList == 0 or 0 in dimList:
        tmp = params2crystal(1, rX[:2] + [rX[2] - 40], 0, 1, rotate)
        tmp = ft3(tmp[0](x).astype('float32'))
        precessed, _, flat = doPrecession(tmp, probe, x, y, rad, alpha, nTheta, keeppath=False)
        save(filename + '_0' + ('_flipped' if rotate else ''), {'precessed':precessed, 'flat':flat})
        print('\n\nNo strain complete\n')

    if dimList == 0:
        exit()
    else:
        dimList = [i for i in dimList if i > 0]

    count = [0, len(stackList) * duplicates]
    with SILENT() as c:
        for i, I in enumerate(dimList):
            print('Starting alpha=%d, dimension=%d %s' % (alpha, I, 'flipped' if rotate else ''))
            random.seed(0)
            count[0], sList, tic = 0, [], toc()
            precessed = empty((1, len(stackList), duplicates, n[0], n[1]), dtype='float32')
            flat = empty(precessed.shape, dtype='float32')

            for j, J in enumerate(stackList):
                for k in range(duplicates):
                    tmp = params2crystal(J,
                                         # extends beyond in [x,y] but zeropadded in z
                                         [1.2 * rX[0], 1.2 * rX[1], rX[2] - 40],
                                         max_strain, I, rotate)
                    FT = ft3(tmp[0](x).astype('float32'))
#                     FT = tmp[0].FT(y).astype('complex64')
                    sList.append(tmp[1])
                    precessed[0, j, k], _, flat[0, j, k] = doPrecession(FT, probe, x, y, rad, alpha, nTheta, keeppath=False)
                    count[0] += 1
                    progressBar(count[0], count[1], tic, context=c)

            save(filename + '_' + str(I) + ('_flipped' if rotate else ''),
                {'precessed':precessed, 'flat':flat, 'strain':sList, 'stackList':stackList})

    print('\n\nComplete\n')
