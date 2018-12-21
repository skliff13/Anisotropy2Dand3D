
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from anisotropy3d import anisotropy3d, build_surface


def example3d():
    sz = 128
    xx = np.linspace(0, 1, sz)
    const_grad, _, _ = np.meshgrid(xx, xx * 0, xx * 0)
    im1 = const_grad * 0.9 + np.random.rand(sz, sz, sz) * 0.1
    im2 = np.random.rand(sz, sz, sz)

    tests = [['3D gradient with little noise\n(middle slice)', 'Diagram', im1, -1, 0],
             ['3D gradient with little noise\n(middle slice)', 'Diagram (min_grad=0.3)', im1, 0.3, 0],
             ['3D gradient with little noise\n(middle slice)', 'Diagram (power=1)', im1, -1, 1],
             ['3D white noise\n(middle slice)', 'Diagram', im2, -1, 0]]

    for test in tests:
        caption1 = test[0]
        caption2 = test[1]
        im = test[2]
        min_grad = test[3]
        power = test[4]

        rho, isotropy, std, entropy = anisotropy3d(im, min_grad=min_grad, power=power)

        xx, yy, zz = build_surface(rho)

        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow(im[:, :, sz // 2], cmap='gray')
        ax.set_title(caption1, va='bottom')

        ax = fig.add_subplot(122, projection='3d')
        ax.plot_surface(
            xx, yy, zz, rstride=1, cstride=1, cmap=plt.get_cmap('jet'),
            linewidth=0, antialiased=False, alpha=0.5)
        ax.set_title('%s\nisotropy=%.03f\nSTD=%.03f\nentropy=%.03f' % (caption2, isotropy, std, entropy), va='bottom')
        ax.set_aspect('equal')
        ax.figure.set_size_inches(10, 5)

        plt.show()

if __name__ == '__main__':
    example3d()

