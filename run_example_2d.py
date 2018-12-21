
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray

from anisotropy2d import anisotropy2d


def example2d():
    test_image1 = rgb2gray(io.imread('lena256.jpg')).astype(float)

    sz = 256
    xx = np.linspace(0, 1, sz)
    const_grad, _ = np.meshgrid(xx, xx * 0)
    test_image2 = np.random.rand(sz, sz)

    tests = [['lena', 'Diagram (default)', test_image1, -1, 0],
             ['lena', 'Diagram (min_grad=0.3)', test_image1, 0.3, 0],
             ['lena', 'Diagram (power=1)', test_image1, -1, 1],
             ['white noise', 'Diagram', test_image2, -1, 0]]

    for test in tests:
        caption1 = test[0]
        caption2 = test[1]
        im = test[2]
        min_grad = test[3]
        power = test[4]

        rho, isotropy, std, entropy = anisotropy2d(im, min_grad=min_grad, power=power)

        step = 2. * np.pi / rho.shape[0]
        angles = np.arange(step / 2, 2 * np.pi + step, step)
        rho = np.append(rho, rho[0])

        ax = plt.subplot(121)
        ax.imshow(im, cmap='gray')
        ax.set_title(caption1, va='bottom')

        ax = plt.subplot(122, projection='polar')
        ax.plot(angles, rho)
        ax.set_title('%s\nisotropy=%.03f\nSTD=%.03f\nentropy=%.03f' % (caption2, isotropy, std, entropy), va='bottom')
        ax.grid(True)
        plt.show()


if __name__ == '__main__':
    example2d()
