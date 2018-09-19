
import numpy as np
from scipy.signal import correlate2d


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def anisotropy2d(gr, min_grad=-1.0, power=0., num_bins=24, symmetric=True):
    sobel = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    gx = correlate2d(gr, sobel, mode='valid')
    gy = correlate2d(gr, np.transpose(sobel), mode='valid')
    theta, gm = cart2pol(gx, gy)
    theta = theta.flatten()
    gm = gm.flatten()
    if min_grad >= 0:
        theta = theta[gm > min_grad]
        gm = gm[gm > min_grad]

    if symmetric:
        theta1 = theta.copy()
        theta2 = theta1 + np.pi
        theta2[theta2 > np.pi] = theta2[theta2 > np.pi] - 2 * np.pi
        theta = np.concatenate((theta1, theta2))
        gm = np.concatenate((gm, gm))

    if power == 0:
        r, _ = np.histogram(theta, bins=num_bins, range=(-np.pi, np.pi))
    else:
        r = np.zeros((num_bins, ))
        theta = (theta + np.pi) / 2. / np.pi * num_bins
        theta = np.floor(theta).astype(int)
        theta[theta == num_bins] = 0
        for bin in range(num_bins):
            r[bin] = np.sum(np.power(gm[theta == bin], power))

    isotropy = np.min(r) / np.max(r)
    r1 = r / np.sum(r)
    std = np.std(r1)
    entropy = -np.sum(r1 * np.log(r1)) / np.log(num_bins)

    return r, isotropy, std, entropy

