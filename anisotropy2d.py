# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import correlate2d


def anisotropy2d(gray_image2d, min_grad=-1.0, power=0., num_bins=24, symmetric=True):
    """Calculates circular histogram and anisotropy features for a 2D Gray-Level image.

    Args:
        gray_image2d (ndarray): 2D numpy array representing the image.
        min_grad (float): Threshold used to ignore small gradient values.
            Negative means no threshold. Defaults to -1.0
        power (float): Circular histogram summarizes (gradient_value)**power.
            Power=0 can be used to treat small and large gradient values equally.
            Power=1 can be used to calculate the sum of gradient values.
            Greater values can be used to suppress small gradient values.
            Defaults to 0.
        num_bins (int): Number of bins of circular histogram. Defaults to 24.
        symmetric (bool): If True, opposite gradient directions are considered as the same.
            Defaults to True.

    Returns:
        ndarray: Rho values for plotting the circular histogram. Example below.
            step = 2. * numpy.pi / r.shape[0]
            angles = numpy.arange(step / 2, 2 * np.pi + step, step)
            r = numpy.append(r, r[0])
            ax = pyplot.subplot(111, projection='polar')
            ax.plot(angles, r)
        float: Isotropy coefficient between 0 and 1. Greater values correspond to more isotropic cases.
            Calculated as max(Rho)/min(Rho).
        float: Standard deviation of Rho values.
        float: Entropy of Rho values.

    """

    grad_x, grad_y = __calculate_gradients(gray_image2d)

    theta, grad_magnitude = __cart2pol(grad_x, grad_y)
    theta = theta.flatten()
    grad_magnitude = grad_magnitude.flatten()

    if min_grad >= 0:
        grad_magnitude, theta = __filter_gradients(grad_magnitude, min_grad, theta)

    if symmetric:
        grad_magnitude, theta = __make_symmetric(grad_magnitude, theta)

    if power == 0:
        rho, _ = np.histogram(theta, bins=num_bins, range=(-np.pi, np.pi))
    else:
        rho = __extended_histogram(grad_magnitude, num_bins, power, theta)

    isotropy = np.min(rho) / np.max(rho)
    rho1 = rho / np.sum(rho)
    std = np.std(rho1)
    entropy = -np.sum(rho1 * np.log(rho1)) / np.log(num_bins)

    return rho, isotropy, std, entropy


def __calculate_gradients(gray_image2d):
    sobel_operator = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gray_image2d = gray_image2d.astype(float)
    grad_x = correlate2d(gray_image2d, sobel_operator, mode='valid')
    grad_y = correlate2d(gray_image2d, np.transpose(sobel_operator), mode='valid')
    return grad_x, grad_y


def __filter_gradients(grad_magnitude, min_grad, theta):
    theta = theta[grad_magnitude > min_grad]
    grad_magnitude = grad_magnitude[grad_magnitude > min_grad]
    return grad_magnitude, theta


def __extended_histogram(grad_magnitude, num_bins, power, theta):
    rho = np.zeros((num_bins,))
    theta = (theta + np.pi) / 2. / np.pi * num_bins
    theta = np.floor(theta).astype(int)
    theta[theta == num_bins] = 0
    for bin in range(num_bins):
        rho[bin] = np.sum(np.power(grad_magnitude[theta == bin], power))
    return rho


def __cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def __make_symmetric(grad_magnitude, theta):
    theta1 = theta.copy()
    theta2 = theta1 + np.pi
    theta2[theta2 > np.pi] = theta2[theta2 > np.pi] - 2 * np.pi
    theta = np.concatenate((theta1, theta2))
    grad_magnitude = np.concatenate((grad_magnitude, grad_magnitude))
    return grad_magnitude, theta
