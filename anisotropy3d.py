# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import correlate


def anisotropy3d(gray_image3d, min_grad=-1.0, power=0., theta_bins=24, phi_bins=12, symmetric=True):
    """Calculates spherical histogram and anisotropy features for a 3D Gray-Level image.

    Args:
        gray_image3d (ndarray): 3D numpy array representing the image.
        min_grad (float): Threshold used to ignore small gradient values.
            Negative means no threshold. Defaults to -1.0
        power (float): Circular histogram summarizes (gradient_value)**power.
            Power=0 can be used to treat small and large gradient values equally.
            Power=1 can be used to calculate the sum of gradient values.
            Greater values can be used to suppress small gradient values.
            Defaults to 0.
        theta_bins (int): Number of bins for Theta coordinate of spherical histogram. Defaults to 24.
        phi_bins (int): Number of bins for Phi coordinate of spherical histogram. Defaults to 12.
        symmetric (bool): If True, opposite gradient directions are considered as the same.
            Defaults to True.

    Returns:
        ndarray: 2D array of Rho values for plotting the circular histogram. Input for 'build_surface'.
        float: Isotropy coefficient between 0 and 1. Greater values correspond to more isotropic cases.
            Calculated as max(Rho)/min(Rho).
        float: Standard deviation of Rho values.
        float: Entropy of Rho values.

    """

    grad_x, grad_y, grad_z = __calculate_gradients(gray_image3d)

    gtheta, gphi, grad_mag = __cart2sph(grad_x, grad_y, grad_z)
    grad_mag = grad_mag.flatten()
    gtheta = gtheta.flatten()
    gphi = gphi.flatten()

    if min_grad >= 0:
        gphi, grad_mag, gtheta = __filter_gradients(gphi, grad_mag, gtheta, min_grad)

    if symmetric:
        grad_mag, gphi, gtheta = __make_symmetric(grad_mag, gphi, gtheta)

    btheta = np.floor((gtheta + np.pi) / 2 / np.pi * theta_bins)
    btheta[btheta >= theta_bins] = theta_bins - 1
    gsinphi = np.sin(gphi)
    bphi = np.floor((gsinphi + 1) / 2 * phi_bins)
    bphi[bphi >= phi_bins] = phi_bins - 1

    idx = btheta + theta_bins * bphi
    n = theta_bins * phi_bins
    if power == 0:
        rho, _ = np.histogram(idx, bins=n, range=(-0.5, n - 0.5))
    else:
        rho = np.zeros((n,))
        for bin in range(n):
            rho[bin] = np.sum(np.power(grad_mag[idx == bin], power))

    rho = np.reshape(rho, (phi_bins, theta_bins))

    isotropy = np.min(rho) / np.max(rho)
    rho1 = rho / np.sum(rho)
    std = np.std(rho1)
    entropy = -np.sum(rho1 * np.log(rho1)) / np.log(n)

    return rho, isotropy, std, entropy


def __calculate_gradients(gray_image3d):
    plane = np.asarray([[3, 2, 3], [2, 1, 2], [3, 2, 3]]) ** (-0.5)
    sobel_operator = np.zeros((3, 3, 3))
    sobel_operator[:, :, 0] = -plane
    sobel_operator[:, :, 2] = plane
    grad_z = correlate(gray_image3d, sobel_operator, mode='valid')
    grad_x = correlate(gray_image3d, np.swapaxes(sobel_operator, 2, 1), mode='valid')
    grad_y = correlate(gray_image3d, np.swapaxes(sobel_operator, 2, 0), mode='valid')
    return grad_x, grad_y, grad_z


def __filter_gradients(gphi, grad_mag, gtheta, min_grad):
    gtheta = gtheta[grad_mag > min_grad]
    gphi = gphi[grad_mag > min_grad]
    grad_mag = grad_mag[grad_mag > min_grad]
    return gphi, grad_mag, gtheta


def __cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


def __make_symmetric(gm, gphi, gtheta):
    gtheta1 = gtheta.copy()
    gtheta2 = gtheta1 + np.pi
    gtheta2[gtheta2 > np.pi] = gtheta2[gtheta2 > np.pi] - 2 * np.pi
    gtheta = np.concatenate((gtheta1, gtheta2))
    gphi = np.concatenate((gphi, -gphi))
    gm = np.concatenate((gm, gm))
    return gm, gphi, gtheta


def build_surface(rho):
    """Calculates X, Y and Z coordinates of spherical histogram surface.

    Plotting example:
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'),
            linewidth=0, antialiased=False, alpha=0.5)

    Args:
        rho (ndarray): 2D array of Rho values for plotting the circular histogram. Output of 'anisotropy3d'.

    Returns:
        ndarray: X coordinates
        ndarray: Y coordinates
        ndarray: Z coordinates

    """

    phi_bins = rho.shape[0]
    theta_bins = rho.shape[1]

    step = 2. * np.pi / theta_bins
    theta = np.linspace(-np.pi + step / 2, np.pi + step / 2, theta_bins + 1)
    step = 2. / theta_bins
    sin_phi = np.linspace(-1 + step / 2, 1 - step / 2, phi_bins)
    sin_phi = np.concatenate((np.asarray([-1]), sin_phi, np.asarray([1])))
    phi = np.arcsin(sin_phi)

    theta_values, phi_values = np.meshgrid(theta, phi)
    rho_values = phi_values * 0 + 1

    rho_values[1:1 + phi_bins, :theta_bins] = rho
    rho_values[:, -1] = rho_values[:, 0]
    rho_values[0, :] = np.mean(rho_values[1, :])
    rho_values[-1, :] = np.mean(rho_values[-2, :])

    xx = rho_values * np.cos(phi_values) * np.cos(theta_values)
    yy = rho_values * np.cos(phi_values) * np.sin(theta_values)
    zz = rho_values * np.sin(phi_values)

    return xx, yy, zz
