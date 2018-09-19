
import numpy as np
from scipy.signal import correlate


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


def anisotropy3d(im3, min_grad=-1.0, power=0., theta_bins=24, phi_bins=12, symmetric=True):
    plane = np.asarray([[3, 2, 3], [2, 1, 2], [3, 2, 3]]) ** (-0.5)
    sobel = np.zeros((3, 3, 3))
    sobel[:, :, 0] = -plane
    sobel[:, :, 2] = plane

    gz = correlate(im3, sobel, mode='valid')
    gx = correlate(im3, np.swapaxes(sobel, 2, 1), mode='valid')
    gy = correlate(im3, np.swapaxes(sobel, 2, 0), mode='valid')

    gtheta, gphi, gm = cart2sph(gx, gy, gz)
    gm = gm.flatten()
    gtheta = gtheta.flatten()
    gphi = gphi.flatten()
    if min_grad >= 0:
        gtheta = gtheta[gm > min_grad]
        gphi = gphi[gm > min_grad]
        gm = gm[gm > min_grad]

    if symmetric:
        gtheta1 = gtheta.copy()
        gtheta2 = gtheta1 + np.pi
        gtheta2[gtheta2 > np.pi] = gtheta2[gtheta2 > np.pi] - 2 * np.pi
        gtheta = np.concatenate((gtheta1, gtheta2))
        gphi = np.concatenate((gphi, -gphi))
        gm = np.concatenate((gm, gm))

    btheta = np.floor((gtheta + np.pi) / 2 / np.pi * theta_bins)
    btheta[btheta >= theta_bins] = theta_bins - 1
    gsinphi = np.sin(gphi)
    bphi = np.floor((gsinphi + 1) / 2 * phi_bins)
    bphi[bphi >= phi_bins] = phi_bins - 1

    idx = btheta + theta_bins * bphi
    n = theta_bins * phi_bins
    if power == 0:
        r, _ = np.histogram(idx, bins=n, range=(-0.5, n - 0.5))
    else:
        r = np.zeros((n,))
        for bin in range(n):
            r[bin] = np.sum(np.power(gm[idx == bin], power))

    r = np.reshape(r, (phi_bins, theta_bins))

    isotropy = np.min(r) / np.max(r)
    r1 = r / np.sum(r)
    std = np.std(r1)
    entropy = -np.sum(r1 * np.log(r1)) / np.log(n)

    return r, isotropy, std, entropy


def build_surface(r):
    phi_bins = r.shape[0]
    theta_bins = r.shape[1]

    step = 2. * np.pi / theta_bins
    theta = np.linspace(-np.pi + step / 2, np.pi + step / 2, theta_bins + 1)
    step = 2. / theta_bins
    sin_phi = np.linspace(-1 + step / 2, 1 - step / 2, phi_bins)
    sin_phi = np.concatenate((np.asarray([-1]), sin_phi, np.asarray([1])))
    phi = np.arcsin(sin_phi)

    THETA, PHI = np.meshgrid(theta, phi)
    R = PHI * 0 + 1

    R[1:1 + phi_bins, :theta_bins] = r
    R[:, -1] = R[:, 0]
    R[0, :] = np.mean(R[1, :])
    R[-1, :] = np.mean(R[-2, :])

    X = R * np.cos(PHI) * np.cos(THETA)
    Y = R * np.cos(PHI) * np.sin(THETA)
    Z = R * np.sin(PHI)

    return X, Y, Z