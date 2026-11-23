"""The three ``sigma_assumed`` rules of the mismatch protocol (section 2).

An AWGN-designed denoiser needs one noise level. On Poisson-Gaussian (PG) data
the noise variance is signal dependent,  Var(z | x) = alpha * x + sigma_read^2,
so a single number is only an approximation. Three ways to pick it:

    A  equal variance (oracle)   sqrt(alpha * mean(x) + sigma_read^2)
    B  blind estimate            skimage.restoration.estimate_sigma(z)
    C  read noise only (naive)   sigma_read (or a floor if it is 0)

All sigmas are in [0, 1] units, like the images.
"""

import numpy as np
from skimage.restoration import estimate_sigma


def sigma_equal_variance(x, alpha, sigma_read):
    """Rule A: std of the AWGN with the same *average* variance as the PG noise.

    Averaging  Var(z | x) = alpha * x + sigma_read^2  over pixels gives
    alpha * mean(x) + sigma_read^2. Uses the CLEAN image ``x``, so it is an
    oracle: it only sets the level, the denoiser never sees ``x``.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if sigma_read < 0:
        raise ValueError("sigma_read must be non-negative")
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(alpha * x.mean() + sigma_read**2))


def sigma_blind(z):
    """Rule B: wavelet-MAD noise estimate from the noisy image only.

    ``skimage.restoration.estimate_sigma`` on a 2D float image, returned as a
    plain float in the same units as ``z``. On signal-dependent noise this is a
    kind of average level, so it mixes model mismatch with estimation error.
    """
    z = np.asarray(z, dtype=np.float64)
    return float(estimate_sigma(z))


def sigma_read_only(sigma_read, floor=1.0 / 255.0):
    """Rule C: the sensor's read-noise std, ignoring shot noise.

    ``floor`` replaces 0 so that a denoiser is never told "no noise".
    """
    if sigma_read < 0:
        raise ValueError("sigma_read must be non-negative")
    return float(sigma_read) if sigma_read > 0 else float(floor)
