"""Spatial-domain denoisers (Part B, Phase 1).

Mean, Gaussian, median and Wiener filters.  Every function takes a noisy 2D
float array ``z`` and returns a denoised float64 array of the same shape.
No clipping happens here; clipping is done once, in ``denoisers.denoise``.

Border policy (identical for all four filters): the image is extended by
reflection (``mode="reflect"``) so that no filter darkens or biases the borders.
``scipy.signal.wiener`` zero-pads internally, so ``wiener_filter`` reflect-pads
the image itself before calling it and crops the result afterwards.
"""

import numpy as np
from scipy import ndimage, signal


def _check_image(z):
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"expected a 2D grayscale image, got shape {z.shape}")
    return z


def _check_size(size):
    """Window side length: positive odd integer (keeps the window centred)."""
    if int(size) != size or size < 1 or size % 2 == 0:
        raise ValueError(f"size must be a positive odd integer, got {size!r}")
    return int(size)


def mean_filter(z, size=3, mode="reflect"):
    """Box (uniform) filter over a ``size x size`` window.

    Linear smoothing baseline: output = local average.
    """
    z = _check_image(z)
    size = _check_size(size)
    return ndimage.uniform_filter(z, size=size, mode=mode)


def gaussian_filter(z, filter_sigma=1.0, mode="reflect", truncate=4.0):
    """Gaussian smoothing with kernel standard deviation ``filter_sigma`` pixels.

    Note: ``filter_sigma`` is the *kernel width in pixels*, not the noise level.
    It is named differently from the noise ``sigma`` on purpose.
    """
    z = _check_image(z)
    if filter_sigma <= 0:
        raise ValueError("filter_sigma must be positive")
    return ndimage.gaussian_filter(z, sigma=filter_sigma, mode=mode, truncate=truncate)


def median_filter(z, size=3, mode="reflect"):
    """Median filter over a ``size x size`` window (order-statistic filter)."""
    z = _check_image(z)
    size = _check_size(size)
    return ndimage.median_filter(z, size=size, mode=mode)


def wiener_filter(z, size=3, noise_sigma=None):
    """Adaptive (local) Wiener filter, ``scipy.signal.wiener``.

    For each pixel, with local mean m and local variance v over a
    ``size x size`` window and noise variance s2::

        out = m + max(v - s2, 0) / v * (z - m)        (out = m where v < s2)

    Parameters
    ----------
    size : odd window side length
    noise_sigma : noise standard deviation in [0, 1] units (the filter squares
        it to get the noise variance).  ``None`` lets scipy estimate the noise
        power as the mean of the local variances, i.e. no knowledge of the
        noise level is used.  Passing a value is how the mismatch protocol
        controls which sigma the Wiener filter "believes".
    """
    z = _check_image(z)
    size = _check_size(size)
    noise = None if noise_sigma is None else float(noise_sigma) ** 2
    if noise is not None and noise < 0:
        raise ValueError("noise_sigma must be non-negative")

    pad = size // 2
    zp = np.pad(z, pad, mode="reflect") if pad > 0 else z
    # lVar == 0 on perfectly flat windows gives harmless 0/0 warnings in scipy;
    # those pixels are replaced by the local mean / the input just below.
    with np.errstate(divide="ignore", invalid="ignore"):
        out = signal.wiener(zp, mysize=(size, size), noise=noise)
    if pad > 0:
        out = out[pad:-pad, pad:-pad]
    # Flat window and noise == 0 -> NaN: the correct output there is the input.
    return np.where(np.isfinite(out), out, z)
