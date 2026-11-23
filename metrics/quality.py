"""Image-quality metrics: PSNR and SSIM (Part B).

Both compare a clean reference ``x`` with an estimate ``x_hat`` and use
``data_range=1.0`` because all images are float arrays in [0, 1].
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

DATA_RANGE = 1.0


def _prepare(x, x_hat):
    x = np.asarray(x, dtype=np.float64)
    x_hat = np.asarray(x_hat, dtype=np.float64)
    if x.shape != x_hat.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {x_hat.shape}")
    if x.ndim != 2:
        raise ValueError(f"expected 2D grayscale images, got shape {x.shape}")
    return x, x_hat


def psnr(x, x_hat):
    """PSNR in dB: ``10 log10(1 / MSE)`` (``inf`` if the images are identical).

    ``x`` is the clean reference, ``x_hat`` the estimate. Neither is clipped
    here: clip ``x_hat`` before calling if the pipeline requires it
    (``denoisers.denoise`` clips by default).
    """
    x, x_hat = _prepare(x, x_hat)
    with np.errstate(divide="ignore"):  # identical images -> MSE 0 -> inf, by design
        return float(peak_signal_noise_ratio(x, x_hat, data_range=DATA_RANGE))


def ssim(x, x_hat):
    """Mean SSIM (skimage defaults: 7x7 uniform window, K1=0.01, K2=0.03)."""
    x, x_hat = _prepare(x, x_hat)
    return float(structural_similarity(x, x_hat, data_range=DATA_RANGE))
