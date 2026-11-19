"""Additive white Gaussian noise (AWGN) generator.

Model:  z = x + n,   n ~ N(0, sigma^2)

Conventions (shared interface contract):
    * x is a float64 array in [0, 1].
    * sigma is in the same [0, 1] units. For "sigma = 25" on an 8-bit scale
      pass ``25 / 255``.
    * Randomness only comes from the ``numpy.random.Generator`` passed in as
      ``rng``, so every result is reproducible from a seed.
    * Clipping policy: by default the noisy image is NOT clipped, because the
      theory (and most denoising papers) assume unclipped Gaussian noise.
      Pass ``clip=True`` to clip to [0, 1] for an 8-bit-like sensor.
"""

import numpy as np


def add_awgn(x, sigma, rng, clip=False):
    """Return ``x + N(0, sigma^2)``.

    Parameters
    ----------
    x : ndarray, float, shape (H, W), values in [0, 1]
    sigma : float, noise standard deviation (in [0, 1] units), must be >= 0
    rng : numpy.random.Generator
    clip : bool, clip the result to [0, 1] (default False)
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    x = np.asarray(x, dtype=np.float64)
    z = x + sigma * rng.normal(size=x.shape)
    return np.clip(z, 0.0, 1.0) if clip else z
