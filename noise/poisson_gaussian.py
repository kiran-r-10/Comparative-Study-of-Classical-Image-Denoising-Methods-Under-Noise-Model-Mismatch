"""Poisson-Gaussian (signal-dependent) noise generator.

Model (Makitalo & Foi, IEEE TIP 2013):

    z = alpha * Poisson(x / alpha) + n,     n ~ N(0, sigma^2)

so that  E[z] = x  and  Var(z) = alpha * x + sigma^2.

Conventions (shared interface contract):
    * x is a float64 array in [0, 1] (must be >= 0, Poisson needs a rate >= 0).
    * alpha is the gain. We describe the Poisson strength by the "peak":
      ``alpha = 1 / peak``. A pixel with x = 1 then has an expected photon
      count of ``peak``. Low peak = very noisy (Poisson dominated).
    * sigma is the standard deviation of the Gaussian (read) noise, in the same
      [0, 1] units as x.
    * Order of random draws (matters for reproducibility): first the Poisson
      draw, then the Gaussian draw, both from the same ``rng``.
    * Clipping policy: by default the output is NOT clipped, so that
      E[z] = x holds exactly and the GAT theory applies. ``clip=True`` clips
      to [0, 1].
"""

import numpy as np


def alpha_from_peak(peak):
    """Gain alpha for a given peak photon count (alpha = 1 / peak)."""
    if peak <= 0:
        raise ValueError("peak must be positive")
    return 1.0 / float(peak)


def add_poisson_gaussian(x, alpha, sigma, rng, clip=False):
    """Return ``alpha * Poisson(x / alpha) + N(0, sigma^2)``.

    Parameters
    ----------
    x : ndarray, float, shape (H, W), values in [0, 1]
    alpha : float > 0, gain (use ``alpha_from_peak(peak)``)
    sigma : float >= 0, Gaussian noise std (in [0, 1] units)
    rng : numpy.random.Generator
    clip : bool, clip the result to [0, 1] (default False)
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    x = np.asarray(x, dtype=np.float64)
    if x.size and x.min() < 0:
        raise ValueError("x must be non-negative for Poisson noise")

    poisson_part = alpha * rng.poisson(x / alpha).astype(np.float64)
    gaussian_part = sigma * rng.normal(size=x.shape)
    z = poisson_part + gaussian_part
    return np.clip(z, 0.0, 1.0) if clip else z
