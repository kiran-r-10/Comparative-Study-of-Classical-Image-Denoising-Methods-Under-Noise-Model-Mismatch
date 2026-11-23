"""Method noise (Part B).

Method noise = what the denoiser removed: ``z - x_hat`` (Buades, Coll, Morel).
For a good denoiser matched to the noise it looks like pure noise with no image
structure in it; visible edges/texture in it mean the method removed signal.

Phase 1 only provides the residual image required by the interface contract.
The energy and structure statistics (Phase 2) will be added to this module.
"""

import numpy as np


def method_noise(z, x_hat):
    """Return the residual ``z - x_hat`` (same shape, float64, not clipped)."""
    z = np.asarray(z, dtype=np.float64)
    x_hat = np.asarray(x_hat, dtype=np.float64)
    if z.shape != x_hat.shape:
        raise ValueError(f"shape mismatch: {z.shape} vs {x_hat.shape}")
    return z - x_hat
