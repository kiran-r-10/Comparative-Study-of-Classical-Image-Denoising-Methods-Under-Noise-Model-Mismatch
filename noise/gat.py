"""Generalized Anscombe Transform (GAT) and its inverses.

Noise model (Makitalo & Foi, IEEE TIP 2013):

    z = alpha * Poisson(x / alpha) + N(g, sigma^2)

Forward GAT (variance stabilization, output has std ~ 1):

    f(z) = (2 / alpha) * sqrt( alpha * z + (3/8) * alpha^2 + sigma^2 - alpha * g )

(the argument of the square root is floored at 0).

Inverses implemented here:
    * algebraic            : solve f(z) = f_hat for z. Simple but BIASED at low counts.
    * closed-form approx.  : closed-form approximation of the exact unbiased inverse.
    * exact unbiased       : Phase 2 stretch goal (stub that raises NotImplementedError).

Working with the normalised variables  y = (z - g) / alpha  and  s = sigma / alpha
the GAT reduces to the classical Anscombe form  f = 2 * sqrt(y + 3/8 + s^2).
The closed-form inverse is written in that normalised domain and mapped back
with  x = alpha * y_hat + g.

All functions work on scalars or arrays and return float64.
"""

import numpy as np

# 2 * sqrt(3/8): value of the Anscombe transform at z = 0 with no read noise.
# Below this the closed-form expression blows up (it has f^-3 terms), so f is floored here.
_F_MIN = np.sqrt(1.5)
_SQRT_3_2 = np.sqrt(1.5)


def _check(alpha, sigma):
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")


def gat_forward(z, alpha, sigma, g=0.0):
    """Forward Generalized Anscombe Transform."""
    _check(alpha, sigma)
    z = np.asarray(z, dtype=np.float64)
    arg = alpha * z + 0.375 * alpha**2 + sigma**2 - alpha * g
    return (2.0 / alpha) * np.sqrt(np.maximum(arg, 0.0))


def gat_inverse_algebraic(f, alpha, sigma, g=0.0):
    """Algebraic inverse: the exact algebraic solution of f(z) = f_hat.

    z = alpha * f^2 / 4 - 3 * alpha / 8 - sigma^2 / alpha + g

    Round trip ``gat_inverse_algebraic(gat_forward(z))`` returns z exactly
    (wherever the square-root argument was positive), but applied to a
    *denoised* f_hat it is biased for low photon counts.
    """
    _check(alpha, sigma)
    f = np.asarray(f, dtype=np.float64)
    return alpha * f**2 / 4.0 - 3.0 * alpha / 8.0 - sigma**2 / alpha + g


def gat_inverse_closed_form(f, alpha, sigma, g=0.0):
    """Closed-form approximation of the exact unbiased inverse GAT.

    In the normalised domain (s = sigma / alpha):

        y = f^2/4 + (1/4) sqrt(3/2) f^-1 - (11/8) f^-2
            + (5/8) sqrt(3/2) f^-3 - 1/8 - s^2

    and  x_hat = alpha * y + g.

    Note: ``gat_forward`` already returns the normalised value
    f = 2 * sqrt(y + 3/8 + s^2) (the alpha factors cancel), so ``f`` from
    ``gat_forward`` / a denoiser is used directly here.

    f is floored at 2*sqrt(3/8) to keep the expression finite.
    """
    _check(alpha, sigma)
    f = np.maximum(np.asarray(f, dtype=np.float64), _F_MIN)
    s2 = (sigma / alpha) ** 2
    y = (
        f**2 / 4.0
        + 0.25 * _SQRT_3_2 / f
        - (11.0 / 8.0) / f**2
        + (5.0 / 8.0) * _SQRT_3_2 / f**3
        - 0.125
        - s2
    )
    return alpha * y + g


def gat_inverse_exact_unbiased(f, alpha, sigma, g=0.0):
    """Exact unbiased inverse GAT (Phase 2 stretch goal, not implemented yet)."""
    raise NotImplementedError(
        "Exact unbiased inverse GAT is a Phase 2 stretch goal. "
        "Use gat_inverse_closed_form for now."
    )
