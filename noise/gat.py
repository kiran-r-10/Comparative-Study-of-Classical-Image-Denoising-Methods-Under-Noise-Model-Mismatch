"""Generalized Anscombe Transform (GAT) and its inverses.

Noise model (Makitalo & Foi, IEEE TIP 2013):

    z = alpha * Poisson(x / alpha) + N(g, sigma^2)

Forward GAT (variance stabilization, output has std ~ 1):

    f(z) = (2 / alpha) * sqrt( alpha * z + (3/8) * alpha^2 + sigma^2 - alpha * g )

(the argument of the square root is floored at 0).

Inverses implemented here:
    * algebraic            : solve f(z) = f_hat for z. Simple but BIASED at low counts.
    * closed-form approx.  : closed-form approximation of the exact unbiased inverse.
    * exact unbiased       : numerical inverse of E[f(z) | x] (Makitalo & Foi, 2013).

Working with the normalised variables  y = (z - g) / alpha  and  s = sigma / alpha
the GAT reduces to the classical Anscombe form  f = 2 * sqrt(y + 3/8 + s^2).
The closed-form inverse is written in that normalised domain and mapped back
with  x = alpha * y_hat + g.

All functions work on scalars or arrays and return float64.
"""

from functools import lru_cache

import numpy as np
from scipy.special import gammaln

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


# ----------------------------------------------------------------------------
# Exact unbiased inverse
# ----------------------------------------------------------------------------
_LAMBDA_CAP = 2000.0   # above this many expected photons the closed form is used
_N_GRID = 400          # points of the lookup table E[f | lambda]
_N_GAUSS = 801         # points of the quadrature over the Gaussian read noise
_GAUSS_SPAN = 8.0      # quadrature covers +/- 8 standard deviations


def _expected_f(lam, s):
    """E[f | lambda] for the normalised model, at every value in ``lam``.

    With y = (z - g) / alpha the model is y = k + e, k ~ Poisson(lambda),
    e ~ N(0, s^2), s = sigma / alpha, and  f = 2 sqrt(max(y + 3/8 + s^2, 0)).
    Conditioning on k,

        h_k = E_e[ 2 sqrt(max(k + 3/8 + s^2 + e, 0)) ],
        E[f | lambda] = sum_k Pois(k; lambda) h_k.

    h_k is computed by quadrature over e (kink at the floor, so a dense
    trapezoid rule rather than a low-order Gauss-Hermite rule), the sum over k
    by an adaptive range [0, k_max] with k_max = lam_max + 10 sqrt(lam_max) + 25.
    """
    lam = np.asarray(lam, dtype=np.float64)
    lam_max = float(lam.max())
    k = np.arange(int(np.ceil(lam_max + 10.0 * np.sqrt(lam_max) + 25.0)) + 1, dtype=np.float64)
    a = k + 0.375 + s**2
    if s > 0:
        u = np.linspace(-_GAUSS_SPAN, _GAUSS_SPAN, _N_GAUSS)
        w = np.exp(-0.5 * u**2)
        w /= w.sum()
        h = 2.0 * np.sqrt(np.maximum(a[:, None] + s * u[None, :], 0.0)) @ w
    else:
        h = 2.0 * np.sqrt(a)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = k[None, :] * np.log(lam[:, None]) - lam[:, None] - gammaln(k + 1.0)[None, :]
    pmf = np.exp(logp)
    pmf[lam == 0.0] = 0.0            # lambda = 0: all mass on k = 0
    pmf[lam == 0.0, 0] = 1.0
    return pmf @ h


@lru_cache(maxsize=32)
def _exact_table(s, lam_top):
    """Lookup table (E[f | lambda], lambda) on a grid that is dense at low counts."""
    lam = np.linspace(0.0, np.sqrt(lam_top), _N_GRID) ** 2
    ef = np.maximum.accumulate(_expected_f(lam, s))   # monotone by theory; guards round-off
    ef.setflags(write=False)
    lam.setflags(write=False)
    return ef, lam


def gat_inverse_exact_unbiased(f, alpha, sigma, g=0.0):
    """Exact unbiased inverse GAT (Makitalo & Foi, IEEE TIP 2013).

    Returns the value x_hat = alpha * lambda_hat + g (same convention as the
    closed form: an estimate of E[z]) such that

        E[ f(z) | E[z] = x_hat ] = f.

    ``E[f | lambda]`` is strictly increasing in lambda, so the inverse is found
    by interpolating a table of the forward expectation (see ``_expected_f``).
    The table depends only on ``s = sigma / alpha`` and on the largest ``f``
    (rounded up to a power of two), so it is cached across calls.

    * ``f`` below ``E[f | lambda = 0]`` maps to lambda = 0 (x_hat = g).
    * Above ``_LAMBDA_CAP`` photons the bias is negligible and the closed-form
      inverse is used for those pixels.
    """
    _check(alpha, sigma)
    f = np.asarray(f, dtype=np.float64)
    if f.size == 0:
        return f.copy()
    s = float(sigma) / float(alpha)
    lam_needed = min(float(np.nanmax(f)) ** 2 / 4.0 + 2.0, _LAMBDA_CAP)
    lam_top = float(max(16.0, 2.0 ** np.ceil(np.log2(lam_needed))))
    ef, lam = _exact_table(s, lam_top)
    lam_hat = np.interp(f, ef, lam)          # clamps to lam[0] = 0 below the table
    x_hat = alpha * lam_hat + g
    beyond = f > ef[-1]
    if np.any(beyond):
        x_hat = np.where(beyond, gat_inverse_closed_form(f, alpha, sigma, g), x_hat)
    return x_hat
