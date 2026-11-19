"""Statistical diagnostics for the noise generators and the GAT.

These helpers are shared by the pytest tests (which assert on them) and by the
Phase 1 notebook (which plots them), so both always check the same thing.

"Flat patch" = an image with a constant intensity. On a flat patch every pixel
is an independent draw of the same distribution, so sample statistics over the
patch estimate the true mean / variance directly.
"""

import numpy as np

from .awgn import add_awgn
from .poisson_gaussian import add_poisson_gaussian
from .gat import gat_forward, gat_inverse_algebraic, gat_inverse_closed_form


def flat_patch(value, shape=(512, 512)):
    """Constant image of the given intensity."""
    return np.full(shape, float(value), dtype=np.float64)


def awgn_stats(sigma, rng, value=0.5, shape=(512, 512)):
    """Empirical mean and std of the AWGN noise ``z - x`` on a flat patch."""
    x = flat_patch(value, shape)
    n = add_awgn(x, sigma, rng) - x
    return {"mean": float(n.mean()), "std": float(n.std()), "sigma": float(sigma)}


def pg_flat_patch_stats(x_value, alpha, sigma, rng, shape=(512, 512)):
    """Empirical mean / variance of Poisson-Gaussian z on a flat patch,
    next to the theory  E[z] = x  and  Var(z) = alpha * x + sigma^2."""
    x = flat_patch(x_value, shape)
    z = add_poisson_gaussian(x, alpha, sigma, rng)
    return {
        "x": float(x_value),
        "mean": float(z.mean()),
        "var": float(z.var()),
        "var_theory": float(alpha * x_value + sigma**2),
    }


def gat_flat_patch_variance(x_value, alpha, sigma, rng, shape=(512, 512)):
    """Variance of f(z) on a flat patch. Should be close to 1 (stabilized)."""
    x = flat_patch(x_value, shape)
    z = add_poisson_gaussian(x, alpha, sigma, rng)
    f = gat_forward(z, alpha, sigma)
    return float(f.var())


def inverse_bias_curve(x_values, alpha, sigma, rng, n_samples=400_000):
    """Test of the *unbiased inverse* property.

    For each true x: draw many z, compute f = GAT(z), take the mean of f
    (this is what an ideal denoiser would output), and apply each inverse to
    that mean. An ideal inverse returns x.

    Returns a dict with arrays 'x', 'algebraic', 'closed_form'.
    """
    x_values = np.asarray(x_values, dtype=np.float64)
    alg, cf = [], []
    for xv in x_values:
        z = add_poisson_gaussian(np.full(n_samples, xv), alpha, sigma, rng)
        f_mean = gat_forward(z, alpha, sigma).mean()
        alg.append(float(gat_inverse_algebraic(f_mean, alpha, sigma)))
        cf.append(float(gat_inverse_closed_form(f_mean, alpha, sigma)))
    return {"x": x_values, "algebraic": np.array(alg), "closed_form": np.array(cf)}
