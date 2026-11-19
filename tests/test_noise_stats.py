"""Statistical tests for the noise generators (Phase 1, Part A).

Each test uses a fixed seed, so results are deterministic. Tolerances are set
from the sampling error of a 512x512 flat patch (262,144 samples):
the relative error of a sample variance is about sqrt(2 / N) ~ 0.3 percent,
so the tolerances below are comfortably loose but still catch real bugs.
"""

import numpy as np
import pytest

from noise import add_awgn, add_poisson_gaussian, alpha_from_peak
from noise.diagnostics import awgn_stats, pg_flat_patch_stats, flat_patch
from utils import make_rng


# ---------------- AWGN ----------------
@pytest.mark.parametrize("sigma255", [15, 25, 50])
def test_awgn_mean_zero_and_std_sigma(sigma255):
    sigma = sigma255 / 255.0
    s = awgn_stats(sigma, make_rng(0, "awgn", sigma255))
    assert abs(s["mean"]) < 0.01 * max(sigma, 0.05)  # mean ~ 0
    assert s["std"] == pytest.approx(sigma, rel=0.01)  # std ~ sigma


def test_awgn_is_reproducible_and_seed_dependent():
    x = flat_patch(0.5, (64, 64))
    a = add_awgn(x, 0.1, make_rng(7))
    b = add_awgn(x, 0.1, make_rng(7))
    c = add_awgn(x, 0.1, make_rng(8))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_awgn_zero_sigma_is_identity():
    x = np.random.default_rng(0).random((16, 16))
    assert np.array_equal(add_awgn(x, 0.0, make_rng(0)), x)


def test_awgn_clip_policy():
    x = flat_patch(0.5, (128, 128))
    z = add_awgn(x, 0.5, make_rng(0), clip=True)
    assert z.min() >= 0.0 and z.max() <= 1.0
    z_unclipped = add_awgn(x, 0.5, make_rng(0))  # default: not clipped
    assert z_unclipped.min() < 0.0 or z_unclipped.max() > 1.0


# ---------------- Poisson-Gaussian ----------------
@pytest.mark.parametrize("peak", [1, 5, 30])
@pytest.mark.parametrize("sigma255", [0, 5, 10])
@pytest.mark.parametrize("x_value", [0.1, 0.5, 0.9])
def test_pg_flat_patch_mean_and_variance(peak, sigma255, x_value):
    alpha = alpha_from_peak(peak)
    sigma = sigma255 / 255.0
    s = pg_flat_patch_stats(x_value, alpha, sigma, make_rng(1, peak, sigma255, int(x_value * 10)))
    # E[z] = x. Standard error of the mean is sqrt(var / N) ~ 0.002 at most here.
    assert s["mean"] == pytest.approx(x_value, abs=0.01)
    # Var(z) = alpha * x + sigma^2
    assert s["var"] == pytest.approx(s["var_theory"], rel=0.03)


def test_pg_variance_is_signal_dependent():
    """Brighter flat patches must be noisier (unlike AWGN)."""
    alpha, sigma = alpha_from_peak(10), 2 / 255
    rng = make_rng(2)
    v_dark = pg_flat_patch_stats(0.1, alpha, sigma, rng)["var"]
    v_bright = pg_flat_patch_stats(0.9, alpha, sigma, rng)["var"]
    assert v_bright > 5 * v_dark


def test_pg_reduces_to_gaussian_when_alpha_small():
    """alpha -> 0 (huge peak): Poisson part vanishes, noise is ~ N(0, sigma^2)."""
    s = pg_flat_patch_stats(0.5, 1e-6, 0.05, make_rng(3))
    assert s["var"] == pytest.approx(0.05**2, rel=0.05)


def test_pg_output_independent_of_how_alpha_enters_the_mean():
    """Changing peak changes the variance but never the mean."""
    means = [
        pg_flat_patch_stats(0.4, alpha_from_peak(p), 0.0, make_rng(4, p))["mean"]
        for p in (1, 2, 5, 30)
    ]
    assert np.allclose(means, 0.4, atol=0.01)


def test_pg_is_reproducible():
    x = np.random.default_rng(0).random((32, 32))
    a = add_poisson_gaussian(x, 0.1, 0.02, make_rng(5))
    b = add_poisson_gaussian(x, 0.1, 0.02, make_rng(5))
    assert np.array_equal(a, b)


def test_pg_rejects_bad_inputs():
    rng = make_rng(0)
    with pytest.raises(ValueError):
        add_poisson_gaussian(np.array([-0.1, 0.2]), 0.1, 0.0, rng)
    with pytest.raises(ValueError):
        add_poisson_gaussian(np.array([0.1]), 0.0, 0.0, rng)
    with pytest.raises(ValueError):
        add_poisson_gaussian(np.array([0.1]), 0.1, -1.0, rng)


def test_pg_clip_policy():
    x = flat_patch(0.9, (128, 128))
    z = add_poisson_gaussian(x, 1.0, 0.1, make_rng(0), clip=True)
    assert z.min() >= 0.0 and z.max() <= 1.0
