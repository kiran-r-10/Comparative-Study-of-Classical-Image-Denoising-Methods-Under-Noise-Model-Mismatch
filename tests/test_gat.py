"""Tests for the Generalized Anscombe Transform and its inverses."""

import numpy as np
import pytest

from noise import (
    add_poisson_gaussian,
    alpha_from_peak,
    gat_forward,
    gat_inverse_algebraic,
    gat_inverse_closed_form,
    gat_inverse_exact_unbiased,
)
from noise.diagnostics import gat_flat_patch_variance, inverse_bias_curve
from utils import make_rng


# ---------------- Variance stabilization ----------------
# The GAT stabilizes variance to ~1 when there are enough photons / enough
# Gaussian noise. At very low counts (e.g. peak = 1) it is known to be inexact,
# so the strict check is restricted to well-conditioned settings.
@pytest.mark.parametrize(
    "peak, sigma255, x_value",
    [
        (30, 0, 0.5),
        (30, 10, 0.1),
        (30, 10, 0.5),
        (30, 10, 0.9),
        (10, 5, 0.5),
        (5, 5, 0.9),
        (5, 5, 0.5),
    ],
)
def test_gat_stabilizes_variance_to_one(peak, sigma255, x_value):
    v = gat_flat_patch_variance(
        x_value, alpha_from_peak(peak), sigma255 / 255.0, make_rng(10, peak, sigma255, int(x_value * 10))
    )
    assert v == pytest.approx(1.0, abs=0.1)


def test_gat_flattens_signal_dependence():
    """After GAT the variance should be (nearly) the same for dark and bright pixels,
    while before GAT it differs by a large factor."""
    alpha, sigma = alpha_from_peak(30), 5 / 255
    rng = make_rng(11)
    v_dark = gat_flat_patch_variance(0.2, alpha, sigma, rng)
    v_bright = gat_flat_patch_variance(0.9, alpha, sigma, rng)
    assert abs(v_dark - v_bright) < 0.15


# ---------------- Forward GAT basics ----------------
def test_forward_matches_formula_and_handles_negative_z():
    alpha, sigma = 0.1, 0.02
    z = np.array([-5.0, -0.5, 0.0, 0.3, 1.0])
    f = gat_forward(z, alpha, sigma)
    expected = (2 / alpha) * np.sqrt(np.maximum(alpha * z + 0.375 * alpha**2 + sigma**2, 0))
    assert np.allclose(f, expected)
    assert np.all(np.isfinite(f)) and np.all(f >= 0)  # no NaN from negative arguments


def test_forward_is_monotone():
    z = np.linspace(-0.2, 1.2, 200)
    f = gat_forward(z, 0.1, 0.02)
    assert np.all(np.diff(f) >= 0)


# ---------------- Round trip ----------------
@pytest.mark.parametrize("alpha, sigma, g", [(1.0, 0.0, 0.0), (0.1, 0.02, 0.0), (0.05, 0.1, 0.0), (0.2, 0.03, 0.01)])
def test_algebraic_inverse_round_trip_is_exact(alpha, sigma, g):
    z = np.linspace(0.0, 1.0, 101)
    z_back = gat_inverse_algebraic(gat_forward(z, alpha, sigma, g), alpha, sigma, g)
    assert np.allclose(z_back, z, atol=1e-10)


def test_algebraic_round_trip_on_noisy_image():
    rng = make_rng(12)
    x = rng.random((64, 64))
    alpha, sigma = alpha_from_peak(20), 3 / 255
    z = add_poisson_gaussian(x, alpha, sigma, rng)
    # Only pixels where the sqrt argument was positive can be recovered exactly.
    ok = alpha * z + 0.375 * alpha**2 + sigma**2 > 0
    z_back = gat_inverse_algebraic(gat_forward(z, alpha, sigma), alpha, sigma)
    assert np.allclose(z_back[ok], z[ok], atol=1e-10)


# ---------------- Inverse quality (unbiasedness) ----------------
def test_closed_form_inverse_is_much_less_biased_than_algebraic():
    """Apply the inverses to E[f(z)]: an unbiased inverse returns x."""
    xs = np.array([0.5, 1.0, 2.0, 4.0, 16.0])  # in photon counts (alpha = 1)
    res = inverse_bias_curve(xs, alpha=1.0, sigma=0.0, rng=make_rng(13), n_samples=300_000)
    err_alg = np.abs(res["algebraic"] - xs)
    err_cf = np.abs(res["closed_form"] - xs)
    # closed form is accurate everywhere in this range ...
    assert np.all(err_cf < 0.05)
    # ... and clearly better than the algebraic inverse at low counts
    assert err_cf[0] < 0.2 * err_alg[0]
    assert err_cf[1] < 0.2 * err_alg[1]
    # algebraic inverse is biased low (it underestimates)
    assert np.all(res["algebraic"] < xs)


def test_closed_form_stays_finite_for_tiny_f():
    f = np.array([0.0, 0.1, 1.0, 1.2247, 2.0])
    out = gat_inverse_closed_form(f, 0.1, 0.02)
    assert np.all(np.isfinite(out))


def test_exact_unbiased_inverse_is_a_phase2_stub():
    with pytest.raises(NotImplementedError):
        gat_inverse_exact_unbiased(np.array([1.0]), 1.0, 0.0)


@pytest.mark.parametrize("fn", [gat_forward, gat_inverse_algebraic, gat_inverse_closed_form])
def test_invalid_parameters_rejected(fn):
    with pytest.raises(ValueError):
        fn(np.array([1.0]), 0.0, 0.1)
    with pytest.raises(ValueError):
        fn(np.array([1.0]), 0.1, -0.1)
