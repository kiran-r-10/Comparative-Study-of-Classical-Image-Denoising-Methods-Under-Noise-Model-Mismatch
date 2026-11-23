"""Smoke and behaviour tests for the Phase 1 denoisers (synthetic images only)."""

import numpy as np
import pytest

from denoisers import (
    METHODS,
    PHASE1_METHODS,
    PHASE2_METHODS,
    denoise,
    get_default_params,
)
from metrics import psnr
from noise import add_awgn
from utils.seed import make_rng


def _test_image(n=64):
    """Smooth gradient + bright square + dark disc, float64 in [0.1, 0.9]."""
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1)
    img = 0.3 + 0.3 * xx
    img[n // 4 : n // 2, n // 4 : n // 2] = 0.85
    img[(yy - 0.7) ** 2 + (xx - 0.7) ** 2 < 0.04] = 0.15
    return img.astype(np.float64)


@pytest.mark.parametrize("method", PHASE1_METHODS)
def test_output_contract(method):
    x = _test_image()
    z = add_awgn(x, 25 / 255, make_rng(0, method))
    z_copy = z.copy()
    out = denoise(method, z, {})
    assert out.shape == z.shape and out.dtype == np.float64
    assert np.all(np.isfinite(out))
    assert out.min() >= 0.0 and out.max() <= 1.0  # clip=True by default
    assert np.array_equal(z, z_copy), "input must not be modified"


@pytest.mark.parametrize("method", PHASE1_METHODS)
def test_params_none_equals_defaults(method):
    z = add_awgn(_test_image(), 0.05, make_rng(1, method))
    a = denoise(method, z)
    b = denoise(method, z, get_default_params(method))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("method", PHASE1_METHODS)
def test_flat_image_stays_flat_including_borders(method):
    """No filter may darken/brighten borders (reflect policy, also for Wiener)."""
    x = np.full((32, 32), 0.5)
    params = {"noise_sigma": 0.05} if method == "wiener" else {}
    out = denoise(method, x, params)
    assert np.allclose(out, 0.5, atol=1e-12)


@pytest.mark.parametrize("method", PHASE1_METHODS)
def test_denoising_improves_psnr_on_awgn(method):
    x = _test_image()
    z = add_awgn(x, 25 / 255, make_rng(2, method))
    params = {"noise_sigma": 25 / 255} if method == "wiener" else {}
    out = denoise(method, z, params)
    assert psnr(x, out) > psnr(x, np.clip(z, 0, 1)) + 1.0


def test_mean_filter_is_local_average():
    z = np.zeros((5, 5))
    z[2, 2] = 9.0
    out = denoise("mean", z, {"size": 3, "clip": False})
    assert np.isclose(out[2, 2], 1.0)  # 9 / 9
    assert np.isclose(out[1, 1], 1.0) and np.isclose(out[0, 0], 0.0)


def test_median_removes_impulse():
    z = np.full((9, 9), 0.4)
    z[4, 4] = 1.0
    out = denoise("median", z, {"size": 3})
    assert np.allclose(out, 0.4)


def test_gaussian_filter_sigma_controls_smoothing():
    x = _test_image()
    z = add_awgn(x, 0.1, make_rng(3))
    weak = denoise("gaussian", z, {"filter_sigma": 0.5, "clip": False})
    strong = denoise("gaussian", z, {"filter_sigma": 2.0, "clip": False})
    assert np.var(strong - x) < np.var(weak - x)  # more smoothing, less noise left here


def test_wiener_mismatched_sigma_changes_output():
    """The sigma handed to Wiener matters: this is what mismatch runs vary."""
    z = add_awgn(_test_image(), 25 / 255, make_rng(4))
    low = denoise("wiener", z, {"noise_sigma": 2 / 255})
    high = denoise("wiener", z, {"noise_sigma": 60 / 255})
    assert not np.allclose(low, high)
    # a far-too-large assumed sigma smooths (almost) everywhere
    assert np.var(np.diff(high, axis=1)) < np.var(np.diff(low, axis=1))


def test_wiener_without_sigma_uses_scipy_estimate():
    z = add_awgn(_test_image(), 25 / 255, make_rng(5))
    out = denoise("wiener", z, {"noise_sigma": None})
    assert np.all(np.isfinite(out))


def test_wiener_zero_noise_returns_input_on_flat_regions():
    z = np.full((16, 16), 0.3)
    out = denoise("wiener", z, {"noise_sigma": 0.0, "clip": False})
    assert np.allclose(out, 0.3)


def test_clip_flag():
    z = np.zeros((9, 9))
    z[4, 4] = 5.0
    raw = denoise("gaussian", z, {"filter_sigma": 0.5, "clip": False})
    clipped = denoise("gaussian", z, {"filter_sigma": 0.5, "clip": True})
    assert raw.max() > 1.0
    assert clipped.max() <= 1.0


def test_unknown_method_and_parameters_are_rejected():
    z = _test_image()
    with pytest.raises(ValueError):
        denoise("nonsense", z, {})
    with pytest.raises(ValueError):
        denoise("median", z, {"filter_sigma": 1.0})  # belongs to gaussian
    with pytest.raises(ValueError):
        denoise("mean", z, {"sizee": 3})  # typo must not run silently


@pytest.mark.parametrize("bad", [0, 2, 4, -3, 3.5])
@pytest.mark.parametrize("method", ["mean", "median", "wiener"])
def test_window_size_must_be_positive_odd_integer(method, bad):
    with pytest.raises(ValueError):
        denoise(method, _test_image(), {"size": bad})


def test_rejects_non_2d_input():
    with pytest.raises(ValueError):
        denoise("mean", np.zeros((4, 4, 3)), {})


def test_phase2_methods_registered_but_not_implemented():
    assert set(PHASE2_METHODS) <= set(METHODS)
    for m in PHASE2_METHODS:
        with pytest.raises(NotImplementedError):
            denoise(m, _test_image(), {})
