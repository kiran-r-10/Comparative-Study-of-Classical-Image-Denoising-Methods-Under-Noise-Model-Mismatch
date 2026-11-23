"""Tests for PSNR, SSIM, method noise and timing."""

import time

import numpy as np
import pytest

from metrics import method_noise, psnr, ssim, timed
from noise import add_awgn
from utils.seed import make_rng


def _img(n=64, seed=0):
    return np.random.default_rng(seed).uniform(0.1, 0.9, size=(n, n))


def test_psnr_known_value():
    x = np.zeros((16, 16))
    x_hat = np.full((16, 16), 0.1)  # MSE = 0.01 -> 10*log10(1/0.01) = 20 dB
    assert psnr(x, x_hat) == pytest.approx(20.0, abs=1e-9)


def test_psnr_identical_is_inf_and_noise_level_matches_theory():
    x = _img()
    assert psnr(x, x) == np.inf
    z = add_awgn(x, 25 / 255, make_rng(0), clip=False)
    expected = -20 * np.log10(25 / 255)  # ~20.17 dB for sigma = 25/255
    assert psnr(x, z) == pytest.approx(expected, abs=0.1)


def test_ssim_identical_is_one_and_decreases_with_noise():
    x = _img()
    assert ssim(x, x) == pytest.approx(1.0, abs=1e-12)
    s_low = ssim(x, add_awgn(x, 0.02, make_rng(1)))
    s_high = ssim(x, add_awgn(x, 0.2, make_rng(1)))
    assert 1.0 > s_low > s_high


def test_metrics_reject_bad_shapes():
    with pytest.raises(ValueError):
        psnr(np.zeros((4, 4)), np.zeros((4, 5)))
    with pytest.raises(ValueError):
        ssim(np.zeros((8, 8)), np.zeros((9, 9)))
    with pytest.raises(ValueError):
        psnr(np.zeros((4, 4, 3)), np.zeros((4, 4, 3)))


def test_method_noise_is_residual():
    z = _img(seed=2)
    x_hat = _img(seed=3)
    r = method_noise(z, x_hat)
    assert np.allclose(r, z - x_hat) and r.dtype == np.float64
    assert np.allclose(method_noise(z, z), 0.0)
    with pytest.raises(ValueError):
        method_noise(z, z[:-1])


def test_method_noise_of_perfect_denoiser_is_the_noise():
    x = _img(seed=4)
    z = add_awgn(x, 0.1, make_rng(5))
    r = method_noise(z, x)  # denoiser returned the clean image
    assert r.std() == pytest.approx(0.1, rel=0.05)
    assert abs(r.mean()) < 0.01


def test_timed_returns_result_and_seconds():
    result, dt = timed(lambda a, b=1: a + b, 2, b=3)
    assert result == 5
    assert isinstance(dt, float) and dt >= 0.0

    _, dt = timed(time.sleep, 0.05)
    assert 0.04 < dt < 1.0
