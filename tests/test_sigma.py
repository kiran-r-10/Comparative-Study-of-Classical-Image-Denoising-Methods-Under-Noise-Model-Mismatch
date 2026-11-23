"""Tests for the three sigma_assumed rules."""

import numpy as np
import pytest

from noise import (
    add_awgn,
    add_poisson_gaussian,
    alpha_from_peak,
    sigma_blind,
    sigma_equal_variance,
    sigma_read_only,
)
from utils import load_sample_images, make_rng


@pytest.mark.parametrize("peak, sr255, value", [(2, 5, 0.5), (10, 10, 0.3), (30, 0, 0.8), (1, 5, 0.25)])
def test_rule_a_matches_empirical_std_on_flat_patch(peak, sr255, value):
    alpha, sr = alpha_from_peak(peak), sr255 / 255
    x = np.full((512, 512), value)
    z = add_poisson_gaussian(x, alpha, sr, make_rng(30, peak, sr255))
    assert sigma_equal_variance(x, alpha, sr) == pytest.approx(np.std(z - x), rel=0.01)


def test_rule_a_uses_mean_of_clean_image():
    x = np.array([[0.0, 1.0], [0.0, 1.0]])
    assert sigma_equal_variance(x, 0.1, 0.0) == pytest.approx(np.sqrt(0.1 * 0.5))


def test_rule_a_rejects_bad_parameters():
    with pytest.raises(ValueError):
        sigma_equal_variance(np.ones((2, 2)), 0.0, 0.1)
    with pytest.raises(ValueError):
        sigma_equal_variance(np.ones((2, 2)), 0.1, -0.1)


@pytest.mark.parametrize("s255", [15, 25, 50])
def test_rule_b_is_close_to_true_sigma_on_awgn(s255):
    name, x = load_sample_images(256)[0]
    z = add_awgn(x, s255 / 255, make_rng(31, s255))
    est = sigma_blind(z)
    assert isinstance(est, float)
    assert est == pytest.approx(s255 / 255, rel=0.15)


def test_rule_c_floor_behaviour():
    assert sigma_read_only(10 / 255) == pytest.approx(10 / 255)
    assert sigma_read_only(0.0) == pytest.approx(1 / 255)
    assert sigma_read_only(0.0, floor=0.01) == 0.01
    with pytest.raises(ValueError):
        sigma_read_only(-1.0)
