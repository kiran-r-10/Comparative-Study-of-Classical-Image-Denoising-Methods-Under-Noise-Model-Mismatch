"""Phase 1 integration test (shared Part A + Part B).

One real Set12 image, every implemented method, one noise setting (AWGN,
sigma = 25/255).  Uses Part A's loader and noise generator and Part B's
denoisers and metrics together.  Skipped (not failed) if Set12 is not in data/.

Phase 2: extend ``METHODS_UNDER_TEST`` with "wavelet", "nlm", "bm3d" once they
exist, and add a Poisson-Gaussian case.
"""

import numpy as np
import pytest

from denoisers import PHASE1_METHODS, denoise
from metrics import method_noise, psnr, ssim, timed
from noise import add_awgn
from utils.data_loader import load_dataset
from utils.seed import make_rng

METHODS_UNDER_TEST = PHASE1_METHODS
SIGMA = 25 / 255


@pytest.fixture(scope="module")
def image():
    try:
        name, x = load_dataset("set12", max_images=1)[0]
    except FileNotFoundError:
        pytest.skip("Set12 not found in data/set12 (see data/README.md)")
    return name, x


@pytest.mark.parametrize("method", METHODS_UNDER_TEST)
def test_end_to_end_one_image_one_noise_setting(image, method):
    name, x = image
    z = add_awgn(x, SIGMA, make_rng(2026, name, 25, "integration"))
    params = {"noise_sigma": SIGMA} if method == "wiener" else {}

    x_hat, dt = timed(denoise, method, z, params)

    assert x_hat.shape == x.shape and x_hat.dtype == np.float64
    assert np.all(np.isfinite(x_hat))
    assert dt > 0
    assert method_noise(z, x_hat).shape == x.shape

    p_noisy, p_den = psnr(x, z), psnr(x, x_hat)
    assert p_den > p_noisy, f"{method}: {p_den:.2f} dB not better than noisy {p_noisy:.2f} dB"
    assert 0.0 < ssim(x, x_hat) <= 1.0
