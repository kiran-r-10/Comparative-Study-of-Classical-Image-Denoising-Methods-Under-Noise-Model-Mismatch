"""Tests for the Phase 1 sanity-check script (synthetic images, no data needed)."""

import numpy as np

from analysis.sanity_check import (
    SANITY_METHODS,
    check_sanity,
    run_sanity,
    summarize,
    to_markdown,
)


def _images():
    out = []
    for k in range(2):
        yy, xx = np.mgrid[0:64, 0:64] / 63.0
        img = 0.3 + 0.3 * xx + 0.05 * k
        img[16:32, 16:32] = 0.85
        out.append((f"img{k}.png", img))
    return out


def test_run_sanity_shape_and_reproducibility():
    df1 = run_sanity(_images(), [15, 25], seed=2026)
    df2 = run_sanity(_images(), [15, 25], seed=2026)
    # 2 images x 2 sigmas x (noisy + 4 methods)
    assert len(df1) == 2 * 2 * (1 + len(SANITY_METHODS))
    assert set(df1["method"]) == {"noisy", *SANITY_METHODS}
    cols = ["image", "sigma_255", "method", "psnr", "ssim"]
    assert df1[cols].equals(df2[cols])  # same seed -> identical noise and results


def test_summary_and_table_and_sanity_flags():
    df = run_sanity(_images(), [25], seed=2026)
    s = summarize(df)
    assert list(s["method"].astype(str)) == ["noisy", *SANITY_METHODS]
    md = to_markdown(s, "synthetic", 2)
    assert "sigma = 25/255" in md and "| wiener |" in md
    assert check_sanity(s) == []  # every method beats the noisy input
