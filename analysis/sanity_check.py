"""Phase 1 sanity check: Part B denoisers on real Set12 images, AWGN only.

Run from the repo root:

    python -m analysis.sanity_check                  # all of Set12, sigma 15/25/50
    python -m analysis.sanity_check --max-images 4   # quick run
    python -m analysis.sanity_check --dataset kodak24 --max-images 3

For every (image, sigma, method) it adds AWGN with Part A's ``add_awgn``,
denoises with ``denoisers.denoise``, and records PSNR, SSIM and runtime.
A "noisy" row (the un-denoised input) is included as the reference: every
method should beat it, otherwise something is wrong.

Outputs (in ``results/`` unless ``--out-dir`` is given):
    phase1_sanity_per_image.csv   one row per (image, sigma, method)
    phase1_sanity_summary.csv     mean over images per (sigma, method)
    phase1_sanity_table.md        the preliminary sanity-check table

Parameters here are fixed, untuned defaults (the proper parameter policy is
Phase 2).  The Wiener filter is given the TRUE noise sigma (matched case).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from denoisers import denoise
from metrics import psnr, ssim, timed
from noise import add_awgn
from utils.config import PROJECT_ROOT, load_config
from utils.data_loader import load_dataset
from utils.seed import make_rng

SANITY_METHODS = ("mean", "gaussian", "median", "wiener")


def sanity_params(method, sigma):
    """Fixed Phase 1 parameters; ``sigma`` is the true noise std in [0, 1] units."""
    if method == "wiener":
        return {"size": 3, "noise_sigma": sigma}  # matched: told the true sigma
    return {}  # mean/gaussian/median: documented defaults (3x3, 3x3, filter_sigma=1)


def run_sanity(images, sigmas_255, seed, clip_noisy=False, methods=SANITY_METHODS):
    """Return a per-image DataFrame for ``images = [(name, array), ...]``."""
    rows = []
    # Untimed warm-up so the first timed call is not paying one-off costs.
    warm = images[0][1]
    for m in methods:
        denoise(m, warm, sanity_params(m, 25 / 255))

    for name, x in images:
        for s255 in sigmas_255:
            sigma = s255 / 255.0
            rng = make_rng(seed, name, int(s255), "sanity")
            z = add_awgn(x, sigma, rng, clip=clip_noisy)
            rows.append(
                dict(image=name, sigma_255=s255, method="noisy",
                     psnr=psnr(x, z), ssim=ssim(x, z), time_s=0.0)
            )
            for m in methods:
                x_hat, dt = timed(denoise, m, z, sanity_params(m, sigma))
                rows.append(
                    dict(image=name, sigma_255=s255, method=m,
                         psnr=psnr(x, x_hat), ssim=ssim(x, x_hat), time_s=dt)
                )
    return pd.DataFrame(rows)


def summarize(df):
    """Mean PSNR / SSIM / runtime over images for each (sigma, method)."""
    order = ["noisy", *SANITY_METHODS]
    g = (
        df.groupby(["sigma_255", "method"], sort=False)
        .agg(psnr=("psnr", "mean"), ssim=("ssim", "mean"),
             time_ms=("time_s", lambda t: 1000.0 * t.mean()),
             n_images=("image", "nunique"))
        .reset_index()
    )
    g["method"] = pd.Categorical(g["method"], categories=order, ordered=True)
    return g.sort_values(["sigma_255", "method"]).reset_index(drop=True)


def to_markdown(summary, dataset, n_images):
    lines = [
        f"# Phase 1 sanity check ({dataset}, {n_images} images, AWGN)",
        "",
        "Mean over images. `noisy` = the un-denoised input. Fixed untuned "
        "parameters: mean 3x3, gaussian filter_sigma=1.0, median 3x3, "
        "wiener 3x3 given the true sigma.",
        "",
    ]
    for s255, grp in summary.groupby("sigma_255"):
        lines += [
            f"## sigma = {s255}/255",
            "",
            "| Method | PSNR (dB) | SSIM | Time (ms/image) |",
            "|---|---|---|---|",
        ]
        for _, r in grp.iterrows():
            t = "-" if r["method"] == "noisy" else f"{r['time_ms']:.2f}"
            lines.append(f"| {r['method']} | {r['psnr']:.2f} | {r['ssim']:.4f} | {t} |")
        lines.append("")
    return "\n".join(lines)


def check_sanity(summary):
    """Return a list of warnings for methods that do not beat the noisy input."""
    bad = []
    for s255, grp in summary.groupby("sigma_255"):
        noisy = grp.loc[grp["method"] == "noisy", "psnr"].iloc[0]
        for _, r in grp.iterrows():
            if r["method"] != "noisy" and r["psnr"] <= noisy:
                bad.append(f"sigma={s255}: {r['method']} PSNR {r['psnr']:.2f} <= noisy {noisy:.2f}")
    return bad


def main(argv=None):
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", default="set12")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--sigmas", type=int, nargs="+",
                   default=cfg["noise"]["awgn"]["sigmas_255"],
                   help="AWGN std on the 8-bit scale")
    p.add_argument("--data-root", default=None,
                   help="override the data folder (default: config data_root)")
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "results"))
    args = p.parse_args(argv)

    data_root = args.data_root or (PROJECT_ROOT / cfg.get("data_root", "data"))
    images = load_dataset(args.dataset, data_root=data_root, max_images=args.max_images)
    print(f"Loaded {len(images)} image(s) from {args.dataset}; sigmas={args.sigmas}")

    df = run_sanity(images, args.sigmas, cfg["seed"], cfg.get("clip_noisy", False))
    summary = summarize(df)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "phase1_sanity_per_image.csv", index=False)
    summary.to_csv(out / "phase1_sanity_summary.csv", index=False)
    md = to_markdown(summary, args.dataset, len(images))
    (out / "phase1_sanity_table.md").write_text(md)

    print(md)
    for w in check_sanity(summary):
        print("WARNING:", w)
    print(f"Wrote 3 files to {out}")


if __name__ == "__main__":
    main()
