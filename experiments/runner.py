"""Experiment loop: noise once per (image, setting, repeat), all methods and arms.

Writes the results CSV incrementally (append + flush after every noise
setting), so a crash loses at most one setting. ``resume=True`` skips rows whose
identifying tuple is already in the CSV. Methods whose ``denoise`` raises
``NotImplementedError`` are skipped with a one-line warning, so the harness
picks up wavelet / NLM / BM3D automatically once Part B implements them.
"""

import csv
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from denoisers import denoise
from metrics import psnr, ssim, timed
from noise import (
    gat_forward,
    gat_inverse_algebraic,
    gat_inverse_closed_form,
    gat_inverse_exact_unbiased,
)
from utils import PROJECT_ROOT, load_dataset

from .arms import ARMS, awgn_settings, build_cases, pg_settings
from .params import params_for

CSV_COLUMNS = [
    "dataset", "image", "method", "arm", "noise_type", "sigma_true", "peak", "sigma_read",
    "sigma_assumed", "repeat", "seed", "psnr", "ssim", "time_s", "noisy_psnr",
]
ESTIMATE_COLUMNS = [
    "dataset", "image", "peak", "sigma_read", "repeat", "seed",
    "sigma_eq", "sigma_blind", "sigma_read_only",
]
_INVERSES = {
    "closed_form": gat_inverse_closed_form,
    "algebraic": gat_inverse_algebraic,
    "exact": gat_inverse_exact_unbiased,
}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def load_images(dataset, max_images=None, data_root=None):
    """list[(name, image)] of a dataset (tests replace this function)."""
    return load_dataset(dataset, data_root=data_root, max_images=max_images)


def apply_preset(cfg, name):
    """Overlay ``cfg[name]`` (e.g. the ``quick`` preset) on the top-level keys."""
    if name is None:
        return cfg
    if name not in cfg or not isinstance(cfg[name], dict):
        raise ValueError(f"preset '{name}' not found in the config")
    merged = {k: v for k, v in cfg.items() if k != name}
    merged.update(cfg[name])
    return merged


def estimates_path(out):
    out = Path(out)
    return out.with_name(out.stem + "_estimates.csv")


def _num(v):
    v = float(v)
    return None if math.isnan(v) else round(v, 9)


def row_key(r):
    """Identifying tuple of a result row (floats rounded: CSV round trips)."""
    return (r["dataset"], r["image"], r["method"], r["arm"], r["noise_type"],
            _num(r["sigma_true"]), _num(r["peak"]), _num(r["sigma_read"]),
            int(float(r["repeat"])), int(float(r["seed"])))


def estimate_key(r):
    return (r["dataset"], r["image"], _num(r["peak"]), _num(r["sigma_read"]),
            int(float(r["repeat"])), int(float(r["seed"])))


def _read_keys(path, key_fn):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with open(path, newline="") as fh:
        return {key_fn(r) for r in csv.DictReader(fh)}


class _CsvAppender:
    """Append rows, write the header once, flush after every batch."""

    def __init__(self, path, columns):
        self.path, self.columns = Path(path), columns
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, rows):
        if not rows:
            return
        new = not self.path.exists() or self.path.stat().st_size == 0
        with open(self.path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=self.columns)
            if new:
                w.writeheader()
            w.writerows(rows)
            fh.flush()


def gat_range(alpha, sigma_read):
    """Nominal GAT output range f(1) - f(0): from the noise model only, not the data."""
    f = gat_forward(np.array([0.0, 1.0]), alpha, sigma_read)
    return float(f[1] - f[0])


def vst_pipeline(method, z, alpha, sigma_read, inverse="closed_form"):
    """GAT -> denoise (sigma = 1, no clipping) -> inverse GAT -> clip to [0, 1].

    Protocol section 4. The stabilised data have unit noise std, so the
    denoiser is told sigma = 1, and its output is not clipped because the GAT
    domain is not [0, 1]. The only clip is at the very end.
    """
    f = gat_forward(z, alpha, sigma_read)
    params = params_for(method, 1.0, "gat", data_range=gat_range(alpha, sigma_read))
    f_hat = denoise(method, f, {**params, "clip": False})
    return np.clip(_INVERSES[inverse](f_hat, alpha, sigma_read), 0.0, 1.0)


def _settings(cfg, arms):
    out = []
    if "M0" in arms:
        out += awgn_settings(cfg)
    if any(a != "M0" for a in arms):
        out += pg_settings(cfg)
    return out


def _cases_per_setting(setting, arms):
    if setting.kind == "awgn":
        return 1 if "M0" in arms else 0
    return sum(1 for a in arms if a != "M0")


# ----------------------------------------------------------------------------
# main entry
# ----------------------------------------------------------------------------
def run_experiments(cfg, datasets, methods, arms, out, max_images=None, repeats=1, seed=None,
                    vst_inverse="closed_form", resume=False, overwrite=False,
                    dry_run=False, progress=True, log=print):
    """Run the benchmark. Returns a dict with the counts (planned, written, skipped...)."""
    arms = list(arms)
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise ValueError(f"Unknown arm(s) {bad}. Choose from {ARMS}.")
    seed = cfg["seed"] if seed is None else seed
    out = Path(out)
    est_out = estimates_path(out)
    settings = _settings(cfg, arms)
    per_image = sum(_cases_per_setting(s, arms) for s in settings) * len(methods) * repeats
    data_root = PROJECT_ROOT / cfg.get("data_root", "data")

    images = {}
    for ds in datasets:
        try:
            images[ds] = load_images(ds, max_images, data_root)
        except FileNotFoundError as err:
            log(f"[skip] dataset '{ds}': {err}")
            images[ds] = []
    planned = per_image * sum(len(v) for v in images.values())
    if dry_run:
        log(f"{planned} runs planned ({per_image} per image, "
            f"{sum(len(v) for v in images.values())} images)")
        return {"planned": planned, "written": 0, "skipped_methods": set()}

    if out.exists() and not resume and not overwrite:
        raise FileExistsError(f"{out} exists. Use --resume to continue it or --overwrite to replace it.")
    if overwrite and not resume:
        for p in (out, est_out):
            p.unlink(missing_ok=True)

    done = _read_keys(out, row_key) if resume else set()
    est_done = _read_keys(est_out, estimate_key) if resume else set()
    writer = _CsvAppender(out, CSV_COLUMNS)
    est_writer = _CsvAppender(est_out, ESTIMATE_COLUMNS)
    unavailable, written = set(), 0

    bar = tqdm(total=planned, disable=not progress, unit="run", desc="experiments")
    for ds, imgs in images.items():
        for name, x in imgs:
            for rep in range(repeats):
                for setting in settings:
                    cases, est = build_cases(x, name, setting, arms, seed, rep, vst_inverse)
                    if est is not None:
                        e = {"dataset": ds, "image": name, "peak": setting.peak,
                             "sigma_read": setting.sigma_read, "repeat": rep, "seed": seed, **est}
                        if estimate_key(e) not in est_done:
                            est_writer.write([e])
                            est_done.add(estimate_key(e))
                    rows = []
                    for case in cases:
                        noisy_psnr = psnr(x, case.z)
                        for method in methods:
                            row = {"dataset": ds, "image": name, "method": method, "arm": case.arm,
                                   "noise_type": case.noise_type, "sigma_true": case.sigma_true,
                                   "peak": case.peak, "sigma_read": case.sigma_read,
                                   "sigma_assumed": case.sigma_assumed, "repeat": rep, "seed": seed}
                            if row_key(row) in done or method in unavailable:
                                bar.update(1)
                                continue
                            try:
                                if case.pipeline == "vst":
                                    x_hat, t = timed(vst_pipeline, method, case.z, case.alpha,
                                                     case.sigma_read, case.vst_inverse)
                                else:
                                    params = params_for(method, case.sigma_assumed, "image")
                                    x_hat, t = timed(denoise, method, case.z, params)
                            except NotImplementedError:
                                unavailable.add(method)
                                log(f"[skip] method '{method}' is not implemented yet (NotImplementedError)")
                                bar.update(1)
                                continue
                            row.update(psnr=psnr(x, x_hat), ssim=ssim(x, x_hat), time_s=t,
                                       noisy_psnr=noisy_psnr)
                            rows.append(row)
                            done.add(row_key(row))
                            bar.update(1)
                    writer.write(rows)          # append + flush after every setting
                    written += len(rows)
    bar.close()
    return {"planned": planned, "written": written, "skipped_methods": unavailable}
