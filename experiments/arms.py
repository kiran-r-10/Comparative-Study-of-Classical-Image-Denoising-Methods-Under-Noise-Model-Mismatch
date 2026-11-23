"""Arms of the mismatch protocol (report/mismatch_protocol.md, sections 1-3).

For one (image, noise setting, repeat) this module draws the noise ONCE and
returns the cases to evaluate, so MM-A, MM-B, MM-C and VST are paired: they see
exactly the same noisy image.

    setting kind   arms
    -------------  ----------------------------------------------------------
    awgn  s        M0   (data AWGN(s), sigma_assumed = s)
    pg    peak,s_r X0   (data AWGN(sigma_eq), fresh draw, sigma_assumed = sigma_eq)
                   MM-A (PG data, sigma_assumed = sigma_eq   [rule A, oracle])
                   MM-B (PG data, sigma_assumed = blind estimate)
                   MM-C (PG data, sigma_assumed = s_r or 1/255)
                   VST  (PG data, GAT -> denoise with sigma = 1 -> inverse)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from noise import (
    add_awgn,
    add_poisson_gaussian,
    alpha_from_peak,
    sigma_blind,
    sigma_equal_variance,
    sigma_read_only,
)
from utils import make_rng

ARMS = ("M0", "X0", "MM-A", "MM-B", "MM-C", "VST")
VST_INVERSES = ("closed_form", "algebraic", "exact")


@dataclass(frozen=True)
class Setting:
    """One noise condition. ``setting_id`` is stable and seeds the noise."""

    kind: str                      # "awgn" or "pg"
    setting_id: str                # e.g. "awgn_s25", "pg_p2_r5"
    sigma: Optional[float] = None  # AWGN std in [0, 1] units
    peak: Optional[float] = None   # PG peak
    sigma_read: Optional[float] = None  # PG read-noise std in [0, 1] units

    @property
    def alpha(self):
        return alpha_from_peak(self.peak)


@dataclass
class Case:
    """One thing to denoise: noisy image + the level the method is told."""

    arm: str            # row label, e.g. "MM-A" or "VST-exact"
    noise_type: str     # "awgn" or "pg"
    sigma_true: float   # AWGN std, or sigma_eq (rule A) for PG data
    peak: float         # NaN when not applicable
    sigma_read: float   # NaN when not applicable
    sigma_assumed: float
    z: np.ndarray       # noisy image (never clipped)
    pipeline: str       # "direct" or "vst"
    alpha: float = float("nan")
    vst_inverse: str = "closed_form"


def _fmt(v):
    return f"{v:g}"


def awgn_settings(cfg):
    """AWGN grid from the config (sigmas in 8-bit units -> [0, 1])."""
    return [
        Setting("awgn", f"awgn_s{_fmt(s)}", sigma=s / 255.0)
        for s in cfg["noise"]["awgn"]["sigmas_255"]
    ]


def pg_settings(cfg):
    """PG grid from the config: peaks x read-noise sigmas."""
    pg = cfg["noise"]["poisson_gaussian"]
    return [
        Setting("pg", f"pg_p{_fmt(p)}_r{_fmt(r)}", peak=float(p), sigma_read=r / 255.0)
        for p in pg["peaks"]
        for r in pg["sigmas_255"]
    ]


def x0_setting_id(setting):
    """Setting id of the X0 control derived from a PG setting: pg_p2_r5 -> x0_p2_r5."""
    return "x0_" + setting.setting_id[len("pg_"):]


def noisy_image(x, image_name, setting, seed, repeat):
    """The noisy image of (image, setting, repeat): one reproducible draw.

    Used for AWGN and PG settings. The generator is
    ``make_rng(seed, image_name, setting_id, repeat)``.
    """
    rng = make_rng(seed, image_name, setting.setting_id, repeat)
    if setting.kind == "awgn":
        return add_awgn(x, setting.sigma, rng)
    return add_poisson_gaussian(x, setting.alpha, setting.sigma_read, rng)


def vst_arm_name(vst_inverse):
    return "VST" if vst_inverse == "closed_form" else f"VST-{vst_inverse}"


def build_cases(x, image_name, setting, arms, seed, repeat, vst_inverse="closed_form"):
    """Cases for one (image, setting, repeat), with the noise drawn once.

    Returns ``(cases, estimates)``. ``estimates`` is a dict with the three
    sigma rules for a PG setting (``None`` for AWGN settings).
    """
    if vst_inverse not in VST_INVERSES:
        raise ValueError(f"vst_inverse must be one of {VST_INVERSES}")
    nan = float("nan")
    nan_pr = dict(peak=nan, sigma_read=nan)
    cases = []

    if setting.kind == "awgn":
        if "M0" in arms:
            z = noisy_image(x, image_name, setting, seed, repeat)
            cases.append(Case("M0", "awgn", setting.sigma, sigma_assumed=setting.sigma,
                              z=z, pipeline="direct", **nan_pr))
        return cases, None

    alpha, sr = setting.alpha, setting.sigma_read
    sigma_eq = sigma_equal_variance(x, alpha, sr)            # rule A, uses clean x
    z_pg = noisy_image(x, image_name, setting, seed, repeat)  # drawn once, shared
    est = {
        "sigma_eq": sigma_eq,
        "sigma_blind": sigma_blind(z_pg),
        "sigma_read_only": sigma_read_only(sr),
    }
    pr = dict(peak=setting.peak, sigma_read=sr)

    if "X0" in arms:
        rng = make_rng(seed, image_name, x0_setting_id(setting), repeat)
        z0 = add_awgn(x, sigma_eq, rng)
        cases.append(Case("X0", "awgn", sigma_eq, sigma_assumed=sigma_eq, z=z0,
                          pipeline="direct", **pr))
    for arm, key in (("MM-A", "sigma_eq"), ("MM-B", "sigma_blind"), ("MM-C", "sigma_read_only")):
        if arm in arms:
            cases.append(Case(arm, "pg", sigma_eq, sigma_assumed=est[key], z=z_pg,
                              pipeline="direct", **pr))
    if "VST" in arms:
        cases.append(Case(vst_arm_name(vst_inverse), "pg", sigma_eq, sigma_assumed=1.0, z=z_pg,
                          pipeline="vst", alpha=alpha, vst_inverse=vst_inverse, **pr))
    return cases, est
