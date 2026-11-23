"""Tests for the experiment harness (arms, parameter adapter, runner, CLI)."""

import csv

import numpy as np
import pytest

import experiments.runner as runner
from denoisers import denoise
from experiments import CSV_COLUMNS, ESTIMATE_COLUMNS, build_cases, noisy_image, params_for, pg_settings
from experiments.arms import awgn_settings
from experiments.runner import apply_preset, estimates_path, run_experiments, vst_pipeline
from noise import sigma_blind, sigma_equal_variance, sigma_read_only
from utils import load_config, load_sample_images
from run_experiments import main

SEED = 7
CFG = {
    "seed": SEED,
    "data_root": "data",
    "noise": {
        "awgn": {"sigmas_255": [25]},
        "poisson_gaussian": {"peaks": [5], "sigmas_255": [5]},
    },
}
ALL_ARMS = ["M0", "X0", "MM-A", "MM-B", "MM-C", "VST"]


@pytest.fixture(scope="module")
def images():
    return load_sample_images(64)[:2]


@pytest.fixture(autouse=True)
def fake_loader(monkeypatch, images):
    monkeypatch.setattr(runner, "load_images", lambda ds, max_images=None, data_root=None: list(images))


def _run(out, methods=("mean", "wiener"), arms=ALL_ARMS, **kw):
    kw.setdefault("overwrite", True)
    return run_experiments(CFG, ["set12"], list(methods), arms, out, progress=False, log=lambda *_: None, **kw)


def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _comparable(rows):
    return [{k: v for k, v in r.items() if k != "time_s"} for r in rows]


# ---------------- schema and content ----------------
def test_schema_and_row_count(tmp_path):
    out = tmp_path / "r.csv"
    _run(out)
    rows = _rows(out)
    assert list(rows[0].keys()) == CSV_COLUMNS
    # per image: M0 x 2 methods + (X0, MM-A, MM-B, MM-C, VST) x 2 methods = 12
    assert len(rows) == 2 * 12
    m0 = [r for r in rows if r["arm"] == "M0"]
    assert all(r["peak"] == "nan" and r["sigma_read"] == "nan" for r in m0)
    assert all(float(r["sigma_assumed"]) == pytest.approx(25 / 255) for r in m0)
    for r in rows:
        assert float(r["time_s"]) >= 0 and np.isfinite(float(r["psnr"])) and 0 <= float(r["ssim"]) <= 1
    vst = [r for r in rows if r["arm"] == "VST"]
    assert vst and all(float(r["sigma_assumed"]) == 1.0 for r in vst)


def test_estimates_csv(tmp_path, images):
    out = tmp_path / "r.csv"
    _run(out)
    est = _rows(estimates_path(out))
    assert list(est[0].keys()) == ESTIMATE_COLUMNS
    assert len(est) == 2                        # 2 images x 1 PG setting
    name, x = images[0]
    row = next(r for r in est if r["image"] == name)
    setting = pg_settings(CFG)[0]
    assert float(row["sigma_eq"]) == pytest.approx(sigma_equal_variance(x, setting.alpha, setting.sigma_read))
    assert float(row["sigma_read_only"]) == pytest.approx(5 / 255)
    z = noisy_image(x, name, setting, SEED, 0)
    assert float(row["sigma_blind"]) == pytest.approx(sigma_blind(z))


def test_same_seed_reproduces_rows_and_other_seed_does_not(tmp_path):
    a, b, c = tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "c.csv"
    _run(a)
    _run(b)
    _run(c, seed=SEED + 1)
    assert _comparable(_rows(a)) == _comparable(_rows(b))
    assert [r["psnr"] for r in _rows(a)] != [r["psnr"] for r in _rows(c)]


# ---------------- resume ----------------
def test_resume_does_not_duplicate(tmp_path):
    out = tmp_path / "r.csv"
    _run(out)
    before = _rows(out)
    s = _run(out, resume=True, overwrite=False)
    assert s["written"] == 0
    assert _rows(out) == before
    assert len(_rows(estimates_path(out))) == 2


def test_resume_adds_only_missing_rows(tmp_path):
    out = tmp_path / "r.csv"
    _run(out, methods=("mean",))
    n1 = len(_rows(out))
    _run(out, methods=("mean", "wiener"), resume=True, overwrite=False)
    rows = _rows(out)
    assert len(rows) == 2 * n1
    keys = [runner.row_key(r) for r in rows]
    assert len(set(keys)) == len(keys)


def test_existing_csv_without_resume_is_refused(tmp_path):
    out = tmp_path / "r.csv"
    _run(out)
    with pytest.raises(FileExistsError):
        _run(out, overwrite=False)


# ---------------- pairing and arms ----------------
def test_mm_arms_and_vst_share_the_same_noisy_image(images):
    name, x = images[0]
    setting = pg_settings(CFG)[0]
    cases, est = build_cases(x, name, setting, ALL_ARMS, SEED, 0)
    by = {c.arm: c for c in cases}
    z = noisy_image(x, name, setting, SEED, 0)
    for arm in ("MM-A", "MM-B", "MM-C", "VST"):
        assert by[arm].z is by["MM-A"].z            # one draw, reused
        assert np.array_equal(by[arm].z, z)         # and it is the reproducible draw
    assert not np.array_equal(by["X0"].z, z)        # X0 is an independent draw
    assert by["MM-A"].sigma_assumed == pytest.approx(est["sigma_eq"])
    assert by["MM-B"].sigma_assumed == pytest.approx(sigma_blind(z))
    assert by["MM-C"].sigma_assumed == pytest.approx(sigma_read_only(setting.sigma_read))
    assert by["VST"].sigma_assumed == 1.0


def test_x0_uses_sigma_eq_per_image(images):
    setting = pg_settings(CFG)[0]
    sig = []
    for name, x in images:
        cases, est = build_cases(x, name, setting, ["X0"], SEED, 0)
        c = cases[0]
        assert c.arm == "X0" and c.noise_type == "awgn"
        assert c.sigma_true == c.sigma_assumed == pytest.approx(sigma_equal_variance(x, setting.alpha, setting.sigma_read))
        assert np.std(c.z - x) == pytest.approx(c.sigma_true, rel=0.05)
        sig.append(c.sigma_true)
    assert sig[0] != sig[1]                         # per image, not on the 15/25/50 grid


def test_setting_ids_are_stable_strings():
    assert [s.setting_id for s in awgn_settings(CFG)] == ["awgn_s25"]
    assert [s.setting_id for s in pg_settings(CFG)] == ["pg_p5_r5"]


def test_vst_ablation_renames_arm(tmp_path):
    out = tmp_path / "r.csv"
    _run(out, arms=["VST"], vst_inverse="exact")
    assert {r["arm"] for r in _rows(out)} == {"VST-exact"}


def test_vst_pipeline_with_identity_denoiser_and_algebraic_inverse_returns_clipped_z(monkeypatch, images):
    name, x = images[0]
    setting = pg_settings(CFG)[0]
    z = noisy_image(x, name, setting, SEED, 0)
    seen = {}

    def identity(method, f, params):
        seen.update(params, f_min=f.min(), f_max=f.max())
        return f

    monkeypatch.setattr(runner, "denoise", identity)
    out = vst_pipeline("mean", z, setting.alpha, setting.sigma_read, "algebraic")
    assert seen["clip"] is False                    # GAT domain must not be clipped
    assert out.min() >= 0 and out.max() <= 1
    ok = setting.alpha * z + 0.375 * setting.alpha**2 + setting.sigma_read**2 > 0
    assert np.allclose(out[ok], np.clip(z, 0, 1)[ok], atol=1e-9)


# ---------------- unimplemented methods ----------------
def test_not_implemented_method_is_skipped_not_fatal(tmp_path, monkeypatch):
    real = denoise

    def fake(method, z, params=None):
        if method == "wavelet":
            raise NotImplementedError("wavelet")
        return real(method, z, params)

    monkeypatch.setattr(runner, "denoise", fake)
    msgs = []
    out = tmp_path / "r.csv"
    s = run_experiments(CFG, ["set12"], ["mean", "wavelet"], ALL_ARMS, out, progress=False,
                        overwrite=True, log=msgs.append)
    assert {r["method"] for r in _rows(out)} == {"mean"}
    assert s["skipped_methods"] == {"wavelet"}
    assert sum("wavelet" in m for m in msgs) == 1   # one warning, not one per run


def test_real_unimplemented_methods_do_not_crash(tmp_path):
    out = tmp_path / "r.csv"
    run_experiments(CFG, ["set12"], ["mean", "bm3d"], ["M0"], out, progress=False, overwrite=True,
                    log=lambda *_: None)
    assert "mean" in {r["method"] for r in _rows(out)}


# ---------------- parameter adapter ----------------
@pytest.mark.parametrize("domain, rng", [("image", 1.0), ("gat", 2.5)])
@pytest.mark.parametrize("method", ["mean", "gaussian", "median", "wiener"])
@pytest.mark.parametrize("sigma", [5 / 255, 25 / 255, 1.0])
def test_params_are_accepted_by_denoise(method, sigma, domain, rng):
    p = params_for(method, sigma, domain, data_range=rng)
    out = denoise(method, np.random.default_rng(0).random((16, 16)), {**p, "clip": False})
    assert out.shape == (16, 16)


def test_params_rules_are_monotone_and_domain_checked():
    sizes = [params_for("median", s / 255, "image")["size"] for s in (15, 25, 50)]
    assert sizes == sorted(sizes) and all(s % 2 == 1 for s in sizes)
    g = [params_for("gaussian", s / 255)["filter_sigma"] for s in (15, 25, 50)]
    assert g == sorted(g)
    assert params_for("wiener", 0.1)["noise_sigma"] == 0.1
    with pytest.raises(ValueError):
        params_for("mean", 0.1, "fourier")


# ---------------- CLI and config ----------------
def test_cli_dry_run_counts_runs(capsys):
    cfg = load_config()
    assert main(["--preset", "quick", "--methods", "mean", "median", "--dry-run"]) == 0
    txt = capsys.readouterr().out
    q = apply_preset(cfg, "quick")
    n_pg = len(q["noise"]["poisson_gaussian"]["peaks"]) * len(q["noise"]["poisson_gaussian"]["sigmas_255"])
    per_image = (len(q["noise"]["awgn"]["sigmas_255"]) + 5 * n_pg) * 2
    assert f"{per_image * 2} runs planned" in txt      # the fake loader returns 2 images


def test_cli_quick_run_writes_csv(tmp_path):
    out = tmp_path / "q.csv"
    assert main(["--preset", "quick", "--methods", "mean", "wiener", "--out", str(out)]) == 0
    assert len(_rows(out)) > 0
    assert estimates_path(out).exists()


def test_config_has_documented_phase2_keys():
    cfg = load_config()
    for k in ("arms", "repeats", "vst_inverse", "results_dir", "quick", "seed", "noise"):
        assert k in cfg
    assert cfg["repeats"] == 1 and cfg["vst_inverse"] == "closed_form"
    assert apply_preset(cfg, None) is cfg
    with pytest.raises(ValueError):
        apply_preset(cfg, "nope")
