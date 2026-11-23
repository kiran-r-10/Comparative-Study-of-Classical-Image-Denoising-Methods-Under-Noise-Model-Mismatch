# Progress Tracker

Checklist mirrors `WORK_SPLIT.md`. Update the status and the "Last updated" line whenever something changes.
Legend: `[x]` done and verified, `[~]` partial / needs follow-up, `[ ]` not started.

**Last updated:** 2026-11-23 · **Tests:** 183 passed (`python3 -m pytest -q`; 179 passed + 4 skipped when `data/set12` is absent)

## Summary

| Area | Phase 1 | Phase 2 |
|---|---|---|
| Part A (Kiran): data, noise, GAT, harness | Done | Done for the 4 current methods; rerun with `--resume` when wavelet/NLM/BM3D land |
| Part B (Roshan): denoisers, metrics, analysis | Done | Not started |
| Shared: integration test, report, README | Integration test done | Not started |

## Part A: Data, Noise & Experiment Engine

### Phase 1

- [x] Repo skeleton, `requirements.txt`, seeds, config format: `utils/seed.py` (`make_rng`), `utils/config.py`, `configs/default.yaml`, `pytest.ini`, `.gitignore`
- [x] `load_dataset`: `utils/data_loader.py` (grayscale, [0,1], optional `max_images`), plus `load_sample_images` fallback. Tests in `tests/test_data_loader.py`
- [x] AWGN generator: `noise/awgn.py`
- [x] Poisson-Gaussian generator with documented clipping policy (no clipping by default, `clip=True` optional): `noise/poisson_gaussian.py`, `alpha_from_peak`
- [x] Forward GAT: `noise/gat.py::gat_forward`
- [x] Algebraic inverse and closed-form inverse: `gat_inverse_algebraic`, `gat_inverse_closed_form`
- [x] Statistical tests (`tests/test_noise_stats.py`, `tests/test_gat.py`, helpers in `noise/diagnostics.py`): AWGN mean/std, PG mean/variance on flat patches, GAT variance ≈ 1, inverse round trip, inverse bias curve
- [x] Phase 1 notebook: `notebooks/01_phase1_noise_and_gat.ipynb`
- [x] Notebook re-run on real Set12 with outputs saved (13/13 code cells, no errors)
- [x] Real datasets in `data/` and loading correctly: set12 (12), kodak24 (24, 500x500 not 768x512, FFDNet crop), bsd68 (68)

### Phase 2

- [x] Exact unbiased inverse GAT: `noise/gat.py::gat_inverse_exact_unbiased` (table of E[f|lambda] + interpolation, cached). Tests in `tests/test_gat.py`. `inverse_bias_curve` in `noise/diagnostics.py` now also returns the `exact` key
- [x] Experiment harness `run_experiments.py` + `experiments/` (arms M0/X0/MM-A/B/C/VST, paired noise, config-driven, resumable CSV). Tests in `tests/test_experiments.py`. `experiments/params.py` uses TEMPORARY parameter rules until Part B's policy is final
- [x] Sigma rules A/B/C: `noise/sigma.py`, tests in `tests/test_sigma.py` (needs `PyWavelets` for `estimate_sigma`, `tqdm` for the progress bar: added to `requirements.txt`)
- [~] Noise grid: draft exists in `configs/default.yaml` (AWGN σ ∈ {15,25,50}/255; PG peak ∈ {1,2,5,10,30}, σ ∈ {0,5,10}/255). Needs final confirmation
- [~] Mismatch protocol: `report/mismatch_protocol.md`. Decided: A headline + B practical view, sigma-dependent mean/gaussian/median, 1 repeat, VST-on-AWGN is future work. Waiting on Roshan to confirm the filter-size rule
- [~] Extra repeats (WHEN TIME PERMITS, do not forget): 3 repeats on Set12 only to show variance is small. First time one BM3D and one NLM call on a Kodak image to check the budget
- [x] Noise-level estimation runs: `results/phase2_results_estimates.csv` (1,560 rows)
- [x] Full benchmark on Set12, Kodak24, BSD68 with the 4 implemented methods: `results/phase2_results.csv` (32,448 rows, 104 images, all 6 arms, no duplicates or NaNs) and `results/phase2_results_estimates.csv`. Wavelet, NLM, BM3D still to be added with `--resume` once Part B lands them
- [x] Presentation notebook with saved outputs: `notebooks/03_phase2_experiments.ipynb`

## Part B: Denoisers, Evaluation & Analysis

Implemented by Roshan (commits of 2026-11-23). Contract: `denoise(method, z, params)` in `denoisers/api.py`.

### Phase 1

- [x] Mean, Gaussian, median and Wiener filters: `denoisers/spatial.py`, entry point `denoisers/api.py::denoise`
- [x] Metrics: PSNR, SSIM, runtime, method noise (residual only so far): `metrics/`
- [x] End-to-end run on Set12 with sanity-check table: `python -m analysis.sanity_check`, `results/phase1_sanity_*`, `notebooks/02_phase1_denoisers_and_metrics.ipynb`

### Phase 2

- [ ] Wavelet soft-thresholding
- [ ] Non-Local Means
- [ ] BM3D
- [ ] Two pipelines per method: direct and VST-mediated (needs Part A's GAT functions, which are ready)
- [ ] Parameter policy per method, documented
- [ ] Method-noise (plus energy and structure statistics)
- [ ] Tables and plots
- [ ] Robustness ranking and decision guide

## Shared

- [x] Integration test (Phase 1 scope): `tests/test_integration.py`, one Set12 image, four filters, sigma=25. Extend to all 7 methods + VST in Phase 2
- [ ] Report: Intro, Methodology, Noise models, GAT (A); Denoisers, Metrics, Results, Decision guide (B); Conclusion and cross-review (both). `report/` is empty
- [x] README updated by Roshan for Part B; `WORK_SPLIT.md` is now in the repo root
- [ ] Reproducibility check (fresh clone runs end to end)

## Notes / open decisions

- Interface contract is implemented for everything Part A promised, so Part B is unblocked and can import `noise` and `utils` today.
- Contract and README say `load_dataset(name)`. The actual signature is `load_dataset(name, data_root=None, max_images=None)`, which is compatible.
- Roshan's Phase 1 Wiener rows in `results/phase1_sanity_table.md` use the true sigma, i.e. the matched case. Mismatch results come only from the Phase 2 harness.
- Next blocker for Part B Phase 2 is the mismatch protocol (which sigma each denoiser receives).
