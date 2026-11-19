# Progress Tracker

Checklist mirrors `WORK_SPLIT.md`. Update the status and the "Last updated" line whenever something changes.
Legend: `[x]` done and verified, `[~]` partial / needs follow-up, `[ ]` not started.

**Last updated:** 2026-11-20 · **Tests:** 65 passed (`python3 -m pytest -q`)

## Summary

| Area | Phase 1 | Phase 2 |
|---|---|---|
| Part A (Kiran): data, noise, GAT, harness | Done | Not started |
| Part B (Roshan): denoisers, metrics, analysis | Not started (folders are empty) | Not started |
| Shared: integration test, report, README | Not started | Not started |

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

- [ ] Exact unbiased inverse GAT: `gat_inverse_exact_unbiased` is a stub that raises `NotImplementedError` (stretch goal; otherwise the report says closed-form is used)
- [ ] Experiment harness `run_experiments.py` (grid, matched and mismatched sweeps, multiple seeds, config-driven, CSV output)
- [~] Noise grid: draft exists in `configs/default.yaml` (AWGN σ ∈ {15,25,50}/255; PG peak ∈ {1,2,5,10,30}, σ ∈ {0,5,10}/255). Needs final confirmation
- [ ] Mismatch protocol: define exactly which `sigma` is passed to each denoiser in matched vs mismatched runs, and write it down for the report
- [ ] Noise-level estimation runs (`skimage.restoration.estimate_sigma`)
- [ ] Full benchmark on Set12, Kodak24, BSD68

## Part B: Denoisers, Evaluation & Analysis

All of `denoisers/`, `metrics/`, `analysis/` currently contain only `.gitkeep`.

### Phase 1

- [ ] Mean and Gaussian filters
- [ ] Median filter
- [ ] Wiener filter
- [ ] Metrics: PSNR, SSIM, runtime
- [ ] End-to-end run on sample Set12 images with sanity-check table

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

- [ ] Integration test: one image, all methods, one noise setting (blocked on Part B Phase 1)
- [ ] Report: Intro, Methodology, Noise models, GAT (A); Denoisers, Metrics, Results, Decision guide (B); Conclusion and cross-review (both). `report/` is empty
- [~] README: written for Phase 1 and the Part A interface. References `WORK_SPLIT.md`, which currently lives one level up (outside this folder). Copy it into the repo root
- [ ] Reproducibility check (fresh clone runs end to end)

## Notes / open decisions

- Interface contract is implemented for everything Part A promised, so Part B is unblocked and can import `noise` and `utils` today.
- Contract and README say `load_dataset(name)`. The actual signature is `load_dataset(name, data_root=None, max_images=None)`, which is compatible.
- The folder is currently a zip-export copy (`denoising-mismatch-phase1/`). Decide which location is the real repo before Part B starts adding code.
