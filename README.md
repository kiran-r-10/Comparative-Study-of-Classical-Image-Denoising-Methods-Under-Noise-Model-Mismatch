# Robustness of Classical Image Denoising Under Noise-Model Mismatch

AWGN vs Poisson-Gaussian noise, evaluated on classical denoisers (mean/Gaussian, median, Wiener, wavelet, NLM, BM3D), with and without the Generalized Anscombe Transform (GAT).

**Team:** Kiran R (2023113002), Roshan K (2023102061). See `WORK_SPLIT.md` for who owns what and `PROGRESS.md` for the current status.

## Setup

```bash
git clone <repo-url> && cd <repo-folder>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put the datasets in `data/` as described in `data/README.md` (Set12, Kodak24, BSD68, one folder each).

## Run the Phase 1 checks

```bash
python -m pytest                          # all tests (noise, GAT, loader, denoisers, metrics, integration)
python -m analysis.sanity_check           # Part B sanity table on all of Set12, AWGN sigma = 15/25/50
python -m analysis.sanity_check --max-images 4   # quick run
jupyter notebook notebooks/01_phase1_noise_and_gat.ipynb          # Part A: noise + GAT
jupyter notebook notebooks/02_phase1_denoisers_and_metrics.ipynb  # Part B: denoisers + metrics
```

The sanity script writes `results/phase1_sanity_per_image.csv`, `results/phase1_sanity_summary.csv` and `results/phase1_sanity_table.md`. The integration test (`tests/test_integration.py`) is skipped automatically if Set12 is not in `data/`.

## Running experiments

The harness (`run_experiments.py`, package `experiments/`) implements the arms of `report/mismatch_protocol.md`: `M0`, `X0`, `MM-A`, `MM-B`, `MM-C`, `VST`. Noise is drawn once per (image, noise setting, repeat) and shared by all methods and by MM-A/B/C/VST, so comparisons are paired. Methods that are not implemented yet (`wavelet`, `nlm`, `bm3d`) are skipped with a one-line warning and picked up automatically once Part B lands them.

Quick check (Set12, first 3 images, the four implemented methods, default noise grid):

```bash
python run_experiments.py --datasets set12 --methods mean gaussian median wiener \
    --arms M0 X0 MM-A MM-B MM-C VST --max-images 3 --repeats 1 --out results/phase2_quick.csv
python run_experiments.py --preset quick --out results/phase2_quick.csv --overwrite   # smaller grid, same idea
python run_experiments.py --preset quick --dry-run                                     # only count the runs
```

Full benchmark (all datasets, all methods, all arms; resumable):

```bash
python run_experiments.py --datasets set12 kodak24 bsd68 \
    --methods mean gaussian median wiener wavelet nlm bm3d \
    --arms M0 X0 MM-A MM-B MM-C VST --repeats 1 --out results/phase2_results.csv --resume
```

Options: `--vst-inverse {closed_form,algebraic,exact}` (anything but `closed_form` renames the arm `VST-<inverse>`), `--seed`, `--config`, `--overwrite`. An existing output CSV is never replaced silently: pass `--resume` (skip rows already there) or `--overwrite`. The CSV is appended and flushed after every noise setting, so a crash loses at most one setting.

Outputs: `<out>` has one row per (dataset, image, method, arm, noise setting, repeat) with columns `dataset, image, method, arm, noise_type, sigma_true, peak, sigma_read, sigma_assumed, repeat, seed, psnr, ssim, time_s, noisy_psnr` (`NaN` where a field does not apply); `<out>_estimates.csv` has the per-image sigma estimates (`sigma_eq`, `sigma_blind`, `sigma_read_only`) of each PG setting. How `sigma_assumed` becomes denoiser parameters lives in one function, `experiments/params.py::params_for`, whose rules are **temporary** until Part B's parameter policy is final.

## Repo layout

| Folder / file | Owner | Content |
|---|---|---|
| `noise/` | A | AWGN, Poisson-Gaussian, GAT and inverses, diagnostics |
| `utils/` | A | data loader, seeding, config loading |
| `configs/` | A | experiment configuration (YAML) |
| `data/` | A | datasets (not committed, see `data/README.md`) |
| `denoisers/` | B | `api.py` (`denoise` entry point, default parameters), `spatial.py` (mean, Gaussian, median, Wiener) |
| `metrics/` | B | `quality.py` (PSNR, SSIM), `method_noise.py`, `timing.py` |
| `analysis/` | B | `sanity_check.py` (Phase 1); tables, plots and ranking come in Phase 2 |
| `tests/` | A + B | statistical tests (A), denoiser, metric, sanity-script and integration tests (B) |
| `notebooks/` | both | notebooks that call the modules and show results |
| `results/` | both | generated CSVs and figures |
| `report/` | both | final report |
| `experiments/` | A | arms, `sigma_assumed` -> parameters adapter (`params.py`), runner (Phase 2) |
| `run_experiments.py` | A | experiment harness CLI (Phase 2) |

## Conventions (the shared interface contract)

* Images are `float64` NumPy arrays with shape `(H, W)` and values in $[0, 1]$.
* Noise levels are in the same $[0, 1]$ units: "$\sigma = 25$" is passed as `25 / 255`.
* Poisson-Gaussian model: $z = \alpha\,\mathrm{Poisson}(x/\alpha) + \mathcal N(0, \sigma^2)$, and the peak is $1/\alpha$ (`alpha_from_peak(peak)`).
* Randomness: always pass a `numpy.random.Generator`. Use `utils.make_rng(seed, *keys)` so each image / noise setting / repeat has its own reproducible stream.
* **Noise clipping policy:** noise is not clipped by default (`clip=False`) so that $\mathbb E[z] = x$ and the GAT theory hold. `clip=True` clips to $[0, 1]$.

```python
# Part A provides (import from noise / utils)
load_dataset(name, data_root=None, max_images=None) -> list[(filename, ndarray)]
add_awgn(x, sigma, rng, clip=False) -> z
add_poisson_gaussian(x, alpha, sigma, rng, clip=False) -> z
gat_forward(z, alpha, sigma, g=0.0) -> f
gat_inverse_algebraic(f, alpha, sigma, g=0.0) -> x_hat
gat_inverse_closed_form(f, alpha, sigma, g=0.0) -> x_hat
gat_inverse_exact_unbiased(f, alpha, sigma, g=0.0) -> x_hat   # numerical exact unbiased inverse
sigma_equal_variance(x, alpha, sigma_read) -> float            # protocol rule A (uses clean x)
sigma_blind(z) -> float                                        # rule B (estimate_sigma)
sigma_read_only(sigma_read, floor=1/255) -> float              # rule C

# Part B provides (import from denoisers / metrics)
denoise(method, z, params=None) -> x_hat   # method in {"mean","gaussian","median","wiener"} now;
                                           # "wavelet","nlm","bm3d" are registered but raise NotImplementedError (Phase 2)
psnr(x, x_hat), ssim(x, x_hat)             # data_range = 1.0
method_noise(z, x_hat)                     # residual z - x_hat
timed(fn, *args, **kwargs) -> (result, seconds)
```

## Denoisers (Part B, Phase 1)

```python
from noise import add_awgn
from denoisers import denoise
from metrics import psnr, ssim, timed
from utils.seed import make_rng

z = add_awgn(x, 25 / 255, make_rng(2026, "img.png", 25))
x_hat, seconds = timed(denoise, "wiener", z, {"size": 3, "noise_sigma": 25 / 255})
print(psnr(x, x_hat), ssim(x, x_hat))
```

Default parameters (a caller overrides only what it needs; unknown keys raise `ValueError`, so a typo never silently runs with defaults):

| Method | Parameters (defaults) | Notes |
|---|---|---|
| `mean` | `size=3`, `mode="reflect"` | box filter, `scipy.ndimage.uniform_filter` |
| `gaussian` | `filter_sigma=1.0`, `mode="reflect"` | kernel std in **pixels**, not the noise level |
| `median` | `size=3`, `mode="reflect"` | `scipy.ndimage.median_filter` |
| `wiener` | `size=3`, `noise_sigma=None` | `scipy.signal.wiener`; `noise_sigma` is the noise std the filter *believes* (in $[0,1]$ units), `None` lets scipy estimate it. This is the knob the Phase 2 mismatch protocol will control. |

Window sizes must be odd positive integers. Policies that apply to every method:

* **Borders:** all filters use reflect padding. `scipy.signal.wiener` zero-pads internally, so `wiener_filter` reflect-pads and crops itself to avoid darkened borders.
* **Output clipping:** `denoise` clips the result to $[0, 1]$ by default (`clip=True`). The Phase 2 VST pipeline must pass `clip=False` for the inner call (data are in the GAT domain) and clip only after the inverse GAT.
* **Metrics:** PSNR and SSIM use `skimage.metrics` with `data_range=1.0` and skimage's default SSIM window. Runtime uses `time.perf_counter`.

## Preliminary results (Phase 1 sanity check)

Set12 (12 images), matched AWGN, mean over images. Parameters are the **fixed, untuned defaults** above, and Wiener is given the true $\sigma$, so this table is a sanity check and **not** a final method ranking. Mismatch and Poisson-Gaussian results come in Phase 2.

| $\sigma$ (8-bit) | noisy | mean | gaussian | median | wiener |
|---|---|---|---|---|---|
| 15 | 24.61 | 27.52 | 27.67 | 27.80 | **29.52** |
| 25 | 20.17 | 26.03 | **26.51** | 25.45 | 26.16 |
| 50 | 14.15 | 22.41 | **23.37** | 21.07 | 21.36 |

PSNR in dB. The noisy-input PSNR matches the theoretical $-20\log_{10}(\sigma/255)$ at every level.

## Status

* Phase 1, Part A (noise module, GAT, statistical tests): done, see `notebooks/01_phase1_noise_and_gat.ipynb`.
* Phase 1, Part B (mean, Gaussian, median and Wiener filters, PSNR/SSIM/method-noise/timing, Set12 sanity run, integration test): done, see `notebooks/02_phase1_denoisers_and_metrics.ipynb`.
* Phase 2, Part A: sigma rules A/B/C, exact unbiased inverse GAT and the experiment harness are implemented and tested (see "Running experiments"); the full benchmark has not been run yet.
* Phase 2, Part B (not started): wavelet, NLM and BM3D; parameter policy; method-noise statistics, tables, plots, robustness ranking and decision guide.
* Details and open items: `PROGRESS.md` and `WORK_SPLIT.md`.
