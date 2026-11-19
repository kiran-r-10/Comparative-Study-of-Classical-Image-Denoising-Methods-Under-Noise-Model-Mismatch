# Robustness of Classical Image Denoising Under Noise-Model Mismatch

AWGN vs Poisson-Gaussian noise, evaluated on classical denoisers (mean/Gaussian, median, Wiener, wavelet, NLM, BM3D), with and without the Generalized Anscombe Transform (GAT).

**Team:** Kiran R (2023113002), Roshan K (2023102061). See `WORK_SPLIT.md` for who owns what.

## Setup

```bash
git clone <repo-url> && cd <repo-folder>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put the datasets in `data/` as described in `data/README.md`.

Run the Phase 1 checks:

```bash
python -m pytest                 # statistical tests
jupyter notebook notebooks/01_phase1_noise_and_gat.ipynb
```

## Repo layout

| Folder / file | Owner | Content |
|---|---|---|
| `noise/` | A | AWGN, Poisson-Gaussian, GAT and inverses, diagnostics |
| `utils/` | A | data loader, seeding, config loading |
| `configs/` | A | experiment configuration (YAML) |
| `data/` | A | datasets (not committed, see `data/README.md`) |
| `tests/` | A + B | statistical tests (A), denoiser smoke tests (B) |
| `notebooks/` | both | notebooks that call the modules and show results |
| `denoisers/` | B | denoiser wrappers + parameter policy |
| `metrics/` | B | PSNR, SSIM, method-noise, timing |
| `analysis/` | B | tables, plots, ranking |
| `results/` | both | generated CSVs and figures |
| `report/` | both | final report |
| `run_experiments.py` | A | experiment harness (Phase 2) |

## Conventions (the shared interface contract)

* Images are `float64` NumPy arrays with shape `(H, W)` and values in $[0, 1]$.
* Noise levels are in the same $[0, 1]$ units: "$\sigma = 25$" is passed as `25 / 255`.
* Poisson-Gaussian model: $z = \alpha\,\mathrm{Poisson}(x/\alpha) + \mathcal N(0, \sigma^2)$, and the peak is $1/\alpha$ (`alpha_from_peak(peak)`).
* Randomness: always pass a `numpy.random.Generator`. Use `utils.make_rng(seed, *keys)` so each image / noise setting / repeat has its own reproducible stream.
* **Clipping policy:** noise is not clipped by default (`clip=False`) so that $\mathbb E[z] = x$ and the GAT theory hold. `clip=True` clips to $[0, 1]$.

```python
# Part A provides (import from noise / utils)
load_dataset(name, data_root=None, max_images=None) -> list[(filename, ndarray)]
add_awgn(x, sigma, rng, clip=False) -> z
add_poisson_gaussian(x, alpha, sigma, rng, clip=False) -> z
gat_forward(z, alpha, sigma, g=0.0) -> f
gat_inverse_algebraic(f, alpha, sigma, g=0.0) -> x_hat
gat_inverse_closed_form(f, alpha, sigma, g=0.0) -> x_hat
gat_inverse_exact_unbiased(f, alpha, sigma, g=0.0) -> x_hat   # Phase 2 stretch (stub)
```

## Status

* Phase 1, Part A (noise module, GAT, statistical tests): done, see `notebooks/01_phase1_noise_and_gat.ipynb`.
* Everything else: see `WORK_SPLIT.md`.
