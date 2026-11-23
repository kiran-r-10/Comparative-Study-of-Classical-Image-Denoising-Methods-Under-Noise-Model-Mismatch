# Mismatch protocol

Goal: define exactly what "a method tuned for AWGN, tested on Poisson-Gaussian" means, in terms of the noise level each denoiser is given. Everything below is fixed by this document, so that the harness (`run_experiments.py`) and the denoiser parameter policy (`denoisers/`) cannot drift apart.

All images are float64 in [0, 1]. All sigmas are in [0, 1] units (`25` means `25/255`). Noise is never clipped; denoiser output is clipped to [0, 1] once, at the very end, before PSNR/SSIM.

## 1. Noise conditions

| Name | Generation | Grid (from `configs/default.yaml`) |
|---|---|---|
| **AWGN(s)** | `z = x + N(0, s^2)` | s in {15, 25, 50}/255 |
| **PG(peak, s_r)** | `z = alpha * Poisson(x/alpha) + N(0, s_r^2)`, `alpha = 1/peak` | peak in {1, 2, 5, 10, 30}, s_r in {0, 5, 10}/255 |

Reproducibility: one generator per (dataset, image, noise setting, repeat) via `make_rng(seed, image_name, setting_id, repeat)`. Default: 1 repeat per setting on all datasets (optional extra: 3 repeats on Set12 only, see section 7).

## 2. The key quantity: assumed sigma

An AWGN-designed method needs one number, `sigma_assumed`. On AWGN data that number is the true sigma. On PG data the noise std varies per pixel (`std(z|x) = sqrt(alpha*x + s_r^2)`), so a single number is an approximation. We fix three definitions, all evaluated, with **(A) as the headline** (decided) and (B) reported alongside it as the practical view:

| Id | `sigma_assumed` on PG data | Meaning |
|---|---|---|
| **A: equal-variance (oracle)** | `sigma_eq = sqrt(alpha * mean(x) + s_r^2)` | Global noise std that matches the average noise variance of the image. Best-case single-sigma AWGN assumption, so any remaining loss is due to the *shape* of the noise (signal dependence), not a wrong level. |
| **B: blind estimate** | `skimage.restoration.estimate_sigma(z)` | What a practitioner would do with no knowledge of the noise. Computed from the noisy image only. |
| **C: read-noise only (naive)** | `s_r` (if `s_r == 0`, use `1/255`) | Lower bound: user knows only the sensor's Gaussian specification and ignores shot noise. |

**Why A is the headline:** our research question is noise-*model* mismatch, not noise-level estimation. Giving the AWGN methods the best possible average level isolates the effect of the noise shape; a method that still degrades cannot blame a bad sigma guess. **B** is reported in the main results as the realistic deployment case (it mixes model mismatch with estimation error). **C** is a lower bound.

`mean(x)` in (A) uses the clean image, which is why it is an oracle and is not available in practice. It is used only to set the level; the denoiser never sees `x`.

## 3. Experimental arms

For each image, noise setting and repeat:

| Arm | Data | Denoiser configuration | Purpose |
|---|---|---|---|
| **M0 matched** | AWGN(s) | direct, `sigma_assumed = s` | Baseline; should match published AWGN numbers (Set12, BSD68) |
| **X0 equal-variance control** | AWGN(`sigma_eq`) with `sigma_eq` from the PG setting (A) | direct, `sigma_assumed = sigma_eq` | Same total noise variance as the PG case but Gaussian. Separates "noisier" from "noise-model mismatch" |
| **MM-A / MM-B / MM-C** | PG(peak, s_r) | direct, `sigma_assumed` from A / B / C | The mismatch condition |
| **VST** | PG(peak, s_r) | GAT with the true `alpha, s_r` -> denoise -> inverse GAT (see 4) | Model-aware pipeline |

Headline quantities:

* **Mismatch penalty** = `PSNR(X0) - PSNR(MM-A)` (and the same for SSIM). Positive means the method suffers from signal-dependent noise even when told the right average level.
* **VST gain** = `PSNR(VST) - PSNR(MM-A)`.
* **Robustness rank**: order methods by mean mismatch penalty and by worst-case penalty over the PG grid.

Note that `x` mean differs per image, so X0 draws a fresh AWGN level per image; it is not on the 15/25/50 grid.

## 4. VST pipeline (fixed definition)

1. `f = gat_forward(z, alpha, s_r)`. Its noise std is about 1 everywhere, by design.
2. `f_hat = denoise(method, f, {..., sigma = 1.0, clip: False})`. The denoiser is told `sigma = 1`, not `sigma_eq`, because the stabilised data have unit noise std. `clip=False` is required, since `f` is not in [0, 1] (see `denoisers/api.py`).
3. `x_hat = gat_inverse_closed_form(f_hat, alpha, s_r)`, then clip to [0, 1].
4. Ablation (cheap, run on Set12 only): replace step 3 with `gat_inverse_algebraic` to show the bias at low peak. If the exact unbiased inverse gets implemented, add it as a third inverse.

Image-scale-dependent denoiser parameters (e.g. NLM `h`, BM3D `sigma_psd`) must be derived from the `sigma` they are given, not from fixed pixel values, so the same parameter policy works in both the [0, 1] domain and the GAT domain.

## 5. What each method receives

The protocol fixes only `sigma_assumed`. The mapping from sigma to concrete parameters belongs to the parameter policy owned by Part B (`denoisers/`) and is documented there.

| Method | Uses sigma through |
|---|---|
| mean, gaussian, median | window / kernel size is a function of sigma (decided, section 7) |
| wiener | `noise_sigma` |
| wavelet | threshold (VisuShrink or BayesShrink) from sigma |
| nlm | `h` proportional to sigma; patch parameters fixed |
| bm3d | `sigma_psd` |

The same policy function must be used in all arms. No arm may tune parameters on its own noisy data.

## 6. Metrics and outputs

* PSNR and SSIM against the clean image (`data_range=1`), runtime per call, method-noise statistics.
* One row per (dataset, image, method, arm, noise setting, repeat) in `results/*.csv` with columns: `dataset, image, method, arm, noise_type, sigma_true or (peak, s_r), sigma_assumed, repeat, psnr, ssim, time_s`.
* Aggregation (Part B `analysis/`): mean and std over images and repeats; ranking by mismatch penalty.

## 7. Decisions

1. **Headline arm: decided.** A is the headline, B is run and reported as the practical view.
2. **Mean / Gaussian / median: decided, sigma-dependent.** Their window or kernel size is a function of `sigma_assumed` (for example Gaussian `filter_sigma` growing with sigma, median window of 3, 5 or 7 chosen by sigma), so they can react to a wrong sigma like every other method. It is set once on the matched AWGN case (M0) and then frozen for all arms. 
3. **Repeats: decided, 1 repeat everywhere** with fixed seeds. Optional, when time permits: 3 repeats on Set12 only, to show the variance is small. Before choosing anything bigger, time one BM3D and one NLM call on a Kodak image. Tracked in `PROGRESS.md`.
4. **VST applied to AWGN data: future work.** Out of scope for this project; mention in the report's limitations section.
