# Work Split: Classical Denoising under Noise-Model Mismatch

**Project:** Robustness of Classical Image Denoising Methods Under Noise-Model Mismatch (AWGN vs Poisson-Gaussian)
**Team:** Kiran R (2023113002), Roshan K (2023102061)

The project is split into two halves with a clean interface between them, so both of us can work in parallel from day one.

- **Part A: Data, Noise & Experiment Engine** (the "input side")
- **Part B: Denoisers, Evaluation & Analysis** (the "output side")

Names are not hard-assigned. Swap A and B if you prefer. A is slightly heavier on math and infrastructure, B is slightly heavier on tuning and analysis.

---

## 1. Dataset choice

| Dataset | Size | Use in this project | Why |
|---|---|---|---|
| **Set12** | 12 grayscale, 256x256 | Phase 1 sanity checks, debugging, quick runs | Tiny, so even NLM and BM3D run in seconds. The standard denoising sanity set, so BM3D numbers are easy to compare with literature. |
| **Kodak24** | 24 colour, 768x512 (convert to grayscale) | Main benchmark | Clean, uncompressed, high-quality natural images with diverse content (textures, edges, flat regions). Standard in denoising papers. Small enough to run the full parameter grid. |
| **BSD68** | 68 grayscale, ~481x321 | Final confirmation run on a subset or the whole set | The most common AWGN benchmark in the literature. Lets us say our matched-AWGN results agree with published numbers, and adds image diversity so conclusions are not tied to 24 images. |

**Why grayscale:** the Poisson-Gaussian noise model and the GAT are defined per channel. Grayscale removes colour-handling as a confound, so the experiment isolates noise-model mismatch, which is the actual research question. Colour (CBSD68) can be listed as future work.

**Why not one dataset only:** Set12 is too small for solid conclusions, BSD68 is too slow for fast iteration. Using all three gives a speed to credibility ladder.

**Important:** the clean images are the ground truth. We add all noise ourselves, so we never use datasets that already contain real noise (e.g. SIDD, DND). Those would break the controlled-mismatch design.

---

## 2. Shared interface contract (agree on this first, 30 minutes)

So that A and B can work independently, fix these signatures now. All images are `float64` NumPy arrays in $[0,1]$, shape `(H, W)`.

```python
# --- Part A provides ---
load_dataset(name: str) -> list[tuple[str, np.ndarray]]        # name in {"set12","kodak24","bsd68"}
add_awgn(x, sigma, rng) -> z
add_poisson_gaussian(x, alpha, sigma, rng) -> z
gat_forward(z, alpha, sigma, g=0.0) -> f
gat_inverse_algebraic(f, alpha, sigma, g=0.0) -> x_hat
gat_inverse_closed_form(f, alpha, sigma, g=0.0) -> x_hat
gat_inverse_exact_unbiased(f, alpha, sigma, g=0.0) -> x_hat    # stretch goal

# --- Part B provides ---
denoise(method: str, z, params: dict) -> x_hat   # method in {"mean","gaussian","median","wiener","wavelet","nlm","bm3d"}
psnr(x, x_hat), ssim(x, x_hat), method_noise(z, x_hat), timed(fn, *args)
```

The GAT is defined as

$$
f(z) = \frac{2}{\alpha}\sqrt{\alpha z + \tfrac{3}{8}\alpha^2 + \sigma^2}
$$

and the model is $z = \alpha\,\mathrm{Poisson}(x/\alpha) + n,\; n \sim \mathcal{N}(0,\sigma^2)$, so $\mathbb{E}[z]=x$ and $\mathrm{Var}(z)=\alpha x+\sigma^2$.

(Use the exact formulation from Makitalo and Foi 2013 [1] when coding. The formula above is the zero-offset form.)

Until A is done, B can use a throwaway AWGN stub (`x + sigma*rng.normal(size=x.shape)`) to test denoisers.

---

## 3. Part A: Data, Noise and Experiment Engine

**Owner: Kiran R (suggested)**

### Phase 1
- [ ] Repo skeleton, `requirements.txt`, seeds and config file format (YAML or JSON)
- [ ] `load_dataset`: image loading with `PIL.Image.open` + `numpy.asarray`, grayscale conversion, scaling to $[0,1]$
- [ ] AWGN generator (`numpy.random.default_rng`, `Generator.normal`)
- [ ] Poisson-Gaussian generator (`Generator.poisson`, `Generator.normal`) with a documented clipping policy
- [ ] Forward GAT
- [ ] Algebraic inverse GAT and closed-form approximation of the unbiased inverse
- [ ] **Statistical tests:**
  - AWGN: empirical mean $\approx 0$, std $\approx \sigma$
  - Poisson-Gaussian on flat patches: $\mathrm{Var}(z)\approx \alpha x + \sigma^2$ and $\mathbb{E}[z]\approx x$
  - GAT: stabilized variance $\approx 1$ on flat patches
  - Inverse GAT round-trip test

### Phase 2
- [ ] **Exact unbiased inverse GAT** (stretch goal; validate against the reference MATLAB routines `GenAnscombe_inverse_exact_unbiased.m` and `..._closed_form.m`). If not reached, report says closed-form is used.
- [ ] **Experiment harness:** parameter grid, matched and mismatched sweeps, multiple seeds, config-driven runs, result files (CSV)
- [ ] Define the **noise grid** (suggested start):
  - AWGN: $\sigma \in \{15, 25, 50\}/255$
  - Poisson-Gaussian: sweep peak (i.e. $1/\alpha$) over a small grid such as $\{1, 2, 5, 10, 30\}$ and $\sigma$ over a small grid, covering Poisson-dominated to Gaussian-dominated regimes
- [ ] Define the **mismatch protocol** (exactly what "method tuned for AWGN, tested on Poisson-Gaussian" means in terms of the `sigma` passed to each denoiser). Write this down in the report.
- [ ] Noise-level estimation runs (`skimage.restoration.estimate_sigma`) for the cases where a method needs $\sigma$
- [ ] Run the full benchmark on Set12, Kodak24 and BSD68

### A's deliverable to B
A working `noise/` module plus a `run_experiments.py` that calls `denoise(...)` from B and writes a results CSV.

---

## 4. Part B: Denoisers, Evaluation and Analysis

**Owner: Roshan K (suggested)**

### Phase 1
- [ ] Mean filter (`scipy.ndimage.uniform_filter`) and Gaussian filter (`scipy.ndimage.gaussian_filter`)
- [ ] Median filter (`scipy.ndimage.median_filter`)
- [ ] Wiener filter (`scipy.signal.wiener`)
- [ ] Metrics: PSNR and SSIM (`skimage.metrics`, `data_range=1.0`), runtime (`time.perf_counter`)
- [ ] End-to-end run of the three methods on sample Set12 images with preliminary sanity-check table

### Phase 2
- [ ] Wavelet soft-thresholding (`skimage.restoration.denoise_wavelet`, `mode='soft'`, VisuShrink or BayesShrink)
- [ ] Non-Local Means (`skimage.restoration.denoise_nl_means`)
- [ ] BM3D (`bm3d.bm3d`, `sigma_psd`)
- [ ] **Two pipelines per method:** direct, and VST-mediated (GAT, then denoise, then inverse GAT), using A's GAT functions
- [ ] **Parameter policy:** per-method parameter settings for the matched case (e.g. $h$ for NLM, patch size, threshold rule). Document them so the comparison is fair and reproducible.
- [ ] **Method-noise** (noisy input minus denoised output, plus energy and structure statistics)
- [ ] Tables and plots: PSNR, SSIM, method-noise, runtime per method, noise type and intensity
- [ ] **Robustness ranking and decision guide:** which method degrades most or least under mismatch, and when each should or should not be trusted

### B's deliverable to A
A `denoisers/` module with the single `denoise(method, z, params)` entry point, plus an `analysis/` folder that reads A's results CSV and produces all tables and plots.

---

## 5. Shared work

| Task | Who |
|---|---|
| Agree interface contract (Section 2) | Both, day 1 |
| Integration test: one image, all methods, one noise setting | Both, end of Phase 1 |
| Final report: Intro, Methodology, Noise models, GAT | A writes |
| Final report: Denoisers, Metrics, Results, Decision guide | B writes |
| Final report: Conclusion, limitations, review of each other's sections | Both |
| README, reproducibility check (fresh clone runs end-to-end) | Both |

---

## 6. Suggested timeline

| Stage | Part A | Part B |
|---|---|---|
| Phase 1, week 1 | Loader, AWGN, PG generator, stat tests | Mean/Gaussian, median, Wiener, PSNR/SSIM/runtime |
| Phase 1, week 2 | Forward GAT, algebraic and closed-form inverse | Sanity-check results on Set12, merge and integration test |
| Phase 2, first half | Harness, grid, mismatch protocol, exact inverse (stretch) | Wavelet, NLM, BM3D, VST pipeline |
| Phase 2, second half | Full runs on all datasets | Method-noise, plots, tables, decision guide |
| Final | Report sections A, README | Report sections B, README |

---

## 7. Repo layout (suggested)

```
.
├── README.md
├── WORK_SPLIT.md
├── requirements.txt
├── configs/            # A: experiment grids
├── data/               # A: datasets (gitignored, with download notes)
├── noise/              # A: awgn, poisson_gaussian, gat
├── tests/              # A: statistical tests; B: denoiser smoke tests
├── denoisers/          # B: wrappers + parameter policy
├── metrics/            # B: psnr, ssim, method_noise, timing
├── run_experiments.py  # A
├── analysis/           # B: tables, plots, ranking
├── results/            # generated CSVs and figures
└── report/
```

## Reference
[1] Makitalo and Foi, "Optimal inversion of the generalized Anscombe transformation for Poisson-Gaussian noise," IEEE TIP, 2013.
