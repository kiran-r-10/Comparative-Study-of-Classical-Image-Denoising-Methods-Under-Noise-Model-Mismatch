# Phase 1 sanity check (set12, 12 images, AWGN)

Mean over images. `noisy` = the un-denoised input. Fixed untuned parameters: mean 3x3, gaussian filter_sigma=1.0, median 3x3, wiener 3x3 given the true sigma.

## sigma = 15/255

| Method | PSNR (dB) | SSIM | Time (ms/image) |
|---|---|---|---|
| noisy | 24.61 | 0.5598 | - |
| mean | 27.52 | 0.7794 | 1.69 |
| gaussian | 27.67 | 0.8080 | 2.15 |
| median | 27.80 | 0.7465 | 17.08 |
| wiener | 29.52 | 0.7891 | 11.14 |

## sigma = 25/255

| Method | PSNR (dB) | SSIM | Time (ms/image) |
|---|---|---|---|
| noisy | 20.17 | 0.3898 | - |
| mean | 26.03 | 0.6683 | 1.71 |
| gaussian | 26.51 | 0.7161 | 2.14 |
| median | 25.45 | 0.6115 | 17.16 |
| wiener | 26.16 | 0.6465 | 11.27 |

## sigma = 50/255

| Method | PSNR (dB) | SSIM | Time (ms/image) |
|---|---|---|---|
| noisy | 14.15 | 0.2017 | - |
| mean | 22.41 | 0.4586 | 1.57 |
| gaussian | 23.37 | 0.5170 | 2.15 |
| median | 21.07 | 0.3958 | 17.15 |
| wiener | 21.36 | 0.4212 | 11.08 |
