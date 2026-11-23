"""Adapter: ``sigma_assumed`` -> concrete denoiser parameters.

THE ONLY PLACE where the harness knows how a noise level becomes parameters.
Part B owns the real parameter policy and it is not final. Until it lands, the
rules below are TEMPORARY. When ``denoisers`` exposes its own policy function,
switching is a one-line change: make ``params_for`` return it (see the marked
line at the end of this file). Nothing else in the harness has to change.

Arguments
---------
method          one of mean, gaussian, median, wiener, wavelet, nlm, bm3d
sigma_assumed   the noise std the method is told, in the units of the data it
                receives ([0, 1] units in the "image" domain, GAT units, where
                the true std is ~1, in the "gat" domain)
domain          "image" or "gat"
data_range      nominal dynamic range of the data (default 1.0 = image domain).
                Only the window / kernel rules use it, through the *relative*
                noise level sigma_assumed / data_range. In the GAT domain the
                runner passes f(1) - f(0), computed from alpha and sigma_read
                only (never from the noisy data). Without it, "sigma = 1" would
                always select the widest filter in the GAT domain.
                (EXTRA argument, not in the original spec; default keeps the
                3-argument call valid. To be confirmed with Part B.)

TEMPORARY rules (replace with Part B policy)
--------------------------------------------
    s255 = 255 * sigma_assumed / data_range     # relative level, 8-bit units
    mean, median:   window size 3 if s255 < 20, 5 if s255 < 40, else 7
    gaussian:       filter_sigma = min(0.5 + 0.04 * s255, 3.0)  (pixels)
    wiener:         size 3, noise_sigma = sigma_assumed (absolute, in data units)
    wavelet, nlm, bm3d: PLACEHOLDERS (parameter names unknown until Part B
                    implements them); they currently raise NotImplementedError
                    inside ``denoise`` before the parameters are used.
"""

DOMAINS = ("image", "gat")


def _odd_window(s255):
    return 3 if s255 < 20 else (5 if s255 < 40 else 7)


def _temporary_policy(method, sigma_assumed, domain="image", data_range=1.0):
    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {DOMAINS}, got '{domain}'")
    if sigma_assumed < 0 or data_range <= 0:
        raise ValueError("sigma_assumed must be >= 0 and data_range > 0")
    s255 = 255.0 * sigma_assumed / data_range

    if method in ("mean", "median"):
        return {"size": _odd_window(s255)}
    if method == "gaussian":
        return {"filter_sigma": min(0.5 + 0.04 * s255, 3.0)}
    if method == "wiener":
        return {"noise_sigma": float(sigma_assumed)}
    # PLACEHOLDERS, to be filled in when Part B implements these methods
    if method == "wavelet":
        return {"sigma": float(sigma_assumed)}
    if method == "nlm":
        return {"h": 0.8 * float(sigma_assumed)}
    if method == "bm3d":
        return {"sigma_psd": float(sigma_assumed)}
    raise ValueError(f"Unknown method '{method}'")


def params_for(method, sigma_assumed, domain="image", data_range=1.0):
    """Parameter dict for ``denoisers.denoise(method, z, params)``."""
    # ONE-LINE SWITCH: replace with  return denoisers.<policy>(method, sigma_assumed, domain)
    return _temporary_policy(method, sigma_assumed, domain, data_range)
