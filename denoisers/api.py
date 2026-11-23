"""Single entry point ``denoise(method, z, params)`` (Part B interface contract).

    x_hat = denoise("median", z, {"size": 5})

Parameter policy (Phase 1)
--------------------------
Each method has a documented default parameter set (``DEFAULT_PARAMS``).  A
caller overrides only what it wants; keys that a method does not understand
raise ``ValueError`` so that a typo can never silently run with defaults.

    method      parameters (defaults)
    ----------  ------------------------------------------------------------
    mean        size=3, mode="reflect"
    gaussian    filter_sigma=1.0, mode="reflect"      (kernel std in pixels)
    median      size=3, mode="reflect"
    wiener      size=3, noise_sigma=None              (noise std, [0,1] units)

Plus one parameter common to all methods:

    clip        True  -> clip the output to [0, 1] (the usual convention: a
                         display image cannot leave the valid range, and PSNR
                         is then computed against a [0, 1] reference).
                False -> return the raw filter output.  The Phase 2
                         VST-mediated pipeline must use ``clip=False`` for the
                         inner call (the data are in the GAT domain, not in
                         [0, 1]) and clip only after the inverse GAT.

Phase 2 methods ("wavelet", "nlm", "bm3d") are registered but not implemented
yet and raise ``NotImplementedError``.
"""

import numpy as np

from .spatial import gaussian_filter, mean_filter, median_filter, wiener_filter

PHASE1_METHODS = ("mean", "gaussian", "median", "wiener")
PHASE2_METHODS = ("wavelet", "nlm", "bm3d")
METHODS = PHASE1_METHODS + PHASE2_METHODS

_IMPLEMENTED = {
    "mean": mean_filter,
    "gaussian": gaussian_filter,
    "median": median_filter,
    "wiener": wiener_filter,
}

DEFAULT_PARAMS = {
    "mean": {"size": 3, "mode": "reflect"},
    "gaussian": {"filter_sigma": 1.0, "mode": "reflect"},
    "median": {"size": 3, "mode": "reflect"},
    "wiener": {"size": 3, "noise_sigma": None},
}


def get_default_params(method):
    """Return a copy of the default parameter dict of ``method``."""
    _check_method(method)
    if method in PHASE2_METHODS:
        raise NotImplementedError(f"'{method}' is a Phase 2 method (not implemented yet)")
    return dict(DEFAULT_PARAMS[method])


def _check_method(method):
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from {METHODS}.")


def denoise(method, z, params=None):
    """Denoise the 2D float image ``z`` with ``method``.

    Parameters
    ----------
    method : one of ``METHODS``
    z : ndarray (H, W), noisy image (float, nominally in [0, 1])
    params : dict or None, overrides of the defaults (see module docstring)

    Returns
    -------
    x_hat : float64 ndarray, same shape as ``z``.  ``z`` is never modified.
    """
    _check_method(method)
    if method in PHASE2_METHODS:
        raise NotImplementedError(f"'{method}' is a Phase 2 method (not implemented yet)")

    params = dict(params or {})
    clip = bool(params.pop("clip", True))

    allowed = DEFAULT_PARAMS[method]
    unknown = set(params) - set(allowed)
    if unknown:
        raise ValueError(
            f"Unknown parameter(s) {sorted(unknown)} for method '{method}'. "
            f"Allowed: {sorted(allowed)} (+ 'clip')."
        )
    kwargs = {**allowed, **params}

    x_hat = _IMPLEMENTED[method](np.array(z, dtype=np.float64, copy=True), **kwargs)
    return np.clip(x_hat, 0.0, 1.0) if clip else x_hat
