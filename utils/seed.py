"""Reproducible random number generators."""

import zlib

import numpy as np


def stable_key(text):
    """Deterministic integer from a string (Python's built-in hash() is NOT
    stable between runs, so we use crc32)."""
    return zlib.crc32(str(text).encode("utf-8"))


def make_rng(seed, *keys):
    """Return a ``numpy.random.Generator`` for ``seed`` plus extra keys.

    Keys can be ints or strings, e.g. ``make_rng(2026, "kodim05.png", 25, run)``.
    The same arguments always give the same stream, and different keys give
    independent streams, so every (image, noise setting, repeat) combination
    in the experiment harness gets its own reproducible noise.
    """
    ints = [k if isinstance(k, (int, np.integer)) else stable_key(k) for k in keys]
    return np.random.default_rng(np.random.SeedSequence([int(seed), *ints]))
