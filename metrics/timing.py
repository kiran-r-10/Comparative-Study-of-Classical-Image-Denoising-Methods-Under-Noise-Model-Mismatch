"""Runtime measurement (Part B)."""

import time


def timed(fn, *args, **kwargs):
    """Call ``fn(*args, **kwargs)`` and return ``(result, seconds)``.

    Uses ``time.perf_counter`` (wall-clock, monotonic, highest resolution).
    """
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0
