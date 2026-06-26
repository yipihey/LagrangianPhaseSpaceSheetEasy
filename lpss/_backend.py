"""Numba availability detection and a no-op fallback.

The whole point of the library is to be *fast* when :mod:`numba` is present and
still *work* (just slower, using pure NumPy) when it is not.  Every numba kernel
in :mod:`lpss` is written so that the same source also runs as plain Python, so
here we only need to provide:

* ``HAVE_NUMBA``      -- a boolean flag,
* ``njit`` / ``prange`` -- the real numba versions, or transparent fall-backs.

Set the environment variable ``LPSS_DISABLE_NUMBA=1`` to force the NumPy path
(handy for debugging or for measuring the speed-up).
"""

from __future__ import annotations

import os

__all__ = ["HAVE_NUMBA", "njit", "prange", "numba_info"]

_DISABLED = os.environ.get("LPSS_DISABLE_NUMBA", "") not in ("", "0", "false", "False")

if _DISABLED:
    HAVE_NUMBA = False
else:
    try:  # pragma: no cover - exercised implicitly by the test-suite
        import numba  # noqa: F401

        HAVE_NUMBA = True
    except Exception:  # pragma: no cover
        HAVE_NUMBA = False


if HAVE_NUMBA:
    from numba import njit, prange  # type: ignore
else:
    # A decorator that ignores numba-specific kwargs (parallel=, fastmath=, ...)
    # and returns the undecorated Python function.
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def wrap(func):
            return func

        return wrap

    # Without numba, ``prange`` is just ``range`` (serial execution).
    prange = range


def numba_info() -> str:
    """A short human-readable description of the active backend."""
    if HAVE_NUMBA:
        import numba

        return f"numba {numba.__version__} (JIT + parallel enabled)"
    if _DISABLED:
        return "numba disabled via LPSS_DISABLE_NUMBA (pure NumPy fallback)"
    return "numba not installed (pure NumPy fallback)"
