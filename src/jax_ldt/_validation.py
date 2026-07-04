"""Boundary input validation shared by the regressors.

These checks live here (not in `tree_core`) so the JIT-compiled inner
kernels stay free of Python-level shape / dtype branching.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np


def _require_x64() -> None:
    """Fail fast if JAX 64-bit mode is disabled.

    jax-ldt fits leaf models via the ridge normal equations, which
    square the condition number of ``X``. In float32 this silently
    produces wrong (or NaN) leaf fits on badly scaled data, and JAX's
    default config demotes float64 inputs to float32. Rather than
    degrade precision behind the user's back, we require x64 at every
    public entry point and refuse to run otherwise.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "jax-ldt requires JAX 64-bit mode, which is currently disabled. "
            "Enable it BEFORE importing jax (and jax_ldt), e.g.:\n"
            '    import os; os.environ["JAX_ENABLE_X64"] = "1"\n'
            "or, if jax is already imported:\n"
            '    import jax; jax.config.update("jax_enable_x64", True)\n'
            "Without x64, float64 inputs are silently demoted to float32 and "
            "the ridge solver loses precision on badly scaled data."
        )


def _maybe_warn_non_float64(arr, name: str) -> None:
    """Emit a UserWarning when ``arr`` is not float64.

    jax-ldt always operates in 64-bit (see ``conftest.py`` enabling
    ``jax_enable_x64``). Float32 inputs would silently lose precision
    in the ridge solver and split-finder, so we warn the caller that
    the array is being upcast.
    """
    dtype = getattr(arr, "dtype", None)
    if dtype is None:
        return
    if dtype != jnp.float64:
        warnings.warn(
            f"Casting {name} from {dtype} to float64; jax-ldt operates in 64-bit.",
            UserWarning,
            stacklevel=3,
        )


def _all_finite(arr) -> bool:
    """Return True iff every element of ``arr`` is finite.

    For a numpy input we reduce on host directly. For a JAX array we
    reduce *on device* and transfer only the resulting scalar bool —
    ``np.asarray(arr)`` would instead copy the whole array to host, an
    O(n) device→host transfer per fit/predict that is costly on GPU/TPU
    under cross-validation or active-learning loops.
    """
    if isinstance(arr, np.ndarray):
        return bool(np.isfinite(arr).all())
    return bool(jnp.isfinite(arr).all())


def _validate_fit_inputs(X: jnp.ndarray, y: jnp.ndarray) -> None:
    _require_x64()
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_samples, n_features); got shape {tuple(X.shape)}")
    if X.shape[0] == 0:
        raise ValueError("X must have at least 1 sample.")
    if X.shape[1] == 0:
        raise ValueError("X must have at least 1 feature; got shape (n, 0).")
    if y.ndim not in (1, 2):
        raise ValueError(f"y must be 1-D or 2-D; got shape {tuple(y.shape)}")
    if y.ndim == 2 and y.shape[1] == 0:
        raise ValueError("y must have at least 1 target column; got shape (n, 0).")
    if y.shape[0] != X.shape[0]:
        raise ValueError(
            f"X and y first-dim mismatch: X.shape[0]={X.shape[0]}, y.shape[0]={y.shape[0]}"
        )
    if not _all_finite(X):
        raise ValueError("X contains non-finite values (NaN or inf).")
    if not _all_finite(y):
        raise ValueError("y contains non-finite values (NaN or inf).")
    _maybe_warn_non_float64(X, "X")
    _maybe_warn_non_float64(y, "y")


def _validate_predict_input(X: jnp.ndarray, n_features_in: int) -> None:
    _require_x64()
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {tuple(X.shape)}")
    if X.shape[1] != n_features_in:
        raise ValueError(
            f"X has {X.shape[1]} features but model was fitted on {n_features_in}."
        )
    if not _all_finite(X):
        raise ValueError("X contains non-finite values (NaN or inf).")
