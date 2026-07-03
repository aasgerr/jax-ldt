"""Regression tests for FIX_PLAN.md M3/M4/M5 (conformal edge cases)."""

from __future__ import annotations

import numpy as np
import pytest

from jax_ldt import LinearTreeRegressor
from jax_ldt.uncertainty import ConformalCalibrator, fit_with_conformal


def _fit(rng, n=400):
    X = rng.uniform(-3, 3, size=(n, 1))
    y = np.sin(X[:, 0]) + 0.3 * rng.standard_normal(n)
    model = LinearTreeRegressor(max_depth=3, max_bins=6, min_samples_leaf=15).fit(X, y)
    return model, X, y


def test_min_calibration_per_leaf_defaults_from_alpha(rng) -> None:
    """M3: with min_calibration_per_leaf=None the threshold is
    max(5, ceil(1/alpha)-1). At alpha=0.05 that is 19."""
    model, X, y = _fit(rng)
    # Route a small per-leaf calibration set: sparse leaves must trigger
    # the fallback with the alpha-derived threshold mentioned in the msg.
    cal = ConformalCalibrator(alpha=0.05)  # -> min_cal = 19
    with pytest.warns(UserWarning, match="fewer than 19"):
        cal.calibrate(model, X[:60], y[:60])


def test_explicit_min_calibration_below_one_raises() -> None:
    with pytest.raises(ValueError, match="min_calibration_per_leaf must be >= 1"):
        ConformalCalibrator(alpha=0.1, min_calibration_per_leaf=0)


def test_per_leaf_clip_warns(rng) -> None:
    """M3: a leaf that passes min_calibration_per_leaf but is still too
    small for a strict per-leaf (1-alpha) quantile must warn about the
    clip (rather than silently clipping to the max residual)."""
    model, X, y = _fit(rng)
    # min_cal=3 lets small leaves through; alpha=0.1 needs 9 per leaf for a
    # strict quantile, so leaves with 3..8 points clip and must warn.
    cal = ConformalCalibrator(alpha=0.1, min_calibration_per_leaf=3)
    with pytest.warns(UserWarning, match="strict per-leaf"):
        cal.calibrate(model, X[:40], y[:40])


def test_empty_calibration_set_raises(rng) -> None:
    """M4: an empty calibration set must raise, not yield NaN intervals."""
    model, X, y = _fit(rng)
    with pytest.raises(ValueError, match="calibration set is empty"):
        ConformalCalibrator(alpha=0.1).calibrate(model, X[:0], y[:0])


def test_fit_with_conformal_empty_split_raises() -> None:
    """M4: fit_with_conformal must reject a split that empties a side."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(2, 1))
    y = X[:, 0]
    with pytest.raises(ValueError, match="non-empty"):
        fit_with_conformal(
            LinearTreeRegressor(), X, y, calibration_size=0.9, alpha=0.1
        )


def test_predict_interval_rejects_refit_model(rng) -> None:
    """M5: passing a model whose tree_ differs from the calibration-time
    tree must raise (stale halfwidths/routing would invalidate coverage)."""
    model, X, y = _fit(rng)
    cal = ConformalCalibrator(alpha=0.1, min_calibration_per_leaf=3)
    with warnings_ignored():
        cal.calibrate(model, X[:120], y[:120])
    # Refit the SAME model object on different data -> new tree_.
    X2 = rng.uniform(-3, 3, size=(400, 1))
    y2 = np.cos(X2[:, 0])
    model.fit(X2, y2)
    with pytest.raises(ValueError, match="differs from"):
        cal.predict_interval(X[:10], model=model)


def test_fit_with_conformal_forwards_kwargs(rng) -> None:
    """Phase 4 item 9: min_calibration_per_leaf / sparse_leaf_strategy
    are forwarded through fit_with_conformal."""
    X = rng.uniform(-3, 3, size=(400, 1))
    y = np.sin(X[:, 0]) + 0.3 * rng.standard_normal(400)
    _, cal = fit_with_conformal(
        LinearTreeRegressor(max_depth=3, min_samples_leaf=15),
        X, y, calibration_size=0.3, alpha=0.1,
        min_calibration_per_leaf=7, sparse_leaf_strategy="skip",
    )
    assert cal.min_calibration_per_leaf == 7
    assert cal.sparse_leaf_strategy == "skip"


import contextlib


@contextlib.contextmanager
def warnings_ignored():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
