"""Regression test for FIX_PLAN.md M6: QuadraticUQ sparse-leaf fallback.

A leaf that receives zero calibration points previously kept all-zero
quadratic params, so its "uncertainty" was |yh_lin - 0| = |prediction|.
Underdetermined leaves (fewer calibration points than quadratic
features) similarly produced exploding |lin - quad|. Both now fall back
to the global mean sigma.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from jax_ldt import LinearTreeRegressor
from jax_ldt.uncertainty import QuadraticUQ
from jax_ldt.uncertainty.quadratic import quadratic_uncertainty


def test_uncovered_leaf_uses_global_sigma_not_prediction(rng) -> None:
    X = rng.uniform(-3, 3, size=(600, 1))
    y = np.sin(X[:, 0]) + 100.0  # large offset -> |prediction| would be huge
    model = LinearTreeRegressor(max_depth=4, max_bins=8, min_samples_leaf=15).fit(X, y)

    # Calibrate on a narrow slice of the domain so several leaves receive
    # zero calibration points.
    mask = X[:, 0] < 0.0
    Xc, yc = X[mask], y[mask]
    with pytest.warns(UserWarning, match="fall.*back to the global mean sigma"):
        quq = QuadraticUQ(ridge=1e-5).calibrate(model.tree_, Xc, yc)

    # Predict uncertainty on the *other* half (routes to uncovered leaves).
    Xtest = X[~mask]
    sigma = np.asarray(quq.predict(model.tree_, Xtest))
    # Pre-fix these would be ~|prediction| ~ 100. The fallback global
    # sigma is on the residual scale (order 1), never the prediction scale.
    assert np.all(sigma < 10.0), f"max sigma {sigma.max()} looks like |prediction|"
    assert np.all(np.isfinite(sigma))


def test_fallback_state_marks_uncovered_leaves(rng) -> None:
    X = rng.uniform(-3, 3, size=(400, 1))
    y = np.sin(X[:, 0])
    model = LinearTreeRegressor(max_depth=4, max_bins=8, min_samples_leaf=10).fit(X, y)
    mask = X[:, 0] < 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quq = QuadraticUQ().calibrate(model.tree_, X[mask], y[mask])
    fb = np.asarray(quq.state_.leaf_is_fallback)
    # At least one leaf should be flagged as fallback (the right-side
    # leaves got zero calibration points).
    assert fb.any()
    # Global sigma is finite and non-negative.
    assert np.all(np.asarray(quq.state_.global_sigma) >= 0.0)
