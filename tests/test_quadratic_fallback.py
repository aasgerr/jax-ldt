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


def test_quadratic_features_matches_outer_product_form(rng):
    """P6: the triu-product form must equal the old (N,F,F) outer-product
    gather exactly."""
    import jax.numpy as jnp
    from jax_ldt.uncertainty.quadratic import _quadratic_features

    X_aug = jnp.asarray(rng.uniform(-1, 1, size=(20, 5)))
    got = np.asarray(_quadratic_features(X_aug))
    F = X_aug.shape[1]
    outer = np.asarray(X_aug)[:, :, None] * np.asarray(X_aug)[:, None, :]
    iu, ju = np.triu_indices(F)
    ref = outer[:, iu, ju]
    np.testing.assert_allclose(got, ref, atol=1e-12)


def test_quadratic_leaf_params_match_direct_fit(rng):
    """P6: fitting only leaves then scattering must give the same per-leaf
    quadratic params as a direct ridge fit on that leaf's points."""
    import jax.numpy as jnp

    from jax_ldt import LinearTreeRegressor
    from jax_ldt.linear_regression import add_bias, fit_ridge
    from jax_ldt.uncertainty.quadratic import _quadratic_features, calibrate_quadratic

    X = rng.uniform(-1, 1, size=(300, 2))
    y = X[:, 0] ** 2 + X[:, 1]
    model = LinearTreeRegressor(max_depth=3, min_samples_leaf=30).fit(X, y)
    state = calibrate_quadratic(model.tree_, X, y, ridge=1e-5)

    tree = model.tree_
    leaf_ids = np.asarray(model.apply(X))
    X_quad = np.asarray(_quadratic_features(add_bias(jnp.asarray(X) @ tree.transform_matrix)))
    is_leaf = np.asarray(tree.is_leaf)
    qp = np.asarray(state.leaf_quad_params)
    for nid in np.where(is_leaf)[0]:
        idx = leaf_ids == nid
        if idx.sum() == 0:
            continue
        ref = np.asarray(fit_ridge(jnp.asarray(X_quad[idx]), jnp.asarray(y[idx]), 1e-5))
        np.testing.assert_allclose(qp[nid], ref, atol=1e-8)
