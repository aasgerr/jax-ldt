"""Regression tests for FIX_PLAN.md Phase 4 low/edge items."""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from jax_ldt import (
    ExpectedImprovement,
    HyperplaneTreeRegressor,
    LinearPropagationUQ,
    LinearTreeRegressor,
    ProbabilityOfImprovement,
    build_transform_matrix,
)


# item 1 — acquisition direction validation
@pytest.mark.parametrize("cls", [ExpectedImprovement, ProbabilityOfImprovement])
def test_acquisition_rejects_bad_direction(cls) -> None:
    with pytest.raises(ValueError, match="direction must be"):
        cls(direction="minimize")


# item 2 — hyperplane guards
def test_num_terms_below_one_raises(rng) -> None:
    X = jnp.asarray(rng.uniform(-1, 1, size=(10, 3)))
    with pytest.raises(ValueError, match="num_terms must be >= 1"):
        build_transform_matrix(X, num_terms=0)


def test_out_of_range_categorical_raises(rng) -> None:
    X = jnp.asarray(rng.uniform(-1, 1, size=(10, 3)))
    with pytest.raises(ValueError, match="categorical_features out of range"):
        build_transform_matrix(X, num_terms=2, categorical_features=(10,))


def test_lc_width_mismatch_raises(rng) -> None:
    X = jnp.asarray(rng.uniform(-1, 1, size=(10, 3)))
    with pytest.raises(ValueError, match="num_terms"):
        build_transform_matrix(X, num_terms=2, LCs=jnp.asarray([[1.0, 2.0, 3.0]]),
                               do_symmetrize=False)


def test_all_zero_lc_row_warns(rng) -> None:
    X = jnp.asarray(rng.uniform(-1, 1, size=(10, 2)))
    with pytest.warns(UserWarning, match="all-zero LC row"):
        build_transform_matrix(X, num_terms=2, LCs=jnp.asarray([[0.0, 0.0]]),
                               do_symmetrize=False)


# item 3 — zero-feature validation
def test_zero_feature_X_raises(rng) -> None:
    with pytest.raises(ValueError, match="at least 1 feature"):
        LinearTreeRegressor().fit(np.zeros((5, 0)), np.zeros(5))


# item 5 — predict shape mirrors 2-D single-target y
def test_predict_shape_mirrors_2d_single_target(rng) -> None:
    X = rng.uniform(-1, 1, size=(100, 2))
    y = (X[:, 0] + X[:, 1]).reshape(-1, 1)  # (N, 1)
    model = LinearTreeRegressor(max_depth=2, min_samples_leaf=10).fit(X, y)
    assert model.predict(X[:10]).shape == (10, 1)
    # 1-D y -> 1-D predictions.
    model1 = LinearTreeRegressor(max_depth=2, min_samples_leaf=10).fit(X, y.reshape(-1))
    assert model1.predict(X[:10]).shape == (10,)


# item 7 — Tree eq=False: hashable, no ambiguous-truth error
def test_tree_is_hashable(rng) -> None:
    X = rng.uniform(-1, 1, size=(60, 2))
    y = X.sum(axis=1)
    tree = LinearTreeRegressor(max_depth=2, min_samples_leaf=10).fit(X, y).tree_
    hash(tree)  # must not raise
    assert (tree == tree) is True  # identity eq


# item 10 — calibration_metrics multi-target guard
def test_calibration_metrics_multitarget_raises() -> None:
    pytest.importorskip("uncertainty_toolbox")
    from jax_ldt.uncertainty.metrics import calibration_metrics

    y = np.zeros((10, 2))
    with pytest.raises(ValueError, match="single-target only"):
        calibration_metrics(y, np.ones((10, 2)), y)


def test_calibration_metrics_nonpositive_uncertainty_raises() -> None:
    pytest.importorskip("uncertainty_toolbox")
    from jax_ldt.uncertainty.metrics import calibration_metrics

    with pytest.raises(ValueError, match="strictly positive"):
        calibration_metrics(np.zeros(10), np.zeros(10), np.zeros(10))


# item 14 — linprop prediction mode
def test_linprop_prediction_mode_nonzero(rng) -> None:
    X = rng.uniform(-2, 2, size=(400, 1))
    y = np.sin(X[:, 0]) + 0.3 * rng.standard_normal(400)
    model = LinearTreeRegressor(max_depth=3, min_samples_leaf=20).fit(X, y)
    up = np.asarray(LinearPropagationUQ().predict(model.tree_, X))
    pr = np.asarray(LinearPropagationUQ().predict(model.tree_, X, mode="prediction"))
    # Prediction std adds the irreducible-noise term, so it does NOT
    # collapse toward 0 at leaf centres the way upstream does: its floor
    # is much higher and its mean sits near the true 0.3 noise scale.
    assert pr.min() > 5 * up.min()
    assert pr.mean() > up.mean()
    assert pr.min() > 0.1  # near the noise floor, not ~0
    with pytest.raises(ValueError, match="mode must be"):
        LinearPropagationUQ().predict(model.tree_, X, mode="bad")
