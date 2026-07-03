"""Regression tests for `linear_features` / categorical leaf-model exclusion.

Pins FIX_PLAN.md H1 and H2: prior to the fix, `linear_features` was
threaded only into `Tree` metadata and never applied, so every leaf
model used all transformed columns (and categorical columns silently
received linear coefficients). The fix zeroes excluded columns in the
fitting design matrix, so their solved coefficients are exactly 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from jax_ldt import HyperplaneTreeRegressor, LinearTreeRegressor, from_json, to_json
from jax_ldt.tree_core import predict


def _leaf_coeffs(model) -> np.ndarray:
    """Leaf-only coefficient block, shape (n_leaves, n_aug, T)."""
    lp = np.asarray(model.tree_.leaf_params)
    is_leaf = np.asarray(model.tree_.is_leaf)
    return lp[is_leaf]


def test_lmdt_linear_features_zeroes_excluded_leaf_coeffs(rng) -> None:
    X = rng.uniform(-1.0, 1.0, size=(300, 3))
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 0.01 * rng.standard_normal(300)
    model = LinearTreeRegressor(
        max_depth=2, min_samples_leaf=20, linear_features=(0,)
    ).fit(X, y)
    # Aug layout is [bias, feat0, feat1, feat2]; excluded transformed
    # features {1, 2} live at aug columns {2, 3} and must be exactly 0.
    coeffs = _leaf_coeffs(model)
    assert np.abs(coeffs[:, 2:, :]).max() == 0.0


def test_lmdt_linear_features_changes_predictions(rng) -> None:
    """Excluding a feature the target depends on must change predictions."""
    X = rng.uniform(-1.0, 1.0, size=(300, 3))
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 0.01 * rng.standard_normal(300)
    full = LinearTreeRegressor(max_depth=2, min_samples_leaf=20).fit(X, y)
    restricted = LinearTreeRegressor(
        max_depth=2, min_samples_leaf=20, linear_features=(0,)
    ).fit(X, y)
    diff = np.abs(np.asarray(full.predict(X)) - np.asarray(restricted.predict(X)))
    assert diff.max() > 1e-3


def test_ht_linear_features_zeroes_excluded_leaf_coeffs(rng) -> None:
    X = rng.uniform(-1.0, 1.0, size=(300, 2))
    y = X[:, 0] + X[:, 1] + 0.01 * rng.standard_normal(300)
    model = HyperplaneTreeRegressor(
        max_depth=2, min_samples_leaf=20, max_weight=1, num_terms=2,
        linear_features=(0,),
    ).fit(X, y)
    coeffs = _leaf_coeffs(model)
    # Every transformed column except index 0 is excluded; only aug
    # columns {0 (bias), 1 (feat0)} may be nonzero.
    assert np.abs(coeffs[:, 2:, :]).max() == 0.0


def test_categorical_feature_excluded_from_leaf_model(rng) -> None:
    """FIX_PLAN H2: a categorical column must not get a leaf coefficient."""
    cat = rng.integers(0, 3, size=300).astype(np.float64)
    cont = rng.uniform(-1.0, 1.0, size=300)
    X = np.column_stack([cat, cont])
    y = 5.0 * cat + cont + 0.01 * rng.standard_normal(300)
    model = LinearTreeRegressor(
        max_depth=1, min_samples_leaf=20, categorical_features=(0,)
    ).fit(X, y)
    coeffs = _leaf_coeffs(model)
    # Categorical column 0 is aug column 1; coefficient must be exactly 0.
    assert np.abs(coeffs[:, 1, :]).max() == 0.0


def test_categorical_relabel_invariance(rng) -> None:
    """Relabeling category codes must not change predictions when the
    categorical column is excluded from splits and the leaf model."""
    cat = rng.integers(0, 4, size=300).astype(np.float64)
    cont = rng.uniform(-1.0, 1.0, size=300)
    X = np.column_stack([cat, cont])
    y = cont + 0.01 * rng.standard_normal(300)
    model = LinearTreeRegressor(
        max_depth=2, min_samples_leaf=20, categorical_features=(0,)
    ).fit(X, y)
    # Swap codes 0<->3 in a fresh test set; predictions must match since
    # the categorical column is inert in both routing and leaf model.
    Xtest = np.column_stack([rng.integers(0, 4, size=50).astype(np.float64),
                             rng.uniform(-1.0, 1.0, size=50)])
    Xswap = Xtest.copy()
    m0, m3 = Xtest[:, 0] == 0, Xtest[:, 0] == 3
    Xswap[m0, 0] = 3
    Xswap[m3, 0] = 0
    np.testing.assert_array_equal(
        np.asarray(model.predict(Xtest)), np.asarray(model.predict(Xswap))
    )


def test_restricted_model_json_roundtrip(rng) -> None:
    """Shape-preservation guard: a linear_features-restricted model must
    still JSON round-trip to identical predictions."""
    X = rng.uniform(-1.0, 1.0, size=(300, 3))
    y = 2 * X[:, 0] + 3 * X[:, 1] + 0.01 * rng.standard_normal(300)
    model = LinearTreeRegressor(
        max_depth=2, min_samples_leaf=20, linear_features=(0, 1)
    ).fit(X, y)
    tree = from_json(to_json(model))
    np.testing.assert_allclose(
        np.asarray(model.predict(X)), np.asarray(predict(tree, X)), atol=1e-12
    )


def test_restricted_model_onnx_roundtrip(rng) -> None:
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from jax_ldt import to_onnx

    X = rng.uniform(-1.0, 1.0, size=(300, 3))
    y = 2 * X[:, 0] + 3 * X[:, 1] + 0.01 * rng.standard_normal(300)
    model = LinearTreeRegressor(
        max_depth=2, min_samples_leaf=20, linear_features=(0, 1)
    ).fit(X, y)
    proto = to_onnx(model)
    sess = ort.InferenceSession(proto.SerializeToString())
    Xtest = X[:40]
    out = np.asarray(sess.run(None, {"X": Xtest.astype(np.float64)})[0])
    np.testing.assert_allclose(
        np.asarray(model.predict(Xtest)), out.reshape(-1), atol=1e-8
    )
