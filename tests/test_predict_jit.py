"""Regression test for FIX_PLAN.md P3: predict/apply are JIT-compiled and
reuse the compiled kernel across same-shape calls (rather than re-tracing
the vmap on every call)."""

from __future__ import annotations

import numpy as np

from jax_ldt import LinearTreeRegressor
from jax_ldt.tree_core import _apply_jit, _predict_jit, predict


def test_predict_traces_once_for_same_shape(rng) -> None:
    X = rng.uniform(-1, 1, size=(200, 2))
    y = X.sum(axis=1)
    model = LinearTreeRegressor(max_depth=3, min_samples_leaf=10).fit(X, y)

    # First call may or may not compile (the shared jit cache may already
    # hold a same-shape specialization from another test); the invariant
    # we care about is that a SECOND identical call adds no compilation.
    r1 = np.asarray(model.predict(X[:50]))
    mid = _predict_jit._cache_size()
    r2 = np.asarray(model.predict(X[:50]))
    after = _predict_jit._cache_size()
    assert after == mid, "second same-shape predict recompiled"
    np.testing.assert_array_equal(r1, r2)


def test_apply_traces_once_for_same_shape(rng) -> None:
    X = rng.uniform(-1, 1, size=(150, 2))
    y = X.sum(axis=1)
    model = LinearTreeRegressor(max_depth=3, min_samples_leaf=10).fit(X, y)
    model.apply(X[:40])
    mid = _apply_jit._cache_size()
    model.apply(X[:40])
    assert _apply_jit._cache_size() == mid, "second same-shape apply recompiled"


def test_jit_predict_matches_reference(rng) -> None:
    """The jitted path must match an independent numpy recomputation."""
    X = rng.uniform(-1, 1, size=(120, 3))
    y = X[:, 0] - X[:, 1] + 2 * X[:, 2]
    model = LinearTreeRegressor(max_depth=3, min_samples_leaf=10).fit(X, y)
    tree = model.tree_
    leaf_ids = np.asarray(model.apply(X))
    A = np.asarray(tree.transform_matrix)
    X_aug = np.column_stack([np.ones(len(X)), X @ A])
    params = np.asarray(tree.leaf_params)[leaf_ids]  # (N, n_aug, 1)
    ref = np.einsum("nf,nft->nt", X_aug, params)[:, 0]
    np.testing.assert_allclose(np.asarray(predict(tree, X)), ref, atol=1e-10)
