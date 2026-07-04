"""Regression tests for FIX_PLAN.md M11 (sklearn conformance)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.exceptions import NotFittedError  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

from jax_ldt import (  # noqa: E402
    SklearnHyperplaneTreeRegressor,
    SklearnLinearTreeRegressor,
)


def test_predict_before_fit_raises_notfitted() -> None:
    """M11(a): unfitted predict must raise NotFittedError (not RuntimeError)."""
    with pytest.raises(NotFittedError):
        SklearnLinearTreeRegressor().predict(np.zeros((3, 2)))
    with pytest.raises(NotFittedError):
        SklearnHyperplaneTreeRegressor().predict(np.zeros((3, 2)))


def test_one_dim_X_raises(rng) -> None:
    """M11 / Phase 4 item 13: 1-D X must raise, not be silently reshaped."""
    model = SklearnLinearTreeRegressor(max_depth=2, min_samples_leaf=5)
    with pytest.raises(ValueError, match="[Rr]eshape|2D|2 dim"):
        model.fit(rng.uniform(size=20), rng.uniform(size=20))


def test_multi_target_score_matches_sklearn_uniform_average(rng) -> None:
    """M11(b): score must equal sklearn's uniform-average R² for
    multi-output y (the old pooled-R² override diverged)."""
    X = rng.uniform(-1, 1, size=(200, 3))
    # Two targets on very different scales — pooled vs uniform-average R²
    # diverge most here.
    y = np.column_stack([X[:, 0] + X[:, 1], 100.0 * (X[:, 2] - X[:, 0])])
    model = SklearnLinearTreeRegressor(max_depth=3, min_samples_leaf=10).fit(X, y)
    got = model.score(X, y)
    expected = r2_score(y, np.asarray(model.predict(X)), multioutput="uniform_average")
    assert abs(got - expected) < 1e-9


def test_hyperplane_wrapper_gridsearch_and_clone(rng) -> None:
    """M11(c): the HT wrapper is tunable via GridSearchCV and clone-able."""
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV

    X = rng.uniform(-1, 1, size=(120, 2))
    y = X[:, 0] + X[:, 1]
    base = SklearnHyperplaneTreeRegressor(max_depth=3, min_samples_leaf=10)
    # clone round-trips ALL params (not just one).
    c = clone(base)
    assert c.get_params() == base.get_params()
    gs = GridSearchCV(base, {"num_terms": [2], "max_weight": [1, 2]}, cv=3)
    gs.fit(X, y)
    # The tuned model must actually learn this additive target well, not
    # merely produce a finite score.
    best = gs.best_estimator_
    assert best.score(X, y) > 0.9, f"HT wrapper fit poorly: R2={best.score(X, y)}"


@pytest.mark.parametrize(
    "cls", [SklearnLinearTreeRegressor, SklearnHyperplaneTreeRegressor]
)
def test_unfitted_predict_raises_notfitted(cls) -> None:
    """M11 target: predict/score before fit must raise NotFittedError."""
    with pytest.raises(NotFittedError):
        cls().predict(np.zeros((2, 2)))
    with pytest.raises(NotFittedError):
        cls().score(np.zeros((2, 2)), np.zeros(2))


def test_fit_predict_score_roundtrip_and_1d_x_raises(rng) -> None:
    """A concrete, version-stable conformance slice: fit→predict→score
    works and 1-D X is rejected (rather than relying on the full,
    version-fragile check_estimator suite)."""
    X = rng.uniform(-1, 1, size=(120, 3))
    y = X[:, 0] + X[:, 1]
    m = SklearnLinearTreeRegressor(max_depth=3, min_samples_leaf=8).fit(X, y)
    assert m.predict(X).shape == (120,)
    assert m.score(X, y) > 0.9
    with pytest.raises(ValueError):
        m.predict(np.zeros(3))  # 1-D X
