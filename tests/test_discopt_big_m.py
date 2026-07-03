"""Regression tests for FIX_PLAN.md H5 (and M1/M2 touch points).

Previously an explicit ``big_m`` was misused as a feature-box
half-width: it set ``lb_t/ub_t = ∓big_m`` and then derived per-node
``m_left = max(ub_t[f] - t, 0)``. Any ``big_m`` smaller than a
threshold made whole subtrees infeasible, silently producing a wrong
optimum. ``big_m`` is now a genuine big-M constant, validated against
derived bounds.
"""

from __future__ import annotations

import numpy as np
import pytest

discopt = pytest.importorskip("discopt")

from jax_ldt import LinearTreeRegressor  # noqa: E402
from jax_ldt.export import embed_in_discopt_model  # noqa: E402


@pytest.mark.discopt
def test_embed_two_sided_fidelity_both_branches() -> None:
    """Two-sided fidelity: with a valid big_m the embedded MIP reproduces
    model.predict at points routed to BOTH the left and right subtree.
    The H5 bug made whole subtrees infeasible when big_m < |threshold|,
    so a right-subtree point would have been cut off. A small depth-2
    tree keeps the big-M solve numerically clean.

    (Uses a large explicit big_m; discopt's big-M solver is numerically
    fragile with deep trees / poorly scaled M — orthogonal to H5, which
    is about big_m being too *small*.)"""
    rng = np.random.default_rng(3)
    X = rng.uniform(0.0, 10.0, size=(300, 1))
    # Piecewise target with a clear split around 5.
    y = np.where(X[:, 0] < 5.0, X[:, 0], 20.0 - X[:, 0])
    model = LinearTreeRegressor(
        max_depth=2, max_bins=8, min_samples_leaf=30
    ).fit(X, y)
    thr = np.asarray(model.tree_.threshold)
    is_leaf = np.asarray(model.tree_.is_leaf)
    root_thr = float(thr[~is_leaf][0])

    # One point comfortably left of the root split, one right.
    left_pt = np.array([root_thr - 2.0])
    right_pt = np.array([root_thr + 2.0])
    for p in (left_pt, right_pt):
        m = discopt.Model("fid")
        x = m.continuous("x", shape=(1,), lb=0.0, ub=10.0)
        y_expr = embed_in_discopt_model(model, m, x, big_m=1e4)
        m.subject_to(x[0] == float(p[0]))
        m.minimize(y_expr[0])
        res = m.solve()
        assert res.objective is not None, f"infeasible at {p} (subtree cut off?)"
        expected = float(np.asarray(model.predict(p[None, :]))[0])
        assert abs(float(res.objective) - expected) < 1e-3, (
            f"at {p}: MIP {res.objective} vs predict {expected}"
        )


@pytest.mark.discopt
def test_undersized_big_m_raises() -> None:
    """A big_m smaller than a valid routing/output bound must raise,
    not silently corrupt the formulation."""
    rng = np.random.default_rng(0)
    # Features on a wide scale so thresholds are ~hundreds.
    X = rng.uniform(0.0, 1000.0, size=(200, 2))
    y = X[:, 0] + X[:, 1]
    model = LinearTreeRegressor(max_depth=3, max_bins=5, min_samples_leaf=15).fit(X, y)

    m = discopt.Model("undersized")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=1000.0)
    with pytest.raises(ValueError, match="big_m.*too small"):
        embed_in_discopt_model(model, m, x, big_m=50.0)


@pytest.mark.discopt
def test_adequate_big_m_accepted() -> None:
    """A big_m large enough to be a valid bound is accepted and faithful."""
    rng = np.random.default_rng(1)
    X = rng.uniform(0.0, 10.0, size=(150, 2))
    y = X[:, 0] + X[:, 1]
    model = LinearTreeRegressor(max_depth=2, max_bins=4, min_samples_leaf=15).fit(X, y)

    p = np.array([3.0, 7.0])
    m = discopt.Model("adequate")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=10.0)
    y_expr = embed_in_discopt_model(model, m, x, big_m=1e4)  # comfortably large
    for i in range(2):
        m.subject_to(x[i] == float(p[i]))
    m.minimize(y_expr[0])
    res = m.solve()
    expected = float(np.asarray(model.predict(p[None, :]))[0])
    assert abs(float(res.objective) - expected) < 1e-3


@pytest.mark.discopt
def test_unbounded_sentinel_triggers_fallback_warning(rng) -> None:
    """M2: discopt's ±1e20 'unbounded' sentinels are finite but toxic as
    a big-M; they must trigger the fallback warning, not become the M."""
    X = rng.uniform(-1.0, 1.0, size=(80, 2))
    y = (X**2).sum(axis=1)
    model = LinearTreeRegressor(max_depth=2, max_bins=4, min_samples_leaf=10).fit(X, y)

    m = discopt.Model("sentinel")
    x = m.continuous("x", shape=(2,))  # no lb/ub → discopt exposes ±1e20
    with pytest.warns(UserWarning, match=r"no usable finite bounds"):
        embed_in_discopt_model(model, m, x)
