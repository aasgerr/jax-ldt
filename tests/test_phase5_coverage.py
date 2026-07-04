"""FIX_PLAN.md Phase 5 — coverage backstops for previously-blind paths."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

import jax.numpy as jnp

from jax_ldt import (
    ExpectedImprovement,
    LinearTreeRegressor,
    MaxVariance,
    UncertaintySampler,
)
from jax_ldt.tree_core import grow_tree, predict


def test_breadth_first_growth_smoke(rng) -> None:
    """depth_first=False was entirely untested."""
    X = rng.uniform(-1, 1, size=(300, 2))
    y = X[:, 0] ** 2 + X[:, 1]
    df = grow_tree(X, y, max_depth=3, max_bins=6, min_samples_leaf=15, depth_first=False)
    bf = grow_tree(X, y, max_depth=3, max_bins=6, min_samples_leaf=15, depth_first=True)
    # Both are valid trees producing finite predictions; node ordering may
    # differ but every sample must route to a leaf.
    yh = np.asarray(predict(df, jnp.asarray(X)))
    assert np.isfinite(yh).all()
    assert int(df.n_leaves) >= 2 and int(bf.n_leaves) >= 2


def test_maxvariance_equals_sigma_squared(rng) -> None:
    X = rng.uniform(-2, 2, size=(300, 1))
    y = np.sin(X[:, 0]) + 0.2 * rng.standard_normal(300)
    model = LinearTreeRegressor(max_depth=3, min_samples_leaf=20).fit(X, y)
    sig = UncertaintySampler().score(model, jnp.asarray(X[:30]))
    var = MaxVariance().score(model, jnp.asarray(X[:30]))
    np.testing.assert_allclose(var, sig**2, atol=1e-10)


def test_ei_matches_closed_form(rng) -> None:
    X = rng.uniform(-2, 2, size=(300, 1))
    y = np.sin(X[:, 0]) + 0.2 * rng.standard_normal(300)
    model = LinearTreeRegressor(max_depth=3, min_samples_leaf=20).fit(X, y)

    acq = ExpectedImprovement(direction="min", xi=0.0)
    acq.update_best(y)
    Xc = jnp.asarray(X[:40])
    got = np.asarray(acq.score(model, Xc))

    mu = np.asarray(model.predict(Xc)).reshape(-1)
    from jax_ldt.uncertainty import linprop_uncertainty

    sigma = np.maximum(np.asarray(linprop_uncertainty(model, Xc)).reshape(-1), 1e-9)
    imp = float(np.min(y)) - mu  # xi=0, direction=min
    z = imp / sigma
    ref = imp * norm.cdf(z) + sigma * norm.pdf(z)
    np.testing.assert_allclose(got, ref, atol=1e-9)


def test_conformal_predict_interval_without_model(rng) -> None:
    """The internal predict(tree, X) path (no model= arg) was never
    exercised."""
    from jax_ldt.uncertainty import ConformalCalibrator

    X = rng.uniform(-3, 3, size=(500, 1))
    y = np.sin(X[:, 0]) + 0.3 * rng.standard_normal(500)
    model = LinearTreeRegressor(max_depth=3, min_samples_leaf=20).fit(X[:300], y[:300])
    cal = ConformalCalibrator(alpha=0.1, min_calibration_per_leaf=3)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal.calibrate(model, X[300:], y[300:])
    lo, hi = cal.predict_interval(X[:20])  # no model= -> internal predict
    lo, hi = np.asarray(lo), np.asarray(hi)
    assert np.all(hi >= lo)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))


def test_mondrian_halfwidths_differ_across_leaves(rng) -> None:
    """Mondrian calibration must produce leaf-specific halfwidths on
    heteroscedastic data (not a single global width)."""
    from jax_ldt.uncertainty import ConformalCalibrator

    x = rng.uniform(-3, 3, size=(1200, 1))
    # Noise scale grows with x -> different leaves should get different
    # residual quantiles.
    y = x[:, 0] + np.abs(x[:, 0]) * rng.standard_normal(1200)
    model = LinearTreeRegressor(max_depth=3, max_bins=8, min_samples_leaf=40).fit(
        x[:700], y[:700]
    )
    cal = ConformalCalibrator(alpha=0.1, min_calibration_per_leaf=5)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal.calibrate(model, x[700:], y[700:])
    hw = np.asarray(cal.state_.leaf_halfwidth)
    is_leaf = np.asarray(model.tree_.is_leaf)
    leaf_hw = hw[is_leaf].reshape(-1)
    # More than one distinct halfwidth across leaves.
    assert np.ptp(leaf_hw) > 1e-6, "mondrian produced a single global halfwidth"
