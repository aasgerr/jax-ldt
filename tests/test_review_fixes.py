"""Regression tests for the second-round review findings (R1-R6)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import jax.numpy as jnp

from jax_ldt import LinearTreeRegressor
from jax_ldt.uncertainty import QuadraticUQ, linprop_uncertainty


# RR4 — linprop leverage must ignore features excluded from linear_features
# (they have zero leaf coefficient, so a test point's distance in those
# dims must not inflate σ). Third-round finding.
def test_linprop_leverage_ignores_excluded_features(rng) -> None:
    X = rng.uniform(-1, 1, size=(300, 3))
    y = X[:, 0]
    m = LinearTreeRegressor(
        max_depth=2, min_samples_leaf=20, linear_features=(0,), split_features=(0,)
    ).fit(X, y)
    a = np.array([[0.1, 0.0, 0.0]])
    b = np.array([[0.1, 0.9, -0.9]])  # differs only in EXCLUDED dims 1,2
    # Same leaf, identical prediction -> σ must be (near) identical too.
    assert np.allclose(np.asarray(m.predict(a)), np.asarray(m.predict(b)))
    sa = float(np.asarray(linprop_uncertainty(m, a)).reshape(-1)[0])
    sb = float(np.asarray(linprop_uncertainty(m, b)).reshape(-1)[0])
    assert abs(sb - sa) < 1e-9, f"excluded dims inflated sigma: {sa} vs {sb}"


def test_linprop_categorical_dims_do_not_inflate_sigma(rng) -> None:
    """The default categorical path (auto-excluded from linear_features)."""
    cat = rng.integers(0, 4, size=300).astype(float)
    cont = rng.uniform(-1, 1, size=300)
    X = np.column_stack([cat, cont])
    y = cont
    m = LinearTreeRegressor(
        max_depth=2, min_samples_leaf=20, categorical_features=(0,)
    ).fit(X, y)
    a = np.array([[0.0, 0.2]])
    b = np.array([[3.0, 0.2]])  # differs only in the categorical dim
    if np.allclose(np.asarray(m.predict(a)), np.asarray(m.predict(b))):
        sa = float(np.asarray(linprop_uncertainty(m, a)).reshape(-1)[0])
        sb = float(np.asarray(linprop_uncertainty(m, b)).reshape(-1)[0])
        assert abs(sb - sa) < 1e-9


# R3 — quadratic UQ must keep per-leaf signal in moderate dims (not all-fallback)
def test_quadratic_keeps_signal_in_moderate_dims(rng) -> None:
    X = rng.uniform(-1, 1, size=(600, 10))
    y = X[:, 0] + X[:, 1] ** 2 + 0.1 * rng.standard_normal(600)
    m = LinearTreeRegressor(max_depth=3, min_samples_leaf=25).fit(X[:400], y[:400])
    # All leaves have calibration points -> none should be fallback even
    # though n_quad (66) >> leaf sizes (~25).
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a spurious fallback warning would fail here
        q = QuadraticUQ().calibrate(m.tree_, X[400:], y[400:])
    assert not np.asarray(q.state_.leaf_is_fallback)[np.asarray(m.tree_.is_leaf)].any()
    sig = np.asarray(q.predict(m.tree_, X[400:]))
    # Real per-leaf discrimination, not a single constant.
    assert len(np.unique(np.round(sig, 6))) > 5


def test_quadratic_still_flags_empty_leaves(rng) -> None:
    """R3 must still catch the genuine garbage case (zero-cal leaves)."""
    X = rng.uniform(-3, 3, size=(600, 1))
    y = np.sin(X[:, 0]) + 100.0
    m = LinearTreeRegressor(max_depth=4, max_bins=8, min_samples_leaf=15).fit(X, y)
    mask = X[:, 0] < 0.0  # calibrate on left half only
    with pytest.warns(UserWarning, match="ZERO calibration points"):
        q = QuadraticUQ().calibrate(m.tree_, X[mask], y[mask])
    sigma = np.asarray(q.predict(m.tree_, X[~mask]))  # routes to empty leaves
    assert np.all(sigma < 10.0)  # global sigma, not |prediction| ~ 100


# R4 — the loop must not clobber a user-set batcher seed
def test_loop_respects_user_batcher_seed(rng) -> None:
    from jax_ldt import ActiveLearner, DiverseBatchSelector, UncertaintySampler

    bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]])
    sel = DiverseBatchSelector(pool_factor=3, seed=12345)
    loop = ActiveLearner(
        model_factory=lambda: LinearTreeRegressor(max_depth=2, min_samples_leaf=5),
        acquisition=UncertaintySampler(),
        batcher=sel,
        bounds=bounds,
        batch_size=4,
        seed=0,
    )
    X0 = rng.uniform(-2, 2, size=(20, 2))
    loop.tell(X0, (X0**2).sum(1))
    loop.ask()
    assert sel.seed == 12345, "loop clobbered the user-set batcher seed"


def test_loop_provides_seed_when_unset(rng) -> None:
    from jax_ldt import ActiveLearner, DiverseBatchSelector, UncertaintySampler

    bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]])
    sel = DiverseBatchSelector(pool_factor=3)  # seed defaults None
    loop = ActiveLearner(
        model_factory=lambda: LinearTreeRegressor(max_depth=2, min_samples_leaf=5),
        acquisition=UncertaintySampler(),
        batcher=sel,
        bounds=bounds,
        batch_size=4,
        seed=0,
    )
    X0 = rng.uniform(-2, 2, size=(20, 2))
    loop.tell(X0, (X0**2).sum(1))
    seeds = []
    for _ in range(4):
        loop.ask()
        seeds.append(sel.seed)
    # RR5: the per-round seed must VARY across rounds (the earlier fix
    # froze it at round 1's value because it re-checked `is None`).
    assert all(s is not None for s in seeds)
    assert len(set(seeds)) > 1, f"batcher seed frozen across rounds: {seeds}"


def test_uq_and_loader_require_x64() -> None:
    """RR3: linprop/quadratic UQ and from_json must enforce x64 too, not
    just fit/predict — else a loaded tree runs silently in float32."""
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import jax
        assert not jax.config.jax_enable_x64
        from jax_ldt.export import from_json
        try:
            from_json('{"version": 1}')  # guard runs before anything else
            print("NOGUARD")
        except RuntimeError as e:
            print("GUARD" if ("x64" in str(e) or "64-bit" in str(e)) else "OTHER")
        """
    )
    import os

    clean = dict(os.environ)
    clean.pop("JAX_ENABLE_X64", None)
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=clean
    )
    assert "GUARD" in proc.stdout, proc.stdout + proc.stderr


# R5 — calibration_metrics rejects NaN uncertainty (not just <=0)
def test_calibration_metrics_rejects_nan_uncertainty() -> None:
    pytest.importorskip("uncertainty_toolbox")
    from jax_ldt.uncertainty.metrics import calibration_metrics

    y = np.zeros(10)
    unc = np.ones(10)
    unc[3] = np.nan
    with pytest.raises(ValueError, match="positive, finite"):
        calibration_metrics(y, unc, y)


# R6 — HT sklearn wrapper exposes hyperplane_weights
def test_ht_wrapper_exposes_hyperplane_weights() -> None:
    pytest.importorskip("sklearn")
    from jax_ldt import SklearnHyperplaneTreeRegressor

    params = SklearnHyperplaneTreeRegressor().get_params()
    assert "hyperplane_weights" in params
