"""Regression tests for FIX_PLAN.md H4.

The split kernel previously combined the two sides' losses as a
sample-count-weighted average `(loss_b*n_b + loss_a*n_a)/n`. That is
exact only for mean-type losses (mae, msle); for `rmse` it must pool
under the sqrt and for `max_abs` it must take the max across sides.
Upstream evaluates the loss of the *combined piecewise prediction* over
the node. These tests pin that the kernel now does the same, and that it
genuinely differs from the old weighted-average for rmse/max_abs.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jax_ldt.linear_regression import add_bias
from jax_ldt.tree_core import (
    _WEIGHTED_LOSS_FNS,
    _compute_thresholds,
    _evaluate_splits,
)


def _recompute(loss_name, X_aug, y, node_mask, X_split, thresholds,
               theta_b, theta_a):
    """Return (combined, weighted_avg) node losses per candidate."""
    fn = _WEIGHTED_LOSS_FNS[loss_name]
    Xa = np.asarray(X_aug)
    yj = jnp.asarray(y)
    nm = np.asarray(node_mask)
    thr = np.asarray(thresholds)
    K, B = theta_b.shape[0], theta_b.shape[1]
    combined = np.full((K, B), np.nan)
    weighted = np.full((K, B), np.nan)
    for c in range(K):
        for b in range(B):
            below = np.asarray(X_split)[:, c] <= thr[b, c]
            mb = (nm * below).astype(float)
            ma = (nm * (~below)).astype(float)
            if mb.sum() < 3 or ma.sum() < 3:
                continue
            yhb = Xa @ np.asarray(theta_b[c, b])
            yha = Xa @ np.asarray(theta_a[c, b])
            yhc = yhb * mb[:, None] + yha * ma[:, None]
            combined[c, b] = float(
                np.sum(np.asarray(fn(yj, jnp.asarray(yhc), jnp.asarray(nm))))
            )
            lb = np.asarray(fn(yj, jnp.asarray(yhb), jnp.asarray(mb)))
            la = np.asarray(fn(yj, jnp.asarray(yha), jnp.asarray(ma)))
            weighted[c, b] = float(
                np.sum((lb * mb.sum() + la * ma.sum()) / nm.sum())
            )
    return combined, weighted


@pytest.mark.parametrize("loss_name", ["max_abs", "rmse", "mae"])
def test_kernel_uses_combined_prediction_loss(loss_name) -> None:
    rng = np.random.default_rng(1)
    N = 150
    X_t = rng.uniform(-1.0, 1.0, size=(N, 1))
    y = (X_t[:, 0] ** 2 + 0.1 * rng.standard_normal(N))[:, None]
    X_aug = add_bias(jnp.asarray(X_t))
    node_mask = jnp.ones(N)
    X_split = jnp.asarray(X_t)
    thresholds = _compute_thresholds(X_t, max_bins=8)

    overall, theta_b, theta_a, _, _ = _evaluate_splits(
        X_aug, jnp.asarray(y), node_mask, X_split, thresholds,
        1e-5, 3.0, loss_name,
    )
    overall = np.asarray(overall)
    combined, weighted = _recompute(
        loss_name, X_aug, y, node_mask, X_split, thresholds,
        np.asarray(theta_b), np.asarray(theta_a),
    )

    finite = np.isfinite(combined)
    assert finite.any()
    # The kernel's reported loss must equal the combined-prediction loss.
    np.testing.assert_allclose(overall[finite], combined[finite], atol=1e-9)

    if loss_name in ("max_abs", "rmse"):
        # ...and must genuinely differ from the old weighted-average, so
        # this test fails against the pre-fix kernel.
        assert np.nanmax(np.abs(combined - weighted)) > 1e-6
    else:  # mae reduces exactly to the weighted average
        np.testing.assert_allclose(
            combined[finite], weighted[finite], atol=1e-9
        )


def test_maxabs_picks_lower_max_split() -> None:
    """End-to-end: with max_abs, the root split must minimize the node's
    max error, which the weighted-average combination could get wrong."""
    rng = np.random.default_rng(2)
    N = 400
    x = rng.uniform(-1.0, 1.0, size=(N, 1))
    # A clean piecewise-linear target with a kink at 0: max_abs should
    # split near 0.
    y = np.where(x[:, 0] < 0, -x[:, 0], 3.0 * x[:, 0])
    from jax_ldt import LinearTreeRegressor

    model = LinearTreeRegressor(
        criterion="max_abs", max_depth=1, max_bins=16, min_samples_leaf=20
    ).fit(x, y)
    feat = np.asarray(model.tree_.feature)
    is_leaf = np.asarray(model.tree_.is_leaf)
    thr = np.asarray(model.tree_.threshold)
    root_thr = thr[~is_leaf][0]
    assert abs(root_thr) < 0.2, f"root split at {root_thr}, expected near 0"


def test_chunking_is_exact(monkeypatch) -> None:
    """P1: chunking the candidate axis (lax.map batch_size) must produce
    the same split evaluation as an all-at-once pass, to float tolerance."""
    import jax
    import jax_ldt.tree_core as tc
    from jax_ldt.linear_regression import add_bias

    rng = np.random.default_rng(7)
    N = 220
    X = rng.uniform(-1, 1, size=(N, 3))
    y = (X[:, 0] ** 2 + X[:, 1])[:, None]
    X_aug = add_bias(jnp.asarray(X))
    node_mask = jnp.ones(N)
    X_split = jnp.asarray(X)
    thr = _compute_thresholds(X, max_bins=8)

    args = (X_aug, jnp.asarray(y), node_mask, X_split, thr, 1e-5, 3.0, "max_abs")
    o_natural, tb_natural, *_ = _evaluate_splits(*args)

    # Force maximal chunking (block=1) and rebuild the jitted kernel.
    monkeypatch.setattr(tc, "_pick_block_size", lambda *a: 1)
    ev_chunked = jax.jit(tc._evaluate_splits_kernel, static_argnames=("loss_name",))
    o_chunked, tb_chunked, *_ = ev_chunked(*args)

    np.testing.assert_allclose(np.asarray(o_natural), np.asarray(o_chunked), atol=1e-9)
    np.testing.assert_allclose(np.asarray(tb_natural), np.asarray(tb_chunked), atol=1e-9)
