"""Quadratic-difference uncertainty.

After fitting a linear leaf model, also fit a quadratic-feature variant
on the same leaf data. The absolute difference between the two
predictions is reported as a "model-form sensitivity" proxy.

Implementation note: rather than refit at predict time, we pre-compute
the quadratic params per leaf at calibration time and store them in
a parallel array. Memory cost: O(n_leaves * n_aug_quadratic^2 * T),
which for typical sizes (n_aug ≈ 10–50, T = 1) is small.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from jax_ldt._types import Tree
from jax_ldt.linear_regression import add_bias
from jax_ldt.tree_core import _fit_many_sides, _route_one
from jax_ldt.uncertainty.linprop import _resolve_tree


def _quadratic_features(X_aug: jnp.ndarray) -> jnp.ndarray:
    """Cross-products X_i * X_j for i <= j (upper triangle incl. diagonal).

    X_aug : (N, n_aug)  -- already has the bias column at index 0
    Returns ``(N, n_aug*(n_aug+1)/2)``. The i=0 row of the triangle
    reproduces the bias (index 0, == 1) and the linear terms, so the
    linear model is a strict subspace of this feature set.
    """
    N, F = X_aug.shape
    # Upper-triangle (incl. diagonal) cross-products, formed directly as
    # X[:, i] * X[:, j] for i <= j — avoids materialising the full
    # (N, F, F) outer product just to gather half of it.
    iu, ju = jnp.triu_indices(F)
    return X_aug[:, iu] * X_aug[:, ju]  # (N, F*(F+1)/2)


@dataclass
class QuadraticUQState:
    """Pre-computed quadratic params per leaf.

    ``leaf_is_fallback`` marks leaves whose quadratic refit is
    unreliable (zero calibration points, or fewer than the number of
    quadratic features → underdetermined). For test points routed
    there, ``quadratic_uncertainty`` returns ``global_sigma`` (the mean
    ``|linear - quadratic|`` over well-fit calibration points) instead of
    the garbage ``|linear - 0|`` a zeroed refit would produce.
    """

    leaf_quad_params: jnp.ndarray  # (n_nodes, n_quad_features, n_targets)
    leaf_is_fallback: jnp.ndarray  # (n_nodes,) bool
    global_sigma: jnp.ndarray  # (n_targets,)


def calibrate_quadratic(
    model_or_tree,
    X: jnp.ndarray,
    y: jnp.ndarray,
    ridge: float = 1e-5,
) -> QuadraticUQState:
    """Refit each leaf with quadratic features. Returns state usable by predict.

    Accepts either a fitted regressor or a `Tree` pytree.
    """
    tree = _resolve_tree(model_or_tree)
    X = jnp.asarray(X, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    if y.ndim == 1:
        y = y[:, None]

    X_t = X @ tree.transform_matrix
    X_aug = add_bias(X_t)
    X_quad = _quadratic_features(X_aug)
    n_quad = X_quad.shape[1]
    T = y.shape[1]

    # Route each sample to its leaf, then materialise once on host so
    # the per-leaf mask construction below doesn't host-sync per node.
    leaf_ids = np.asarray(jax.vmap(lambda x: _route_one(tree, x))(X_t))
    n_nodes = int(tree.n_nodes)

    # Build a (n_nodes, N) leaf-membership matrix on host. Non-leaf
    # nodes get an all-zero row, which the ridge regulariser absorbs to
    # produce a zero-coefficient fit (overwritten below to keep the
    # zero-init invariant that downstream code relies on).
    is_leaf_np = np.asarray(tree.is_leaf)
    leaf_node_ids = np.where(is_leaf_np)[0]

    # Fit ONLY the leaves. The previous version built a dense (n_nodes, N)
    # mask and vmapped the ridge fit over every node — half the solves
    # (internal nodes) were pure waste that got zeroed afterward, and the
    # mask doubled the memory. We fit the (n_leaves, N) membership masks
    # and scatter the results back into the (n_nodes, ...) array, leaving
    # internal nodes at zero (the contract downstream relies on).
    quad_params = jnp.zeros((n_nodes, n_quad, T), dtype=jnp.float64)
    if leaf_node_ids.size > 0:
        membership = (leaf_ids[None, :] == leaf_node_ids[:, None]).astype(np.float64)
        leaf_masks = jnp.asarray(membership)  # (n_leaves, N)
        fitted_leaves = _fit_many_sides(X_quad, y, leaf_masks, ridge)  # (n_leaves, n_quad, T)
        quad_params = quad_params.at[jnp.asarray(leaf_node_ids)].set(fitted_leaves)

    # Flag leaves whose quadratic refit is unreliable: zero calibration
    # points (→ σ would be |lin - 0| = |prediction|, pure garbage) or
    # fewer points than quadratic features (→ underdetermined, near
    # interpolation, |lin - quad| explodes). Such leaves fall back to a
    # global σ at predict time.
    counts = np.zeros(n_nodes, dtype=np.int64)
    if leaf_ids.size > 0:
        binc = np.bincount(leaf_ids, minlength=n_nodes)
        counts[: binc.shape[0]] = binc
    fallback_np = is_leaf_np & (counts < n_quad)

    # Global σ = mean |lin - quad| over calibration points routed to
    # well-fit leaves. Fall back to the mean over all points if every
    # leaf is a fallback (tiny calibration set).
    lin_params_cal = np.asarray(tree.leaf_params)[leaf_ids]  # (N, n_aug, T)
    yh_lin_cal = np.einsum("nf,nft->nt", np.asarray(X_aug), lin_params_cal)
    yh_quad_cal = np.einsum(
        "nq,nqt->nt", np.asarray(X_quad), np.asarray(quad_params)[leaf_ids]
    )
    sigma_cal = np.abs(yh_lin_cal - yh_quad_cal)  # (N, T)
    well_fit = ~fallback_np[leaf_ids] if leaf_ids.size > 0 else np.zeros(0, bool)
    if well_fit.any():
        global_sigma_np = sigma_cal[well_fit].mean(axis=0)
    elif sigma_cal.shape[0] > 0:
        global_sigma_np = sigma_cal.mean(axis=0)
    else:
        global_sigma_np = np.zeros(T, dtype=np.float64)

    n_fallback = int(fallback_np.sum())
    if n_fallback > 0:
        fb_ids = np.where(fallback_np)[0]
        preview = fb_ids[:10].tolist()
        more = "" if fb_ids.size <= 10 else f" (+{fb_ids.size - 10} more)"
        warnings.warn(
            f"{n_fallback} leaves had fewer than n_quad={n_quad} calibration "
            f"points (quadratic refit underdetermined or absent); "
            f"quadratic_uncertainty falls back to the global mean sigma there "
            f"(leaf ids: {preview}{more}). Provide more calibration data for "
            "per-leaf model-form uncertainty.",
            UserWarning,
            stacklevel=2,
        )

    return QuadraticUQState(
        leaf_quad_params=quad_params,
        leaf_is_fallback=jnp.asarray(fallback_np),
        global_sigma=jnp.asarray(global_sigma_np),
    )


def quadratic_uncertainty(
    model_or_tree, state: QuadraticUQState, X: jnp.ndarray
) -> jnp.ndarray:
    """|linear_pred - quadratic_pred| per sample.

    Accepts either a fitted regressor or a `Tree` pytree.
    """
    tree = _resolve_tree(model_or_tree)
    X = jnp.asarray(X, dtype=jnp.float64)
    X_t = X @ tree.transform_matrix
    X_aug = add_bias(X_t)
    X_quad = _quadratic_features(X_aug)

    leaf_ids = jax.vmap(lambda x: _route_one(tree, x))(X_t)

    # Linear prediction
    lin_params = tree.leaf_params[leaf_ids]  # (N, n_aug, T)
    yh_lin = jnp.einsum("nf,nft->nt", X_aug, lin_params)

    # Quadratic prediction
    quad_params_per_sample = state.leaf_quad_params[leaf_ids]  # (N, n_quad, T)
    yh_quad = jnp.einsum("nq,nqt->nt", X_quad, quad_params_per_sample)

    out = jnp.abs(yh_lin - yh_quad)  # (N, T)

    # For points routed to fallback leaves (uncovered / underdetermined
    # quadratic refit), substitute the global sigma instead of the
    # garbage |lin - 0| the zeroed refit would give.
    is_fallback = state.leaf_is_fallback[leaf_ids]  # (N,)
    out = jnp.where(
        is_fallback[:, None],
        jnp.broadcast_to(state.global_sigma, out.shape),
        out,
    )

    if tree.n_targets == 1:
        out = out[:, 0]
    return out


class QuadraticUQ:
    """Marker class for quadratic-difference UQ. Stateful: holds calibration."""

    def __init__(self, ridge: float = 1e-5) -> None:
        self.ridge = ridge
        self.state_: Optional[QuadraticUQState] = None
        self._tree: Optional[Tree] = None

    def calibrate(self, model_or_tree, X: jnp.ndarray, y: jnp.ndarray) -> "QuadraticUQ":
        from jax_ldt._validation import _require_x64

        _require_x64()
        self._tree = _resolve_tree(model_or_tree)
        self.state_ = calibrate_quadratic(self._tree, X, y, ridge=self.ridge)
        return self

    def predict(self, model_or_tree, X: jnp.ndarray) -> jnp.ndarray:
        if self.state_ is None or self._tree is None:
            raise RuntimeError("Call .calibrate(model, X, y) before .predict.")
        # Allow caller to pass either form; we route through the tree
        # captured at calibration time.
        _ = _resolve_tree(model_or_tree)
        return quadratic_uncertainty(self._tree, self.state_, X)
