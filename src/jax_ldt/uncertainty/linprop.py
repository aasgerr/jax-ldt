"""Linear-propagation uncertainty (epistemic).

For a leaf with `n` training points, per-feature mean `μ`, per-feature
variance `σ²`, and leaf MSE `m`:

    σ(x) = sqrt( m/n + Σⱼ (xⱼ - μⱼ)² / ((n-1) σⱼ²) )

This is a faithful port of upstream's
`TorchLinearRegression.linprop_uncertainty`. It is the **epistemic**
variance of the fitted *mean* — it collapses toward ~0 at leaf centres as
`n` grows and is *not* a predictive std. The leverage sum runs only over
the features the leaf model actually uses (``linear_features``); features
excluded by ``linear_features`` (or categoricals) have exactly-zero leaf
coefficients, so a test point's distance from the leaf mean in those
dimensions must not inflate σ.

For a **predictive** interval (irreducible noise included, with a
finite-sample coverage guarantee), use conformal prediction
(:class:`jax_ldt.ConformalCalibrator`) instead — a parametric predictive
std cannot be estimated reliably from a tree's own leaf residuals
(interpolating / over-parameterized leaves have ~0 residual, and trees
deliberately create heteroscedastic leaves).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jax_ldt._types import Tree
from jax_ldt.tree_core import _route_one


def _resolve_tree(model_or_tree) -> Tree:
    """Accept either a Tree pytree or a fitted regressor (with .tree_)."""
    if isinstance(model_or_tree, Tree):
        return model_or_tree
    tree = getattr(model_or_tree, "tree_", None)
    if tree is None:
        raise TypeError(
            "Pass either a Tree pytree or a fitted regressor (with `.tree_`)."
        )
    return tree


def _per_sample_uncertainty(
    tree: Tree, x_in: jnp.ndarray, lin_mask: jnp.ndarray
) -> jnp.ndarray:
    """Compute σ for one sample (shape: (n_features_in,)).

    ``σ² = mse/n + Σⱼ (xⱼ − μⱼ)² / ((n−1) σ_x²ⱼ)`` — the epistemic
    mean-variance. ``lin_mask`` is a boolean ``(n_transformed,)`` mask of
    the features in ``linear_features`` (the ones the leaf model uses); the
    deviation sum is restricted to them.

    Edge cases:

    * Effective ``n < 2``: the deviation term is undefined (no within-leaf
      variance to scale against); we return only the ``sqrt(mse/n)`` term.
    * Constant feature inside a leaf (``x_var <= eps``): dropped from the
      deviation sum — the leaf model cannot have used a feature with no
      within-leaf variation, so it must not inflate σ.
    """
    eps = 1e-10  # "effectively zero" variance under float64.

    x_t = x_in @ tree.transform_matrix  # (n_features_transformed,)
    leaf_id = _route_one(tree, x_t)

    if tree.leaf_uq is None:
        raise RuntimeError("Tree has no leaf UQ; refit model.")

    uq = tree.leaf_uq
    n = uq.n[leaf_id].astype(jnp.float64)
    x_mean = uq.x_mean[leaf_id]  # (n_transformed,)
    x_var = uq.x_var[leaf_id]
    mse = uq.mse[leaf_id]  # (n_targets,)

    safe_n = jnp.where(n > 0.0, n, 1.0)
    base = mse / safe_n  # (n_targets,)

    # Per-feature deviation: include a feature only if it (a) is in the
    # leaf's linear model (`lin_mask`) and (b) actually varies within the
    # leaf (`x_var > eps`). Excluded / constant features have zero leaf
    # coefficient and must not inflate σ.
    feature_active = lin_mask & (x_var > eps)
    safe_var = jnp.where(feature_active, x_var, 1.0)
    per_feature = jnp.where(
        feature_active,
        (x_t - x_mean) ** 2 / safe_var,
        0.0,
    )
    raw_deviation = jnp.sum(per_feature)

    has_variance = n >= 2.0
    safe_n_minus_one = jnp.where(has_variance, n - 1.0, 1.0)
    deviation = jnp.where(has_variance, raw_deviation / safe_n_minus_one, 0.0)

    sigma_sq = base + deviation  # broadcast: (n_targets,)
    return jnp.sqrt(jnp.clip(sigma_sq, min=0.0))


def linprop_uncertainty(model_or_tree, X: jnp.ndarray) -> jnp.ndarray:
    """Per-sample epistemic uncertainty, (n_samples, n_targets) or (n_samples,).

    Accepts a fitted regressor (with `.tree_`) or a `Tree` pytree. This is
    an epistemic (fitted-mean) std, NOT a predictive std — for calibrated
    predictive intervals use conformal prediction.
    """
    from jax_ldt._validation import _require_x64

    _require_x64()
    tree = _resolve_tree(model_or_tree)
    if tree.leaf_uq is None:
        raise RuntimeError("Tree has no leaf UQ; refit model.")
    X = jnp.asarray(X, dtype=jnp.float64)
    # Static boolean mask of the transformed columns the leaf model uses.
    lin_mask_np = np.zeros(tree.n_features_transformed, dtype=bool)
    for f in tree.linear_features:
        lin_mask_np[int(f)] = True
    lin_mask = jnp.asarray(lin_mask_np)
    out = jax.vmap(lambda x: _per_sample_uncertainty(tree, x, lin_mask))(X)  # (N, T)
    if tree.n_targets == 1:
        out = out[:, 0]
    return out


class LinearPropagationUQ:
    """Marker class that selects linear-propagation (epistemic) UQ.

    UQ params are always saved during tree growth (cheap enough), so this
    class just signals the desired predict path. Accepts either a fitted
    regressor or a `Tree` pytree.
    """

    def predict(self, model_or_tree, X: jnp.ndarray) -> jnp.ndarray:
        return linprop_uncertainty(model_or_tree, X)
