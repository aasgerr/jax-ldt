"""Linear-propagation uncertainty.

Two modes (see :func:`_per_sample_uncertainty`):

``"upstream"`` (default) — a faithful port of upstream's
``TorchLinearRegression.linprop_uncertainty``:

    σ(x) = sqrt(m/n + Σⱼ (xⱼ - μⱼ)² / ((n-1) σⱼ²)),   m = SSR/n

This is the epistemic variance of the fitted mean; it collapses to ~0 at
leaf centres as n grows and is *not* a predictive std. It is kept as the
default so existing acquisition behaviour and saved-model numbers do not
change.

``"prediction"`` — a proper predictive std with the irreducible-noise
term and an unbiased residual variance ``s² = SSR/max(n−p, 1)``:

    σ(x) = sqrt( s² · (1 + 1/n + Σⱼ (xⱼ - μⱼ)² / ((n-1) σⱼ²)) )

Use this when feeding σ to :func:`calibration_metrics` (which treats it
as a Gaussian predictive std); ``"upstream"`` there under-covers badly.
Both operate on the per-leaf UQ blob populated during tree growth.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

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


def _leaf_param_count(tree: Tree) -> float:
    """Number of free leaf-model parameters = |linear_features| + bias.

    With `linear_features` restriction (H1) the excluded columns have
    exactly-zero coefficients, so the effective DoF is the size of the
    linear set plus the intercept — not `n_features_transformed + 1`.
    """
    return float(len(tree.linear_features)) + 1.0


def _global_noise_var(tree: Tree) -> jnp.ndarray:
    """Pooled residual variance across *informative* leaves, ``(n_targets,)``.

    Estimated as ``Σ SSR / Σ (n − p)`` over leaves with residual degrees of
    freedom (``n > p``); interpolating leaves (``n ≤ p``) carry no noise
    information and are excluded. Used as an irreducible-noise FLOOR in
    prediction mode so σ does not collapse to ~0 on small/interpolating
    leaves whose own ``mse`` is ~0.

    Caveat: if the tree is fully over-parameterized (EVERY leaf has
    ``n ≤ p``, e.g. a very deep tree on few samples), there are no
    informative leaves, the estimate is 0, and no floor can be provided —
    the residuals genuinely carry no noise signal in that regime.
    """
    uq = tree.leaf_uq
    n = uq.n.astype(jnp.float64)  # (n_nodes,)
    p = _leaf_param_count(tree)
    informative = (tree.is_leaf & (n > p)).astype(jnp.float64)  # (n_nodes,)
    ssr_per_leaf = uq.mse * n[:, None]  # (n_nodes, T)  (mse = SSR/n)
    total_ssr = jnp.sum(ssr_per_leaf * informative[:, None], axis=0)  # (T,)
    total_dof = jnp.sum((n - p) * informative)  # scalar
    return jnp.where(total_dof > 0.0, total_ssr / jnp.maximum(total_dof, 1.0), 0.0)


def _per_sample_uncertainty(
    tree: Tree, x_in: jnp.ndarray, mode: str, global_noise_var: jnp.ndarray
) -> jnp.ndarray:
    """Compute σ_pred for one sample (shape: (n_features_in,)).

    ``mode``:

    * ``"upstream"`` (default): the historical port formula
      ``σ² = mse/n + Σ(x−μ)²/((n−1)σ_x²)`` with ``mse = SSR/n``. This is
      the variance of the fitted *mean* (epistemic only) and is what the
      acquisition functions have always consumed; kept as the default so
      numbers do not change silently.
    * ``"prediction"``: a predictive std
      ``σ² = s²·(1 + 1/n + Σ(x−μ)²/((n−1)σ_x²))`` where the residual
      variance ``s²`` is the per-leaf ``SSR/max(n−p, 1)`` *floored* by the
      global pooled residual variance (``global_noise_var``). The floor
      matters: a leaf's own ``mse`` is ~0 when it interpolates its few
      points (``n ≤ p`` or a smooth region), which would otherwise
      collapse σ→~0 precisely on the least-trustworthy leaves. Flooring by
      the global noise keeps σ at a sensible irreducible-noise level, so
      the value is appropriate to feed to ``calibration_metrics`` as a
      Gaussian predictive std.

    Edge cases:

    * Effective ``n < 2``: the deviation term is undefined (no
      within-leaf variance to scale against). We return only the base
      ``sqrt(mse / n)`` term in that case rather than fabricating a
      pseudo-denominator of ``1``.
    * Constant feature inside a leaf (``x_var <= eps``): the leaf's
      linear model cannot have used that feature (no within-leaf
      variation), so we drop it from the deviation sum entirely. The
      old code clamped ``x_var`` to ``1e-12`` which would inflate σ by
      ~1e13 for any test point a unit away from the leaf mean.

    The ``eps = 1e-10`` floor matches the JAX 64-bit zero threshold we
    use elsewhere (see ``_validation.py``); features with variance below
    this are treated as constant within the leaf.
    """
    # Floor for "effectively zero" variance under JAX float64 arithmetic.
    eps = 1e-10

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

    # Per-feature deviation contribution: drop constant features (the
    # leaf model could not have used them, so they should not inflate σ).
    feature_active = x_var > eps
    safe_var = jnp.where(feature_active, x_var, 1.0)
    per_feature = jnp.where(
        feature_active,
        (x_t - x_mean) ** 2 / safe_var,
        0.0,
    )
    raw_deviation = jnp.sum(per_feature)

    # When n < 2, the deviation term is undefined; vanish it. Otherwise
    # divide by (n-1). Use a safe denominator inside the where to avoid
    # NaN propagation from the discarded branch.
    has_variance = n >= 2.0
    safe_n_minus_one = jnp.where(has_variance, n - 1.0, 1.0)
    deviation = jnp.where(has_variance, raw_deviation / safe_n_minus_one, 0.0)

    if mode == "prediction":
        # Unbiased residual variance s² = SSR/(n-p); the stored mse is
        # SSR/n, so s² = mse·n/(n-p). Floor by the global pooled noise so
        # interpolating leaves (mse≈0) don't report ~0 uncertainty.
        p = _leaf_param_count(tree)
        ssr = mse * n  # (n_targets,)
        denom_df = jnp.where(n > p, n - p, 1.0)
        s_sq = jnp.maximum(ssr / denom_df, global_noise_var)  # (n_targets,)
        sigma_sq = s_sq * (1.0 + 1.0 / safe_n + deviation)
    else:  # "upstream" (default)
        sigma_sq = base + deviation  # broadcast: (n_targets,)
    return jnp.sqrt(jnp.clip(sigma_sq, min=0.0))


def linprop_uncertainty(model_or_tree, X: jnp.ndarray, *, mode: str = "upstream") -> jnp.ndarray:
    """Per-sample uncertainty (n_samples, n_targets) or (n_samples,) if T=1.

    Accepts either a fitted regressor (with `.tree_`) or a `Tree` pytree.
    ``mode`` is ``"upstream"`` (default, epistemic mean variance — the
    historical formula) or ``"prediction"`` (a full predictive std); see
    :func:`_per_sample_uncertainty`.
    """
    if mode not in ("upstream", "prediction"):
        raise ValueError(f"mode must be 'upstream' or 'prediction'; got {mode!r}")
    tree = _resolve_tree(model_or_tree)
    if tree.leaf_uq is None:
        raise RuntimeError("Tree has no leaf UQ; refit model.")
    X = jnp.asarray(X, dtype=jnp.float64)
    # Global noise floor computed once (only used in prediction mode).
    gnv = _global_noise_var(tree) if mode == "prediction" else jnp.zeros(tree.n_targets)
    out = jax.vmap(lambda x: _per_sample_uncertainty(tree, x, mode, gnv))(X)  # (N, T)
    if tree.n_targets == 1:
        out = out[:, 0]
    return out


class LinearPropagationUQ:
    """Marker class that selects linear-propagation UQ.

    UQ params are always saved during tree growth (cheap enough), so
    this class just signals the desired predict path. Accepts either a
    fitted regressor or a `Tree` pytree.
    """

    def predict(self, model_or_tree, X: jnp.ndarray, *, mode: str = "upstream") -> jnp.ndarray:
        return linprop_uncertainty(model_or_tree, X, mode=mode)
