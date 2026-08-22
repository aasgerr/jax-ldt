# Algorithm notes

This page summarises the math for what the trees actually compute. For
the user-facing API, see [the quickstart](quickstart.md).

## Tree structure

A trained tree is a flat parallel-array layout (see `jax_ldt._types.Tree`):

| field              | shape                          | meaning                              |
|--------------------|--------------------------------|--------------------------------------|
| `is_leaf`          | (n_nodes,)                     | leaf marker                          |
| `feature`          | (n_nodes,)                     | split feature in *transformed* space |
| `threshold`        | (n_nodes,)                     | split threshold; NaN at leaves       |
| `left`, `right`    | (n_nodes,)                     | child node ids                       |
| `leaf_params`      | (n_nodes, n_aug, n_targets)    | `[bias; weights]` per leaf           |
| `transform_matrix` | (n_features_in, n_lifted)      | maps X to lifted features            |

For a sample `x`:
1. **Lift**: `x' = x @ A` where `A = transform_matrix`.
2. **Augment**: `x_aug = [1; x']`.
3. **Route**: starting at node 0, follow `left` if `x'[feature] ≤
   threshold`, else `right`, until a leaf.
4. **Predict**: `ŷ = x_aug · leaf_params[leaf_id]`.

For axis-aligned LMDTs, `A` is the identity. For HTs, the first
`n_features_in` columns of `A` are the identity (preserving original
features) and the rest encode oblique directions.

## Hyperplane enumeration

`hyperplanes.build_transform_matrix` constructs `A` in three steps:

1. **Generate integer-weight rows** via Miller-index enumeration. For
   `(num_terms=2, max_weight=W)`, this yields all unique rows
   `[1, w]` with `w ∈ {0, 1/W, …, W}`.
2. **Symmetrise**: apply ±-parity and column permutations, prune rows
   with non-trailing zeros, normalise to leftmost-1, dedup at
   `tol_decimals` precision.
3. **Permute over input features**: for each ordered length-`num_terms`
   selection of non-categorical input columns, place each LC row at
   those positions.
4. **Optional scaling**: divide each row by feature range so oblique
   angles aren't dominated by feature units; renormalise.

The result is appended to an identity block so axis-aligned splits are
still candidates.

## Recursive split-and-grow

Tree growth is a Python while-loop over a queue of (node_id, depth).
Per node:

1. Compute candidate thresholds: `max_bins - 1` quantile cuts of each
   split feature on the active subdomain.
2. **Inner kernel** (JIT-compiled): for each candidate (column, bin),
   evaluate two ridge regressions (left and right side) using the side
   mask as a 0/1 sample weight, then score the combined piecewise
   prediction under the chosen criterion. The K·B candidates are
   vmapped — a single fused kernel handles them.
3. Pick the candidate with lowest combined loss; if no valid split or
   the depth limit is reached, finalise the node as a leaf.

The split kernel is shape-stable (sample axis is always `N`, the full
training set) so JIT compiles once. Subdomain selection is via mask
multiplication, not slicing.

Each node's own regression is fitted exactly once, when the node is
created: the same parameters serve as its `min_impurity_decrease`
baseline and, if growth stops there, as its leaf model.

### Candidate fits live in the input space, not the lifted space

The candidate kernel never forms an `n_aug × n_aug` system, and this is
what keeps hyperplane fits affordable. The fitting design matrix always
factors as

```
X_fit = Z @ B        Z = [1, X]  (N × m,  m = n_features_in + 1)
                     B = blockdiag(1, A), columns outside `linear_features` zeroed
```

because every lifted column is by construction a linear function of the
original features. The ridge solve then satisfies the push-through
identity

```
(Bᵀ G B + λI)⁻¹ Bᵀ v  =  Bᵀ (G BBᵀ + λI)⁻¹ v      G = ZᵀWZ,  v = ZᵀWy
```

so a candidate's fitted values are `ŷ = (Z·BBᵀ) · (G·BBᵀ + λI)⁻¹ v` —
all in `m` dimensions, whatever the lift's width. Cost per candidate
drops from `O(N·n_aug² + n_aug³)` to `O(N·m² + m³)`.

This is an algebraic identity, not an approximation: candidate losses
agree with the dense `n_aug`-space fit to round-off (~1e-15 relative),
pinned by `tests/test_hyperplane_scaling.py`. The *leaf* parameters are
still solved densely in the lifted space, because those coefficients are
what gets exported (JSON / ONNX / discopt) and reproducibility of the
exported model matters more there than the cubic term.

Candidates are evaluated in column chunks sized to a fixed memory budget
(`tree_core.SPLIT_MEMORY_BUDGET_BYTES`, 64 MiB), so peak allocation does
not grow with the lifted-feature count. The chunk width is constant for
a given fit — the tail chunk is padded with `+inf` thresholds, which
score as invalid — so the kernel still traces once. Problems whose
candidate set fits the budget (nearly all of them) take a single call
and are bit-identical to an unchunked evaluation.

See [Benchmarks → Cost model](benchmarks.md#cost-model) for what this
means in wall-clock terms.

## Why ridge regression?

We solve `(XᵀWX + ridge·I) β = XᵀWy` because:
- the ridge regularisation guarantees positive-definiteness of `XᵀWX`,
  so `jax.scipy.linalg.solve(..., assume_a='pos')` is faster and more
  stable than `lstsq` (the reduced-space candidate solve above uses the
  general solver, since `G·BBᵀ` is not symmetric — it is still
  invertible for any `ridge > 0`, as `G·BBᵀ` has real non-negative
  eigenvalues);
- it's well-defined even when `W` has very few nonzero entries — important
  because invalid candidate splits still produce *some* coefficients
  during the vmapped batch (we discard them via a validity mask after).

The order of ridge application is fixed: ridge is added to `XᵀWX`
*before* the solve, never after. This is pinned by
`tests/test_principled_choices.py` (planned follow-up) but is also
documented in `linear_regression.py`.
