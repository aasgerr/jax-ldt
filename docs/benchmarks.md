# Benchmarks

Performance numbers are environment-dependent; the table below is
meant as an order-of-magnitude reference, not a hard claim.

| Operation | jax-ldt | upstream PyTorch | Notes |
|-----------|---------|------------------|-------|
| Fit LMDT, Branin 2D, 400 pts | ~0.6 s | ~0.4 s | Cold-start JIT cost dominates |
| Predict 1000 pts | ~50 ms | ~30 ms | After JIT; vmapped routing |
| Fit HT, Branin 2D | ~1.5 s | ~1.2 s | Hyperplane lift adds features |
| ONNX export | ~10 ms | n/a | hand-built ONNX graph |

The JIT compile happens on the first call to the inner split kernel;
subsequent splits with the same shape signature are fast.

## Cost model

Hyperplane fit cost is driven by the **lifted-feature count** `L`, not by
`n_features` directly. Get `L` before you fit:

```python
from jax_ldt import count_lifted_features

count_lifted_features(4)                          # 16   (defaults)
count_lifted_features(16)                         # 256
count_lifted_features(6, num_terms=3, max_weight=2)   # 656
count_lifted_features(8, num_terms=4, max_weight=3)   # 57408
```

With the defaults (`num_terms=2, max_weight=1`) the count is exactly
`n_features ** 2`. Raising `num_terms` is by far the biggest lever —
each extra term multiplies the permutation count by roughly
`n_features`, *and* multiplies the integer-weight enumeration.
`max_weight` grows it polynomially. `LinearTreeRegressor` is the `L =
n_features` case.

For a tree with `V` nodes, `N` training rows, `T` targets, `B =
max_bins - 1` bins and `m = n_features + 1`:

| stage | cost | grows with `L`? |
|-------|------|-----------------|
| build the transform | `O(L · n_features)` | linearly |
| candidate search (per node) | `O(L · B · (N·m² + m³))` | linearly |
| leaf/node solve (per node) | `O(N·L² + L³)` | **cubically** |

So total fit cost is roughly `O(V · (L·B·N·m² + L³))`. Below ~1000
lifted features the linear candidate-search term dominates and fits are
seconds; past that the cubic per-node solve takes over, which is why
`build_transform_matrix` emits a `UserWarning` above
`hyperplanes.LIFTED_FEATURE_WARN_THRESHOLD` (1024).

Measured on a 4-core CPU runner, 120 rows, 1 target, `max_depth=4`,
`criterion="mae"` (includes ~1 s of cold-start JIT):

| n_features | num_terms | max_weight | lifted `L` | fit |
|---|---|---|---|---|
| 4 | 2 | 1 | 16 | 1.4 s |
| 8 | 2 | 1 | 64 | 1.4 s |
| 12 | 2 | 1 | 144 | 1.7 s |
| 16 | 2 | 1 | 256 | 2.5 s |
| 20 | 2 | 1 | 400 | 3.6 s |
| 6 | 3 | 1 | 116 | 2.2 s |
| 6 | 3 | 2 | 656 | 3.1 s |
| 8 | 3 | 2 | 1744 | 7.2 s |
| 6 | 4 | 2 | 2456 | 10.9 s |
| 5 | 4 | 3 | 4865 | 30.9 s |

Rules of thumb:

- `L ≤ 1000` — fits in seconds. No action needed.
- `1000 < L ≤ 5000` — tens of seconds; usable, but check `L` first.
- `L > 5000` — minutes to hours. Lower `num_terms` (the biggest lever),
  lower `max_weight`, or cap the search with `max_hyperplanes=...`,
  which keeps an evenly spaced subset of the enumerated directions:

  ```python
  HyperplaneTreeRegressor(num_terms=3, max_weight=2, max_hyperplanes=500)
  ```

Sample count `N` enters only linearly (through the candidate search and
the Gram matrices), so widening the *lift* is far more expensive than
adding rows.

## Reproducing

The benchmark suite (skipped in regular CI) is `tests/test_performance.py`.
Run with:

```bash
pytest tests/test_performance.py --benchmark-only
```

The cost-model claims above are pinned by
`tests/test_hyperplane_scaling.py`: `count_lifted_features` is checked
against the transform the fit actually builds, and the compiled split
kernel's flop count is asserted not to move when the lifted width does.
Run the wall-clock half with `pytest -m perf`.

## Where speed comes from

- **Inner kernel JIT.** `_evaluate_splits_kernel` is JIT-compiled, so
  each candidate-split evaluation is one compiled kernel call rather
  than a chain of Python ops.
- **Vmapped ridge fits.** Per-candidate ridge regressions are batched
  via `jax.vmap`, fusing K·B fits into one call.
- **Reduced-space candidate solves.** Each candidate fit is solved in
  the `m = n_features + 1` dimensional space the lifted design matrix
  actually spans, not in the `n_aug`-dimensional lifted space — an exact
  algebraic reformulation (see
  [Algorithm](algorithm.md#candidate-fits-live-in-the-input-space-not-the-lifted-space)).
  This is what makes candidate cost independent of the lift width.
- **Mask-based subdomain selection.** No Python-level slicing /
  reallocation between splits — the kernel sees the full (N, n_aug)
  tensor and a (N,) mask.
- **One solve per node.** A node's regression is fitted when the node is
  created and reused as both its impurity baseline and its leaf model.

## Where it slows down

- **Tree growth Python loop.** Topology is data-dependent and not
  JIT-compatible. Each split call re-enters Python.
- **First split has JIT compile cost** (~0.5 s on CPU). Re-fitting trees
  with the same hyperparameters reuses the cached compilation.
- **Wide lifts.** The per-node leaf solve is still cubic in the lifted
  width (those coefficients are the exported model, so they are computed
  densely for reproducibility). Past ~1000 lifted features it dominates
  — see the cost model above.
