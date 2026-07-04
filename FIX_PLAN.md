# FIX_PLAN — remediation plan for the 2026-07 full-codebase review

This plan is written to be executed by an autonomous agent. It covers
every finding from the review: 5 high-severity correctness bugs, ~14
medium items, ~6 performance items, and a tail of low/edge items.

**Non-negotiable workflow for every issue:**

1. **CONFIRM** — reproduce the issue first (a failing test or a
   short script) before touching the fix. If you cannot reproduce it,
   stop and record why in the commit message instead of "fixing" it.
2. **FIX** — smallest change that resolves the confirmed behavior.
3. **PROTECT** — add a regression test that fails on the pre-fix code
   and passes after. Put the reproduction from step 1 into the test
   when practical.
4. Run the full suite (`pytest tests/ -q`) after each phase, not just
   the new tests. `pytest -m parity` additionally needs `[parity]`
   extras (torch) — run it at least once after Phase 1 since Phase 1
   intentionally changes numerics.

**Environment setup:**

```bash
uv venv .venv
uv pip install -e ".[dev,onnx,viz,sklearn]"
.venv/bin/python -m pytest tests/ -q     # baseline: 133 passed, 11 skipped
```

**Commit discipline:** one commit per issue (or per tightly-related
group), message format `fix(<area>): <issue id> — <summary>`. Update
`GAPS.md` when a fix invalidates or resolves an entry (see H1 note on
G-17, which is currently recorded as resolved but is not).

---

## Phase 1 — HIGH: silent wrong answers

### H1. `linear_features` is a no-op (leaf fits always use every column)

- **Where:** `src/jax_ldt/tree_core.py` — `grow_tree` threads
  `linear_features` only into `Tree` metadata (line ~611). All fits
  (`_fit_one_side`, `_initial_leaf`) and `predict` use the full
  `X_aug = add_bias(X_t)`.
- **Confirm:** fit `LinearTreeRegressor(linear_features=(0,))` on
  3-feature data where y depends on features 1 and 2; predictions are
  identical (max |diff| = 0.0) to the unrestricted model. Upstream
  reference subsets explicitly (`tests/_reference/_classes.py:416-421`,
  `:522-524`).
- **Fix (recommended approach — preserves all shapes):** zero out the
  excluded columns of `X_aug` *for fitting only*. Build once in
  `grow_tree`:

  ```python
  lin_mask = np.zeros(n_aug); lin_mask[0] = 1.0          # bias always in
  lin_mask[[f + 1 for f in linear_features]] = 1.0
  X_fit = X_aug * jnp.asarray(lin_mask)[None, :]
  ```

  Use `X_fit` in every ridge fit (`_evaluate_splits` inputs,
  `_initial_leaf`, `_leaf_stats_kernel`'s `yh`). With a zeroed column,
  `XtX` has only `ridge` on that diagonal and `Xty = 0`, so the solved
  coefficient is **exactly 0**. `leaf_params` stays `(n_aug, T)`, so
  `predict`, ONNX export, JSON spec, and the discopt adapter all keep
  working unchanged (a zero coefficient is inert everywhere).
  Prediction inside the split kernel must also use `X_fit` (the loss
  must be evaluated with the restricted model), but routing
  (`X_split`, thresholds) still uses the raw columns.
- **Watch out:** `split_features` and `linear_features` are in
  *transformed* space; for `HyperplaneTreeRegressor` the lifted
  columns beyond `n_features_in` must remain in the leaf model by
  default. Preserve the current default (`all non-categorical
  transformed columns`) exactly; only user-supplied restrictions
  change behavior.
- **Protect:**
  - Test: leaf_params coefficients for excluded features are exactly 0
    at every leaf (LMDT and HT).
  - Test: predictions of `linear_features=(0,)` differ from the
    unrestricted model when y depends on the excluded feature.
  - Test: JSON and ONNX round-trips of a restricted model still match
    `predict` (guards the shape-preservation assumption).
- **Also:** correct `GAPS.md` — G-17 claims this is resolved; move it
  back or annotate it, and note the fix in the changelog section.

### H2. Categorical features receive leaf-model coefficients

- **Where:** same root cause as H1; `linear_tree.py:60-62` documents
  categoricals as "excluded from splitting **and from the linear leaf
  model**", but only split exclusion works.
- **Confirm:** categorical column with codes {0,1,2} gets a leaf slope
  of ~5.0 when y depends on the category (already reproduced in
  review; script pattern in H1 confirm).
- **Fix:** falls out of H1 automatically (categoricals are excluded
  from the *default* `linear_features`). Verify that after H1 the
  default path zeroes categorical coefficients.
- **Protect:** extend `tests/test_categorical_features.py` with a
  leaf-model exclusion assertion: every leaf's coefficient on the
  categorical column is exactly 0, and predictions are invariant to
  relabeling category codes ({0↔3} swap) for points routed to the same
  leaves.

### H3. Library requires x64 but never enables/enforces it; silently
runs float32 with an inverted warning

- **Where:** only `tests/conftest.py:23` and the notebooks enable
  `jax_enable_x64`. `src/jax_ldt/__init__.py` does not.
  `_validation.py:15-31` emits "Casting X from float32 to float64;
  jax-ldt operates in 64-bit" for float64 input that JAX already
  demoted — factually backwards. README.md:87 claims "computation is
  `float64` by default".
- **Confirm:** in a fresh process **without** x64
  (`JAX_ENABLE_X64` unset), fit a model: `tree_.leaf_params.dtype ==
  float32`, warning text is wrong, JAX truncation warnings flood.
- **Fix (decision made — do not re-litigate):** do **not** mutate
  global JAX config on import (that breaks co-resident libraries).
  Instead:
  1. Add `_require_x64()` in `_validation.py`: if
     `not jax.config.jax_enable_x64`, raise `RuntimeError` with an
     actionable message (how to enable via env var or
     `jax.config.update`), called at the top of every public `fit`,
     `predict`, `calibrate`, and `to_onnx`/`embed_in_discopt_model`
     boundary.
  2. Fix `_maybe_warn_non_float64` to inspect the dtype of the
     *original user input* (before `jnp.asarray`), so the warning text
     is truthful; drop the warning entirely when input was float64.
  3. README quickstart + SKILL.md: add the two x64-enabling lines
     before any import of jax_ldt; soften "float64 by default" to
     "float64 (requires `jax_enable_x64`, enforced at fit time)".
- **Protect:** a subprocess-based test (conftest force-enables x64 in
  this process, so spawn `sys.executable -c ...` with a clean env) that
  asserts `fit` raises the clear RuntimeError when x64 is off. A second
  subprocess test asserts the error message contains the remediation
  string.

### H4. `rmse` / `max_abs` split criteria don't compute what they name

- **Where:** `src/jax_ldt/tree_core.py:198-206` — candidate loss is
  `(loss_b·n_b + loss_a·n_a)/n_total`. Exact for mae/msle; wrong for
  rmse (needs pooling under the sqrt) and max_abs (needs
  `max(max_b, max_a)`). Upstream evaluates the loss of the **combined
  piecewise prediction** over the node
  (`tests/_reference/_classes.py:423-427`), correct for all criteria.
- **Confirm:** construct the review's counterexample for `max_abs`:
  candidate A side-maxes (10, n=5)/(0.1, n=95) scores 0.595 under
  current code and beats candidate B (1, n=50)/(1, n=50) scoring 1.0;
  true criterion prefers B. A unit test on `_evaluate_splits_kernel`
  directly (feed hand-built masks) is the cleanest confirmation.
- **Fix:** in `_evaluate_splits_kernel`, replace the weighted-average
  combination with the upstream construction:

  ```python
  yh_combined = yh_below * mb_flat[:, :, None] + yh_above * ma_flat[:, :, None]
  overall_flat = jax.vmap(lambda yh: weighted_loss(y, yh, node_mask))(yh_combined)
  overall_flat = jnp.sum(overall_flat, axis=-1)
  ```

  This is criterion-agnostic (also correct for mae/msle — verify the
  mae parity tests still pass bit-for-bit or within tolerance).
  Keep `n_below`/`n_above` for the validity mask. This also makes the
  `min_impurity_decrease` comparison consistent, since `_initial_leaf`
  already computes a node-level loss.
- **Watch out:** `yh_combined` adds one more (K·B, N, T) temporary;
  coordinate with P1 (memory) — you can subtract instead of
  materializing a third tensor by reusing one buffer, or fold this into
  the P1 chunking. Correctness first, then optimize.
- **Protect:**
  - Kernel-level test pinning the max_abs counterexample (candidate B
    must win).
  - rmse test: combined loss equals
    `sqrt((n_b·rmse_b² + n_a·rmse_a²)/n)` on a hand-computed example.
  - If torch is available in CI, extend
    `tests/test_parity_vs_pytorch.py` to `criterion in
    ("mae", "rmse", "max_abs")`; otherwise add a numpy re-implementation
    parity check so it runs in the default lane.

### H5. discopt: explicit `big_m` is misused as a feature-box half-width

- **Where:** `src/jax_ldt/export/discopt_adapter.py:223-266` — passing
  `big_m` skips bounds derivation and sets `lb_t/ub_t = ∓big_m`; then
  `m_left = max(ub_t[f] − t, 0) + 1e-3` (line ~287). Any
  `big_m < max|threshold|` silently makes whole subtrees infeasible.
- **Confirm:** build a tree on features in [0, 1000] with a threshold
  near 800, embed with `big_m=500`, minimize y with x fixed (via
  constraints) in the right subtree → solver reports infeasible or a
  wrong optimum vs `model.predict`.
- **Fix:** treat `big_m` as an actual big-M constant:
  - When `big_m` is given: `m_left = m_right = m_k = float(big_m)` for
    every node/leaf; do **not** fabricate `lb_t/ub_t` from it. Still
    attempt bounds derivation, and if derived requirements exceed the
    user's `big_m`, raise `ValueError` (silently wrong MIPs are the
    worst outcome; loud is correct here).
  - When bounds are available (no `big_m`): current per-node derivation
    stays.
- **Protect:**
  - Two-sided fidelity test: for several x points fixed via equality
    constraints (including deep-left and deep-right leaves), assert
    `solve(y) == model.predict(x)` within tolerance — this replaces the
    current one-sided `objective <= in_sample_min + 1e-3` assertion
    (keep it, but add the two-sided one).
  - Test that an undersized `big_m` raises.
  - Mark tests `skipif discopt missing` consistent with existing suite.

---

## Phase 2 — MEDIUM: correctness and API contract

### M1. MIP boundary ambiguity — no epsilon on the right branch

- **Where:** `discopt_adapter.py:292-293` — left `x'_f ≤ t + M(1−Σz_L)`,
  right `t − x'_f ≤ M(1−Σz_R)`; at `x'_f = t` both leaves feasible, and
  the optimizer exploits it (thresholds are data quantiles, optima land
  on them often).
- **Confirm:** 1-D tree with two leaves of different slopes; minimize y
  with x fixed at exactly the threshold → solver picks the lower leaf,
  `predict` uses the left leaf; values differ.
- **Fix:** add a small epsilon on the right branch:
  `t + eps − x'_f ≤ M(1−Σz_R)` with `eps` configurable
  (default ~1e-6·scale, documented caveat that MIP optima within eps of
  a threshold are boundary-ambiguous). Include eps in `m_right`.
- **Protect:** the confirm scenario as a test (predict/solve agree at
  a threshold point).

### M2. Solver unbounded sentinels (±1e20) pass the finiteness check

- **Where:** `discopt_adapter.py:228-236`; discopt uses ±1e20 for
  unbounded (see `tests/test_discopt_adapter.py:73-75`).
- **Fix:** treat `|bound| > 1e10` as "no usable bounds" → existing
  fallback + warning path.
- **Protect:** test with a mock `x_vars` exposing ±1e20 bounds →
  asserts the fallback warning fires and derived Ms are ≤ the fallback.

### M3. Conformal per-leaf quantile silently clipped for small leaves

- **Where:** `uncertainty/conformal.py:163-177`; global path warns
  (135-144), per-leaf path doesn't. `min_calibration_per_leaf=5`
  default is not tied to alpha (needs `ceil(1/alpha)−1` points, i.e. 9
  at alpha=0.1).
- **Confirm:** mondrian calibration with a leaf holding 5 points at
  alpha=0.1 → q_level clips to 1.0, no warning.
- **Fix:** (a) warn (once, with leaf ids, matching the existing
  truncated-list style) whenever any leaf's `raw_q_level > 1`;
  (b) default `min_calibration_per_leaf=None` → resolve to
  `max(5, ceil(1/alpha) − 1)` at calibrate time; explicit user values
  are respected but validated `>= 1`.
- **Protect:** tests for the warning, the alpha-derived default, and
  `min_calibration_per_leaf=0` raising.

### M4. Conformal: empty calibration set → silent all-NaN intervals

- **Where:** `conformal.py:133-148` (divide-by-zero → inf → NaN).
- **Fix:** raise `ValueError("calibration set is empty")` when N=0 in
  `calibrate`; also validate in `fit_with_conformal` that the split
  leaves ≥1 row on each side (its `n=2, calibration_size=0.9` case
  currently gives `model.fit` zero rows).
- **Protect:** tests for both raises.

### M5. Conformal `predict_interval(X, model=...)` mixes stale state

- **Where:** `conformal.py:203-232` — predictions from the passed
  model, halfwidths/routing from the calibration-time tree.
- **Fix:** if `model.tree_ is not self._tree`, raise (or at minimum
  warn) that the calibrator must be re-calibrated after refit.
- **Protect:** test: calibrate, refit the model on different data, call
  `predict_interval(X, model=model)` → raises/warns.

### M6. QuadraticUQ: zero-calibration leaves report σ = |prediction|

- **Where:** `uncertainty/quadratic.py:83-95,120-123`.
- **Confirm:** deep tree + small calibration set missing ≥1 leaf;
  σ at those leaves ≈ |yh|.
- **Fix:** track leaves with zero calibration points; for them fall
  back to the global mean σ (mirror conformal's
  `sparse_leaf_strategy="global"`) and warn with the same style; also
  add a minimum-points guard: leaves with
  `n_cal < n_quad` (underdetermined refit) use the same fallback
  (fixes the |lin − quad| blowup too).
- **Protect:** tests for both fallback paths (uncovered leaf; leaf with
  `3 < n_cal < n_quad`) asserting finite, y-scale-reasonable σ and the
  warning.

### M7. ActiveLearner `ask()` re-proposes observed/pending points (pool mode)

- **Where:** `active_learning/loop.py:185-208`.
- **Confirm:** fixed `candidate_pool`, two `ask()` calls without
  `tell()` → identical X batches.
- **Fix:** in pool mode, exclude (by row identity within the pool)
  indices already in `X_train` and in pending batches before scoring;
  raise a clear error if the remaining pool is smaller than the batch.
  For the Sobol path, no change (fresh key per ask).
- **Protect:** tests: (a) two asks without tell are disjoint;
  (b) told points never reappear across rounds; (c) exhausted pool
  raises.

### M8. ActiveLearner `tell()` accepts duplicate / negative indices

- **Where:** `loop.py:156-174`.
- **Fix:** validate `indices`: within `[0, len(pending.X))`, no
  duplicates within the call, none already answered (last one exists).
- **Protect:** tests for `[0,0,1,1]` and `[-1]` raising `ValueError`.

### M9. DiverseBatchSelector: duplicate picks + hardcoded RNG + dead param

- **Where:** `active_learning/batching.py:99,145-157,86`.
- **Fix:** (a) non-empty-cluster branch must skip candidates in `used`
  (fall through to that cluster's next-best member, then to the global
  fallback); (b) add a `seed: Optional[int] = None` field, use
  `np.random.default_rng(self.seed)`; thread the learner's per-round
  seed through `ask()` where the selector is invoked; (c) remove the
  dead `diversity_weight` field or implement it — removing is fine, it
  is documented as unused.
- **Protect:** test constructing the empty-cluster→duplicate scenario
  (assert all returned indices unique — a property test over random
  pools is acceptable); test that different seeds change selection on a
  tie-rich pool.

### M10. GreedyMaxMinBatchSelector: tanh saturation / docstring mismatch

- **Where:** `batching.py:57-67`.
- **Fix:** add `scale: float = 1.0` and use `tanh(dist/scale)`; fix the
  docstring to describe the actual formula (delete the two formulas it
  promises but doesn't implement).
- **Protect:** test that with small `scale` the selector spreads points
  (min pairwise distance of the batch > that of top-k) on a clustered
  pool.

### M11. sklearn wrapper conformance

- **Where:** `sklearn_compat.py:120-121` (RuntimeError),
  `:132-138` (pooled R²), missing HT wrapper.
- **Fix:** (a) raise `sklearn.exceptions.NotFittedError`; (b) delete
  the `score` override — `RegressorMixin.score` already handles
  multioutput correctly; (c) add `SklearnHyperplaneTreeRegressor`
  mirroring the LMDT wrapper (constructor params = HT dataclass fields,
  `get_params`/`set_params` inherited from `BaseEstimator`), export via
  the same lazy `__getattr__` in `jax_ldt/__init__.py`.
- **Protect:** run
  `sklearn.utils.estimator_checks.check_estimator` on both wrappers in
  a test (xfail-mark specific checks only with a written reason);
  multi-target `score` test comparing to
  `sklearn.metrics.r2_score(..., multioutput="uniform_average")`.

### M12. Spurious dedup warning on every default HT fit

- **Where:** `hyperplanes.py:29-49` (fires at 136 and 228) — ±-parity
  expansion guarantees ≥50% exact duplicates, so the >10% threshold
  always trips.
- **Confirm:** default `HyperplaneTreeRegressor().fit` emits two
  UserWarnings (visible in current test-suite output).
- **Fix:** warn only on *rounding-induced* collapse: compare the number
  of unique rows at full precision vs after `round(tol_decimals)`; warn
  when rounding itself drops >10%. Exact duplicates from
  symmetrization are by-design and must not count.
- **Protect:** test that a default fit is **warning-free**
  (`warnings.catch_warnings(record=True)` / `pytest.warns(None)`
  equivalent), and that a genuinely-collapsing case (directions
  differing at the 5th decimal with `tol_decimals=4`) still warns. The
  existing `test_dedup_warning_fires_when_aggressive_rounding` passes
  vacuously today — tighten it.

### M13. `do_symmetrize=False` silently drops legitimate LCs

- **Where:** `hyperplanes.py:192-194` — `keep = abs_sums != 1.0`
  assumes symmetrize-normalized rows; `[[0.5, 0.5]]` is discarded and
  the tree silently degrades to axis-aligned.
- **Fix:** replace the abs-sum heuristic with an explicit test for
  "axis-aligned duplicate of the identity block": drop a row iff it has
  exactly one nonzero entry. Normalize user rows (leading-nonzero → 1)
  before dedup so float-equality is meaningful.
- **Protect:** test `build_transform_matrix(X, LCs=[[0.5, 0.5]],
  do_symmetrize=False)` yields a 4th column ∝ (1,1); test one-hot rows
  are still dropped.

### M14. `plot_calibration` band drawn at wrong x-coordinates

- **Where:** `viz.py:99-128` — band at x=y_pred, points at x=y_true.
- **Fix:** draw the band in y_true order
  (`fill_between(y_true[order], lo[order], hi[order])`) so it brackets
  the plotted points.
- **Protect:** matplotlib-Agg test asserting the fill path's x-vertices
  equal sorted y_true (the existing viz tests show the pattern).

### M15. ONNX `ir_version` pinned to the installed lib's newest

- **Where:** `export/onnx_export.py:246-252` — comment claims
  `make_model` picks the minimum ir_version; it actually sets
  `onnx.IR_VERSION`.
- **Fix:** set explicitly:
  `model.ir_version = onnx.helper.find_min_ir_version_for(opset_imports)`.
  Also validate `opset >= 13` up front with a clear `ValueError`
  (Squeeze/Unsqueeze axes-as-input requires it).
- **Protect:** test asserting the exported model's `ir_version <=` the
  value required by opset 18 (pin the expected constant), and
  `to_onnx(model, opset=12)` raising `ValueError`.

---

## Phase 3 — performance

Order by measured impact; add a timing note to the commit message for
each (before/after on the benchmark below), and where feasible tighten
`tests/test_performance.py` budgets so regressions get caught. Bench
harness: fit on Friedman-1, N=20_000, 10 features, `max_depth=8`;
predict on N=100_000.

### P1. Split-kernel peak memory is O(K·B·N) (+ H4 adds a tensor)

- **Where:** `tree_core.py:172-192`.
- **Fix:** chunk candidates: `lax.map` over blocks of the flattened
  K·B axis (block size chosen so block·N·T stays ~tens of MB), keeping
  per-block vmap. Verify identical split decisions vs unchunked on a
  seeded problem (exact equality expected — same math, same order).
- **Protect:** correctness: seeded fit equality test vs pre-change
  predictions (tolerance 0); memory is hard to assert portably — add a
  comment-documented manual check instead.

### P2. Growth-loop host syncs and redundant work

- **Where/fixes** (`tree_core.py`):
  - Return `loss_below_flat`/`loss_above_flat` from
    `_evaluate_splits_kernel` and delete `_weighted_loss_python`
    (lines 563-568, 616-631) — the values are already computed.
  - Hoist `X_split = X_t[:, split_feats_arr]` out of the while loop
    (line 513) — loop-invariant.
  - Delete the discarded root fit (line 418) — `root_loss` is needed
    but `root_params` is not; compute the loss lazily on first use or
    keep `_initial_leaf` and pass its params to `_finalize_leaf`
    when the root stops immediately (avoid the double fit at 498).
  - `_fit_leaf`: actually jit it (`_fit_leaf = jax.jit(_fit_one_side)`)
    and fix the false comment at 236-240 (the current alias runs
    eagerly).
  - Batch `int(nm)` conversions: keep `leaf_n` as a device scalar in
    `_GrowNode` and convert once during the final packing loop
    (line ~593), removing one sync per leaf.
- **Protect:** seeded end-to-end equality test (same tree, same
  predictions before/after); existing suite covers behavior.

### P3. `predict`/`apply_tree` unjitted; kernel recompiles when N changes

- **Fix:** wrap `predict`/`apply_tree` bodies in `jax.jit` (Tree is a
  registered pytree with static aux — confirm via a retrace-counter
  test). For the N-recompile: document it, and in
  `ActiveLearner._refit_model` pad `X`/`y` to the next power-of-two row
  count with zero-weight… **do not** — masks are 0/1 sample weights in
  the kernel, but `grow_tree` takes raw X/y. Instead: leave the
  algorithm alone and only document the compile-per-round cost in
  `docs/active_learning.md`; padding inside `grow_tree` (rows with
  node_mask 0 everywhere) is a clean follow-up but touches quantile
  computation — out of scope unless the timing shows it dominating.
- **Protect:** retrace-count test using
  `jax.monitoring`/trace-counter or a `jax.jit`-cache probe: two
  `predict` calls with same-shape X trigger one trace.

### P4. JSON NaN encode/decode is per-element Python

- **Where:** `export/spec.py:54-67,123-131`.
- **Fix:** vectorize: build an object ndarray via
  `arr.astype(object); obj[np.isnan(arr)] = None; obj.tolist()`, and on
  load `np.array([...], dtype=float)` handles `None`→NaN natively
  (verify) or use a mask. Preserve exact float64 repr round-trip
  (existing tests pin it).
- **Protect:** existing round-trip tests + a timing sanity bound is
  optional; add a round-trip test for a tree with NaN thresholds
  (leaf nodes) and multi-target leaf_params.

### P5. `_all_finite` copies whole arrays to host

- **Where:** `_validation.py:34-41`.
- **Fix:** `bool(jnp.isfinite(arr).all())` — O(1) bytes transferred;
  fix the inverted docstring rationale. For numpy inputs keep the numpy
  path (no device round-trip): branch on input type.
- **Protect:** behavior covered by existing validation tests.

### P6. QuadraticUQ fits all nodes with dense masks

- **Where:** `quadratic.py:83-95` (~2.3 GB intermediates; half the
  solves discarded).
- **Fix:** fit only rows for `leaf_ids` (index the mask matrix to
  leaves before vmap) — removes the dead internal-node fits and halves
  the mask; scatter results back to `n_nodes` slots. The (n_leaves, N)
  mask stays but is half the size; the bigger `Xw` intermediate is
  bounded by chunking the vmap the same way as P1 if needed.
- **Protect:** equality test: σ values identical before/after on a
  seeded model. Also fix the `_quadratic_features` docstring shape
  claim (off by F−1) and use `X_aug[:, iu] * X_aug[:, ju]` instead of
  the (N, F, F) outer product while you're there.

---

## Phase 4 — LOW / edge / docs (batch by file)

Each item: confirm with a 3-line repro, fix, add a one-assert test
unless marked (doc).

1. **acquisitions.py** — validate `direction in ("min", "max")` in
   `__post_init__` of EI/PI (a typo silently inverts the campaign).
   Test: `direction="minimize"` raises.
2. **hyperplanes.py** — `num_terms < 1` → `ValueError` (currently
   IndexError); `hyperplane_weights` row width must equal `num_terms`
   or be inferred (pick: infer, matching upstream) — cryptic broadcast
   error today; all-zero LC rows dropped with a warning; out-of-range
   `categorical_features` indices → `ValueError` (also check in
   `linear_tree.py` / `tree_core.grow_tree` for split/linear features).
   Delete dead `_factorial`.
3. **_validation.py** — reject `X.shape[1] == 0` and `y.shape[1] == 0`
   with clear messages.
4. **losses.py** — make `_safe_weighted_mean` return `inf` on zero
   total weight (docstring claims it mirrors the kernel, which returns
   inf; pick inf to actually mirror); document that `max_abs` expects
   0/1 masks (or implement a true weighted max via `where(w>0, |err|,
   -inf)` — pick the `where` fix, it's strictly better).
5. **tree_core.py** — `predict` shape asymmetry: remember
   `y_ndim_in == 2` on the regressor and return `(N, 1)` when trained
   on 2-D single-target y (sklearn convention). Keep `Tree`-level
   `predict` behavior; do the reshape at the regressor layer to avoid
   breaking `tree_core` consumers. Test both cases.
6. **ridge conditioning (doc)** — document in `linear_regression.py`
   and the regressor docstrings that `ridge` is absolute, with the
   guidance to standardize features at scale ≥1e6; do **not** silently
   change to relative ridge (would shift all fitted models). Optionally
   add `ridge_mode="absolute"|"relative"` later — out of scope.
7. **_types.py** — fix `LeafUQ` docstrings (`x_mean`/`x_var` are in
   *transformed* space, length `n_features_transformed`); set
   `eq=False` on the frozen `Tree` dataclass (auto `__eq__`/`__hash__`
   are broken on array fields) — verify pytree flatten/unflatten tests
   still pass.
8. **__init__.py** — add `build_transform_matrix`,
   `generate_planes_to_index`, `lift`, `symmetrize` to `__all__`.
9. **conformal.py (doc)** — document that multi-target intervals are
   per-target marginal (joint box coverage ≈ (1−α)^T); expose
   `min_calibration_per_leaf` and `sparse_leaf_strategy` through
   `fit_with_conformal` kwargs.
10. **metrics.py** — raise a clear `ValueError` on multi-target input
    instead of `reshape(-1)`; clamp/validate `uncertainty > 0` with a
    clear message (linprop can return exact 0 → utb division by zero).
11. **discopt_adapter.py** — `to_discopt_decision_tree`: raise when
    `transform_matrix` is not the identity (silently dropping it
    mis-routes consumers feeding raw x); pass the derived `lb_t/ub_t`
    as bounds on the `x_t` continuous vars (free LP tightening).
12. **spec.py** — pass `allow_nan=False` to `json.dumps` and route
    `leaf_x_mean/x_var/mse` + `transform_matrix` through the same
    NaN→null encoding as thresholds (after P4's vectorization, this is
    cheap); test a NaN in `leaf_mse` round-trips.
13. **sklearn_compat.py** — raise on 1-D X (sklearn "Reshape your
    data" convention) instead of silently reshaping to a column.
14. **linprop.py (decision point — do not silently change numbers):**
    the formula is a faithful upstream port but statistically
    non-standard (leverage not scaled by mse; SSR/n not SSR/(n−p);
    epistemic-only). Fix as follows: add
    `linprop_uncertainty(..., mode="upstream")` default preserving
    current numbers, plus `mode="prediction"` implementing
    `σ² = mse·(1 + 1/n + Σ(x−μ)²/((n−1)σ_x²))` with `mse = SSR/max(n−p, 1)`.
    Document the difference and point `calibration_metrics` docs at
    `mode="prediction"`. Update `docs/uncertainty.md`. Tests for both
    modes; keep `test_linprop_single_sample_leaf_returns_sqrt_mse`
    pinned to `mode="upstream"` and add the n=1 → nonzero-σ test for
    `mode="prediction"`.

---

## Phase 5 — test-coverage backstops (beyond per-fix regression tests)

- `depth_first=False` (breadth-first) growth: seeded fit + predict
  smoke and a shape/route sanity test — currently zero coverage.
- ONNX: evaluate inputs **exactly at split thresholds** and assert
  equality with `predict` (the `LessOrEqual` fidelity the code comments
  emphasize).
- JSON: compare `leaf_uq` (n, x_mean, x_var, mse) field-by-field after
  round-trip, not just predictions; multi-target round-trip.
- discopt: HT (oblique, non-identity transform) embed test;
  multi-target embed test.
- AL: v1 `rng_key` load-compat branch (`loop.py:279-283`);
  numeric EI test against the closed form `imp·Φ(z) + σ·φ(z)`;
  `MaxVariance.score == σ²` numerically.
- Conformal: `predict_interval` without `model=`; mondrian actually
  produces *different* halfwidths across leaves on heteroscedastic
  data (the current mondrian test passes for mondrian=False too).

---

## Execution order & verification gates

1. Phase 1 (H1→H5), each with its regression tests. Gate: full suite
   green **plus** the new subprocess x64 tests **plus** parity lane if
   torch available.
2. Phase 2 (M1→M15). Gate: full suite green; run
   `check_estimator` lane.
3. Phase 3 (P1→P6). Gate: seeded-equality tests green; record
   before/after timings in commit messages.
4. Phase 4, batched by file. Phase 5 alongside or after.
5. Final: update `GAPS.md` (move fixed items to the changelog; add the
   linprop `mode=` decision as a documented design note), regenerate
   nothing in `examples/` unless a fix changes example output
   (H3's README edit does; re-run `examples/_make_notebooks.py` only if
   notebook text references the changed quickstart).

**Known-risk notes for the executor:**

- H1/H4 change fitted models. Any test pinning exact predictions
  (parity, seeded snapshots) may need regeneration — regenerate only
  after confirming the new behavior is the intended one, and say so in
  the commit message.
- Do not fix by mutating global JAX config at import time (H3 decision
  is final).
- Do not switch the ridge to relative scaling (Phase 4 item 6) — doc
  only.
- Anything you cannot reproduce: leave the code alone, add a note to
  `GAPS.md` instead.
