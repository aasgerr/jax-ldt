"""Regression tests for FIX_PLAN.md M7-M10 (active-learning correctness)."""

from __future__ import annotations

import numpy as np
import pytest

from jax_ldt import (
    ActiveLearner,
    DiverseBatchSelector,
    GreedyMaxMinBatchSelector,
    LinearTreeRegressor,
    TopKBatchSelector,
    UncertaintySampler,
)


def _model_factory():
    return LinearTreeRegressor(max_depth=3, max_bins=6, min_samples_leaf=5)


def _bounds_2d():
    return np.array([[-2.0, 2.0], [-2.0, 2.0]])


# --- M7: pool mode must not re-propose observed / pending points ---


def _pool_loop(pool, batch_size=4, seed=0):
    return ActiveLearner(
        model_factory=_model_factory,
        acquisition=UncertaintySampler(),
        batcher=TopKBatchSelector(),
        bounds=_bounds_2d(),
        batch_size=batch_size,
        seed=seed,
        candidate_pool=pool,
    )


def test_two_asks_without_tell_are_disjoint(rng) -> None:
    pool = rng.uniform(-2, 2, size=(40, 2))
    loop = _pool_loop(pool)
    X0 = rng.uniform(-2, 2, size=(10, 2))
    loop.tell(X0, (X0**2).sum(1))
    _, b1 = loop.ask()
    _, b2 = loop.ask()
    s1 = {tuple(r) for r in b1}
    s2 = {tuple(r) for r in b2}
    assert s1.isdisjoint(s2), "two asks without tell returned overlapping points"


def test_told_points_do_not_reappear(rng) -> None:
    pool = rng.uniform(-2, 2, size=(40, 2))
    loop = _pool_loop(pool)
    X0 = rng.uniform(-2, 2, size=(10, 2))
    loop.tell(X0, (X0**2).sum(1))
    bid, batch = loop.ask()
    loop.tell(bid, (batch**2).sum(1))
    _, batch2 = loop.ask()
    told = {tuple(r) for r in batch}
    assert told.isdisjoint({tuple(r) for r in batch2})


def test_exhausted_pool_raises(rng) -> None:
    pool = rng.uniform(-2, 2, size=(6, 2))
    loop = _pool_loop(pool, batch_size=4)
    loop.tell(pool[:4], (pool[:4] ** 2).sum(1))  # 4 of 6 observed
    # Only 2 unlabeled remain < batch_size 4 -> must raise.
    with pytest.raises(RuntimeError, match="pool exhausted"):
        loop.ask()


# --- M8: tell() index validation ---


def _seeded_loop():
    loop = ActiveLearner(
        model_factory=_model_factory,
        acquisition=UncertaintySampler(),
        batcher=TopKBatchSelector(),
        bounds=_bounds_2d(),
        batch_size=4,
        seed=1,
    )
    rng = np.random.default_rng(0)
    X0 = rng.uniform(-2, 2, size=(15, 2))
    loop.tell(X0, (X0**2).sum(1))
    return loop


def test_tell_duplicate_indices_raise() -> None:
    loop = _seeded_loop()
    bid, batch = loop.ask()
    with pytest.raises(ValueError, match="duplicate indices"):
        loop.tell(bid, np.zeros(4), indices=[0, 0, 1, 1])


def test_tell_negative_index_raises() -> None:
    loop = _seeded_loop()
    bid, batch = loop.ask()
    with pytest.raises(ValueError, match=r"indices must be in"):
        loop.tell(bid, np.zeros(1), indices=[-1])


def test_tell_out_of_range_index_raises() -> None:
    loop = _seeded_loop()
    bid, batch = loop.ask()
    with pytest.raises(ValueError, match=r"indices must be in"):
        loop.tell(bid, np.zeros(1), indices=[999])


# --- M9: DiverseBatchSelector uniqueness + seed control ---


def test_diverse_selector_returns_unique_indices(rng) -> None:
    # Duplicate rows + tie-rich scores stress the empty-cluster fallback.
    X = rng.uniform(-1, 1, size=(30, 2))
    X = np.vstack([X, X[:5]])  # exact duplicates
    scores = np.ones(len(X))
    scores[:5] = 5.0  # a few high-scorers
    sel = DiverseBatchSelector(pool_factor=3, seed=0)
    idx = sel.select(X, scores, k=8)
    assert len(idx) == len(set(idx.tolist())), "selector returned duplicate indices"


def test_diverse_selector_seed_is_respected(rng) -> None:
    """Pre-fix the RNG was hardcoded default_rng(0), so seed had no effect
    and every call was identical. Across several seeds the k-means++
    seeding must produce more than one distinct selection, and a fixed
    seed must be reproducible."""
    X = rng.uniform(-1, 1, size=(60, 2))
    scores = rng.uniform(0, 1, size=60)
    results = {
        tuple(sorted(DiverseBatchSelector(pool_factor=5, seed=s).select(X, scores, 6).tolist()))
        for s in range(8)
    }
    assert len(results) >= 2, "seed had no effect on selection"
    # Reproducibility: same seed -> same result.
    r1 = DiverseBatchSelector(pool_factor=5, seed=3).select(X, scores, 6)
    r2 = DiverseBatchSelector(pool_factor=5, seed=3).select(X, scores, 6)
    np.testing.assert_array_equal(r1, r2)


# --- M10: GreedyMaxMin scale actually spreads points ---


def test_greedy_maxmin_covers_both_clusters(rng) -> None:
    """Two tight clusters, all high scores in cluster 1. Plain top-k stays
    in cluster 1; the diversity term must pull at least one pick into
    cluster 2. With scale saturating tanh (the pre-fix behavior) the
    diversity term is nearly constant and this coverage is lost."""
    c1 = rng.normal([-3, -3], 0.05, size=(40, 2))
    c2 = rng.normal([3, 3], 0.05, size=(40, 2))
    X = np.vstack([c1, c2])
    scores = np.r_[np.linspace(1, 0.9, 40), np.linspace(0.89, 0.8, 40)]

    def clusters(idx):
        return {"c1" if i < 40 else "c2" for i in idx}

    topk = TopKBatchSelector().select(X, scores, 6)
    greedy = GreedyMaxMinBatchSelector(diversity_weight=0.7, scale=1.0).select(
        X, scores, 6
    )
    assert clusters(topk) == {"c1"}
    assert clusters(greedy) == {"c1", "c2"}
