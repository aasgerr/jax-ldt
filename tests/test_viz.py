"""Smoke tests for `jax_ldt.viz` plotting helpers.

These tests import matplotlib lazily via `pytest.importorskip` so the
package's optional dependency on matplotlib is not promoted to a
required test dependency.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_plot_tree_partition_2d_returns_axes(branin_2d) -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)

    from jax_ldt import LinearTreeRegressor, plot_tree_partition_2d

    X, y = branin_2d
    model = LinearTreeRegressor(max_depth=3, max_bins=6, min_samples_leaf=20).fit(X, y)

    fig, ax = plt.subplots()
    out = plot_tree_partition_2d(model, X, ax=ax)
    assert isinstance(out, matplotlib.axes.Axes)
    plt.close(fig)


def test_plot_tree_partition_2d_default_ax(branin_2d) -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)

    from jax_ldt import LinearTreeRegressor, plot_tree_partition_2d

    X, y = branin_2d
    model = LinearTreeRegressor(max_depth=2, max_bins=5, min_samples_leaf=20).fit(X, y)
    out = plot_tree_partition_2d(model, X)
    assert isinstance(out, matplotlib.axes.Axes)
    plt.close(out.figure)


def test_plot_calibration_basic(rng) -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)

    from jax_ldt import plot_calibration

    y_true = rng.normal(size=50)
    y_pred = y_true + 0.1 * rng.normal(size=50)
    fig, ax = plt.subplots()
    out = plot_calibration(y_true, y_pred, ax=ax)
    assert isinstance(out, matplotlib.axes.Axes)
    plt.close(fig)


def test_plot_calibration_with_sigma_and_intervals(rng) -> None:
    plt = pytest.importorskip("matplotlib.pyplot")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)

    from jax_ldt import plot_calibration

    y_true = rng.normal(size=40)
    y_pred = y_true + 0.05 * rng.normal(size=40)
    sigma = 0.1 * np.ones_like(y_pred)
    lo = y_pred - 0.2
    hi = y_pred + 0.2

    out = plot_calibration(y_true, y_pred, sigma=sigma, intervals=(lo, hi))
    assert isinstance(out, matplotlib.axes.Axes)
    plt.close(out.figure)


def test_plot_calibration_band_x_matches_y_true(rng) -> None:
    """M14: the prediction band must be drawn on the y_true x-axis (where
    the points sit), not the y_pred axis."""
    plt = pytest.importorskip("matplotlib.pyplot")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)

    from jax_ldt import plot_calibration

    y_true = rng.normal(size=30)
    y_pred = y_true + 0.5 * rng.normal(size=30)  # meaningful deviation
    lo = y_pred - 0.2
    hi = y_pred + 0.2

    ax = plot_calibration(y_true, y_pred, intervals=(lo, hi))
    # The fill_between poly's x-vertices must span sorted y_true, not y_pred.
    coll = ax.collections[0]
    xs = coll.get_paths()[0].vertices[:, 0]
    assert abs(xs.min() - y_true.min()) < 1e-9
    assert abs(xs.max() - y_true.max()) < 1e-9
    plt.close(ax.figure)
