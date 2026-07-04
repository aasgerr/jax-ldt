"""Calibration / sharpness metrics.

Lazy import of `uncertainty-toolbox` so it stays a soft dependency.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np


def calibration_metrics(
    y_pred: jnp.ndarray,
    uncertainty: jnp.ndarray,
    y_true: jnp.ndarray,
    *,
    num_bins: int = 100,
    resolution: int = 99,
) -> dict[str, Any]:
    """Compute accuracy / sharpness / calibration / scoring metrics.

    Lazy-imports `uncertainty-toolbox`; raises a clear error with install
    instructions if missing.
    """
    try:
        import uncertainty_toolbox as utb
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "calibration_metrics requires the `uncertainty-toolbox` package. "
            "Install with `pip install jax-ldt[uq-metrics]`."
        ) from exc

    y_pred = np.asarray(y_pred)
    uncertainty = np.asarray(uncertainty)
    y_true = np.asarray(y_true)
    # uncertainty-toolbox is single-target only; silently flattening a
    # (N, T) array to length N*T destroys per-target semantics and can
    # even pass by coincidence when the flattened lengths happen to match.
    for name, arr in (("y_pred", y_pred), ("uncertainty", uncertainty), ("y_true", y_true)):
        if arr.ndim > 1 and arr.shape[-1] > 1:
            raise ValueError(
                f"calibration_metrics is single-target only; {name} has shape "
                f"{arr.shape}. Call it once per target column."
            )
    y_pred = y_pred.reshape(-1)
    uncertainty = uncertainty.reshape(-1)
    y_true = y_true.reshape(-1)
    # utb's NLL / calibration divide by the std, so a non-positive
    # uncertainty (linprop can legitimately return exact 0) blows up.
    if np.any(uncertainty <= 0.0):
        raise ValueError(
            "calibration_metrics requires strictly positive uncertainty "
            "(uncertainty-toolbox divides by the std). Got a value <= 0; "
            "clamp with a small floor or use a predictive-variance UQ mode."
        )

    metrics: dict[str, Any] = {}
    metrics["accuracy"] = utb.metrics_accuracy.get_all_accuracy_metrics(y_pred, y_true)
    metrics["sharpness"] = {
        "rms_unc": float(np.sqrt(np.mean(uncertainty**2))),
        "mean_unc": float(np.mean(uncertainty)),
        "median_unc": float(np.median(uncertainty)),
        "75_percentile_unc": float(np.quantile(uncertainty, 0.75)),
        "99_percentile_unc": float(np.quantile(uncertainty, 0.99)),
    }
    metrics["calibration"] = utb.metrics_calibration.get_all_calibration_metrics(
        y_pred, uncertainty, y_true, num_bins=num_bins, resolution=resolution
    )
    metrics["scoring_rule"] = utb.metrics_scoring_rule.get_all_scoring_rule_metrics(
        y_pred, uncertainty, y_true
    )
    return metrics
