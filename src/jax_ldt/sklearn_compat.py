"""Optional sklearn-compatible wrapper around :class:`LinearTreeRegressor`.

This module is opt-in: importing it requires scikit-learn at runtime.
The core jax-ldt package has no sklearn dependency. The wrapper exists
for users who want to plug a jax-ldt regressor into sklearn machinery
such as :func:`sklearn.model_selection.cross_val_score`,
:class:`sklearn.pipeline.Pipeline`, or
:class:`sklearn.model_selection.GridSearchCV`.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from jax_ldt.linear_tree import LinearTreeRegressor


def _require_sklearn():
    try:
        import sklearn  # type: ignore  # noqa: F401
        from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised by import-guard test
        raise ImportError(
            "jax_ldt.sklearn_compat requires scikit-learn. Install it with "
            "`pip install scikit-learn`."
        ) from exc
    return BaseEstimator, RegressorMixin


# Resolve the sklearn base classes at import time. We accept the cost of
# this eager import (this module is itself opt-in) so the resulting class
# inherits from the genuine sklearn bases — which is what
# `check_estimator`, `cross_val_score`, and friends look for.
BaseEstimator, RegressorMixin = _require_sklearn()


class _SklearnTreeBase(BaseEstimator, RegressorMixin):
    """Shared fit/predict plumbing for the sklearn wrappers.

    Subclasses store their hyperparameters verbatim in ``__init__`` (so
    ``clone`` / ``get_params`` work) and implement ``_make_estimator``.
    Scoring is inherited from :class:`sklearn.base.RegressorMixin`, whose
    ``score`` computes the uniform-average R² for multi-output targets —
    the sklearn convention. (A prior hand-rolled ``score`` flattened y and
    computed a pooled R², diverging from sklearn and re-ranking
    hyperparameters under GridSearchCV.)
    """

    def _make_estimator(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, X, y):
        from sklearn.utils.validation import check_X_y

        # check_X_y raises on 1-D X ("Reshape your data"), non-finite
        # values, and length mismatch — the sklearn input contract.
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True, multi_output=True)
        self._estimator = self._make_estimator()
        self._estimator.fit(X, y)
        self.n_features_in_ = int(self._estimator.n_features_in_)
        return self

    def predict(self, X):
        from sklearn.exceptions import NotFittedError
        from sklearn.utils.validation import check_array, check_is_fitted

        # Raise the sklearn-canonical NotFittedError (a RuntimeError does
        # not satisfy check_is_fitted-based meta-estimators / checks).
        try:
            check_is_fitted(self)
        except NotFittedError:
            raise
        X = check_array(X, dtype=np.float64)
        return np.asarray(self._estimator.predict(X))

    # ------------------------------------------------------------------
    # Tag overrides for sklearn's estimator-check machinery.
    # sklearn 1.6 deprecated `_more_tags` in favour of
    # `__sklearn_tags__()`; both are implemented for cross-version
    # compatibility (sklearn falls back to whichever it knows about).
    # ------------------------------------------------------------------
    def _more_tags(self) -> dict[str, Any]:
        return {
            "requires_y": True,
            "allow_nan": False,
            "X_types": ["2darray"],
        }

    def __sklearn_tags__(self):  # type: ignore[no-untyped-def]
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.input_tags.allow_nan = False
        return tags

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "_estimator") and self._estimator is not None


class SklearnLinearTreeRegressor(_SklearnTreeBase):
    """sklearn-compatible wrapper around :class:`LinearTreeRegressor`.

    Useful for :func:`sklearn.model_selection.cross_val_score`,
    :class:`sklearn.pipeline.Pipeline`,
    :class:`sklearn.model_selection.GridSearchCV`, and
    :func:`sklearn.base.clone`. All hyperparameters of the underlying
    :class:`LinearTreeRegressor` are forwarded as constructor keywords.
    """

    def __init__(
        self,
        criterion: str = "mae",
        max_depth: int = 32,
        min_samples_split: Union[int, float] = 6,
        min_samples_leaf: Union[int, float] = 0.01,
        max_bins: int = 25,
        min_impurity_decrease: float = 0.0,
        categorical_features: Optional[tuple] = None,
        split_features: Optional[tuple] = None,
        linear_features: Optional[tuple] = None,
        ridge: float = 1e-5,
        depth_first: bool = True,
    ) -> None:
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_bins = max_bins
        self.min_impurity_decrease = min_impurity_decrease
        self.categorical_features = categorical_features
        self.split_features = split_features
        self.linear_features = linear_features
        self.ridge = ridge
        self.depth_first = depth_first

    def _make_estimator(self):
        return LinearTreeRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_bins=self.max_bins,
            min_impurity_decrease=self.min_impurity_decrease,
            categorical_features=self.categorical_features,
            split_features=self.split_features,
            linear_features=self.linear_features,
            ridge=self.ridge,
            depth_first=self.depth_first,
        )


class SklearnHyperplaneTreeRegressor(_SklearnTreeBase):
    """sklearn-compatible wrapper around :class:`HyperplaneTreeRegressor`.

    Exposes the hyperplane knobs (``num_terms``, ``max_weight``, …) as
    tunable constructor keywords so the flagship oblique model can be
    used with ``GridSearchCV`` / ``Pipeline`` / ``clone``.
    """

    def __init__(
        self,
        criterion: str = "mae",
        max_depth: int = 32,
        min_samples_split: Union[int, float] = 6,
        min_samples_leaf: Union[int, float] = 0.01,
        max_bins: int = 10,
        min_impurity_decrease: float = 0.0,
        categorical_features: Optional[tuple] = None,
        split_features: Optional[tuple] = None,
        linear_features: Optional[tuple] = None,
        ridge: float = 1e-5,
        depth_first: bool = True,
        num_terms: int = 2,
        max_weight: int = 1,
        do_symmetrize: bool = True,
        do_scaling: bool = True,
        tol_decimals: int = 4,
        hyperplane_weights=None,
    ) -> None:
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_bins = max_bins
        self.min_impurity_decrease = min_impurity_decrease
        self.categorical_features = categorical_features
        self.split_features = split_features
        self.linear_features = linear_features
        self.ridge = ridge
        self.depth_first = depth_first
        self.num_terms = num_terms
        self.max_weight = max_weight
        self.do_symmetrize = do_symmetrize
        self.do_scaling = do_scaling
        self.tol_decimals = tol_decimals
        self.hyperplane_weights = hyperplane_weights

    def _make_estimator(self):
        from jax_ldt.hyperplane_tree import HyperplaneTreeRegressor

        return HyperplaneTreeRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_bins=self.max_bins,
            min_impurity_decrease=self.min_impurity_decrease,
            categorical_features=self.categorical_features,
            split_features=self.split_features,
            linear_features=self.linear_features,
            ridge=self.ridge,
            depth_first=self.depth_first,
            num_terms=self.num_terms,
            max_weight=self.max_weight,
            do_symmetrize=self.do_symmetrize,
            do_scaling=self.do_scaling,
            tol_decimals=self.tol_decimals,
            hyperplane_weights=self.hyperplane_weights,
        )
