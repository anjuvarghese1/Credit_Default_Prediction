# custom defined transformers for data cleaning

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import DELINQUENCY_VOIDS, DELINQUENCY_CAP

# replace the delinquency voids (96/98) with Nan
class SentinelReplacer(BaseEstimator, TransformerMixin):
    def __init__ (self, voids: tuple[int, ...] = delinquency_voids) -> None:
    self.voids = voids

  def fit(self, X, y=None):
    self.feature_names_in = _column_names(X)
    self.n_features_in_ = _n_features(X)
    return self

  def transform(self, X):
    X = _as_frame(X, names=getattr(self, "feature_names_in_", None)).copy()
    X = X.replace(list(self.voids), np.nan)
    return X

  def get_feature_names_out(self, input_features=None):
    return _resolve_names(self, input_features)

# prune outliers
class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        upper_quantile: float = 0.99,
        hard_caps: dict[str, float] | None = None,
    ) -> None:
        self.upper_quantile = upper_quantile
        self.hard_caps = hard_caps

    @property
    def _caps(self) -> dict[str, float]:
        return self.hard_caps or {}

    def fit(self, X, y=None):
        X = _as_frame(X)
        # Learn the upper bound for every column from TRAIN data only
        self.upper_bounds_ = X.quantile(self.upper_quantile)
        self.columns_ = list(X.columns)
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = _as_frame(X, names=getattr(self, "feature_names_in_", None)).copy()
        caps = self._caps
        for col in self.columns_:
            upper = self.upper_bounds_[col]
            if col in caps:
                upper = min(upper, caps[col])
            X[col] = X[col].clip(upper=upper)
        return X

    def get_feature_names_out(self, input_features=None):
        return _resolve_names(self, input_features)

# create dataframe
def _as_frame(X, names: list[str] | None = None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if names is not None and len(names) == arr.shape[1]:
        cols = list(names)
    else:
        cols = [f"x{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols)


def _n_features(X) -> int:
    arr = np.asarray(X)
    return 1 if arr.ndim == 1 else arr.shape[1]


def _column_names(X) -> list[str] | None:
    if isinstance(X, pd.DataFrame):
        return list(X.columns)
    return None


def _resolve_names(estimator, input_features):
    if input_features is not None:
        return np.asarray(input_features, dtype=object)
    captured = getattr(estimator, "feature_names_in_", None)
    if captured is not None:
        return np.asarray(captured, dtype=object)
    raise ValueError(
        "Cannot resolve output feature names: no input_features supplied and "
        "none captured at fit time."
    )


def delinquency_hard_caps(columns: list[str]) -> dict[str, float]:
    return {col: float(DELINQUENCY_CAP) for col in columns}
