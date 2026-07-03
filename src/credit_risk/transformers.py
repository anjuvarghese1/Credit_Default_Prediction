"""Custom, leakage-safe transformers for credit data cleaning.

Each transformer follows the scikit-learn estimator API (``fit`` / ``transform``)
so it can live inside a :class:`~sklearn.pipeline.Pipeline` and learn its
parameters from *training* data only. This is the mechanism that prevents test
information from leaking into preprocessing -- a subtle but critical
correctness property for any credible modeling pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import DELINQUENCY_CAP, DELINQUENCY_SENTINELS


class SentinelReplacer(BaseEstimator, TransformerMixin):
    """Replace delinquency sentinel codes (96 / 98) with NaN.

    In the raw "Give Me Some Credit" data, the delinquency-count columns use
    96 and 98 to encode "not applicable / unknown" rather than literal counts.
    Left untouched they masquerade as extreme counts and corrupt both scaling
    and model fit. We convert them to NaN here so a downstream imputer can
    handle them coherently with other missing values.

    This transformer is stateless (nothing is learned in ``fit``); it is a
    class rather than a function so it composes inside a Pipeline.
    """

    def __init__(self, sentinels: tuple[int, ...] = DELINQUENCY_SENTINELS) -> None:
        self.sentinels = sentinels

    def fit(self, X, y=None):  # noqa: D102  (sklearn API)
        self.feature_names_in_ = _column_names(X)
        self.n_features_in_ = _n_features(X)
        return self

    def transform(self, X):  # noqa: D102
        X = _as_frame(X, names=getattr(self, "feature_names_in_", None)).copy()
        X = X.replace(list(self.sentinels), np.nan)
        return X

    def get_feature_names_out(self, input_features=None):
        """Names are unchanged by this transformer (passthrough)."""
        return _resolve_names(self, input_features)


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Cap heavy-tailed features at learned quantiles (winsorization).

    Financial ratios such as ``RevolvingUtilizationOfUnsecuredLines`` and
    ``DebtRatio`` have extreme right tails in the real data (values in the
    thousands where the sensible range is roughly [0, 1]). Rather than drop
    rows, we winsorize: the upper bound is the ``upper_quantile`` of each
    column, **learned on the training split only** and then applied to any
    future data. Delinquency counts additionally receive a fixed domain cap.

    Parameters
    ----------
    upper_quantile:
        Quantile used as the per-column upper bound (default 0.99).
    hard_caps:
        Optional mapping of column -> absolute ceiling, applied in addition
        to the learned quantile (used here for delinquency counts).
    """

    def __init__(
        self,
        upper_quantile: float = 0.99,
        hard_caps: dict[str, float] | None = None,
    ) -> None:
        # Store constructor args verbatim. sklearn's clone() requires that
        # get_params() returns exactly what was passed in, so we must NOT
        # coerce None -> {} here; that is deferred to fit/transform.
        self.upper_quantile = upper_quantile
        self.hard_caps = hard_caps

    @property
    def _caps(self) -> dict[str, float]:
        """Resolved hard-cap mapping (None treated as empty)."""
        return self.hard_caps or {}

    def fit(self, X, y=None):
        X = _as_frame(X)
        # Learn the upper bound for every column from TRAIN data only.
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
        """Winsorizing does not change column identity (passthrough)."""
        return _resolve_names(self, input_features)


def _as_frame(X, names: list[str] | None = None) -> pd.DataFrame:
    """Coerce array-like input to a DataFrame, preserving column names.

    ColumnTransformer may hand transformers a numpy array; we restore a frame
    so quantile/replace operations stay column-aware. If ``names`` (captured
    at fit time) is supplied and matches the array width, those names are
    reattached so downstream pipeline steps see consistent column identities.
    """
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
    """Number of columns in array-like input."""
    arr = np.asarray(X)
    return 1 if arr.ndim == 1 else arr.shape[1]


def _column_names(X) -> list[str] | None:
    """Best-effort capture of input column names at fit time."""
    if isinstance(X, pd.DataFrame):
        return list(X.columns)
    return None


def _resolve_names(estimator, input_features):
    """Resolve output feature names for a passthrough transformer.

    Prefers names passed by the caller, then names captured at fit time,
    falling back to positional names if neither is available.
    """
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
    """Build the hard-cap mapping for delinquency columns."""
    return {col: float(DELINQUENCY_CAP) for col in columns}
