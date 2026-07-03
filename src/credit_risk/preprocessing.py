"""Assembly of the leakage-safe preprocessing pipeline.

The public entry point is :func:`build_preprocessor`, which returns an
unfitted :class:`~sklearn.compose.ColumnTransformer`. Because every step is a
fitted estimator, calling ``fit`` on the *training* split learns all
imputation values, outlier bounds, and scaling statistics from training data
alone -- the test split never influences preprocessing.

Pipeline per column group
-------------------------
Numeric features:
    impute (median) -> winsorize (1st/99th pct) -> standard-scale
Delinquency counts:
    replace sentinels (96/98 -> NaN) -> impute (median)
    -> cap (domain ceiling) -> standard-scale
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import SCHEMA
from .transformers import OutlierCapper, SentinelReplacer, delinquency_hard_caps


def build_numeric_pipeline() -> Pipeline:
    """Impute -> winsorize -> scale for continuous financial features."""
    return Pipeline(
        steps=[
            # Median is robust to the skew typical of income / debt ratios.
            ("impute", SimpleImputer(strategy="median")),
            ("winsorize", OutlierCapper(upper_quantile=0.99)),
            ("scale", StandardScaler()),
        ]
    )


def build_delinquency_pipeline() -> Pipeline:
    """Sentinel-clean -> impute -> domain-cap -> scale for count features."""
    caps = delinquency_hard_caps(list(SCHEMA.delinquency_counts))
    return Pipeline(
        steps=[
            ("sentinels", SentinelReplacer()),
            ("impute", SimpleImputer(strategy="median")),
            # Quantile is permissive here; the hard cap does the real work.
            ("cap", OutlierCapper(upper_quantile=1.0, hard_caps=caps)),
            ("scale", StandardScaler()),
        ]
    )


def get_output_feature_names() -> list[str]:
    """Output feature names after preprocessing, in column order.

    The preprocessing steps (impute, winsorize, cap, scale) all preserve
    column identity, so the output names are simply the numeric features
    followed by the delinquency-count features. Returning these from the
    known schema is more robust than relying on sklearn's name introspection
    through nested pipelines, and it keeps downstream reporting readable.
    """
    return list(SCHEMA.numeric_features) + list(SCHEMA.delinquency_counts)


def build_preprocessor() -> ColumnTransformer:
    """Return the full, unfitted preprocessing ColumnTransformer.

    The returned object is meant to be the first step of a model Pipeline, or
    fitted directly on training data. It maps each column group to its
    dedicated cleaning pipeline and concatenates the results.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                build_numeric_pipeline(),
                list(SCHEMA.numeric_features),
            ),
            (
                "delinquency",
                build_delinquency_pipeline(),
                list(SCHEMA.delinquency_counts),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor
