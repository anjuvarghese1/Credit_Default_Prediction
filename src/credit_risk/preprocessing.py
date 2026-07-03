# pre-processing

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import SCHEMA
from .transformers import OutlierCapper, SentinelReplacer, delinquency_hard_caps

# numeric features
def build_numeric_pipeline() -> Pipeline:    
    return Pipeline(
        steps=[
            # Median is robust to the skew typical of income / debt ratios.
            ("impute", SimpleImputer(strategy="median")),
            ("winsorize", OutlierCapper(upper_quantile=0.99)),
            ("scale", StandardScaler()),
        ]
    )

# delinquency voids
def build_delinquency_pipeline() -> Pipeline:    
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
    return list(SCHEMA.numeric_features) + list(SCHEMA.delinquency_counts)


def build_preprocessor() -> ColumnTransformer:    
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
