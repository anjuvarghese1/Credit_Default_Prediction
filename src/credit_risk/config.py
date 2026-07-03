"""Central configuration for the credit-risk pipeline.

Keeping the dataset schema, file paths, and key hyperparameters in one
place means the rest of the codebase never hard-codes column names or
magic numbers. To point the pipeline at a different dataset, edit the
``SCHEMA`` block below and nothing else needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Resolve project root as two levels above this file (src/credit_risk/config.py)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# Default input file. Drop the real Kaggle ``cs-training.csv`` here and the
# pipeline will use it automatically; otherwise a schema-matched sample is
# generated for a self-contained demo run.
RAW_DATA_FILE: Path = DATA_DIR / "cs-training.csv"


# --------------------------------------------------------------------------- #
# Dataset schema  --  "Give Me Some Credit" (Kaggle, 2011)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Schema:
    """Column-level description of the credit dataset.

    Centralizing this lets the preprocessing layer treat columns by *role*
    (target / numeric / count-with-sentinels) rather than by hard-coded name.
    """

    target: str = "SeriousDlqin2yrs"

    # An identifier column present in the raw Kaggle file ("Unnamed: 0").
    # Dropped before modeling; kept here so ingestion can recognize it.
    id_col: str = "Unnamed: 0"

    # Continuous financial ratios / amounts -> scale these.
    numeric_features: tuple[str, ...] = (
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberRealEstateLoansOrLines",
        "NumberOfDependents",
    )

    # Delinquency counts. These carry the infamous 96 / 98 sentinel codes
    # in the raw data and are treated with dedicated cleaning logic.
    delinquency_counts: tuple[str, ...] = (
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfTime60-89DaysPastDueNotWorse",
    )

    # Columns known to contain missing values in the raw data.
    missing_value_cols: tuple[str, ...] = (
        "MonthlyIncome",
        "NumberOfDependents",
    )

    @property
    def feature_columns(self) -> list[str]:
        """All model input columns, in a stable order."""
        return list(self.numeric_features) + list(self.delinquency_counts)

    @property
    def all_columns(self) -> list[str]:
        """Target + all features (excludes the id column)."""
        return [self.target] + self.feature_columns


SCHEMA = Schema()

# Sentinel codes that encode "not applicable / unknown" in the delinquency
# count columns of the raw Kaggle data. They are not real counts and must be
# handled before modeling (see preprocessing.SentinelCapper).
DELINQUENCY_SENTINELS: tuple[int, ...] = (96, 98)

# Domain-plausible hard ceiling for delinquency counts after sentinel removal.
DELINQUENCY_CAP: int = 20


# --------------------------------------------------------------------------- #
# Train / evaluation parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.2
    random_state: int = 42
    # Stratify on the target to preserve the ~7% default rate in both splits.
    stratify: bool = True


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the benchmarked models.

    Values are deliberately modest so the whole benchmark runs quickly and
    deterministically. They are sensible defaults, not tuned optima --
    hyperparameter search is intentionally out of scope for this pipeline.
    """

    random_state: int = 42

    # Logistic regression (interpretable baseline).
    logreg_max_iter: int = 1000
    logreg_C: float = 1.0

    # Histogram gradient boosting (sklearn-native; XGBoost/LightGBM drop-in).
    hgb_max_iter: int = 300
    hgb_learning_rate: float = 0.05
    hgb_max_depth: int | None = None
    hgb_l2_regularization: float = 1.0
    hgb_early_stopping: bool = True

    # Multilayer perceptron (the "complex" neural model). Wrapped with
    # minority oversampling at train time. On the full dataset (150k rows)
    # early stopping is safe and avoids grinding all iterations; a smaller
    # network and lighter oversampling keep this — the slowest model — fast.
    mlp_hidden_layer_sizes: tuple[int, ...] = (32, 16)
    mlp_alpha: float = 1e-2
    mlp_max_iter: int = 150
    mlp_learning_rate_init: float = 1e-3
    mlp_sampling_ratio: float = 0.3
    mlp_early_stopping: bool = True
    mlp_n_iter_no_change: int = 8
    # The MLP cross-validates with fewer folds than the cheap models, since
    # it dominates runtime; its held-out test score is unaffected.
    mlp_cv_folds: int = 3


SPLIT = SplitConfig()
MODELS = ModelConfig()


def ensure_dirs() -> None:
    """Create output directories if they do not yet exist."""
    for path in (DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)
