"""
Configuration page - maintains the dataset schema, file paths and key hyperparameters 
To use a different dataset, edit the SCHEMA block below and rerun the notebook
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

# paths
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT/"data"
MODELS_DIR: Path = PROJECT_ROOT/"models"
REPORTS_DIR: Path = PROJECT_ROOT/"reports"
FIGURES_DIR: Path = REPORTS_DIR/"figures"

# default input file
DEFAULT_INPUT_FILE: Path = DATA_DIR/"cs-training.csv"

# dataclass schema : Give_Me_Some_Credit (Kaggle)
@dataclass(frozen=True)
class Schema:
  target: str = 'SeriousDlqin2yrs'          # target
  id_col: str = 'Unnamed: 0'                # unnamed index column, drop it
  numerical_features: tuple[str, ...] = (   # continuous figures, scale it
      'RevolvingUtilizationOfUnsecuredLines',
      'age',
      'DebtRatio',
      'MonthlyIncome',
      'NumberOfOpenCreditLinesAndLoans',
      'NumberRealEstateLoansOrLines',
      'NumberOfDependents',
  )
  delinquency_days: tuple[str, ...] = (     # contains 96/98 - no history/closed ccount, special handling
      'NumberOfTime30-59DaysPastDueNotWorse',
      'NumberOfTimes90DaysLate',
      'NumberOfTime60-89DaysPastDueNotWorse'
  )
  missing_value_cols: tuple[str, ...] = (    # columns missing data values
      'MonthlyIncome',
      'NumberOfDependents'
  )

  @property
  def feature_columns(self) -> list[str]:
    return list(self.numerical_features) + list(self.delinquency_days)

  @property
  def all_columns(self) -> list[str]:
    return[self.target] + self.feature_columns

SCHEMA = Schema()

delinquency_voids: tuple[int, ...] = (96,98)
delinquency_bar: int = 20

# training | evaluation
@dataclass(frozen=True)
class TrainSplit:
  test_size: float = 0.2
  random_state: int = 42
  stratify: bool = True

# hyperparameters
@dataclass(frozen=True)
class HParams:
  random_state: int = 42

  # logistic regression
  lr_max_iter: int = 1000
  lr_C: float = 1.0

  # histogram gradient boosting
  hgb_max_iter: int = 300
  hgb_learning_rate: float = 0.05
  hgb_max_depth: int | None = None
  hgb_l2_regularization: float = 1.0
  hgb_early_stopping: bool = True

  # mlp
  mlp_hidden_layer_sizes: tuple[int, ...] = (32, 16)
  mlp_max_iter: int = 150
  mlp_alpha: float = 1e-2
  mlp_learning_rate_init: float = 1e-3
  mlp_sampling_ratio: float = 0.3
  mlp_early_stopping: bool = True
  mlp_n_iter_no_change: int = 8
  mlp_cv_folds: int = 3

SPLIT = TrainSplit()
HPARAMS = HParams()

# create output directories if they do not exist
def ensure_dirs() -> None:
    for path in (DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)
