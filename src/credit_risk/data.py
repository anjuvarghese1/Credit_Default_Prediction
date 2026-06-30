"""Data ingestion for the credit-risk pipeline.

Two entry points:

* :func:`load_raw_data` -- read the real Kaggle ``cs-training.csv`` if it is
  present in ``data/``; otherwise fall back to a generated, schema-matched
  sample so the pipeline is runnable out of the box.
* :func:`generate_synthetic_sample` -- build a small dataset that mirrors the
  real file's *structure and defects* (class imbalance, missing values,
  delinquency sentinel codes, heavy-tailed outliers). This is a stand-in for
  development and CI, **not** a substitute for the real data in any reported
  result.

Swap-in note
------------
To use the real competition data, place ``cs-training.csv`` in ``data/`` and
re-run. No code changes are required -- the loader detects and prefers it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    DELINQUENCY_SENTINELS,
    RAW_DATA_FILE,
    SCHEMA,
)

logger = logging.getLogger(__name__)


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """Load the raw credit dataset.

    Parameters
    ----------
    path:
        Explicit path to a CSV. If ``None``, uses ``config.RAW_DATA_FILE``.
        When that file is absent, a schema-matched synthetic sample is
        generated and a clear warning is logged.

    Returns
    -------
    DataFrame with the target column and all feature columns. The raw id
    column ("Unnamed: 0"), if present, is dropped.
    """
    path = path or RAW_DATA_FILE

    if path.exists():
        logger.info("Loading real dataset from %s", path)
        df = pd.read_csv(path)
    else:
        logger.warning(
            "Real dataset not found at %s -- generating a schema-matched "
            "synthetic sample. Place 'cs-training.csv' in data/ to use the "
            "real Kaggle data.",
            path,
        )
        df = generate_synthetic_sample()

    # Drop the Kaggle id column if present (it is not a feature).
    if SCHEMA.id_col in df.columns:
        df = df.drop(columns=[SCHEMA.id_col])

    _validate_schema(df)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Fail fast if expected columns are missing."""
    missing = set(SCHEMA.all_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"Loaded data is missing expected columns: {sorted(missing)}. "
            f"Check that the file matches the 'Give Me Some Credit' schema."
        )


def generate_synthetic_sample(
    n_rows: int = 8_000,
    default_rate: float = 0.07,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a schema-matched synthetic credit dataset.

    The goal is *structural realism* on two axes:

    1. **Same columns, dtypes, and defects** as the real file (missing income
       and dependents, delinquency sentinel codes, heavy right tails).
    2. **Realistic learnable structure** -- default risk depends on the
       features through threshold effects and interactions (e.g., high
       utilization is far more dangerous when income is low), not a single
       linear combination. This is what gives tree ensembles a genuine edge
       over a linear model, reproducing the real-data finding rather than an
       artifact of weak/linear synthetic signal.

    Even so, these numbers are illustrative scaffolding only. Drop the real
    ``cs-training.csv`` in ``data/`` for authentic results.
    """
    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------ #
    # 1. Draw realistic marginal feature distributions (independent of y).
    # ------------------------------------------------------------------ #
    age = np.clip(rng.normal(52, 14, n_rows), 21, 99).round().astype(int)

    util = np.abs(rng.normal(0.3, 0.28, n_rows))
    util_tail = rng.random(n_rows) < 0.01
    util[util_tail] *= rng.uniform(50, 5000, util_tail.sum())  # extreme outliers

    monthly_income = np.clip(rng.lognormal(8.6, 0.5, n_rows), 0, None).round()

    debt_ratio = np.abs(rng.normal(0.35, 0.35, n_rows))
    dr_tail = rng.random(n_rows) < 0.02
    debt_ratio[dr_tail] *= rng.uniform(100, 5000, dr_tail.sum())

    open_lines = np.clip(rng.normal(8, 5, n_rows), 0, None).round().astype(int)
    real_estate = np.clip(rng.normal(1, 1.1, n_rows), 0, None).round().astype(int)
    dependents = np.clip(rng.normal(0.8, 1.1, n_rows), 0, None).round()

    dpd_30_59 = rng.poisson(0.25, n_rows).astype(float)
    dpd_90 = rng.poisson(0.12, n_rows).astype(float)
    dpd_60_89 = rng.poisson(0.10, n_rows).astype(float)

    # ------------------------------------------------------------------ #
    # 2. Build a non-linear risk score (log-odds) from the features.
    #    Threshold effects + interactions => trees can exceed a linear model.
    # ------------------------------------------------------------------ #
    # Normalize a few drivers to keep coefficients interpretable.
    income_k = monthly_income / 1000.0
    util_c = np.clip(util, 0, 2.0)              # ignore absurd outliers in signal
    dr_c = np.clip(debt_ratio, 0, 2.0)

    logit = np.full(n_rows, -3.1)               # base rate anchor

    # --- Smooth, broadly-learnable component (helps all models, incl. MLP) ---
    logit += 1.8 * (util_c - 0.3)               # linear-ish utilization effect
    logit += 0.9 * np.tanh(any_dpd_smooth(dpd_30_59, dpd_60_89, dpd_90))
    logit += 0.45 * np.maximum(2.5 - income_k, 0.0)   # low-income gradient
    logit += 0.35 * (dr_c - 0.35)

    # --- Tree-favorable component (thresholds + interactions) ----------------
    # Threshold (non-linear) effect: utilization bites harder past ~0.6.
    logit += 1.5 * np.maximum(util_c - 0.6, 0.0)
    # Interaction: high utilization is much worse when income is low.
    logit += 0.8 * np.maximum(util_c - 0.5, 0.0) * np.maximum(4.0 - income_k, 0.0)
    # Any 90-day delinquency is a strong discrete signal.
    logit += 1.0 * (dpd_90 > 0).astype(float)
    # Young + high debt ratio interaction.
    logit += 0.6 * (age < 35).astype(float) * np.maximum(dr_c - 0.4, 0.0)
    # Mild protective effect of more open credit lines (saturating).
    logit -= 0.15 * np.tanh(open_lines / 5.0)
    # Small noise so the problem is not perfectly separable.
    logit += rng.normal(0, 0.35, n_rows)

    prob = 1.0 / (1.0 + np.exp(-logit))

    # ------------------------------------------------------------------ #
    # 3. Sample labels, then calibrate the base rate to `default_rate` by
    #    shifting the intercept until the realized prevalence matches.
    # ------------------------------------------------------------------ #
    shift = _solve_intercept(logit, target_rate=default_rate)
    prob = 1.0 / (1.0 + np.exp(-(logit + shift)))
    y = (rng.random(n_rows) < prob).astype(int)

    df = pd.DataFrame(
        {
            SCHEMA.target: y,
            "RevolvingUtilizationOfUnsecuredLines": util,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": dpd_30_59,
            "DebtRatio": debt_ratio,
            "MonthlyIncome": monthly_income,
            "NumberOfOpenCreditLinesAndLoans": open_lines,
            "NumberOfTimes90DaysLate": dpd_90,
            "NumberRealEstateLoansOrLines": real_estate,
            "NumberOfTime60-89DaysPastDueNotWorse": dpd_60_89,
            "NumberOfDependents": dependents,
        }
    )

    df = _inject_defects(df, rng)
    return df[SCHEMA.all_columns]


def any_dpd_smooth(*cols: np.ndarray) -> np.ndarray:
    """Sum of delinquency counts -- a smooth aggregate risk driver."""
    total = np.zeros_like(cols[0], dtype=float)
    for c in cols:
        total = total + np.asarray(c, dtype=float)
    return total


def _solve_intercept(
    logit: np.ndarray, target_rate: float, iters: int = 60
) -> float:
    """Find an intercept shift so mean sigmoid(logit + shift) == target_rate.

    Simple bisection on a monotone function; keeps the synthetic prevalence
    locked to the desired default rate regardless of the signal terms above.
    """
    lo, hi = -10.0, 10.0
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        rate = float(np.mean(1.0 / (1.0 + np.exp(-(logit + mid)))))
        if rate > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def _inject_defects(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add the real dataset's characteristic data-quality problems."""
    n = len(df)

    # 1) Missing values in MonthlyIncome (~20% missing in the real data) and
    #    NumberOfDependents (~2.6% missing).
    income_missing = rng.random(n) < 0.20
    df.loc[income_missing, "MonthlyIncome"] = np.nan
    dep_missing = rng.random(n) < 0.026
    df.loc[dep_missing, "NumberOfDependents"] = np.nan

    # 2) Sentinel codes (96 / 98) in the delinquency-count columns.
    for col in SCHEMA.delinquency_counts:
        sentinel_mask = rng.random(n) < 0.002
        sentinel_vals = rng.choice(DELINQUENCY_SENTINELS, sentinel_mask.sum())
        df.loc[sentinel_mask, col] = sentinel_vals

    return df


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    sample = generate_synthetic_sample()
    print(sample.head())
    print("\nShape:", sample.shape)
    print("Default rate:", round(sample[SCHEMA.target].mean(), 4))
    print("\nMissing values per column:")
    print(sample.isna().sum())
