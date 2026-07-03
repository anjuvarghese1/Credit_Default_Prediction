# tests for the data ingestion and synthetic-generation layer

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.config import DELINQUENCY_VOIDS, SCHEMA
from credit_risk.data import generate_synthetic_sample, load_raw_data

# ensure no surprise columns
def test_synthetic_sample_has_expected_schema():
    df = generate_synthetic_sample(n_rows=1_000, random_state=0)    
    assert list(df.columns) == SCHEMA.all_columns
    assert df.shape[0] == 1_000


def test_synthetic_default_rate_is_calibrated():
    df = generate_synthetic_sample(n_rows=10_000, default_rate=0.07, random_state=0)
    rate = df[SCHEMA.target].mean()    
    assert 0.05 < rate < 0.09


def test_synthetic_injects_missing_values():
    df = generate_synthetic_sample(n_rows=5_000, random_state=0)
    for col in SCHEMA.missing_value_cols:
        assert df[col].isna().any(), f"expected missing values in {col}"


def test_synthetic_injects_sentinels():
    df = generate_synthetic_sample(n_rows=20_000, random_state=0)
    found = (
        df[list(SCHEMA.delinquency_days)]
        .isin(list(DELINQUENCY_VOIDS))
        .to_numpy()
        .any()
    )
    assert found, "expected at least one 96/98 sentinel code in delinquency cols"


def test_synthetic_has_outliers():
    df = generate_synthetic_sample(n_rows=10_000, random_state=0)    
    assert df["RevolvingUtilizationOfUnsecuredLines"].max() > 1.0


def test_load_raw_data_falls_back_to_synthetic(tmp_path, monkeypatch):    
    import credit_risk.data as data_mod

    fake = tmp_path / "does_not_exist.csv"
    df = data_mod.load_raw_data(path=fake)
    assert set(SCHEMA.all_columns).issubset(df.columns)
    assert SCHEMA.id_col not in df.columns  # id column dropped if present


def test_load_raw_data_reads_real_csv(tmp_path):    
    import credit_risk.data as data_mod

    df_src = generate_synthetic_sample(n_rows=200, random_state=1)
    df_src.insert(0, SCHEMA.id_col, range(len(df_src)))  # add a Kaggle-style id
    path = tmp_path / "cs-training.csv"
    df_src.to_csv(path, index=False)

    df = data_mod.load_raw_data(path=path)
    assert SCHEMA.id_col not in df.columns
    assert len(df) == 200
