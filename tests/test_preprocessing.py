"""Tests for the preprocessing layer and custom transformers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from credit_risk.config import SCHEMA
from credit_risk.data import generate_synthetic_sample
from credit_risk.preprocessing import build_preprocessor, get_output_feature_names
from credit_risk.transformers import OutlierCapper, SentinelReplacer


def _xy(n=2_000, seed=0):
    df = generate_synthetic_sample(n_rows=n, random_state=seed)
    return df[SCHEMA.feature_columns], df[SCHEMA.target].to_numpy()


def test_preprocessor_removes_all_nans():
    X, y = _xy()
    pre = build_preprocessor()
    Xt = pre.fit_transform(X, y)
    assert not np.isnan(Xt).any()


def test_preprocessor_scales_to_unit_variance():
    X, y = _xy()
    pre = build_preprocessor()
    Xt = pre.fit_transform(X, y)
    # After StandardScaler, columns should be ~0 mean, ~1 std.
    assert np.allclose(Xt.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(Xt.std(axis=0), 1, atol=1e-6)


def test_preprocessor_caps_outliers():
    X, y = _xy()
    pre = build_preprocessor()
    Xt = pre.fit_transform(X, y)
    # Winsorization should remove extreme scaled magnitudes.
    assert np.abs(Xt).max() < 20


def test_preprocessor_is_sklearn_clonable():
    pre = build_preprocessor()
    clone(pre)  # must not raise (get_params returns ctor args verbatim)


def test_output_feature_names_match_width():
    X, y = _xy()
    pre = build_preprocessor()
    Xt = pre.fit_transform(X, y)
    assert Xt.shape[1] == len(get_output_feature_names())


def test_no_leakage_train_bounds_applied_to_test():
    # The capper must learn bounds on TRAIN and apply them to TEST unchanged,
    # not re-fit on test. We verify by checking that a transformer fitted on a
    # low-valued train set caps a high-valued test row to the train bound.
    train = pd.DataFrame({"a": np.arange(0, 100, dtype=float)})
    test = pd.DataFrame({"a": [10_000.0]})
    capper = OutlierCapper(upper_quantile=0.99)
    capper.fit(train)
    out = capper.transform(test)
    train_upper = float(train["a"].quantile(0.99))
    assert out["a"].iloc[0] <= train_upper + 1e-9


def test_sentinel_replacer_converts_codes_to_nan():
    df = pd.DataFrame({"x": [0, 1, 96, 98, 2]})
    out = SentinelReplacer().fit_transform(df)
    assert out["x"].isna().sum() == 2
    assert out["x"].tolist()[:2] == [0, 1]
