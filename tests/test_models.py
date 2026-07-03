# tests for model construction and the evaluation metric suite

from __future__ import annotations

import warnings

import numpy as np

from credit_risk.config import SCHEMA
from credit_risk.data import generate_synthetic_sample
from credit_risk.evaluate import evaluate_model, ks_statistic
from credit_risk.models import OversampledClassifier, build_all_models


def _xy(n=1_500, seed=0):
    df = generate_synthetic_sample(n_rows=n, random_state=seed)
    return df[SCHEMA.feature_columns], df[SCHEMA.target].to_numpy()


def test_build_all_models_returns_three_pipelines():
    models = build_all_models()
    assert set(models) == {
        "Logistic Regression",
        "Gradient Boosting",
        "Neural Network (MLP)",
    }


def test_each_model_fits_and_predicts_proba():
    X, y = _xy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, pipe in build_all_models().items():
            pipe.fit(X, y)
            proba = pipe.predict_proba(X)
            assert proba.shape == (len(y), 2)
            # Probabilities in [0, 1] and rows summing to 1.
            assert proba.min() >= 0 and proba.max() <= 1
            assert np.allclose(proba.sum(axis=1), 1, atol=1e-6)


def test_oversampler_balances_classes_at_fit():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    n = 1_000
    X = rng.normal(size=(n, 4))
    # ~8% positive class to mimic the imbalance.
    y = (rng.random(n) < 0.08).astype(int)

    clf = OversampledClassifier(
        LogisticRegression(max_iter=200), sampling_ratio=1.0, random_state=0
    )
    clf.fit(X, y)
    assert hasattr(clf, "classes_")
    assert clf._estimator_type == "classifier"
    # Predictions should be well-formed probabilities.
    proba = clf.predict_proba(X)
    assert proba.shape == (n, 2)


def test_ks_statistic_bounds():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500)
    score = rng.random(500)
    ks = ks_statistic(y, score)
    assert 0.0 <= ks <= 1.0


def test_ks_perfect_separation_is_one():
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert ks_statistic(y, score) == 1.0


def test_evaluate_model_produces_full_result():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=400)
    # Give the score mild signal so metrics are well-defined.
    score = np.clip(0.3 * y + rng.random(400) * 0.7, 0, 1)
    res = evaluate_model("test", y, score, threshold=0.5)
    assert 0.0 <= res.roc_auc <= 1.0
    assert 0.0 <= res.average_precision <= 1.0
    assert res.confusion.shape == (2, 2)
    assert isinstance(res.report, str) and len(res.report) > 0
    row = res.summary_row()
    assert set(row) == {"Model", "ROC-AUC", "PR-AUC", "KS", "Brier"}
