from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

from .config import MODELS, ModelConfig
from .preprocessing import build_preprocessor


class OversampledClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self, estimator, sampling_ratio: float = 1.0, random_state: int = 42
    ) -> None:
        self.estimator = estimator
        self.sampling_ratio = sampling_ratio
        self.random_state = random_state
    
    _estimator_type = "classifier"

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        majority = counts.max()        
        target = max(int(round(majority * self.sampling_ratio)), int(counts.min()))

        idx_parts: list[np.ndarray] = []
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            n_cls = len(cls_idx)
            if n_cls < target and n_cls < majority:
                cls_idx = resample(
                    cls_idx, replace=True, n_samples=min(target, majority),
                    random_state=self.random_state,
                )
            idx_parts.append(cls_idx)

        idx = np.concatenate(idx_parts)
        rng = np.random.default_rng(self.random_state)
        rng.shuffle(idx)

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[idx], y[idx])
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(np.asarray(X))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(np.asarray(X))


def build_logistic_regression(cfg: ModelConfig = MODELS) -> Pipeline:
    """Interpretable baseline: L2-regularized logistic regression."""
    estimator = LogisticRegression(
        C=cfg.logreg_C,
        max_iter=cfg.logreg_max_iter,
        class_weight="balanced",               # offset class imbalance
        random_state=cfg.random_state,
    )
    return _wrap(estimator)


def build_gradient_boosting(cfg: ModelConfig = MODELS) -> Pipeline:    
    estimator = HistGradientBoostingClassifier(
        max_iter=cfg.hgb_max_iter,
        learning_rate=cfg.hgb_learning_rate,
        max_depth=cfg.hgb_max_depth,
        l2_regularization=cfg.hgb_l2_regularization,
        early_stopping=cfg.hgb_early_stopping,
        random_state=cfg.random_state,
    )
    # swap-in: from xgboost import XGBClassifier; estimator = XGBClassifier(...)
    return _wrap(estimator)


def build_mlp(cfg: ModelConfig = MODELS) -> Pipeline:    
    estimator = MLPClassifier(
        hidden_layer_sizes=cfg.mlp_hidden_layer_sizes,
        alpha=cfg.mlp_alpha,
        max_iter=cfg.mlp_max_iter,
        learning_rate="adaptive",
        learning_rate_init=cfg.mlp_learning_rate_init,
        early_stopping=cfg.mlp_early_stopping,
        n_iter_no_change=cfg.mlp_n_iter_no_change,
        random_state=cfg.random_state,
    )
    # swap-in: a PyTorch nn.Module wrapped via skorch.NeuralNetClassifier
    wrapped = OversampledClassifier(
        estimator, sampling_ratio=cfg.mlp_sampling_ratio,
        random_state=cfg.random_state,
    )
    return _wrap(wrapped)


def _wrap(estimator) -> Pipeline:    
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


def build_all_models(cfg: ModelConfig = MODELS) -> dict[str, Pipeline]:    
    return {
        "Logistic Regression": build_logistic_regression(cfg),
        "Gradient Boosting": build_gradient_boosting(cfg),
        "Neural Network (MLP)": build_mlp(cfg),
    }
