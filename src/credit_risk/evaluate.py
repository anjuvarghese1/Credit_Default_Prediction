"""Evaluation metrics tailored to credit-default modeling.

Accuracy is nearly useless on a 7%-default problem (predicting "no default"
for everyone scores ~93%). This module instead reports the metrics a credit
model-risk audience expects:

* **AUC-ROC** -- ranking quality, threshold-independent.
* **Average Precision (PR-AUC)** -- precision/recall trade-off, the right lens
  under heavy class imbalance.
* **KS statistic** -- the maximum separation between the cumulative score
  distributions of defaulters and non-defaulters; a staple of scorecard
  validation.
* **Brier score** -- calibration of the predicted probabilities (lower is
  better), important when scores feed downstream expected-loss calculations.
* **Confusion matrix & classification report** at a chosen decision threshold,
  for an operational view of the trade-offs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


@dataclass
class EvaluationResult:
    """Container for a single model's evaluation metrics."""

    model_name: str
    roc_auc: float
    average_precision: float
    ks_statistic: float
    brier_score: float
    threshold: float
    confusion: np.ndarray
    report: str = field(repr=False)

    def summary_row(self) -> dict[str, float | str]:
        """Flat dict for tabular comparison across models."""
        return {
            "Model": self.model_name,
            "ROC-AUC": round(self.roc_auc, 4),
            "PR-AUC": round(self.average_precision, 4),
            "KS": round(self.ks_statistic, 4),
            "Brier": round(self.brier_score, 4),
        }


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov separation between class score distributions.

    Computed as the maximum gap between the true-positive rate and the
    false-positive rate across all thresholds -- equivalently, max(TPR - FPR)
    along the ROC curve. Ranges from 0 (no separation) to 1 (perfect).
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> EvaluationResult:
    """Compute the full metric suite for one model's scored predictions.

    Parameters
    ----------
    model_name:
        Human-readable label.
    y_true:
        Ground-truth binary labels.
    y_score:
        Predicted probability of default (positive class).
    threshold:
        Probability cutoff for the confusion matrix / classification report.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    return EvaluationResult(
        model_name=model_name,
        roc_auc=roc_auc_score(y_true, y_score),
        average_precision=average_precision_score(y_true, y_score),
        ks_statistic=ks_statistic(y_true, y_score),
        brier_score=brier_score_loss(y_true, y_score),
        threshold=threshold,
        confusion=confusion_matrix(y_true, y_pred),
        report=classification_report(y_true, y_pred, digits=3, zero_division=0),
    )
