"""Plotting utilities for model comparison and diagnostics.

All figures are written to ``reports/figures`` as PNGs and are referenced from
the generated report. Matplotlib is used with a non-interactive backend so the
pipeline runs headless (CI, containers) without a display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from .config import FIGURES_DIR


def plot_roc_curves(
    scored: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path | None = None,
) -> Path:
    """Overlay ROC curves for every model.

    Parameters
    ----------
    scored:
        Mapping of model name -> (y_true, y_score).
    """
    out_path = out_path or (FIGURES_DIR / "roc_curves.png")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, (y_true, y_score) in scored.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = np.trapezoid(tpr, fpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves -- Default Prediction")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_pr_curves(
    scored: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path | None = None,
) -> Path:
    """Overlay precision-recall curves (the right view under imbalance)."""
    out_path = out_path or (FIGURES_DIR / "pr_curves.png")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, (y_true, y_score) in scored.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ax.plot(recall, precision, lw=2, label=name)
    # Baseline precision = prevalence of the positive class.
    any_true = next(iter(scored.values()))[0]
    prevalence = float(np.mean(any_true))
    ax.axhline(prevalence, ls="--", color="k", alpha=0.6,
               label=f"Baseline ({prevalence:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves -- Default Prediction")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_calibration(
    scored: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: Path | None = None,
    n_bins: int = 10,
) -> Path:
    """Reliability diagram: predicted vs. observed default frequency."""
    out_path = out_path or (FIGURES_DIR / "calibration.png")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, (y_true, y_score) in scored.items():
        frac_pos, mean_pred = calibration_curve(
            y_true, y_score, n_bins=n_bins, strategy="quantile"
        )
        ax.plot(mean_pred, frac_pos, "o-", lw=2, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed default frequency")
    ax.set_title("Calibration -- Default Prediction")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_permutation_importance(
    feature_names: list[str],
    importances: np.ndarray,
    out_path: Path | None = None,
    title: str = "Permutation Importance",
) -> Path:
    """Horizontal bar chart of feature importances (sorted)."""
    out_path = out_path or (FIGURES_DIR / "feature_importance.png")
    order = np.argsort(importances)
    names_sorted = [feature_names[i] for i in order]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh(range(len(order)), importances[order], color="#3b6ea5")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names_sorted, fontsize=8)
    ax.set_xlabel("Mean importance (AUC drop when shuffled)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
