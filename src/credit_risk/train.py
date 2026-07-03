# run as a modlue: python -m credit_risk.train

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from .config import (
    MODELS,
    MODELS_DIR,
    REPORTS_DIR,
    SCHEMA,
    SPLIT,
    ensure_dirs,
)

from .data import load_raw_data
from .evaluate import EvaluationResult, evaluate_model
from .models import build_all_models
from .plots import (
    plot_calibration,
    plot_permutation_importance,
    plot_pr_curves,
    plot_roc_curves,
)
from .preprocessing import get_output_feature_names
from .report import write_report

logger = logging.getLogger(__name__)

N_CV_FOLDS = 5
DECISION_THRESHOLD = 0.5

def run(cv_folds: int = N_CV_FOLDS) -> dict[str, EvaluationResult]:
    """Execute the full benchmark and return per-model evaluation results."""
    ensure_dirs()

    # load data
    df = load_raw_data()
    X = df[SCHEMA.feature_columns]
    y = df[SCHEMA.target].to_numpy()
    logger.info("Loaded %d rows; default rate = %.3f", len(df), y.mean())

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=SPLIT.test_size,
        random_state=SPLIT.random_state,
        stratify=y if SPLIT.stratify else None,
    )
    logger.info(
        "Train/test split: %d / %d rows (test default rate = %.3f)",
        len(X_train), len(X_test), y_test.mean(),
    )

    models = build_all_models()

    results: dict[str, EvaluationResult] = {}
    cv_scores: dict[str, tuple[float, float]] = {}
    scored: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # cross validation, refit, score
    mlp_folds = getattr(MODELS, "mlp_cv_folds", cv_folds)
    for name, pipe in models.items():
        n_folds = mlp_folds if "MLP" in name else cv_folds
        model_cv = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=SPLIT.random_state
        )
        logger.info("Cross-validating: %s (%d folds)", name, n_folds)
        scores = cross_val_score(
            pipe, X_train, y_train, cv=model_cv, scoring="roc_auc"
        )
        cv_scores[name] = (float(scores.mean()), float(scores.std()))
        logger.info("  CV ROC-AUC = %.4f +/- %.4f", scores.mean(), scores.std())

        pipe.fit(X_train, y_train)
        y_score = pipe.predict_proba(X_test)[:, 1]
        scored[name] = (y_test, y_score)

        result = evaluate_model(name, y_test, y_score, threshold=DECISION_THRESHOLD)
        results[name] = result
        logger.info(
            "  Test ROC-AUC = %.4f | PR-AUC = %.4f | KS = %.4f",
            result.roc_auc, result.average_precision, result.ks_statistic,
        )

    # plots
    roc_path = plot_roc_curves(scored)
    pr_path = plot_pr_curves(scored)
    cal_path = plot_calibration(scored)

    # best model by ROC-AUC.
    best_name = max(results, key=lambda n: results[n].roc_auc)
    logger.info("Best model by test ROC-AUC: %s", best_name)

    best_pipe = models[best_name]
    feat_names = get_output_feature_names()
    perm = permutation_importance(
        best_pipe, X_test, y_test, scoring="roc_auc",
        n_repeats=10, random_state=SPLIT.random_state,
    )
    imp_path = plot_permutation_importance(
        feat_names, perm.importances_mean,
        title=f"Permutation Importance -- {best_name}",
    )

    # report
    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_pipe, model_path)
    logger.info("Saved best model -> %s", model_path)

    metrics_path = REPORTS_DIR / "metrics.json"
    _dump_metrics(metrics_path, results, cv_scores, best_name)

    write_report(
        results=results,
        cv_scores=cv_scores,
        best_name=best_name,
        figures={
            "roc": roc_path,
            "pr": pr_path,
            "calibration": cal_path,
            "importance": imp_path,
        },
        n_train=len(X_train),
        n_test=len(X_test),
        default_rate=float(y.mean()),
    )
    return results

def _dump_metrics(
    path: Path,
    results: dict[str, EvaluationResult],
    cv_scores: dict[str, tuple[float, float]],
    best_name: str,
) -> None:
    # metrics summary
    payload = {
        "best_model": best_name,
        "models": {
            name: {
                **res.summary_row(),
                "cv_roc_auc_mean": round(cv_scores[name][0], 4),
                "cv_roc_auc_std": round(cv_scores[name][1], 4),
            }
            for name, res in results.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved metrics -> %s", path)

def main() -> None:  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    results = run()
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    table = pd.DataFrame([r.summary_row() for r in results.values()])
    print(table.to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    main()
