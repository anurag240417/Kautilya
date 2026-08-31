"""Temporal evaluation for ML classifiers.

Provides per-time-step performance analysis to detect concept drift
and identify time periods with degraded detection.

See CONTEXT.md — Experimental Progression and AGENTS.md §7.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from backend.ml.classifier import TransactionClassifier

logger = logging.getLogger(__name__)


@dataclass
class StepMetrics:
    """Classification metrics for a single time step."""

    time_step: int
    n_samples: int
    n_illicit: int
    n_licit: int
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    roc_auc: float | None = None


@dataclass
class TemporalEvaluation:
    """Complete temporal evaluation result.

    Contains both aggregate and per-time-step metrics, plus class
    distribution information for investigating model behavior over time.
    """

    model_version: str
    feature_set_name: str
    overall_metrics: dict = field(default_factory=dict)
    per_step_metrics: dict[int, StepMetrics] = field(default_factory=dict)
    time_steps_evaluated: list[int] = field(default_factory=list)


def evaluate_temporal(
    classifier: TransactionClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    time_steps: np.ndarray,
) -> TemporalEvaluation:
    """Evaluate a classifier with per-time-step granularity.

    Computes aggregate metrics across all test data, then breaks
    down performance by each individual time step. This reveals
    concept drift or degraded detection in newer time periods.

    Args:
        classifier: A fitted ``TransactionClassifier``.
        X_test: Test feature matrix of shape (n_samples, n_features).
        y_test: Test target vector of shape (n_samples,).
        time_steps: Time step array of shape (n_samples,), one per row.

    Returns:
        A ``TemporalEvaluation`` with overall and per-step metrics.

    Raises:
        RuntimeError: If the classifier has not been trained.
        ValueError: If inputs are empty or shapes mismatch.
    """
    if len(X_test) == 0:
        msg = "Test data is empty"
        raise ValueError(msg)
    if len(X_test) != len(y_test):
        msg = f"Shape mismatch: X_test has {len(X_test)} rows, y_test has {len(y_test)}"
        raise ValueError(msg)
    if len(X_test) != len(time_steps):
        msg = f"Shape mismatch: X_test has {len(X_test)} rows, time_steps has {len(time_steps)}"
        raise ValueError(msg)

    # Overall predictions
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[:, 1]

    # Aggregate metrics
    overall = _compute_step_metrics_dict(y_test, y_pred, y_prob)

    result = TemporalEvaluation(
        model_version=classifier.model_version,
        feature_set_name=classifier.feature_set_name,
        overall_metrics=overall,
    )

    # Per-step metrics
    unique_steps = sorted(set(time_steps))
    result.time_steps_evaluated = list(unique_steps)

    for step in unique_steps:
        mask = time_steps == step
        step_y_true = y_test[mask]
        step_y_pred = y_pred[mask]
        step_y_prob = y_prob[mask]

        n_illicit = int((step_y_true == 1).sum())
        n_licit = int((step_y_true == 0).sum())
        n_samples = len(step_y_true)

        step_result = StepMetrics(
            time_step=int(step),
            n_samples=n_samples,
            n_illicit=n_illicit,
            n_licit=n_licit,
        )

        # Only compute metrics if we have at least one sample
        if n_samples > 0:
            step_result.accuracy = float(accuracy_score(step_y_true, step_y_pred))
            step_result.precision = float(
                precision_score(step_y_true, step_y_pred, zero_division=0)
            )
            step_result.recall = float(
                recall_score(step_y_true, step_y_pred, zero_division=0)
            )
            step_result.f1 = float(
                f1_score(step_y_true, step_y_pred, zero_division=0)
            )

            # ROC-AUC requires both classes present
            if n_illicit > 0 and n_licit > 0:
                try:
                    step_result.roc_auc = float(
                        roc_auc_score(step_y_true, step_y_prob)
                    )
                except ValueError:
                    step_result.roc_auc = None

        result.per_step_metrics[int(step)] = step_result

    logger.info(
        "Temporal evaluation complete: %d steps, overall F1=%.4f, overall AUC=%s",
        len(unique_steps),
        overall.get("f1", 0.0),
        f"{overall.get('roc_auc', 'N/A'):.4f}" if overall.get("roc_auc") is not None else "N/A",
    )

    return result


def summarize_evaluation(evaluation: TemporalEvaluation) -> dict:
    """Produce a JSON-serializable summary of a temporal evaluation.

    Useful for inclusion in model metadata or reporting.

    Args:
        evaluation: A completed ``TemporalEvaluation``.

    Returns:
        Dictionary with overall metrics, per-step breakdown, and
        model provenance.
    """
    per_step = {}
    for step, metrics in evaluation.per_step_metrics.items():
        per_step[int(step)] = {
            "n_samples": metrics.n_samples,
            "n_illicit": metrics.n_illicit,
            "n_licit": metrics.n_licit,
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "roc_auc": metrics.roc_auc,
        }

    return {
        "model_version": evaluation.model_version,
        "feature_set_name": evaluation.feature_set_name,
        "overall_metrics": evaluation.overall_metrics,
        "per_step_metrics": per_step,
        "time_steps_evaluated": [int(s) for s in evaluation.time_steps_evaluated],
        "n_steps": len(evaluation.time_steps_evaluated),
    }


def _compute_step_metrics_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Compute aggregate classification metrics as a dict."""
    metrics: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = None

    return metrics
