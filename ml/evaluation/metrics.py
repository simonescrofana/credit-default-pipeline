"""Compute classification metrics for model evaluation.

Provide a single entry point for computing threshold-independent metrics
(AUC-ROC, AUC-PR), threshold-dependent metrics (precision, recall, F1), and
raw confusion matrix counts, all ready to be logged directly as MLflow
metrics.

"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

RATE_METRICS = {"auc_roc", "auc_pr", "precision", "recall", "f1"}
COUNT_METRICS = {
    "true_positives",
    "false_positives",
    "true_negatives",
    "false_negatives",
}


def compute_metrics(
    y_true: pd.Series, y_pred_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute classification metrics from predicted probabilities.

    Includes threshold-independent metrics (AUC-ROC, AUC-PR), threshold-dependent
    metrics (precision, recall, F1) computed at the given threshold, and the
    raw confusion matrix counts, which allow reconstructing the confusion
    matrix later without rerunning inference. The confusion matrix is always
    computed with explicit labels [0, 1], so a fold with no positive examples
    still yields a well-formed 2x2 matrix instead of raising an error.

    Args:
        y_true (pd.Series): The true binary target labels.
        y_pred_proba (np.ndarray): The predicted probabilities for the
            positive class.
        threshold (float, optional): The decision threshold used to convert
            probabilities into binary predictions for the threshold-dependent
            metrics. Defaults to 0.5.

    Returns:
        dict[str, float]: A flat dictionary of metric names to values,
            ready to be logged directly as MLflow metrics.

    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "auc_roc": roc_auc_score(y_true, y_pred_proba),
        "auc_pr": average_precision_score(y_true, y_pred_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def aggregate_fold_metrics(
    fold_metrics: list[dict[str, float]], fold_sizes: list[int]
) -> dict[str, float]:
    """Aggregate per-fold metrics into a single cross-validation summary.

    Rate-like metrics (AUC-ROC, AUC-PR, precision, recall, F1) are aggregated
    with a size-weighted mean, giving more influence to folds evaluated on
    more data than an unweighted mean would (relevant since TimeSeriesSplit
    produces folds of unequal size). Confusion matrix counts (true/false
    positives/negatives) are aggregated by summation instead, since each
    validation row is scored exactly once across the whole cross-validation:
    summing preserves their meaning as a total count, while averaging them
    would produce a fold-size-dependent number with no clear interpretation.

    Args:
        fold_metrics (list[dict[str, float]]): The per-fold metrics, as
            returned by run_cross_validation.
        fold_sizes (list[int]): The validation set size of each fold, in the
            same order as fold_metrics.

    Returns:
        dict[str, float]: The aggregated metrics: a size-weighted mean for
            rate-like metrics, and a sum for confusion matrix counts.

    """
    total_size = sum(fold_sizes)

    aggregated = {}
    for name in fold_metrics[0]:
        values = [m[name] for m in fold_metrics]

        if name in COUNT_METRICS:
            aggregated[name] = sum(values)
        else:
            aggregated[name] = (
                sum(v * size for v, size in zip(values, fold_sizes, strict=True))
                / total_size
            )

    return aggregated
