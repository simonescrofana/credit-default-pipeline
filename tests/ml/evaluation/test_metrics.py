"""Tests for ml.evaluation.metrics.

Cover the happy path of compute_metrics, along with the edge case of a
validation set with no positive examples, which exercises the explicit
labels=[0, 1] safeguard against a malformed confusion matrix.

"""

import numpy as np
import pandas as pd
import pytest

from ml.evaluation.metrics import aggregate_fold_metrics, compute_metrics


@pytest.fixture
def sample_predictions() -> tuple[pd.Series, np.ndarray]:
    """Provide plausible true labels and predicted probabilities for a small batch.

    Returns:
        tuple[pd.Series, np.ndarray]: 8 true labels (2 positive, 6 negative)
            and their corresponding predicted probabilities, mixing confident
            correct predictions, a false positive, and a false negative.

    """
    y_true = pd.Series([0, 0, 0, 0, 0, 0, 1, 1])
    y_pred_proba = np.array([0.05, 0.10, 0.20, 0.55, 0.15, 0.30, 0.90, 0.40])
    return y_true, y_pred_proba


def test_compute_metrics_happy_path(
    sample_predictions: tuple[pd.Series, np.ndarray],
) -> None:
    """Verify the metric dict shape and confusion matrix counts on a plausible batch."""
    y_true, y_pred_proba = sample_predictions

    metrics = compute_metrics(y_true, y_pred_proba, threshold=0.5)

    assert set(metrics.keys()) == {
        "auc_roc",
        "auc_pr",
        "precision",
        "recall",
        "f1",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
    }

    # at threshold 0.5: predicted positives are index 3 (0.55, false positive)
    # and index 6 (0.90, true positive); index 7 (0.40, actual positive) is a
    # false negative
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_negatives"] == 5

    assert 0.0 <= metrics["auc_roc"] <= 1.0
    assert 0.0 <= metrics["auc_pr"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0


def test_compute_metrics_threshold_changes_confusion_counts(
    sample_predictions: tuple[pd.Series, np.ndarray],
) -> None:
    """Verify that a lower decision threshold increases recall at the same data."""
    y_true, y_pred_proba = sample_predictions

    metrics_low = compute_metrics(y_true, y_pred_proba, threshold=0.1)
    metrics_high = compute_metrics(y_true, y_pred_proba, threshold=0.9)

    # a lower threshold flags more predictions as positive, raising recall
    # at the expense of precision
    assert metrics_low["recall"] >= metrics_high["recall"]


def test_compute_metrics_handles_fold_with_no_positive_examples() -> None:
    """Verify that a fold with no positive examples yields a 2x2 confusion matrix."""
    y_true = pd.Series([0, 0, 0, 0, 0])
    y_pred_proba = np.array([0.1, 0.2, 0.05, 0.4, 0.15])

    metrics = compute_metrics(y_true, y_pred_proba)

    # confusion matrix must still resolve to a well-formed 2x2 matrix
    assert metrics["true_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["true_negatives"] == 5
    assert metrics["false_positives"] == 0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_compute_metrics_returns_native_python_ints(
    sample_predictions: tuple[pd.Series, np.ndarray],
) -> None:
    """Verify that confusion matrix counts are native Python ints, not np.int64.

    MLflow metric logging requires plain Python numeric types, so this guards
    against a future refactor silently dropping the explicit int(...) cast.

    """
    y_true, y_pred_proba = sample_predictions

    metrics = compute_metrics(y_true, y_pred_proba)

    for key in (
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
    ):
        assert isinstance(metrics[key], int)


@pytest.fixture
def two_fold_metrics() -> tuple[list[dict], list[int]]:
    """Provide 2 folds of unequal size with distinct rate and count metrics.

    Returns:
        tuple[list[dict], list[int]]: Fold metrics and their validation set
        sizes (1000 and 3000 rows), chosen to make weighted vs unweighted
        aggregation produce clearly different results.

    """
    fold_metrics = [
        {
            "auc_roc": 0.60,
            "precision": 0.50,
            "true_positives": 3,
            "false_positives": 5,
            "true_negatives": 900,
            "false_negatives": 92,
        },
        {
            "auc_roc": 0.90,
            "precision": 0.80,
            "true_positives": 7,
            "false_positives": 2,
            "true_negatives": 2900,
            "false_negatives": 91,
        },
    ]
    fold_sizes = [1000, 3000]
    return fold_metrics, fold_sizes


def test_aggregate_fold_metrics_weights_rate_metrics_by_fold_size(
    two_fold_metrics: tuple[list[dict], list[int]],
) -> None:
    """Verify rate-like metrics use a size-weighted mean, not a plain average."""
    fold_metrics, fold_sizes = two_fold_metrics

    result = aggregate_fold_metrics(fold_metrics, fold_sizes)

    # plain average would give 0.75; size-weighted mean should lean toward
    # the larger fold's value (0.90) since it has 3x the weight
    expected_auc_roc = (0.60 * 1000 + 0.90 * 3000) / 4000
    assert result["auc_roc"] == pytest.approx(expected_auc_roc)
    assert result["auc_roc"] != pytest.approx((0.60 + 0.90) / 2)


def test_aggregate_fold_metrics_sums_confusion_matrix_counts(
    two_fold_metrics: tuple[list[dict], list[int]],
) -> None:
    """Verify confusion matrix counts are summed, not averaged."""
    fold_metrics, fold_sizes = two_fold_metrics

    result = aggregate_fold_metrics(fold_metrics, fold_sizes)

    assert result["true_positives"] == 3 + 7
    assert result["false_positives"] == 5 + 2
    assert result["true_negatives"] == 900 + 2900
    assert result["false_negatives"] == 92 + 91


def test_aggregate_fold_metrics_confusion_counts_remain_native_ints(
    two_fold_metrics: tuple[list[dict], list[int]],
) -> None:
    """Verify summed confusion matrix counts stay plain ints, not floats."""
    fold_metrics, fold_sizes = two_fold_metrics

    result = aggregate_fold_metrics(fold_metrics, fold_sizes)

    for key in (
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
    ):
        assert isinstance(result[key], int)


def test_aggregate_fold_metrics_single_fold_returns_same_values(
    two_fold_metrics: tuple[list[dict], list[int]],
) -> None:
    """Verify aggregating a single fold returns that fold's own values unchanged."""
    fold_metrics, fold_sizes = two_fold_metrics

    result = aggregate_fold_metrics([fold_metrics[0]], [fold_sizes[0]])

    assert result == fold_metrics[0]
