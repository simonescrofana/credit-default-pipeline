"""Orchestrate model training across CV folds and the final test set.

Provide a model-agnostic training loop that accepts any scikit-learn-compatible
model builder, delegating metric computation to `evaluation.metrics` and
experiment tracking to `training.mlflow_utils`. Training duration, at both
the overall and per-fold level, is captured via Logfire spans.

"""

import logging
from collections.abc import Callable

import logfire
import mlflow
import pandas as pd
from sklearn.base import BaseEstimator

from ml.dataset.split import Fold
from ml.evaluation.metrics import aggregate_fold_metrics, compute_metrics
from ml.training.mlflow_utils import log_final_run, log_fold_run, timestamped_run_name

logger = logging.getLogger(__name__)


def run_cross_validation(
    folds: list[Fold],
    model_builder: Callable[..., BaseEstimator],
    model_params: dict,
    experiment_name: str,
    threshold: float = 0.5,
) -> list[dict[str, float]]:
    """Train and evaluate a model across all cross-validation folds.

    For each fold, build a fresh model instance, fit it on the training
    split, predict on the validation split, compute evaluation metrics, and
    log the run to MLflow as a child run nested under a single parent
    cross-validation run. After all folds are processed, a size-weighted
    aggregate of the metrics is computed and logged on the parent run
    itself. The overall cross-validation run and each individual fold are
    also wrapped in Logfire spans, capturing their duration.

    Args:
        folds (list[Fold]): The preprocessed cross-validation folds.
        model_builder (Callable[..., BaseEstimator]): A factory function that
            returns a fresh, unfitted model instance (e.g.
            `build_baseline_model`).
        model_params (dict): Keyword arguments passed to `model_builder` on
            every call.
        experiment_name (str): The MLflow experiment to log runs under (one
            experiment per model family, e.g. `"baseline"`).
        threshold (float): The value of the classification threshold. In
            `evaluation/plots.ipynb` the optimal value is computed maximizing
            `F2-score` on aggregated CV folds. Defaults to 0.5.

    Returns:
        list[dict[str, float]]: The computed metrics for each fold, in fold order.

    """
    logger.info("Starting cross-validation across %d folds...", len(folds))
    fold_metrics = []
    fold_sizes = []

    mlflow.set_experiment(experiment_name)

    with logfire.span("cross_validation_run", n_folds=len(folds)):
        with mlflow.start_run(run_name=timestamped_run_name("cross_validation")):
            for i, fold in enumerate(folds, start=1):
                with logfire.span("fold_training", fold_index=i):
                    model = model_builder(**model_params)
                    model.fit(fold.X_train, fold.y_train)

                    y_pred_proba = model.predict_proba(fold.X_val)[:, 1]
                    metrics = compute_metrics(fold.y_val, y_pred_proba, threshold)

                    log_fold_run(
                        model=model,
                        params=model_params,
                        metrics=metrics,
                        fold_index=i,
                        y_val=fold.y_val,
                        y_pred_proba=y_pred_proba,
                    )

                fold_metrics.append(metrics)
                fold_sizes.append(len(fold.y_val))
                logger.info("Fold %d/%d completed. Metrics: %s", i, len(folds), metrics)

            aggregated_metrics = aggregate_fold_metrics(fold_metrics, fold_sizes)
            mlflow.log_metrics({f"agg_{k}": v for k, v in aggregated_metrics.items()})
            logger.info("Aggregated cross-validation metrics: %s", aggregated_metrics)

    logger.info("Cross-validation completed for all %d folds.", len(folds))
    return fold_metrics


def train_final_model(
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_builder: Callable[..., BaseEstimator],
    model_params: dict,
    experiment_name: str,
    threshold: float = 0.5,
) -> tuple[BaseEstimator, dict[str, float]]:
    """Fit and evaluate the final model on the full training/CV and test data.

    This must be called only after model selection and hyperparameter tuning
    are complete, using the full pre-test data rather than a single
    cross-validation fold. The fit and evaluation steps are wrapped in
    Logfire spans, capturing their duration separately.

    Args:
        X_train_full (pd.DataFrame): The full training/CV feature matrix.
        y_train_full (pd.Series): The full training/CV target labels.
        X_test (pd.DataFrame): The held-out test feature matrix.
        y_test (pd.Series): The held-out test target labels.
        model_builder (Callable[..., BaseEstimator]): A factory function that
            returns a fresh, unfitted model instance.
        model_params (dict): Keyword arguments passed to `model_builder`.
        experiment_name (str): The MLflow experiment to log this run under
            (the same experiment used for the model's cross-validation runs).
        threshold (float, optional): The value of the classification threshold. In
            `evaluation/plots.ipynb` the optimal value is computed maximizing
            `F2-score` on aggregated CV folds. Defaults to 0.5.

    Returns:
        tuple[BaseEstimator, dict[str, float]]: The fitted model and its
            metrics on the holdout test set.

    """
    logger.info("Fitting the final model on the full training/CV data...")

    mlflow.set_experiment(experiment_name)

    with logfire.span("final_model_fit"):
        model = model_builder(**model_params)
        model.fit(X_train_full, y_train_full)

    with logfire.span("final_model_evaluation"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred_proba, threshold)

    log_final_run(
        model=model,
        params=model_params,
        metrics=metrics,
        y_test=y_test,
        y_pred_proba=y_pred_proba,
    )

    logger.info("Final model evaluated on the holdout test set. Metrics: %s", metrics)
    return model, metrics
