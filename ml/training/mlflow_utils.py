"""Encapsulate MLflow experiment tracking details for the training pipeline.

Provide logging utilities for cross-validation fold runs (nested under a
parent run) and the final holdout evaluation run, keeping ``training.trainer``
free of MLflow-specific implementation details.

"""

import datetime
import logging

import mlflow
import mlflow.sklearn
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


def timestamped_run_name(base_name: str) -> str:
    """Append a UTC timestamp to a run name for readability across repeated runs.

    MLflow already guarantees run uniqueness via its own Run ID, but repeated
    executions of the same training script otherwise produce runs and models
    with identical, indistinguishable names in the UI (e.g. multiple runs all
    named "cross_validation" or artifacts all named "model"). Appending a
    timestamp makes runs visually distinguishable without introducing a
    manually-tracked counter, which could drift out of sync if a run crashes
    partway through.

    Args:
        base_name (str): The semantic run name (e.g. "cross_validation",
            "fold_1", "final_model").

    Returns:
        str: The base name suffixed with a "YYYYMMDD_HHMMSS" UTC timestamp.

    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}"

def log_fold_run(
    model: BaseEstimator, params: dict, metrics: dict[str, float], fold_index: int
) -> None:
    """Log a single cross-validation fold as a nested MLflow run.

    Must be called while a parent MLflow run is already active, so the
    resulting run appears nested under it in the MLflow UI. Does not log the
    model itself as an artifact, since fold-level models are only used for
    performance comparison, not for inference.

    Args:
        model (BaseEstimator): The fitted model for this fold (used only to
            infer its class name for logging context, not persisted).
        params (dict): The hyperparameters used to build the model.
        metrics (dict[str, float]): The computed validation metrics for this fold.
        fold_index (int): The 1-based index of this fold, used to name the run.

    """
    with mlflow.start_run(run_name=timestamped_run_name(f"fold_{fold_index}"), nested=True):
        mlflow.log_param("model_class", type(model).__name__)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

    logger.info("Logged MLflow run for fold %d.", fold_index)


def log_final_run(
    model: BaseEstimator, params: dict, metrics: dict[str, float]
) -> None:
    """Log the final model, fitted on the full training/CV data, as an MLflow run.

    Unlike `log_fold_run`, this also logs the fitted model itself as an
    artifact, since this is the model intended for downstream inference.

    Args:
        model (BaseEstimator): The final fitted model.
        params (dict): The hyperparameters used to build the model.
        metrics (dict[str, float]): The computed test set metrics.

    """
    with mlflow.start_run(run_name=timestamped_run_name("final_model")):
        mlflow.log_param("model_class", type(model).__name__)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model")

    logger.info("Logged final MLflow run with model artifact.")
