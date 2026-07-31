"""Encapsulate MLflow experiment tracking details for the training pipeline.

Provide logging utilities for cross-validation fold runs (nested under a
parent run) and the final holdout evaluation run, keeping ``training.trainer``
free of MLflow-specific implementation details.

"""

import datetime
import logging

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBModel

from ml.models.mlp import MLPClassifier
from ml.models.protocol import Estimator

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
    model: Estimator,
    params: dict,
    metrics: dict[str, float],
    fold_index: int,
    y_val: pd.Series,
    y_pred_proba: np.ndarray,
) -> None:
    """Log a single cross-validation fold as a nested MLflow run.

    Must be called while a parent MLflow run is already active, so the
    resulting run appears nested under it in the MLflow UI. Does not log the
    model itself as an artifact, since fold-level models are only used for
    performance comparison, not for inference. Logs the raw validation
    predictions as a CSV artifact, so threshold tuning and ROC/PR curves can
    be reconstructed later without refitting the model.

    Args:
        model (Estimator): The fitted model for this fold (used only to
            infer its class name for logging context, not persisted).
        params (dict): The hyperparameters used to build the model.
        metrics (dict[str, float]): The computed validation metrics for this
            fold.
        fold_index (int): The 1-based index of this fold, used to name the
            run.
        y_val (pd.Series): The true validation target labels for this fold.
        y_pred_proba (np.ndarray): The predicted probabilities for this
            fold's validation split.

    """
    mlflow.log_param("model_class", type(model).__name__)
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    predictions_df = pd.DataFrame(
        {"y_true": y_val.values, "y_pred_proba": y_pred_proba}
    )
    predictions_df.to_csv(f"/tmp/fold_{fold_index}_predictions.csv", index=False)
    mlflow.log_artifact(
        f"/tmp/fold_{fold_index}_predictions.csv", artifact_path="predictions"
    )

    logger.info("Logged MLflow run for fold %d.", fold_index)


def log_final_run(
    model: Estimator,
    encoder: OneHotEncoder,
    scaler: StandardScaler | None,
    params: dict,
    metrics: dict[str, float],
    y_test: pd.Series,
    y_pred_proba: np.ndarray,
) -> None:
    """Log the final model, fitted on the full training/CV data, as an MLflow run.

    Unlike `log_fold_run`, this also logs the fitted model, the fitted
    OneHotEncoder, and (if used) the fitted StandardScaler as artifacts,
    since this is the model intended for downstream inference — the
    inference layer must apply the exact same fitted transformations used
    during training, never refitting them on new data. Also logs the raw
    test set predictions as a CSV artifact, so threshold tuning and ROC/PR
    curves on the holdout test set can be reconstructed later without
    rerunning inference.

    Args:
        model (Estimator): The final fitted model.
        encoder (OneHotEncoder): The OneHotEncoder fitted on the full
            training/CV data.
        scaler (StandardScaler | None): The StandardScaler fitted on the
            full training/CV data, or None if scaling was not applied
            (e.g. for tree-based models like XGBoost).
        params (dict): The hyperparameters used to build the model.
        metrics (dict[str, float]): The computed test set metrics.
        y_test (pd.Series): The true holdout test target labels.
        y_pred_proba (np.ndarray): The predicted probabilities on the
            holdout test set.

    """
    mlflow.log_param("model_class", type(model).__name__)
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    if isinstance(model, XGBModel):
        mlflow.xgboost.log_model(xgb_model=model, name="model")
    elif isinstance(model, BaseEstimator):
        mlflow.sklearn.log_model(model, name="model")
    elif isinstance(model, MLPClassifier):
        # the serialization_format bypass the data example typically required in input
        mlflow.pytorch.log_model(
            model.network, name="model", serialization_format="pickle"
        )
    else:
        logger.error("Unrecognized model type: %s", type(model).__name__)
        raise TypeError(f"Cannot log model of type {type(model).__name__}")

    mlflow.sklearn.log_model(encoder, name="encoder")
    if scaler is not None:
        mlflow.sklearn.log_model(scaler, name="scaler")

    predictions_df = pd.DataFrame(
        {"y_true": y_test.values, "y_pred_proba": y_pred_proba}
    )
    predictions_df.to_csv("/tmp/final_model_predictions.csv", index=False)
    mlflow.log_artifact("/tmp/final_model_predictions.csv", artifact_path="predictions")

    logger.info(
        "Logged final MLflow run with model, encoder, and scaler artifacts "
        "(when are used)."
    )
