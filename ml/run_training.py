"""Entrypoint for the ML training pipeline.

Prepare the training data once, then run cross-validation and final model
fitting for each selected model family.

"""

import gc
import logging
from collections.abc import Callable
from typing import NamedTuple

import mlflow
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import settings
from database.connection import get_db
from ml.dataset.loader import load_data
from ml.dataset.preprocessing import preprocess_test_set, preprocess_train_folds
from ml.dataset.split import Fold, train_val_test_split
from ml.models.baseline import DEFAULT_C, DEFAULT_MAX_ITER, build_baseline_model
from ml.models.mlp import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_LAYERS,
    DEFAULT_LEARNING_RATE_MLP,
    DEFAULT_WEIGHT_DECAY,
    build_mlp_model,
)
from ml.models.protocol import Estimator
from ml.models.xgboost_model import (
    DEFAULT_EVAL_METRIC,
    DEFAULT_LEARNING_RATE_XGB,
    DEFAULT_MAX_DEPTH,
    DEFAULT_N_ESTIMATORS,
    build_xgboost_model,
)
from ml.training.trainer import run_cross_validation, train_final_model
from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


class FinalSplit(NamedTuple):
    """Represent the final train/test split, ready for the final model fit.

    Attributes:
        X_train (pd.DataFrame): Full training/CV feature matrix.
        y_train (pd.Series): Full training/CV target labels.
        X_test (pd.DataFrame): Held-out test feature matrix.
        y_test (pd.Series): Held-out test target labels.
        encoder (OneHotEncoder): The OneHotEncoder fitted on the full
            training/CV data.
        scaler (StandardScaler | None): The StandardScaler fitted on the
            full training/CV data, or None if scaling was not applied.

    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    encoder: OneHotEncoder
    scaler: StandardScaler | None


class ModelConfig(NamedTuple):
    """Represent the configuration needed to train one model family.

    Attributes:
        experiment_name (str): The MLflow experiment name for this model
            family (e.g. "baseline"). Also used to select which models to
            train via `main(models_to_train=...)`.
        model_builder (Callable[..., Estimator]): A factory function that
            returns a fresh, unfitted model instance.
        model_params (dict): Keyword arguments passed to `model_builder`.
        scale (bool): Whether this model family requires scaled features
            (True for linear models and the MLP, False for tree-based models
            like XGBoost, which are scale-invariant).
        handle_nan (bool): Whether to create informative missing
            value flags and apply group-specific fallback constants. Set to
            ``False`` for model families with native missing value handling
            (e.g. XGBoost), which learn the optimal split direction for
            missing data directly rather than relying on an explicit
            imputation strategy.
        threshold (float, optional): The value of the classification threshold. In
            `evaluation/plots.ipynb` the optimal value is computed maximizing
            `F2-score` on aggregated CV folds. Defaults to 0.5.

    """

    experiment_name: str
    model_builder: Callable[..., Estimator]
    model_params: dict
    scale: bool
    handle_nan: bool
    threshold: float = 0.5


MODEL_CONFIGS = [
    ModelConfig(
        experiment_name="baseline",
        model_builder=build_baseline_model,
        model_params={
            "C": DEFAULT_C,
            "max_iter": DEFAULT_MAX_ITER,
        },
        scale=True,
        handle_nan=True,
        # optimal threshold, see plots.ipynb (F2-optimal on aggregated CV folds)
        threshold=0.4078,
    ),
    ModelConfig(
        experiment_name="mlp",
        model_builder=build_mlp_model,
        model_params={
            "hidden_layers": DEFAULT_HIDDEN_LAYERS,
            "dropout": DEFAULT_DROPOUT,
            "epochs": DEFAULT_EPOCHS,
            "batch_size": DEFAULT_BATCH_SIZE,
            "learning_rate": DEFAULT_LEARNING_RATE_MLP,
            "weight_decay": DEFAULT_WEIGHT_DECAY,
            "pos_weight": 5.25,
        },
        scale=True,
        handle_nan=True,
        # optimal threshold, see plots.ipynb (F2-optimal on aggregated CV folds)
        threshold=0.5,  # 0.0012: ignored: suspiciously low, and not affecting results
    ),
    ModelConfig(
        experiment_name="xgboost_model",
        model_builder=build_xgboost_model,
        model_params={
            "n_estimators": DEFAULT_N_ESTIMATORS,
            "max_depth": DEFAULT_MAX_DEPTH,
            "learning_rate": DEFAULT_LEARNING_RATE_XGB,
            "scale_pos_weight": 5.25,
            "eval_metric": DEFAULT_EVAL_METRIC,
        },
        scale=False,
        handle_nan=False,
        # optimal threshold, see plots.ipynb (F2-optimal on aggregated CV folds)
        threshold=0.4580,
    ),
]


def prepare_training_data(
    scale: bool, handle_nan: bool = True
) -> tuple[list[Fold], FinalSplit]:
    """Load, split, and preprocess the star schema data for training.

    Encapsulates the full data preparation pipeline: loading the star schema
    into a single DataFrame, isolating the temporal holdout test set,
    generating cross-validation folds, and preprocessing both the folds and
    the final train/test split.

    Args:
        scale (bool): Whether to apply feature scaling, required for the
            baseline and MLP models but not for XGBoost.
        handle_nan (bool, optional): Whether to create informative missing
            value flags and apply group-specific fallback constants. Set to
            ``False`` for model families with native missing value handling
            (e.g. XGBoost), which learn the optimal split direction for
            missing data directly rather than relying on an explicit
            imputation strategy. Defaults to ``True``.

    Returns:
        tuple[list[Fold], FinalSplit]: The preprocessed cross-validation
            folds, and the final train/test split ready for the final model fit.

    """
    session_generator = get_db()
    session = next(session_generator)

    dataset = load_data(session)
    train_folds, df_remaining, df_test = train_val_test_split(df=dataset)

    train_folds = preprocess_train_folds(
        folds=train_folds, scale=scale, handle_nan=handle_nan
    )
    X_train, y_train, X_test, y_test, encoder, scaler = preprocess_test_set(
        df_train_full=df_remaining, df_test=df_test, scale=scale, handle_nan=handle_nan
    )

    return train_folds, FinalSplit(X_train, y_train, X_test, y_test, encoder, scaler)


def main(models_to_train: list[str] | None = None) -> None:
    """Run cross-validation and final training for the selected model families.

    Args:
        models_to_train (list[str] | None, optional): Names of the
            experiments to train, matching each `ModelConfig.experiment_name`
            (e.g. `["baseline", "xgboost"]`). If `None`, every configured
            model is trained. Defaults to `None`.

    """
    for config in MODEL_CONFIGS:
        if (
            models_to_train is not None
            and config.experiment_name not in models_to_train
        ):
            logger.info(
                "Skipping experiment '%s' (not selected).", config.experiment_name
            )
            continue

        logger.info("Starting training for experiment '%s'...", config.experiment_name)

        train_folds, final_split = prepare_training_data(
            scale=config.scale, handle_nan=config.handle_nan
        )

        run_cross_validation(
            folds=train_folds,
            model_builder=config.model_builder,
            model_params=config.model_params,
            experiment_name=config.experiment_name,
        )

        train_final_model(
            X_train_full=final_split.X_train,
            y_train_full=final_split.y_train,
            X_test=final_split.X_test,
            y_test=final_split.y_test,
            encoder=final_split.encoder,
            scaler=final_split.scaler,
            model_builder=config.model_builder,
            model_params=config.model_params,
            experiment_name=config.experiment_name,
            threshold=config.threshold,
        )

        del train_folds, final_split
        gc.collect()


if __name__ == "__main__":
    setup_logging("INFO")
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    # main(models_to_train=["baseline"]) # to train only baseline model
    # main(models_to_train=["mlp"])  # to train only the MLP model
    # main(models_to_train=["xgboost_model"])  # to train only the XGBoost model
    main()  # to train all models
