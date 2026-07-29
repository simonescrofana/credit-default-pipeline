"""Tests for ml.training.trainer.

Use a fake model builder and a spy on compute_metrics to verify the
orchestration logic of run_cross_validation and train_final_model in
isolation from real scikit-learn models and MLflow.

"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.dataset.split import Fold
from ml.training.trainer import run_cross_validation, train_final_model


class FakeModel:
    """A minimal stand-in for a scikit-learn-compatible model.

    Records how many times `fit` was called and always predicts a fixed
    probability, so tests can verify orchestration behavior without any real
    training happening.

    Attributes:
        instances_created (int): Class variable tracking the total number of
            `FakeModel` instances instantiated.
        fit_called_with (tuple[pd.DataFrame, pd.Series] | None): Stored tuple
            of `(X, y)` passed during the last call to `fit`, or `None`
            if `fit` has not been called yet.
        kwargs (dict): Arbitrary keyword arguments captured during
            initialization.

    """

    instances_created = 0

    def __init__(self, **kwargs) -> None:
        """Initialize a FakeModel instance and track global instantiation count.

        Args:
            **kwargs: Arbitrary keyword arguments passed to simulate model
                hyperparameters.

        """
        FakeModel.instances_created += 1
        self.fit_called_with = None
        self.kwargs = kwargs

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Record the training data passed to the model.

        Args:
            X (pd.DataFrame): Training feature matrix.
            y (pd.Series): Training target labels.

        """
        self.fit_called_with = (X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return constant predicted probabilities for each row.

        Args:
            X (pd.DataFrame): Feature matrix for which to predict probabilities.

        Returns:
            np.ndarray: A 2D array of shape `(len(X), 2)` where each row contains
            fixed class probabilities `[0.4, 0.6]`.

        """
        n = len(X)
        return np.column_stack([np.full(n, 0.4), np.full(n, 0.6)])


@pytest.fixture(autouse=True)
def reset_fake_model_counter() -> None:
    """Reset the FakeModel instance counter before each test."""
    FakeModel.instances_created = 0


def fake_model_builder(**kwargs) -> FakeModel:
    """Build a fresh FakeModel instance, mirroring a real model builder's signature."""
    return FakeModel(**kwargs)


@pytest.fixture
def synthetic_folds() -> list[Fold]:
    """Provide 3 minimal synthetic folds with a single feature column.

    Returns:
        list[Fold]: Three folds, each with 4 training rows and 2 validation
            rows, sufficient to verify per-fold orchestration behavior.

    """
    folds = []
    for i in range(3):
        X_train = pd.DataFrame({"feature": range(4)})
        y_train = pd.Series([0, 0, 1, 0])
        X_val = pd.DataFrame({"feature": range(2)})
        y_val = pd.Series([0, 1])
        folds.append(Fold(X_train, y_train, X_val, y_val))
    return folds


@patch("ml.training.trainer.mlflow.set_experiment")
@patch("ml.training.trainer.log_fold_run")
@patch("ml.training.trainer.compute_metrics")
def test_run_cross_validation_builds_one_model_per_fold(
    mock_compute_metrics,
    mock_log_fold_run,
    mock_set_experiment,
    synthetic_folds: list[Fold],
) -> None:
    """Verify a new model instance is built for every fold, not shared across folds."""
    mock_compute_metrics.return_value = {"auc_roc": 0.5}

    run_cross_validation(
        folds=synthetic_folds,
        model_builder=fake_model_builder,
        model_params={},
        experiment_name="test_experiment",
    )

    assert FakeModel.instances_created == len(synthetic_folds)


@patch("ml.training.trainer.mlflow.set_experiment")
@patch("ml.training.trainer.log_fold_run")
@patch("ml.training.trainer.compute_metrics")
def test_run_cross_validation_computes_metrics_on_validation_split(
    mock_compute_metrics,
    mock_log_fold_run,
    mock_set_experiment,
    synthetic_folds: list[Fold],
) -> None:
    """Verify compute_metrics is called with each fold's own y_val, not y_train."""
    mock_compute_metrics.return_value = {"auc_roc": 0.5}

    run_cross_validation(
        folds=synthetic_folds,
        model_builder=fake_model_builder,
        model_params={},
        experiment_name="test_experiment",
    )

    assert mock_compute_metrics.call_count == len(synthetic_folds)

    for call, fold in zip(
        mock_compute_metrics.call_args_list, synthetic_folds, strict=True
    ):
        y_true_arg = call.args[0]
        pd.testing.assert_series_equal(y_true_arg, fold.y_val)


@patch("ml.training.trainer.mlflow.set_experiment")
@patch("ml.training.trainer.log_fold_run")
@patch("ml.training.trainer.compute_metrics")
def test_run_cross_validation_returns_one_metrics_dict_per_fold(
    mock_compute_metrics,
    mock_log_fold_run,
    mock_set_experiment,
    synthetic_folds: list[Fold],
) -> None:
    """Verify the returned list has exactly one metrics dict per fold, in order."""
    mock_compute_metrics.return_value = {"auc_roc": 0.5}

    result = run_cross_validation(
        folds=synthetic_folds,
        model_builder=fake_model_builder,
        model_params={},
        experiment_name="test_experiment",
    )

    assert result == [{"auc_roc": 0.5}] * len(synthetic_folds)


@patch("ml.training.trainer.mlflow.set_experiment")
@patch("ml.training.trainer.log_final_run")
@patch("ml.training.trainer.compute_metrics")
def test_train_final_model_computes_metrics_on_test_set(
    mock_compute_metrics, mock_log_final_run, mock_set_experiment
) -> None:
    """Verify compute_metrics is called with y_test, not y_train_full."""
    mock_compute_metrics.return_value = {"auc_roc": 0.7}

    X_train_full = pd.DataFrame({"feature": range(6)})
    y_train_full = pd.Series([0, 0, 1, 0, 0, 1])
    X_test = pd.DataFrame({"feature": range(2)})
    y_test = pd.Series([0, 1])

    fake_encoder = MagicMock(spec=OneHotEncoder)
    fake_scaler = MagicMock(spec=StandardScaler)

    train_final_model(
        X_train_full=X_train_full,
        y_train_full=y_train_full,
        X_test=X_test,
        y_test=y_test,
        encoder=fake_encoder,
        scaler=fake_scaler,
        model_builder=fake_model_builder,
        model_params={},
        experiment_name="test_experiment",
    )

    y_true_arg = mock_compute_metrics.call_args.args[0]
    pd.testing.assert_series_equal(y_true_arg, y_test)


@patch("ml.training.trainer.mlflow.set_experiment")
@patch("ml.training.trainer.log_final_run")
@patch("ml.training.trainer.compute_metrics")
def test_train_final_model_fits_on_full_training_data(
    mock_compute_metrics, mock_log_final_run, mock_set_experiment
) -> None:
    """Verify the model is fitted on the full train/CV data, not a single fold."""
    mock_compute_metrics.return_value = {"auc_roc": 0.7}

    X_train_full = pd.DataFrame({"feature": range(6)})
    y_train_full = pd.Series([0, 0, 1, 0, 0, 1])
    X_test = pd.DataFrame({"feature": range(2)})
    y_test = pd.Series([0, 1])

    fake_encoder = MagicMock(spec=OneHotEncoder)
    fake_scaler = MagicMock(spec=StandardScaler)

    model, _ = train_final_model(
        X_train_full=X_train_full,
        y_train_full=y_train_full,
        X_test=X_test,
        y_test=y_test,
        encoder=fake_encoder,
        scaler=fake_scaler,
        model_builder=fake_model_builder,
        model_params={},
        experiment_name="test_experiment",
    )

    fitted_X, fitted_y = model.fit_called_with
    pd.testing.assert_frame_equal(fitted_X, X_train_full)
    pd.testing.assert_series_equal(fitted_y, y_train_full)


@patch("ml.training.trainer.mlflow.set_experiment")
@patch("ml.training.trainer.log_final_run")
@patch("ml.training.trainer.compute_metrics")
def test_train_final_model_returns_fitted_model_and_metrics(
    mock_compute_metrics, mock_log_final_run, mock_set_experiment
) -> None:
    """Verify the function returns the fitted model instance and its metrics."""
    mock_compute_metrics.return_value = {"auc_roc": 0.7}

    X_train_full = pd.DataFrame({"feature": range(6)})
    y_train_full = pd.Series([0, 0, 1, 0, 0, 1])
    X_test = pd.DataFrame({"feature": range(2)})
    y_test = pd.Series([0, 1])

    fake_encoder = MagicMock(spec=OneHotEncoder)
    fake_scaler = MagicMock(spec=StandardScaler)

    model, metrics = train_final_model(
        X_train_full=X_train_full,
        y_train_full=y_train_full,
        X_test=X_test,
        y_test=y_test,
        encoder=fake_encoder,
        scaler=fake_scaler,
        model_builder=fake_model_builder,
        model_params={},
        experiment_name="test_experiment",
    )

    assert isinstance(model, FakeModel)
    assert metrics == {"auc_roc": 0.7}
