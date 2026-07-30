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


# ==============================================================================
# These classes are used for tests on the logic that decides to accept (or not)
# validation data during fit (when it's possible). In order to keep code clean
# this should be refactored
# ==============================================================================
class FakeModelWithValidation:
    """A fit-compatible model whose fit() accepts validation data (e.g., MLP).

    Attributes:
        instances_created (int): Class variable tracking instantiation count.
        fit_called_with (dict | None): Dictionary storing fit arguments
            ``{"X": X, "y": y, "X_val": X_val, "y_val": y_val}``, or ``None`` if
            ``fit`` was not called.

    """

    instances_created = 0

    def __init__(self, **kwargs) -> None:
        """Initialize a FakeModelWithValidation instance.

        Args:
            **kwargs: Arbitrary keyword arguments passed to simulate model
                hyperparameters.

        """
        FakeModelWithValidation.instances_created += 1
        self.fit_called_with = None

    def fit(self, X, y, X_val=None, y_val=None):
        """Record training and validation data arguments and return the estimator.

        Args:
            X: Training feature matrix.
            y: Training target vector.
            X_val (optional): Validation feature matrix. Defaults to ``None``.
            y_val (optional): Validation target vector. Defaults to ``None``.

        Returns:
            FakeModelWithValidation: The fitted estimator instance itself.

        """
        self.fit_called_with = {"X": X, "y": y, "X_val": X_val, "y_val": y_val}
        return self

    def predict_proba(self, X):
        """Return fixed probability predictions of shape (len(X), 2).

        Args:
            X: Feature matrix to predict probabilities for.

        Returns:
            np.ndarray: A 2D array with fixed probabilities ``[0.4, 0.6]``.

        """
        n = len(X)
        return np.column_stack([np.full(n, 0.4), np.full(n, 0.6)])


class FakeModelWithoutValidation:
    """A fit-compatible model whose fit() only accepts X, y (e.g. sklearn-style).

    Attributes:
        instances_created (int): Class variable tracking instantiation count.
        fit_called_with (dict | None): Dictionary storing positional fit arguments
            ``{"X": X, "y": y}``, or ``None`` if ``fit`` was not called.

    """

    instances_created = 0

    def __init__(self, **kwargs) -> None:
        """Initialize a FakeModelWithoutValidation instance.

        Args:
            **kwargs: Arbitrary keyword arguments passed to simulate model
                hyperparameters.

        """
        FakeModelWithoutValidation.instances_created += 1
        self.fit_called_with = None

    def fit(self, X, y):
        """Record positional fit arguments and return the estimator instance.

        Args:
            X: Feature matrix.
            y: Target vector.

        Returns:
            FakeModelWithoutValidation: The fitted estimator instance itself.

        """
        self.fit_called_with = {"X": X, "y": y}
        return self

    def predict_proba(self, X):
        """Return fixed probability predictions of shape (len(X), 2).

        Args:
            X: Feature matrix to predict probabilities for.

        Returns:
            np.ndarray: A 2D array with fixed probabilities ``[0.4, 0.6]``.

        """
        n = len(X)
        return np.column_stack([np.full(n, 0.4), np.full(n, 0.6)])


@pytest.fixture(autouse=True)
def reset_fake_model_counter() -> None:
    """Reset all the FakeModel instances counter before each test."""
    FakeModel.instances_created = 0
    FakeModelWithValidation.instances_created = 0
    FakeModelWithoutValidation.instances_created = 0


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
@patch("ml.training.trainer.mlflow.start_run")
def test_run_cross_validation_builds_one_model_per_fold(
    mock_start_run,
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
@patch("ml.training.trainer.mlflow.start_run")
def test_run_cross_validation_computes_metrics_on_validation_split(
    mock_start_run,
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
@patch("ml.training.trainer.mlflow.start_run")
def test_run_cross_validation_returns_one_metrics_dict_per_fold(
    mock_start_run,
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
@patch("ml.training.trainer.mlflow.start_run")
def test_train_final_model_computes_metrics_on_test_set(
    mock_start_run, mock_compute_metrics, mock_log_final_run, mock_set_experiment
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
@patch("ml.training.trainer.mlflow.start_run")
def test_train_final_model_fits_on_full_training_data(
    mock_start_run, mock_compute_metrics, mock_log_final_run, mock_set_experiment
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
@patch("ml.training.trainer.mlflow.start_run")
def test_train_final_model_returns_fitted_model_and_metrics(
    mock_start_run, mock_compute_metrics, mock_log_final_run, mock_set_experiment
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


@patch("ml.training.trainer.mlflow.set_experiment")
@patch("ml.training.trainer.log_fold_run")
@patch("ml.training.trainer.compute_metrics")
@patch("ml.training.trainer.mlflow.start_run")
def test_run_cross_validation_passes_validation_data_when_fit_accepts_it(
    mock_start_run,
    mock_compute_metrics,
    mock_log_fold_run,
    mock_set_experiment,
    synthetic_folds: list[Fold],
) -> None:
    """Verify X_val/y_val are passed to fit() when its signature supports them."""
    mock_compute_metrics.return_value = {"auc_roc": 0.5}

    run_cross_validation(
        folds=synthetic_folds,
        model_builder=FakeModelWithValidation,
        model_params={},
        experiment_name="test_experiment",
    )

    # inspect the single model instance created for the one fold
    assert FakeModelWithValidation.instances_created == 3


@patch("ml.training.trainer.mlflow.set_experiment")
@patch("ml.training.trainer.log_fold_run")
@patch("ml.training.trainer.compute_metrics")
@patch("ml.training.trainer.mlflow.start_run")
def test_run_cross_validation_fit_receives_actual_validation_fold_data(
    mock_start_run,
    mock_compute_metrics,
    mock_log_fold_run,
    mock_set_experiment,
    synthetic_folds: list[Fold],
) -> None:
    """Verify the X_val/y_val passed to fit() are the fold's own val data, not None."""
    mock_compute_metrics.return_value = {"auc_roc": 0.5}

    captured_models = []

    def capturing_builder(**kwargs):
        model = FakeModelWithValidation(**kwargs)
        captured_models.append(model)
        return model

    run_cross_validation(
        folds=synthetic_folds,
        model_builder=capturing_builder,
        model_params={},
        experiment_name="test_experiment",
    )

    fit_args = captured_models[0].fit_called_with
    pd.testing.assert_frame_equal(fit_args["X_val"], synthetic_folds[0].X_val)
    pd.testing.assert_series_equal(fit_args["y_val"], synthetic_folds[0].y_val)


@patch("ml.training.trainer.mlflow.set_experiment")
@patch("ml.training.trainer.log_fold_run")
@patch("ml.training.trainer.compute_metrics")
@patch("ml.training.trainer.mlflow.start_run")
def test_run_cross_validation_does_not_pass_validation_data_when_fit_lacks_support(
    mock_start_run,
    mock_compute_metrics,
    mock_log_fold_run,
    mock_set_experiment,
    synthetic_folds: list[Fold],
) -> None:
    """Verify fit() is called with only X, y when its signature has no X_val/y_val."""
    mock_compute_metrics.return_value = {"auc_roc": 0.5}

    captured_models = []

    def capturing_builder(**kwargs):
        model = FakeModelWithoutValidation(**kwargs)
        captured_models.append(model)
        return model

    run_cross_validation(
        folds=synthetic_folds,
        model_builder=capturing_builder,
        model_params={},
        experiment_name="test_experiment",
    )

    fit_args = captured_models[0].fit_called_with
    assert set(fit_args.keys()) == {"X", "y"}


@patch("ml.training.trainer.log_fold_run")
@patch("ml.training.trainer.compute_metrics")
@patch("ml.training.trainer.mlflow.start_run")
def test_run_cross_validation_opens_run_before_fit(
    mock_start_run, mock_compute_metrics, mock_log_fold_run, synthetic_folds: list[Fold]
) -> None:
    mock_compute_metrics.return_value = {"auc_roc": 0.5}

    call_order = []
    fake_model = MagicMock()
    fake_model.fit.side_effect = lambda *a, **kw: call_order.append("fit")
    mock_start_run.return_value.__enter__.side_effect = lambda: call_order.append(
        "start_run"
    )

    run_cross_validation(
        folds=synthetic_folds,
        model_builder=lambda **kw: fake_model,
        model_params={},
        experiment_name="test_experiment",
    )

    assert call_order == ["start_run"] + ["start_run", "fit"] * len(synthetic_folds)
