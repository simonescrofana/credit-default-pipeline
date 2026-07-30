"""Tests for the MLP model.

Cover fit's happy path with and without validation data, predict_proba's
guard against being called before fit, build_mlp_model's independent
instances across calls, and the active_run guard around MLflow logging.

"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ml.models.mlp import MLPClassifier, build_mlp_model

# Small settings to keep tests fast: few epochs, tiny network, small batch
FAST_TEST_KWARGS = {
    "hidden_layers": (4, 2),
    "dropout": 0.1,
    "epochs": 3,
    "batch_size": 5,
    "learning_rate": 1e-2,
    "weight_decay": 0.0,
    "pos_weight": 1.0,
}


@pytest.fixture
def synthetic_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provide a small synthetic training set with 2 classes present.

    Returns:
        tuple[pd.DataFrame, pd.Series]: 20 rows, 3 numeric features, and a
            binary target with both classes represented.

    """
    X = pd.DataFrame(
        {
            "feature_1": range(10),
            "feature_2": [i * 0.5 for i in range(10)],
            "feature_3": [1, 0] * 5,
        }
    )
    y = pd.Series([0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
    return X, y


@pytest.fixture
def synthetic_validation_data() -> tuple[pd.DataFrame, pd.Series]:
    """Provide a small synthetic validation set, disjoint from training.

    Returns:
        tuple[pd.DataFrame, pd.Series]: 6 rows, same schema as the training
            fixture.

    """
    X_val = pd.DataFrame(
        {
            "feature_1": range(20, 23),
            "feature_2": [i * 0.5 for i in range(20, 23)],
            "feature_3": [1, 0, 1],
        }
    )
    y_val = pd.Series([0, 1, 0])
    return X_val, y_val


def test_fit_happy_path_without_validation(
    synthetic_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verify fit trains and returns self when no validation data is given."""
    X, y = synthetic_training_data
    model = MLPClassifier(**FAST_TEST_KWARGS)

    result = model.fit(X, y)

    assert result is model
    assert model.network is not None


def test_fit_happy_path_with_validation(
    synthetic_training_data: tuple[pd.DataFrame, pd.Series],
    synthetic_validation_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verify fit trains successfully and returns self when validation data is given."""
    X, y = synthetic_training_data
    X_val, y_val = synthetic_validation_data
    model = MLPClassifier(**FAST_TEST_KWARGS)

    result = model.fit(X, y, X_val=X_val, y_val=y_val)

    assert result is model
    assert model.network is not None


@patch("ml.models.mlp.mlflow.log_metric")
@patch("ml.models.mlp.mlflow.active_run")
def test_fit_logs_validation_loss_when_validation_data_is_given(
    mock_active_run,
    mock_log_metric,
    synthetic_training_data: tuple[pd.DataFrame, pd.Series],
    synthetic_validation_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verify mlflow.log_metric logs val_loss per epoch."""
    mock_active_run.return_value = MagicMock()

    X, y = synthetic_training_data
    X_val, y_val = synthetic_validation_data
    model = MLPClassifier(**FAST_TEST_KWARGS)

    model.fit(X, y, X_val=X_val, y_val=y_val)

    val_loss_calls = [
        call for call in mock_log_metric.call_args_list if call.args[0] == "val_loss"
    ]
    assert len(val_loss_calls) == FAST_TEST_KWARGS["epochs"]


def test_predict_proba_happy_path(
    synthetic_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Test predict_proba returns correctly shaped probabilities after fitting."""
    X, y = synthetic_training_data
    model = MLPClassifier(**FAST_TEST_KWARGS)
    model.fit(X, y)

    X_test = pd.DataFrame(
        {"feature_1": [21, 22], "feature_2": [10.5, 11.0], "feature_3": [1, 0]}
    )
    probabilities = model.predict_proba(X_test)

    assert probabilities.shape == (2, 2)
    assert (probabilities >= 0).all()
    assert (probabilities <= 1).all()


def test_predict_proba_raises_if_called_before_fit() -> None:
    """Verify predict_proba raises error instead of failing on a None network."""
    model = MLPClassifier(**FAST_TEST_KWARGS)
    X = pd.DataFrame(
        {"feature_1": [1, 2], "feature_2": [0.5, 1.0], "feature_3": [1, 0]}
    )

    with pytest.raises(
        RuntimeError, match="MLPClassifier must be fitted before calling predict_proba."
    ):
        model.predict_proba(X)


def test_build_mlp_model_returns_independent_instances() -> None:
    """Verify each call to build_mlp_model returns a fresh, independent instance."""
    model_1 = build_mlp_model(**FAST_TEST_KWARGS)
    model_2 = build_mlp_model(**FAST_TEST_KWARGS)

    assert model_1 is not model_2
    assert model_1.network is None
    assert model_2.network is None


@patch("ml.models.mlp.mlflow.log_metric")
@patch("ml.models.mlp.mlflow.active_run")
def test_fit_logs_metrics_when_run_is_active(
    mock_active_run,
    mock_log_metric,
    synthetic_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verify log_metric is called once per epoch when an MLflow run is active."""
    mock_active_run.return_value = MagicMock()

    X, y = synthetic_training_data
    model = MLPClassifier(**FAST_TEST_KWARGS)

    model.fit(X, y)

    assert mock_log_metric.call_count == FAST_TEST_KWARGS["epochs"]


@patch("ml.models.mlp.mlflow.log_metric")
@patch("ml.models.mlp.mlflow.active_run")
def test_fit_skips_logging_when_no_run_is_active(
    mock_active_run,
    mock_log_metric,
    synthetic_training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verify mlflow.log_metric is never called when no MLflow run is active."""
    mock_active_run.return_value = None

    X, y = synthetic_training_data
    model = MLPClassifier(**FAST_TEST_KWARGS)

    model.fit(X, y)

    mock_log_metric.assert_not_called()
