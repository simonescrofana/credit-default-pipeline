"""Test suite for the script loading models for inference.

Cover the happy path of loading a model with a scaler artifact present,
without one, and with an explicit run_id (skipping the search entirely).

"""

from unittest.mock import MagicMock, patch

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.inference.model_loader import LoadedModel, load_model


def _fake_artifact(path: str) -> MagicMock:
    """Build a fake MLflow FileInfo-like object with the given artifact path."""
    artifact = MagicMock()
    artifact.path = path
    return artifact


@patch("ml.inference.model_loader.MlflowClient")
@patch("ml.inference.model_loader.mlflow.sklearn.load_model")
@patch("ml.inference.model_loader.mlflow.search_runs")
@patch("ml.inference.model_loader.mlflow.get_experiment_by_name")
def test_load_model_happy_path_with_scaler(
    mock_get_experiment, mock_search_runs, mock_sklearn_load_model, mock_client_class
) -> None:
    """Verify load_model returns model, encoder, scaler when all 3 artifacts exist."""
    mock_experiment = MagicMock(experiment_id="1")
    mock_get_experiment.return_value = mock_experiment

    fake_search_result = MagicMock()
    fake_search_result.iloc = [{"run_id": "run_abc"}]
    mock_search_runs.return_value = fake_search_result

    fake_model = LogisticRegression()
    fake_encoder = OneHotEncoder()
    fake_scaler = StandardScaler()
    mock_sklearn_load_model.side_effect = [fake_model, fake_encoder, fake_scaler]

    mock_client = MagicMock()
    mock_client.list_artifacts.return_value = [
        _fake_artifact("model"),
        _fake_artifact("encoder"),
        _fake_artifact("scaler"),
    ]
    mock_client_class.return_value = mock_client

    result = load_model(experiment_name="baseline")

    assert isinstance(result, LoadedModel)
    assert isinstance(result.model, LogisticRegression)
    assert isinstance(result.encoder, OneHotEncoder)
    assert isinstance(result.scaler, StandardScaler)

    mock_sklearn_load_model.assert_any_call("runs:/run_abc/model")
    mock_sklearn_load_model.assert_any_call("runs:/run_abc/encoder")
    mock_sklearn_load_model.assert_any_call("runs:/run_abc/scaler")


@patch("ml.inference.model_loader.MlflowClient")
@patch("ml.inference.model_loader.mlflow.sklearn.load_model")
@patch("ml.inference.model_loader.mlflow.search_runs")
@patch("ml.inference.model_loader.mlflow.get_experiment_by_name")
def test_load_model_happy_path_without_scaler(
    mock_get_experiment, mock_search_runs, mock_sklearn_load_model, mock_client_class
) -> None:
    """Verify load_model returns scaler=None when no scaler artifact exists."""
    mock_experiment = MagicMock(experiment_id="2")
    mock_get_experiment.return_value = mock_experiment

    fake_search_result = MagicMock()
    fake_search_result.iloc = [{"run_id": "run_xyz"}]
    mock_search_runs.return_value = fake_search_result

    fake_model = LogisticRegression()
    fake_encoder = OneHotEncoder()
    mock_sklearn_load_model.side_effect = [fake_model, fake_encoder]

    mock_client = MagicMock()
    mock_client.list_artifacts.return_value = [
        _fake_artifact("model"),
        _fake_artifact("encoder"),
    ]
    mock_client_class.return_value = mock_client

    result = load_model(experiment_name="xgboost")

    assert isinstance(result, LoadedModel)
    assert isinstance(result.model, LogisticRegression)
    assert isinstance(result.encoder, OneHotEncoder)
    assert result.scaler is None

    # scaler was never requested, since it wasn't in the artifact list
    for call in mock_sklearn_load_model.call_args_list:
        assert "scaler" not in call.args[0]


@patch("ml.inference.model_loader.MlflowClient")
@patch("ml.inference.model_loader.mlflow.sklearn.load_model")
@patch("ml.inference.model_loader.mlflow.search_runs")
@patch("ml.inference.model_loader.mlflow.get_experiment_by_name")
def test_load_model_with_explicit_run_id_skips_search(
    mock_get_experiment, mock_search_runs, mock_sklearn_load_model, mock_client_class
) -> None:
    """Verify passing run_id explicitly skips the experiment/run search entirely."""
    fake_model = LogisticRegression()
    fake_encoder = OneHotEncoder()
    fake_scaler = StandardScaler()
    mock_sklearn_load_model.side_effect = [fake_model, fake_encoder, fake_scaler]

    mock_client = MagicMock()
    mock_client.list_artifacts.return_value = [
        _fake_artifact("model"),
        _fake_artifact("encoder"),
        _fake_artifact("scaler"),
    ]
    mock_client_class.return_value = mock_client

    result = load_model(experiment_name="baseline", run_id="run_pinned")

    assert isinstance(result, LoadedModel)
    mock_get_experiment.assert_not_called()
    mock_search_runs.assert_not_called()

    mock_sklearn_load_model.assert_any_call("runs:/run_pinned/model")
    mock_sklearn_load_model.assert_any_call("runs:/run_pinned/encoder")
    mock_sklearn_load_model.assert_any_call("runs:/run_pinned/scaler")
