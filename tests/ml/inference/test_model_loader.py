"""Test suite for the script loading models for inference.

Cover the happy path of loading a model with a scaler artifact present,
without one, and with an explicit run_id (skipping the search entirely).

"""

from unittest.mock import MagicMock, patch

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from ml.inference.model_loader import LoadedModel, load_model


def _fake_artifact(path: str) -> MagicMock:
    """Build a fake MLflow FileInfo-like object with the given artifact path."""
    artifact = MagicMock()
    artifact.path = path
    return artifact


@patch("ml.inference.model_loader.build_explainer")
@patch("ml.inference.model_loader.MlflowClient")
@patch("ml.inference.model_loader.mlflow.sklearn.load_model")
@patch("ml.inference.model_loader.mlflow.xgboost.load_model")
@patch("ml.inference.model_loader.mlflow.search_runs")
@patch("ml.inference.model_loader.mlflow.get_experiment_by_name")
def test_load_model_happy_path_with_scaler(
    mock_get_experiment,
    mock_search_runs,
    mock_xgboost_load_model,
    mock_sklearn_load_model,
    mock_client_class,
    mock_build_explainer,
) -> None:
    """Verify load_model returns model, encoder, scaler and explainer."""
    mock_experiment = MagicMock(experiment_id="1")
    mock_get_experiment.return_value = mock_experiment

    fake_search_result = MagicMock()
    fake_search_result.iloc = [{"run_id": "run_abc"}]
    mock_search_runs.return_value = fake_search_result

    fake_model = MagicMock(spec=XGBClassifier)
    mock_xgboost_load_model.return_value = fake_model

    fake_encoder = OneHotEncoder()
    fake_scaler = StandardScaler()
    mock_sklearn_load_model.side_effect = [fake_encoder, fake_scaler]

    fake_explainer = MagicMock()
    mock_build_explainer.return_value = fake_explainer

    mock_client = MagicMock()
    mock_client.list_artifacts.return_value = [
        _fake_artifact("model"),
        _fake_artifact("encoder"),
        _fake_artifact("scaler"),
    ]
    mock_client_class.return_value = mock_client

    result = load_model(experiment_name="xgboost_model")

    assert isinstance(result, LoadedModel)
    assert result.model is fake_model
    assert isinstance(result.encoder, OneHotEncoder)
    assert isinstance(result.scaler, StandardScaler)
    assert result.explainer is fake_explainer

    mock_xgboost_load_model.assert_called_once_with("runs:/run_abc/model")
    mock_sklearn_load_model.assert_any_call("runs:/run_abc/encoder")
    mock_sklearn_load_model.assert_any_call("runs:/run_abc/scaler")
    mock_build_explainer.assert_called_once_with(fake_model)


@patch("ml.inference.model_loader.build_explainer")
@patch("ml.inference.model_loader.MlflowClient")
@patch("ml.inference.model_loader.mlflow.sklearn.load_model")
@patch("ml.inference.model_loader.mlflow.xgboost.load_model")
@patch("ml.inference.model_loader.mlflow.search_runs")
@patch("ml.inference.model_loader.mlflow.get_experiment_by_name")
def test_load_model_happy_path_without_scaler(
    mock_get_experiment,
    mock_search_runs,
    mock_xgboost_load_model,
    mock_sklearn_load_model,
    mock_client_class,
    mock_build_explainer,
) -> None:
    """Verify load_model returns scaler=None when no scaler artifact exists."""
    mock_experiment = MagicMock(experiment_id="2")
    mock_get_experiment.return_value = mock_experiment

    fake_search_result = MagicMock()
    fake_search_result.iloc = [{"run_id": "run_xyz"}]
    mock_search_runs.return_value = fake_search_result

    fake_model = MagicMock(spec=XGBClassifier)
    mock_xgboost_load_model.return_value = fake_model

    fake_encoder = OneHotEncoder()
    mock_sklearn_load_model.side_effect = [fake_encoder]

    mock_client = MagicMock()
    mock_client.list_artifacts.return_value = [
        _fake_artifact("model"),
        _fake_artifact("encoder"),
    ]
    mock_client_class.return_value = mock_client

    result = load_model(experiment_name="xgboost_model")

    assert isinstance(result, LoadedModel)
    assert result.model is fake_model
    assert isinstance(result.encoder, OneHotEncoder)
    assert result.scaler is None

    # scaler was never requested, since it wasn't in the artifact list
    for call in mock_sklearn_load_model.call_args_list:
        assert "scaler" not in call.args[0]


@patch("ml.inference.model_loader.build_explainer")
@patch("ml.inference.model_loader.MlflowClient")
@patch("ml.inference.model_loader.mlflow.sklearn.load_model")
@patch("ml.inference.model_loader.mlflow.xgboost.load_model")
@patch("ml.inference.model_loader.mlflow.search_runs")
@patch("ml.inference.model_loader.mlflow.get_experiment_by_name")
def test_load_model_with_explicit_run_id_skips_search(
    mock_get_experiment,
    mock_search_runs,
    mock_xgboost_load_model,
    mock_sklearn_load_model,
    mock_client_class,
    mock_build_explainer,
) -> None:
    """Verify passing run_id explicitly skips the experiment/run search entirely."""
    fake_model = MagicMock(spec=XGBClassifier)
    mock_xgboost_load_model.return_value = fake_model

    fake_encoder = OneHotEncoder()
    fake_scaler = StandardScaler()
    mock_sklearn_load_model.side_effect = [fake_encoder, fake_scaler]

    mock_client = MagicMock()
    mock_client.list_artifacts.return_value = [
        _fake_artifact("model"),
        _fake_artifact("encoder"),
        _fake_artifact("scaler"),
    ]
    mock_client_class.return_value = mock_client

    result = load_model(experiment_name="xgboost_model", run_id="run_pinned")

    assert isinstance(result, LoadedModel)
    mock_get_experiment.assert_not_called()
    mock_search_runs.assert_not_called()

    mock_xgboost_load_model.assert_called_once_with("runs:/run_pinned/model")
    mock_sklearn_load_model.assert_any_call("runs:/run_pinned/encoder")
    mock_sklearn_load_model.assert_any_call("runs:/run_pinned/scaler")


@patch("ml.inference.model_loader.build_explainer")
@patch("ml.inference.model_loader.MlflowClient")
@patch("ml.inference.model_loader.mlflow.sklearn.load_model")
@patch("ml.inference.model_loader.mlflow.xgboost.load_model")
@patch("ml.inference.model_loader.mlflow.search_runs")
@patch("ml.inference.model_loader.mlflow.get_experiment_by_name")
def test_load_model_uses_default_experiment_name(
    mock_get_experiment,
    mock_search_runs,
    mock_xgboost_load_model,
    mock_sklearn_load_model,
    mock_client_class,
    mock_build_explainer,
) -> None:
    """Verify the default experiment_name is "xgboost_model" when omitted."""
    mock_get_experiment.return_value = MagicMock(experiment_id="3")

    fake_search_result = MagicMock()
    fake_search_result.iloc = [{"run_id": "run_default"}]
    mock_search_runs.return_value = fake_search_result

    fake_model = MagicMock(spec=XGBClassifier)
    mock_xgboost_load_model.return_value = fake_model
    mock_sklearn_load_model.side_effect = [OneHotEncoder()]

    mock_build_explainer.return_value = MagicMock()

    mock_client = MagicMock()
    mock_client.list_artifacts.return_value = [
        _fake_artifact("model"),
        _fake_artifact("encoder"),
    ]
    mock_client_class.return_value = mock_client

    load_model()

    mock_get_experiment.assert_called_once_with("xgboost_model")
