"""Tests for the script orchestrating the models training.

Mock every underlying data/model dependency to verify only the orchestration
logic: that prepare_training_data wires its pieces together correctly, and
that main() correctly selects which model families to train.

"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.dataset.split import Fold
from ml.run_training import FinalSplit, ModelConfig, main, prepare_training_data


@pytest.fixture
def synthetic_folds() -> list[Fold]:
    """Provide 2 minimal synthetic folds for orchestration testing.

    Returns:
        list[Fold]: Two folds, each with a single feature column.

    """
    return [
        Fold(
            X_train=pd.DataFrame({"feature": [1, 2]}),
            y_train=pd.Series([0, 1]),
            X_val=pd.DataFrame({"feature": [3]}),
            y_val=pd.Series([0]),
        )
        for _ in range(2)
    ]


@pytest.fixture
def fake_model_configs() -> list[ModelConfig]:
    """Provide 2 fake ModelConfig entries: one to select, one to skip.

    Returns:
        list[ModelConfig]: A "baseline" config and an "mock_model" config, each
            with a MagicMock builder so no real model is ever built.

    """
    return [
        ModelConfig(
            experiment_name="baseline",
            model_builder=MagicMock(),
            model_params={},
            scale=True,
            handle_nan=True,
        ),
        ModelConfig(
            experiment_name="mock_model",
            model_builder=MagicMock(),
            model_params={},
            scale=False,
            handle_nan=True,
        ),
        ModelConfig(
            experiment_name="mock_xgboost",
            model_builder=MagicMock(),
            model_params={},
            scale=False,
            handle_nan=False,
        ),
    ]


@patch("ml.run_training.preprocess_test_set")
@patch("ml.run_training.preprocess_train_folds")
@patch("ml.run_training.train_val_test_split")
@patch("ml.run_training.load_data")
@patch("ml.run_training.get_db")
def test_prepare_training_data_wires_pipeline_together(
    mock_get_db,
    mock_load_data,
    mock_train_val_test_split,
    mock_preprocess_train_folds,
    mock_preprocess_test_set,
    synthetic_folds: list[Fold],
) -> None:
    """Verify prepare_training_data passes each function's output to the next step."""
    fake_session_generator = MagicMock()
    mock_get_db.return_value = iter([fake_session_generator])

    fake_dataset = pd.DataFrame({"feature": [1, 2, 3]})
    mock_load_data.return_value = fake_dataset

    fake_df_remaining = pd.DataFrame({"feature": [1, 2]})
    fake_df_test = pd.DataFrame({"feature": [3]})
    mock_train_val_test_split.return_value = (
        synthetic_folds,
        fake_df_remaining,
        fake_df_test,
    )

    mock_preprocess_train_folds.return_value = synthetic_folds

    fake_X_train = pd.DataFrame({"feature": [1, 2]})
    fake_y_train = pd.Series([0, 1])
    fake_X_test = pd.DataFrame({"feature": [3]})
    fake_y_test = pd.Series([0])
    fake_encoder = MagicMock(spec=OneHotEncoder)
    fake_scaler = MagicMock(spec=StandardScaler)
    mock_preprocess_test_set.return_value = (
        fake_X_train,
        fake_y_train,
        fake_X_test,
        fake_y_test,
        fake_encoder,
        fake_scaler,
    )

    train_folds, final_split = prepare_training_data(scale=True)

    mock_load_data.assert_called_once_with(fake_session_generator)
    mock_train_val_test_split.assert_called_once_with(df=fake_dataset)
    mock_preprocess_train_folds.assert_called_once_with(
        folds=synthetic_folds, scale=True, handle_nan=True
    )
    mock_preprocess_test_set.assert_called_once_with(
        df_train_full=fake_df_remaining,
        df_test=fake_df_test,
        scale=True,
        handle_nan=True,
    )

    assert train_folds == synthetic_folds
    assert final_split == FinalSplit(
        fake_X_train, fake_y_train, fake_X_test, fake_y_test, fake_encoder, fake_scaler
    )


@patch("ml.run_training.MODEL_CONFIGS")
@patch("ml.run_training.train_final_model")
@patch("ml.run_training.run_cross_validation")
@patch("ml.run_training.prepare_training_data")
@patch("ml.run_training.mlflow.set_tracking_uri")
@patch("ml.run_training.setup_logging")
@patch("ml.run_training.mlflow.start_run")
def test_main_trains_only_selected_models(
    mock_start_run,
    mock_setup_logging,
    mock_set_tracking_uri,
    mock_prepare_training_data,
    mock_run_cross_validation,
    mock_train_final_model,
    mock_model_configs,
    fake_model_configs: list[ModelConfig],
) -> None:
    """Verify main() trains only the experiment listed in models_to_train."""
    # force the iterability of mock_model_configs
    mock_model_configs.__iter__.return_value = iter(fake_model_configs)
    mock_prepare_training_data.return_value = (
        [],
        FinalSplit(None, None, None, None, None, None),
    )

    main(models_to_train=["baseline"])

    assert mock_run_cross_validation.call_count == 1
    assert mock_train_final_model.call_count == 1

    called_config = mock_run_cross_validation.call_args.kwargs["experiment_name"]
    assert called_config == "baseline"


@patch("ml.run_training.MODEL_CONFIGS")
@patch("ml.run_training.train_final_model")
@patch("ml.run_training.run_cross_validation")
@patch("ml.run_training.prepare_training_data")
@patch("ml.run_training.mlflow.set_tracking_uri")
@patch("ml.run_training.setup_logging")
@patch("ml.run_training.mlflow.start_run")
def test_main_trains_all_models_when_none_selected(
    mock_start_run,
    mock_setup_logging,
    mock_set_tracking_uri,
    mock_prepare_training_data,
    mock_run_cross_validation,
    mock_train_final_model,
    mock_model_configs,
    fake_model_configs: list[ModelConfig],
) -> None:
    """Verify main() trains every configured model when models_to_train is None."""
    mock_model_configs.__iter__.return_value = iter(fake_model_configs)
    mock_prepare_training_data.return_value = (
        [],
        FinalSplit(None, None, None, None, None, None),
    )

    main(models_to_train=None)

    assert mock_run_cross_validation.call_count == len(fake_model_configs)
    assert mock_train_final_model.call_count == len(fake_model_configs)


@patch("ml.run_training.MODEL_CONFIGS")
@patch("ml.run_training.train_final_model")
@patch("ml.run_training.run_cross_validation")
@patch("ml.run_training.prepare_training_data")
@patch("ml.run_training.mlflow.set_tracking_uri")
@patch("ml.run_training.setup_logging")
@patch("ml.run_training.mlflow.start_run")
def test_main_trains_nothing_when_selection_matches_no_config(
    mock_start_run,
    mock_setup_logging,
    mock_set_tracking_uri,
    mock_prepare_training_data,
    mock_run_cross_validation,
    mock_train_final_model,
    mock_model_configs,
    fake_model_configs: list[ModelConfig],
) -> None:
    """Verify main() behaviour if models_to_train matches no configured experiment."""
    mock_model_configs.__iter__.return_value = iter(fake_model_configs)

    main(models_to_train=["fake_model"])

    mock_prepare_training_data.assert_not_called()
    mock_run_cross_validation.assert_not_called()
    mock_train_final_model.assert_not_called()
