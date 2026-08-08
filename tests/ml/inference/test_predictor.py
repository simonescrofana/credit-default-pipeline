"""Test the model inference prediction pipeline and database retrieval.

Provide unit tests to verify single-company star schema data retrieval, SQL
parameter binding, the shared scoring helper's usage from both entry points
(`predict` for case A, `predict_from_raw_data` for case B), and prediction
result construction using isolated mocks.

"""

import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import Session

from ml.inference.model_loader import LoadedModel
from ml.inference.predictor import (
    PredictionResult,
    predict,
    predict_from_raw_data,
    retrieve_company_data,
    score_features,
)
from schemas.ml.insolvency_prediction import InsolvencyPredictionRequest


@patch("ml.inference.predictor.pd.read_sql")
def test_retrieve_company_data_returns_single_row_indexed_by_legal_name(
    mock_read_sql, db_session: Session
) -> None:
    """Verify the returned DataFrame has the expected format."""
    fake_data = pd.DataFrame(
        {
            "company_id": [1],
            "snapshot_date": [pd.Timestamp("2026-06-01")],
            "legal_name": ["Alpha Srl"],
            "industry_sector": ["manufacturing"],
            "is_insolvent": [0],
        }
    )
    mock_read_sql.return_value = fake_data

    result = retrieve_company_data(session=db_session, company_id=1)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result.index.name == "legal_name"
    assert "company_id" not in result.columns
    assert "snapshot_date" not in result.columns
    assert "is_insolvent" not in result.columns


@patch("ml.inference.predictor.pd.read_sql")
def test_retrieve_company_data_passes_company_id_as_param(
    mock_read_sql, db_session: Session
) -> None:
    """Verify company_id is passed as a query parameter, never string-interpolated."""
    mock_read_sql.return_value = pd.DataFrame(
        {
            "company_id": [99],
            "snapshot_date": [pd.Timestamp("2026-01-01")],
            "legal_name": ["X"],
            "is_insolvent": [1],
        }
    )

    retrieve_company_data(session=db_session, company_id=99)

    call_kwargs = mock_read_sql.call_args.kwargs
    assert call_kwargs["params"] == {"company_id": 99}


@patch("ml.inference.predictor.score_features")
@patch("ml.inference.predictor.retrieve_company_data")
def test_predict_happy_path(mock_retrieve, mock_score, db_session: Session) -> None:
    """Verify predict returns a PredictionResult for a specific stored company."""
    fake_model = MagicMock()
    loaded_model = LoadedModel(
        model=fake_model,
        encoder=MagicMock(),
        scaler=MagicMock(),
        explainer=MagicMock(),
        threshold=0.5,
    )

    retrieved_df = pd.DataFrame(
        {"feature": [1]}, index=pd.Index(["Alpha Srl"], name="legal_name")
    )
    mock_retrieve.return_value = retrieved_df
    fake_explanation = {"feature": {"value": 1, "shap": 0.2}}
    mock_score.return_value = (0.7, 1, fake_explanation)

    result = predict(session=db_session, company_id=1, loaded_model=loaded_model)

    assert isinstance(result, PredictionResult)
    assert result.company_id == 1
    assert result.company_name == "Alpha Srl"
    assert result.probability == pytest.approx(0.7)
    assert result.predicted_class == 1
    assert result.explanation == fake_explanation

    mock_retrieve.assert_called_once_with(db_session, 1)
    called_args = mock_score.call_args.args
    assert called_args[0].equals(retrieved_df)
    assert called_args[1] is loaded_model


@patch("ml.inference.predictor.score_features")
@patch("ml.inference.predictor.retrieve_company_data")
def test_predict_from_raw_data_happy_path(mock_retrieve, mock_score) -> None:
    """Verify predictor returns a PredictionResult for requested company."""
    fake_model = MagicMock()
    loaded_model = LoadedModel(
        model=fake_model,
        encoder=MagicMock(),
        scaler=None,
        explainer=MagicMock(),
        threshold=0.5,
    )

    request = InsolvencyPredictionRequest(
        foundation_date=datetime.date(2015, 3, 1),
        industry_sector="manufacturing",
        unpaid_ratio_trailing_90d=0.3,
        total_outstanding_debt=15000.0,
    )

    fake_explanation = {"unpaid_ratio_trailing_90d": {"value": 0.3, "shap": 0.5}}
    mock_score.return_value = (0.6, 1, fake_explanation)

    result = predict_from_raw_data(request=request, loaded_model=loaded_model)

    assert isinstance(result, PredictionResult)
    assert result.company_id is None
    assert result.company_name is None
    assert result.probability == pytest.approx(0.6)
    assert result.predicted_class == 1
    assert result.explanation == fake_explanation

    # never touches the database: this is the whole point of case B
    mock_retrieve.assert_not_called()

    called_args = mock_score.call_args.args
    passed_df = called_args[0]
    assert len(passed_df) == 1
    assert passed_df.iloc[0]["unpaid_ratio_trailing_90d"] == 0.3
    assert passed_df.iloc[0]["total_outstanding_debt"] == 15000.0
    assert passed_df.iloc[0]["industry_sector"] == "manufacturing"
    assert called_args[1] is loaded_model


@patch("ml.inference.predictor.score_features")
def test_predict_from_raw_data_casts_numeric_columns_to_float64(mock_score) -> None:
    """Verify every column except categorical ones is cast to float64 before scoring."""
    loaded_model = LoadedModel(
        model=MagicMock(),
        encoder=MagicMock(),
        scaler=None,
        explainer=MagicMock(),
        threshold=0.5,
    )

    request = InsolvencyPredictionRequest(
        foundation_date=datetime.date(2015, 3, 1),
        industry_sector="manufacturing",
        unpaid_ratio_trailing_90d=0.3,
        total_outstanding_debt=15000.0,
    )

    mock_score.return_value = (0.5, 0, {})

    predict_from_raw_data(request=request, loaded_model=loaded_model)

    passed_df = mock_score.call_args.args[0]

    categorical_cols = {"industry_sector", "registered_office_region"}
    for col in passed_df.columns:
        if col in categorical_cols:
            continue
        assert passed_df[col].dtype == "float64", (
            f"column '{col}' has dtype {passed_df[col].dtype}, expected float64"
        )


@patch("ml.inference.predictor.score_features")
def test_predict_from_raw_data_derives_year_quarter_month(mock_score) -> None:
    """Verify year, quarter, and month are derived server-side, not left missing."""
    loaded_model = LoadedModel(
        model=MagicMock(),
        encoder=MagicMock(),
        scaler=None,
        explainer=MagicMock(),
        threshold=0.5,
    )
 
    request = InsolvencyPredictionRequest(
        foundation_date=datetime.date(2015, 3, 1),
        industry_sector="manufacturing",
        unpaid_ratio_trailing_90d=0.3,
        total_outstanding_debt=15000.0,
    )
 
    mock_score.return_value = (0.5, 0, {})
 
    predict_from_raw_data(request=request, loaded_model=loaded_model)
 
    passed_df = mock_score.call_args.args[0]
 
    now = datetime.datetime.now()
    expected_quarter = (now.month - 1) // 3 + 1
 
    assert passed_df.iloc[0]["year"] == float(now.year)
    assert passed_df.iloc[0]["quarter"] == float(expected_quarter)
    assert passed_df.iloc[0]["month"] == float(now.month)


@patch("ml.inference.predictor.explain_prediction")
@patch("ml.inference.predictor.scale_features")
@patch("ml.inference.predictor.handle_missing_and_encode")
def test_score_features_applies_scaling_when_scaler_is_present(
    mock_encode, mock_scale, mock_explain
) -> None:
    """Verify score_features scales the encoded features when scaler is present."""
    fake_model = MagicMock()
    fake_model.predict_proba.return_value = np.array([[0.4, 0.6]])
    fake_model.feature_names_in_ = ["feature"]
    fake_scaler = MagicMock()
    fake_explainer = MagicMock()
    loaded_model = LoadedModel(
        model=fake_model,
        encoder=MagicMock(),
        scaler=fake_scaler,
        explainer=fake_explainer,
        threshold=0.5,
    )

    encoded_df = pd.DataFrame({"feature": [1]})
    scaled_df = pd.DataFrame({"feature": [0.5]})
    mock_encode.return_value = (encoded_df, MagicMock())
    mock_scale.return_value = (scaled_df, fake_scaler)
    fake_explanation = {"feature": {"value": 0.5, "shap": 0.3}}
    mock_explain.return_value = fake_explanation

    raw_features = pd.DataFrame({"feature": [1]})
    probability, predicted_class, explanation = score_features(
        raw_features, loaded_model
    )

    assert probability == pytest.approx(0.6)
    assert predicted_class == 1
    assert explanation == fake_explanation

    mock_scale.assert_called_once_with(encoded_df, scaler=fake_scaler)

    predict_proba_arg = fake_model.predict_proba.call_args.args[0]
    assert predict_proba_arg.equals(scaled_df)
    explain_arg = mock_explain.call_args.args[1]
    assert explain_arg.equals(scaled_df)


@patch("ml.inference.predictor.explain_prediction")
@patch("ml.inference.predictor.scale_features")
@patch("ml.inference.predictor.handle_missing_and_encode")
def test_score_features_skips_scaling_when_scaler_is_none(
    mock_encode, mock_scale, mock_explain
) -> None:
    """Verify score_features never calls scale_features when no scaler is passed."""
    fake_model = MagicMock()
    fake_model.predict_proba.return_value = np.array([[0.9, 0.1]])
    fake_model.feature_names_in_ = ["feature"]
    fake_explainer = MagicMock()
    loaded_model = LoadedModel(
        model=fake_model,
        encoder=MagicMock(),
        scaler=None,
        explainer=fake_explainer,
        threshold=0.5,
    )

    encoded_df = pd.DataFrame({"feature": [1]})
    mock_encode.return_value = (encoded_df, MagicMock())
    fake_explanation = {"feature": {"value": 1, "shap": -0.1}}
    mock_explain.return_value = fake_explanation

    raw_features = pd.DataFrame({"feature": [1]})
    probability, predicted_class, explanation = score_features(
        raw_features,
        loaded_model,
    )

    assert probability == pytest.approx(0.1)
    assert predicted_class == 0
    assert explanation == fake_explanation

    mock_scale.assert_not_called()

    predict_proba_arg = fake_model.predict_proba.call_args.args[0]
    assert predict_proba_arg.equals(encoded_df)
    explain_arg = mock_explain.call_args.args[1]
    assert explain_arg.equals(encoded_df)


@patch("ml.inference.predictor.explain_prediction")
@patch("ml.inference.predictor.scale_features")
@patch("ml.inference.predictor.handle_missing_and_encode")
def test_score_features_reorders_columns_to_match_model_training_order(
    mock_encode, mock_scale, mock_explain
) -> None:
    """Verify columns are reordered to loaded_model.model.feature_names_in_."""
    fake_model = MagicMock()
    fake_model.predict_proba.return_value = np.array([[0.7, 0.3]])
    # training order differs from the order encoded_df happens to have below
    fake_model.feature_names_in_ = ["b", "a", "c"]
    loaded_model = LoadedModel(
        model=fake_model,
        encoder=MagicMock(),
        scaler=None,
        explainer=MagicMock(),
        threshold=0.5,
    )
 
    # encoded_df's own column order is deliberately different from training order
    encoded_df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    mock_encode.return_value = (encoded_df, MagicMock())
    mock_explain.return_value = {}
 
    score_features(pd.DataFrame({"a": [1], "b": [2], "c": [3]}), loaded_model)
 
    predict_proba_arg = fake_model.predict_proba.call_args.args[0]
    assert list(predict_proba_arg.columns) == ["b", "a", "c"]


@patch("ml.inference.predictor.explain_prediction")
@patch("ml.inference.predictor.scale_features")
@patch("ml.inference.predictor.handle_missing_and_encode")
def test_score_features_applies_threshold_to_predicted_class(
    mock_encode, mock_scale, mock_explain
) -> None:
    """Test predicted_class is derived by the probability with threshold."""
    fake_model = MagicMock()
    fake_model.predict_proba.return_value = np.array([[0.7, 0.3]])
    fake_model.feature_names_in_ = ["feature"]

    loaded_model_high_threshold = LoadedModel(
        model=fake_model,
        encoder=MagicMock(),
        scaler=None,
        explainer=MagicMock(),
        threshold=0.5,
    )
    loaded_model_low_threshold = LoadedModel(
        model=fake_model,
        encoder=MagicMock(),
        scaler=None,
        explainer=MagicMock(),
        threshold=0.2,
    )

    mock_encode.return_value = (pd.DataFrame({"feature": [1]}), MagicMock())
    mock_explain.return_value = {}

    _, predicted_class_below, _ = score_features(
        pd.DataFrame({"feature": [1]}), loaded_model_high_threshold
    )
    assert predicted_class_below == 0

    _, predicted_class_above, _ = score_features(
        pd.DataFrame({"feature": [1]}), loaded_model_low_threshold
    )
    assert predicted_class_above == 1
