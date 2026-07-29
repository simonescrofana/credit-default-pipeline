"""Test the model inference prediction pipeline and database retrieval.

Provide unit tests to verify single-company star schema data retrieval, SQL parameter
binding, preprocessing execution (encoding and conditional scaling), and prediction
result construction using isolated mocks.

"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy.orm import Session

from ml.inference.model_loader import LoadedModel
from ml.inference.predictor import PredictionResult, predict, retrieve_company_data


@patch("ml.inference.predictor.pd.read_sql")
def test_retrieve_company_data_returns_single_row_indexed_by_legal_name(
    mock_read_sql, db_session: Session
) -> None:
    """Verify the returned DataFrame has the expected format."""
    fake_data = pd.DataFrame(
        {
            "company_id": [42],
            "snapshot_date": [pd.Timestamp("2026-06-01")],
            "legal_name": ["Alpha Srl"],
            "industry_sector": ["manufacturing"],
        }
    )
    mock_read_sql.return_value = fake_data

    result = retrieve_company_data(session=db_session, company_id=42)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result.index.name == "legal_name"
    assert "company_id" not in result.columns
    assert "snapshot_date" not in result.columns


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
        }
    )

    retrieve_company_data(session=db_session, company_id=99)

    call_kwargs = mock_read_sql.call_args.kwargs
    assert call_kwargs["params"] == {"company_id": 99}


@patch("ml.inference.predictor.scale_features")
@patch("ml.inference.predictor.handle_missing_and_encode")
@patch("ml.inference.predictor.retrieve_company_data")
def test_predict_happy_path(
    mock_retrieve, mock_encode, mock_scale, db_session: Session
) -> None:
    """Verify predict returns prediction with the expected format."""
    fake_model = MagicMock()
    fake_model.predict_proba.return_value = np.array([[0.3, 0.7]])

    fake_encoder = MagicMock(spec=OneHotEncoder)
    fake_scaler = MagicMock(spec=StandardScaler)
    loaded_model = LoadedModel(
        model=fake_model, encoder=fake_encoder, scaler=fake_scaler
    )

    mock_retrieve.return_value = pd.DataFrame({"feature": [1]})
    mock_encode.return_value = (pd.DataFrame({"feature": [1]}), fake_encoder)
    mock_scale.return_value = (pd.DataFrame({"feature": [0.5]}), fake_scaler)

    result = predict(
        session=db_session, company_id=1, loaded_model=loaded_model, threshold=0.5
    )

    assert isinstance(result, PredictionResult)
    assert result.probability == pytest.approx(0.7)
    assert result.predicted_class == 1
    assert result.company_id == 1


@patch("ml.inference.predictor.scale_features")
@patch("ml.inference.predictor.handle_missing_and_encode")
@patch("ml.inference.predictor.retrieve_company_data")
def test_predict_skips_scaling_when_scaler_is_none(
    mock_retrieve, mock_encode, mock_scale, db_session: Session
) -> None:
    """Verify scale_features is never called when the loaded model has no scaler."""
    fake_model = MagicMock()
    fake_model.predict_proba.return_value = np.array([[0.9, 0.1]])

    fake_encoder = MagicMock(spec=OneHotEncoder)
    loaded_model = LoadedModel(model=fake_model, encoder=fake_encoder, scaler=None)

    mock_retrieve.return_value = pd.DataFrame({"feature": [1]})
    mock_encode.return_value = (pd.DataFrame({"feature": [1]}), fake_encoder)

    predict(session=db_session, company_id=1, loaded_model=loaded_model, threshold=0.5)

    mock_scale.assert_not_called()
