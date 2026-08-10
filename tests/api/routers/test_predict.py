"""Test suite for the /predict/* endpoints.

Covers the happy path for both `POST /predict/ad-hoc` and
`POST /predict/company`, plus the 404 raised by `POST /predict/company`
when the requested identifier does not resolve to any company. Dependency
injection is bypassed via `app.dependency_overrides`; the internal
`ml.inference.predictor` and `pandas.read_sql` calls are mocked, so no real
model or database is needed.

"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.dependencies import get_loaded_model
from api.routers.predict import router
from database.connection import get_db
from ml.inference.predictor import PredictionResult

AD_HOC_PAYLOAD = {
    "unpaid_ratio_trailing_90d": 0.1,
    "total_outstanding_debt": 1000.0,
    "foundation_date": "2020-01-01",
    "industry_sector": "manufacturing",
    "registered_office_region": "Lazio",
}


@pytest.fixture
def client() -> TestClient:
    """Build a TestClient with get_db and get_loaded_model overridden.

    Returns:
        TestClient: A client for the isolated `predict` router, with its
            model and database dependencies replaced by mocks so no real
            model or database connection is required.

    """
    app = FastAPI()
    app.include_router(router)

    def fake_get_db():
        yield MagicMock()

    app.dependency_overrides[get_loaded_model] = lambda: MagicMock()
    app.dependency_overrides[get_db] = fake_get_db

    return TestClient(app)


@patch("api.routers.predict.predict_from_raw_data")
def test_predict_ad_hoc_returns_prediction(
    mock_predict_from_raw_data,
    client: TestClient,
) -> None:
    """Test POST /predict/ad-hoc returns the scored prediction."""
    mock_predict_from_raw_data.return_value = PredictionResult(
        company_id=None,
        company_name=None,
        probability=0.3,
        predicted_class=0,
        explanation={"total_outstanding_debt": 0.1},
    )

    response = client.post("/predict/ad-hoc", json=AD_HOC_PAYLOAD)

    assert response.status_code == 200
    assert response.json() == {
        "company_name": None,
        "probability": 0.3,
        "predicted_class": 0,
        "explanation": {"total_outstanding_debt": 0.1},
    }


@patch("api.routers.predict.predict")
@patch("api.routers.predict.pd.read_sql")
def test_predict_company_returns_prediction(
    mock_read_sql,
    mock_predict,
    client: TestClient,
) -> None:
    """Test POST /predict/company returns scored prediction for a resolved company."""
    mock_read_sql.return_value = pd.DataFrame({"company_id": [1]})
    mock_predict.return_value = PredictionResult(
        company_id=1,
        company_name="Rossi S.r.l.",
        probability=0.7,
        predicted_class=1,
        explanation={"leverage_ratio": 0.4},
    )

    response = client.post("/predict/company", json={"identifier": "Rossi S.r.l."})

    assert response.status_code == 200
    assert response.json() == {
        "company_name": "Rossi S.r.l.",
        "probability": 0.7,
        "predicted_class": 1,
        "explanation": {"leverage_ratio": 0.4},
    }
    assert mock_predict.call_args.args[1] == 1


@patch("api.routers.predict.pd.read_sql")
def test_predict_company_raises_404_when_identifier_not_found(
    mock_read_sql: MagicMock,
    client: TestClient,
) -> None:
    """Test POST /predict/company returns 404 when no company matches the identifier."""
    mock_read_sql.return_value = pd.DataFrame({"company_id": []})

    response = client.post("/predict/company", json={"identifier": "Ghost Srl"})

    assert response.status_code == 404
    assert response.json() == {"detail": "Company 'Ghost Srl' not found."}
