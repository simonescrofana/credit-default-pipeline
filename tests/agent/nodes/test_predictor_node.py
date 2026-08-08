"""Test suite for the predictor node.

Covers `predict_case_a` (multi-company happy path, a single company failing
with a database error without stopping the others), `predict_case_b`
(happy path and the defensive ValidationError guard), and `predictor_node`
itself as a dispatcher, mocking `get_loaded_model` and the underlying
`predict`/`predict_from_raw_data` calls throughout so no real model or
database is ever touched.

"""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import SQLAlchemyError

from agent.nodes.predictor_node import predict_case_a, predict_case_b, predictor_node
from agent.state import AgentState
from ml.inference.predictor import PredictionResult


def fake_get_db(session):
    """Build a generator function mimicking get_db(), yielding the given session."""

    def gen():
        yield session

    return gen


@patch("agent.nodes.predictor_node.get_db")
@patch("agent.nodes.predictor_node.predict")
def test_predict_case_a_happy_path_multiple_companies(
    mock_predict, mock_get_db
) -> None:
    """Verify predict_case_a scores every resolved company_id."""
    fake_session = MagicMock()
    mock_get_db.return_value = fake_get_db(fake_session)()

    mock_predict.side_effect = [
        PredictionResult(
            company_id=1,
            company_name="Company 1",
            probability=0.7,
            predicted_class=1,
            explanation={},
        ),
        PredictionResult(
            company_id=2,
            company_name="Company 2",
            probability=0.1,
            predicted_class=0,
            explanation={},
        ),
    ]

    state = AgentState(user_input="...", resolved_company_ids=[1, 2])
    loaded_model = MagicMock()
    result = predict_case_a(state, loaded_model)

    assert len(result["prediction_results"]) == 2
    assert result["prediction_results"][0]["company_id"] == 1
    assert result["prediction_results"][1]["company_id"] == 2
    assert result["prediction_errors"] == []
    assert mock_predict.call_count == 2


@patch("agent.nodes.predictor_node.get_db")
@patch("agent.nodes.predictor_node.predict")
def test_predict_case_a_one_failure_does_not_stop_the_others(
    mock_predict, mock_get_db
) -> None:
    """Verify a SQLAlchemyError for one company is recorded, not propagated."""
    fake_session = MagicMock()
    mock_get_db.return_value = fake_get_db(fake_session)()

    mock_predict.side_effect = [
        PredictionResult(
            company_id=1,
            company_name="Company 1",
            probability=0.7,
            predicted_class=1,
            explanation={},
        ),
        SQLAlchemyError("connection lost"),
        PredictionResult(
            company_id=3,
            company_name="Company 3",
            probability=0.2,
            predicted_class=0,
            explanation={},
        ),
    ]

    state = AgentState(user_input="...", resolved_company_ids=[1, 2, 3])
    loaded_model = MagicMock()
    result = predict_case_a(state, loaded_model)

    assert len(result["prediction_results"]) == 2
    assert [r["company_id"] for r in result["prediction_results"]] == [1, 3]
    assert len(result["prediction_errors"]) == 1
    assert "company_id=2" in result["prediction_errors"][0]


@patch("agent.nodes.predictor_node.get_db")
@patch("agent.nodes.predictor_node.predict")
def test_predict_case_a_closes_session_even_on_failure(
    mock_predict, mock_get_db
) -> None:
    """Verify the session generator is closed even if every prediction fails."""
    fake_session = MagicMock()
    fake_gen = fake_get_db(fake_session)()
    mock_get_db.return_value = fake_gen

    mock_predict.side_effect = SQLAlchemyError("connection lost")

    state = AgentState(user_input="...", resolved_company_ids=[1])
    loaded_model = MagicMock()
    result = predict_case_a(state, loaded_model)

    assert result["prediction_results"] == []
    assert len(result["prediction_errors"]) == 1
    with pytest.raises(StopIteration):
        next(fake_gen)


@patch("agent.nodes.predictor_node.predict_from_raw_data")
def test_predict_case_b_happy_path(mock_predict_from_raw_data) -> None:
    """Verify predict_case_b validates raw_prediction_input and scores it."""
    mock_predict_from_raw_data.return_value = PredictionResult(
        company_id=None,
        company_name=None,
        probability=0.6,
        predicted_class=1,
        explanation={},
    )

    state = AgentState(
        user_input="...",
        raw_prediction_input={
            "foundation_date": "2015-03-01",
            "industry_sector": "manufacturing",
            "unpaid_ratio_trailing_90d": 0.3,
            "total_outstanding_debt": 15000.0,
        },
    )
    loaded_model = MagicMock()
    result = predict_case_b(state, loaded_model)

    assert len(result["prediction_results"]) == 1
    assert result["prediction_results"][0]["probability"] == 0.6
    assert result["prediction_errors"] == []


@patch("agent.nodes.predictor_node.predict_from_raw_data")
def test_predict_case_b_invalid_data_is_guarded(mock_predict_from_raw_data) -> None:
    """Verify predict_case_b does not raise if raw_prediction_input is invalid."""
    state = AgentState(user_input="...", raw_prediction_input={})

    loaded_model = MagicMock()
    result = predict_case_b(state, loaded_model)

    assert result["prediction_results"] == []
    mock_predict_from_raw_data.assert_not_called()


@patch("agent.nodes.predictor_node.get_loaded_model")
@patch("agent.nodes.predictor_node.predict_case_a")
def test_predictor_node_dispatches_case_a(
    mock_predict_case_a, mock_get_loaded_model
) -> None:
    """Verify predictor_node calls predict_case_a when route is case_a."""
    mock_get_loaded_model.return_value = MagicMock()
    mock_predict_case_a.return_value = {
        "prediction_results": [],
        "prediction_errors": [],
    }

    state = AgentState(user_input="...", route="case_a")
    predictor_node(state)

    mock_predict_case_a.assert_called_once()


@patch("agent.nodes.predictor_node.get_loaded_model")
@patch("agent.nodes.predictor_node.predict_case_b")
def test_predictor_node_dispatches_case_b(
    mock_predict_case_b, mock_get_loaded_model
) -> None:
    """Verify predictor_node calls predict_case_b when route is case_b."""
    mock_get_loaded_model.return_value = MagicMock()
    mock_predict_case_b.return_value = {
        "prediction_results": [],
        "prediction_errors": [],
    }

    state = AgentState(user_input="...", route="case_b")
    predictor_node(state)

    mock_predict_case_b.assert_called_once()
