"""Test suite for the extractor node.

Covers the happy path and the not-found branch of `extract_case_a` (company
identifier resolution against the database), the SQLAlchemyError re-raise
path, and the happy path and validation-failure branch of `extract_case_b`
(ad hoc company data extraction and validation).

"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy.exc import SQLAlchemyError

from agent.nodes.extractor import extract_case_a, extract_case_b, extractor_node
from agent.state import AgentState
from schemas.agent.extraction_validation import CompanyIdentifiers, ExtractedCompanyData


@patch("agent.nodes.extractor.get_db")
@patch("agent.nodes.extractor.pd.read_sql")
@patch("agent.nodes.extractor.get_responder_llm")
def test_extract_case_a_happy_path_resolves_identifier(
    mock_get_responder_llm, mock_read_sql, mock_get_db
) -> None:
    """Verify a matched identifier is resolved into resolved_company_ids."""
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = CompanyIdentifiers(
        identifiers=["Rossi SRL"]
    )
    mock_llm.with_structured_output.return_value = mock_structured_llm
    mock_get_responder_llm.return_value = mock_llm

    fake_session = MagicMock()

    def fake_get_db():
        yield fake_session

    mock_get_db.return_value = fake_get_db()
    mock_read_sql.return_value = pd.DataFrame({"company_id": [1]})

    state = AgentState(user_input="Rischia default Rossi SRL?")
    result = extract_case_a(state)

    assert result["company_identifiers"] == ["Rossi SRL"]
    assert result["resolved_company_ids"] == [1]
    assert result["prediction_errors"] == []
    call_kwargs = mock_read_sql.call_args.kwargs
    assert call_kwargs["params"] == {"identifier": "Rossi SRL"}


@patch("agent.nodes.extractor.get_db")
@patch("agent.nodes.extractor.pd.read_sql")
@patch("agent.nodes.extractor.get_responder_llm")
def test_extract_case_a_identifier_not_found_records_error(
    mock_get_responder_llm, mock_read_sql, mock_get_db
) -> None:
    """Verify an unmatched identifier is recorded in prediction_errors, not raised."""
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = CompanyIdentifiers(
        identifiers=["Ghost SRL"]
    )
    mock_llm.with_structured_output.return_value = mock_structured_llm
    mock_get_responder_llm.return_value = mock_llm

    fake_session = MagicMock()

    def fake_get_db():
        yield fake_session

    mock_get_db.return_value = fake_get_db()
    mock_read_sql.return_value = pd.DataFrame(columns=["company_id"])

    state = AgentState(user_input="Rischia default Ghost SRL?")
    result = extract_case_a(state)

    assert result["resolved_company_ids"] == []
    assert result["prediction_errors"] == [
        "Company 'Ghost SRL' not found in the database."
    ]


@patch("agent.nodes.extractor.get_db")
@patch("agent.nodes.extractor.pd.read_sql")
@patch("agent.nodes.extractor.get_responder_llm")
def test_extract_case_a_sqlalchemy_error_is_reraised(
    mock_get_responder_llm, mock_read_sql, mock_get_db
) -> None:
    """Verify a database failure is re-raised, not swallowed into prediction_errors."""
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = CompanyIdentifiers(
        identifiers=["Rossi SRL"]
    )
    mock_llm.with_structured_output.return_value = mock_structured_llm
    mock_get_responder_llm.return_value = mock_llm

    fake_session = MagicMock()

    def fake_get_db():
        yield fake_session

    mock_get_db.return_value = fake_get_db()
    mock_read_sql.side_effect = SQLAlchemyError("Connection lost")

    state = AgentState(user_input="Rischia default Rossi SRL?")

    with pytest.raises(SQLAlchemyError, match="Connection lost"):
        extract_case_a(state)


@patch("agent.nodes.extractor.get_responder_llm")
def test_extract_case_b_happy_path_validates_successfully(
    mock_get_responder_llm,
) -> None:
    """Verify sufficient extracted data validates with no error recorded."""
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = ExtractedCompanyData(
        foundation_date="2015-03-01",
        industry_sector="manufacturing",
        unpaid_ratio_trailing_90d=0.3,
        total_outstanding_debt=15000.0,
    )
    mock_llm.with_structured_output.return_value = mock_structured_llm
    mock_get_responder_llm.return_value = mock_llm

    state = AgentState(user_input="Azienda manifatturiera con debito di 15000...")
    result = extract_case_b(state)

    assert result["prediction_errors"] == []
    assert result["raw_prediction_input"]["unpaid_ratio_trailing_90d"] == 0.3
    assert result["raw_prediction_input"]["total_outstanding_debt"] == 15000.0


@patch("agent.nodes.extractor.get_responder_llm")
def test_extract_case_b_missing_required_fields_records_error(
    mock_get_responder_llm,
) -> None:
    """Verify missing required fields produce a prediction_errors entry, not a raise."""
    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()

    mock_structured_llm.invoke.return_value = ExtractedCompanyData(
        industry_sector="manufacturing"
    )
    mock_llm.with_structured_output.return_value = mock_structured_llm
    mock_get_responder_llm.return_value = mock_llm

    state = AgentState(user_input="Un'azienda manifatturiera")
    result = extract_case_b(state)

    assert len(result["prediction_errors"]) == 1
    assert "incomplete or invalid" in result["prediction_errors"][0]


@patch("agent.nodes.extractor.extract_case_a")
def test_extractor_node_dispatches_case_a(mock_extract_case_a) -> None:
    """Verify extractor_node calls extract_case_a when route is case_a."""
    mock_extract_case_a.return_value = {
        "company_identifiers": ["Rossi SRL"],
        "resolved_company_ids": [1],
        "prediction_errors": [],
    }

    state = AgentState(user_input="Rischia default Rossi SRL?", route="case_a")
    result = extractor_node(state)

    mock_extract_case_a.assert_called_once_with(state)
    assert result == mock_extract_case_a.return_value


@patch("agent.nodes.extractor.extract_case_b")
def test_extractor_node_dispatches_case_b(mock_extract_case_b) -> None:
    """Verify extractor_node calls extract_case_b when route is case_b."""
    mock_extract_case_b.return_value = {
        "raw_prediction_input": {"unpaid_ratio_trailing_90d": 0.3},
        "prediction_errors": [],
    }

    state = AgentState(user_input="What if a company had high debt?", route="case_b")
    result = extractor_node(state)

    mock_extract_case_b.assert_called_once_with(state)
    assert result == mock_extract_case_b.return_value
