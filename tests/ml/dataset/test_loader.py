"""Test the dataset loader module for the analytics data pipeline.

Provide unit tests for loading company credit profile data from the SQL
database into a pandas DataFrame, verifying proper indexing and error handling
using mocks.

"""

from unittest.mock import patch

import pandas as pd
import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ml.dataset.loader import load_data


@pytest.fixture
def fake_star_schema_data() -> pd.DataFrame:
    """Generate a mock DataFrame representing company credit profile records.

    Returns:
        pd.DataFrame: A populated DataFrame containing synthetic feature data,
        company metadata, date attributes, and target insolvency labels.

    """
    return pd.DataFrame(
        {
            "company_id": [1, 2, 3],
            "snapshot_date": pd.to_datetime(["2026-01-31", "2026-01-31", "2026-02-28"]),
            "company_age_days": [365, 1200, 40],
            "active_contracts_count": [1, 2, 1],
            "has_active_electricity_contract": [True, True, False],
            "has_active_gas_contract": [False, True, True],
            "leverage_ratio": [0.5, 1.2, None],
            "cash_to_debt_ratio": [0.3, 0.1, 0.8],
            "net_profit_margin": [0.05, -0.02, 0.1],
            "ebitda": [10000.0, -2000.0, 5000.0],
            "max_dpd_trailing_90d": [0, 120, 15],
            "avg_dpd_trailing_90d": [0.0, 85.5, 5.0],
            "unpaid_ratio_trailing_90d": [0.0, 0.4, 0.1],
            "total_outstanding_debt": [0.0, 15000.0, 500.0],
            "days_since_last_login": [2, 60, None],
            "login_velocity": [1.1, 0.2, None],
            "billing_disputes_count": [0, 3, 1],
            "average_satisfaction_score": [4.5, 2.1, None],
            "is_insolvent": [0, 1, 0],
            "legal_name": ["Alpha Srl", "Beta Spa", "Gamma Srl"],
            "industry_sector": ["manufacturing", "retail", "manufacturing"],
            "registered_office_region": ["Lazio", "Lombardia", "Lazio"],
            "year": [2026, 2026, 2026],
            "quarter": [1, 1, 1],
            "month": [1, 1, 2],
        }
    )


@patch("ml.dataset.loader.pd.read_sql")
def test_load_data_happy_path(
    mock_read_sql, db_session: Session, fake_star_schema_data
) -> None:
    """Test loading and indexing data into a multi-indexed DataFrame.

    Args:
        fake_star_schema_data (pd.DataFrame): The fixture providing
            sample credit profile data.

    """
    mock_read_sql.return_value = fake_star_schema_data

    result = load_data(db_session)

    assert isinstance(result.index, pd.MultiIndex)
    assert list(result.index.names) == ["company_id", "snapshot_date"]
    assert len(result) == 3
    assert "is_insolvent" in result.columns
    mock_read_sql.assert_called_once()


@patch("ml.dataset.loader.pd.read_sql")
def test_load_data_raises_on_sqlalchemy_error(
    mock_read_sql, db_session: Session
) -> None:
    """Ensure that load_data propagates SQLAlchemyError when database queries fail."""
    mock_read_sql.side_effect = SQLAlchemyError("Connection error!")

    with pytest.raises(SQLAlchemyError):
        load_data(db_session)
