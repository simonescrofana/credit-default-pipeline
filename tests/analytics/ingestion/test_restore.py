"""Tests for the database restoration pipeline.

This module validates the correctness of the database recovery utility using
mock Parquet sources. It tests successful end-to-end data insertion across
relational tables, PostgreSQL primary key sequence counter re-alignment,
graceful handling and logging of missing source files, and transactional safety
(rollback and session closure) under database exception states.

"""

import datetime
import os
from collections.abc import Iterator
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from analytics.ingestion.restore import MAPPING, restore_database
from database.models import Company, EnergyContract


@pytest.fixture
def setup_tmp_parquet() -> Iterator[str]:
    """Create and clean up temporary Parquet files for database testing.

    This fixture initializes a local mock directory structure populated with
    temporary Parquet files. It feeds sample mock data into specific business
    tables (companies and energy contracts) while creating empty Parquet files
    for all other remaining tables in the MAPPING configuration to isolate the
    test run environment.

    Yields:
        str: The system filesystem path targeting the created temporary raw data
            directory.

    """
    tmp_repo = "tests/analytics/ingestion/data/raw"
    os.makedirs(tmp_repo)

    companies_data = pd.DataFrame(
        [
            {
                "id": 42,
                "vat_number": "12345678901",
                "legal_name": "Test Restore SPA",
                "legal_form": "SPA",
                "industry_sector": "manufacturing",
                "foundation_date": datetime.date(2010, 1, 1),
                "is_active": True,
            }
        ]
    )
    companies_data.to_parquet(tmp_repo + "/companies.parquet")

    contracts_data = pd.DataFrame(
        [
            {
                "id": 99,
                "company_id": 42,
                "commodity_type": "electricity",
                "market_type": "deregulated",
                "voltage_level": "medium",
                "pressure_level": None,
                "power_committed_kw": Decimal("150.00"),
                "gas_meter_class": None,
                "activation_date": datetime.date(2020, 5, 20),
                "termination_date": None,
                "contract_status": "active",
            }
        ]
    )
    contracts_data.to_parquet(tmp_repo + "/energy_contracts.parquet")

    for step in MAPPING:
        name = step["table_name"]
        if name not in ["companies", "energy_contracts"]:
            pd.DataFrame().to_parquet(tmp_repo + f"/{name}.parquet")

    yield tmp_repo

    for step in MAPPING:
        name = step["table_name"]
        os.remove(tmp_repo + f"/{name}.parquet")
    os.removedirs(tmp_repo)


def test_restore_database_integration(
    db_session: Session, setup_tmp_parquet: Iterator[str]
) -> None:
    """Integration test validating full database restoration and sequence resets.

    This test patches the active database provider to inject the isolated test
    session and triggers the `restore_database` function using mock Parquet sources.
    It verifies that data is accurately loaded into the respective relational tables
    (Company and EnergyContract) and validates that PostgreSQL primary key
    sequence counters are re-aligned by asserting that a subsequent record insert
    receives the correct auto-incremented identifier.

    """
    with patch("analytics.ingestion.restore.get_db", return_value=iter([db_session])):
        restore_database("tests/analytics/ingestion/data/raw")

    company = db_session.query(Company).filter_by(id=42).first()
    assert company is not None
    assert company.legal_name == "Test Restore SPA"
    assert company.vat_number == "12345678901"

    contract = db_session.query(EnergyContract).filter_by(id=99).first()
    assert contract is not None
    assert contract.company_id == 42
    assert contract.commodity_type == "electricity"

    new_company = Company(
        vat_number="00000000000",
        legal_name="Next Autoincrement SRL",
        legal_form="SRL",
        industry_sector="services",
        foundation_date=datetime.date(2025, 1, 1),
        is_active=True,
    )
    db_session.add(new_company)
    db_session.flush()

    assert new_company.id == 43


def test_restore_database_skips_missing_files(db_session, tmp_path, caplog):
    """Verify warning logs when Parquet source files are missing.

    This test executes the restore process targeting an empty directory. It
    ensures that the system handles the absence of files gracefully by emitting
    a warning log for each missing table defined in the configuration instead of
    crashing.

    Args:
        caplog (LogCaptureFixture): Pytest fixture to capture and inspect log
            records generated during execution.

    """
    empty_dir = str(tmp_path)

    with patch("analytics.ingestion.restore.get_db") as mock_get_db:
        mock_get_db.return_value = iter([db_session])

        with caplog.at_level("WARNING"):
            restore_database(dir_path=empty_dir)

    for step in MAPPING:
        table_name = step["table_name"]
        expected_warning = (
            f"File not found: {empty_dir}/{table_name}.parquet. "
            + f"Skipping table {table_name}."
        )
        assert any(expected_warning in record.message for record in caplog.records)


def test_restore_database_rollback_on_exception(db_session):
    """Verify that a database exception triggers a rollback and closes the session.

    This test simulates a critical database failure during the early truncate
    phase using a mock session. It asserts that the restoration pipeline correctly
    propagates the SQLAlchemyError, triggers a single atomic transaction rollback,
    and safely closes the session via the finally block.

    Raises:
        SQLAlchemyError: Expected database failure simulated via mocking.

    """
    mock_session = MagicMock()
    mock_session.bind.dialect.name = "sqlite"
    mock_session.execute.side_effect = SQLAlchemyError("Simulated database crash.")

    with patch("analytics.ingestion.restore.get_db") as mock_get_db:
        mock_get_db.return_value = iter([mock_session])

        with pytest.raises(SQLAlchemyError, match="Simulated database crash."):
            restore_database(dir_path="fake_repo")

    mock_session.rollback.assert_called_once()

    mock_session.close.assert_called_once()
