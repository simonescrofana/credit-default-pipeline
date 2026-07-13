"""Integration tests for the data extraction pipeline.

This module validates the correctness of the table extraction routines from the
OLTP database into the local OLAP raw storage. It verifies successful Parquet
file generation, proper handling of database connection timeouts, and strict
resource cleanup (closing writers) when I/O operations fail mid-stream.

"""

import datetime
import logging
import os
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from analytics.ingestion.extract import extract_table_data
from database.models import Company


def test_data_extraction(db_session: Session) -> None:
    """Verify successful data extraction and Parquet file generation.

    This test populates the database with a mock company entry, executes the
    extraction pipeline for that table, and asserts that the data is correctly
    fetched and written as a Parquet file in the designated raw data directory.

    """
    company = Company(
        vat_number="01234567890",
        legal_name="Acme Energy Consumer SPA",
        legal_form="SPA",
        industry_sector="manufacturing",
        foundation_date=datetime.date(2018, 4, 15),
        is_active=True,
    )
    db_session.add(company)
    db_session.flush()

    extract_table_data("company", Company)

    assert os.path.exists("data/raw/company.parquet")


def test_data_extraction_database_failure(db_session: Session) -> None:
    """Verify that a database operational error is properly caught and re-raised.

    This test mocks the pandas SQL query execution to simulate a database
    connection timeout and ensures that the extraction pipeline propagates
    the resulting OperationalError instead of failing silently.

    Raises:
        OperationalError: If the database communication fails during execution.

    """
    with patch("analytics.ingestion.extract.pd.read_sql_query") as mock_read:
        mock_read.side_effect = OperationalError(
            "SELECT ...", params={}, orig=Exception("Connection timed out")
        )
        with pytest.raises(OperationalError):
            extract_table_data("company", Company)


def test_data_extraction_writer_closes_on_error(db_session: Session) -> None:
    """Verify that ParquetWriter is explicitly closed even if write operations fail.

    Ensures that resource cleanup inside the finally block is triggered when an
    I/O or filesystem exception occurs mid-stream.

    Raises:
        OSError: Expected filesystem write failure simulated via mocking.

    """
    company = Company(
        vat_number="99999999999",
        legal_name="Fail Test SRL",
        legal_form="SRL",
        industry_sector="services",
        foundation_date=datetime.date(2018, 4, 15),
        is_active=True,
    )
    db_session.add(company)
    db_session.flush()

    with patch("analytics.ingestion.extract.pq.ParquetWriter") as mock_writer_class:
        mock_writer_instance = MagicMock()
        mock_writer_class.return_value = mock_writer_instance

        mock_writer_instance.write_table.side_effect = OSError(
            "No space left on device"
        )

        with pytest.raises(OSError):
            extract_table_data("company", Company)

        mock_writer_instance.close.assert_called_once()


def test_extract_table_data_empty_chunk(
    db_session: Session, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify system warnings when the overall data extraction yields zero total rows.

    This test patches the database session provider to inject the active test
    session, simulates a scenario where the extraction returns no records, and
    asserts that the system generates a warning log indicating that the process
    completed without retrieving any data.

    Args:
        caplog (LogCaptureFixture): Pytest fixture to capture and inspect log
            records generated during execution.

    """

    def mock_get_db():
        yield db_session

    with patch("analytics.ingestion.extract.get_db", mock_get_db):
        with caplog.at_level(logging.WARNING, logger="analytics.ingestion.extract"):
            extract_table_data("company_empty_chunk", Company)

    assert any(
        "Extraction completed but no data was retrieved." in record.message
        for record in caplog.records
        if record.levelname == "WARNING"
    )


def test_extract_table_data_no_chunks_at_all(caplog: pytest.LogCaptureFixture) -> None:
    """Verify system logging when pandas returns an empty iterator during extraction.

    This test mocks `read_sql_query` to return an empty list/iterator, simulating
    a structural lack of data chunks. It ensures that the pipeline correctly
    logs warnings for both the first empty chunk and the final zero-record
    extraction summary.

    Args:
        caplog (LogCaptureFixture): Pytest fixture to capture and inspect log
            records generated during execution.

    """
    with patch("analytics.ingestion.extract.pd.read_sql_query", return_value=[]):
        with caplog.at_level(logging.WARNING, logger="analytics.ingestion.extract"):
            extract_table_data("company_empty_iter", Company)

    assert any(
        "Chunk number 1 is empty." in record.message
        for record in caplog.records
        if record.levelname == "WARNING"
    )

    assert any(
        "Extraction completed but no data was retrieved." in record.message
        for record in caplog.records
        if record.levelname == "WARNING"
    )
