"""Tests for the relational database layer.

This script tests, with pytest, the functions in the following
scripts:
    *connection.py:
        tests the connection to the PostgreSQL database.
"""

from collections.abc import Iterator

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from database.connection import SessionLocal


@pytest.fixture(scope="module")
def get_db() -> Iterator[Session]:
    """Give a SQLAlchemy Session to work as a fixture.

    It provides a SQLAlchemy Session to work with the PostgreSQL
    database. It is defined as a fixture to test connection
    to the PostgreSQL database.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_db_connection(get_db: Iterator[Session]) -> None:
    """Test connection to PostgreSQL database.

    The function tests the connection asking a simple "SELECT 1" statement.
    """
    statement = text("SELECT 1")
    result = get_db.execute(statement).scalar()
    assert result == 1
