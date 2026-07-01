"""Tests for the relational database layer.

This script tests, with pytest, the functions in the following
scripts:
    *connection.py:
        tests the connection to the PostgreSQL database.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from database.connection import get_db


def test_db_connection() -> None:
    """Test connection to PostgreSQL database.

    The function tests the connection asking a simple "SELECT 1" statement.
    """
    db_generator = get_db()
    db_session = next(db_generator)

    statement = text("SELECT 1")
    result = db_session.execute(statement).scalar()
    assert result == 1


def test_get_db_raises_and_closes_on_error() -> None:
    """Test on closure of connection in error case.

    The function tests the correct closure of the database Local Session
    when an error occurs.
    """
    db_generator = get_db()
    db_session = next(db_generator)

    assert isinstance(db_session, Session)

    with pytest.raises(RuntimeError):
        db_generator.throw(RuntimeError)

    with pytest.raises(StopIteration):
        next(db_generator)
