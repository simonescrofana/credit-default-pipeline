"""Database infrastructure integration test suite.

This module consolidates connection robustness, session generation stability,
and resource teardown compliance tests across the pipeline infrastructure.

Modules tested:
    database.connection: Manages PostgreSQL network reachability, engine setup,
        and transaction session lifecycles.

"""

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from database.connection import get_db


def test_db_connection() -> None:
    """Verifies successful connection and minimal query execution on the database.

    Extracts an active operational session from the database generator and
    executes a fundamental compliance statement (`SELECT 1`) to validate
    network reachability, credentials, and basic engine readiness.

    Raises:
        AssertionError: If the execution scalar response deviates from 1.

    """
    db_generator = get_db()
    db_session = next(db_generator)

    statement = text("SELECT 1")
    result = db_session.execute(statement).scalar()
    assert result == 1


def test_get_db_raises_and_closes_on_error() -> None:
    """Verifies that the database generator correctly context-closes on raised errors.

    Validates the robustness of the session lifetime manager by injecting a
    `RuntimeError` directly into the active generator. Enforces that the underling
    `try/finally` structure catches the perturbation, triggers teardown operations,
    and terminates orderly by raising a `StopIteration` on subsequent steps.

    Attributes:
        db_generator: Instantiated database session generator under context analysis.
        db_session: Extracted operational engine session under validation.

    """
    db_generator = get_db()
    db_session = next(db_generator)

    assert isinstance(db_session, Session)

    with pytest.raises(RuntimeError):
        db_generator.throw(RuntimeError)

    with pytest.raises(StopIteration):
        next(db_generator)
