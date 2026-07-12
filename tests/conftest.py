"""Test configuration and global instrumentation module.

This module initializes global configurations, logging providers, and subsystem
instrumentation required across the test suite execution.

Supported tools and integrations:
    * Logfire: Distributed tracing and structured test logging execution.
    * SQLAlchemy: Database engine query monitoring and ORM instrumentation.

"""

import logging
from collections.abc import Iterator

import logfire
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from config import settings
from database.base import Base
from database.connection import engine

LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logfire.configure(token=settings.LOGFIRE_TOKEN.get_secret_value())
logging.basicConfig(
    level=logging.DEBUG,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[logging.StreamHandler(), logfire.LogfireLoggingHandler()],
)

logfire.instrument_sqlalchemy(engine=engine)


@pytest.fixture(scope="function")
def db_session() -> Iterator[Session]:
    """Provide an isolated, in-memory SQLite session with enforced foreign keys.

    Creates a transient, serverless SQLite database initialized with the full
    declarative schema metadata. This setup ensures high-speed execution and
    zero-dependency environments for database testing compared to PostgreSQL.

    Yields:
        Session: Active SQLAlchemy session bound to the clean memory engine.

    """
    engine = create_engine("sqlite:///:memory:")

    # This function bypasses a SQLite limit and allows fk constraints.
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)

    TestingSessionLocal = sessionmaker(bind=engine)
    session = TestingSessionLocal()

    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()
