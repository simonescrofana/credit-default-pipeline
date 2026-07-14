"""Test configuration and global instrumentation module.

This module initializes global configurations, logging providers, and subsystem
instrumentation required across the test suite execution.

Supported tools and integrations:
    * Logfire: Distributed tracing and structured test logging execution.
    * SQLAlchemy: Database engine query monitoring and ORM instrumentation.

"""

import logging
import os
from collections.abc import Iterator

import logfire
import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

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


@pytest.fixture(scope="session", autouse=True)
def setup_test_database() -> Iterator[None]:
    """Set up and tears down the database environment for the test session.

    Configures a PostgreSQL connection with NullPool if running in a GitHub
    Actions CI environment to prevent pandas from freezing active connections.
    Otherwise, initializes an in-memory SQLite database and explicitly enables
    foreign key support.

    Yields:
        None: Control is yielded back to the test suite for the session duration.

    """
    is_ci_environment = os.getenv("GITHUB_ACTIONS") == "true"

    if is_ci_environment:
        db_url = settings.database_url

        # pandas may keep active connections freezing Postgres during tests
        # NullPool forces connection closure
        global_engine = create_engine(db_url, poolclass=NullPool)

    else:
        global_engine = create_engine("sqlite:///:memory:")

        # This function bypasses a SQLite limit and allows fk constraints.
        @event.listens_for(global_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    Base.metadata.create_all(global_engine)

    yield

    Base.metadata.drop_all(global_engine)
    global_engine.dispose()


@pytest.fixture(scope="function")
def db_session() -> Iterator[Session]:
    """Provide a transactional database session for a single test function.

    Create a clean database environment for each test. For CI environments, it
    initializes a PostgreSQL engine and truncates all tables after execution to
    ensure isolation. For local environments, it uses an in-memory SQLite
    database with foreign key support enabled and drops the schema afterward.

    Yields:
        Session: A SQLAlchemy database session object.

    """
    is_ci_environment = os.getenv("GITHUB_ACTIONS") == "true"

    if is_ci_environment:
        db_url = settings.database_url
        test_engine = create_engine(db_url, poolclass=NullPool)

    else:
        test_engine = create_engine("sqlite:///:memory:")

        @event.listens_for(test_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        Base.metadata.create_all(test_engine)

    TestingSessionLocal = sessionmaker(bind=test_engine)
    session = TestingSessionLocal()

    try:
        yield session

    finally:
        session.close()

        if is_ci_environment:
            with test_engine.begin() as conn:
                for table in reversed(Base.metadata.sorted_tables):
                    conn.execute(
                        text(f"TRUNCATE TABLE {table.name} RESTART IDENTITY CASCADE;")
                    )
        else:
            Base.metadata.drop_all(test_engine)

        test_engine.dispose()
