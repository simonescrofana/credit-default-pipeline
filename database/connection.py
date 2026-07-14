"""Module for configuring the SQLAlchemy connection to PostgreSQL.

This module initializes the SQLAlchemy database engine using credentials
injected from the application settings. It provides a session factory for
generating local database sessions and exposes a context generator to manage
the session lifecycle safely.

Attributes:
    engine: The SQLAlchemy Engine instance managing the database connection pool.
    SessionLocal: The SQLAlchemy sessionmaker factory for creating new Session
        instances.

"""

import logging
import os
from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from config import settings

logger = logging.getLogger(__name__)

is_ci_environment = os.getenv("GITHUB_ACTIONS") == "true"

DATABASE_URL = settings.database_url

logger.debug("Initializing SQLAlchemy engine with pool_pre_ping=True")

if is_ci_environment:
    # pool_pre_ping, on connection request, checks the connection
    # and re-creates it if it was broken
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, poolclass=NullPool)

else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(
    autoflush=False,  # avoids sending queries automatically
    bind=engine,
)


def get_db() -> Iterator[Session]:
    """Provide a transactional database session context.

    Yields a SQLAlchemy Session connected to the PostgreSQL database.
    The try-finally block guarantees that the local session is reliably
    closed after the caller finishes operations, even if exceptions occur
    during processing.

    Yields:
        Iterator[Session]: A SQLAlchemy database session object.

    Raises:
        Exception: Re-raises any exception encountered during the session
            lifecycle after logging the failure.

    """
    logger.debug("Opening a new database Local Session")
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error("Database transaction error occurred: %s", e, exc_info=True)
        raise
    finally:
        logger.debug("Closing database Local Session")
        db.close()
