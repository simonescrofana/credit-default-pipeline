"""Module for connecting SQLAlchemy to PostgreSQL.

This module uses the script in config.py to import secretely
the credentials required to build the URL to the PostgreSQL database.
It creates the SQLAlchemy engine for the database and
it gives a generator of SQLAlchemy local sessions to work with the
database.
"""

import logging
from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config import settings

logger = logging.getLogger(__name__)

DATABASE_URL = settings.database_url

# pool_pre_ping, on connection request, checks the connection
# and re-creates it if it was broken
logger.debug("Initializing SQLAlchemy engine with pool_pre_ping=True")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(
    autoflush=False,  # avoids sending queries automatically
    bind=engine,
)


def get_db() -> Iterator[Session]:
    """Give a SQLAlchemy Session to work.

    It provides a SQLAlchemy Session to work with the PostgreSQL
    database connected thanks to the engine defined above.
    The function provides a generator of Session and the try finally
    statement ensures the closure of the connection when the work
    on this Local Session is done.
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
