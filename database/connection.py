"""Module for connecting SQLAlchemy to PostgreSQL.

This module uses the script in config.py to import secretely
the credentials required to build the URL to the PostgreSQL database.
It creates the SQLAlchemy engine for the database and
it gives a generator of SQLAlchemy local sessions to work with the
database.
"""

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config import settings

DATABASE_URL = settings.DATABASE_URL

# pool_pre_ping, on connection request, checks the connection
# and re-creates it if it was broken
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
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
