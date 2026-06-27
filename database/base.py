"""Module to define the declarative base for SQLAlchemy ORM models.

This module provides the central schema metadata container used by both
the application models and the Alembic migration framework.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Abstract base class for all SQLAlchemy ORM models.

    Serves as the registry for database schema metadata. All domain models
    must inherit from this class to be tracked by SQLAlchemy and Alembic.
    """

    pass
