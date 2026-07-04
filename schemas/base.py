"""Base Pydantic Configuration Module for Schemas validation."""

from pydantic import BaseModel, ConfigDict


class BaseResponseSchema(BaseModel):
    """Core Pydantic model for all outbound API responses.

    Enables native compatibility with SQLAlchemy ORM attributes, allowing
    automatic serialization of database objects into validated JSON primitives.
    """

    model_config = ConfigDict(from_attributes=True)
