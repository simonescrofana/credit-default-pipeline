"""Base Pydantic configuration schemas for database validation layers.

This module provides the abstract base schemas and configuration models used
to safely hydrate, parse, and validate operational records extracted from the
OLTP database layer, preparing them for downstream processing.

"""

from pydantic import BaseModel, ConfigDict


class BaseResponseSchema(BaseModel):
    """Core Pydantic model for database data hydration and validation.

    Acts as the base validation layer for loading and parsing operational records
    extracted from the database. It enables native compatibility with SQLAlchemy
    ORM attributes, ensuring that raw database objects are safely deserialized
    and validated against strict type constraints before downstream processing.

    Attributes:
        model_config: Pydantic configuration dictionary set to enable
            attribute hydration (`from_attributes=True`) for ORM compatibility.

    """

    model_config = ConfigDict(from_attributes=True)
