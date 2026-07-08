"""Application configuration and environment variable validation.

Defines the centralized configuration schema using Pydantic Settings. This
module manages validation rules for environment variables required by the
application, specifically orchestrating service configurations for:
    * PostgreSQL (database connectivity)
    * Logfire (telemetry and observability)

"""

from typing import Annotated

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Field definition for passwords
SecretPassword = Annotated[SecretStr, Field(description="Protected secret password.")]


class Settings(BaseSettings):
    """Application configuration manager.

    Orchestrates the loading and validation of environment variables via `.env`
    files. This class serves as the central source of truth for service
    configurations, including PostgreSQL connection parameters and Logfire
    telemetry credentials.

    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",  # not every system uses same enconding
        case_sensitive=True,
    )

    # POSTGRES
    POSTGRES_USER: Annotated[
        str, Field(description="Username for Postgres authentication.")
    ]
    POSTGRES_PASSWORD: SecretPassword
    POSTGRES_DB: Annotated[str, Field(description="Name of the relational database.")]
    POSTGRES_PORT: Annotated[
        int, Field(description="Port relative to the PostgreSQL service.")
    ] = 5432
    POSTGRES_HOST: Annotated[
        str, Field(description="Network host for the database.")
    ] = "localhost"

    # LOGFIRE
    LOGFIRE_TOKEN: SecretPassword

    @computed_field
    @property
    def database_url(self) -> str:
        """Builds dynamically the URL string for connecting to PostgreSQL.

        It uses the PostgreSQL settings found in the environment to dynamically
        build the URL string to the database of PostgreSQL. It allows
        connecting to the PostgreSQL database using the relevant tools such as
        SQLAlchemy present in the other scripts.

        Returns:
            str: connection URL string with the format:
                'postgresql://user:password@host:port/database_name'.

        """
        secret_password = self.POSTGRES_PASSWORD.get_secret_value()
        return (
            f"postgresql://{self.POSTGRES_USER}:{secret_password}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


settings = Settings()
