"""Configuration module for the software.

This module, based on Pydantic Settings defines the validation rules for each
environment variable required by the software to work. It configures the
following services:
    * PostgreSQL
    * Logfire
"""

from typing import Annotated

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Field definition for passwords
SecretPassword = Annotated[SecretStr, Field(description="Protected secret password.")]


class Settings(BaseSettings):
    """Manager of the configuration settings.

    It scans the system environment (specially the .env file) to load and to validate
    the required setting variables for each service requiring configuration.

    Services:

        * PostgreSQL:
        Attributes:
            * POSTGRES_USER (str): Username for PostgreSQL authentication.
            * POSTGRES_PASSWORD (SecretStr): Protected secret user's password.
            * POSTGRES_DB (str): Name of the relational database.
            * POSTGRES_PORT (int): Port relative to the PostgreSQL service.
            Default: 5432.
            * POSTGRES_HOST (str): Network host for the database. Default: "localhost".

        * Logfire:
        Attributes:
            * LOGFIRE_TOKEN (str): secret API key for writing tokens (logs) on
            Logfire dashboard.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",  # not every system uses same enconding
        case_sensitive=True,
        extra="ignore",
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
