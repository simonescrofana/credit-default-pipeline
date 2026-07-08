"""Test configuration and global instrumentation module.

This module initializes global configurations, logging providers, and subsystem
instrumentation required across the test suite execution.

Supported tools and integrations:
    * Logfire: Distributed tracing and structured test logging execution.
    * SQLAlchemy: Database engine query monitoring and ORM instrumentation.

"""

import logging

import logfire

from config import settings
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
