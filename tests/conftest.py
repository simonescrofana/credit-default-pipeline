"""Test configuration module.

This module configures global aspects that every test has to
consider. This scripts contains the configuration of
Logfire for test logging including the configuration of the tools
used in tests, which are:
    * SQLAlchemy.
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
