"""Test configuration module.

This module configures global aspects that every test has to
consider. This scripts contains the configuration of
Logfire for test logging including the configuration of the tools
used in tests, which are:
    * SQLAlchemy.
"""

from logging import basicConfig

import logfire

from config import settings
from database.connection import engine

logfire.configure(token=settings.LOGFIRE_TOKEN.get_secret_value())
basicConfig(handlers=[logfire.LogfireLoggingHandler()])

logfire.instrument_sqlalchemy(engine=engine)
