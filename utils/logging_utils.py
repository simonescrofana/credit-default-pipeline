"""Configuration moduel for logs.

This module configures logging and logfire exporting a function
`setup_logging()` taking the log level as parameter (default is INFO).
It allows logging during the execution of single scripts.
"""

import logging

import logfire

from config import settings


def setup_logging(log_level: str = "INFO") -> None:
    """Configure the logs of the scripts.

    Calling this function only in the __main__ esnures the configuration of
    logging when scripts are not imported or executed by pytest or FastAPI.
    """
    LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=[logging.StreamHandler(), logfire.LogfireLoggingHandler()],
    )
    logfire.configure(token=settings.LOGFIRE_TOKEN.get_secret_value())
