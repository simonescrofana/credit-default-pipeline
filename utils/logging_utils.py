"""Configures application logging and export telemetry pipelines.

This module initializes the standard library `logging` system and configures
the `logfire` exporter. It exposes `setup_logging()` to standardize log output
formats and telemetry ingestion for standalone execution environments.

"""

import logging

import logfire

from config import settings


def setup_logging(log_level: str = "INFO") -> None:
    """Configure the global logging system and telemetry exporter.

    Initializes `logging.basicConfig` with a standardized output format and
    attaches both a local stream handler and the `logfire` logging handler.

    Args:
        log_level (str): The logging threshold level (e.g., "DEBUG", "INFO").
            Defaults to "INFO".

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
