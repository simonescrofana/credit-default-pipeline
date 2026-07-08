"""Timezone utility functions for temporal data standardization.

This module provides utility functions to manage, convert, and enforce timezone
consistency across the application's data models and pipelines. It centralizes
the system-wide convention of working exclusively with Coordinated Universal
Time (UTC) for all datetime-aware operations, avoiding timezone mismatches during
database persistence and automated testing.

Typical usage example:
    from utils.timezone_utils import ensure_utc_aware
    aware_dt = ensure_utc_aware(naive_dt)

"""

import datetime


def ensure_utc_aware(dt: datetime.datetime) -> datetime.datetime:
    """Guarantee that a datetime object is offset-aware and set to UTC.

    Inspects the input `datetime` metadata. If the object is naive (lacks
    timezone info), it assigns the UTC timezone. If the object is already
    offset-aware, it converts the timestamp to UTC, preserving the absolute
    moment in time.

    Args:
        dt (datetime.datetime): The datetime object to be standardized.

    Returns:
        datetime.datetime: An offset-aware datetime object localized in UTC.

    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)

    return dt.astimezone(datetime.timezone.utc)
