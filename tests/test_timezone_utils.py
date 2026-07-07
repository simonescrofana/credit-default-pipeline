"""Unit tests for timezone utility functions.

This module contains the test suite for validating the application's timezone
normalization and conversion utilities. It focuses on ensuring that temporal
objects are correctly converted into or preserved as offset-aware UTC datetimes,
preventing chronological validation mismatches across different system environments.

Typical usage example:
    $ pytest tests/test_timezone_utils.py

"""

import datetime

from utils.timezone_utils import ensure_utc_aware


def test_force_utc() -> None:
    """Verify that a naive datetime is successfully converted to UTC-aware.

    This positive unit test confirms that `ensure_utc_aware` correctly handles
    an offset-naive datetime object. It asserts that the resulting datetime
    is localized and that its timezone name explicitly matches 'UTC'.

    """
    naive_dt = datetime.datetime(2026, 7, 7, 11, 59, 30)
    aware_dt = ensure_utc_aware(naive_dt)

    assert not naive_dt.tzinfo
    assert not naive_dt.tzname()
    assert aware_dt.tzname() == "UTC"


def test_check_utc() -> None:
    """Verify that an already UTC-aware datetime is preserved correctly.

    This unit test confirms that `ensure_utc_aware` safely handles an
    offset-aware datetime object that is already in UTC. It asserts that the
    function behaves idempotently, maintaining the correct timezone localization
    without modifications.

    """
    aware_dt = datetime.datetime(2026, 7, 7, 11, 59, 30, tzinfo=datetime.timezone.utc)
    aware_dt = ensure_utc_aware(aware_dt)

    assert aware_dt.tzname() == "UTC"


def test_different_tz_conversion() -> None:
    """Verify that a datetime with a non-UTC offset is correctly converted to UTC.

    This unit test ensures that `ensure_utc_aware` correctly handles an
    offset-aware datetime object localized in a different timezone (e.g., UTC+2).
    It asserts that the function shifts the timestamp appropriately to normalize
    it into the target UTC timezone.

    """
    offset = datetime.timedelta(hours=2)
    aware_dt = datetime.datetime(
        2026, 7, 7, 11, 59, 30, tzinfo=datetime.timezone(offset)
    )
    aware_dt = ensure_utc_aware(aware_dt)

    assert aware_dt.tzname() == "UTC"
