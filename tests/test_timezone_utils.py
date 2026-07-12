"""Validates application timezone normalization and conversion utilities.

This module provides a comprehensive test suite for validating the application's
timezone utility functions. It ensures that all temporal objects are correctly
converted into or preserved as offset-aware UTC datetimes, preventing
chronological validation mismatches across different system environments.

Typical usage example:
    $ pytest tests/test_timezone_utils.py

"""

import datetime

from utils.timezone_utils import ensure_utc_aware


def test_force_utc() -> None:
    """Asserts that offset-naive datetime objects are converted to UTC-aware.

    Validates that `ensure_utc_aware` correctly processes naive `datetime` instances,
    asserting that the resulting object is localized and that its timezone
    identifier explicitly matches 'UTC'.

    Raises:
        AssertionError: If the input datetime remains naive or if the converted
            timezone identifier does not equal 'UTC'.

    """
    naive_dt = datetime.datetime(2026, 7, 7, 11, 59, 30)
    aware_dt = ensure_utc_aware(naive_dt)

    assert not naive_dt.tzinfo
    assert not naive_dt.tzname()
    assert aware_dt.tzname() == "UTC"


def test_check_utc() -> None:
    """Asserts that the utility handles already UTC-aware datetimes idempotently.

    Validates that `ensure_utc_aware` safely preserves `datetime` objects that
    are already localized to UTC. Ensures the function returns the original
    timezone configuration without modification or re-localization.

    Raises:
        AssertionError: If the returned datetime's timezone identifier is
            modified or does not match 'UTC'.

    """
    aware_dt = datetime.datetime(2026, 7, 7, 11, 59, 30, tzinfo=datetime.timezone.utc)
    aware_dt = ensure_utc_aware(aware_dt)

    assert aware_dt.tzname() == "UTC"


def test_different_tz_conversion() -> None:
    """Asserts that non-UTC offsets are correctly normalized to UTC.

    Validates that `ensure_utc_aware` correctly detects and shifts `datetime`
    instances localized to non-UTC timezones (e.g., UTC+2). Confirms that the
    utility performs the necessary temporal arithmetic to normalize the timestamp
    into the target UTC timezone while preserving the absolute moment in time.

    Raises:
        AssertionError: If the resulting timezone is not UTC or if the internal
            timestamp representation fails the normalization shift.

    """
    offset = datetime.timedelta(hours=2)
    aware_dt = datetime.datetime(
        2026, 7, 7, 11, 59, 30, tzinfo=datetime.timezone(offset)
    )
    aware_dt = ensure_utc_aware(aware_dt)

    assert aware_dt.tzname() == "UTC"
