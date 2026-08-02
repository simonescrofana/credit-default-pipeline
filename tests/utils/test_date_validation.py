"""Test suite for the shared "not in the future" date validation utility.

Covers the happy path (past/today dates pass silently), the sad path (a
future date raises ValueError), and that the error message reflects the
caller-supplied field_label, since a broken label would fail silently
otherwise.

"""

import datetime

import pytest

from utils.date_validation import validate_not_future_date


def test_validate_not_future_date_accepts_past_and_present_dates() -> None:
    """Verify a past date and today's date both pass without raising."""
    today = datetime.datetime.now(tz=datetime.timezone.utc).date()
    past_date = today - datetime.timedelta(days=365)

    validate_not_future_date(past_date, "Some date")
    validate_not_future_date(today, "Some date")


def test_validate_not_future_date_rejects_future_date() -> None:
    """Verify a future date raises ValueError."""
    today = datetime.datetime.now(tz=datetime.timezone.utc).date()
    future_date = today + datetime.timedelta(days=1)

    with pytest.raises(ValueError, match="can not be in the future"):
        validate_not_future_date(future_date, "Some date")


def test_validate_not_future_date_error_message_includes_field_label() -> None:
    """Verify the error message reflects the caller-supplied field_label."""
    today = datetime.datetime.now(tz=datetime.timezone.utc).date()
    future_date = today + datetime.timedelta(days=1)

    with pytest.raises(ValueError, match="Foundation date can not be in the future"):
        validate_not_future_date(future_date, "Foundation date")
