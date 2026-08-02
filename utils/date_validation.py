"""Provide validation helper functions for input dates and timestamps.

Define reusable validator functions to enforce temporal integrity constraints across
data schemas, such as rejecting future-dated entries.

"""

import datetime


def validate_not_future_date(value: datetime.date, field_label: str) -> None:
    """Raise ValueError if the given date is in the future relative to today (UTC).

    Args:
        value (datetime.date): The date to validate.
        field_label (str): Human-readable field name for the error message.

    Raises:
        ValueError: If `value` is later than today's UTC date.

    """
    today = datetime.datetime.now(tz=datetime.timezone.utc).date()
    if value > today:
        raise ValueError(f"{field_label} can not be in the future.")
