"""Test suite for the ad hoc insolvency prediction request schema (predictor case B).

Covers the happy path (required-only and fully-populated instances), missing
required fields, per-field value constraints (parametrized), the
foundation_date-not-in-the-future validator, and the company_age_days
computed value.

"""

import datetime

import pytest
from pydantic import ValidationError

from schemas.insolvency_prediction import InsolvencyPredictionRequest

VALID_REQUIRED_ONLY = {
    "foundation_date": datetime.date(2015, 3, 1),
    "industry_sector": "manufacturing",
    "unpaid_ratio_trailing_90d": 0.3,
    "total_outstanding_debt": 15000.0,
}

VALID_FULLY_POPULATED = {
    **VALID_REQUIRED_ONLY,
    "registered_office_region": "Lombardia",
    "days_since_last_login": 5,
    "login_velocity": 1.5,
    "cash_to_debt_ratio": 0.8,
    "ebitda": -2000.0,
    "net_profit_margin": -0.05,
    "leverage_ratio": 2.1,
    "has_active_gas_contract": True,
    "has_active_electricity_contract": False,
    "average_satisfaction_score": 4.0,
    "billing_disputes_count": 2,
    "active_contracts_count": 1,
}


def test_required_only_instance_defaults_optional_fields_to_none() -> None:
    """Verify an instance built with only required fields is valid."""
    request = InsolvencyPredictionRequest(**VALID_REQUIRED_ONLY)

    assert request.registered_office_region is None
    assert request.days_since_last_login is None
    assert request.login_velocity is None
    assert request.cash_to_debt_ratio is None
    assert request.ebitda is None
    assert request.net_profit_margin is None
    assert request.leverage_ratio is None
    assert request.has_active_gas_contract is None
    assert request.has_active_electricity_contract is None
    assert request.average_satisfaction_score is None
    assert request.billing_disputes_count is None
    assert request.active_contracts_count is None


def test_fully_populated_instance_is_valid() -> None:
    """Verify a fully-populated instance is accepted."""
    request = InsolvencyPredictionRequest(**VALID_FULLY_POPULATED)

    assert request.foundation_date == VALID_FULLY_POPULATED["foundation_date"]
    assert request.industry_sector == VALID_FULLY_POPULATED["industry_sector"]
    assert (
        request.registered_office_region
        == VALID_FULLY_POPULATED["registered_office_region"]
    )
    assert (
        request.unpaid_ratio_trailing_90d
        == VALID_FULLY_POPULATED["unpaid_ratio_trailing_90d"]
    )
    assert (
        request.total_outstanding_debt
        == VALID_FULLY_POPULATED["total_outstanding_debt"]
    )
    assert (
        request.days_since_last_login == VALID_FULLY_POPULATED["days_since_last_login"]
    )
    assert request.login_velocity == VALID_FULLY_POPULATED["login_velocity"]
    assert request.cash_to_debt_ratio == VALID_FULLY_POPULATED["cash_to_debt_ratio"]
    assert request.ebitda == VALID_FULLY_POPULATED["ebitda"]
    assert request.net_profit_margin == VALID_FULLY_POPULATED["net_profit_margin"]
    assert request.leverage_ratio == VALID_FULLY_POPULATED["leverage_ratio"]
    assert (
        request.has_active_gas_contract
        == VALID_FULLY_POPULATED["has_active_gas_contract"]
    )
    assert (
        request.has_active_electricity_contract
        == VALID_FULLY_POPULATED["has_active_electricity_contract"]
    )
    assert (
        request.average_satisfaction_score
        == VALID_FULLY_POPULATED["average_satisfaction_score"]
    )
    assert (
        request.billing_disputes_count
        == VALID_FULLY_POPULATED["billing_disputes_count"]
    )
    assert (
        request.active_contracts_count
        == VALID_FULLY_POPULATED["active_contracts_count"]
    )


@pytest.mark.parametrize(
    "missing_field",
    [
        "foundation_date",
        "industry_sector",
        "unpaid_ratio_trailing_90d",
        "total_outstanding_debt",
    ],
)
def test_missing_required_field_raises(missing_field: str) -> None:
    """Verify omitting any required field raises ValidationError."""
    data = {k: v for k, v in VALID_REQUIRED_ONLY.items() if k != missing_field}

    with pytest.raises(ValidationError):
        InsolvencyPredictionRequest(**data)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("unpaid_ratio_trailing_90d", -0.01),
        ("unpaid_ratio_trailing_90d", 1.01),
        ("login_velocity", -0.01),
        ("login_velocity", 3.01),
        ("leverage_ratio", 0),
        ("leverage_ratio", -1.0),
        ("average_satisfaction_score", 0.99),
        ("average_satisfaction_score", 5.01),
        ("days_since_last_login", -1),
        ("billing_disputes_count", -1),
        ("active_contracts_count", -1),
        ("industry_sector", "not_a_real_sector"),
        ("registered_office_region", "not_a_real_region"),
    ],
)
def test_invalid_field_value_raises(field: str, invalid_value: object) -> None:
    """Verify each field's constraint rejects an out-of-range or unrecognized value."""
    data = {**VALID_REQUIRED_ONLY, field: invalid_value}

    with pytest.raises(ValidationError):
        InsolvencyPredictionRequest(**data)


def test_future_foundation_date_raises() -> None:
    """Verify a foundation_date in the future is rejected."""
    tomorrow = datetime.datetime.now(
        tz=datetime.timezone.utc
    ).date() + datetime.timedelta(days=1)
    data = {**VALID_REQUIRED_ONLY, "foundation_date": tomorrow}

    with pytest.raises(ValidationError, match="can not be in the future"):
        InsolvencyPredictionRequest(**data)


def test_company_age_days_matches_foundation_date() -> None:
    """Verify company_age_days is computed correctly from foundation_date."""
    hundred_days_ago = datetime.datetime.now(
        tz=datetime.timezone.utc
    ).date() - datetime.timedelta(days=100)
    data = {**VALID_REQUIRED_ONLY, "foundation_date": hundred_days_ago}

    request = InsolvencyPredictionRequest(**data)

    assert request.company_age_days == 100
