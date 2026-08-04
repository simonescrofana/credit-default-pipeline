"""Pydantic schemas for the extractor node's structured LLM output.

Validates what the extractor node's LLM calls produce before any database
query or downstream validation. Two schemas cover the two routes handled by
the extractor:

- `CompanyIdentifiers`, for case_a: the free-text company identifiers
  (legal name or VAT number) mentioned in the prompt, later resolved
  against the database.
- `ExtractedCompanyData`, for case_b: a deliberately permissive mirror of
  `InsolvencyPredictionRequest` with every field optional and none of its
  numeric/categorical constraints enforced. The extractor's job is only to
  report what the prompt actually contains; whether that is enough to
  produce a prediction — required fields present, values within range —
  is decided by `InsolvencyPredictionRequest` itself, not by this schema.
  Keeping the two separate means an LLM extraction quirk (e.g. defaulting
  an unset value instead of leaving it out) cannot silently masquerade as
  a valid request.

"""

import datetime
from typing import Optional

from pydantic import BaseModel, Field

from schemas.database.types import IndustrySectorType
from schemas.ml.insolvency_prediction import RegisteredOfficeRegion


class CompanyIdentifiers(BaseModel):
    """Represent the company identifiers extracted from the prompt (case_a).

    Attributes:
        identifiers (list[str]): Company legal names or VAT numbers
            mentioned in the request, as free text. Resolution against the
            database (and handling of any that don't match) happens
            downstream, not here.

    """

    identifiers: list[str] = Field(
        description="Company legal names or VAT numbers mentioned in the "
        "user's request, exactly as they appear in the text."
    )


class ExtractedCompanyData(BaseModel):
    """Represent ad hoc company data extracted from the prompt (case_b).

    Mirrors the fields of `InsolvencyPredictionRequest`, but with every
    field optional and unconstrained — this schema only captures what the
    prompt states, not whether it is sufficient or valid for a prediction.

    Attributes:
        foundation_date (date | None): Date of the company's foundation,
            if mentioned.
        industry_sector (str | None): Macro industrial sector, if
            mentioned.
        registered_office_region (str | None): Region where the company's
            offices are located, if mentioned.
        unpaid_ratio_trailing_90d (float | None): Share of trailing-90-day
            billing left unpaid, if mentioned.
        total_outstanding_debt (float | None): Total outstanding debt, in
            EUR, if mentioned.
        days_since_last_login (int | None): Days since the company's last
            platform login, if mentioned.
        login_velocity (float | None): Login frequency over the trailing
            90 days, if mentioned.
        cash_to_debt_ratio (float | None): Ratio of liquid cash to total
            debt, if mentioned.
        ebitda (float | None): EBITDA, in EUR, if mentioned.
        net_profit_margin (float | None): Net income as a share of
            revenue, if mentioned.
        leverage_ratio (float | None): Total debt divided by share
            capital, if mentioned.
        has_active_gas_contract (bool | None): Whether the company has an
            active gas supply contract, if mentioned.
        has_active_electricity_contract (bool | None): Whether the company
            has an active electricity supply contract, if mentioned.
        average_satisfaction_score (float | None): Average customer
            support satisfaction rating, if mentioned.
        billing_disputes_count (int | None): Number of billing disputes
            raised, if mentioned.
        active_contracts_count (int | None): Number of currently active
            energy contracts, if mentioned.

    """

    foundation_date: Optional[datetime.date] = Field(
        default=None,
        description="Date the company was founded, if mentioned in the request.",
    )
    industry_sector: Optional[IndustrySectorType] = Field(
        default=None, description="The company's industry sector, if mentioned."
    )
    registered_office_region: Optional[RegisteredOfficeRegion] = Field(
        default=None,
        description="The Italian region of the company's registered office, "
        "if mentioned.",
    )
    unpaid_ratio_trailing_90d: Optional[float] = Field(
        default=None,
        description="Share of trailing-90-day billing left unpaid, as a "
        "fraction between 0 and 1, if mentioned.",
    )
    total_outstanding_debt: Optional[float] = Field(
        default=None, description="Total outstanding debt in EUR, if mentioned."
    )
    days_since_last_login: Optional[int] = Field(
        default=None,
        description="Days since the company's last platform login, if mentioned.",
    )
    login_velocity: Optional[float] = Field(
        default=None,
        description="Login frequency over the trailing 90 days (0-3), if mentioned.",
    )
    cash_to_debt_ratio: Optional[float] = Field(
        default=None, description="Ratio of liquid cash to total debt, if mentioned."
    )
    ebitda: Optional[float] = Field(
        default=None, description="EBITDA in EUR, if mentioned."
    )
    net_profit_margin: Optional[float] = Field(
        default=None, description="Net income as a share of revenue, if mentioned."
    )
    leverage_ratio: Optional[float] = Field(
        default=None,
        description="Total debt divided by share capital, if mentioned.",
    )
    has_active_gas_contract: Optional[bool] = Field(
        default=None,
        description="Whether the company has an active gas supply contract, "
        "if mentioned.",
    )
    has_active_electricity_contract: Optional[bool] = Field(
        default=None,
        description="Whether the company has an active electricity supply "
        "contract, if mentioned.",
    )
    average_satisfaction_score: Optional[float] = Field(
        default=None,
        description="Average customer support satisfaction rating (1-5), if mentioned.",
    )
    billing_disputes_count: Optional[int] = Field(
        default=None,
        description="Number of billing disputes raised, if mentioned.",
    )
    active_contracts_count: Optional[int] = Field(
        default=None,
        description="Number of currently active energy contracts, if mentioned.",
    )
