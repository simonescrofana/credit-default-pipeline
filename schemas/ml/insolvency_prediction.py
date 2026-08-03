"""Pydantic schema for ad hoc insolvency predictions (predictor case B).

Validates user/agent-supplied data for a company not present in the star
schema, before it reaches `ml.inference.predictor`. Which fields are
mandatory versus optional is driven directly by SHAP feature importance
(see `ml/evaluation/plots.ipynb`, beeswarm section): the two dominant
features are required, the rest are optional since XGBoost's native
missing-value handling degrades gracefully for them. Fields whose value
depends on the moment of the request (month, quarter, year) are never
accepted from the caller, they are derived server-side at scoring time,
not supplied by the user/agent. Field descriptions are written for an LLM
agent to relay to a non-technical end user (e.g. explaining what a field
means, or why a prediction was declined for missing it), not just as
internal documentation.

"""

import datetime
from typing import Literal, Optional, Self

from pydantic import Field, computed_field, model_validator

from schemas.database.base import BaseResponseSchema
from schemas.database.types import FoundationDate, IndustrySectorType
from utils.date_validation import validate_not_future_date

RegisteredOfficeRegion = Literal[
    "Abruzzo",
    "Basilicata",
    "Calabria",
    "Campania",
    "Emilia-Romagna",
    "Friuli-Venezia Giulia",
    "Lazio",
    "Liguria",
    "Lombardia",
    "Marche",
    "Molise",
    "Piemonte",
    "Puglia",
    "Sardegna",
    "Sicilia",
    "Toscana",
    "Trentino-Alto Adige",
    "Umbria",
    "Valle d'Aosta",
    "Veneto",
]


class InsolvencyPredictionRequest(BaseResponseSchema):
    r"""Schema validating ad hoc insolvency prediction requests.

    Covers companies not present in the star schema (predictor case B):
    the caller supplies raw company data directly, rather than an existing
    `company_id`. `foundation_date` is required instead of a direct
    `company_age_days` figure — the age in days is derived from it at
    scoring time, since deriving it server-side (a single, testable
    computation against the actual request date) is more reliable than
    relying on the caller to compute it correctly.

    Attributes:
        foundation_date (FoundationDate): Date of the company's foundation.
            Never in the future. `company_age_days` is computed from this
            field, not accepted directly.
        industry_sector (IndustrySectorType): Macro industrial sector, from
            the same closed set used in training.
        registered_office_region (RegisteredOfficeRegion | None): Region
            where the company's offices are located, from the same closed
            set used in training. Optional: SHAP shows region has
            near-zero impact on the model's predictions.
        unpaid_ratio_trailing_90d (float): Share of trailing-90-day billing
            left unpaid. Required: the single most influential feature in
            the model (see beeswarm plot in `ml/evaluation/plots.ipynb`).
        total_outstanding_debt (float): Total outstanding debt, in EUR.
            Required: the second most influential feature in the model.
        days_since_last_login (int | None): Days since the company's last
            platform login. Optional; leave unset if there is no login
            history to report.
        login_velocity (float | None): Login frequency over the trailing 90
            days (0-3). Optional; leave unset (not 0) if there were no
            logins in the trailing 90 days.
        cash_to_debt_ratio (float | None): Ratio of liquid cash to total
            debt. Optional.
        ebitda (float | None): Earnings before interest, taxes,
            depreciation, and amortization, in EUR. Can be negative.
            Optional.
        net_profit_margin (float | None): Net income as a share of revenue.
            Can be negative. Optional.
        leverage_ratio (float | None): Total debt divided by share capital.
            Always positive when available; leave unset if share capital
            is zero or negative (the ratio is not meaningful in that case).
        has_active_gas_contract (bool | None): Whether the company has an
            active gas supply contract. Optional; not every company has
            one, including inactive or defunct companies.
        has_active_electricity_contract (bool | None): Whether the company
            has an active electricity supply contract. Optional; not every
            company has one, including inactive or defunct companies.
        average_satisfaction_score (float | None): Average post-resolution
            customer support rating (1-5 stars). Optional; leave unset if
            the company has no resolved support tickets to average.
        billing_disputes_count (int | None): Number of billing disputes
            raised by the company. Optional.
        active_contracts_count (int | None): Number of currently active
            energy contracts (gas and/or electricity combined). Optional.

    """

    foundation_date: FoundationDate
    industry_sector: IndustrySectorType
    registered_office_region: Optional[RegisteredOfficeRegion] = None

    # required: dominant SHAP features
    unpaid_ratio_trailing_90d: float = Field(
        ...,
        ge=0,
        le=1,
        description="Share of trailing-90-day billing left unpaid, as a "
        "fraction between 0 and 1 (e.g. 0.25 for 25%% unpaid). This is the "
        "single most influential factor in the prediction.",
    )
    total_outstanding_debt: float = Field(
        ...,
        description="Total outstanding debt, in EUR. This is the second "
        "most influential factor in the prediction.",
    )

    # optional: everything below degrades gracefully via XGBoost's native
    # missing-value handling if left unset
    days_since_last_login: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of days since the company last logged into "
        "the platform. Leave unset if there is no login history at all.",
    )
    login_velocity: Optional[float] = Field(
        default=None,
        ge=0,
        le=3,
        description="How often the company logged in over the trailing 90 "
        "days, on a 0-3 scale. Leave unset (do not use 0) if there were no "
        "logins in the trailing 90 days.",
    )
    cash_to_debt_ratio: Optional[float] = Field(
        default=None, description="Ratio of liquid cash to total debt."
    )
    ebitda: Optional[float] = Field(
        default=None,
        description="Earnings before interest, taxes, depreciation, and "
        "amortization, in EUR. Can be negative for loss-making companies.",
    )
    net_profit_margin: Optional[float] = Field(
        default=None,
        description="Net income as a share of revenue. Can be negative "
        "for loss-making companies.",
    )
    leverage_ratio: Optional[float] = Field(
        default=None,
        gt=0,
        description="Total debt divided by share capital. Leave unset if "
        "share capital is zero or negative, since the ratio would not be "
        "meaningful in that case.",
    )
    has_active_gas_contract: Optional[bool] = Field(
        default=None,
        description="Whether the company currently has an active gas "
        "supply contract. Leave unset if unknown; not every company has "
        "one (including inactive or defunct companies).",
    )
    has_active_electricity_contract: Optional[bool] = Field(
        default=None,
        description="Whether the company currently has an active "
        "electricity supply contract. Leave unset if unknown; not every "
        "company has one (including inactive or defunct companies).",
    )
    average_satisfaction_score: Optional[float] = Field(
        default=None,
        ge=1,
        le=5,
        description="Average customer support satisfaction rating (1 to 5 "
        "stars) across the company's resolved support tickets. Leave "
        "unset if the company has no resolved tickets to average.",
    )
    billing_disputes_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of billing disputes raised by the company.",
    )
    active_contracts_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of currently active energy contracts (gas "
        "and/or electricity combined).",
    )

    @model_validator(mode="after")
    def foundation_date_constraint(self) -> Self:
        """Validate that the company foundation date resides in the past or present.

        Returns:
            Self: The validated instance of the model if constraints are met.

        Raises:
            ValueError: If the foundation date is set to a future date.

        """
        validate_not_future_date(self.foundation_date, "Foundation date")
        return self

    @computed_field
    @property
    def company_age_days(self) -> int:
        """Compute the company's age in days from `foundation_date` to today.

        Derived server-side at access time (not stored, not accepted as
        caller input), so it always reflects the actual moment of the
        request rather than a value the caller could compute incorrectly
        or let go stale.

        Returns:
            int: The number of days between `foundation_date` and today
                (UTC).

        """
        today = datetime.datetime.now(tz=datetime.timezone.utc).date()
        return (today - self.foundation_date).days
