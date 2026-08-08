"""Global domain types and type aliases for Pydantic validation.

This module centralizes all core corporate data types, categorical literals,
and numerical metrics used across the pipeline layers. By establishing these
Annotated types as a single source of truth, it enforces strict type safety
and structural consistency during database seeding, API data ingestion,
and machine learning feature validation.

"""

import datetime
from decimal import Decimal
from typing import Annotated, Literal, Optional

from pydantic import Field

# ==============================================================================
# 1. ID Fields
# ==============================================================================

CompanyId = Annotated[
    int,
    Field(
        ...,
        gt=0,
        description="Unique database identifier for companies"
        "(this is also the primary key of financial_statements table).",
    ),
]
ContractId = Annotated[
    int,
    Field(
        ...,
        gt=0,
        description="Unique database identifier for energy contracts"
        "(primary key of energy_contracts table).",
    ),
]
InvoiceId = Annotated[
    int,
    Field(
        ...,
        gt=0,
        description="Unique database identifier for invoices"
        "(primary key of invoices table).",
    ),
]
PaymentId = Annotated[
    int,
    Field(
        ...,
        gt=0,
        description="Unique database identifier for transaction payments"
        "(primary key of payments table).",
    ),
]
TicketId = Annotated[
    int,
    Field(
        ...,
        gt=0,
        description="Unique database identifier for CRM support tickets"
        "(primary key of crm_support_tickets table).",
    ),
]
LoginId = Annotated[
    int,
    Field(
        ...,
        gt=0,
        description="Unique database identifier for user web logins"
        "(primary key of user_web_logins table).",
    ),
]
UserId = Annotated[
    int,
    Field(..., gt=0, description="Unique identifier for the specific corporate user."),
]


# ==============================================================================
# 2. Temporal Fields
# ==============================================================================

FoundationDate = Annotated[
    datetime.date, Field(..., description="Date of the company's foundation.")
]
ContractActivationDate = Annotated[
    datetime.date,
    Field(
        ...,
        ge=datetime.date(2000, 1, 1),
        description="Activation date of the energy contract.",
    ),
]
ContractTerminationDate = Annotated[
    Optional[datetime.date],
    Field(description="Termination date of the energy contract."),
]
FiscalYear = Annotated[
    int,
    Field(
        ...,
        ge=2000,
        description="Reference fiscal year for the financial statement documentation.",
    ),
]
InvoiceIssueDate = Annotated[
    datetime.date,
    Field(
        ...,
        description="The date the utility invoice was emitted.",
    ),
]
InvoiceDueDate = Annotated[
    datetime.date,
    Field(
        ...,
        description="The final deadline date for invoice payment.",
    ),
]
PaymentDate = Annotated[
    datetime.date,
    Field(
        ...,
        description="The day the financial payment execution occurred.",
    ),
]
TicketCreationDateTime = Annotated[
    datetime.datetime,
    Field(
        ...,
        description="UTC Timestamp recording when the support event has been created.",
    ),
]
TicketResolutionDateTime = Annotated[
    Optional[datetime.datetime],
    Field(
        description="UTC Timestamp recording when the support event has been resolved.",
    ),
]
LoginTimestamp = Annotated[
    datetime.datetime,
    Field(
        ..., description="UTC Timestamp tracking user authentication session liveness."
    ),
]


# ==============================================================================
# 3. Monetary Fields
# ==============================================================================

FinancialAmount = Annotated[
    Decimal,
    Field(
        ...,
        max_digits=15,
        decimal_places=2,
        description="Monetary balance metric"
        "(allows negative boundaries for net losses).",
    ),
]
FinancialAmountGeZero = Annotated[
    Decimal,
    Field(
        ...,
        ge=0,
        max_digits=15,
        decimal_places=2,
        description="Monetary balance metrics strictly bound to non-negative domains.",
    ),
]
InvoiceAmount = Annotated[
    Decimal,
    Field(
        ...,
        ge=0,
        max_digits=15,
        decimal_places=2,
        description="Invoice monetary amount, allowing 0.00 for "
        "zero-consumption or perfect adjustments.",
    ),
]
PaymentAmount = Annotated[
    Decimal,
    Field(
        ...,
        gt=0,
        max_digits=15,
        decimal_places=2,
        description="Strictly positive financial settlement amount paid.",
    ),
]


# ==============================================================================
# 4. "company" Table Specific Fields
# ==============================================================================

VatNumber = Annotated[
    str,
    Field(
        ...,
        # min and max length are present for debug reasons and for
        # how Pydantic throws errors
        min_length=11,
        max_length=11,
        pattern=r"^\d{11}$",
        description="VAT number of the company.",
    ),
]
LegalName = Annotated[str, Field(..., description="Legal name of the company.")]
LegalForm = Annotated[
    str,
    Field(
        ...,
        max_length=50,
        description="Legal form of the company (e.g. S.p.A., s.r.l.,...).",
    ),
]
IndustrySectorType = Annotated[
    Literal[
        "manufacturing",
        "chemical_heavy_industry",
        "services",
        "commerce",
        "food_beverage",
        "agriculture",
        "construction",
        "tech",
        "hospitality",
        "healthcare",
        "transportation",
        "utilities",
    ],
    Field(
        ...,
        description="Macro industrial sector of the corporate customer for profiling.",
    ),
]
OfficeRegion = Annotated[
    Optional[str],
    Field(
        max_length=100, description="Region where the company's offices are located."
    ),
]
IsActiveFlag = Annotated[
    bool,
    Field(
        ...,
        description="Boolean flag indicating whether the entity is"
        "currently active and operational.",
    ),
]


# ==============================================================================
# 5. "energy_contracts" Table Specific Fields
# ==============================================================================

CommodityType = Annotated[
    Literal["electricity", "gas"], Field(..., description="Type of commodity for sale.")
]
MarketType = Annotated[
    Literal["regulated", "deregulated"],
    Field(..., description="Type of market structure (regulated or deregulated)."),
]
VoltageLevel = Annotated[
    Optional[Literal["low", "medium", "high"]],
    Field(
        description="Voltage tier of the electricity contract"
        "(low, medium, or high voltage)."
    ),
]
PressureLevel = Annotated[
    Optional[Literal["low", "medium", "high"]],
    Field(description="Gas supply pressure tier (low, medium, or high pressure)"),
]
PowerKw = Annotated[
    Optional[Decimal],
    Field(
        gt=0,
        max_digits=10,
        decimal_places=2,
        description="Contracted power capacity specified in the contract,"
        "expressed in kilowatt (kW).",
    ),
]
GasMeterClass = Annotated[
    Optional[str],
    Field(
        pattern=r"^G\d+$",
        description="Gas meter nominal flow capacity class (e.g., G4, G6, G10).",
    ),
]
ContractStatus = Annotated[
    Literal["active", "suspended", "terminated"],
    Field(
        ...,
        description="Status of the energy contract (active, suspended or terminated).",
    ),
]


# ==============================================================================
# 6. "invoices" Table Specific Fields
# ==============================================================================

# CommodityType already defined for energy contracts
InvoiceNumber = Annotated[
    str,
    Field(
        ...,
        max_length=50,
        description="Unique commercial identification string of the invoice.",
    ),
]
ElectricityConsumption = Annotated[
    Optional[Decimal],
    Field(
        ge=0,
        max_digits=10,
        decimal_places=2,
        description="Actual energy consumption measured in kilowatt-hours (kWh).",
    ),
]
GasConsumption = Annotated[
    Optional[Decimal],
    Field(
        ge=0,
        max_digits=10,
        decimal_places=2,
        description="Actual gas volume consumed measured in "
        "Standard Cubic Meters (SMC).",
    ),
]
InvoiceStatus = Annotated[
    Literal["unpaid", "paid", "overdue", "cancelled"],
    Field(..., description="Accounting status of the generated utility invoice."),
]


# ==============================================================================
# 7. "payments" Table Specific Fields
# ==============================================================================

PaymentMethod = Annotated[
    Literal["direct_debit", "bank_transfer", "credit_card", "postal_bulletin", "cash"],
    Field(..., description="Financial settlement channel chosen by the customer."),
]
TransactionReference = Annotated[
    Optional[str],
    Field(
        max_length=100,
        description="Unique banking/transaction clearings reference string.",
    ),
]
PaymentStatus = Annotated[
    Literal["pending", "completed", "failed", "refunded"],
    Field(
        ...,
        max_length=20,
        description="Lifecycle state of the specific financial transaction.",
    ),
]


# ==============================================================================
# 8. "crm_support_tickets" Table Specific Fields
# ==============================================================================

TicketCategory = Annotated[
    Literal["billing", "technical", "onboarding", "commercial"],
    Field(..., description="CRM categorization of the inbound support request."),
]
SatisfactionScore = Annotated[
    Optional[int],
    Field(
        ge=1,
        le=5,
        description="Post-resolution customer feedback rating scale(1 to 5 stars).",
    ),
]


# ==============================================================================
# 9. "user_web_logins" Table Specific Fields
# ==============================================================================

IpAddress = Annotated[
    str,
    Field(
        ...,
        max_length=45,
        description="Client IPv4 or IPv6 address captured duringauthentication.",
    ),
]
DeviceType = Annotated[
    Literal["desktop", "mobile", "tablet", "api"],
    Field(
        ..., description="Hardware platform or interface channel used forweb logging."
    ),
]
