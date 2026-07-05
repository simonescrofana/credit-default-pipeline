"""Pydantic execution schemas for multi-layer pipeline data validation.

This module centralizes the data contracts and structural validation schemas used
across database seeding, API ingestion layers, and service interfaces. By inheriting
from centralized domain types, these models enforce strict operational integrity
before data persists into the OLTP storage layer.

The schemas map one-to-one with the following relational database entities:
    * Companies
    * Energy Contracts
    * Financial Statements
    * Invoices
    * Payments
    * CRM Support Tickets
    * User Web Logins

"""

import datetime
from typing import Optional, Self

from pydantic import model_validator

from schemas.base import BaseResponseSchema
from schemas.types import (
    CommodityType,
    CompanyId,
    ContractActivationDate,
    ContractId,
    ContractStatus,
    ContractTerminationDate,
    DeviceType,
    ElectricityConsumption,
    FinancialAmount,
    FinancialAmountGeZero,
    FiscalYear,
    FoundationDate,
    GasConsumption,
    GasMeterClass,
    IndustrySectorName,
    InvoiceAmount,
    InvoiceDueDate,
    InvoiceId,
    InvoiceIssueDate,
    InvoiceNumber,
    InvoiceStatus,
    IpAddress,
    IsActiveFlag,
    LegalForm,
    LegalName,
    LoginId,
    LoginTimestamp,
    MarketType,
    OfficeRegion,
    PaymentAmount,
    PaymentDate,
    PaymentId,
    PaymentMethod,
    PaymentStatus,
    PowerKw,
    PressureLevel,
    SatisfactionScore,
    TicketCategory,
    TicketCreationDateTime,
    TicketId,
    TicketResolutionDateTime,
    TransactionReference,
    UserId,
    VatNumber,
    VoltageLevel,
)


class CompanyCreate(BaseResponseSchema):
    r"""Schema for validating the ingestion and creation of a new company record.

    Enforces data contracts and format constraints for corporate entities
    entering the relational OLTP database layer. This schema excludes the
    autoincrementing primary key.

    Attributes:
        vat_number (VatNumber): Unique 11-digit Italian VAT identification
            number (Regex: ^\d{11}$).
        legal_name (LegalName): Full registered commercial name of the company.
        legal_form (LegalForm): Legal structure of the company (e.g., S.p.A., s.r.l.).
        industry_sector (IndustrySectorName): Macro industrial sector literal
            used for profiling.
        foundation_date (FoundationDate): Official date of the company's foundation.
        registered_office_region (OfficeRegion): Administrative region where the
            company's headquarters are located.
        is_active (IsActiveFlag): Boolean flag indicating operational status.

    """

    vat_number: VatNumber
    legal_name: LegalName
    legal_form: LegalForm
    industry_sector: IndustrySectorName
    foundation_date: FoundationDate
    registered_office_region: Optional[OfficeRegion] = None
    is_active: IsActiveFlag = True

    @model_validator(mode="after")
    def foundation_date_constraint(self) -> Self:
        """Validate that the company foundation date resides in the past or present.

        This model-level validator checks the temporal consistency of the record
        against the current system date.

        Returns:
            Self: The validated instance of the model if constraints are met.

        Raises:
            ValueError: If the foundation date is set to a future date.

        """
        if self.foundation_date > datetime.date.today():
            raise ValueError("Foundation date can not be in the future.")

        return self


class CompanyResponse(CompanyCreate):
    """Schema for company database outputs and API response payloads.

    Inherits all core attributes from CompanyCreate and guarantees the inclusion
    of the database primary key.

    Attributes:
        id (CompanyId): Unique database identifier and autoincrementing primary key.

    """

    id: CompanyId


class EnergyContractCreate(BaseResponseSchema):
    """Schema for validating the ingestion and creation of an energy contract.

    This schema enforces domain constraints for corporate energy contracts,
    handling attributes for both electricity and gas commodities. It excludes
    the autoincrementing primary key.

    Attributes:
        company_id (CompanyId): Foreign key referencing the associated company.
        commodity_type (CommodityType): The type of utility ("electricity" or "gas").
        market_type (MarketType): Market structure ("regulated" or "deregulated").
        voltage_level (VoltageLevel): Voltage tier for electricity. Defaults to None.
        pressure_level (PressureLevel): Pressure tier for gas. Defaults to None.
        power_committed_kw (PowerKw): Contracted capacity in kW for electricity.
            Defaults to None.
        gas_meter_class (GasMeterClass): Nominal flow class (e.g. G4, G10, G25,...).
            Defaults to None.
        activation_date (ContractActivationDate): Contract start date (>= 2000-01-01).
        termination_date (ContractTerminationDate): Optional contract end date.
        contract_status (ContractStatus): Current state ("active", "suspended",
            "terminated").

    """

    company_id: CompanyId
    commodity_type: CommodityType
    market_type: MarketType
    voltage_level: Optional[VoltageLevel] = None
    pressure_level: Optional[PressureLevel] = None
    power_committed_kw: Optional[PowerKw] = None
    gas_meter_class: Optional[GasMeterClass] = None
    activation_date: ContractActivationDate
    termination_date: Optional[ContractTerminationDate] = None
    contract_status: ContractStatus = "active"

    @model_validator(mode="after")
    def electricity_contract_constraints(self) -> Self:
        """Validate that electricity contracts contain only required power metrics.

        Returns:
            Self: The validated instance of the model.

        Raises:
            ValueError: If an electricity contract is missing voltage or power data,
                or if it contains gas-specific technical configurations.

        """
        if self.commodity_type == "electricity" and (
            self.voltage_level is None or self.power_committed_kw is None
        ):
            raise ValueError(
                "An electricity contract must specify both voltage level "
                "and electric power."
            )

        if self.commodity_type == "electricity" and (
            self.pressure_level is not None or self.gas_meter_class is not None
        ):
            raise ValueError("An electricity contract can not specify gas metrics.")

        return self

    @model_validator(mode="after")
    def gas_contract_constraints(self) -> Self:
        """Validate that gas contracts contain only required gas metrics.

        Returns:
            Self: The validated instance of the model.

        Raises:
            ValueError: If a gas contract is missing pressure or meter class data,
                or if it contains electricity-specific technical configurations.

        """
        if self.commodity_type == "gas" and (
            self.pressure_level is None or self.gas_meter_class is None
        ):
            raise ValueError(
                "A gas contract must specify both pressure level and gas meter class."
            )

        if self.commodity_type == "gas" and (
            self.voltage_level is not None or self.power_committed_kw is not None
        ):
            raise ValueError("A gas contract can not specify electricity metrics.")

        return self

    @model_validator(mode="after")
    def termination_after_activation(self) -> Self:
        """Ensure the contract termination date does not precede its activation date.

        Returns:
            Self: The validated instance of the model.

        Raises:
            ValueError: If the termination date is chronologically prior to or
                equal to the activation date.

        """
        if self.termination_date and self.termination_date <= self.activation_date:
            raise ValueError(
                "The termination date of a contract can not be before "
                "its activation date."
            )

        return self

    @model_validator(mode="after")
    def termination_date_on_terminated_contract(self) -> Self:
        """Guarantees that terminated contracts explicitly provide a termination date.

        This mirrors the relational database's check constraint for lifecycle
        validation.

        Returns:
            Self: The validated instance of the model.

        Raises:
            ValueError: If the contract status is set to 'terminated' but the
                termination date is missing (None).

        """
        if self.contract_status == "terminated" and self.termination_date is not None:
            raise ValueError("A terminated contract must specify its termination date.")

        return self


class EnergyContractResponse(EnergyContractCreate):
    """Schema for energy contract database outputs and API responses.

    Inherits all core attributes from EnergyContractCreate and guarantees
    the inclusion of the database primary key.

    Attributes:
        id (ContractId): Unique database identifier and primary key.

    """

    id: ContractId


class FinancialStatementCreate(BaseResponseSchema):
    """Schema for validating corporate financial statement ingestion.

    Enforces data contracts and technical constraints for financial metrics
    stored in the database.

    Attributes:
        company_id (CompanyId): Shared primary key and foreign key referencing
            the associated company record.
        fiscal_year (FiscalYear): The reference accounting year for the financial
            statement data (>= 2000).
        total_revenue (FinancialAmountGeZero): Total gross revenue generated
            by the company (>= 0.00).
        net_income (FinancialAmount): Final corporate net profit or loss
            (allows negative values for losses).
        total_debt (FinancialAmountGeZero): Total financial debt obligations
            accumulated by the company (>= 0.00).
        liquidity_cash (FinancialAmountGeZero): Cash and liquid assets
            currently held (>= 0.00).
        share_capital (FinancialAmountGeZero): Total capital funded by
            shareholders or owners (>= 0.00).
        ebitda (FinancialAmount): Earnings Before Interest, Taxes, Depreciation,
            and Amortization (allows negative boundaries).

    """

    company_id: CompanyId
    fiscal_year: FiscalYear
    total_revenue: FinancialAmountGeZero
    net_income: FinancialAmount
    total_debt: FinancialAmountGeZero
    liquidity_cash: FinancialAmountGeZero
    share_capital: FinancialAmountGeZero
    ebitda: FinancialAmount


class FinancialStatementResponse(FinancialStatementCreate):
    """Schema for financial statement database outputs and API responses.

    Inherits all core attributes from FinancialStatementCreate to validate dat also
    in the database output.
    """

    pass


class InvoiceCreate(BaseResponseSchema):
    """Schema for validating utility invoice ingestion and billing data.

    Captures financial billing amounts and actual physical consumptions for energy
    risk profiling. It excludes the autoincrementing primary key.

    Attributes:
        contract_id (ContractId): Foreign key referencing the source energy contract.
        commodity_type (CommodityType): The type of utility ("electricity" or "gas").
        invoice_number (InvoiceNumber): Unique commercial identification string.
        electricity_consumption_kwh (ElectricityConsumption): Measured
            electricity (>= 0). Defaults to None.
        gas_consumption_smc (GasConsumption): Measured gas volume (>= 0).
            Defaults to None.
        amount_excluding_tax (InvoiceAmount): Net monetary amount excluding
            taxes (>= 0.00).
        tax_amount (InvoiceAmount): Total tax and duties applied to
            the invoice (>= 0.00).
        total_amount (InvoiceAmount): Total monetary amount due (>= 0.00).
        issue_date (InvoiceIssueDate): The date the invoice was generated and emitted.
        due_date (InvoiceDueDate): The final deadline date for payment settlement.
        invoice_status (InvoiceStatus): Lifecycle status ("unpaid", "paid",
            "overdue", "cancelled").

    """

    contract_id: ContractId
    commodity_type: CommodityType
    invoice_number: InvoiceNumber
    electricity_consumption_kwh: Optional[ElectricityConsumption] = None
    gas_consumption_smc: Optional[GasConsumption] = None
    amount_excluding_tax: InvoiceAmount
    tax_amount: InvoiceAmount
    total_amount: InvoiceAmount
    issue_date: InvoiceIssueDate
    due_date: InvoiceDueDate
    invoice_status: InvoiceStatus = "unpaid"

    @model_validator(mode="after")
    def electricity_invoice_constraints(self) -> Self:
        """Validate electricity invoices to ensure target metrics are isolated.

        Returns:
            Self: The validated instance of the model.

        Raises:
            ValueError: If the commodity is electricity but gas metrics are present,
                or if the mandatory electricity consumption value is missing.

        """
        if self.commodity_type == "electricity":
            if self.electricity_consumption_kwh is None:
                raise ValueError(
                    "An electricity invoice must specify electricity consumption."
                )
            if self.gas_consumption_smc is not None:
                raise ValueError(
                    "An electricity invoice must not specify gas consumption."
                )

        return self

    @model_validator(mode="after")
    def gas_invoice_constraints(self) -> Self:
        """Validate gas invoices to ensure target metrics are isolated.

        Returns:
            Self: The validated instance of the model.

        Raises:
            ValueError: If the commodity is gas but electricity metrics are present,
                or if the mandatory gas consumption value is missing.

        """
        if self.commodity_type == "gas":
            if self.gas_consumption_smc is None:
                raise ValueError("A gas invoice must specify gas consumption.")
            if self.electricity_consumption_kwh is not None:
                raise ValueError(
                    "A gas invoice must not contain information about "
                    "electricity consumption."
                )

        return self

    @model_validator(mode="after")
    def amount_integrity_validation(self) -> Self:
        """Verify mathematical consistency across invoice billing breakdown fields.

        Returns:
            Self: The validated instance of the model.

        Raises:
            ValueError: If the sum of amount_excluding_tax and tax_amount does not
                match the total_amount.

        """
        if (self.amount_excluding_tax + self.tax_amount) != self.total_amount:
            raise ValueError(
                "Total costs of the invoice must be exactly equal to the "
                "sum of the taxes amount and the energy costs without taxes."
            )

        return self

    @model_validator(mode="after")
    def due_date_after_issue_date(self) -> Self:
        """Ensure the payment due date is chronologically consistent with emission.

        Returns:
            Self: The validated instance of the model.

        Raises:
            ValueError: If the payment due date occurs before the invoice issue date.

        """
        if self.due_date < self.issue_date:
            raise ValueError(
                "Due date can not anticipate the issue date of the invoice."
            )

        return self


class InvoiceResponse(InvoiceCreate):
    """Schema for invoice database outputs and API responses.

    Inherits all core attributes from InvoiceCreate and guarantees
    the inclusion of the database primary key.

    Attributes:
        id (InvoiceId): Unique database identifier and primary key.

    """

    id: InvoiceId


class PaymentCreate(BaseResponseSchema):
    """Schema for validating financial transaction settlements.

    Enforces constraints on transaction clearing records and payment tracking.
    It excludes the autoincrementing primary key.

    Attributes:
        invoice_id (InvoiceId): Foreign key referencing the target utility invoice.
        payment_date (PaymentDate): The specific day the execution occurred.
        amount_paid (PaymentAmount): Strictly positive financial settlement
            amount (> 0).
        payment_method (PaymentMethod): Channel used
            ("direct_debit", "bank_transfer", etc.).
        transaction_reference (TransactionReference): Unique banking clearing string.
            Defaults to None.
        payment_status (PaymentStatus): Transaction state ("pending", "completed",
            "failed", "refunded").

    """

    invoice_id: InvoiceId
    payment_date: PaymentDate
    amount_paid: PaymentAmount
    payment_method: PaymentMethod
    transaction_reference: Optional[TransactionReference] = None
    payment_status: PaymentStatus = "completed"

    @model_validator(mode="after")
    def payment_date_constraint(self) -> Self:
        """Validate that the payment execution date does not reside in the future.

        This model-level validator checks the temporal consistency of the transaction
        against the current system date.

        Returns:
            Self: The validated instance of the model.

        Raises:
            ValueError: If the payment date is set to a future date.

        """
        if self.payment_date > datetime.date.today():
            raise ValueError("Payment date can not be in the future.")

        return self


class PaymentResponse(PaymentCreate):
    """Schema for payment database outputs and API responses.

    Inherits all core attributes from PaymentCreate and guarantees
    the inclusion of the database primary key.

    Attributes:
        id (PaymentId): Unique database identifier and primary key.

    """

    id: PaymentId


class CRMSupportTicketCreate(BaseResponseSchema):
    """Schema for validating customer relationship management (CRM) interactions.

    Tracks inbound corporate support logs, resolution timelines, and customer
    satisfaction metrics. It excludes the autoincrementing primary key.

    Attributes:
        company_id (CompanyId): Foreign key referencing the associated company.
        ticket_category (TicketCategory): Operational domain
            ("billing", "technical", etc.).
        created_at (TicketCreationDateTime): UTC Timestamp recording ticket creation.
        resolved_at (TicketResolutionDateTime): Optional UTC Timestamp recording
            resolution. Defaults to None.
        satisfaction_score (SatisfactionScore): Post-resolution rating (1 to 5 stars).
            Defaults to None.

    """

    company_id: CompanyId
    ticket_category: TicketCategory
    created_at: TicketCreationDateTime
    resolved_at: Optional[TicketResolutionDateTime] = None
    satisfaction_score: Optional[SatisfactionScore] = None

    @model_validator(mode="after")
    def creation_date_validation(self) -> Self:
        """Validate that the ticket creation timestamp does not reside in the future.

        Returns:
            Self: The validated instance of the model.

        Raises:
            ValueError: If the creation date is set to a future timestamp.

        """
        if self.created_at > datetime.datetime.now(tz=datetime.timezone.utc):
            raise ValueError("Ticket creation date can not be in the future.")

        return self

    @model_validator(mode="after")
    def resolution_after_creation(self) -> Self:
        """Ensure the ticket resolution time is chronologically after its creation.

        Returns:
            Self: The validated instance of the model.

        Raises:
            ValueError: If the resolution timestamp precedes the creation timestamp.

        """
        if self.resolved_at and self.resolved_at < self.created_at:
            raise ValueError("Ticket can not be resolved before its creation.")

        return self

    @model_validator(mode="after")
    def feedback_validation(self) -> Self:
        """Guarantee satisfaction scores are exclusively assigned to resolved tickets.

        This ensures consistency between the lifecycle state and the feedback presence.

        Returns:
            Self: The validated instance of the model.

        Raises:
            ValueError: If a satisfaction score is provided for an open ticket
                (where resolved_at is missing).

        """
        if self.satisfaction_score and not self.resolved_at:
            raise ValueError(
                "Satisfaction score can not be assigned to an open ticket."
            )

        return self


class CRMSupportTicketResponse(CRMSupportTicketCreate):
    """Schema for CRM ticket database outputs and API responses.

    Inherits all core attributes from CRMSupportTicketCreate and guarantees
    the inclusion of the database primary key.

    Attributes:
        id (TicketId): Unique database identifier and primary key.

    """

    id: TicketId


class UserWebLoginCreate(BaseResponseSchema):
    """Schema for validating user authentication session logs.

    Captures hardware platforms and network digital footprints for audit trails
    and behavioral liveness tracking. It excludes the autoincrementing primary key.

    Attributes:
        company_id (CompanyId): Foreign key referencing the associated company.
        user_id (UserId): Unique identifier for the specific corporate user.
        login_timestamp (LoginTimestamp): UTC Timestamp tracking
            authentication liveness.
        ip_address (IpAddress): Client IPv4 or IPv6 address (max length 45).
        device_type (DeviceType): Interface channel used ("desktop", "mobile", etc.).

    """

    company_id: CompanyId
    user_id: UserId
    login_timestamp: LoginTimestamp
    ip_address: IpAddress
    device_type: DeviceType

    @model_validator(mode="after")
    def login_timestamp_validation(self) -> Self:
        """Validate that the login attempt timestamp does not occur in the future.

        Returns:
            Self: The validated instance of the model.

        Raises:
            ValueError: If the login timestamp is set to a future temporal point.

        """
        if self.login_timestamp > datetime.datetime.now(tz=datetime.timezone.utc):
            raise ValueError("Login timestamp can not be in the future.")

        return self


class UserWebLoginResponse(UserWebLoginCreate):
    """Schema for user web login database outputs and API responses.

    Inherits all core attributes from UserWebLoginCreate and guarantees
    the inclusion of the database primary key.

    Attributes:
        id (LoginId): Unique database identifier and primary key.

    """

    id: LoginId
