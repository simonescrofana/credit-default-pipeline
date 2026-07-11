"""Data models architecture for the credit default prediction pipeline.

This module defines the core OLTP (OnLine Transaction Processing) database schema
using SQLAlchemy 2.0 Declarative Mapping. The database architecture models a
large utility company providing electricity and gas services to corporate
B2B clients.

The schema serves as the primary operational data store, capturing demographic,
financial, contract-level, and multi-layered behavioral telemetry. These signals
form the historical and real-time basis for alternative credit scoring and
automated corporate insolvency prediction within the machine learning lifecycle.

The operational perimeter encompasses seven core corporate data dimensions:
    Companies: Core registry capturing B2B customer demographics, operational
        states, and standardized economic activity sectors.
    Financial Statements: Annual balance sheets mapping traditional financial
        stability profiles including revenue, leverage, and liquidity metrics.
    Energy Contracts: Technical and market dimensions of energy supply tracking
        activation cycles, capacity parameters, and utility meter classes.
    Invoices: Transactional accounting records detailing continuous energy
        consumption, billing splits, and immediate aging balances.
    Payments: High-granularity transactional tracking mapping payment methods,
        settlement execution, and anomalies used as early default indicators.
    CRM Support Tickets: Non-financial behavioral telemetry capturing customer
        friction, resolution performance, and operational disputes.
    User Web Logins: Digital engagement signals mapping real-time corporate user
        liveness and platform interaction patterns.

Implementation Design Patterns:
    * Strong typing using modern SQLAlchemy `Mapped[...]` and `mapped_column()`.
    * Domain constraints and logical cross-field dependencies enforced via
      native database-level `CheckConstraints`.
    * Multi-column optimized B-Tree indexing and compound constraints for
      high-throughput operational ingestion.
    * Advanced indexing patterns, including compound descending timelines on
      temporal vectors to optimize application state retrieval.
    * Transaction-level integrity safety via `DEFERRABLE INITIALLY IMMEDIATE`
      foreign keys, isolating execution during automated bulk-seeding routines.

"""

import datetime
from decimal import Decimal
from typing import List, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    text,
)
from sqlalchemy.dialects.postgresql import INET
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.base import Base
from database.types import ExactNumeric, dialect_aware_check


class Company(Base):
    """Represents the 'companies' table within the database.

    This model stores core administrative and registration data for simulated
    corporate entities, ensuring industry sector integrity and valid historical
    foundation dates via database constraints.

    Attributes:
        id: Unique identifier for the company (BigInteger with SQLite fallback).
        vat_number: 11-character unique tax registration number.
        legal_name: Official registered name of the corporate entity.
        legal_form: Legal structure of the company (e.g., S.r.l., S.p.A.).
        industry_sector: Economic sector, constrained to pre-defined categories.
        foundation_date: Historical date when the company was established.
        registered_office_region: Geographical region of the registered office.
        is_active: Flag indicating whether the entity is currently operational.
        contracts: List of associated energy contracts.
        financial_statements: Historical financial records for the company.
        support_tickets: CRM support tickets raised by or linked to the entity.
        logins: Recorded web platform login events for company users.

    """

    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    vat_number: Mapped[str] = mapped_column(String(11), unique=True, nullable=False)
    legal_name: Mapped[str] = mapped_column(String, nullable=False)
    legal_form: Mapped[str] = mapped_column(String(50), nullable=False)
    industry_sector: Mapped[str] = mapped_column(String(30), nullable=False)
    foundation_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    registered_office_region: Mapped[Optional[str]] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default=text("true")
    )

    contracts: Mapped[List["EnergyContract"]] = relationship(back_populates="company")
    financial_statements: Mapped[List["FinancialStatement"]] = relationship(
        back_populates="company"
    )
    support_tickets: Mapped[List["CRMSupportTicket"]] = relationship(
        back_populates="company"
    )
    logins: Mapped[List["UserWebLogin"]] = relationship(back_populates="company")

    __table_args__ = (
        CheckConstraint(
            """industry_sector IN (
                'manufacturing',
                'chemical_heavy_industry',
                'services',
                'commerce',
                'food_beverage',
                'agriculture',
                'construction',
                'tech',
                'hospitality',
                'healthcare',
                'transportation',
                'utilities')""",
            name="industry_sector_constraint",
        ),
        CheckConstraint(
            "foundation_date <= CURRENT_DATE", name="past_foundation_date_constraint"
        ),
    )


class EnergyContract(Base):
    """Represents the 'energy_contracts' table within the database.

    This model manages the supply utility agreements (electricity or gas) linked
    to corporate entities. It enforces rigorous database-level constraints to Ensure
    data integrity across conditional fields specific to each commodity type, as well
    as contract states and chronological validity.

    Attributes:
        id: Unique identifier for the contract (BigInteger with SQLite fallback).
        company_id: Foreign key linking the contract to its associated company.
        commodity_type: Type of energy vehicle provided ('electricity' or 'gas').
        market_type: Regulatory environment governing the contract
            ('regulated' or 'deregulated').
        voltage_level: Supply voltage tier for electricity ('low', 'medium', 'high').
        pressure_level: Distribution pressure tier for gas ('low', 'medium', 'high').
        power_committed_kw: Contracted active power capacity allocation in kilowatts.
        gas_meter_class: Meter sizing classification prefix (e.g., 'G4', 'G10').
        activation_date: Effective calendar date when service delivery begins.
        termination_date: Optional closure date if the contract is no longer active.
        contract_status: Current operational standing ('active',
            'suspended', 'terminated').
        company: ORM relationship to the parent Company entity.
        invoices: ORM relationship to the historical list of associated Invoices.

    """

    __tablename__ = "energy_contracts"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    company_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("companies.id", deferrable=True, initially="IMMEDIATE"),
        nullable=False,
    )
    commodity_type: Mapped[str] = mapped_column(String(11), nullable=False)
    market_type: Mapped[str] = mapped_column(String(11), nullable=False)
    voltage_level: Mapped[Optional[str]] = mapped_column(String(6))
    pressure_level: Mapped[Optional[str]] = mapped_column(String(6))
    power_committed_kw: Mapped[Optional[Decimal]] = mapped_column(ExactNumeric(10, 2))
    gas_meter_class: Mapped[Optional[str]] = mapped_column(String(5))
    activation_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    termination_date: Mapped[Optional[datetime.date]] = mapped_column(Date)
    contract_status: Mapped[str] = mapped_column(
        String(10), default="active", server_default=text("'active'")
    )

    company: Mapped["Company"] = relationship(back_populates="contracts")
    invoices: Mapped[List["Invoice"]] = relationship(back_populates="contract")

    __table_args__ = (
        Index("idx_energy_contracts_company_id", "company_id"),
        Index("uq_energy_contracts_id_commodity", "id", "commodity_type", unique=True),
        CheckConstraint(
            """commodity_type IN (
                'electricity',
                'gas')""",
            name="commodity_type_constraint",
        ),
        CheckConstraint(
            """market_type IN (
                'regulated',
                'deregulated')""",
            name="market_type_constraint",
        ),
        CheckConstraint(
            """voltage_level IN (
                'low',
                'medium',
                'high')""",
            name="voltage_level_constraint",
        ),
        CheckConstraint(
            """pressure_level IN (
                'low',
                'medium',
                'high')""",
            name="pressure_level_constraint",
        ),
        CheckConstraint(
            """(power_committed_kw IS NULL) OR
            (power_committed_kw > 0)""",
            name="positive_power_committed_constraint",
        ),
        # The rigid version with regex expression will be imposed by Pydantic validation
        CheckConstraint("gas_meter_class LIKE 'G%'", name="gas_meter_class_constraint"),
        CheckConstraint(
            """(commodity_type != 'electricity') OR
            (voltage_level IS NOT NULL AND
            power_committed_kw IS NOT NULL)""",
            name="electricity_fields_required_constraint",
        ),
        CheckConstraint(
            """(commodity_type != 'gas') OR
            (pressure_level IS NOT NULL AND
            gas_meter_class IS NOT NULL)""",
            name="gas_fields_required_constraint",
        ),
        CheckConstraint(
            """(commodity_type != 'gas') OR
            (voltage_level IS NULL AND
            power_committed_kw IS NULL)""",
            name="gas_isolated_fields_constraint",
        ),
        CheckConstraint(
            """(commodity_type != 'electricity') OR
            (pressure_level IS NULL AND
            gas_meter_class IS NULL)""",
            name="electricity_isolated_fields_constraint",
        ),
        CheckConstraint(
            "termination_date > activation_date",
            name="contract_date_chronology_constraint",
        ),
        CheckConstraint(
            "activation_date >= '2000-01-01'", name="minimum_activation_date_constraint"
        ),
        CheckConstraint(
            """(contract_status != 'terminated') OR
            (termination_date IS NOT NULL)""",
            name="terminated_contract_date_required_constraint",
        ),
        CheckConstraint(
            """contract_status IN (
                'active',
                'suspended',
                'terminated')""",
            name="contract_status_constraint",
        ),
    )


class FinancialStatement(Base):
    """Represents the 'financial_statements' table within the database.

    This model stores annual corporate financial metrics and performance records
    for companies. It tracks core balance sheet and income statement indicators
    using a composite primary key consisting of the company identifier and the
    target fiscal year to maintain historical records.

    Attributes:
        company_id: Foreign key and composite primary key linking to the Company.
        fiscal_year: Calendar year of the financial reporting period
            (composite primary key).
        total_revenue: Gross top-line earnings generated during the fiscal year.
        net_income: Bottom-line profit or loss after all expenses, taxes, and interest.
        total_debt: Aggregate outstanding short-term and
            long-term financial liabilities.
        liquidity_cash: Total available cash reserves and liquid equivalents.
        share_capital: Total funding provided by shareholders/owners.
        ebitda: Earnings Before Interest, Taxes, Depreciation, and Amortization.
        company: ORM relationship to the parent Company entity.

    """

    __tablename__ = "financial_statements"

    company_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("companies.id", deferrable=True, initially="IMMEDIATE"),
        primary_key=True,
    )
    fiscal_year: Mapped[int] = mapped_column(Integer, primary_key=True)
    total_revenue: Mapped[Decimal] = mapped_column(ExactNumeric(15, 2), nullable=False)
    net_income: Mapped[Decimal] = mapped_column(ExactNumeric(15, 2), nullable=False)
    total_debt: Mapped[Decimal] = mapped_column(ExactNumeric(15, 2), nullable=False)
    liquidity_cash: Mapped[Decimal] = mapped_column(ExactNumeric(15, 2), nullable=False)
    share_capital: Mapped[Decimal] = mapped_column(ExactNumeric(15, 2), nullable=False)
    ebitda: Mapped[Decimal] = mapped_column(ExactNumeric(15, 2), nullable=False)

    company: Mapped["Company"] = relationship(back_populates="financial_statements")

    __table_args__ = (
        CheckConstraint("fiscal_year >= 2000", name="fiscal_year_constraint"),
        CheckConstraint("total_revenue >= 0", name="positive_revenue_constraint"),
        CheckConstraint("share_capital >= 0", name="positive_share_capital_constraint"),
        CheckConstraint("total_debt >= 0", name="positive_total_debt_constraint"),
        CheckConstraint(
            "liquidity_cash >= 0", name="positive_liquidity_cash_constraint"
        ),
    )


class Invoice(Base):
    """Represents the 'invoices' table within the database.

    This model manages individual billing statements issued under specific energy
    contracts. It enforces rigorous financial integrity through database-level total
    sum validations and strictly seals resource consumption metrics to match the linked
    contract's commodity vehicle type (electricity or gas).

    Attributes:
        id: Unique internal sequential identifier for the invoice (BigInteger).
        contract_id: Foreign key core constituent linking to the target contract.
        commodity_type: Foreign key validation tier classifying the utility vehicle.
        invoice_number: Alphanumeric unique billing sequence identifier.
        electricity_consumption_kwh: Active power usage during the billing cycle in kWh.
        gas_consumption_scm: Natural gas usage volume during the billing cycle
            in Standard Cubic Meters.
        amount_excluding_tax: Net base cost configuration before taxation.
        tax_amount: Calculated tax value application (e.g., VAT equivalent).
        total_amount: Gross billing total, constrained to equal net plus tax amount.
        issue_date: Documentation issuance calendar date.
        due_date: Contractual payment deadline date.
        invoice_status: Lifecycle standing ('unpaid', 'paid', 'overdue', 'cancelled').
        contract: ORM relationship to the parent EnergyContract entity.
        payments: ORM relationship tracking historical Payment entries for this invoice.

    """

    __tablename__ = "invoices"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    contract_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), nullable=False
    )
    commodity_type: Mapped[str] = mapped_column(String(11), nullable=False)
    invoice_number: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    electricity_consumption_kwh: Mapped[Optional[Decimal]] = mapped_column(
        ExactNumeric(10, 2)
    )
    gas_consumption_scm: Mapped[Optional[Decimal]] = mapped_column(ExactNumeric(10, 2))
    amount_excluding_tax: Mapped[Decimal] = mapped_column(
        ExactNumeric(15, 2), nullable=False
    )
    tax_amount: Mapped[Decimal] = mapped_column(ExactNumeric(15, 2), nullable=False)
    total_amount: Mapped[Decimal] = mapped_column(ExactNumeric(15, 2), nullable=False)
    issue_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    due_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    invoice_status: Mapped[str] = mapped_column(
        String(20), default="unpaid", server_default=text("'unpaid'")
    )

    contract: Mapped["EnergyContract"] = relationship(back_populates="invoices")
    payments: Mapped[List["Payment"]] = relationship(back_populates="invoice")

    __table_args__ = (
        ForeignKeyConstraint(
            ["contract_id", "commodity_type"],
            ["energy_contracts.id", "energy_contracts.commodity_type"],
            deferrable=True,
            initially="IMMEDIATE",
            name="commodity_type_composite_fk_constraint",
        ),
        Index("idx_invoices_fk_contract_commodity", "contract_id", "commodity_type"),
        Index("idx_invoices_operational_status", "invoice_status"),
        CheckConstraint(
            "due_date >= issue_date", name="invoice_date_chronology_constraint"
        ),
        CheckConstraint(
            dialect_aware_check(
                pg_expr="total_amount = (amount_excluding_tax + tax_amount)",
                sqlite_expr=(
                    "ABS(CAST(total_amount AS REAL) - "
                    "(CAST(amount_excluding_tax AS REAL) + "
                    "CAST(tax_amount AS REAL))) < 0.005"
                ),
            ),
            name="invoice_amount_integrity_constraint",
        ),
        CheckConstraint(
            """invoice_status IN (
                'unpaid',
                'paid',
                'overdue',
                'cancelled')""",
            name="invoice_status_constraint",
        ),
        CheckConstraint(
            """commodity_type IN (
                'electricity',
                'gas')""",
            name="invoice_commodity_type_constraint",
        ),
        CheckConstraint(
            "amount_excluding_tax >= 0", name="positive_invoice_amount_constraint"
        ),
        CheckConstraint("tax_amount >= 0", name="positive_tax_amount_constraint"),
        CheckConstraint(
            """(electricity_consumption_kwh IS NULL) OR
            (electricity_consumption_kwh >= 0)""",
            name="positive_electricity_consumption_constraint",
        ),
        CheckConstraint(
            """(gas_consumption_scm IS NULL) OR
            (gas_consumption_scm >= 0)""",
            name="positive_gas_consumption_constraint",
        ),
        CheckConstraint(
            """(commodity_type != 'electricity') OR
            (electricity_consumption_kwh IS NOT NULL AND
            gas_consumption_scm IS NULL)""",
            name="electricity_consumption_integrity_constraint",
        ),
        CheckConstraint(
            """(commodity_type != 'gas') OR
            (gas_consumption_scm IS NOT NULL AND
            electricity_consumption_kwh IS NULL)""",
            name="gas_consumption_integrity_constraint",
        ),
    )


class Payment(Base):
    """Represents the 'payments' table within the database.

    This model tracks financial transactions executed against issued invoices.
    It enforces strict database-level integration constraints regarding payment
    methods, transaction statuses, chronological compliance, and monetary amounts
    to guarantee accounting precision.

    Attributes:
        id: Unique internal sequential identifier for the payment record (BigInteger).
        invoice_id: Foreign key linking the transaction to its target Invoice.
        payment_date: Calendar date when the payment transaction occurred.
        amount_paid: Total monetary value processed, constrained
            to be strictly positive.
        payment_method: Transaction channel used ('direct_debit', 'bank_transfer',
            'credit_card', 'postal_bulletin', 'cash').
        transaction_reference: Unique external alphanumeric gateway identifier code.
        payment_status: Current transactional lifecycle state ('pending', 'completed',
            'failed', 'refunded').
        invoice: ORM relationship to the associated parent Invoice entity.

    """

    __tablename__ = "payments"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    invoice_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("invoices.id", deferrable=True, initially="IMMEDIATE"),
        nullable=False,
    )
    payment_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    amount_paid: Mapped[Decimal] = mapped_column(ExactNumeric(15, 2), nullable=False)
    payment_method: Mapped[str] = mapped_column(String(25), nullable=False)
    transaction_reference: Mapped[Optional[str]] = mapped_column(
        String(100), unique=True
    )
    payment_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="completed",
        server_default=text("'completed'"),
    )

    invoice: Mapped["Invoice"] = relationship(back_populates="payments")

    __table_args__ = (
        Index("idx_payments_invoice_id", "invoice_id"),
        CheckConstraint(
            """payment_method IN (
                'direct_debit',
                'bank_transfer',
                'credit_card',
                'postal_bulletin',
                'cash')""",
            name="payment_method_constraint",
        ),
        CheckConstraint(
            """payment_status IN (
                'pending',
                'completed',
                'failed',
                'refunded')""",
            name="payment_status_constraint",
        ),
        CheckConstraint("amount_paid > 0", name="positive_payment_amount_constraint"),
        CheckConstraint(
            "payment_date <= CURRENT_DATE", name="past_payment_date_constraint"
        ),
    )


class CRMSupportTicket(Base):
    """Represents the 'crm_support_tickets' table within the database.

    This model manages customer operations and assistance requests. It tracks
    ticket lifecycles from creation to resolution, enforcing conditional database
    constraints to ensure a satisfaction score can only be recorded for resolved
    incidents, alongside strict timeline progression.

    Attributes:
        id: Unique internal sequential identifier for the support ticket (BigInteger).
        company_id: Foreign key linking the ticket to the issuing Company entity.
        ticket_category: Nature of the request ('billing', 'technical',
            'onboarding', 'commercial').
        created_at: Datetime with timezone metadata capturing ticket initialization.
        resolved_at: Optional datetime with timezone capturing final resolution.
        satisfaction_score: Optional customer feedback rating,
            constrained between 1 and 5.
        company: ORM relationship to the associated parent Company entity.

    """

    __tablename__ = "crm_support_tickets"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    company_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("companies.id", deferrable=True, initially="IMMEDIATE"),
        nullable=False,
    )
    ticket_category: Mapped[str] = mapped_column(String(30), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        server_default=text("now()"),
    )
    resolved_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    satisfaction_score: Mapped[Optional[int]] = mapped_column(Integer)

    company: Mapped["Company"] = relationship(back_populates="support_tickets")

    __table_args__ = (
        Index("idx_crm_support_tickets_company_id", "company_id"),
        CheckConstraint(
            "resolved_at >= created_at", name="ticket_date_chronology_constraint"
        ),
        CheckConstraint(
            "satisfaction_score BETWEEN 1 AND 5",
            name="satisfaction_score_range_constraint",
        ),
        CheckConstraint(
            """(resolved_at IS NOT NULL) OR
            (satisfaction_score IS NULL)""",
            name="resolved_ticket_score_dependency_constraint",
        ),
        CheckConstraint(
            """ticket_category IN (
                'billing',
                'technical',
                'onboarding',
                'commercial')""",
            name="ticket_category_constraint",
        ),
        CheckConstraint(
            "created_at <= CURRENT_TIMESTAMP", name="past_ticket_creation_constraint"
        ),
    )


class UserWebLogin(Base):
    """Represents the 'user_web_logins' table within the database.

    This model functions as an audit trail for user authentication events on the
    web platform. It tracks login times, source network addresses, and access channels,
    utilizing custom indices to optimize chronological user timeline queries.

    Attributes:
        id: Unique internal sequential identifier for the login event (BigInteger).
        company_id: Foreign key linking the login event to the user's parent Company.
        user_id: Unique identifier for the specific user executing the authentication.
        login_timestamp: Datetime with timezone metadata capturing when the
            login occurred.
        ip_address: Network address of the client (INET type with a 45-character
            string fallback).
        device_type: Client classification medium used ('desktop', 'mobile',
            'tablet', 'api').
        company: ORM relationship to the associated parent Company entity.

    """

    __tablename__ = "user_web_logins"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    company_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("companies.id", deferrable=True, initially="IMMEDIATE"),
        nullable=False,
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), nullable=False
    )
    login_timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    ip_address: Mapped[str] = mapped_column(
        INET().with_variant(String(45), "sqlite"), nullable=False
    )
    device_type: Mapped[str] = mapped_column(String(20), nullable=False)

    company: Mapped["Company"] = relationship(back_populates="logins")

    __table_args__ = (
        Index("idx_user_web_logins_company_id", "company_id"),
        # this one is always ascendent in the drawn schema because dbdiagram.io does not
        # support the DESC keyword in the index
        Index("idx_logins_user_timeline", "user_id", text("login_timestamp DESC")),
        CheckConstraint(
            """device_type IN (
                'desktop',
                'mobile',
                'tablet',
                'api')""",
            name="device_type_constraint",
        ),
        CheckConstraint(
            "login_timestamp <= CURRENT_TIMESTAMP",
            name="past_login_timestamp_constraint",
        ),
    )
