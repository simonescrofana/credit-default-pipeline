"""Data Models Architecture Module for Credit Default Pipeline.

This module defines the Core OLTP (OnLine Transaction Processing) database schema
using SQLAlchemy 2.0 Declarative Mapping. The database architecture is designed
from the perspective of a large Utility Company providing electricity and gas services
to corporate clients (B2B).

The schema serves as the primary operational data store, capturing demographic,
financial, contract-level, and multi-layered behavioral data. The collected signals
form the historical and real-time basis for advanced alternative credit scoring
and automated corporate insolvency prediction (Machine Learning lifecycle).

Operational Perimeter and Table Topology:
    - Companies: Core registry capturing B2B customer demographics, entity state,
      and structural industry sectors.
    - Financial Statements: Official historical balance sheets mapping the traditional
      financial stability profile (revenue, leverage, liquidity) of the enterprises.
    - Energy Contracts: Technical and market dimensions of energy supply, tracking
      activation cycles, capacity parameters, and utility-specific meter classes.
    - Invoices: Transactional accounting records detailing continuous energy
      consumption, billing splits, and immediate aging balances.
    - Payments: High-granularity behavior tracking execution status, payment methods,
      and settlement failures (early-warning default indicators).
    - CRM Support Tickets: Non-financial behavioral telemetry representing customer
      friction, resolution times, and operational administration disputes.
    - User Web Logins: Appended digital engagement signals mapping customer operational
      liveness and real-time platform interaction patterns.

Design Patterns and Implementation Standards:
    - Strong typing using modern SQLAlchemy `Mapped[...]` and `mapped_column()`.
    - Domain constraints and logical cross-field dependencies enforced via native
      database-level CheckConstraints.
    - Multi-column optimized B-Tree indexing and compound constraints for
      high-throughput operational ingestion.
    - Advanced indexing patterns, including compound descending timelines on temporal
      vectors to optimize immediate application state retrieval.
    - Transaction-level integrity safety via DEFERRABLE INITIALLY IMMEDIATE foreign
      keys, allowing programmatic isolation during automated bulk-seeding operations.
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
    Numeric,
    String,
    text,
)
from sqlalchemy.dialects.postgresql import INET
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.base import Base


class Company(Base):
    """Table to store companies' data.

    This class specifies the 'companies' table, storing data about the companies.
    In particular, the table contains the columns: 'id' number identifying
    the companies, 'vat_number' associated to the company, 'legal_name' of
    the company, 'legal_form' (s.r.l., S.p.A.,...) of the company, 'industry_sector'
    of the company, 'foundation_date' of the company, 'registered_office_region'
    locating the company and 'is_active' column specifying if the company is
    still active today.
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
    """Table to store energy contracts' data.

    This class specifies the 'energy_contracts' table, storing data about
    the energy contracts. In particular, the table contains the columns:
    'id' number identifying the contract, 'company_id' associated to the company
    buying energy, 'commodity_type' specifying if the company is buying electricity
    or gas, 'market_type' that can be regulated or deregulated, 'voltage_level' and
    'pressure_level' dividing the companies in three groups based on the quantity
    of energy consumed, 'power_committed_kw' and 'gas_meter_class' specifying in
    detail the energy that the companies are going to consume, 'activation_date'
    of the contract, 'termination_date' of the contract (if it is terminated) and
    'contract_status' specifying the status of the contract.
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
    power_committed_kw: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2))
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
    """Table to store companies' financial data.

    This class specifies the 'financial_statements' table, storing data about
    the financial data of companies. In particular, the table contains the columns:
    'company_id' identifying the company, ' fiscal_year' specifying the reference year
    of the related data, 'total_revenue', 'net_income', 'total_debt', 'liquidity_cash'
    and 'share_capital' for the related company for the related fiscal year and
    'ebitda' (Earnings Before Interest, Taxes, Depreciation and Amortization) to measure
    financial performances.
    """

    __tablename__ = "financial_statements"

    company_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("companies.id", deferrable=True, initially="IMMEDIATE"),
        primary_key=True,
    )
    fiscal_year: Mapped[int] = mapped_column(Integer, primary_key=True)
    total_revenue: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    net_income: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    total_debt: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    liquidity_cash: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    share_capital: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    ebitda: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)

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
    """Table to store single invoices' data.

    This class specifies the 'invoices' table, storing data about
    the single energy invoices. In particular, the table contains the columns:
    'id' of the invoice, 'contract_id' which is contract related to the invoice,
    'commodity_type' indicating the type of energy (electricity, gas), 'invoice_number'
    which is the unique identification number of the invoice (not the id in the database
    which is the first column), 'energy_consumption_kwh' and 'gas_consumption_scm'
    reporting the actual energy consumed by the company for the period specified by
    the invoice, 'amount_excluding_tax', 'tax_amount' and 'total_amount' specifying
    the costs of the invoices, 'issue_date' and 'due_date' for the invoice and
    'invoice_status' indicating if the invoice is unpaid, paid, overdue or cancelled.
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
    energy_consumption_kwh: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2))
    gas_consumption_scm: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2))
    amount_excluding_tax: Mapped[Decimal] = mapped_column(
        Numeric(15, 2), nullable=False
    )
    tax_amount: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
    total_amount: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
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
            "total_amount = (amount_excluding_tax + tax_amount)",
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
            """(energy_consumption_kwh IS NULL) OR
            (energy_consumption_kwh >= 0)""",
            name="positive_electricity_consumption_constraint",
        ),
        CheckConstraint(
            """(gas_consumption_scm IS NULL) OR
            (gas_consumption_scm >= 0)""",
            name="positive_gas_consumption_constraint",
        ),
        CheckConstraint(
            """(commodity_type != 'electricity') OR
            (energy_consumption_kwh IS NOT NULL AND
            gas_consumption_scm IS NULL)""",
            name="electricity_consumption_integrity_constraint",
        ),
        CheckConstraint(
            """(commodity_type != 'gas') OR
            (gas_consumption_scm IS NOT NULL AND
            energy_consumption_kwh IS NULL)""",
            name="gas_consumption_integrity_constraint",
        ),
    )


class Payment(Base):
    """Table to store payments data.

    This class specifies the 'payments' table, storing data about payments.
    In particular, the table contains the columns: 'id' identifying the payment,
    'invoice_id' identifying the invoice related to the specific payment,
    'payment_date', 'amount_paid', 'payment_method' (direct debit, bank transfer,
    credit card, postal bulletin or cash), 'transaction_reference' for payment
    identification (not the 'id' which is related to the database) and
    'payment_status' (pending, completed, failed or refunded).
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
    amount_paid: Mapped[Decimal] = mapped_column(Numeric(15, 2), nullable=False)
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
    """Table to store CRM support tickets data.

    This class specifies the 'crm_support_tickets' table, storing data about
    tickets opened by the customer support. In particular, the table contains the
    columns: 'id' identifying the ticket, 'company_id' to specify which company opened
    the ticket, 'ticket_category' specifying the category of the ticket (billing,
    technical, onboarding or commercial), 'created_at' specifying the datetime at which
    the ticket has been opened, 'resolved_at' containing the datetime at which the
    ticket has been resolved (if it has already happened) and 'satisfaction_score'
    storing data about the user's feedback.
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
    """Table to store data about the users' web logins.

    This class specifies the 'user_web_logins' table, storing data about
    the web logins done from users. In particular, the table contains the columns:
    'id' identifying the login, 'company_id' specifying the company related to the
    user's login, 'user_id' and 'ip_address' identifying the user, 'login_timestamp'
    storing the datetime of the login and 'device_type' to store information about
    the device type used by the user to log in.
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
        )
    )
