"""Database schema validation and relational constraint integration tests.

This module provides a comprehensive suite of integration tests to validate
the database schema, table relationships, and integrity constraints using pytest
and SQLAlchemy.

An isolated, in-memory SQLite database is initialized for each test function to
ensure fast, serverless, and independent execution without external database
dependencies (e.g., PostgreSQL). Foreign key constraints are explicitly enabled
via SQLite pragmas.

Tested Constraints & Features:
    * Entity Relationships: Verification of hierarchical data mapping
      from Companies down to Contracts, Invoices, Payments, Logins, and Tickets.
    * Conditional Column Requirements: Validation of conditional fields based on
      commodity type (e.g., electricity vs. gas specifications).
    * Chronological Integrity: Enforcement of valid date ranges (e.g., contract
      activation occurring before termination).
    * Domain Inclusion & Ranges: Allowed values for industry sectors and bounded
      scales for customer satisfaction scores.
    * Composite Foreign Keys: Strict matching across multi-column references.
    * Mathematical Integrity: Calculated database-level check constraints (e.g.,
      ensuring Total Amount equals Taxable Amount plus Tax).
    * Schema Definitions: Presence of database indices, unique constraints
      (single and multi-column), non-nullable column rules, and restricted
      deletion behaviors.

"""

import datetime
from decimal import Decimal

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database.models import (
    Company,
    CRMSupportTicket,
    EnergyContract,
    FinancialStatement,
    Invoice,
    Payment,
    UserWebLogin,
)


def test_database_references(db_session: Session) -> None:
    """Asserts relational mapping integrity across all schema entities.

    Seeds a complete, interdependent B2B operational graph (Company, Contract,
    Financial Statement, Invoice, Payment, Ticket, and Login records) ensuring that
    foreign key cascades, composite keys, and back-populations resolve without
    perturbation through SQLAlchemy ORM relationship navigation paths.

    Raises:
        AssertionError: If any structural relationship path fails to hydrate or
            deviates from the expected cardinality.

    """
    company = Company(
        vat_number="01234567890",
        legal_name="Acme Energy Consumer SPA",
        legal_form="SPA",
        industry_sector="manufacturing",
        foundation_date=datetime.date(2018, 4, 15),
        is_active=True,
    )
    db_session.add(company)
    db_session.flush()  # this forces sqlalchemy to insert the id autoincrementally

    contract = EnergyContract(
        company_id=company.id,
        commodity_type="electricity",
        market_type="deregulated",
        voltage_level="medium",
        power_committed_kw=Decimal("150.00"),
        activation_date=datetime.date(2024, 1, 1),
        contract_status="active",
    )
    db_session.add(contract)
    db_session.flush()

    financial_statement = FinancialStatement(
        company_id=company.id,
        fiscal_year=2025,
        total_revenue=Decimal("1500000.00"),
        net_income=Decimal("120000.00"),
        total_debt=Decimal("450000.00"),
        liquidity_cash=Decimal("300000.00"),
        share_capital=Decimal("100000.00"),
        ebitda=Decimal("180000.00"),
    )
    db_session.add(financial_statement)
    db_session.flush()

    invoice = Invoice(
        contract_id=contract.id,
        commodity_type="electricity",
        invoice_number="INV-2026-001",
        electricity_consumption_kwh=Decimal("25000.00"),
        amount_excluding_tax=Decimal("3510.39"),
        tax_amount=Decimal("990.11"),
        total_amount=Decimal("4500.50"),
        issue_date=datetime.date(2026, 6, 1),
        due_date=datetime.date(2026, 6, 30),
        invoice_status="unpaid",
    )
    db_session.add(invoice)
    db_session.flush()

    payment = Payment(
        invoice_id=invoice.id,
        payment_date=datetime.date(2026, 6, 15),
        amount_paid=Decimal("4500.50"),
        payment_method="direct_debit",
        transaction_reference="TXN-9988776655",
        payment_status="completed",
    )
    db_session.add(payment)
    db_session.flush()

    ticket_created_at = datetime.datetime(
        2024, 6, 20, 13, 40, 52, tzinfo=datetime.timezone.utc
    )
    ticket = CRMSupportTicket(
        company_id=company.id,
        ticket_category="billing",
        created_at=ticket_created_at,
        resolved_at=ticket_created_at + datetime.timedelta(hours=2),
        satisfaction_score=5,
    )
    db_session.add(ticket)
    db_session.flush()

    login = UserWebLogin(
        company_id=company.id,
        user_id=101,
        login_timestamp=datetime.datetime(
            2025, 3, 2, 9, 10, 13, tzinfo=datetime.timezone.utc
        ),
        ip_address="192.168.1.50",
        device_type="desktop",
    )
    db_session.add(login)
    db_session.flush()

    db_session.commit()

    queried_company = (
        db_session.query(Company).filter_by(vat_number="01234567890").first()
    )

    assert queried_company is not None
    assert len(queried_company.contracts) == 1
    assert len(queried_company.financial_statements) == 1
    assert len(queried_company.contracts[0].invoices) == 1
    assert len(queried_company.contracts[0].invoices[0].payments) == 1
    assert len(queried_company.support_tickets) == 1
    assert len(queried_company.logins) == 1


def test_commodity_data_constraints(db_session: Session) -> None:
    """Asserts commodity conditional data constraints.

    Validates that the database prevents the insertion of an energy contract when
    conditional required fields are omitted. Specifically, ensures that a 'gas'
    commodity type contract without a defined pressure level triggers a native
    `gas_fields_required_constraint` violation.

    Raises:
        AssertionError: If the database flushes the invalid contract configuration
            successfully without raising an `IntegrityError`.

    """
    company = Company(
        vat_number="11111111111",
        legal_name="Gas Class Tester SpA",
        legal_form="S.p.A.",
        industry_sector="utilities",
        foundation_date=datetime.date(2020, 1, 1),
    )
    db_session.add(company)
    db_session.flush()

    gas_contract = EnergyContract(
        company_id=company.id,
        commodity_type="gas",
        market_type="deregulated",
        activation_date=datetime.date(2026, 1, 1),
        pressure_level=None,  # required data
        gas_meter_class="G10",
    )
    db_session.add(gas_contract)

    with pytest.raises(IntegrityError, match=".*gas_fields_required_constraint.*"):
        db_session.flush()


def test_chronological_constraints(db_session: Session) -> None:
    """Asserts contract chronological date constraints.

    Validates chronological constraint enforcement on the `energy_contracts` table.
    Ensures that generating a record where the `termination_date` precedes the
    `activation_date` successfully triggers a native database-level
    `contract_date_chronology_constraint` violation.

    Raises:
        AssertionError: If the database flushes the invalid temporal sequence
            without raising an `IntegrityError`.

    """
    company = Company(
        vat_number="22222222222",
        legal_name="Time Paradox SRL",
        legal_form="s.r.l.",
        industry_sector="services",
        foundation_date=datetime.date(2021, 5, 10),
    )
    db_session.add(company)
    db_session.flush()

    energy_contract = EnergyContract(
        company_id=company.id,
        commodity_type="electricity",
        market_type="regulated",
        voltage_level="low",
        power_committed_kw=Decimal("15.00"),
        activation_date=datetime.date(2026, 6, 1),
        termination_date=datetime.date(2026, 5, 15),  # impossible date
    )
    db_session.add(energy_contract)
    with pytest.raises(IntegrityError, match=".*contract_date_chronology_constraint.*"):
        db_session.flush()


def test_inclusion_constraints(db_session: Session) -> None:
    """Asserts that the database rejects invalid categorical string variables.

    Validates inclusion constraint enforcement (`IN` clauses) on the `companies`
    table. Verifies that attempting to register an enterprise with an unlisted
    macroeconomic activity sector (e.g., 'money_laundering') correctly triggers
    a native database-level `industry_sector_constraint` violation.

    Raises:
        AssertionError: If the database flushes the invalid industry string
            successfully without raising an `IntegrityError`.

    """
    company = Company(
        vat_number="66666666666",
        legal_name="Inclusion Tester SRL",
        legal_form="s.r.l.",
        industry_sector="money_laundering",  # clearly invalid
        foundation_date=datetime.date(2024, 2, 2),
    )
    db_session.add(company)

    with pytest.raises(IntegrityError, match=".*industry_sector_constraint.*"):
        db_session.flush()


def test_range_constraint(db_session: Session) -> None:
    """Test on range constraint.

    This function tests if the range constraint (BETWEEN) refuses wrong values for
    the the user satisfaction score value in the crm_support_tickets table.
    """
    company = Company(
        vat_number="33333333333",
        legal_name="CRM Care SRL",
        legal_form="s.r.l.",
        industry_sector="tech",
        foundation_date=datetime.date(2022, 3, 3),
    )
    db_session.add(company)
    db_session.flush()

    ticket = CRMSupportTicket(
        company_id=company.id,
        ticket_category="billing",
        created_at=datetime.datetime.now(datetime.timezone.utc),
        satisfaction_score=6,  # out-of-range score
    )
    db_session.add(ticket)

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_composite_foreign_key_constraint(db_session: Session) -> None:
    """Asserts composite foreign key matching across business domains.

    Validates that the database enforces strict entity alignment by linking the
    `contract_id` and `commodity_type` multi-columns of the `invoices` table to
    their parent definitions in the `energy_contracts` table. Verifies that mismatching
    the parent contract profile (e.g., billing gas data on an active electricity
    contract) correctly breaks integrity invariants, throwing a `IntegrityError`.

    Raises:
        AssertionError: If the database engine flushes the domain mismatch
            successfully without throwing a foreign key violation.

    """
    company = Company(
        vat_number="44444444444",
        legal_name="Mismatch Power SRL",
        legal_form="s.r.l.",
        industry_sector="manufacturing",
        foundation_date=datetime.date(2019, 11, 20),
    )
    db_session.add(company)
    db_session.flush()

    electricity_contract = EnergyContract(
        company_id=company.id,
        commodity_type="electricity",
        market_type="deregulated",
        voltage_level="medium",
        power_committed_kw=Decimal("100.00"),
        activation_date=datetime.date(2025, 1, 1),
    )
    db_session.add(electricity_contract)
    db_session.flush()

    invoice = Invoice(
        contract_id=electricity_contract.id,
        commodity_type="gas",  # impossible for an electricity contract
        invoice_number="INV-MISMATCH-2026",
        gas_consumption_scm=Decimal("450.00"),
        amount_excluding_tax=Decimal("100.00"),
        tax_amount=Decimal("22.00"),
        total_amount=Decimal("122.00"),
        issue_date=datetime.date(2026, 6, 1),
        due_date=datetime.date(2026, 6, 15),
    )
    db_session.add(invoice)

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_mathematical_integrity_constraint(db_session: Session) -> None:
    """Asserts database invoice arithmetic constraints.

    Validates financial algebraic balance on the `invoices` table. Ensures that
    attempting to record an invoice where the sum of `amount_excluding_tax` and
    `tax_amount` does not mathematically match the specified `total_amount`
    correctly triggers an operational `invoice_amount_integrity_constraint` failure.

    Raises:
        AssertionError: If the database engine persists the asymmetric financial amounts
            without throwing an accounting integrity exception.

    """
    company = Company(
        vat_number="55555555555",
        legal_name="Math Check SpA",
        legal_form="S.p.A.",
        industry_sector="manufacturing",
        foundation_date=datetime.date(2023, 1, 1),
    )
    db_session.add(company)
    db_session.flush()

    contract = EnergyContract(
        company_id=company.id,
        commodity_type="electricity",
        market_type="deregulated",
        voltage_level="low",
        power_committed_kw=Decimal("20.00"),
        activation_date=datetime.date(2026, 1, 1),
    )
    db_session.add(contract)
    db_session.flush()

    invoice = Invoice(
        contract_id=contract.id,
        commodity_type="electricity",
        invoice_number="INV-MATH-ERR",
        electricity_consumption_kwh=Decimal("1500.00"),
        amount_excluding_tax=Decimal("100.00"),
        tax_amount=Decimal("22.00"),
        total_amount=Decimal("999.00"),  # Impossible given the other costs
        issue_date=datetime.date(2026, 2, 1),
        due_date=datetime.date(2026, 3, 1),
    )
    db_session.add(invoice)

    with pytest.raises(IntegrityError, match=".*invoice_amount_integrity_constraint.*"):
        db_session.flush()


def test_database_indices_exist(db_session: Session) -> None:
    """Verifies the programmatic existence of database operational indices.

    Leverages the SQLAlchemy inspection utility to query the active database schema
    metadata for the `user_web_logins` table. Enforces that optimization structures,
    specifically the chronological reverse-ordered tracking index
    (`idx_logins_user_timeline`), are properly defined and deployed.

    Raises:
        AssertionError: If the expected index name is missing from the table's
            structural index registry.

    """
    inspector = inspect(db_session.get_bind())
    indices = inspector.get_indexes("user_web_logins")
    index_names = [idx["name"] for idx in indices]
    expected_index_name = "idx_logins_user_timeline"
    assert expected_index_name in index_names


def test_single_column_unique_constraint(db_session: Session) -> None:
    """Asserts schema unique constraint enforcement.

    Validates unique invariant enforcement across the schema on two layers:
    1. Single-column: Verifies that attempting to register two distinct companies
       with the same VAT number (`vat_number`) throws an `IntegrityError`.
    2. Entity identity: Verifies that forcing overlapping primary identifiers (`id`)
       on the `energy_contracts` table breaks structural unique constraints.

    Raises:
        AssertionError: If the database flushes duplicate unique fields or overlapping
            identities successfully without raising an `IntegrityError`.

    """
    company_1 = Company(
        vat_number="66666666666",
        legal_name="One SpA",
        legal_form="S.p.A.",
        industry_sector="tech",
        foundation_date=datetime.date(2020, 1, 1),
    )
    db_session.add(company_1)
    db_session.flush()

    company_2 = Company(
        vat_number="66666666666",  # same as "One SpA"
        legal_name="Second SRL",
        legal_form="s.r.l.",
        industry_sector="services",
        foundation_date=datetime.date(2022, 5, 5),
    )
    db_session.add(company_2)

    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()

    company_3 = Company(
        vat_number="77777777777",
        legal_name="Energy Client SRL",
        legal_form="s.r.l.",
        industry_sector="manufacturing",
        foundation_date=datetime.date(2018, 10, 10),
    )
    db_session.add(company_3)
    db_session.flush()

    contract_1 = EnergyContract(
        id=100,
        company_id=company_3.id,
        commodity_type="electricity",
        market_type="deregulated",
        voltage_level="low",
        power_committed_kw=Decimal("15.00"),
        activation_date=datetime.date(2026, 1, 1),
    )
    db_session.add(contract_1)
    db_session.flush()

    # suppress the warning for same ids which should be (auto)incremental
    db_session.expunge(contract_1)

    contract_2 = EnergyContract(
        id=100,  # same id
        company_id=company_3.id,
        commodity_type="electricity",  # same commodity
        market_type="regulated",
        voltage_level="low",
        power_committed_kw=Decimal("10.00"),
        activation_date=datetime.date(2026, 2, 1),
    )
    db_session.add(contract_2)

    with pytest.raises(IntegrityError, match=".*UNIQUE constraint failed.*"):
        db_session.flush()


def test_nullable_false_constraint(db_session: Session) -> None:
    """Asserts that non-nullable columns trigger integrity errors when omitted.

    Validates the strict enforcement of `nullable=False` constraints across the schema.
    Verifies that attempting to persist an entity missing an obligatory attribute
    (e.g., a `Company` record with `legal_name=None`) is correctly intercepted by
    the database engine, raising a native `IntegrityError`.

    Raises:
        AssertionError: If the database engine flushes the incomplete entity
            successfully without throwing a NOT NULL violation.

    """
    invalid_company = Company(
        vat_number="88888888888",
        legal_name=None,  # This is not nullable
        legal_form="S.p.A.",
        industry_sector="tech",
        foundation_date=datetime.date(2020, 1, 1),
    )
    db_session.add(invalid_company)

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_on_delete_restriction_behavior(db_session: Session) -> None:
    """Asserts that relational delete constraints protect child records from orphanhood.

    Validates cascading behavioral invariants across linked tables. Verifies that
    attempting to delete an active `EnergyContract` record while it still possesses
    associated child `Invoice` entities forces an integrity violation, as the system
    blocks the contract identity from becoming null on historical ledger records.

    Raises:
        AssertionError: If the database engine drops the parent entity successfully
            without raising a NOT NULL foreign key constraint error.

    """
    company = Company(
        vat_number="99999999999",
        legal_name="Anti Delete SpA",
        legal_form="S.p.A.",
        industry_sector="utilities",
        foundation_date=datetime.date(2019, 1, 1),
    )
    db_session.add(company)
    db_session.flush()

    contract = EnergyContract(
        company_id=company.id,
        commodity_type="electricity",
        market_type="deregulated",
        voltage_level="low",
        power_committed_kw=Decimal("30.00"),
        activation_date=datetime.date(2025, 1, 1),
    )
    db_session.add(contract)
    db_session.flush()

    invoice = Invoice(
        contract_id=contract.id,
        commodity_type="electricity",
        invoice_number="INV-PREVENT-DEL",
        electricity_consumption_kwh=Decimal("500.00"),
        amount_excluding_tax=Decimal("200.00"),
        tax_amount=Decimal("44.00"),
        total_amount=Decimal("244.00"),
        issue_date=datetime.date(2025, 2, 1),
        due_date=datetime.date(2025, 3, 1),
    )
    db_session.add(invoice)
    db_session.flush()

    db_session.delete(contract)

    # We should have a not null constraint failed for the invoice contract_id becoming
    # null with the contract deletion
    with pytest.raises(IntegrityError):
        db_session.flush()
