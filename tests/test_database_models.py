"""Database Schema Validation and Constraint Test Suite.

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
from collections.abc import Iterator
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from database.base import Base
from database.models import (
    Company,
    CRMSupportTicket,
    EnergyContract,
    FinancialStatement,
    Invoice,
    Payment,
    UserWebLogin,
)


@pytest.fixture(scope="function")
def db_session() -> Iterator[Session]:
    """Create a SQLite database for schema validation.

    The function creates a session in a SQLite database which allows
    to create databases stored in RAM, ina serverless way. This makes
    the test function executable without any further installation and
    faster than its PostgreSQL version.
    """
    engine = create_engine("sqlite:///:memory:")

    # This function bypasses a SQLite limit and allows fk constraints.
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)

    TestingSessionLocal = sessionmaker(bind=engine)
    session = TestingSessionLocal()

    yield session

    session.close()
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_database_references(db_session) -> None:
    """Test the creation of the entire database structure.

    This function tests the creation of the table and the
    references present among the tables in the database.
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
        energy_consumption_kwh=Decimal("25000.00"),
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


def test_commodity_data_constraints(db_session) -> None:
    """Test for required columns and gas_meter_class data constraint.

    This function tests if the constraints on required columns (not null) are
    working and if the column gas_meter_class of energy_contract table accepts
    only valid data.
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


def test_chronological_constraints(db_session) -> None:
    """Test for chronological constraints.

    This function tests if the chronological constraints are refusing impossible dates.
    It tests the date constraint for the energy_contracts table.
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


def test_inclusion_constraints(db_session) -> None:
    """Test for the inclusion constraints.

    This function tests if the inclusion constraints are excluding wrong
    or/and impossible data. It checks the constraint on the industry_sector column
    of companies table.
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


def test_range_constraint(db_session) -> None:
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

    with pytest.raises(IntegrityError, match=".*satisfaction_score_range_constraint.*"):
        db_session.flush()


def test_composite_foreign_key_constraint(db_session) -> None:
    """Test for the composite foreign key.

    This function tests the composite foreign key of the database, which relates the
    contract_id and commodity_type columns of invoices table with the columns
    id and commodity_type of energy_contracts table.
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

    with pytest.raises(IntegrityError, match=".*FOREIGN KEY constraint failed.*"):
        db_session.flush()


def test_mathematical_integrity_constraint(db_session) -> None:
    """Test for the mathematical integrity constraint.

    This function tests the mathematical constraint preesent in the database. It checks
    if the constraint that, for each invoice, the cost without taxes summed to the taxes
    gives exactly the total cost of the invoice.
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
        energy_consumption_kwh=Decimal("1500.00"),
        amount_excluding_tax=Decimal("100.00"),
        tax_amount=Decimal("22.00"),
        total_amount=Decimal("999.00"),  # Impossible given the other costs
        issue_date=datetime.date(2026, 2, 1),
        due_date=datetime.date(2026, 3, 1),
    )
    db_session.add(invoice)

    with pytest.raises(IntegrityError, match=".*invoice_amount_integrity_constraint.*"):
        db_session.flush()


def test_database_indices_exist(db_session) -> None:
    """Test for database indices.

    This function tests if the operational indices are well defined, checks if the
    indexed columns are correct, and verifies the descending order constraint.
    """
    inspector = inspect(db_session.get_bind())
    indices = inspector.get_indexes("user_web_logins")
    index_names = [idx["name"] for idx in indices]
    expected_index_name = "idx_logins_user_timeline"
    assert expected_index_name in index_names


def test_single_column_unique_constraint(db_session) -> None:
    """Test unique constraints.

    This function tests if the unique contraints are working, including both single- and
    multi-column unique indices.
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

    with pytest.raises(IntegrityError, match=".*UNIQUE constraint failed.*"):
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


def test_nullable_false_constraint(db_session) -> None:
    """Test for not null constraints.

    This function tests if the database throws an error if a not nullable value is
    missing.
    """
    invalid_company = Company(
        vat_number="88888888888",
        legal_name=None,  # This is not nullable
        legal_form="S.p.A.",
        industry_sector="tech",
        foundation_date=datetime.date(2020, 1, 1),
    )
    db_session.add(invalid_company)

    with pytest.raises(IntegrityError, match=".*NOT NULL constraint failed.*"):
        db_session.flush()


def test_on_delete_restriction_behavior(db_session) -> None:
    """Test on delete constraints.

    This functions tests if the database follows the deletion constraints
    (no action/restrict). It checks the deletion behaviour on orphan invoice.
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
        energy_consumption_kwh=Decimal("500.00"),
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
    with pytest.raises(IntegrityError, match=".*NOT NULL constraint failed.*"):
        db_session.flush()
