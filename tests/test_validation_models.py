"""Validation tests for the Pydantic data schemas.

This module contains the comprehensive test suite for verifying the integrity,
constraints, and business logic enforced by the Pydantic creation models. It covers
both positive verification (happy paths) and negative validation failures (sad paths)
across various operational, financial, and corporate data structures.

The test suite systematically ensures the following model behaviors:
    - CompanyCreate: Enforces chronological validity of historical foundation dates.
    - EnergyContractCreate: Enforces commodity isolation (gas vs. electricity metrics)
      and logical timeline sequences between activation and termination.
    - FinancialStatementCreate: Verifies accurate mapping of corporate financial
      profiles.
    - InvoiceCreate: Guarantees strict mathematical integrity between line items and
      totals, due dates, and distinct commodity-specific billing records.
    - PaymentCreate: Rejects transaction records post-dated into the future.
    - CRMSupportTicketCreate: Controls chronological validity of support interactions
      and ensures satisfaction scores are blocked on open tickets.
    - UserWebLoginCreate: Validates audit fields and timestamps for system access logs.

"""

import datetime

import pytest
from pydantic import ValidationError

from schemas.models_validation import (
    CompanyCreate,
    CRMSupportTicketCreate,
    EnergyContractCreate,
    FinancialStatementCreate,
    InvoiceCreate,
    PaymentCreate,
    UserWebLoginCreate,
)


@pytest.fixture
def valid_company() -> dict:
    """Provide a dictionary of valid configuration data for a company profile.

    This fixture returns a complete, structurally sound set of key-value pairs
    representing the legal and operational details of a corporate entity. It
    serves as the standard happy-path baseline data to simplify instantiation
    and avoid repetition across company-related validation and database seeding
    tests.

    Returns:
        dict: A dictionary containing valid company field names mapped to
        their respective mock values, legal forms, and datetime objects.

    """
    return {
        "vat_number": "11111111111",
        "legal_name": "Foundation S.p.A.",
        "legal_form": "S.p.A.",
        "foundation_date": datetime.date(2015, 2, 3),
        "industry_sector": "commerce",
        "registered_office_region": "Lazio",
        "is_active": True,
    }


def test_company_creation(valid_company) -> None:
    """Verify that a valid past foundation date passes company schema validation.

    This test ensures that the `CompanyCreate` Pydantic model successfully
    instantiates when provided with a chronologically correct historical date,
    confirming the happy path of the foundation date validator.

    Args:
        valid_company (dict): A pytest fixture providing a dictionary populated
            with valid company record data.

    """
    company = CompanyCreate(**valid_company)

    assert company.vat_number == "11111111111"
    assert company.legal_name == "Foundation S.p.A."
    assert company.legal_form == "S.p.A."
    assert company.foundation_date == datetime.date(2015, 2, 3)
    assert company.industry_sector == "commerce"
    assert company.registered_office_region == "Lazio"
    assert company.is_active


def test_future_foundation_date_rejected(valid_company) -> None:
    """Verify that a future foundation date correctly triggers a ValidationError.

    This test checks the negative scenario (sad path) for company validation. It
    dynamically computes a date three days into the future relative to the current
    UTC date, ensuring the model rejects the record with the expected error message.

    Args:
        valid_company (dict): A pytest fixture providing a dictionary populated
            with valid company record data.

    """
    today = datetime.datetime.now(tz=datetime.timezone.utc).date()
    future_date = today + datetime.timedelta(days=3)

    invalid_company = valid_company.copy()
    invalid_company["foundation_date"] = future_date

    with pytest.raises(
        ValidationError, match="Foundation date can not be in the future."
    ):
        CompanyCreate(**invalid_company)


@pytest.fixture
def valid_electricity_contract() -> dict:
    """Provide a dictionary of valid configuration data for an electricity contract.

    This fixture returns a complete, structurally sound set of key-value pairs
    representing the attributes of an electricity commodity contract. It serves
    as the standard happy-path baseline data to simplify instantiation and
    prevent code duplication across electricity-related unit tests.

    Returns:
        dict: A dictionary containing valid contract field names mapped to
        their respective mock values and datetime objects.

    """
    return {
        "company_id": 1,
        "commodity_type": "electricity",
        "market_type": "regulated",
        "voltage_level": "low",
        "power_committed_kw": 10.00,
        "activation_date": datetime.date(2023, 5, 6),
        "termination_date": datetime.date(2026, 4, 27),
        "contract_status": "terminated",
    }


@pytest.fixture
def valid_gas_contract() -> dict:
    """Provide a dictionary of valid configuration data for a gas contract.

    This fixture returns a complete, structurally sound set of key-value pairs
    representing the attributes of a gas commodity contract. It serves as the
    standard happy-path baseline data to simplify instantiation and prevent code
    duplication across gas-related unit tests.

    Returns:
        dict: A dictionary containing valid contract field names mapped to
        their respective mock values and datetime objects.

    """
    return {
        "company_id": 1,
        "commodity_type": "gas",
        "market_type": "regulated",
        "pressure_level": "low",
        "gas_meter_class": "G6",
        "activation_date": datetime.date(2023, 5, 6),
        "termination_date": datetime.date(2026, 4, 27),
        "contract_status": "terminated",
    }


def test_electricity_contract_creation(valid_electricity_contract) -> None:
    """Verify successful validation of an electricity energy contract schema.

    This unit test ensures that the `EnergyContractCreate` Pydantic model correctly
    instantiates when provided with valid parameters specific to the electricity
    commodity type. It validates that commodity-specific fields are correctly mapped
    and that irrelevant fields (such as `pressure_level`, which belongs to gas
    contracts) remain unpopulated or evaluate to False.

    Args:
        valid_electricity_contract (dict): A pytest fixture providing a dictionary
            populated with valid electricity energy contract data.

    """
    electricity_contract = EnergyContractCreate(**valid_electricity_contract)

    assert electricity_contract.company_id == 1
    assert electricity_contract.commodity_type == "electricity"
    assert electricity_contract.market_type == "regulated"
    assert electricity_contract.voltage_level == "low"
    assert electricity_contract.power_committed_kw == 10.00
    assert not electricity_contract.pressure_level
    assert not electricity_contract.gas_meter_class
    assert electricity_contract.activation_date == datetime.date(2023, 5, 6)
    assert electricity_contract.termination_date == datetime.date(2026, 4, 27)
    assert electricity_contract.contract_status == "terminated"


def test_electricity_contract_with_gas_metrics_failure(
    valid_electricity_contract,
) -> None:
    """Ensure validation fails if an electricity contract contains gas metrics.

    This negative unit test (sad path) verifies that the `EnergyContractCreate`
    Pydantic model correctly enforces commodity separation by rejecting records
    that cross-contaminate an electricity profile with gas-specific attributes
    (such as pressure levels or gas meter classes).

    Args:
        valid_electricity_contract (dict): A pytest fixture providing a dictionary
            populated with valid electricity energy contract data.

    """
    invalid_contract = valid_electricity_contract.copy()
    invalid_contract["pressure_level"] = "low"
    invalid_contract["gas_meter_class"] = "G6"

    with pytest.raises(
        ValidationError, match="An electricity contract can not specify gas metrics."
    ):
        EnergyContractCreate(**invalid_contract)


def test_electricity_contract_without_required_electricity_metrics_failure(
    valid_electricity_contract,
) -> None:
    """Ensure validation fails if an electricity contract misses required power metrics.

    This negative unit test (sad path) verifies that the `EnergyContractCreate`
    Pydantic model correctly enforces data completeness for electricity profiles
    by rejecting records that provide a voltage level but omit the mandatory
    committed power capacity.

    Args:
        valid_electricity_contract (dict): A pytest fixture providing a dictionary
            populated with valid electricity energy contract data.

    """
    invalid_contract = valid_electricity_contract.copy()
    invalid_contract["power_committed_kw"] = None

    with pytest.raises(
        ValidationError,
        match="An electricity contract must specify both voltage level "
        "and electric power.",
    ):
        EnergyContractCreate(**invalid_contract)


def test_gas_contract_creation(valid_gas_contract: dict) -> None:
    """Verify successful validation of a gas energy contract schema.

    This unit test ensures that the `EnergyContractCreate` Pydantic model correctly
    instantiates when provided with valid parameters specific to the gas
    commodity type. It validates that gas-specific attributes (such as `pressure_level`
    and `gas_meter_class`) are properly mapped, and checks that electricity-specific
    fields (like `voltage_level` and `power_committed_kw`) remain unpopulated or
    evaluate to False.

    Args:
        valid_gas_contract (dict): A pytest fixture providing a dictionary
            populated with valid gas energy contract data.

    """
    gas_contract = EnergyContractCreate(**valid_gas_contract)

    assert gas_contract.company_id == 1
    assert gas_contract.commodity_type == "gas"
    assert gas_contract.market_type == "regulated"
    assert gas_contract.pressure_level == "low"
    assert gas_contract.gas_meter_class == "G6"
    assert not gas_contract.voltage_level
    assert not gas_contract.power_committed_kw
    assert gas_contract.activation_date == datetime.date(2023, 5, 6)
    assert gas_contract.termination_date == datetime.date(2026, 4, 27)
    assert gas_contract.contract_status == "terminated"


def test_gas_contract_with_electricity_metrics(valid_gas_contract) -> None:
    """Ensure validation fails if a gas contract contains electricity metrics.

    This negative unit test (sad path) verifies that the `EnergyContractCreate`
    Pydantic model correctly enforces commodity separation by rejecting records
    that cross-contaminate a gas profile with electricity-specific attributes
    (such as voltage levels or committed power capacity).

    Args:
        valid_gas_contract (dict): A pytest fixture providing a dictionary
            populated with valid gas energy contract data.

    """
    invalid_contract = valid_gas_contract.copy()
    invalid_contract["voltage_level"] = "medium"
    invalid_contract["power_committed_kw"] = 1000

    with pytest.raises(
        ValidationError, match="A gas contract can not specify electricity metrics."
    ):
        EnergyContractCreate(**invalid_contract)


def test_gas_contract_without_required_gas_metrics(valid_gas_contract) -> None:
    """Ensure validation fails if a gas contract misses required gas meter metrics.

    This negative unit test (sad path) verifies that the `EnergyContractCreate`
    Pydantic model correctly enforces data completeness for gas profiles by
    rejecting records that provide a pressure level but omit the mandatory
    gas meter class.

    Args:
        valid_gas_contract (dict): A pytest fixture providing a dictionary
            populated with valid gas energy contract data.

    """
    invalid_contract = valid_gas_contract.copy()
    invalid_contract["gas_meter_class"] = None

    with pytest.raises(
        ValidationError,
        match="A gas contract must specify both pressure level and gas meter class.",
    ):
        EnergyContractCreate(**invalid_contract)


def test_contract_termination_before_activation_failure(valid_gas_contract) -> None:
    """Ensure validation fails if the termination date precedes the activation date.

    This negative unit test verifies that the `EnergyContractCreate` Pydantic model
    correctly rejects contract initialization when the provided chronological
    timeline is impossible (i.e., when the termination date occurs before the
    activation date).

    Args:
        valid_gas_contract (dict): A pytest fixture providing a dictionary
            populated with valid gas energy contract data.

    """
    past_date = valid_gas_contract["activation_date"] - datetime.timedelta(days=3)
    invalid_contract = valid_gas_contract.copy()
    invalid_contract["termination_date"] = past_date

    with pytest.raises(
        ValidationError,
        match="The termination date of a contract can not be before "
        "its activation date.",
    ):
        EnergyContractCreate(**invalid_contract)


def test_missing_termination_date_on_terminated_contract(valid_gas_contract) -> None:
    """Ensure validation fails if a terminated contract lacks a termination date.

    This negative unit test verifies that the `EnergyContractCreate` Pydantic model
    correctly enforces data completeness by rejecting records where the status is
    set to 'terminated' but no corresponding termination date is provided.

    Args:
        valid_gas_contract (dict): A pytest fixture providing a dictionary
            populated with valid gas energy contract data.

    """
    invalid_contract = valid_gas_contract.copy()
    invalid_contract["contract_status"] = "terminated"
    invalid_contract["termination_date"] = None

    with pytest.raises(
        ValidationError,
        match="A terminated contract must specify its termination date.",
    ):
        EnergyContractCreate(**invalid_contract)


def test_valid_financial_statement() -> None:
    """Verify that a valid financial statement passes validation and maps correctly.

    This positive unit test ensures that the `FinancialStatementCreate` Pydantic
    model successfully instantiates when provided with realistic, logically
    consistent corporate financial metrics. It validates that all attributes
    are preserved with their correct values.

    """
    valid_financial_statement = {
        "company_id": 1,
        "fiscal_year": 2025,
        "total_revenue": 15500000.00,
        "net_income": 850000.00,
        "total_debt": 4200000.00,
        "liquidity_cash": 1200000.00,
        "share_capital": 500000.00,
        "ebitda": 2100000.00,
    }
    fin_stmt = FinancialStatementCreate(**valid_financial_statement)

    assert fin_stmt.company_id == 1
    assert fin_stmt.fiscal_year == 2025
    assert fin_stmt.total_revenue == 15500000.00
    assert fin_stmt.net_income == 850000.00
    assert fin_stmt.total_debt == 4200000.00
    assert fin_stmt.liquidity_cash == 1200000.00
    assert fin_stmt.share_capital == 500000.00
    assert fin_stmt.ebitda == 2100000.00


@pytest.fixture
def valid_electricity_invoice() -> dict:
    """Provide a dictionary of valid configuration data for an electricity invoice.

    This fixture returns a complete, structurally sound set of key-value pairs
    representing the billing details of an electricity commodity contract. It
    serves as the standard happy-path baseline data to simplify instantiation
    and prevent code duplication across electricity billing unit tests.

    Returns:
        dict: A dictionary containing valid invoice field names mapped to
        their respective mock values, financial amounts, and datetime objects.

    """
    return {
        "contract_id": 1,
        "commodity_type": "electricity",
        "invoice_number": "INV-00001",
        "electricity_consumption_kwh": 1000.00,
        "amount_excluding_tax": 230.00,
        "tax_amount": 70.00,
        "total_amount": 300.00,
        "issue_date": datetime.date(2025, 7, 1),
        "due_date": datetime.date(2025, 8, 1),
        "invoice_status": "paid",
    }


@pytest.fixture
def valid_gas_invoice() -> dict:
    """Provide a dictionary of valid configuration data for a gas invoice.

    This fixture returns a complete, structurally sound set of key-value pairs
    representing the billing details of a gas commodity contract.

    Returns:
        dict: A dictionary containing valid invoice field names mapped to
        their respective mock values, financial amounts, and datetime objects.

    """
    return {
        "contract_id": 1,
        "commodity_type": "gas",
        "invoice_number": "INV-00001",
        "gas_consumption_scm": 5.00,
        "amount_excluding_tax": 21.00,
        "tax_amount": 9.00,
        "total_amount": 30.00,
        "issue_date": datetime.date(2025, 7, 1),
        "due_date": datetime.date(2025, 8, 1),
        "invoice_status": "paid",
    }


def test_electricity_invoice_creation(valid_electricity_invoice) -> None:
    """Verify successful instantiation and field mapping of an electricity invoice.

    This positive unit test confirms that the `InvoiceCreate`
    Pydantic model correctly processes a valid electricity dataset. It ensures
    that attributes are properly assigned, data types are preserved, and
    gas-specific metrics remain unpopulated.

    Args:
        valid_electricity_invoice (dict): A pytest fixture providing a dictionary
            populated with valid electricity invoice record data.

    """
    electricity_invoice = InvoiceCreate(**valid_electricity_invoice)

    assert electricity_invoice.contract_id == 1
    assert electricity_invoice.commodity_type == "electricity"
    assert electricity_invoice.invoice_number == "INV-00001"
    assert electricity_invoice.electricity_consumption_kwh == 1000.00
    assert not electricity_invoice.gas_consumption_scm
    assert electricity_invoice.amount_excluding_tax == 230.00
    assert electricity_invoice.tax_amount == 70.00
    assert electricity_invoice.total_amount == 300.00
    assert electricity_invoice.issue_date == datetime.date(2025, 7, 1)
    assert electricity_invoice.due_date == datetime.date(2025, 8, 1)
    assert electricity_invoice.invoice_status == "paid"


def test_gas_invoice_creation(valid_gas_invoice) -> None:
    """Verify successful instantiation and field mapping of a gas invoice.

    This positive unit test (happy path) confirms that the `InvoiceCreate`
    Pydantic model correctly processes a valid gas dataset. It ensures
    that attributes are properly assigned, data types are preserved, and
    electricity-specific metrics remain unpopulated.

    Args:
        valid_gas_invoice (dict): A pytest fixture providing a dictionary
            populated with valid gas invoice record data.

    """
    gas_invoice = InvoiceCreate(**valid_gas_invoice)

    assert gas_invoice.contract_id == 1
    assert gas_invoice.commodity_type == "gas"
    assert gas_invoice.invoice_number == "INV-00001"
    assert not gas_invoice.electricity_consumption_kwh
    assert gas_invoice.gas_consumption_scm == 5.00
    assert gas_invoice.amount_excluding_tax == 21.00
    assert gas_invoice.tax_amount == 9.00
    assert gas_invoice.total_amount == 30.00
    assert gas_invoice.issue_date == datetime.date(2025, 7, 1)
    assert gas_invoice.due_date == datetime.date(2025, 8, 1)
    assert gas_invoice.invoice_status == "paid"


def test_electricity_invoice_without_electricity_consumption_failure(
    valid_electricity_invoice,
) -> None:
    """Ensure validation fails if an electricity invoice misses consumption data.

    This negative unit test (sad path) verifies that the `InvoiceCreate`
    Pydantic model correctly enforces data completeness for electricity
    billing by rejecting records that specify the electricity commodity type
    but omit the mandatory kilowatt-hour (kWh) consumption metrics.

    Args:
        valid_electricity_invoice (dict): A pytest fixture providing a dictionary
            populated with valid electricity invoice record data.

    """
    invalid_invoice = valid_electricity_invoice.copy()
    invalid_invoice["electricity_consumption_kwh"] = None

    with pytest.raises(
        ValidationError,
        match="An electricity invoice must specify electricity consumption.",
    ):
        InvoiceCreate(**invalid_invoice)


def test_electricity_invoice_with_gas_consumption_failure(
    valid_electricity_invoice,
) -> None:
    """Ensure validation fails if an electricity invoice contains gas consumption data.

    This negative unit test (sad path) verifies that the `InvoiceCreate`
    Pydantic model correctly enforces commodity isolation by rejecting records
    that attempt to mix billing metrics, specifically ensuring that an
    electricity invoice cannot contain gas-specific consumption fields.

    Args:
        valid_electricity_invoice (dict): A pytest fixture providing a dictionary
            populated with valid electricity invoice record data.

    """
    invalid_invoice = valid_electricity_invoice.copy()
    invalid_invoice["gas_consumption_scm"] = 5

    with pytest.raises(
        ValidationError,
        match="An electricity invoice must not specify gas consumption.",
    ):
        InvoiceCreate(**invalid_invoice)


def test_gas_invoice_without_gas_consumption_failure(valid_gas_invoice) -> None:
    """Ensure validation fails if a gas invoice misses consumption data.

    This negative unit test (sad path) verifies that the `InvoiceCreate`
    Pydantic model correctly enforces data completeness for gas billing
    by rejecting records that specify the gas commodity type but omit the
    mandatory Standard Cubic Meter (SCM) consumption metrics.

    Args:
        valid_gas_invoice (dict): A pytest fixture providing a dictionary
            populated with valid gas invoice record data.

    """
    invalid_invoice = valid_gas_invoice.copy()
    invalid_invoice["gas_consumption_scm"] = None

    with pytest.raises(
        ValidationError, match="A gas invoice must specify gas consumption."
    ):
        InvoiceCreate(**invalid_invoice)


def test_gas_invoice_with_electricity_consumption_failure(valid_gas_invoice) -> None:
    """Ensure validation fails if a gas invoice contains electricity metrics.

    This negative unit test (sad path) verifies that the `InvoiceCreate`
    Pydantic model correctly enforces commodity isolation by rejecting records
    that attempt to mix billing metrics, specifically ensuring that a gas
    invoice cannot specify electricity-specific kilowatt-hour (kWh) fields.

    Args:
        valid_gas_invoice (dict): A pytest fixture providing a dictionary
            populated with valid gas invoice record data.

    """
    invalid_invoice = valid_gas_invoice.copy()
    invalid_invoice["electricity_consumption_kwh"] = 1000

    with pytest.raises(
        ValidationError,
        match="A gas invoice must not contain information about "
        "electricity consumption.",
    ):
        InvoiceCreate(**invalid_invoice)


def test_mathematical_integrity_of_invoice_costs(valid_electricity_invoice):
    """Ensure validation fails if the invoice total does not equal the sum of costs.

    This negative unit test (sad path) verifies that the `InvoiceCreate`
    Pydantic model enforces strict mathematical integrity. It ensures that
    an invoice is rejected if the provided total amount does not precisely
    match the sum of the amount excluding tax and the tax amount.

    Args:
        valid_electricity_invoice (dict): A pytest fixture providing a dictionary
            populated with valid electricity invoice record data.

    """
    invalid_invoice = valid_electricity_invoice.copy()
    taxes = valid_electricity_invoice["tax_amount"]
    pure_energy_costs = valid_electricity_invoice["amount_excluding_tax"]
    wrong_total_amount = 1 + taxes + pure_energy_costs
    invalid_invoice["total_amount"] = wrong_total_amount

    with pytest.raises(
        ValidationError,
        match="Total costs of the invoice must be exactly equal to the "
        "sum of the taxes amount and the energy costs without taxes.",
    ):
        InvoiceCreate(**invalid_invoice)


def test_invoice_due_date_before_issue_date_failure(valid_electricity_invoice):
    """Ensure validation fails if the invoice due date predates the issue date.

    This negative unit test (sad path) verifies that the `InvoiceCreate`
    Pydantic model enforces chronological integrity. It ensures that an
    invoice is rejected if the deadline for payment (due date) is set to a
    day prior to when the invoice was actually generated (issue date).

    Args:
        valid_electricity_invoice (dict): A pytest fixture providing a dictionary
            populated with valid electricity invoice record data.

    """
    invalid_invoice = valid_electricity_invoice.copy()
    iss_date = valid_electricity_invoice["issue_date"]
    past_date = iss_date - datetime.timedelta(days=3)
    invalid_invoice["due_date"] = past_date

    with pytest.raises(
        ValidationError,
        match="Due date can not anticipate the issue date of the invoice.",
    ):
        InvoiceCreate(**invalid_invoice)


@pytest.fixture
def valid_payment() -> dict:
    """Provide a dictionary of valid configuration data for a payment record.

    This fixture returns a complete, structurally sound set of key-value pairs
    representing a successful billing transaction. It serves as the standard
    happy-path baseline data to simplify instantiation and avoid replication
    across payment validation and database seeding tests.

    Returns:
        dict: A dictionary containing valid payment field names mapped to
        their respective mock values, financial amounts, and datetime objects.

    """
    return {
        "invoice_id": 1,
        "payment_date": datetime.date(2025, 7, 28),
        "amount_paid": 300.00,
        "payment_method": "bank_transfer",
        "transaction_reference": "TX_9a8b7c6d5e",
        "payment_status": "completed",
    }


def test_payment_creation(valid_payment) -> None:
    """Verify successful instantiation and field mapping of a payment record.

    This positive unit test (happy path) confirms that the `PaymentCreate`
    Pydantic model correctly processes a valid payment dataset. It ensures
    that attributes—such as the invoice reference, financial amounts, transaction
    identifiers, and payment statuses—are properly assigned and maintain their
    intended data types.

    Args:
        valid_payment (dict): A pytest fixture providing a dictionary populated
            with valid payment transaction data.

    """
    payment = PaymentCreate(**valid_payment)

    assert payment.invoice_id == 1
    assert payment.payment_date == datetime.date(2025, 7, 28)
    assert payment.amount_paid == 300.00
    assert payment.payment_method == "bank_transfer"
    assert payment.transaction_reference == "TX_9a8b7c6d5e"
    assert payment.payment_status == "completed"


def test_future_payment_date_failure(valid_payment) -> None:
    """Ensure validation fails if the payment date is set in the future.

    This negative unit test (sad path) verifies that the `PaymentCreate`
    Pydantic model enforces chronological integrity by rejecting payment records
    with a future timestamp. It ensures that only processed, historical, or
    current-day transactions can be logged in the system.

    Args:
        valid_payment (dict): A pytest fixture providing a dictionary populated
            with valid payment transaction data.

    """
    today = datetime.datetime.now(tz=datetime.timezone.utc).date()
    future_date = today + datetime.timedelta(days=3)
    invalid_payment = valid_payment.copy()
    invalid_payment["payment_date"] = future_date

    with pytest.raises(ValidationError, match="Payment date can not be in the future."):
        PaymentCreate(**invalid_payment)


@pytest.fixture
def valid_ticket() -> dict:
    """Provide a dictionary of valid configuration data for a support ticket.

    This fixture returns a complete, structurally sound set of key-value pairs
    representing a customer support interaction. It serves as the standard
    happy-path baseline data to simplify instantiation and avoid repetition
    across ticketing validation, operational metric calculations, and database
    seeding tests.

    Returns:
        dict: A dictionary containing valid ticket field names mapped to
        their respective mock values, categories, timestamps, and satisfaction
        scores.

    """
    return {
        "company_id": 1,
        "ticket_category": "billing",
        "created_at": datetime.datetime(
            2025, 9, 3, 11, 30, 9, tzinfo=datetime.timezone.utc
        ),
        "resolved_at": datetime.datetime(
            2025, 9, 3, 12, 2, 56, tzinfo=datetime.timezone.utc
        ),
        "satisfaction_score": 4,
    }


def test_ticket_creation(valid_ticket) -> None:
    """Verify that a CRM support ticket is successfully created with valid data.

    This positive unit test ensures that `CRMSupportTicketCreate` correctly
    instantiates the model when provided with compliant fields. It validates
    that all attributes, including company identifiers, categories, behavioral
    scores, and UTC-standardized timestamps, are mapped accurately.

    Args:
        valid_ticket (dict): A pytest fixture providing a dictionary populated
            with valid CRM support ticket data.

    """
    ticket = CRMSupportTicketCreate(**valid_ticket)

    assert ticket.company_id == 1
    assert ticket.ticket_category == "billing"
    assert ticket.created_at == datetime.datetime(
        2025, 9, 3, 11, 30, 9, tzinfo=datetime.timezone.utc
    )
    assert ticket.resolved_at == datetime.datetime(
        2025, 9, 3, 12, 2, 56, tzinfo=datetime.timezone.utc
    )
    assert ticket.satisfaction_score == 4


def test_ticket_future_creation_date_failure(valid_ticket) -> None:
    """Verify that ticket instantiation fails if the creation date is in the future.

    This negative unit test ensures that `CRMSupportTicketCreate` raises a
    `ValidationError` when `created_at` is set to a future timestamp. It verifies
    that the system properly triggers the chronological integrity constraint and
    matches the specific error message.

    Args:
        valid_ticket (dict): A pytest fixture providing a dictionary populated
            with valid CRM support ticket data.

    """
    today = datetime.datetime.now(tz=datetime.timezone.utc)
    future_date = today + datetime.timedelta(days=3)
    invalid_ticket = valid_ticket.copy()
    invalid_ticket["created_at"] = future_date
    invalid_ticket["resolved_at"] = None

    with pytest.raises(
        ValidationError, match="Ticket creation date can not be in the future."
    ):
        CRMSupportTicketCreate(**invalid_ticket)


def test_ticket_creation_after_resolution_failure(valid_ticket) -> None:
    """Ensure validation fails if a ticket's resolution date precedes its creation date.

    This negative unit test (sad path) verifies that the `CRMSupportTicketCreate`
    Pydantic model enforces chronological integrity. It ensures that a support
    ticket is rejected if the closure timestamp (`resolved_at`) is set to a
    point in time prior to when the ticket was actually opened (`created_at`).

    Args:
        valid_ticket (dict): A pytest fixture providing a dictionary populated
            with valid CRM support ticket data.

    """
    creation_date = valid_ticket["created_at"]
    past_date = creation_date - datetime.timedelta(days=3)
    invalid_ticket = valid_ticket.copy()
    invalid_ticket["resolved_at"] = past_date

    with pytest.raises(
        ValidationError, match="Ticket can not be resolved before its creation."
    ):
        CRMSupportTicketCreate(**invalid_ticket)


def test_feedback_validation(valid_ticket) -> None:
    """Verify that a satisfaction score cannot be assigned to an open ticket.

    This negative unit test ensures that the `CRMSupportTicketCreate` Pydantic
    model enforces business logic constraints regarding customer feedback. It
    asserts that a `ValidationError` is raised if a ticket has a satisfaction
    score but lacks a `resolved_at` timestamp (meaning the ticket is still open).

    Args:
        valid_ticket (dict): A pytest fixture providing a dictionary populated
            with valid CRM support ticket data.

    """
    invalid_ticket = valid_ticket.copy()
    invalid_ticket["resolved_at"] = None

    with pytest.raises(
        ValidationError,
        match="Satisfaction score can not be assigned to an open ticket.",
    ):
        CRMSupportTicketCreate(**invalid_ticket)


@pytest.fixture
def valid_login() -> dict:
    """Provide a dictionary populated with valid user login audit data.

    This pytest fixture generates a realistic payload representing a successful
    user authentication event. It is designed to test the happy path of the
    login validation schemas and database model mapping.

    Returns:
        dict: A dictionary containing compliant user login record fields.

    """
    return {
        "company_id": 1,
        "user_id": 42,
        "login_timestamp": datetime.datetime(
            2026, 7, 6, 13, 15, 0, tzinfo=datetime.timezone.utc
        ),
        "ip_address": "192.168.1.45",
        "device_type": "desktop",
    }


def test_login_creation(valid_login) -> None:
    """Verify that a user web login event is successfully created with valid data.

    This positive unit test ensures that `UserWebLoginCreate` correctly
    instantiates the model when provided with compliant fields. It validates
    that all attributes, including company and user identifiers, network
    IP addresses, device classifications, and UTC-standardized login timestamps,
    are mapped accurately.

    Args:
        valid_login (dict): A pytest fixture providing a dictionary populated
            with valid user login audit data.

    """
    login = UserWebLoginCreate(**valid_login)

    assert login.company_id == 1
    assert login.user_id == 42
    assert login.login_timestamp == datetime.datetime(
        2026, 7, 6, 13, 15, 0, tzinfo=datetime.timezone.utc
    )
    assert login.ip_address == "192.168.1.45"
    assert login.device_type == "desktop"


def test_future_timestamp_login_failure(valid_login) -> None:
    """Ensure validation fails if the login timestamp is set in the future.

    This negative unit test (sad path) verifies that the `UserWebLoginCreate`
    Pydantic model enforces chronological integrity by rejecting login records
    with a future timestamp. It ensures that login authentication events cannot
    be logged ahead of the current systemic time.

    Args:
        valid_login (dict): A pytest fixture providing a dictionary populated
            with valid user login audit data.

    """
    now_ts = datetime.datetime.now(tz=datetime.timezone.utc)
    future_date = now_ts + datetime.timedelta(days=3)
    invalid_login = valid_login.copy()
    invalid_login["login_timestamp"] = future_date

    with pytest.raises(
        ValidationError, match="Login timestamp can not be in the future."
    ):
        UserWebLoginCreate(**invalid_login)
