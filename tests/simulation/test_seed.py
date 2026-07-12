"""Test suite for the corporate metadata and operational telemetry seeding engine.

Validates mathematical constraints, core stochastic samplers, macroeconomic
simulation pathways, and relational ORM persistence loops. It verifies nominal
multi-year corporate lifecycles as well as transactional error boundaries, including
pipeline rollbacks, future payment date clamping, and contract state transitions.

"""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from faker import Faker
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
from simulation.profiles import (
    MONTHS_PER_YEAR,
    SEASONALITY_PROFILE_CURVES,
    SECTOR_PROFILES,
)
from simulation.seed import (
    COMPANY_SIZE_WEIGHTS,
    SECTOR_WEIGHTS,
    generate_company_history,
    generate_consumption_series,
    generate_identity,
    persist_billing_and_telemetry,
    run_seeding,
    sigmoid,
    simulate_financial_year,
    simulate_operational_telemetry,
)


def test_seasonality_curves_match_months_in_year() -> None:
    """Assert that seasonality curves comply with structural invariants.

    Iterates through all configured profiles within the simulation matrix to
    validate that each curve contains exactly twelve monthly elements and that
    the cumulative sum of these elements precisely reflects the annual vector.

    """
    for name, curve in SEASONALITY_PROFILE_CURVES.items():
        assert len(curve) == MONTHS_PER_YEAR, (
            f"Curve '{name}' has {len(curve)} elements, expected {MONTHS_PER_YEAR}"
        )
        assert abs(sum(curve) - MONTHS_PER_YEAR) < 1e-6, (
            f"Curve '{name}' sums to {sum(curve)}, expected {float(MONTHS_PER_YEAR)}"
        )


def test_sector_weights_contraint() -> None:
    """Assert that industrial sector weights define a valid probability space.

    Validates that the cumulative sum of all values within the sector weight
    matrix equals exactly 1.0, ensuring consistent and bound-safe stochastic
    sampling during corporate initialization.

    """
    assert abs(sum(SECTOR_WEIGHTS.values()) - 1.0) < 1e-6, (
        f"SECTOR_WEIGHTS does not sum to 1.0: {sum(SECTOR_WEIGHTS.values())}"
    )


def test_company_size_weights_constraint() -> None:
    """Assert that corporate size weights define a valid probability space.

    Validates that the cumulative sum of all values within the company size
    weight matrix equals exactly 1.0, ensuring safe and predictable stochastic
    sampling during corporate scale distribution.

    """
    assert (
        abs(sum(COMPANY_SIZE_WEIGHTS.values()) - 1.0) < 1e-6
    ), f"COMPANY_SIZE_WEIGHTS doesn not sum to 1.0: {
        sum(COMPANY_SIZE_WEIGHTS.values())
    }"


@pytest.fixture
def fake() -> Faker:
    """Provide a localized Italian Faker instance for data synthesis.

    Returns:
        Faker: An initialized data generation instance tailored to the Italian
            corporate and regional naming conventions.

    """
    return Faker("it_IT")


@pytest.fixture
def rg() -> np.random.Generator:
    """Provide a deterministic random number generator for reproducible tests.

    Returns:
        np.random.Generator: A seed-bounded NumPy random generator instance
            ensuring consistent test executions.

    """
    return np.random.default_rng(42)


def test_sigmoid() -> None:
    """Validate the sigmoid function performance against boundary limits.

    Tests the mathematical properties of the standard logistic function at
    critical nodes, ensuring precise tracking at the center point and structural
    convergence near asymptotic thresholds.

    """
    assert sigmoid(0.0) == 0.5
    assert sigmoid(10.0) > 0.99
    assert sigmoid(-10.0) < 0.01


def test_generate_identity(rg: np.random.Generator, fake: Faker) -> None:
    """Validate schema integrity and types of synthesized corporate identities.

    Verifies that the generated metadata dictionary complies with the expected OLTP
    schema structures, checking critical constraints such as the Italian VAT character
    length, structural status flags, and categorical size boundaries.

    """
    info, size_mult, size = generate_identity(rg, fake)

    assert isinstance(info, dict)
    assert "vat_number" in info
    assert len(info["vat_number"]) == 11
    assert info["is_active"] is True
    assert size in ["small", "medium", "large"]
    assert isinstance(size_mult, float)


def test_generate_consumption_series(rg: np.random.Generator) -> None:
    """Validate volumetric and seasonality constraints for consumption profiles.

    Tests that the computed energy baselines for a target sector contain the expected
    commodity keys, enforce positive floating-point bounds, and correctly map the
    corresponding structural monthly curve configurations.

    """
    vols = generate_consumption_series(rg, "manufacturing", size_mult=1.0)

    assert "annual_electricity" in vols
    assert "annual_gas" in vols
    assert isinstance(vols["annual_electricity"], float)
    assert vols["annual_electricity"] > 0
    assert isinstance(vols["curves"]["electricity"], tuple)


def test_simulate_financial_year_healthy_case(rg: np.random.Generator) -> None:
    """Validate fiscal statement simulation under stable operating margins.

    Tests that a profitable entity correctly derives positive EBITDA balances,
    enforces tax-adjusted net income reductions, and generates compliant capital
    and cash distributions within realistic stochastic boundaries.

    """
    initial_margin = 0.15
    fin_data, next_margin = simulate_financial_year(
        rg,
        sector="tech",
        size="medium",
        total_energy=500.0,
        current_margin=initial_margin,
    )

    assert isinstance(next_margin, float)
    assert next_margin != initial_margin
    assert initial_margin - 0.035 <= next_margin <= initial_margin + 0.035
    assert -0.20 <= next_margin <= 0.40
    assert fin_data["ebitda"] > 0
    assert fin_data["net_income"] < fin_data["ebitda"]
    assert fin_data["liquidity_cash"] > 0


def test_simulate_financial_year_crisis_case(rg: np.random.Generator) -> None:
    """Validate fiscal statement simulation under negative operating margins.

    Tests that an enterprise in financial distress correctly generates negative
    EBITDA structures, bypasses corporate tax deductions on net losses, and triggers
    distressed balance sheet distributions with compressed cash and high leverage.

    """
    fin_data, next_margin = simulate_financial_year(
        rg,
        sector="construction",
        size="medium",
        total_energy=500.0,
        current_margin=-0.05,
    )

    assert next_margin >= -0.20
    assert fin_data["ebitda"] < 0
    assert fin_data["net_income"] == fin_data["ebitda"]

    revenue = float(fin_data["total_revenue"])
    assert float(fin_data["liquidity_cash"]) <= revenue * 0.015
    assert float(fin_data["total_debt"]) >= revenue * 0.35


def test_simulate_operational_telemetry_healthy() -> None:
    """Validate operational data generation under healthy financial conditions.

    Tests that an enterprise with solid cash reserves and active contracts
    consistently results in fully paid invoices, swift payment behaviors, compliant
    operational ticket lifecycle tracking, and stable user engagement telemetry.

    """
    contract_mock = {"commodity_type": "electricity", "contract_status": "active"}
    vols_mock = {"annual_electricity": 10.0, "curves": {"electricity": (1.0,) * 12}}

    fin_statement_mock = {
        "total_revenue": 500000.0,
        "total_debt": 0.0,
        "liquidity_cash": 100000.0,
        "ebitda": 150000.0,
    }
    sector = "tech"
    sector_prof = {"support_bias": "low", "digitalization_bias": "high"}

    all_tickets = []
    for seed in range(200):
        billing, tickets, logins = simulate_operational_telemetry(
            np.random.default_rng(seed=seed),
            2026,
            5,
            contract_mock,
            vols_mock,
            fin_statement_mock,
            sector,
            sector_prof,
        )
        all_tickets.extend(tickets)

    assert isinstance(billing, dict)
    assert billing["invoice_status"] == "paid"
    assert billing["payment_behavior"] == "completed"
    assert billing["pay_days"] is not None
    assert billing["pay_days"] <= 44
    assert billing["amount_excluding_tax"] > 0
    assert (
        billing["total_amount"]
        == billing["amount_excluding_tax"] + billing["tax_amount"]
    )

    assert isinstance(tickets, list)
    assert all_tickets
    for ticket in all_tickets:
        assert "ticket_category" in ticket
        assert ticket["ticket_category"] in ["billing", "technical", "commercial"]
        assert ticket["created_at"] is not None

        assert ticket["resolved_at"] is not None
        assert ticket["resolved_at"] >= ticket["created_at"]

        assert ticket["satisfaction_score"] is not None
        assert 3 <= ticket["satisfaction_score"] <= 5

    assert len(logins) > 0
    for log in logins:
        assert "ip_address" in log
        assert log["device_type"] in ["desktop", "mobile", "tablet"]


def test_simulate_operational_telemetry_terminated_contract(
    rg: np.random.Generator,
) -> None:
    """Validate telemetry bypass behaviors for terminated contracts.

    Tests that a corporate entity with a terminated status flag correctly silences
    all transactional operational loops, ensuring that no billing records, active
    support tickets, or user login events are generated.

    """
    contract_mock = {"commodity_type": "electricity", "contract_status": "terminated"}
    vols_mock = {"annual_electricity": 100.0, "curves": {"electricity": (1.0,) * 12}}
    fin_mock = {
        "total_revenue": 100000,
        "total_debt": 20000,
        "liquidity_cash": 5000,
        "ebitda": 10000,
    }

    billing, tickets, logins = simulate_operational_telemetry(
        rg, 2026, 5, contract_mock, vols_mock, fin_mock, "tech", SECTOR_PROFILES["tech"]
    )

    assert billing == {}
    assert tickets == []
    assert logins == []


def test_persist_billing_and_telemetry(db_session: Session, fake: Faker) -> None:
    """Validate mapping and database persistence of operational datasets.

    Tests that raw telemetry dictionaries representing historical invoicing, support
    tickets, and web interaction logs are correctly transformed into their corresponding
    SQLAlchemy ORM models, validating foreign key constraints and precise
    numeric conversions.

    """
    company = Company(
        vat_number="99999999999",
        legal_name="Test Persist",
        legal_form="S.r.l.",
        industry_sector="services",
        foundation_date=datetime.date(2024, 1, 1),
        registered_office_region="Lazio",
        is_active=True,
    )
    db_session.add(company)
    db_session.flush()

    contract = EnergyContract(
        company_id=company.id,
        commodity_type="electricity",
        market_type="deregulated",
        power_committed_kw=50.00,
        voltage_level="low",
        activation_date=company.foundation_date,
        contract_status="active",
    )
    db_session.add(contract)
    db_session.flush()

    billing_history = [
        {
            "month": 6,
            "consumption": 1250.5,
            "amount_excluding_tax": Decimal("312.63"),
            "tax_amount": Decimal("68.78"),
            "total_amount": Decimal("381.41"),
            "invoice_status": "paid",
            "payment_behavior": "completed",
            "pay_days": 5,
            "payment_method": "direct_debit",
            "issue_date": datetime.date(2026, 6, 1),
        }
    ]
    tickets = [
        {
            "ticket_category": "technical",
            "created_at": datetime.datetime(
                2026, 6, 5, 10, 0, tzinfo=datetime.timezone.utc
            ),
        }
    ]
    logins = [
        {
            "user_id": 101,
            "login_timestamp": datetime.datetime(
                2026, 6, 2, 22, 15, tzinfo=datetime.timezone.utc
            ),
            "ip_address": "192.168.1.50",
            "device_type": "desktop",
        }
    ]

    persist_billing_and_telemetry(
        db_session,
        company.id,
        contract.id,
        "electricity",
        billing_history,
        tickets,
        logins,
        fake,
    )

    assert db_session.query(Invoice).filter_by(contract_id=contract.id).count() == 1
    assert db_session.query(Payment).count() == 1
    assert (
        db_session.query(CRMSupportTicket).filter_by(company_id=company.id).count() == 1
    )
    assert db_session.query(UserWebLogin).filter_by(company_id=company.id).count() == 1

    saved_invoice = db_session.query(Invoice).first()
    assert saved_invoice.electricity_consumption_kwh == Decimal("1250.50")
    assert saved_invoice.gas_consumption_scm is None


def test_generate_company_history(
    db_session: Session, rg: np.random.Generator, fake: Faker
) -> None:
    """Validate multi-year lifecycle orchestration and database persistence.

    Tests that the end-to-end data generation pipeline properly instantiates a
    corporate entity, tracks its historical chronological progression through the
    current operational fiscal year, and commits all multi-layered
    relational structures.

    """
    generate_company_history(db_session, fake, rg)
    db_session.flush()

    company = db_session.query(Company).first()
    assert company is not None
    assert company.id is not None

    contracts = db_session.query(EnergyContract).filter_by(company_id=company.id).all()
    assert len(contracts) > 0

    statements = (
        db_session.query(FinancialStatement).filter_by(company_id=company.id).all()
    )
    assert len(statements) > 0

    years = [st.fiscal_year for st in statements]
    assert max(years) == 2026
    assert company.foundation_date.year in years

    total_invoices = db_session.query(Invoice).count()
    assert total_invoices > 0


def test_run_seeding_populates_every_single_table(db_session: Session) -> None:
    """Assert that the seed automation script completely populates the database schema.

    Executes the global data seeding pipeline with a fixed company count and
    validates that the transaction block commits expected record densities across
    all underlying multi-layered relational tables.

    """

    def mock_session_gen():
        yield db_session

    num_companies = 5
    run_seeding(num_companies=num_companies, seed=42, session_gen=mock_session_gen)

    assert db_session.query(Company).count() == num_companies

    total_contracts = db_session.query(EnergyContract).count()
    assert num_companies <= total_contracts <= (num_companies * 2)

    total_statements = db_session.query(FinancialStatement).count()
    assert total_statements >= num_companies

    total_invoices = db_session.query(Invoice).count()
    assert total_invoices > 0

    total_payments = db_session.query(Payment).count()
    assert total_payments > 0
    assert total_payments <= total_invoices

    total_tickets = db_session.query(CRMSupportTicket).count()
    assert total_tickets >= 0

    total_logins = db_session.query(UserWebLogin).count()
    assert total_logins > 0

    sample_invoice = db_session.query(Invoice).first()
    assert sample_invoice is not None
    assert isinstance(sample_invoice.total_amount, Decimal)
    assert sample_invoice.contract is not None
    assert sample_invoice.contract.company is not None


def test_run_seeding_exception_rollback() -> None:
    """Validate transaction rollback and error propagation during seeding failures.

    Tests that an unexpected database exception injected during the pipeline flush
    correctly triggers the underlying error handling block, ensures transaction
    rollback execution, and clean bubbles up the exception to the caller.
    """

    def mock_session_gen_error():
        session_mock = MagicMock()
        session_mock.flush.side_effect = Exception("Simulated database failure")
        yield session_mock

    with pytest.raises(Exception, match="Simulated database failure"):
        run_seeding(num_companies=1, seed=123, session_gen=mock_session_gen_error)


def test_simulate_operational_telemetry_suspended_and_default_paths() -> None:
    """Validate telemetry generation for suspended contracts and distressed firms.

    Tests through multi-seed stochastic exploration that entities under structural
    financial distress with suspended energy contracts restrict operational outputs,
    ensuring generated invoices strictly transition into unpaid or
    overdue accounting states.

    """
    contract_suspended = {
        "commodity_type": "electricity",
        "contract_status": "suspended",
    }
    vols_mock = {
        "annual_electricity": 50.0,
        "annual_gas": 0.0,
        "curves": {"electricity": (1.0,) * 12},
    }

    fin_distressed = {
        "total_revenue": 10000.0,
        "total_debt": 950000.0,
        "liquidity_cash": 5.0,
        "ebitda": -85000.0,
    }
    sector_prof = {"support_bias": "high", "digitalization_bias": "low"}

    for seed in range(50):
        rg = np.random.default_rng(seed)
        billing, tickets, logins = simulate_operational_telemetry(
            rg,
            2025,
            3,
            contract_suspended,
            vols_mock,
            fin_distressed,
            "construction",
            sector_prof,
        )
        if billing:
            assert billing["invoice_status"] in ["unpaid", "overdue"]


def test_simulate_operational_telemetry_future_payment_adjustment() -> None:
    """Validate payment date clamping when delayed payments land in the future.

    Tests that when the sampled payment delay would push the derived payment date
    beyond today's date, the invoice correctly reverts to an 'overdue'/'unpaid'
    state with a null pay_days, instead of persisting a payment dated in the future.
    Searches dynamically across seeds at runtime, since the target behavior
    depends on the current calendar date rather than a fixed historical one.

    """
    contract_active = {"commodity_type": "electricity", "contract_status": "active"}
    vols_mock = {"annual_electricity": 10.0, "curves": {"electricity": (1.0,) * 12}}
    fin_healthy = {
        "total_revenue": 500000.0,
        "total_debt": 0.0,
        "liquidity_cash": 100000.0,
        "ebitda": 150000.0,
    }
    sector_prof = {"support_bias": "low", "digitalization_bias": "low"}

    today = datetime.date.today()
    year, month = today.year, today.month

    found_future_payment = False
    for seed in range(500):
        rg = np.random.default_rng(seed)
        billing, _, _ = simulate_operational_telemetry(
            rg,
            year,
            month,
            contract_active,
            vols_mock,
            fin_healthy,
            "tech",
            sector_prof,
        )
        if (
            billing.get("pay_days") is None
            and billing.get("invoice_status") == "overdue"
        ):
            found_future_payment = True
            break

    assert found_future_payment, (
        "Nessun seed su 500 tentativi ha prodotto un pagamento clampato nel futuro"
    )
    assert billing["payment_behavior"] == "unpaid"


def test_persist_billing_and_telemetry_unhappy_paths(
    db_session: Session, fake: Faker
) -> None:
    """Verify billing persistence bounds for empty inputs and missing records.

    Tests that the persistence layer gracefully skips empty payload dictionaries,
    correctly handles unpaid invoice records with an undefined numerical payment
    delay, and accurately records failing payment events into the
    database tracking tables.

    """
    company = Company(
        vat_number="12345678901",
        legal_name="Test Unhappy Persist S.r.l.",
        legal_form="S.r.l.",
        industry_sector="services",
        foundation_date=datetime.date(2024, 1, 1),
        registered_office_region="Lombardia",
        is_active=True,
    )
    db_session.add(company)
    db_session.flush()

    contract = EnergyContract(
        company_id=company.id,
        commodity_type="gas",
        market_type="deregulated",
        pressure_level="low",
        gas_meter_class="G4",
        activation_date=company.foundation_date,
        contract_status="active",
    )
    db_session.add(contract)
    db_session.flush()

    billing_history = [
        {},
        {
            "month": 1,
            "consumption": 100.0,
            "amount_excluding_tax": Decimal("85.00"),
            "tax_amount": Decimal("18.70"),
            "total_amount": Decimal("103.70"),
            "invoice_status": "overdue",
            "payment_behavior": "unpaid",
            "pay_days": None,
            "payment_method": "bank_transfer",
            "issue_date": datetime.date(2025, 1, 1),
        },
        {
            "month": 2,
            "consumption": 200.0,
            "amount_excluding_tax": Decimal("170.00"),
            "tax_amount": Decimal("37.40"),
            "total_amount": Decimal("207.40"),
            "invoice_status": "unpaid",
            "payment_behavior": "failed",
            "pay_days": 10,
            "payment_method": "credit_card",
            "issue_date": datetime.date(2025, 2, 1),
        },
    ]

    persist_billing_and_telemetry(
        db_session, company.id, contract.id, "gas", billing_history, [], [], fake
    )

    invoices = db_session.query(Invoice).filter_by(contract_id=contract.id).all()
    assert len(invoices) == 2

    failed_payment = (
        db_session.query(Payment).filter_by(payment_status="failed").first()
    )
    assert failed_payment is not None


def test_generate_company_history_large_corporate_and_transitions(
    db_session: Session, fake: Faker
) -> None:
    """Verify large-cap generation and margin-driven contract transitions.

    Tests that a chemical heavy industry company classified as 'large' (S.p.A.)
    correctly generates dual-fuel contracts with high-tier voltage/pressure levels,
    and that a naturally occurring negative margin trajectory (seed-verified) drives
    the linked contract into a 'suspended' status without mocking the random
    generator's internal call sequence.

    """
    with patch(
        "simulation.seed.generate_identity",
        return_value=(
            {
                "vat_number": "12345678901",
                "legal_name": "Test Large Corp S.p.A.",
                "legal_form": "S.p.A.",
                "industry_sector": "chemical_heavy_industry",
                "foundation_date": datetime.date(2025, 1, 1),
                "registered_office_region": "Piemonte",
                "is_active": True,
            },
            15.0,
            "large",
        ),
    ):
        rg = np.random.default_rng(8)
        generate_company_history(db_session, fake, rg)

    db_session.flush()

    company = db_session.query(Company).first()
    assert company is not None
    assert company.legal_form == "S.p.A."
    assert company.industry_sector == "chemical_heavy_industry"

    contracts = db_session.query(EnergyContract).filter_by(company_id=company.id).all()
    assert len(contracts) > 0

    commodity_types = {c.commodity_type for c in contracts}
    assert "electricity" in commodity_types
    assert "gas" in commodity_types

    assert all(c.contract_status == "suspended" for c in contracts)


def test_generate_company_history_contract_reactivation(
    db_session: Session, fake: Faker
) -> None:
    """Verify lifecycle reactivation for previously suspended contracts.

    Tests that a company experiencing a transient financial dip (negative margin)
    followed by a sustained economic recovery successfully triggers a state
    transition, reverting its energy supply agreements from suspended back to active.

    """
    fin_data = {
        "total_revenue": Decimal("100000"),
        "net_income": Decimal("10000"),
        "ebitda": Decimal("15000"),
        "share_capital": Decimal("20000"),
        "liquidity_cash": Decimal("10000"),
        "total_debt": Decimal("15000"),
    }

    margins = iter(
        [
            (fin_data, -0.10),
            (fin_data, 0.10),
            (fin_data, 0.10),
            (fin_data, 0.10),
            (fin_data, 0.10),
            (fin_data, 0.10),
            (fin_data, 0.10),
            (fin_data, 0.10),
            (fin_data, 0.10),
        ]
    )

    with (
        patch(
            "simulation.seed.generate_identity",
            return_value=(
                {
                    "vat_number": "11122233344",
                    "legal_name": "Reactivate Test",
                    "legal_form": "S.r.l.",
                    "industry_sector": "tech",
                    "foundation_date": datetime.date(2018, 1, 1),
                    "registered_office_region": "Lazio",
                    "is_active": True,
                },
                4.0,
                "medium",
            ),
        ),
        patch(
            "simulation.seed.simulate_financial_year",
            side_effect=lambda *args, **kwargs: next(margins),
        ),
    ):
        rg = np.random.default_rng(42)
        generate_company_history(db_session, fake, rg)

    contracts = db_session.query(EnergyContract).all()

    assert len(contracts) > 0
    assert all(c.contract_status == "active" for c in contracts)


def test_generate_company_history_contract_termination(
    db_session: Session, fake: Faker
) -> None:
    """Validate stochastic contract termination during the current fiscal year.

    Searches across seeds for one that triggers the 8% termination probability
    check within the current year's processing loop, then verifies the affected
    contract transitions to 'terminated' with a populated termination date.

    """
    terminated_found = False
    contracts = []
    for seed in range(300):
        rg = np.random.default_rng(seed)
        generate_company_history(db_session, fake, rg)
        db_session.flush()

        company = db_session.query(Company).order_by(Company.id.desc()).first()
        contracts = (
            db_session.query(EnergyContract).filter_by(company_id=company.id).all()
        )
        if any(c.contract_status == "terminated" for c in contracts):
            terminated_found = True
            break

    assert terminated_found, "Nessun seed su 300 tentativi ha prodotto una terminazione"
    terminated_contracts = [c for c in contracts if c.contract_status == "terminated"]
    assert len(terminated_contracts) > 0
    assert all(c.termination_date is not None for c in terminated_contracts)


def test_run_seeding_rollback_and_logging_on_failure(db_session: Session) -> None:
    """Validate transactional rollback and error logging on mid-pipeline failure.

    Tests that an exception raised during company history generation triggers
    a session rollback (leaving no partial company records committed), logs the
    failure through the module logger, and re-raises the original exception.

    """

    def mock_session_gen():
        yield db_session

    with (
        patch(
            "simulation.seed.generate_company_history",
            side_effect=RuntimeError("Simulated failure"),
        ),
        patch("simulation.seed.logger") as mock_logger,
    ):
        with pytest.raises(RuntimeError, match="Simulated failure"):
            run_seeding(num_companies=3, seed=42, session_gen=mock_session_gen)

        mock_logger.error.assert_called_once()
        assert (
            "Error during seeding, executed rollback"
            in mock_logger.error.call_args[0][0]
        )

    assert db_session.query(Company).count() == 0
