"""Database seeding and operational lifecycle simulation script.

This module acts as the core orchestration engine for synthesizing historical
B2B corporate data. It runs a longitudinal stochastic pipeline that models
macroeconomic drifts, structural credit risk trajectories, and synchronous
operational telemetry (billing ledgers, CRM tickets, digital platform traffic)
integrated directly with the target OLTP relational database schema.

"""

import datetime
import logging
from collections.abc import Iterator
from decimal import ROUND_HALF_UP, Decimal
from typing import Callable, Optional

import numpy as np
from faker import Faker
from sqlalchemy.orm import Session

from database.connection import get_db
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
    COMPANY_SIZE_SCALE,
    ENERGY_INTENSITY_SCALE,
    EVENT_RATE_SCALE,
    MONTHS_PER_YEAR,
    SEASONALITY_PROFILE_CURVES,
    SECTOR_PROFILES,
    CompanySize,
    SectorName,
)

logger = logging.getLogger(__name__)

ITALIAN_REGIONS: tuple[str, ...] = (
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
)

ELECTRICITY_PRICE_PER_KWH: float = 0.25
GAS_PRICE_PER_SMC: float = 0.85
ENERGY_TO_REVENUE_FACTOR: float = 6.5

PHYSICAL_CONSUMPTION_SCALE_KWH: float = 25_000.0

SECTOR_WEIGHTS: dict[SectorName, float] = {
    "manufacturing": 0.20,
    "chemical_heavy_industry": 0.05,
    "food_beverage": 0.10,
    "services": 0.25,
    "commerce": 0.15,
    "hospitality": 0.05,
    "healthcare": 0.05,
    "agriculture": 0.05,
    "construction": 0.04,
    "transportation": 0.03,
    "utilities": 0.01,
    "tech": 0.02,
}
COMPANY_SIZE_WEIGHTS: dict[CompanySize, float] = {
    "small": 0.75,
    "medium": 0.20,
    "large": 0.05,
}

SECTOR_RISK_PREMIUM: dict[str, float] = {
    "construction": 0.4,
    "hospitality": 0.3,
    "agriculture": 0.2,
    "manufacturing": 0.1,
    "services": -0.1,
    "tech": -0.2,
    "healthcare": -0.2,
    "chemical_heavy_industry": 0.1,
    "food_beverage": 0.0,
    "commerce": 0.0,
    "transportation": 0.1,
    "utilities": -0.3,
}

TODAY: datetime.date = datetime.date.today()


def _max_day_for_month(year: int, month: int) -> int:
    """Calculate the upper boundary day for generation within a given month matrix.

    Evaluates chronological limits by returning the actual calendar end of the target
    month for past periods, while capping the returned integer at the current day
    threshold if the parameters match the active system date to prevent
    futuristic events.

    Args:
        year (int): The four-digit calendar year under simulation evaluation.
        month (int): The specific month integer ranging from 1 to 12.

    Returns:
        int: The maximum valid day integer available for generation loops.

    """
    if year == TODAY.year and month == TODAY.month:
        return TODAY.day
    if month == 12:
        next_month_first = datetime.date(year + 1, 1, 1)
    else:
        next_month_first = datetime.date(year, month + 1, 1)
    return (next_month_first - datetime.timedelta(days=1)).day


def sigmoid(x: float) -> float:
    """Map a linear credit risk score to a probability constraint.

    Computes the standard logistic function to bound any real-valued input
    into a closed probability interval between 0 and 1. This is utilized to
    convert raw structural risk factors into non-linear default probabilities.

    Args:
        x (float): The raw linear credit risk score or log-odds value.

    Returns:
        float: The mapped probability value strictly constrained within [0, 1].

    """
    return 1.0 / (1.0 + np.exp(-x))


def generate_identity(rg: np.random.Generator, fake: Faker) -> tuple[dict, float, str]:
    """Generate a synthetic corporate identity and its operational scaling factors.

    Selects an industry sector and company size based on predefined domain
    weights, then constructs standard corporate metadata tailored to the Italian
    business ecosystem (VAT number, legal name, region, and foundation date).

    Args:
        rg (np.random.Generator): The initialized NumPy random generator for
            reproducible sampling.
        fake (Faker): The Faker instance used to synthesize realistic corporate
            legal names.

    Returns:
        tuple[dict, float, str]: A triplet containing:
            - dict: Corporate registration records matching the OLTP schema.
            - float: Quantitative capacity multiplier derived from the company size.
            - str: The selected company size category ("small", "medium", "large").

    """
    sector = str(
        rg.choice(list(SECTOR_WEIGHTS.keys()), p=list(SECTOR_WEIGHTS.values()))
    )
    size = str(
        rg.choice(
            list(COMPANY_SIZE_WEIGHTS.keys()), p=list(COMPANY_SIZE_WEIGHTS.values())
        )
    )

    size_mult = COMPANY_SIZE_SCALE[size]
    foundation_year = int(rg.integers(2018, 2021))
    foundation_date = datetime.date(foundation_year, 1, 1)

    legal_form = (
        str(rg.choice(["S.r.l.", "S.p.A."], p=[0.92, 0.08]))
        if size == "large"
        else "S.r.l."
    )

    info = {
        "vat_number": f"{rg.integers(10000000000, 99999999999)}",
        "legal_name": f"{
            str(rg.choice(['Industrie', 'Gruppo', 'Sistemi', 'Logistica']))
        } {fake.last_name()} {legal_form}",
        "legal_form": legal_form,
        "industry_sector": sector,
        "foundation_date": foundation_date,
        "registered_office_region": str(rg.choice(ITALIAN_REGIONS)),
        "is_active": True,
    }
    return info, size_mult, size


def generate_consumption_series(
    rg: np.random.Generator, sector: str, size_mult: float
) -> dict:
    """Calculate annualized volumetric consumption parameters for energy commodities.

    Computes baseline electricity and gas consumption volumes by compounding the
    sector's core energy intensity scale with company size multipliers and stochastic
    variance. It maps the designated monthly seasonality curves to each utility type.

    Args:
        rg (np.random.Generator): The initialized NumPy random generator for
            stochastic baseline variance.
        sector (str): The industry sector name used to look up behavioral profiles.
        size_mult (float): The capacity scaling multiplier derived from corporate size.

    Returns:
        dict: A structured configuration dictionary containing:
            - "annual_electricity" (float): Scaled annual power volume.
            - "annual_gas" (float): Scaled annual thermal gas volume.
            - "curves" (dict): Nested monthly seasonality factor tuples for
              "electricity" and "gas".

    """
    profile = SECTOR_PROFILES[sector]
    base_energy = (
        ENERGY_INTENSITY_SCALE[profile["energy_intensity"]]
        * size_mult
        * rg.uniform(0.85, 1.15)
    )

    annual_elec = (
        base_energy
        if profile["electricity_weight"] in ["high", "very_high"]
        else base_energy * 0.4
    )
    annual_gas = (
        base_energy * 0.6
        if profile["gas_weight"] in ["medium", "high", "very_high"]
        else 0.0
    )

    return {
        "annual_electricity": float(annual_elec),
        "annual_gas": float(annual_gas),
        "curves": {
            "electricity": SEASONALITY_PROFILE_CURVES[
                profile["seasonality"]["electricity"]
            ],
            "gas": SEASONALITY_PROFILE_CURVES[profile["seasonality"]["gas"]],
        },
    }


def simulate_financial_year(
    rg: np.random.Generator,
    sector: str,
    size: str,
    total_energy: float,
    current_margin: float,
) -> tuple[dict, float]:
    """Simulate a single fiscal year corporate financial statement.

    Computes top-line revenue by compounding company size baselines with utility
    consumption vectors. It synthesizes core financial metrics—including EBITDA, net
    income, share capital, cash reserves, and total liabilities—leveraging structural
    stochastic adjustments influenced by the company's current operating margin.

    Args:
        rg (np.random.Generator): The initialized NumPy random generator for
            macroeconomic and operational variance.
        sector (str): The industrial sector identifier determining risk and cost bounds.
        size (str): The corporate size category ("small", "medium", "large") used
            to set revenue baselines.
        total_energy (float): Combined annual volumetric utility consumption.
        current_margin (float): The operational profit margin driving asset and
            liability distribution behaviors.

    Returns:
        tuple[dict, float]: A pair containing:
            - dict: A balance sheet record mapped to `Decimal` types
              for financial safety:
                * "total_revenue" (Decimal)
                * "net_income" (Decimal)
                * "ebitda" (Decimal)
                * "share_capital" (Decimal)
                * "liquidity_cash" (Decimal)
                * "total_debt" (Decimal)
            - float: The original or adjusted operational margin carried over
              for downstream iterations.

    """
    size_base_rev = {"small": 120000, "medium": 750000, "large": 4500000}[size]
    revenue = size_base_rev * rg.uniform(0.80, 1.25) + (
        total_energy * ENERGY_TO_REVENUE_FACTOR
    )

    ebitda = revenue * current_margin
    net_inc = ebitda * 0.72 if ebitda > 0 else ebitda
    share_cap = revenue * rg.uniform(0.10, 0.20)

    if current_margin < 0.01:
        cash = revenue * rg.uniform(0.002, 0.015)
        debt = revenue * rg.uniform(0.35, 0.65)
    else:
        cash = revenue * rg.uniform(0.04, 0.12)
        debt = revenue * rg.uniform(0.05, 0.25)

    next_margin = current_margin + rg.uniform(-0.03, 0.03)
    next_margin = max(-0.20, min(0.40, next_margin))

    return {
        "total_revenue": Decimal(str(round(revenue, 2))),
        "net_income": Decimal(str(round(net_inc, 2))),
        "ebitda": Decimal(str(round(ebitda, 2))),
        "share_capital": Decimal(str(round(share_cap, 2))),
        "liquidity_cash": Decimal(str(round(cash, 2))),
        "total_debt": Decimal(str(round(debt, 2))),
    }, next_margin


def simulate_operational_telemetry(
    rg: np.random.Generator,
    year: int,
    month: int,
    contract: dict,
    vols: dict,
    fin_statement: dict,
    sector: str,
    sector_prof: dict,
) -> tuple[dict, list[dict], list[dict]]:
    """Generate synchronous monthly billing records, CRM tickets, and user logs.

    Simulates realistic corporate behavior by linking financial ratios (leverage,
    liquidity, profitability) to operational outcomes like payment failures,
    customer support tickets, and digital platform interaction rates without
    introducing target leakage.

    Args:
        rg (np.random.Generator): The initialized NumPy random generator for
            stochastic operational sampling.
        year (int): The calendar year of the simulation step.
        month (int): The calendar month of the simulation step.
        contract (dict): Active contract attributes including commodity type and status.
        vols (dict): Volumetric benchmarks and baseline monthly consumption curves.
        fin_statement (dict): Fiscal indicators used to dynamically compute default
            probabilities.
        sector (str): Industrial sector token driving targeted risk premiums.
        sector_prof (dict): Qualitative sector matrices controlling support and
            digitalization biases.

    Returns:
        tuple[dict, list[dict], list[dict]]: A triplet containing:
            - dict: A structured billing/invoice record matching the OLTP schema.
            - list[dict]: Chronological CRM tickets generated during the month.
            - list[dict]: Micro-traffic user interaction and login timestamps.

    """
    max_day = _max_day_for_month(year, month)

    comm = contract["commodity_type"]
    curve = vols["curves"][comm]
    base_vol = (
        vols["annual_electricity"] if comm == "electricity" else vols["annual_gas"]
    )
    base_vol_physical = base_vol * PHYSICAL_CONSUMPTION_SCALE_KWH
    price = ELECTRICITY_PRICE_PER_KWH if comm == "electricity" else GAS_PRICE_PER_SMC

    issue_dt = datetime.date(year, month, 1)

    if contract["contract_status"] == "terminated":
        return {}, [], []
    elif contract["contract_status"] == "suspended":
        if rg.random() < 0.75:
            return {}, [], []
        consumption = (
            (base_vol_physical / MONTHS_PER_YEAR)
            * curve[month - 1]
            * rg.uniform(0.01, 0.04)
        )
    else:
        consumption = (
            (base_vol_physical / MONTHS_PER_YEAR)
            * curve[month - 1]
            * rg.uniform(0.9, 1.1)
        )

    consumption_dec = Decimal(str(round(consumption, 2)))

    TWO_PLACES = Decimal("0.01")
    amount_ex_tax_not_rounded = consumption_dec * Decimal(str(round(price, 2)))
    amount_ex_tax = amount_ex_tax_not_rounded.quantize(
        TWO_PLACES, rounding=ROUND_HALF_UP
    )
    tax_amount = (amount_ex_tax * Decimal("0.22")).quantize(
        TWO_PLACES, rounding=ROUND_HALF_UP
    )
    total_amount = amount_ex_tax + tax_amount

    rev = float(fin_statement["total_revenue"])
    debt_ratio = float(fin_statement["total_debt"]) / rev if rev > 0 else 1.0
    cash_ratio = float(fin_statement["liquidity_cash"]) / rev if rev > 0 else 0.0
    ebitda_margin = float(fin_statement["ebitda"]) / rev if rev > 0 else -0.1

    risk_score = (
        -8.0 * ebitda_margin
        + 2.5 * debt_ratio
        - 4.0 * cash_ratio
        + SECTOR_RISK_PREMIUM.get(sector, 0.0)
        + rg.normal(0.0, 0.35)
    )

    default_prob = sigmoid(risk_score - 2.2)
    roll = rg.random()
    pay_method = str(
        rg.choice(
            ["direct_debit", "bank_transfer", "credit_card"], p=[0.75, 0.20, 0.05]
        )
    )

    if roll > default_prob:
        if rg.random() < 0.94:
            behavior, status, days = "completed", "paid", int(rg.integers(2, 12))
        else:
            behavior, status, days = "completed", "paid", int(rg.integers(13, 44))
    else:
        if rg.random() < 0.40:
            behavior, status, days = "failed", "unpaid", int(rg.integers(3, 14))
        else:
            behavior, status, days = "unpaid", "overdue", None
    if days is not None and (issue_dt + datetime.timedelta(days=days)) > TODAY:
        days = None
        status = "overdue" if status == "paid" and behavior == "completed" else status
        behavior = "unpaid" if behavior == "completed" else behavior

    billing = {
        "month": month,
        "consumption": float(consumption_dec),
        "amount_excluding_tax": amount_ex_tax,
        "tax_amount": tax_amount,
        "total_amount": total_amount,
        "invoice_status": status,
        "payment_behavior": behavior,
        "pay_days": days,
        "payment_method": pay_method,
        "issue_date": issue_dt,
    }

    tickets = []
    crm_rate = EVENT_RATE_SCALE[sector_prof["support_bias"]] * (
        2.8 if ebitda_margin < 0.02 else 0.7
    )
    if rg.random() < (crm_rate / 12.0):
        category = str(
            rg.choice(
                ["billing", "technical", "commercial"],
                p=[0.55, 0.35, 0.10] if ebitda_margin < 0.02 else [0.20, 0.65, 0.15],
            )
        )
        created_ts = datetime.datetime(
            year,
            month,
            int(rg.integers(1, max_day)),
            int(rg.integers(8, 18)),
            0,
            tzinfo=datetime.timezone.utc,
        )

        resolved_ts = (
            created_ts + datetime.timedelta(days=int(rg.integers(1, 12)))
            if ebitda_margin > -0.02
            else None
        )
        satisfaction = (
            int(rg.integers(1, 4)) if ebitda_margin < 0.02 else int(rg.integers(3, 6))
        )

        tickets.append(
            {
                "ticket_category": category,
                "created_at": created_ts,
                "resolved_at": resolved_ts,
                "satisfaction_score": satisfaction if resolved_ts else None,
            }
        )

    logins = []
    login_base_rate = int(
        EVENT_RATE_SCALE[sector_prof["digitalization_bias"]] * rg.integers(4, 14)
    )
    actual_logins = (
        int(login_base_rate * 0.15) if ebitda_margin < -0.04 else login_base_rate
    )

    for _ in range(max(1, actual_logins)):
        logins.append(
            {
                "user_id": int(rg.integers(101, 105)),
                "login_timestamp": datetime.datetime(
                    year,
                    month,
                    int(rg.integers(1, max_day)),
                    int(rg.integers(0, 24)),
                    int(rg.integers(0, 60)),
                    tzinfo=datetime.timezone.utc,
                ),
                "ip_address": f"192.168.{rg.integers(1, 254)}.{rg.integers(1, 254)}",
                "device_type": str(
                    rg.choice(["desktop", "mobile", "tablet"], p=[0.70, 0.25, 0.05])
                ),
            }
        )

    return billing, tickets, logins


def persist_billing_and_telemetry(
    session: Session,
    company_id: int,
    contract_id: int,
    comm_type: str,
    billing_history: list[dict],
    tickets: list[dict],
    logins: list[dict],
    fake: Faker,
) -> None:
    """Map and commits generated synthetic operational records into the OLTP schema.

    Iterates through simulated historical operational logs to instantiate, link, and
    persist SQLAlchemy ORM models. Manages database synchronization via interim flushes
    to resolve foreign key dependencies dynamically between invoices and payments.

    Args:
        session (Session): The active SQLAlchemy database session context.
        company_id (int): Foreign key identifier linking tickets and logins
            to the enterprise.
        contract_id (int): Foreign key identifier anchoring generated invoice instances.
        comm_type (str): The active utility commodity stream ("electricity" or "gas").
        billing_history (list[dict]): Simulated monthly raw financial billing events.
        tickets (list[dict]): Collection of unstructured synthesized CRM
            customer interactions.
        logins (list[dict]): Collection of structured digital platform interaction
            traffic logs.
        fake (Faker): The Faker instance utilized for unique invoice and transaction
            sequence tokenization.

    """
    for b in billing_history:
        if not b:
            continue
        inv = Invoice(
            contract_id=contract_id,
            commodity_type=comm_type,
            invoice_number=f"INV-{comm_type[:3].upper()}-{fake.unique.bothify(text='?#?#?#?#?')}",
            electricity_consumption_kwh=Decimal(str(round(b["consumption"], 2)))
            if comm_type == "electricity"
            else None,
            gas_consumption_scm=Decimal(str(round(b["consumption"], 2)))
            if comm_type == "gas"
            else None,
            amount_excluding_tax=b["amount_excluding_tax"],
            tax_amount=b["tax_amount"],
            total_amount=b["total_amount"],
            issue_date=b["issue_date"],
            due_date=b["issue_date"] + datetime.timedelta(days=20),
            invoice_status=b["invoice_status"],
        )
        session.add(inv)
        session.flush()

        if b["pay_days"] is not None:
            session.add(
                Payment(
                    invoice_id=inv.id,
                    payment_date=inv.issue_date
                    + datetime.timedelta(days=b["pay_days"]),
                    amount_paid=inv.total_amount,
                    payment_method=b["payment_method"],
                    transaction_reference=f"TRX-{fake.unique.bothify(text='?#?#?#?#?#?#')}",
                    payment_status="completed"
                    if b["payment_behavior"] == "completed"
                    else "failed",
                )
            )

    for t in tickets:
        session.add(CRMSupportTicket(company_id=company_id, **t))
    for login in logins:
        session.add(UserWebLogin(company_id=company_id, **login))

    session.flush()


def generate_company_history(
    session: Session, fake: Faker, rg: np.random.Generator
) -> None:
    """Simulate a multi-year corporate operational lifecycle.

    Orchestrates the transactional sequence from core identity definition and contract
    blueprint construction to longitudinal state updates. It advances the corporate
    entity through fiscal iterations, computing structural margin drifts,
    dynamically adapting contract statuses (active, suspended, terminated),
    and emitting operational time-series.

    Args:
        session (Session): The active SQLAlchemy database session context used
            for multi-stage model persistence.
        fake (Faker): The Faker instance utilized for name, token, and string synthesis.
        rg (np.random.Generator): The initialized NumPy random generator driving
            macroeconomic drifts and structural rolling behaviors.

    """
    comp_data, size_mult, size_str = generate_identity(rg, fake)
    sector = comp_data["industry_sector"]
    sector_prof = SECTOR_PROFILES[sector]

    db_company = Company(**comp_data)
    session.add(db_company)
    session.flush()

    vols = generate_consumption_series(rg, sector, size_mult)

    contracts_blueprint = []
    if vols["annual_electricity"] > 0:
        v_level = (
            "low"
            if size_str == "small"
            else ("medium" if size_str == "medium" else "high")
        )

        if size_str == "small":
            p_kw = Decimal(str(rg.choice([6.0, 10.0, 15.0], p=[0.60, 0.30, 0.10])))
        elif size_str == "medium":
            p_kw = Decimal(str(round(rg.uniform(30.0, 150.0), 2)))
        else:
            p_kw = Decimal(str(round(rg.uniform(500.0, 3500.0), 2)))

        contracts_blueprint.append(
            {
                "commodity_type": "electricity",
                "voltage_level": v_level,
                "power_committed_kw": p_kw,
                "pressure_level": None,
                "gas_meter_class": None,
            }
        )

    if vols["annual_gas"] > 0:
        p_level = (
            "low"
            if size_str == "small"
            else ("medium" if size_str == "medium" else "high")
        )
        if size_str == "small":
            g_meter = str(rg.choice(["G4", "G6"], p=[0.85, 0.15]))
        elif size_str == "medium":
            g_meter = str(rg.choice(["G16", "G25", "G40"], p=[0.40, 0.40, 0.20]))
        else:
            g_meter = str(rg.choice(["G65", "G100", "G250"], p=[0.50, 0.35, 0.15]))

        contracts_blueprint.append(
            {
                "commodity_type": "gas",
                "voltage_level": None,
                "power_committed_kw": None,
                "pressure_level": p_level,
                "gas_meter_class": g_meter,
            }
        )

    db_contracts_map = {}
    for cb in contracts_blueprint:
        db_c = EnergyContract(
            company_id=db_company.id,
            commodity_type=cb["commodity_type"],
            market_type="deregulated",
            activation_date=db_company.foundation_date,
            contract_status="active",
            voltage_level=cb["voltage_level"],
            power_committed_kw=cb["power_committed_kw"],
            pressure_level=cb["pressure_level"],
            gas_meter_class=cb["gas_meter_class"],
        )
        session.add(db_c)
        session.flush()
        db_contracts_map[cb["commodity_type"]] = db_c

    start_year = db_company.foundation_date.year
    end_year = 2026
    current_margin = float(rg.normal(0.06, 0.04))
    prev_fin_data: Optional[dict] = None

    for year in range(start_year, end_year + 1):
        drift = float(rg.normal(-0.005, 0.025))
        current_margin = float(np.clip(current_margin + drift, -0.15, 0.35))

        fin_data, final_margin = simulate_financial_year(
            rg,
            sector,
            size_str,
            vols["annual_electricity"] + vols["annual_gas"],
            current_margin,
        )
        fin_statement = FinancialStatement(
            company_id=db_company.id, fiscal_year=year, **fin_data
        )
        session.add(fin_statement)
        session.flush()

        risk_driving_fin_data = prev_fin_data if prev_fin_data is not None else fin_data

        termination_month_by_commodity: dict[str, int] = {}
        for comm, k in db_contracts_map.items():
            if k.contract_status == "terminated":
                continue
            if final_margin < -0.07:
                k.contract_status = "suspended"
            elif final_margin > 0.04 and k.contract_status == "suspended":
                k.contract_status = "active"

            if year == 2026 and rg.random() < 0.08:
                termination_month_by_commodity[comm] = int(rg.integers(1, 8))

        for m in range(1, MONTHS_PER_YEAR + 1):
            if year == 2026 and m > 7:
                break

            for comm, db_contract in db_contracts_map.items():
                term_month = termination_month_by_commodity.get(comm)
                if term_month is not None and m >= term_month:
                    effective_status = "terminated"
                else:
                    effective_status = db_contract.contract_status

                c_mock = {"commodity_type": comm, "contract_status": effective_status}

                b_rec, tickets, logins = simulate_operational_telemetry(
                    rg,
                    year,
                    m,
                    c_mock,
                    vols,
                    risk_driving_fin_data,
                    sector,
                    sector_prof,
                )

                persist_billing_and_telemetry(
                    session,
                    db_company.id,
                    db_contract.id,
                    comm,
                    [b_rec],
                    tickets,
                    logins,
                    fake,
                )

        for comm, term_month in termination_month_by_commodity.items():
            k = db_contracts_map[comm]
            k.contract_status = "terminated"
            k.termination_date = datetime.date(2026, term_month, 1)

        prev_fin_data = fin_data


def run_seeding(
    num_companies: int = 100,
    seed: int = 202607,
    session_gen: Callable[[], Iterator[Session]] = get_db,
) -> None:
    """Initialize simulation engines and orchestrates database pipeline seeding.

    Instantiates the localized synthesis generators, enforces strict global
    reproducibility configurations, and executes the historical corporate lifecycle
    loop. Manages transactional integrity across all generated entity boundaries
    by handling atomic database sessions with explicit rollback guarantees.

    Args:
        num_companies (int): Total number of synthetic corporate profiles to generate
            and persist. Defaults to 100.
        seed (int): Global pseudo-random number generator seed to guarantee
            deterministic simulation outputs. Defaults to 202607.
        session_gen (Callable[[], Iterator[Session]]): A factory function yielding
            an active SQLAlchemy Session context iterator. Defaults to get_db.

    Raises:
        Exception: Any upstream database or generation error, triggering a safe
            transaction rollback before re-raising.

    """
    logger.info(f"Starting simulation for {num_companies} companies (Seed: {seed})...")

    fake = Faker("it_IT")
    rg = np.random.default_rng(seed)
    Faker.seed(seed)

    db_generator = session_gen()
    session = next(db_generator)
    try:
        for i in range(1, num_companies + 1):
            if i % 10 == 0 or i == num_companies:
                logger.info(f"Generation company n. {i}/{num_companies}...")

            generate_company_history(session, fake, rg)

        session.commit()
        logger.info("Seeding completed successfully in the database!")

    except Exception as e:
        session.rollback()
        logger.error(f"Error during seeding, executed rollback: {str(e)}")
        raise e
    finally:
        session.close()


if __name__ == "__main__":
    from utils.logging_utils import setup_logging

    setup_logging("INFO")

    run_seeding(num_companies=10000, seed=202607)
