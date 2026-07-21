"""Extract company credit profile data from the analytical star schema.

Provide functionality to query the OLAP data warehouse tables (fact table
fct_company_credit_profile joined with company and date dimensions) and load
the result into a structured pandas DataFrame indexed for analytical processing.

"""

import logging

import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

QUERY = """
    SELECT
        f.company_id,
        f.snapshot_date,
        f.company_age_days,
        f.active_contracts_count,
        f.has_active_electricity_contract,
        f.has_active_gas_contract,
        f.leverage_ratio,
        f.cash_to_debt_ratio,
        f.net_profit_margin,
        f.ebitda,
        f.max_dpd_trailing_90d,
        f.avg_dpd_trailing_90d,
        f.unpaid_ratio_trailing_90d,
        f.total_outstanding_debt,
        f.days_since_last_login,
        f.login_velocity,
        f.billing_disputes_count,
        f.average_satisfaction_score,
        f.is_insolvent,
        d.legal_name,
        d.industry_sector,
        d.registered_office_region,
        dt.year,
        dt.quarter,
        dt.month_number AS month
    FROM fct_company_credit_profile f
    JOIN dim_companies d ON f.company_key = d.company_key
    JOIN dim_date dt ON f.snapshot_date = dt.date_day
"""


def load_data(session: Session) -> pd.DataFrame:
    """Load credit profile features and metrics from the database into a DataFrame.

    Execute a SQL query joining the fact table and related dimension tables to
    extract financial metrics, operational KPIs, and company attributes. Set a
    multi-level index based on `company_id` and `snapshot_date`.

    Args:
        session (Session): The active SQLAlchemy database session used to execute
            the query.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the joined credit profile
        data, indexed by `company_id` and `snapshot_date`.

    Raises:
        SQLAlchemyError: If an error occurs during database query execution.

    """
    logger.info(
        "Starting querying the star schema to import data in a pandas DataFrame..."
    )
    try:
        df = pd.read_sql(QUERY, con=session.bind)

    except SQLAlchemyError:
        logger.exception("Error while reading SQL query with pandas!")
        raise

    df = df.set_index(["company_id", "snapshot_date"])
    logger.info("DataFrame successfully created: %d rows loaded.", len(df))
    return df
