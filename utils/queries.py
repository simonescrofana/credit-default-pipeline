"""Shared, parameterized SQL queries used across the application layers.

Centralizing raw queries here lets both the LangGraph agent (`agent/`) and
the FastAPI layer (`api/`) resolve the same data the same way, without one
importing from the other's package.

"""

from sqlalchemy import text

# Wrapped in text() explicitly: a raw string with named (:name) parameters
# passed to pd.read_sql is not reliably translated to the driver's own
# paramstyle (e.g. psycopg2's %(name)s) unless SQLAlchemy's text() marks it
# as a parameterized statement to interpret, rather than a literal string
# to pass through almost as-is.
RESOLVE_COMPANY_ID_QUERY = text(
    """
    SELECT company_id
    FROM public_marts.dim_companies
    WHERE legal_name = :identifier OR vat_number = :identifier
    LIMIT 1
    """
)
