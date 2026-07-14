"""Custom SQLAlchemy types and compiler extensions for cross-dialect compatibility.

Provides custom type decorators and dialect-aware compilation clauses designed
to maintain strict fixed-point arithmetic precision and schema constraint
integrity across disparate backends, specifically targeting SQLite and PostgreSQL.

"""

from decimal import Decimal
from typing import Any, Optional

from sqlalchemy import Numeric, String
from sqlalchemy.engine import Dialect
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.elements import TextClause
from sqlalchemy.types import TypeDecorator


class ExactNumeric(TypeDecorator):
    """Custom decorator ensuring cross-dialect exact fixed-point math storage.

    Bypasses SQLite floating-point binding limitations by transparently mapping
    exact numeric decimals to string storage under SQLite while maintaining native
    high-precision database-level numeric structures on PostgreSQL.

    """

    impl = Numeric
    cache_ok = True

    def __init__(self, precision: int, scale: int, **kw):
        """Initialize the backend type system with localized layout parameters.

        Args:
            precision (int): Total maximum number of digits allowed in the fields.
            scale (int): Fixed number of decimal digits positioned after the point.
            **kw: Arbitrary keyword arguments forwarded to the parent implementation.

        """
        self.scale = scale
        super().__init__(precision=precision, scale=scale, **kw)

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        """Determine the specific underlying type descriptor based on the active engine.

        Args:
            dialect (Dialect): The dialect object currently executing the transaction.

        Returns:
            Any: A standard string storage descriptor for SQLite engines or a default
                high-precision numeric descriptor for compliant engines.

        """
        if dialect.name == "sqlite":
            return dialect.type_descriptor(String())
        return dialect.type_descriptor(self.impl)

    def process_bind_param(  ## pragma: no cover
        self,
        value: Optional[Decimal],
        dialect: Dialect,  # pragma: no cover
    ) -> Optional[Decimal | str]:  # pragma: no cover
        """Intercept outbound values to apply fixed-point text formatting if needed.

        Args:
            value (Optional[Decimal]): The local precise application-level memory value.
            dialect (Dialect): The targeted active relational backend system.

        Returns:
            Optional[Decimal | str]: A normalized and quantized textual string
                representation when writing to SQLite files, or the
                unmodified object for native backends.

        """
        if value is None or dialect.name != "sqlite":
            return value
        return str(Decimal(value).quantize(Decimal(10) ** -self.scale))

    def process_result_value(
        self, value: Optional[str], dialect: Dialect
    ) -> Optional[Decimal]:
        """Convert incoming backend column buffers back into validated precise objects.

        Args:
            value (Optional[str]): The plain source data retrieved from the driver row.
            dialect (Dialect): The executing database system engine context.

        Returns:
            Optional[Decimal]: A fully parsed high-precision application object instance
                or None if the stored column value was null.

        """
        return None if value is None else Decimal(value)


class DialectAwareCheck(TextClause):
    """Custom SQL check clause enabling dialect-specific syntax compilation.

    Overrides standard compilation blocks to bypass cross-dialect mathematical
    handling, addressing SQLite's internal tendency to convert check operands into
    floating-point representations during validation loops.

    """

    inherit_cache = True


def dialect_aware_check(pg_expr: str, sqlite_expr: str) -> DialectAwareCheck:
    """Instantiate an isolated conditional constraint tracking distinct SQL clauses.

    Args:
        pg_expr (str): The exact structural SQL expression designed for PostgreSQL.
        sqlite_expr (str): The epsilon-tolerant expression optimized for SQLite engines.

    Returns:
        DialectAwareCheck: A composite dialect-aware textual statement instance.

    """
    clause = DialectAwareCheck(pg_expr)
    clause._pg_expr = pg_expr
    clause._sqlite_expr = sqlite_expr
    return clause


@compiles(DialectAwareCheck, "postgresql")
def _compile_dialect_check_pg(element, compiler, **kw):
    """Emit the exact fixed-point structural expression for PostgreSQL backends.

    Args:
        element (DialectAwareCheck): The structural constraint node to compile.
        compiler (Any): The system compiler engine driving the active transaction.
        **kw (Any): Additional internal keyword arguments passed by SQLAlchemy.

    Returns:
        str: The raw PostgreSQL-compliant check string.

    """
    return element._pg_expr


@compiles(DialectAwareCheck, "sqlite")
def _compile_dialect_check_sqlite(element, compiler, **kw):
    """Emit the custom epsilon-tolerant numerical comparison statement for SQLite.

    Args:
        element (DialectAwareCheck): The structural constraint node to compile.
        compiler (Any): The system compiler engine driving the active transaction.
        **kw (Any): Additional internal keyword arguments passed by SQLAlchemy.

    Returns:
        str: The specialized SQLite-compliant check string.

    """
    return element._sqlite_expr
