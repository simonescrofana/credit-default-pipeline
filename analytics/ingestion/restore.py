"""Database recovery utility script.

This script manages the complete restoration of core business tables by reading
historical or snapshot data from Parquet files and performing atomic bulk
insertions into PostgreSQL. It maps file inputs to SQLAlchemy models for critical
entities (companies, invoices, payments, CRM tickets, etc.) and guarantees
referential integrity via cascading operations and automated sequence alignment.

"""

import logging
import os

import pandas as pd
import pyarrow.parquet as pq
from sqlalchemy import text

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
from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

MAPPING = [
    {"table_name": "companies", "model": Company},
    {"table_name": "financial_statements", "model": FinancialStatement},
    {"table_name": "energy_contracts", "model": EnergyContract},
    {"table_name": "invoices", "model": Invoice},
    {"table_name": "payments", "model": Payment},
    {"table_name": "crm_support_tickets", "model": CRMSupportTicket},
    {"table_name": "user_web_logins", "model": UserWebLogin},
]


def restore_database(dir_path: str = "data/raw", chunk_size: int = 50000) -> None:
    """Restore the database tables using historical data from Parquet files.

    This function coordinates the full recovery pipeline by first wiping any
    residual data from the database using a cascading truncate in reverse dependency
    order. It then iterates through the mapping configuration to read Parquet
    files in memory-efficient chunks, performs bulk inserts into the respective
    SQLAlchemy models, and re-synchronizes PostgreSQL primary key sequence counters
    to avoid collision errors on subsequent writes.

    Args:
        dir_path: Path to the directory containing the Parquet files.
        chunk_size: Number of rows to process and insert at a time.

    Raises:
        Exception: If any critical error occurs during the truncate, read, or bulk
            insertion phases. The active session is safely rolled back before
            re-raising the exception.

    """
    session = next(get_db())

    logger.info("Starting recovery of the data from the Parquet files...")

    try:
        db_type = session.bind.dialect.name
        is_sqlite = False
        if db_type == "sqlite":
            is_sqlite = True

        logger.info("Deleting eventual residual data from database...")

        for step in reversed(MAPPING):
            table_name = step["table_name"]
            if is_sqlite:
                session.execute(text(f"DELETE FROM {table_name};"))
            elif not is_sqlite:
                session.execute(text(f"DELETE FROM {table_name};"))
                session.execute(text(f"TRUNCATE TABLE {table_name} CASCADE;"))
        session.commit()

        for step in MAPPING:
            table_name = step["table_name"]
            model = step["model"]
            file_path = dir_path + f"/{table_name}.parquet"

            if not os.path.exists(file_path):
                logger.warning(
                    f"File not found: {file_path}. Skipping table {table_name}."
                )
                continue

            logger.info(
                f"Opening table {table_name} from {file_path} for chunked streaming..."
            )

            parquet_file = pq.ParquetFile(file_path)

            if parquet_file.metadata.num_rows == 0:
                logger.info(
                    f"The Parquet file {file_path} for {table_name} is empty. "
                    "Skipping file..."
                )
                continue

            total_rows_inserted = 0

            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                df = batch.to_pandas()

                df = df.replace({pd.NA: None, float("nan"): None})
                df = df.where(pd.notnull(df), None)
                records = df.to_dict(orient="records")

                session.bulk_insert_mappings(model, records)
                session.commit()

                total_rows_inserted += len(records)
                logger.debug(
                    f"Inserted chunk of {len(records)} rows into {table_name}."
                )

            logger.info(
                f"Recovered total {total_rows_inserted} rows in the table {table_name}!"
            )

        if not is_sqlite:
            logger.info("Updating sequences counters (ID) on Postgres...")
            for step in MAPPING:
                table_name = step["table_name"]
                if "id" in step["model"].__table__.columns:
                    seq_query = (
                        f"SELECT setval(pg_get_serial_sequence('{table_name}', 'id'),"
                        f"COALESCE(MAX(id), 1)) FROM {table_name};"
                    )
                    session.execute(text(seq_query))
            session.commit()

        else:
            logger.info(
                "SQLite environment detected: "
                "skipping Postgres update of sequence counters (ID)..."
            )

        logger.info("Database recovered!")

    except Exception as e:
        session.rollback()
        logger.error(f"Critical error during the database restore: {e}", exc_info=True)
        raise
    finally:
        session.close()


if __name__ == "__main__":
    setup_logging("INFO")
    restore_database()
