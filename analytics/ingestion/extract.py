"""Extract transactional database tables into local cold-storage Parquet files.

This module orchestrates the extraction pipeline by querying mapped ORM models
in memory-efficient chunks and persisting them sequentially into a dedicated
raw data repository directory.

"""

import logging
import os

import pandas as pd
from sqlalchemy import select

from database.base import Base
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

logger = logging.getLogger(__name__)

DATA_REPO = "../data/raw"


def build_file_name(file_name: str) -> str:
    """Resolve the absolute path for a data file within the repository.

    Combines the input file name with the static target directory pathway
    defined in DATA_REPO ('../data/raw').

    Args:
        file_name (str): The target file name or relative path component.

    Returns:
        str: The fully resolved filesystem path inside the repository root.

    """
    return os.path.join(DATA_REPO, file_name)


def extract_table_data(
    db_table_name: str, sqlalchemy_model_name: type[Base], chunk_size: int = 100000
) -> None:
    """Stream database table records into a partitioned Parquet file.

    Extracts rows in chunks via SQLAlchemy and appends them to a target Parquet
    storage file, ensuring low memory consumption for large-scale datasets.

    Args:
        db_table_name (str): Name of the source database table used for file naming.
        sqlalchemy_model_name (type[Base]): The mapped SQLAlchemy ORM model class
            to query.
        chunk_size (int): Numerical limit of rows processed per memory buffer segment.
            Defaults to 100000.

    Raises:
        Exception: Any upstream database connection fault or filesystem IO error
            encountered during execution.

    """
    logger.info("Starting data extraction from %s table...", db_table_name)

    if not os.path.exists(DATA_REPO):
        os.makedirs(DATA_REPO)
        logger.info("Created destination repository: %s", DATA_REPO)

    try:
        session = next(get_db())
        query = select(sqlalchemy_model_name)

        logger.info("Dividing data in chunks of %s row...", chunk_size)
        chunks = pd.read_sql_query(sql=query, con=session.bind, chunksize=chunk_size)

        is_first_chunk = True
        total_rows = 0
        OUTPUT_FILE = build_file_name(db_table_name + ".parquet")

        for i, chunk in enumerate(chunks):
            if is_first_chunk:
                chunk.to_parquet(OUTPUT_FILE, index=False, append=False)
                is_first_chunk = False
            else:
                chunk.to_parquet(OUTPUT_FILE, index=False, append=True)

            chunk_rows = len(chunk)
            total_rows += chunk_rows

            if chunk_rows == 0:
                logger.warning("Chunk number %d is empty.", i + 1)
            logger.info(
                "Successfully extracted data from block number %d"
                "(%d rows, for a cumulative amount of %d rows)",
                i + 1,
                chunk_rows,
                total_rows,
            )

        if total_rows == 0:
            logger.warning("Extraction completed but no data was retrieved.")
        else:
            logger.info(
                "Data extraction from %s table completed!"
                "Stored %d rows in the file: %s",
                db_table_name,
                total_rows,
                OUTPUT_FILE,
            )

    except Exception as e:
        logger.error("Critical error during data extraction: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    from utils.logging_utils import setup_logging

    DB_TABLES = (
        "companies",
        "energy_contracts",
        "financial_statements",
        "invoices",
        "payments",
        "crm_support_tickets",
        "user_web_logins",
    )
    MODEL_TABLES = (
        Company,
        EnergyContract,
        FinancialStatement,
        Invoice,
        Payment,
        CRMSupportTicket,
        UserWebLogin,
    )

    setup_logging("INFO")
    logger.info("Starting data extracion from the transational database...")

    for table, model in zip(DB_TABLES, MODEL_TABLES):
        extract_table_data(table, model, chunk_size=100000)

    logger.info("Data extraction from the transational database completed!")
