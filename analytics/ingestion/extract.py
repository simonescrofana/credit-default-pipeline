"""Extract transactional database tables into local cold-storage Parquet files.

This module orchestrates the extraction pipeline by querying mapped ORM models
in memory-efficient chunks and persisting them sequentially into a dedicated
raw data repository directory.

"""

import logging
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import select

from database import connection
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

logger = logging.getLogger(__name__)

DATA_REPO = "data/raw"


def build_file_name(file_name: str, output_dir: str = DATA_REPO) -> str:
    """Resolve the absolute path for a data file within the repository.

    Combines the input file name with the target directory pathway, defaulting
    to the static repository location defined in DATA_REPO ('root/data/raw').

    Args:
        file_name (str): The target file name or relative path component.
        output_dir (str): The destination directory for the resolved path.
            Defaults to DATA_REPO, preserving the production extraction target.

    Returns:
        str: The fully resolved filesystem path inside the repository root.

    """
    return os.path.join(output_dir, file_name)


def extract_table_data(
    db_table_name: str,
    sqlalchemy_model: type[Base],
    chunk_size: int = 100000,
    output_dir: str = DATA_REPO,
) -> None:
    """Stream database table records into a partitioned Parquet file.

    Extracts rows in chunks via SQLAlchemy and appends them to a target Parquet
    storage file, ensuring low memory consumption for large-scale datasets.

    Args:
        db_table_name (str): Name of the source database table used for file naming.
        sqlalchemy_model (type[Base]): The mapped SQLAlchemy ORM model class to query.
        chunk_size (int): Numerical limit of rows processed per memory buffer segment.
            Defaults to 100000.
        output_dir (str): Destination directory for the extracted Parquet file.
            Defaults to DATA_REPO. Override in tests (e.g. with pytest's tmp_path)
            to avoid writing to -- and overwriting -- the real production output.

    Raises:
        Exception: Any upstream database connection fault or filesystem IO error
            encountered during execution.

    """
    logger.info("Starting data extraction from %s table...", db_table_name)

    writer = None
    session = None
    try:
        session = next(connection.get_db())
        query = select(sqlalchemy_model)

        logger.info("Dividing data in chunks of %s row...", chunk_size)
        chunks = pd.read_sql_query(
            sql=query, con=session.connection(), chunksize=chunk_size
        )

        total_rows = 0
        OUTPUT_FILE = build_file_name(db_table_name + ".parquet", output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # has_chunks is a flag required by tests to verify log reports when the
        # iterator logic broke for some reason
        has_chunks = False
        for i, chunk in enumerate(chunks):
            has_chunks = True
            chunk_rows = len(chunk)
            if chunk_rows == 0:
                logger.warning("Chunk number %d is empty.", i + 1)
                continue

            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(OUTPUT_FILE, table.schema)
            writer.write_table(table)

            total_rows += chunk_rows
            logger.info(
                "Successfully extracted data from block number %d"
                "(%d rows, for a cumulative amount of %d rows)",
                i + 1,
                chunk_rows,
                total_rows,
            )

        if not has_chunks or total_rows == 0:
            if not has_chunks:
                logger.warning("Chunk number 1 is empty.")
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

    finally:
        if writer is not None:
            writer.close()
        if session is not None:
            session.close()


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

    #    the repo creation now is inside the function itself
    #    if not os.path.exists(DATA_REPO):
    #        os.makedirs(DATA_REPO)
    #        logger.info("Created destination repository: %s", DATA_REPO)

    for table, model in zip(DB_TABLES, MODEL_TABLES):
        extract_table_data(table, model, chunk_size=100000)

    logger.info("Data extraction from the transational database completed!")
