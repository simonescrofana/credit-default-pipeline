import datetime
import os
import pytest

from sqlalchemy.orm import Session

from analytics.ingestion.extract import extract_table_data
from database.models import Company


def test_data_extraction(db_session: Session) -> None:

    company = Company(
        vat_number="01234567890",
        legal_name="Acme Energy Consumer SPA",
        legal_form="SPA",
        industry_sector="manufacturing",
        foundation_date=datetime.date(2018, 4, 15),
        is_active=True,
    )
    db_session.add(company)
    db_session.flush()

    extract_table_data("company", Company)

    assert os.path.exists("../data/raw/company.parquet")
