"""Extractor node: turns free-text requests into structured data.

Handles the two data-gathering routes decided upstream by the router node:

- `extract_case_a`: resolves the company identifiers (legal name or VAT
  number) mentioned in the prompt into `company_id` values, via a
  parameterized database query. Identifiers with no match are recorded in
  `prediction_errors` rather than raising, so the graph can still proceed
  with whichever companies were found.
- `extract_case_b`: extracts ad hoc company data from the prompt into a
  permissive intermediate schema, then validates it against
  `InsolvencyPredictionRequest`. A validation failure is recorded in
  `prediction_errors` with the missing/invalid fields, rather than raising,
  so the responder node can explain to the user what is missing.

Neither function queries or validates more than necessary: `extract_case_a`
only resolves identity (company_id), leaving feature retrieval to
`ml.inference.predictor`; `extract_case_b` never invents a value the
prompt doesn't state.

"""

import logging

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from agent.models.llm_responder import get_responder_llm
from agent.prompts.extractor_prompt import (
    CASE_A_EXTRACTION_PROMPT,
    CASE_B_EXTRACTION_PROMPT,
)
from agent.state import AgentState
from database.connection import get_db
from schemas.agent.extraction_validation import CompanyIdentifiers, ExtractedCompanyData
from schemas.ml.insolvency_prediction import InsolvencyPredictionRequest

logger = logging.getLogger(__name__)

RESOLVE_COMPANY_ID_QUERY = """
    SELECT company_id
    FROM public_marts.dim_companies
    WHERE legal_name = :identifier OR vat_number = :identifier
    LIMIT 1
"""


def extract_case_a(state: AgentState) -> dict:
    """Extract company identifiers and resolve them to database company_ids.

    Args:
        state (AgentState): The current graph state. Reads `user_input`.

    Returns:
        dict: A partial state update setting `company_identifiers` (as
            extracted), `resolved_company_ids` (those that matched a
            company in the database), and `prediction_errors` (appended
            with one message per identifier that found no match).

    Raises:
        SQLAlchemyError: If an error occurs while querying the database.
            A missing company is not an error and is instead recorded in
            `prediction_errors`; this is only raised for infrastructural
            failures (e.g. the database is unreachable).

    """
    llm = get_responder_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(CompanyIdentifiers)

    messages = [
        SystemMessage(content=CASE_A_EXTRACTION_PROMPT),
        HumanMessage(content=state.user_input),
    ]

    logger.info("Extracting company identifiers...")
    extracted: CompanyIdentifiers = structured_llm.invoke(messages)
    logger.info("Extracted %d identifier(s).", len(extracted.identifiers))

    resolved_ids: list[int] = []
    errors: list[str] = list(state.prediction_errors)

    session_gen = get_db()
    session = next(session_gen)
    try:
        for identifier in extracted.identifiers:
            try:
                result = pd.read_sql(
                    RESOLVE_COMPANY_ID_QUERY,
                    con=session.bind,
                    params={"identifier": identifier},
                )
            except SQLAlchemyError:
                logger.exception(
                    "Connection error while resolving company identifier '%s'!",
                    identifier,
                )
                raise

            if result.empty:
                errors.append(f"Company '{identifier}' not found in the database.")
            else:
                resolved_ids.append(int(result.iloc[0]["company_id"]))

    finally:
        session_gen.close()

    return {
        "company_identifiers": extracted.identifiers,
        "resolved_company_ids": resolved_ids,
        "prediction_errors": errors,
    }


def extract_case_b(state: AgentState) -> dict:
    """Extract ad hoc company data from the prompt and validate it.

    Args:
        state (AgentState): The current graph state. Reads `user_input`.

    Returns:
        dict: A partial state update setting `raw_prediction_input` (the
            extracted data as a dict) and, if validation against
            `InsolvencyPredictionRequest` fails, an entry appended to
            `prediction_errors` describing what is missing or invalid.

    """
    llm = get_responder_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(ExtractedCompanyData)

    messages = [
        SystemMessage(content=CASE_B_EXTRACTION_PROMPT),
        HumanMessage(content=state.user_input),
    ]

    logger.info("Extracting ad hoc company data...")
    extracted: ExtractedCompanyData = structured_llm.invoke(messages)
    raw_data = extracted.model_dump(exclude_none=True)

    errors: list[str] = list(state.prediction_errors)
    try:
        InsolvencyPredictionRequest(**raw_data)
        logger.info("Extracted data validated successfully.")
    except ValidationError as exc:
        logger.info("Extracted data failed validation: %s", exc)
        errors.append(
            f"The provided company data is incomplete or invalid: {exc.errors()}"
        )

    return {
        "raw_prediction_input": raw_data,
        "prediction_errors": errors,
    }
