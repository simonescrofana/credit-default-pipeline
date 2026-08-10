"""Direct, deterministic prediction endpoints, bypassing the LLM.

Exposes two parallel endpoints, mirroring the case_a/case_b distinction
used throughout the rest of the project:

- POST /predict/ad-hoc: score a fully-specified, ad hoc company profile
    not present in the database (case_b).
- POST /predict/company: score a company already present in the database,
    looked up by legal name or VAT number (case_a).

Neither endpoint goes through the agent or the LLM: both call directly into
`ml.inference.predictor`, the same functions the agent's predictor node
uses internally.

"""

import logging

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import get_loaded_model
from database.connection import get_db
from ml.inference.model_loader import LoadedModel
from ml.inference.predictor import predict, predict_from_raw_data
from schemas.api.predict import ExistingCompanyRequest, PredictionResponse
from schemas.ml.insolvency_prediction import InsolvencyPredictionRequest
from utils.queries import RESOLVE_COMPANY_ID_QUERY

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("/ad-hoc", response_model=PredictionResponse)
def predict_ad_hoc(
    request: InsolvencyPredictionRequest,
    loaded_model: LoadedModel = Depends(get_loaded_model),
) -> PredictionResponse:
    """Score a fully-specified, ad hoc company profile (case_b).

    Args:
        request (InsolvencyPredictionRequest): The ad hoc company data to
            score, already validated by Pydantic against the request body.
        loaded_model (LoadedModel): The model, encoder, scaler, explainer,
            and decision threshold bundle, injected via
            `Depends(get_loaded_model)`.

    Returns:
        PredictionResponse: The predicted probability, class, and a
            SHAP-based explanation of the prediction.

    """
    result = predict_from_raw_data(request, loaded_model)

    return PredictionResponse(**result._asdict())


@router.post("/company", response_model=PredictionResponse)
def predict_company(
    request: ExistingCompanyRequest,
    session: Session = Depends(get_db),
    loaded_model: LoadedModel = Depends(get_loaded_model),
) -> PredictionResponse:
    """Score a company already present in the database (case_a).

    Resolves `request.identifier` (legal name or VAT number) into a
    `company_id` the same way the agent's extractor node does, then scores
    the company's latest star schema snapshot.

    Args:
        request (ExistingCompanyRequest): The company identifier (legal
            name or VAT number) to look up and score.
        session (Session): An active SQLAlchemy session, injected via
            `Depends(get_db)`.
        loaded_model (LoadedModel): The model, encoder, scaler, explainer,
            and decision threshold bundle, injected via
            `Depends(get_loaded_model)`.

    Returns:
        PredictionResponse: The predicted probability, class, and a
            SHAP-based explanation of the prediction.

    Raises:
        HTTPException: 404 if no company matches `request.identifier`.

    """
    logger.info("Resolving company identifier...")
    resolved = pd.read_sql(
        RESOLVE_COMPANY_ID_QUERY,
        con=session.bind,
        params={"identifier": request.identifier},
    )

    if resolved.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Company '{request.identifier}' not found.",
        )

    company_id = int(resolved.iloc[0]["company_id"])
    result = predict(session, company_id, loaded_model)

    return PredictionResponse(**result._asdict())
