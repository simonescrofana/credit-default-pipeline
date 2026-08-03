"""Serve predictions for existing companies using the production XGBoost model.

Provide the inference entry point for scoring a company already present in
the star schema, reusing the same fitted preprocessing transformations
produced during training. Every prediction is explained via SHAP (see
`ml.evaluation.explainability`), since XGBoost is the only model family
served in production. Support for scoring companies not present in the
database (arbitrary user-supplied data) is a separate, still-to-be-implemented
entry point.

"""

import logging
from typing import NamedTuple

import pandas as pd
from sqlalchemy.orm import Session

from ml.dataset.loader import QUERY
from ml.dataset.preprocessing import handle_missing_and_encode, scale_features
from ml.evaluation.explainability import explain_prediction
from ml.inference.model_loader import LoadedModel
from schemas.ml.insolvency_prediction import InsolvencyPredictionRequest

logger = logging.getLogger(__name__)


class PredictionResult(NamedTuple):
    """Represent the outcome of a single company's insolvency prediction.

    Attributes:
        company_id (int): The identifier of the scored company.
        probability (float): The predicted probability of insolvency.
        predicted_class (int): The binary prediction (0 or 1), obtained by
            applying the decision threshold to `probability`.
        explanation (dict): A SHAP-based explanation of the prediction,
            mapping each feature to its original (encoded) value and
            Shapley contribution, sorted by descending absolute impact.

    """

    company_id: int
    probability: float
    predicted_class: int
    explanation: dict


def retrieve_company_data(session: Session, company_id: int) -> pd.DataFrame:
    """Retrieve the most recent star schema snapshot for a single company.

    Reuses the same feature-selection query as the training data loader,
    filtered down to a single company's latest snapshot via a parameterized
    query, never by string interpolation of the company_id.

    Args:
        session (Session): An active SQLAlchemy session, managed by the
            caller.
        company_id (int): The identifier of the company to retrieve.

    Returns:
        pd.DataFrame: A single-row DataFrame indexed by `legal_name`, with
            `company_id` and `snapshot_date` dropped as they are not model
            features.

    """
    query = f"""
        WITH dataset AS ({QUERY})
        SELECT * FROM dataset
        WHERE company_id = :company_id
        ORDER BY snapshot_date DESC
        LIMIT 1
    """
    df = pd.read_sql(query, con=session.bind, params={"company_id": company_id})

    df = df.set_index("legal_name")
    df = df.drop(columns=["company_id", "snapshot_date"])

    return df


def score_features(
    raw_features: pd.DataFrame,
    loaded_model: LoadedModel,
    threshold: float,
) -> tuple[float, int, dict]:
    """Apply preprocessing, scoring, and SHAP explanation to a single-row DataFrame.

    Shared by both `predict` (case A) and `predict_from_raw_data` (case B),
    which differ only in how `raw_features` is obtained (a star schema
    query versus caller-supplied data), the preprocessing and scoring
    steps that follow are identical, and are kept in one place so the two
    entry points cannot silently drift apart.

    Args:
        raw_features (pd.DataFrame): A single-row feature DataFrame, not
            yet encoded or scaled.
        loaded_model (LoadedModel): The model, encoder, scaler, and
            explainer bundle, as returned by
            `ml.inference.model_loader.load_model`.
        threshold (float): The decision threshold used to convert the
            predicted probability into a binary class.

    Returns:
        tuple[float, int, dict]: `(probability, predicted_class,
            explanation)`.

    """
    features, _ = handle_missing_and_encode(raw_features, encoder=loaded_model.encoder)

    if loaded_model.scaler is not None:
        features, _ = scale_features(features, scaler=loaded_model.scaler)

    probability = loaded_model.model.predict_proba(features)[0, 1]
    predicted_class = int(probability >= threshold)
    explanation = explain_prediction(loaded_model.explainer, features)

    return float(probability), predicted_class, explanation


def predict(
    session: Session,
    company_id: int,
    loaded_model: LoadedModel,
    threshold: float,
) -> PredictionResult:
    """Predict insolvency risk for an existing company using its latest snapshot.

    Retrieve the most recent star schema snapshot for the given company,
    apply the same fitted preprocessing transformations used during training
    (never refitting them), score the company, and explain the prediction
    via SHAP.

    Args:
        session (Session): An active SQLAlchemy session, managed by the
            caller.
        company_id (int): The identifier of the company to score.
        loaded_model (LoadedModel): The model, encoder, scaler, and
            explainer bundle, as returned by
            `ml.inference.model_loader.load_model`. Must come directly from
            a single `load_model` call, since its artifacts are only
            guaranteed to be mutually consistent when loaded together from
            the same run.
        threshold (float): The decision threshold used to convert the
            predicted probability into a binary class.

    Returns:
        PredictionResult: The predicted probability, class, and (once SHAP
            is integrated) an explanation of the prediction.

    """
    logger.info("Retrieving latest snapshot for company_id=%d...", company_id)
    company_df = retrieve_company_data(session, company_id)

    probability, predicted_class, explanation = score_features(
        company_df, loaded_model, threshold
    )

    logger.info(
        "Prediction for company_id=%d: probability=%.4f, predicted_class=%d",
        company_id,
        probability,
        predicted_class,
    )

    return PredictionResult(
        company_id=company_id,
        probability=float(probability),
        predicted_class=predicted_class,
        explanation=explanation,
    )


def predict_from_raw_data(
    request: InsolvencyPredictionRequest,
    loaded_model: LoadedModel,
    threshold: float,
) -> PredictionResult:
    """Predict insolvency risk for a company not present in the star schema.

    Builds a single-row feature DataFrame directly from an already-validated
    `InsolvencyPredictionRequest` (validation is the caller's
    responsibility, e.g. the agent layer constructing it from user-supplied
    data), then applies the same preprocessing, scoring, and SHAP
    explanation as `predict`. `company_age_days` is taken from the
    request's computed property (derived from `foundation_date` at request
    time), never supplied directly. Fields left unset on the request are
    passed through as NaN, which XGBoost's native missing-value handling
    (the model was trained with `handle_nan=False`) accounts for directly.

    Args:
        request (InsolvencyPredictionRequest): The validated ad hoc company
            data to score.
        loaded_model (LoadedModel): The model, encoder, scaler, and
            explainer bundle, as returned by
            `ml.inference.model_loader.load_model`.
        threshold (float): The decision threshold used to convert the
            predicted probability into a binary class.

    Returns:
        PredictionResult: The predicted probability, class, and a SHAP-based
            explanation of the prediction. `company_id` is `None`, since
            this company has no star schema identifier.

    """
    logger.info("Scoring ad hoc company data (no company_id)...")

    raw_features = pd.DataFrame(
        [
            {
                "company_age_days": request.company_age_days,
                "industry_sector": request.industry_sector,
                "registered_office_region": request.registered_office_region,
                "unpaid_ratio_trailing_90d": request.unpaid_ratio_trailing_90d,
                "total_outstanding_debt": request.total_outstanding_debt,
                "days_since_last_login": request.days_since_last_login,
                "login_velocity": request.login_velocity,
                "cash_to_debt_ratio": request.cash_to_debt_ratio,
                "ebitda": request.ebitda,
                "net_profit_margin": request.net_profit_margin,
                "leverage_ratio": request.leverage_ratio,
                "has_active_gas_contract": request.has_active_gas_contract,
                "has_active_electricity_contract": request.has_active_electricity_contract,  # noqa: E501
                "average_satisfaction_score": request.average_satisfaction_score,
                "billing_disputes_count": request.billing_disputes_count,
                "active_contracts_count": request.active_contracts_count,
            }
        ]
    )

    numeric_cols = [
        col
        for col in raw_features.columns
        if col not in ("industry_sector", "registered_office_region")
    ]
    raw_features[numeric_cols] = raw_features[numeric_cols].astype("float64")

    probability, predicted_class, explanation = score_features(
        raw_features, loaded_model, threshold
    )

    logger.info(
        "Ad hoc prediction: probability=%.4f, predicted_class=%d",
        probability,
        predicted_class,
    )

    return PredictionResult(
        company_id=None,
        probability=probability,
        predicted_class=predicted_class,
        explanation=explanation,
    )
