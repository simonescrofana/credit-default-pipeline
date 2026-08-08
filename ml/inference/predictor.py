"""Serve predictions for existing companies using the production XGBoost model.

Provide the inference entry points for scoring a company already present in
the star schema (`predict`) and for ad hoc, user-supplied company data not
in the database (`predict_from_raw_data`), reusing the same fitted
preprocessing transformations produced during training. Every prediction is
explained via SHAP (see `ml.evaluation.explainability`), since XGBoost is
the only model family served in production. Both entry points always score
against `loaded_model.threshold`, the decision threshold selected for the
loaded run, never a caller-supplied value.

"""

import datetime
import logging
from typing import NamedTuple

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from ml.dataset.loader import QUERY
from ml.dataset.preprocessing import handle_missing_and_encode, scale_features
from ml.dataset.split import TARGET_COLUMN
from ml.evaluation.explainability import explain_prediction
from ml.inference.model_loader import LoadedModel
from schemas.ml.insolvency_prediction import InsolvencyPredictionRequest

logger = logging.getLogger(__name__)


class PredictionResult(NamedTuple):
    """Represent the outcome of a single company's insolvency prediction.

    Attributes:
        company_id (int | None): The identifier of the scored company, or
            `None` for ad hoc data not present in the database (case_b).
        company_name (str | None): The company's canonical legal name from
            the database, or `None` for ad hoc data (case_b), which has no
            database record to draw a canonical name from.
        probability (float): The predicted probability of insolvency.
        predicted_class (int): The binary prediction (0 or 1), obtained by
            applying the decision threshold to `probability`.
        explanation (dict): A SHAP-based explanation of the prediction,
            mapping each feature to its original (encoded) value and
            Shapley contribution, sorted by descending absolute impact.

    """

    company_id: int | None
    company_name: str | None
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
            `company_id`, `snapshot_date`, and the target column dropped,
            as none of them are model features.

    """
    query = text(
        f"""
        WITH dataset AS ({QUERY})
        SELECT * FROM dataset
        WHERE company_id = :company_id
        ORDER BY snapshot_date DESC
        LIMIT 1
        """
    )
    # Wrapped in text() explicitly: a raw string with named (:name) parameters
    # passed to pd.read_sql is not reliably translated to the driver's own
    # paramstyle (e.g. psycopg2's %(name)s) unless SQLAlchemy's text() marks
    # it as a parameterized statement to interpret, rather than a literal
    # string to pass through almost as-is.
    df = pd.read_sql(query, con=session.bind, params={"company_id": company_id})

    df = df.set_index("legal_name")
    df = df.drop(columns=["company_id", "snapshot_date", TARGET_COLUMN])

    # A single-row result from pd.read_sql can leave a column with a NULL
    # value as dtype=object rather than float64 (pandas cannot always infer
    # a clean numeric dtype from one row); cast explicitly, same as
    # predict_from_raw_data does for its own single-row DataFrame.
    numeric_cols = [
        col
        for col in df.columns
        if col not in ("industry_sector", "registered_office_region")
    ]
    df[numeric_cols] = df[numeric_cols].astype("float64")

    return df


def score_features(
    raw_features: pd.DataFrame,
    loaded_model: LoadedModel,
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
        loaded_model (LoadedModel): The model, encoder, scaler, explainer,
            and decision threshold bundle, as returned by
            `ml.inference.model_loader.load_model`. The threshold used to
            convert the predicted probability into a binary class is
            always `loaded_model.threshold`, never a caller-supplied
            value, so a prediction can never be scored against a
            threshold inconsistent with the run it came from.

    Returns:
        tuple[float, int, dict]: `(probability, predicted_class,
            explanation)`.

    """
    features, _ = handle_missing_and_encode(
        raw_features, encoder=loaded_model.encoder, handle_nan=False
    )

    if loaded_model.scaler is not None:
        features, _ = scale_features(features, scaler=loaded_model.scaler)

    # XGBoost's inplace_predict validates column order, not just column
    # names — reorder to match the order the model was trained on (which
    # this project's own feature-construction order does not guarantee,
    # e.g. case_b's hand-written dict groups columns conceptually rather
    # than in training order) rather than relying on callers to get it
    # right themselves.
    features = features[loaded_model.model.feature_names_in_]

    probability = loaded_model.model.predict_proba(features)[0, 1]
    predicted_class = int(probability >= loaded_model.threshold)
    explanation = explain_prediction(loaded_model.explainer, features)

    return float(probability), predicted_class, explanation


def predict(
    session: Session,
    company_id: int,
    loaded_model: LoadedModel,
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
        loaded_model (LoadedModel): The model, encoder, scaler, explainer,
            and decision threshold bundle, as returned by
            `ml.inference.model_loader.load_model`. Must come directly from
            a single `load_model` call, since its artifacts are only
            guaranteed to be mutually consistent when loaded together from
            the same run.

    Returns:
        PredictionResult: The predicted probability, class, and (once SHAP
            is integrated) an explanation of the prediction.

    """
    logger.info("Retrieving latest snapshot for company_id=%d...", company_id)
    company_df = retrieve_company_data(session, company_id)
    company_name = str(company_df.index[0])

    probability, predicted_class, explanation = score_features(
        company_df,
        loaded_model,
    )

    logger.info(
        "Prediction for company_id=%d: probability=%.4f, predicted_class=%d",
        company_id,
        probability,
        predicted_class,
    )

    return PredictionResult(
        company_id=company_id,
        company_name=company_name,
        probability=float(probability),
        predicted_class=predicted_class,
        explanation=explanation,
    )


def predict_from_raw_data(
    request: InsolvencyPredictionRequest,
    loaded_model: LoadedModel,
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
        loaded_model (LoadedModel): The model, encoder, scaler, explainer,
            and decision threshold bundle, as returned by
            `ml.inference.model_loader.load_model`.

    Returns:
        PredictionResult: The predicted probability, class, and a SHAP-based
            explanation of the prediction. `company_id` is `None`, since
            this company has no star schema identifier.

    """
    logger.info("Scoring ad hoc company data (no company_id)...")

    now = datetime.datetime.now()

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
                # Derived server-side from the moment of the request, never
                # accepted from the caller — these represent the scoring
                # snapshot's period, the same role they play in training
                # (derived from snapshot_date there), not the company's
                # foundation date.
                "year": now.year,
                "quarter": (now.month - 1) // 3 + 1,
                "month": now.month,
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
        raw_features, loaded_model
    )

    logger.info(
        "Ad hoc prediction: probability=%.4f, predicted_class=%d",
        probability,
        predicted_class,
    )

    return PredictionResult(
        company_id=None,
        company_name=None,
        probability=probability,
        predicted_class=predicted_class,
        explanation=explanation,
    )
