"""Serve predictions for existing companies using a trained model.

Provide the inference entry point for scoring a company already present in
the star schema, reusing the same fitted preprocessing transformations
produced during training. Support for scoring companies not present in the
database (arbitrary user-supplied data) is deferred until SHAP is
integrated, since which features are safe to leave as optional depends on
their actual contribution to the model, not assumption.

"""

import logging
from typing import NamedTuple

import pandas as pd
from sqlalchemy.orm import Session

from ml.dataset.loader import QUERY
from ml.dataset.preprocessing import handle_missing_and_encode, scale_features
from ml.inference.model_loader import LoadedModel

logger = logging.getLogger(__name__)


class PredictionResult(NamedTuple):
    """Represent the outcome of a single company's insolvency prediction.

    Attributes:
        company_id (int): The identifier of the scored company.
        probability (float): The predicted probability of insolvency.
        predicted_class (int): The binary prediction (0 or 1), obtained by
            applying the decision threshold to `probability`.
        explanation (dict | None): A SHAP-based explanation of the
            prediction, mapping each feature to its original value and
            Shapley contribution. `None` until SHAP is integrated.

    """

    company_id: int
    probability: float
    predicted_class: int
    explanation: dict | None = None


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


def predict(
    session: Session,
    company_id: int,
    loaded_model: LoadedModel,
    threshold: float,
) -> PredictionResult:
    """Predict insolvency risk for an existing company using its latest snapshot.

    Retrieve the most recent star schema snapshot for the given company,
    apply the same fitted preprocessing transformations used during training
    (never refitting them), and score the company with the provided model.

    Args:
        session (Session): An active SQLAlchemy session, managed by the
            caller.
        company_id (int): The identifier of the company to score.
        loaded_model (LoadedModel): The model, encoder, and scaler bundle,
            as returned by `ml.inference.model_loader.load_model`. Must
            come directly from a single `load_model` call, since its three
            artifacts are only guaranteed to be mutually consistent when
            loaded together from the same run.
        threshold (float): The decision threshold used to convert the
            predicted probability into a binary class.

    Returns:
        PredictionResult: The predicted probability, class, and (once SHAP
            is integrated) an explanation of the prediction.

    """
    logger.info("Retrieving latest snapshot for company_id=%d...", company_id)
    company_df = retrieve_company_data(session, company_id)

    features, _ = handle_missing_and_encode(company_df, encoder=loaded_model.encoder)

    if loaded_model.scaler is not None:
        features, _ = scale_features(features, scaler=loaded_model.scaler)

    probability = loaded_model.model.predict_proba(features)[0, 1]
    predicted_class = int(probability >= threshold)

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
        explanation=None,
    )
