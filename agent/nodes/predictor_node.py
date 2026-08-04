"""Predictor node: scores companies using the production XGBoost model.

Provides `predict_case_a` and `predict_case_b`, one per prediction route
decided upstream, plus `predictor_node`, a thin orchestrator that dispatches
to the right one based on `state.route`. case_a scores one or more existing
companies, resolved by the extractor node into `resolved_company_ids`;
case_b scores ad hoc company data, validated by the extractor node into
`raw_prediction_input`. Both converge on the same output shape in
`AgentState`, so the responder node downstream doesn't need to know which
path a given result came from.

The model, its preprocessing artifacts, and its decision threshold are
loaded once per process (see `get_loaded_model`) rather than on every
node invocation, since `load_model` involves querying MLflow and
downloading artifacts, too costly to repeat per request.

"""

import logging
from functools import cache

from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from agent.state import AgentState
from database.connection import get_db
from ml.inference.model_loader import LoadedModel, load_model
from ml.inference.predictor import predict, predict_from_raw_data
from schemas.ml.insolvency_prediction import InsolvencyPredictionRequest

logger = logging.getLogger(__name__)


@cache
def get_loaded_model() -> LoadedModel:
    """Load and cache the production model bundle for the lifetime of the process.

    Returns:
        LoadedModel: The model, encoder, scaler, explainer, and decision
            threshold bundle, loaded once and reused across every
            prediction node invocation.

    """
    return load_model()  # pragma: no cover


def predict_case_a(state: AgentState, loaded_model: LoadedModel) -> dict:
    """Score every company resolved by the extractor node (case_a).

    Args:
        state (AgentState): The current graph state. Reads
            `resolved_company_ids`.
        loaded_model (LoadedModel): The model, encoder, scaler, explainer,
            and decision threshold bundle used to score each company.

    Returns:
        dict: A partial state update appending one entry to
            `prediction_results` per successfully scored company, and one
            entry to `prediction_errors` per company that failed to score
            due to a database error.

    """
    results: list[dict] = list(state.prediction_results)
    errors: list[str] = list(state.prediction_errors)

    session_gen = get_db()
    session = next(session_gen)
    try:
        for company_id in state.resolved_company_ids or []:
            try:
                result = predict(
                    session=session,
                    company_id=company_id,
                    loaded_model=loaded_model,
                )
                results.append(result._asdict())
            except SQLAlchemyError:
                logger.exception("Error while scoring company_id=%d!", company_id)
                errors.append(
                    f"Could not score company_id={company_id} due to a database error."
                )
    finally:
        session_gen.close()

    return {"prediction_results": results, "prediction_errors": errors}


def predict_case_b(state: AgentState, loaded_model: LoadedModel) -> dict:
    """Score the ad hoc company data extracted from the prompt (case_b).

    Args:
        state (AgentState): The current graph state. Reads
            `raw_prediction_input`.
        loaded_model (LoadedModel): The model, encoder, scaler, explainer,
            and decision threshold bundle used to score the company.

    Returns:
        dict: A partial state update appending the prediction to
            `prediction_results`, or an entry to `prediction_errors` if
            `raw_prediction_input` fails validation.

    """
    results: list[dict] = list(state.prediction_results)
    errors: list[str] = list(state.prediction_errors)

    try:
        request = InsolvencyPredictionRequest(**(state.raw_prediction_input or {}))
        result = predict_from_raw_data(request=request, loaded_model=loaded_model)
        results.append(result._asdict())
    except ValidationError:
        logger.warning("predict_case_b called with invalid raw_prediction_input.")

    return {"prediction_results": results, "prediction_errors": errors}


def predictor_node(state: AgentState) -> dict:
    """Dispatch to the scoring function for the route decided upstream.

    Args:
        state (AgentState): The current graph state. Reads `route` to
            decide whether to call `predict_case_a` or `predict_case_b`.

    Returns:
        dict: The partial state update produced by the dispatched
            function. See `predict_case_a` and `predict_case_b`.

    """
    loaded_model = get_loaded_model()

    if state.route == "case_a":
        return predict_case_a(state, loaded_model)
    return predict_case_b(state, loaded_model)
