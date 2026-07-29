"""Load a trained model and its fitted preprocessing artifacts from MLflow.

Provide a single entry point for retrieving the model, encoder, and scaler
produced by a training run, bundled together since the inference layer must
apply the exact same fitted transformations used during training.

"""

import logging
from typing import NamedTuple

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class LoadedModel(NamedTuple):
    """Bundle a fitted model with its fitted preprocessing artifacts.

    Attributes:
        model (BaseEstimator): The fitted model.
        encoder (OneHotEncoder): The OneHotEncoder fitted during training.
        scaler (StandardScaler | None): The StandardScaler fitted during
            training, or None if scaling was not applied for this model
            family (e.g. XGBoost).

    """

    model: BaseEstimator
    encoder: OneHotEncoder
    scaler: StandardScaler | None


def load_model(experiment_name: str, run_id: str | None = None) -> LoadedModel:
    """Load the final model, encoder, and scaler from a training run.

    If `run_id` is not provided, the most recent `final_model` run under the
    given experiment is used. This makes retraining and serving reproducible
    by default (always the latest validated run) without requiring a
    separately maintained pointer to "the good run", while still allowing a
    specific run to be pinned when needed (run_id can be retrived from the mlflow
    ui). Whether a scaler was logged is determined by inspecting the run's actual
    artifacts.

    Args:
        experiment_name (str): The MLflow experiment to load the model from
            (e.g. "baseline").
        run_id (str | None, optional): A specific MLflow run ID to load
            instead of searching for the latest one. Defaults to `None`.

    Returns:
        LoadedModel: The fitted model, encoder, and scaler (`scaler` is
            `None` if the run has no scaler artifact).

    """
    if run_id is None:
        logger.info(
            "No run_id provided, searching for the latest final_model run "
            "in experiment '%s'...",
            experiment_name,
        )
        experiment = mlflow.get_experiment_by_name(experiment_name)
        final_runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.mlflow.runName LIKE 'final_model%'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        run_id = final_runs.iloc[0]["run_id"]

    logger.info("Loading model artifacts from run '%s'...", run_id)

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    encoder = mlflow.sklearn.load_model(f"runs:/{run_id}/encoder")

    client = MlflowClient()
    artifact_names = {artifact.path for artifact in client.list_artifacts(run_id)}

    scaler = None
    if "scaler" in artifact_names:
        scaler = mlflow.sklearn.load_model(f"runs:/{run_id}/scaler")
    else:
        logger.info("No scaler artifact found for run '%s' (scale=False).", run_id)

    logger.info("Model artifacts loaded successfully.")
    return LoadedModel(model=model, encoder=encoder, scaler=scaler)
