"""Explain individual XGBoost predictions using SHAP (TreeExplainer).

Provides per-prediction feature attributions for the model actually served
in production. XGBoost is the only model family explained here: since only
the best-performing model is ever deployed, and prior benchmarking already
points to XGBoost, investing explainability effort in the baseline or MLP
would not translate into production value.

"""

import pandas as pd
import shap
from xgboost import XGBClassifier


def build_explainer(model: XGBClassifier) -> shap.TreeExplainer:
    """Build a TreeExplainer for a fitted XGBoost model.

    Construction does upfront work analyzing the tree structure, so this
    should be called once when the model is loaded (e.g. alongside the
    model itself in `ml.inference.model_loader.LoadedModel`), never
    per-prediction.

    Args:
        model (XGBClassifier): A fitted XGBoost classifier.

    Returns:
        shap.TreeExplainer: An explainer bound to the given model, ready to
            be reused across any number of predictions.

    """
    return shap.TreeExplainer(model)


def explain_prediction(
    explainer: shap.TreeExplainer,
    features: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Compute per-feature SHAP contributions for a single prediction.

    Uses TreeExplainer's exact computation path (no sampling/approximation
    needed for tree-based models), and natively supports the missing values
    XGBoost was trained on directly, with no additional preprocessing.

    Args:
        explainer (shap.TreeExplainer): An explainer built via
            `build_explainer`.
        features (pd.DataFrame): A single-row feature DataFrame, already
            preprocessed exactly as passed to `model.predict_proba` (i.e.
            after categorical encoding, before any scaling — XGBoost does
            not require scaling).

    Returns:
        dict[str, dict[str, float]]: Feature name -> `{"value": ..., "shap":
            ...}`, where `value` is the feature's original (encoded) value
            for this prediction and `shap` is its Shapley contribution.
            Sorted by descending absolute SHAP contribution, so the most
            influential features come first.

    """
    shap_values = explainer(features)

    contributions = {
        feature_name: {
            "value": float(features.iloc[0][feature_name]),
            "shap": float(shap_values.values[0][i]),
        }
        for i, feature_name in enumerate(features.columns)
    }

    return dict(
        sorted(
            contributions.items(),
            key=lambda item: abs(item[1]["shap"]),
            reverse=True,
        )
    )


def explain_prediction_for_plot(
    explainer: shap.TreeExplainer,
    features: pd.DataFrame,
) -> shap.Explanation:
    """Compute the raw SHAP Explanation object, for plotting.

    Unlike `explain_prediction`, which returns a flattened dict for a single
    prediction, this keeps the full `shap.Explanation` object intact
    (`.values`, `.base_values`, `.data`) — required by SHAP's own plotting
    functions, several of which need the base value in addition to each
    feature's contribution. `features` can hold any number of rows: a
    single row for a per-prediction plot (e.g. a waterfall explaining one
    company), or the full dataset for a corpus-level plot (e.g. a beeswarm
    summarizing feature importance across the whole test set) — the shape
    of the returned `Explanation.values` follows accordingly.

    Intended to be called only by the presentation layer (e.g. Streamlit)
    or by offline analysis notebooks, when a plot is actually requested,
    kept separate from `explain_prediction`, so the production inference
    path (called on every single-row prediction) never pays for data it
    won't use.

    Args:
        explainer (shap.TreeExplainer): An explainer built via
            `build_explainer`.
        features (pd.DataFrame): A feature DataFrame (one or many rows),
            already preprocessed exactly as passed to `model.predict_proba`.

    Returns:
        shap.Explanation: The raw SHAP explanation for the given rows,
            ready to be passed directly to SHAP's plotting functions.

    """
    return explainer(features)
