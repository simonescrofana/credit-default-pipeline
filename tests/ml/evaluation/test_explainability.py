"""Test suite for SHAP-based explainability of the production XGBoost model.

Cover building the TreeExplainer, computing per-feature contributions for a
single prediction (including the descending-by-magnitude ordering), and
computing the raw Explanation object used for plotting, for both a single
row and multiple rows.

"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from ml.evaluation.explainability import (
    build_explainer,
    explain_prediction,
    explain_prediction_for_plot,
)


@patch("ml.evaluation.explainability.shap.TreeExplainer")
def test_build_explainer_wraps_model_in_tree_explainer(mock_tree_explainer) -> None:
    """Verify build_explainer constructs a TreeExplainer around the given model."""
    fake_model = MagicMock(spec=XGBClassifier)
    fake_explainer = MagicMock()
    mock_tree_explainer.return_value = fake_explainer

    result = build_explainer(fake_model)

    mock_tree_explainer.assert_called_once_with(fake_model)
    assert result is fake_explainer


def test_explain_prediction_returns_value_and_shap_per_feature() -> None:
    """Verify explain_prediction pairs feature's original and SHAP values."""
    features = pd.DataFrame({"leverage_ratio": [0.8], "days_past_due": [12]})

    fake_shap_values = MagicMock()
    fake_shap_values.values = np.array([[0.3, -0.1]])
    fake_explainer = MagicMock(return_value=fake_shap_values)

    result = explain_prediction(fake_explainer, features)

    assert result["leverage_ratio"] == {"value": 0.8, "shap": 0.3}
    assert result["days_past_due"] == {"value": 12, "shap": -0.1}


def test_explain_prediction_sorts_by_descending_absolute_shap() -> None:
    """Verify the returned dict is ordered by |shap| descending."""
    features = pd.DataFrame(
        {
            "small_positive": [1],
            "large_negative": [1],
            "medium_positive": [1],
        }
    )

    fake_shap_values = MagicMock()
    fake_shap_values.values = np.array([[0.05, -0.9, 0.4]])
    fake_explainer = MagicMock(return_value=fake_shap_values)

    result = explain_prediction(fake_explainer, features)

    assert list(result.keys()) == [
        "large_negative",
        "medium_positive",
        "small_positive",
    ]


def test_explain_prediction_for_plot_returns_explanation_for_single_multi_row() -> None:
    """Verify explain_prediction_for_plot works for both single- and multi-row cases."""
    single_row = pd.DataFrame({"leverage_ratio": [0.8]})
    multi_row = pd.DataFrame({"leverage_ratio": [0.8, 0.3, 0.5]})

    fake_explanation_single = MagicMock()
    fake_explanation_multi = MagicMock()
    fake_explainer = MagicMock(
        side_effect=[fake_explanation_single, fake_explanation_multi]
    )

    result_single = explain_prediction_for_plot(fake_explainer, single_row)
    result_multi = explain_prediction_for_plot(fake_explainer, multi_row)

    assert result_single is fake_explanation_single
    assert result_multi is fake_explanation_multi

    # DataFrames don't support == inside Mock's default call comparison
    # (it raises instead of returning a bool), so compare the two recorded
    # calls explicitly with .equals() rather than assert_any_call.
    called_with = [call.args[0] for call in fake_explainer.call_args_list]
    # assert called_with[0] == single_row
    # assert called_with[1] == multi_row
    # assert with == does not work with mocking in this case -> equals
    assert called_with[0].equals(single_row)
    assert called_with[1].equals(multi_row)
