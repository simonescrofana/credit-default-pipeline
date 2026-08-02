"""Tests for ml.models.xgboost_model.

Cover the happy path of build_xgboost_model with default and custom
parameters (including early_stopping_rounds), and its independence
guarantee across calls (mirroring the same check already applied to
build_baseline_model).

"""

from xgboost import XGBClassifier

from ml.models.xgboost_model import (
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_EVAL_METRIC,
    DEFAULT_LEARNING_RATE_XGB,
    DEFAULT_MAX_DEPTH,
    DEFAULT_N_ESTIMATORS,
    build_xgboost_model,
)


def test_build_xgboost_model_returns_xgb_classifier_instance() -> None:
    """Verify build_xgboost_model returns an XGBClassifier instance."""
    model = build_xgboost_model()

    assert isinstance(model, XGBClassifier)


def test_build_xgboost_model_applies_default_params() -> None:
    """Verify default hyperparameters are applied when none are overridden."""
    model = build_xgboost_model()

    assert model.n_estimators == DEFAULT_N_ESTIMATORS
    assert model.max_depth == DEFAULT_MAX_DEPTH
    assert model.learning_rate == DEFAULT_LEARNING_RATE_XGB
    assert model.eval_metric == DEFAULT_EVAL_METRIC
    assert model.scale_pos_weight == 1.0
    assert model.early_stopping_rounds == DEFAULT_EARLY_STOPPING_ROUNDS


def test_build_xgboost_model_accepts_custom_params() -> None:
    """Verify custom hyperparameters override the defaults."""
    model = build_xgboost_model(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=5.25,
        eval_metric="logloss",
        early_stopping_rounds=10,
    )

    assert model.n_estimators == 50
    assert model.max_depth == 3
    assert model.learning_rate == 0.1
    assert model.scale_pos_weight == 5.25
    assert model.eval_metric == "logloss"
    assert model.early_stopping_rounds == 10


def test_build_xgboost_model_returns_independent_instances() -> None:
    """Verify each call to build_xgboost_model returns a fresh, independent instance."""
    model_1 = build_xgboost_model()
    model_2 = build_xgboost_model()

    assert model_1 is not model_2
