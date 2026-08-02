"""Define the XGBoost model family as a scikit-learn-compatible factory function.

Unlike the PyTorch MLP, XGBClassifier natively implements the fit/predict_proba
interface expected by ``ml.training.trainer``, so no wrapper is needed here —
only a factory function that builds a fresh, unfitted instance with the
project's default hyperparameters, mirroring ``ml.models.baseline``.

"""

from xgboost import XGBClassifier

RANDOM_STATE = 202607

DEFAULT_N_ESTIMATORS = 200
DEFAULT_MAX_DEPTH = 10
DEFAULT_LEARNING_RATE_XGB = 0.05
DEFAULT_EVAL_METRIC = "aucpr"
DEFAULT_EARLY_STOPPING_ROUNDS = 30


def build_xgboost_model(
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    learning_rate: float = DEFAULT_LEARNING_RATE_XGB,
    scale_pos_weight: float = 1.0,
    eval_metric: str = DEFAULT_EVAL_METRIC,
    early_stopping_rounds: int | None = DEFAULT_EARLY_STOPPING_ROUNDS,
) -> XGBClassifier:
    """Build a fresh, unfitted XGBClassifier instance.

    Mirrors ``ml.models.baseline.build_baseline_model``'s role as a factory
    function passed to ``ml.training.trainer``, ensuring a new, independent
    model is created for every fold instead of reusing mutable state. No
    feature scaling is required, since tree-based splits are invariant to
    monotonic transformations of the input features.

    Args:
        n_estimators (int, optional): The number of sequentially boosted
            trees. Defaults to ``DEFAULT_N_ESTIMATORS``.
        max_depth (int, optional): The maximum depth of each tree,
            controlling model complexity. Defaults to ``DEFAULT_MAX_DEPTH``.
        learning_rate (float, optional): The shrinkage applied to each
            tree's contribution to the ensemble. Defaults to
            ``DEFAULT_LEARNING_RATE_XGB``.
        scale_pos_weight (float, optional): The positive class weight,
            addressing class imbalance. Computed as the ratio of negative
            to positive examples (``n_negatives / n_positives``). Defaults
            to ``1.0`` (no reweighting).
        eval_metric (str, optional): The metric XGBoost monitors internally
            during training, relevant for early stopping. Defaults to
            ``DEFAULT_EVAL_METRIC`` (Area Under the Precision-Recall Curve),
            consistent with the metric used throughout this project's
            evaluation for imbalanced classification. Defaults to
            ``DEFAULT_EVAL_METRIC``.
        early_stopping_rounds (int | None, optional): The number of consecutive
            rounds without improvement in `eval_metric` on the validation set
            before training stops early. Only takes effect when `eval_set` is
            passed to `.fit()` (done only during cross-validation, never for the
            final fit on the full training set, to avoid ever using held-out data
            to influence training). Defaults to `DEFAULT_EARLY_STOPPING_ROUNDS`.

    Returns:
        XGBClassifier: A fresh, unfitted model instance.

    """
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        eval_metric=eval_metric,
        early_stopping_rounds=early_stopping_rounds,
        random_state=RANDOM_STATE,
    )
