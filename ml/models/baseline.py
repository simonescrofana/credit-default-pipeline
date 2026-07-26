"""Provide baseline linear model builders for binary classification.

Define factory functions to construct baseline scikit-learn models,
pre-configured with robust defaults (such as L1, L2 or ElasticNet regularization
via the SAGA solver) for credit risk and insolvency prediction.

"""

from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 202607


def build_baseline_model(
    l1_ratio: float = 0.0,
    C: float = 1.0,
    class_weight: str | dict = "balanced",
) -> LogisticRegression:
    """Build a baseline Logistic Regression model using the SAGA solver.

    Configure a `LogisticRegression` instance supporting L1, L2 or ElasticNet
    regularization via the SAGA solver. Designed as a linear baseline for
    imbalanced binary classification tasks.

    Args:
        l1_ratio (float, optional): The ElasticNet mixing parameter. Set to
            `0.0` for L2 penalty, `1.0` for L1 penalty, or a value in
            between for a combination. Defaults to `0.0`.
        C (float, optional): Inverse of regularization strength; smaller values
            specify stronger regularization. Defaults to `1.0`.
        class_weight (str | dict, optional): Weights associated with classes.
            If `"balanced"`, uses class frequencies to automatically adjust
            weights inversely proportional to class frequencies. Defaults to
            `"balanced"`.

    Returns:
        LogisticRegression: An un-fitted scikit-learn `LogisticRegression`
        estimator configured with the specified parameters and a fixed random
        seed.

    """
    return LogisticRegression(
        l1_ratio=l1_ratio,
        C=C,
        solver="saga",
        class_weight=class_weight,
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
