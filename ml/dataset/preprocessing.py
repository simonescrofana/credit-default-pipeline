"""Preprocess datasets for model training and evaluation.

Provide utilities to handle missing values with informative flags, encode
categorical features, scale numeric features, and orchestrate the full
preprocessing pipeline across cross-validation folds.

"""

import logging

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.dataset.split import TARGET_COLUMN, Fold

logger = logging.getLogger(__name__)

FINANCIAL_COLS = ["leverage_ratio", "cash_to_debt_ratio", "net_profit_margin", "ebitda"]
LOGIN_COLS = ["days_since_last_login", "login_velocity"]
SATISFACTION_COL = "average_satisfaction_score"

FINANCIAL_FALLBACK = 0.0
LOGIN_FALLBACK = -1
SATISFACTION_FALLBACK = -1

CATEGORICAL_COLS = ["industry_sector", "registered_office_region"]

NUMERIC_COLS_TO_SCALE = [
    "company_age_days",
    "active_contracts_count",
    "leverage_ratio",
    "cash_to_debt_ratio",
    "net_profit_margin",
    "ebitda",
    # "avg_dpd_trailing_90d",
    "unpaid_ratio_trailing_90d",
    "total_outstanding_debt",
    "days_since_last_login",
    "login_velocity",
    "billing_disputes_count",
    "average_satisfaction_score",
    "year",
    "quarter",
    "month",
]


def handle_missing_and_encode(
    df: pd.DataFrame, encoder: OneHotEncoder | None = None, handle_nan: bool = True
) -> tuple[pd.DataFrame, OneHotEncoder]:
    """Handle informative missing values and one-hot encode categorical columns.

    Create binary flags marking the absence of financial, login, and
    satisfaction data before replacing the missing values with group-specific
    fallback constants. Then one-hot encode the categorical columns, fitting a
    new encoder if none is provided.

    Args:
        df (pd.DataFrame): The input feature DataFrame.
        encoder (OneHotEncoder | None, optional): A previously fitted encoder.
            If ``None``, a new encoder is fitted on ``df``. Defaults to
            ``None``.
        handle_nan (bool, optional): Whether to create informative missing
            value flags and apply group-specific fallback constants. Set to
            ``False`` for model families with native missing value handling
            (e.g. XGBoost), which learn the optimal split direction for
            missing data directly rather than relying on an explicit
            imputation strategy. Defaults to ``True``.

    Returns:
        tuple[pd.DataFrame, OneHotEncoder]: The transformed DataFrame and the
        (possibly newly fitted) encoder.

    """
    logger.info("Handling missing values and encoding categorical columns...")
    df = df.copy()

    if handle_nan:
        df["has_recent_financials"] = df[FINANCIAL_COLS].notna().all(axis=1).astype(int)
        df["has_login_activity"] = df[LOGIN_COLS].notna().all(axis=1).astype(int)
        df["has_recent_satisfaction_score"] = df[SATISFACTION_COL].notna().astype(int)

        df[FINANCIAL_COLS] = df[FINANCIAL_COLS].fillna(FINANCIAL_FALLBACK)
        df[LOGIN_COLS] = df[LOGIN_COLS].fillna(LOGIN_FALLBACK)
        df[SATISFACTION_COL] = df[SATISFACTION_COL].fillna(SATISFACTION_FALLBACK)

    if encoder is None:
        logger.info("No encoder provided, fitting a new OneHotEncoder...")
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded = encoder.fit_transform(df[CATEGORICAL_COLS])
    else:
        encoded = encoder.transform(df[CATEGORICAL_COLS])

    encoded_cols = encoder.get_feature_names_out(CATEGORICAL_COLS)
    encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)

    df = df.drop(columns=CATEGORICAL_COLS)
    df = pd.concat([df, encoded_df], axis=1)

    logger.info("Missing value handling and categorical encoding completed.")
    return df, encoder


def scale_features(
    df: pd.DataFrame, scaler: StandardScaler | None = None
) -> tuple[pd.DataFrame, StandardScaler]:
    """Scale numeric feature columns using standardization.

    Fit a new StandardScaler if none is provided, otherwise apply the
    previously fitted scaler. Boolean, one-hot encoded, and missing-value
    flag columns are intentionally excluded from scaling.

    Args:
        df (pd.DataFrame): The input feature DataFrame.
        scaler (StandardScaler | None, optional): A previously fitted scaler.
            If ``None``, a new scaler is fitted on ``df``. Defaults to
            ``None``.

    Returns:
        tuple[pd.DataFrame, StandardScaler]: The transformed DataFrame and
            the (possibly newly fitted) scaler.

    """
    logger.info("Scaling numeric feature columns...")
    df = df.copy()

    if scaler is None:
        logger.info("No scaler provided, fitting a new StandardScaler...")
        scaler = StandardScaler()
        df[NUMERIC_COLS_TO_SCALE] = scaler.fit_transform(df[NUMERIC_COLS_TO_SCALE])
    else:
        df[NUMERIC_COLS_TO_SCALE] = scaler.transform(df[NUMERIC_COLS_TO_SCALE])

    logger.info("Feature scaling completed.")
    return df, scaler


def preprocess_train_folds(
    folds: list[Fold], scale: bool = True, handle_nan: bool = True
) -> list[Fold]:
    """Orchestrate the full preprocessing pipeline across cross-validation folds.

    For each fold, fit missing-value handling and categorical encoding on the
    training split, then apply the same fitted transformations to the
    validation split. Optionally repeat the same fit/transform pattern for
    feature scaling.

    Args:
        folds (list[Fold]): The cross-validation folds produced by
            ``ml.dataset.split.generate_cv_folds``.
        scale (bool, optional): Whether to apply feature scaling, required
            for the baseline and PyTorch models but not for XGBoost. Defaults
            to ``True``.
        handle_nan (bool, optional): Whether to create informative missing
            value flags and apply group-specific fallback constants. Set to
            ``False`` for model families with native missing value handling
            (e.g. XGBoost), which learn the optimal split direction for
            missing data directly rather than relying on an explicit
            imputation strategy. Defaults to ``True``.

    Returns:
        list[Fold]: The list of preprocessed folds, ready for model training.

    """
    logger.info("Starting preprocessing for %d folds (scale=%s)...", len(folds), scale)
    processed_folds = []

    for i, fold in enumerate(folds, start=1):
        X_train, encoder = handle_missing_and_encode(
            fold.X_train, handle_nan=handle_nan
        )
        X_val, _ = handle_missing_and_encode(
            fold.X_val, encoder=encoder, handle_nan=handle_nan
        )

        if scale:
            X_train, scaler = scale_features(X_train)
            X_val, _ = scale_features(X_val, scaler=scaler)

        processed_folds.append(Fold(X_train, fold.y_train, X_val, fold.y_val))
        logger.info("Fold %d/%d preprocessed.", i, len(folds))

    logger.info("Preprocessing completed for all folds.")
    return processed_folds


def preprocess_test_set(
    df_train_full: pd.DataFrame,
    df_test: pd.DataFrame,
    scale: bool = True,
    handle_nan: bool = True,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    OneHotEncoder,
    StandardScaler | None,
]:
    """Preprocess the final holdout test set using full training data.

    Fit missing-value handling, categorical encoding, and (optionally) feature
    scaling on the entire training/cross-validation block, then apply the same
    fitted transformations to the held-out test set. This must be called only
    after model selection is complete, using the full pre-test data rather than
    a single cross-validation fold. The fitted encoder (and scaler, if used)
    are also returned so they can be persisted alongside the final model,
    since the inference layer must apply the exact same fitted transformations
    to new data rather than refitting them.

    Args:
        df_train_full (pd.DataFrame): The full training/CV DataFrame (the
            `df_remaining` output of `isolate_test_set`), features and
            target still combined.
        df_test (pd.DataFrame): The held-out test DataFrame, features and
            target still combined.
        scale (bool, optional): Whether to apply feature scaling. Defaults to
            `True`.
        handle_nan (bool, optional): Whether to create informative missing
            value flags and apply group-specific fallback constants. Set to
            ``False`` for model families with native missing value handling
            (e.g. XGBoost), which learn the optimal split direction for
            missing data directly rather than relying on an explicit
            imputation strategy. Defaults to ``True``.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, OneHotEncoder,
        StandardScaler | None]:
            `(X_train_full, y_train_full, X_test, y_test, encoder, scaler)`,
            ready for a final model fit and evaluation. `scaler` is `None`
            if `scale=False` (e.g. for tree-based models like XGBoost).

    """
    y_train_full = df_train_full[TARGET_COLUMN]
    y_test = df_test[TARGET_COLUMN]

    X_train_full = df_train_full.drop(columns=TARGET_COLUMN)
    X_test = df_test.drop(columns=TARGET_COLUMN)

    X_train_full, encoder = handle_missing_and_encode(
        X_train_full, handle_nan=handle_nan
    )
    X_test, _ = handle_missing_and_encode(
        X_test, encoder=encoder, handle_nan=handle_nan
    )

    scaler = None
    if scale:
        X_train_full, scaler = scale_features(X_train_full)
        X_test, _ = scale_features(X_test, scaler=scaler)

    return X_train_full, y_train_full, X_test, y_test, encoder, scaler
