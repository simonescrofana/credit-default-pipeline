"""Split datasets for time-series evaluation and cross-validation.

Provide utilities to isolate temporal holdout test sets and generate time-series
cross-validation folds for machine learning model training and validation.

"""

import logging
from typing import NamedTuple

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

N_TEST_MONTHS = 12
N_CV_SPLITS = 5
TARGET_COLUMN = "is_insolvent"


class Fold(NamedTuple):
    """Represent a single cross-validation split containing features and targets.

    Attributes:
        X_train (pd.DataFrame): Training feature matrix for the fold.
        y_train (pd.Series): Training target labels for the fold.
        X_val (pd.DataFrame): Validation feature matrix for the fold.
        y_val (pd.Series): Validation target labels for the fold.

    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series


def isolate_test_set(
    df: pd.DataFrame, n_test_months: int = N_TEST_MONTHS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate a temporal test set from the input DataFrame based on snapshot dates.

    Filter the dataset into a historical training/validation subset and a recent
    holdout test set using the last `n_test_months` unique snapshot dates as the
    cutoff threshold.

    Args:
        df (pd.DataFrame): The input DataFrame indexed by `company_id` and
            `snapshot_date`.
        n_test_months (int, optional): The number of recent unique snapshot months
            to reserve for testing. Defaults to `N_TEST_MONTHS`.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing `(df_remaining, df_test)`.

    """
    logger.info(
        "Starting test separation. Test dataset contains the data of the last"
        "{%d} months.",
        n_test_months,
    )
    snapshot_dates = df.index.get_level_values("snapshot_date")
    cutoff_date = snapshot_dates.unique().sort_values()[-n_test_months]

    df_remaining = df[snapshot_dates < cutoff_date]
    df_test = df[snapshot_dates >= cutoff_date]

    logger.info("Test dataset extracted.")
    return df_remaining, df_test


def generate_cv_folds(df: pd.DataFrame, n_splits: int = N_CV_SPLITS) -> list[Fold]:
    """Create cross-validation folds using time-series splitting logic.

    Partition the input dataset into sequential, expanding-window temporal
    splits, separating target labels from feature sets for each training and
    validation fold.

    Args:
        df (pd.DataFrame): The feature matrix and target dataset.
        n_splits (int, optional): The number of time-series splits to generate.
            Defaults to `N_CV_SPLITS`.

    Returns:
        list[Fold]: A list of `Fold` instances containing `X_train`, `y_train`,
        `X_val`, and `y_val` objects.

    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    folds = []
    count = 0
    logger.info(
        "Generating %d cross-validation folds splitting train and validation data...",
        n_splits,
    )

    unique_dates = df.index.get_level_values("snapshot_date").unique().sort_values()

    for train_date_idx, val_date_idx in tscv.split(unique_dates):
        count += 1

        train_dates = unique_dates[train_date_idx]
        val_dates = unique_dates[val_date_idx]

        snapshot_dates = df.index.get_level_values("snapshot_date")
        train_fold = df[snapshot_dates.isin(train_dates)]
        val_fold = df[snapshot_dates.isin(val_dates)]

        X_train = train_fold.drop(columns=TARGET_COLUMN)
        y_train = train_fold[TARGET_COLUMN]
        X_val = val_fold.drop(columns=TARGET_COLUMN)
        y_val = val_fold[TARGET_COLUMN]

        folds.append(Fold(X_train, y_train, X_val, y_val))
        logger.info("Built fold number %d.", count)

    logger.info("Train and validation data splitted. Returning %d folds...", n_splits)
    return folds


def train_val_test_split(
    df: pd.DataFrame, n_test_months: int = N_TEST_MONTHS, n_splits: int = N_CV_SPLITS
) -> tuple[list[Fold], pd.DataFrame]:
    """Execute the full dataset splitting pipeline for cross-validation and testing.

    Isolate a temporal holdout test set using the specified number of recent
    months, then partition the remaining historical data into time-series
    cross-validation folds.

    Args:
        df (pd.DataFrame): The complete dataset indexed by `company_id` and
            `snapshot_date`.
        n_test_months (int, optional): The number of recent unique snapshot months
            to reserve for the test set. Defaults to `N_TEST_MONTHS`.
        n_splits (int, optional): The number of time-series cross-validation
            splits to generate from historical data. Defaults to `N_CV_SPLITS`.

    Returns:
        tuple[list[Fold], pd.DataFrame]: A tuple containing a list of `Fold`
        instances for cross-validation and the holdout test DataFrame (`df_test`).

    """
    df_remaining, df_test = isolate_test_set(df=df, n_test_months=n_test_months)
    train_val_folds = generate_cv_folds(df=df_remaining, n_splits=n_splits)

    return train_val_folds, df_test
