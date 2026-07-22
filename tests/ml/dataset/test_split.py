"""Test the dataset splitting logic for cross-validation and evaluation.

Provide unit tests to verify temporal test set isolation, cross-validation
fold generation, data integrity, and end-to-end splitting pipeline orchestration.

"""

import pandas as pd
import pytest

from ml.dataset.split import (
    Fold,
    generate_cv_folds,
    isolate_test_set,
    train_val_test_split,
)


@pytest.fixture
def synthetic_star_schema_df() -> pd.DataFrame:
    """Generate a multi-indexed DataFrame spanning 18 months for test usage.

    Returns:
        pd.DataFrame: A synthetic DataFrame indexed by company_id and snapshot_date,
        containing sample feature columns and the target label.

    """
    dates = pd.date_range("2025-01-31", periods=18, freq="ME")
    company_ids = [1, 2, 3]

    rows = []
    for date in dates:
        for company_id in company_ids:
            rows.append(
                {
                    "company_id": company_id,
                    "snapshot_date": date,
                    "leverage_ratio": 0.5,
                    "is_insolvent": 0,
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values("snapshot_date")
    df = df.set_index(["company_id", "snapshot_date"])
    return df


def test_isolate_test_set_splits_on_correct_month_count(
    synthetic_star_schema_df,
) -> None:
    """Verify that isolate_test_set correctly allocates specified test months.

    Args:
        synthetic_star_schema_df (pd.DataFrame): Synthetic multi-indexed DataFrame.

    """
    n_test_months = 4

    df_remaining, df_test = isolate_test_set(
        synthetic_star_schema_df, n_test_months=n_test_months
    )

    remaining_dates = df_remaining.index.get_level_values("snapshot_date").unique()
    test_dates = df_test.index.get_level_values("snapshot_date").unique()

    assert len(test_dates) == n_test_months
    assert len(remaining_dates) == 18 - n_test_months


def test_isolate_test_set_has_no_temporal_overlap(synthetic_star_schema_df) -> None:
    """Ensure no date overlap exists between remaining and holdout test sets.

    Args:
        synthetic_star_schema_df (pd.DataFrame): Synthetic multi-indexed DataFrame.

    """
    df_remaining, df_test = isolate_test_set(synthetic_star_schema_df, n_test_months=4)

    max_remaining_date = df_remaining.index.get_level_values("snapshot_date").max()
    min_test_date = df_test.index.get_level_values("snapshot_date").min()

    assert max_remaining_date < min_test_date


def test_isolate_test_set_preserves_total_row_count(synthetic_star_schema_df) -> None:
    """Confirm that the total row count is preserved across split subsets.

    Args:
        synthetic_star_schema_df (pd.DataFrame): Synthetic multi-indexed DataFrame.

    """
    df_remaining, df_test = isolate_test_set(synthetic_star_schema_df, n_test_months=4)

    assert len(df_remaining) + len(df_test) == len(synthetic_star_schema_df)


def test_generate_cv_folds_returns_expected_number_of_folds(
    synthetic_star_schema_df,
) -> None:
    """Check that generate_cv_folds produces the requested fold count and types.

    Args:
        synthetic_star_schema_df (pd.DataFrame): Synthetic multi-indexed DataFrame.

    """
    n_splits = 3

    folds = generate_cv_folds(synthetic_star_schema_df, n_splits=n_splits)

    assert len(folds) == n_splits
    assert all(isinstance(fold, Fold) for fold in folds)


def test_generate_cv_folds_excludes_target_from_features(
    synthetic_star_schema_df,
) -> None:
    """Ensure target columns are stripped from feature matrices in generated folds.

    Args:
        synthetic_star_schema_df (pd.DataFrame): Synthetic multi-indexed DataFrame.

    """
    folds = generate_cv_folds(synthetic_star_schema_df, n_splits=3)

    for fold in folds:
        assert "is_insolvent" not in fold.X_train.columns
        assert "is_insolvent" not in fold.X_val.columns


def test_generate_cv_folds_respects_temporal_order(synthetic_star_schema_df) -> None:
    """Verify that training set dates strictly precede validation set dates in folds.

    Args:
        synthetic_star_schema_df (pd.DataFrame): Synthetic multi-indexed DataFrame.

    """
    folds = generate_cv_folds(synthetic_star_schema_df, n_splits=3)

    for fold in folds:
        max_train_date = fold.X_train.index.get_level_values("snapshot_date").max()
        min_val_date = fold.X_val.index.get_level_values("snapshot_date").min()

        assert max_train_date < min_val_date


def test_generate_cv_folds_training_window_expands(synthetic_star_schema_df) -> None:
    """Check that training set size expands monotonically across sequential folds.

    Args:
        synthetic_star_schema_df (pd.DataFrame): Synthetic multi-indexed DataFrame.

    """
    folds = generate_cv_folds(synthetic_star_schema_df, n_splits=3)

    train_sizes = [len(fold.X_train) for fold in folds]

    assert train_sizes == sorted(train_sizes)


def test_train_val_test_split_orchestrates_full_pipeline(
    synthetic_star_schema_df,
) -> None:
    """Validate full pipeline execution returning folds and test DataFrame.

    Args:
        synthetic_star_schema_df (pd.DataFrame): Synthetic multi-indexed DataFrame.

    """
    n_test_months = 4
    n_splits = 3

    train_val_folds, df_test = train_val_test_split(
        synthetic_star_schema_df, n_test_months=n_test_months, n_splits=n_splits
    )

    assert len(train_val_folds) == n_splits
    assert [isinstance(fold, Fold) for fold in train_val_folds]
    assert isinstance(df_test, pd.DataFrame)

    test_dates = df_test.index.get_level_values("snapshot_date").unique()
    assert len(test_dates) == n_test_months
