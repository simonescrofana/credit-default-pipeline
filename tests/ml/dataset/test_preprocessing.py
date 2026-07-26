"""Test the machine learning dataset preprocessing pipeline.

Provide unit tests to verify missing-value handling with informative flags,
categorical encoding, feature scaling, and orchestration entry points for
cross-validation folds and final test set preprocessing.

"""

import numpy as np
import pandas as pd
import pytest

from ml.dataset.preprocessing import (
    CATEGORICAL_COLS,
    FINANCIAL_COLS,
    LOGIN_COLS,
    NUMERIC_COLS_TO_SCALE,
    SATISFACTION_COL,
    handle_missing_and_encode,
    preprocess_test_set,
    preprocess_train_folds,
    scale_features,
)
from ml.dataset.split import TARGET_COLUMN, Fold


def build_synthetic_features(
    n_rows: int, industries: list[str], regions: list[str]
) -> pd.DataFrame:
    """Build a synthetic feature DataFrame with plausible business ranges.

    Args:
        n_rows (int): Number of rows to include, up to 4.
        industries (list[str]): Industry sector values, one per row.
        regions (list[str]): Registered office region values, one per row.

    Returns:
        pd.DataFrame: A synthetic DataFrame mirroring the star schema feature
        set, including deliberate NaN values in the nullable columns and the
        ``is_insolvent`` target.

    """
    return pd.DataFrame(
        {
            "company_age_days": np.linspace(30, 3000, n_rows),
            "active_contracts_count": [1, 2, 1, 3][:n_rows],
            "has_active_electricity_contract": [True, True, False, True][:n_rows],
            "has_active_gas_contract": [False, True, True, False][:n_rows],
            "leverage_ratio": [0.5, 1.2, np.nan, 2.1][:n_rows],
            "cash_to_debt_ratio": [0.3, 0.1, np.nan, 0.8][:n_rows],
            "net_profit_margin": [0.05, -0.02, np.nan, 0.1][:n_rows],
            "ebitda": [10000.0, -2000.0, np.nan, 5000.0][:n_rows],
            "max_dpd_trailing_90d": [0, 120, 15, 5][:n_rows],
            "avg_dpd_trailing_90d": [0.0, 85.5, 5.0, 1.2][:n_rows],
            "unpaid_ratio_trailing_90d": [0.0, 0.4, 0.1, 0.02][:n_rows],
            "total_outstanding_debt": [0.0, 15000.0, 500.0, 100.0][:n_rows],
            "days_since_last_login": [2.0, np.nan, 10.0, 1.0][:n_rows],
            "login_velocity": [1.1, np.nan, 0.5, 1.4][:n_rows],
            "billing_disputes_count": [0, 3, 1, 0][:n_rows],
            "average_satisfaction_score": [4.5, np.nan, 2.1, np.nan][:n_rows],
            "industry_sector": industries[:n_rows],
            "registered_office_region": regions[:n_rows],
            "year": [2025, 2025, 2026, 2026][:n_rows],
            "quarter": [4, 4, 1, 1][:n_rows],
            "month": [11, 12, 1, 2][:n_rows],
            TARGET_COLUMN: [0, 1, 0, 0][:n_rows],
        }
    )


@pytest.fixture
def synthetic_train_df() -> pd.DataFrame:
    """Provide a 4-row synthetic training DataFrame with known NaN positions.

    Returns:
        pd.DataFrame: Row 0 has no missing values, row 1 is missing login and
        satisfaction data, row 2 is missing all financial data, row 3 has no
        missing values.

    """
    return build_synthetic_features(
        n_rows=4,
        industries=["manufacturing", "retail", "manufacturing", "services"],
        regions=["Lazio", "Lombardia", "Lazio", "Lombardia"],
    )


@pytest.fixture
def synthetic_val_df() -> pd.DataFrame:
    """Provide a 2-row synthetic validation/test DataFrame.

    Includes an "energy" industry sector never present in
    ``synthetic_train_df``, in order to exercise the fitted
    ``OneHotEncoder``'s ``handle_unknown="ignore"`` behavior.

    Returns:
        pd.DataFrame: A synthetic DataFrame with the same schema as
        ``synthetic_train_df``.

    """
    return build_synthetic_features(
        n_rows=2,
        industries=["energy", "retail"],
        regions=["Lazio", "Lazio"],
    )


def test_handle_missing_and_encode_creates_flags(synthetic_train_df) -> None:
    """Verify that per-group missing-value flags reflect the correct rows."""
    result, _ = handle_missing_and_encode(synthetic_train_df)

    assert result.loc[2, "has_recent_financials"] == 0
    assert result.loc[0, "has_recent_financials"] == 1

    assert result.loc[1, "has_login_activity"] == 0
    assert result.loc[0, "has_login_activity"] == 1

    assert result.loc[1, "has_recent_satisfaction_score"] == 0
    assert result.loc[0, "has_recent_satisfaction_score"] == 1


def test_handle_missing_and_encode_applies_correct_fallbacks(
    synthetic_train_df,
) -> None:
    """Verify that each column group receives its designated fallback value."""
    result, _ = handle_missing_and_encode(synthetic_train_df)

    assert result.loc[2, FINANCIAL_COLS].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert result.loc[1, LOGIN_COLS].tolist() == [-1, -1]
    assert result.loc[1, SATISFACTION_COL] == -1


def test_handle_missing_and_encode_no_nans_remain(synthetic_train_df) -> None:
    """Verify that no NaN values remain in the nullable columns after processing."""
    result, _ = handle_missing_and_encode(synthetic_train_df)

    assert (
        result[FINANCIAL_COLS + LOGIN_COLS + [SATISFACTION_COL]].isna().sum().sum() == 0
    )


def test_handle_missing_and_encode_removes_categorical_columns(
    synthetic_train_df,
) -> None:
    """Verify that the original categorical columns are dropped after encoding."""
    result, _ = handle_missing_and_encode(synthetic_train_df)

    for col in CATEGORICAL_COLS:
        assert col not in result.columns


def test_handle_missing_and_encode_creates_one_hot_columns(synthetic_train_df) -> None:
    """Verify that one-hot columns are created and correctly flag each category."""
    result, _ = handle_missing_and_encode(synthetic_train_df)

    assert "industry_sector_manufacturing" in result.columns
    assert "registered_office_region_Lazio" in result.columns
    assert result.loc[0, "industry_sector_manufacturing"] == 1
    assert result.loc[1, "industry_sector_manufacturing"] == 0


def test_handle_missing_and_encode_reuses_fitted_encoder(
    synthetic_train_df, synthetic_val_df
) -> None:
    """Verify that passing a fitted encoder reuses it instead of fitting a new one."""
    train_result, encoder = handle_missing_and_encode(synthetic_train_df)
    val_result, val_encoder = handle_missing_and_encode(
        synthetic_val_df, encoder=encoder
    )

    assert val_encoder is encoder
    assert set(val_result.columns) == set(train_result.columns)


def test_handle_missing_and_encode_ignores_unknown_category(
    synthetic_train_df, synthetic_val_df
) -> None:
    """Verify that a category unseen during fit produces an all-zero one-hot row."""
    _, encoder = handle_missing_and_encode(synthetic_train_df)
    val_result, _ = handle_missing_and_encode(synthetic_val_df, encoder=encoder)

    energy_row = val_result.iloc[0]
    industry_columns = [
        c for c in val_result.columns if c.startswith("industry_sector_")
    ]

    assert energy_row[industry_columns].sum() == 0


def test_scale_features_produces_standardized_columns(synthetic_train_df) -> None:
    """Verify that a freshly fitted scaler standardizes the target columns to mean 0."""
    encoded_df, _ = handle_missing_and_encode(synthetic_train_df)
    scaled_df, _ = scale_features(encoded_df)

    means = scaled_df[NUMERIC_COLS_TO_SCALE].mean()
    assert np.allclose(means, 0, atol=1e-8)


def test_scale_features_reuses_fitted_scaler(
    synthetic_train_df, synthetic_val_df
) -> None:
    """Verify that reusing a fitted scaler does not leak validation statistics.

    Applying the training-fitted scaler to the validation set should not
    force the validation set's own mean to 0, which would indicate a new fit
    occurred on validation data instead of a reused transform.

    """
    train_encoded, encoder = handle_missing_and_encode(synthetic_train_df)
    val_encoded, _ = handle_missing_and_encode(synthetic_val_df, encoder=encoder)

    _, scaler = scale_features(train_encoded)
    val_scaled, val_scaler = scale_features(val_encoded, scaler=scaler)

    assert val_scaler is scaler
    assert not np.allclose(val_scaled[NUMERIC_COLS_TO_SCALE].mean(), 0, atol=1e-8)


def test_scale_features_excludes_non_numeric_columns(synthetic_train_df):
    """Verify that boolean and one-hot encoded columns are left untouched by scaling."""
    encoded_df, _ = handle_missing_and_encode(synthetic_train_df)
    scaled_df, _ = scale_features(encoded_df)

    assert scaled_df["has_active_electricity_contract"].tolist() == (
        encoded_df["has_active_electricity_contract"].tolist()
    )
    assert scaled_df["industry_sector_manufacturing"].tolist() == (
        encoded_df["industry_sector_manufacturing"].tolist()
    )


@pytest.fixture
def synthetic_folds(synthetic_train_df, synthetic_val_df) -> list[Fold]:
    """Build a single synthetic cross-validation Fold from the train/val fixtures.

    Returns:
        list[Fold]: A one-element list containing a Fold with features and
        target already separated, mirroring the output of
        ``ml.dataset.split.generate_cv_folds``.

    """

    def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        return df.drop(columns=TARGET_COLUMN), df[TARGET_COLUMN]

    X_train, y_train = split_xy(synthetic_train_df)
    X_val, y_val = split_xy(synthetic_val_df)

    return [Fold(X_train, y_train, X_val, y_val)]


def test_preprocess_train_folds_returns_same_number_of_folds(synthetic_folds) -> None:
    """Verify that the orchestrator returns one processed Fold per input fold."""
    result = preprocess_train_folds(synthetic_folds)

    assert len(result) == len(synthetic_folds)
    assert all(isinstance(fold, Fold) for fold in result)


def test_preprocess_train_folds_applies_scaling_when_requested(synthetic_folds) -> None:
    """Verify that scale=True standardizes the training split of each fold."""
    result = preprocess_train_folds(synthetic_folds, scale=True)

    means = result[0].X_train[NUMERIC_COLS_TO_SCALE].mean()
    assert np.allclose(means, 0, atol=1e-8)


def test_preprocess_train_folds_skips_scaling_when_disabled(synthetic_folds) -> None:
    """Verify that scale=False leaves feature values unstandardized."""
    original_values = synthetic_folds[0].X_train["ebitda"].replace(np.nan, 0.0).tolist()

    result = preprocess_train_folds(synthetic_folds, scale=False)

    assert result[0].X_train["ebitda"].tolist() == original_values


def test_preprocess_test_set_returns_expected_shapes(
    synthetic_train_df, synthetic_val_df
) -> None:
    """Verify that preprocess_test_set returns well-shaped, target-free features."""
    X_train_full, y_train_full, X_test, y_test = preprocess_test_set(
        synthetic_train_df, synthetic_val_df
    )

    assert len(X_train_full) == len(y_train_full) == len(synthetic_train_df)
    assert len(X_test) == len(y_test) == len(synthetic_val_df)
    assert TARGET_COLUMN not in X_train_full.columns
    assert TARGET_COLUMN not in X_test.columns


def test_preprocess_train_folds_fits_only_on_train_per_fold(synthetic_folds) -> None:
    """Verify that each fold's validation split is transformed, never fitted."""
    result = preprocess_train_folds(synthetic_folds, scale=True)

    train_means = result[0].X_train[NUMERIC_COLS_TO_SCALE].mean()
    val_means = result[0].X_val[NUMERIC_COLS_TO_SCALE].mean()

    assert np.allclose(train_means, 0, atol=1e-8)
    assert not np.allclose(val_means, 0, atol=1e-8)


def test_preprocess_test_set_fits_only_on_train(
    synthetic_train_df, synthetic_val_df
) -> None:
    """Verify that the holdout test set is transformed, never fitted.

    The full training block should be standardized to mean 0; the test set,
    only transformed with the training-fitted scaler, should not have its
    own mean forced to 0.

    """
    X_train_full, _, X_test, _ = preprocess_test_set(
        synthetic_train_df, synthetic_val_df
    )

    means = X_train_full[NUMERIC_COLS_TO_SCALE].mean()
    assert np.allclose(means, 0, atol=1e-8)
    assert not np.allclose(X_test[NUMERIC_COLS_TO_SCALE].mean(), 0, atol=1e-8)
