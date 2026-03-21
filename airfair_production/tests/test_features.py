"""
tests/test_features.py
Unit tests for src/pipeline/features.py
"""

import pytest
import pandas as pd
import numpy as np
from src.pipeline.features import (
    FEATURE_COLUMNS, TARGET_COLUMN, CAT_COLUMNS,
    fit_encoders, engineer, build_single_row,
)


def test_feature_column_count():
    assert len(FEATURE_COLUMNS) == 32, f"Expected 32 features, got {len(FEATURE_COLUMNS)}"


def test_no_duplicate_features():
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS)), "Duplicate feature names found"


def test_fit_encoders_returns_all_cats(fitted_encoders):
    for col in CAT_COLUMNS:
        assert col in fitted_encoders, f"Encoder missing for: {col}"


def test_fit_encoders_are_fitted(fitted_encoders):
    from sklearn.preprocessing import LabelEncoder
    for name, enc in fitted_encoders.items():
        assert isinstance(enc, LabelEncoder)
        assert hasattr(enc, "classes_"), f"Encoder not fitted: {name}"


def test_engineer_returns_correct_columns(engineered_X):
    assert list(engineered_X.columns) == FEATURE_COLUMNS


def test_engineer_no_nulls(engineered_X):
    assert engineered_X.isnull().sum().sum() == 0


def test_engineer_row_count_preserved(small_synth_df, engineered_X):
    assert len(engineered_X) == len(small_synth_df)


def test_log_distance_positive(engineered_X):
    assert (engineered_X["Log_Distance"] > 0).all()


def test_log_days_positive(engineered_X):
    assert (engineered_X["Log_Days_Until"] > 0).all()


def test_is_long_haul_binary(engineered_X):
    assert engineered_X["Is_Long_Haul"].isin([0,1]).all()


def test_is_last_minute_binary(engineered_X):
    assert engineered_X["Is_Last_Minute"].isin([0,1]).all()


def test_is_advance_booking_binary(engineered_X):
    assert engineered_X["Is_Advance_Booking"].isin([0,1]).all()


def test_long_haul_consistent_with_distance(small_synth_df, engineered_X):
    expected = (small_synth_df["Distance_km"] > 5000).astype(int).values
    assert (engineered_X["Is_Long_Haul"].values == expected).all()


def test_last_minute_consistent_with_days(small_synth_df, engineered_X):
    expected = (small_synth_df["Days_Until_Departure"] < 7).astype(int).values
    assert (engineered_X["Is_Last_Minute"].values == expected).all()


def test_class_dist_interact_is_product(small_synth_df, engineered_X, fitted_encoders):
    class_enc = fitted_encoders["Class"].transform(small_synth_df["Class"].astype(str))
    expected  = class_enc * small_synth_df["Distance_km"].values
    np.testing.assert_array_almost_equal(
        engineered_X["Class_Dist_Interact"].values, expected)


def test_build_single_row_shape(sample_input_dict, fitted_encoders):
    row = build_single_row(sample_input_dict, fitted_encoders)
    assert row.shape == (1, len(FEATURE_COLUMNS))


def test_build_single_row_columns(sample_input_dict, fitted_encoders):
    row = build_single_row(sample_input_dict, fitted_encoders)
    assert list(row.columns) == FEATURE_COLUMNS


def test_build_single_row_no_nulls(sample_input_dict, fitted_encoders):
    row = build_single_row(sample_input_dict, fitted_encoders)
    assert row.isnull().sum().sum() == 0


def test_build_single_row_log_distance(sample_input_dict, fitted_encoders):
    row = build_single_row(sample_input_dict, fitted_encoders)
    expected = np.log1p(sample_input_dict["Distance_km"])
    assert abs(row["Log_Distance"].iloc[0] - expected) < 1e-9


def test_build_single_row_is_long_haul(sample_input_dict, fitted_encoders):
    row = build_single_row(sample_input_dict, fitted_encoders)
    expected = int(sample_input_dict["Distance_km"] > 5000)
    assert row["Is_Long_Haul"].iloc[0] == expected


def test_build_single_row_is_weekend(sample_input_dict, fitted_encoders):
    row = build_single_row(sample_input_dict, fitted_encoders)
    expected = int(sample_input_dict["Journey_Weekday"] >= 5)
    assert row["Is_Weekend"].iloc[0] == expected


@pytest.mark.parametrize("stops,enc", [
    ("non-stop", 0), ("1 stop", 1), ("2 stops", 2)
])
def test_stops_dist_interact_values(sample_input_dict, fitted_encoders, stops, enc):
    d = {**sample_input_dict, "Total_Stops": stops}
    row = build_single_row(d, fitted_encoders)
    expected = enc * d["Distance_km"]
    assert abs(row["Stops_Dist_Interact"].iloc[0] - expected) < 1e-9
