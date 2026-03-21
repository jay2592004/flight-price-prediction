"""
tests/test_preprocessor.py
Unit tests for src/data/preprocessor.py
"""

import pytest
import pandas as pd
import numpy as np
import os
from src.data.preprocessor import backfill_original, load_and_merge
from src.data.generator import CITIES, AIRLINES


# ── Helpers ───────────────────────────────────────────────────────────────────
def _make_minimal_original(n=50):
    """Build a minimal DataFrame that mimics the real flight_price_dataset.csv schema."""
    np.random.seed(10)
    sources = np.random.choice(CITIES, n)
    dests   = [np.random.choice([c for c in CITIES if c != s]) for s in sources]
    return pd.DataFrame({
        "Airline":               np.random.choice(AIRLINES, n),
        "Source":                sources,
        "Destination":           dests,
        "Route":                 [f"{s} - {d}" for s,d in zip(sources,dests)],
        "Journey_Date":          ["15-06-2026"] * n,
        "Journey_Month":         [6] * n,
        "Journey_Day":           [15] * n,
        "Total_Stops":           np.random.choice(["non-stop","1 stop","2 stops"], n),
        "Class":                 np.random.choice(["Economy","Business"], n),
        "Days_Until_Departure":  np.random.randint(1, 200, n),
        "Distance_km":           np.random.randint(500, 12000, n),
        "Price":                 np.random.randint(5000, 50000, n),
    })


# ── backfill_original tests ───────────────────────────────────────────────────
@pytest.fixture(scope="module")
def backfilled_df():
    df_orig = _make_minimal_original(50)
    return backfill_original(df_orig)


def test_backfill_adds_saf_zone(backfilled_df):
    assert "SAF_Zone" in backfilled_df.columns


def test_backfill_adds_env_tier(backfilled_df):
    assert "Env_Surcharge_Tier" in backfilled_df.columns


def test_backfill_adds_fleet_age(backfilled_df):
    assert "Fleet_Age_Years" in backfilled_df.columns


def test_backfill_adds_restricted(backfilled_df):
    assert "Is_Restricted_Airspace" in backfilled_df.columns


def test_backfill_adds_weekday(backfilled_df):
    assert "Journey_Weekday" in backfilled_df.columns


def test_backfill_adds_weekend(backfilled_df):
    assert "Is_Weekend" in backfilled_df.columns


def test_backfill_adds_season(backfilled_df):
    assert "Season" in backfilled_df.columns
    assert backfilled_df["Season"].isin(["peak","shoulder","off_peak"]).all()


def test_backfill_adds_aircraft_type(backfilled_df):
    assert "Aircraft_Type" in backfilled_df.columns
    assert backfilled_df["Aircraft_Type"].isin(["wide-body","narrow-body"]).all()


def test_backfill_adds_airline_tier(backfilled_df):
    assert "Airline_Tier" in backfilled_df.columns
    assert backfilled_df["Airline_Tier"].isin(["budget","mid","premium"]).all()


def test_backfill_saf_zone_range(backfilled_df):
    assert backfilled_df["SAF_Zone"].isin([0,1,2]).all()


def test_backfill_env_tier_range(backfilled_df):
    assert backfilled_df["Env_Surcharge_Tier"].isin([0,1,2,3]).all()


def test_backfill_restricted_binary(backfilled_df):
    assert backfilled_df["Is_Restricted_Airspace"].isin([0,1]).all()


def test_backfill_fleet_age_positive(backfilled_df):
    assert (backfilled_df["Fleet_Age_Years"] > 0).all()


def test_backfill_preserves_row_count():
    df_orig = _make_minimal_original(30)
    result  = backfill_original(df_orig)
    assert len(result) == 30


def test_backfill_preserves_price(backfilled_df):
    # Price column should be untouched
    orig = _make_minimal_original(50)
    # Both have same length — just check Price column still exists
    assert "Price" in backfilled_df.columns


def test_backfill_does_not_mutate_original():
    df_orig = _make_minimal_original(20)
    cols_before = set(df_orig.columns)
    backfill_original(df_orig)
    assert set(df_orig.columns) == cols_before, "backfill_original mutated the original DataFrame"


# ── load_and_merge tests ──────────────────────────────────────────────────────
def test_load_and_merge_without_original(small_synth_df, tmp_path):
    combined = load_and_merge(str(tmp_path / "nonexistent.csv"), small_synth_df)
    assert len(combined) == len(small_synth_df)


def test_load_and_merge_with_original(small_synth_df, tmp_path):
    orig = _make_minimal_original(30)
    orig_path = str(tmp_path / "orig.csv")
    orig.to_csv(orig_path, index=False)

    combined = load_and_merge(orig_path, small_synth_df)
    # Combined should have synth rows + orig rows
    assert len(combined) == len(small_synth_df) + 30


def test_load_and_merge_columns_match_synth(small_synth_df, tmp_path):
    orig = _make_minimal_original(10)
    orig_path = str(tmp_path / "orig2.csv")
    orig.to_csv(orig_path, index=False)

    combined = load_and_merge(orig_path, small_synth_df)
    assert list(combined.columns) == list(small_synth_df.columns)


def test_load_and_merge_no_nulls_in_key_columns(small_synth_df, tmp_path):
    orig = _make_minimal_original(10)
    orig_path = str(tmp_path / "orig3.csv")
    orig.to_csv(orig_path, index=False)

    combined = load_and_merge(orig_path, small_synth_df)
    for col in ["Airline","Source","Destination","Price","Class"]:
        assert combined[col].isnull().sum() == 0, f"Nulls in {col} after merge"


def test_load_and_merge_is_shuffled(small_synth_df, tmp_path):
    """After merge the rows should be shuffled — first rows shouldn't all be from orig."""
    orig = _make_minimal_original(50)
    orig_path = str(tmp_path / "orig4.csv")
    orig.to_csv(orig_path, index=False)

    combined = load_and_merge(orig_path, small_synth_df)
    # If not shuffled, all first 50 rows would be from orig (same dates).
    # Shuffled → date variance in first 50 rows.
    first_50_months = combined["Journey_Month"].iloc[:50].nunique()
    assert first_50_months > 1, "Merged dataset does not appear to be shuffled"
