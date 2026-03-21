"""
tests/test_generator.py
Unit tests for src/data/generator.py
"""

import pytest
import pandas as pd
import numpy as np
from src.data.generator import generate, AIRLINES, CITIES, CLASSES, STOPS


def test_generate_row_count(small_synth_df):
    assert len(small_synth_df) == 200


def test_generate_column_count(small_synth_df):
    # 25 columns: 11 original + 4 BRD + 9 extra + 1 target
    assert len(small_synth_df.columns) == 25


def test_required_columns_present(small_synth_df):
    required = [
        "Airline","Source","Destination","Route","Journey_Date",
        "Journey_Month","Journey_Day","Total_Stops","Class",
        "Days_Until_Departure","Distance_km","Price",
        "SAF_Zone","Env_Surcharge_Tier","Fleet_Age_Years","Is_Restricted_Airspace",
        "Journey_Weekday","Is_Weekend","Season","Geo_Risk_Score",
        "Fuel_Price_Index","Seat_Availability","Layover_Hours",
        "Aircraft_Type","Airline_Tier",
    ]
    for col in required:
        assert col in small_synth_df.columns, f"Missing column: {col}"


def test_no_null_values(small_synth_df):
    assert small_synth_df.isnull().sum().sum() == 0, "Dataset contains null values"


def test_price_floor(small_synth_df):
    assert (small_synth_df["Price"] >= 1500).all(), "Some prices are below floor (₹1500)"


def test_price_ceiling(small_synth_df):
    assert (small_synth_df["Price"] <= 250000).all(), "Some prices exceed ceiling (₹2.5L)"


def test_airlines_are_valid(small_synth_df):
    assert set(small_synth_df["Airline"].unique()).issubset(set(AIRLINES))


def test_cities_are_valid(small_synth_df):
    assert set(small_synth_df["Source"].unique()).issubset(set(CITIES))
    assert set(small_synth_df["Destination"].unique()).issubset(set(CITIES))


def test_source_dest_never_equal(small_synth_df):
    assert (small_synth_df["Source"] != small_synth_df["Destination"]).all()


def test_classes_are_valid(small_synth_df):
    assert set(small_synth_df["Class"].unique()).issubset(set(CLASSES))


def test_stops_are_valid(small_synth_df):
    assert set(small_synth_df["Total_Stops"].unique()).issubset(set(STOPS))


def test_saf_zone_range(small_synth_df):
    assert small_synth_df["SAF_Zone"].isin([0,1,2]).all()


def test_env_tier_range(small_synth_df):
    assert small_synth_df["Env_Surcharge_Tier"].isin([0,1,2,3]).all()


def test_fleet_age_range(small_synth_df):
    assert (small_synth_df["Fleet_Age_Years"] >= 3).all()
    assert (small_synth_df["Fleet_Age_Years"] <= 25).all()


def test_is_restricted_binary(small_synth_df):
    assert small_synth_df["Is_Restricted_Airspace"].isin([0,1]).all()


def test_is_weekend_binary(small_synth_df):
    assert small_synth_df["Is_Weekend"].isin([0,1]).all()


def test_is_weekend_consistent_with_weekday(small_synth_df):
    expected = (small_synth_df["Journey_Weekday"] >= 5).astype(int)
    assert (small_synth_df["Is_Weekend"] == expected).all()


def test_season_values(small_synth_df):
    assert small_synth_df["Season"].isin(["peak","shoulder","off_peak"]).all()


def test_aircraft_type_values(small_synth_df):
    assert small_synth_df["Aircraft_Type"].isin(["wide-body","narrow-body"]).all()


def test_aircraft_type_consistent_with_distance(small_synth_df):
    wide = small_synth_df[small_synth_df["Aircraft_Type"] == "wide-body"]
    narrow = small_synth_df[small_synth_df["Aircraft_Type"] == "narrow-body"]
    assert (wide["Distance_km"] > 4000).all()
    assert (narrow["Distance_km"] <= 4000).all()


def test_layover_hours_nonzero_for_stops(small_synth_df):
    nonstop = small_synth_df[small_synth_df["Total_Stops"] == "non-stop"]
    assert (nonstop["Layover_Hours"] == 0).all()


def test_days_until_departure_range(small_synth_df):
    assert (small_synth_df["Days_Until_Departure"] >= 1).all()
    assert (small_synth_df["Days_Until_Departure"] <= 365).all()


def test_reproducible_with_seed():
    df1 = generate(n_rows=50, seed=42)
    df2 = generate(n_rows=50, seed=42)
    assert df1["Price"].equals(df2["Price"]), "Same seed should produce identical output"


def test_different_seeds_differ():
    df1 = generate(n_rows=50, seed=1)
    df2 = generate(n_rows=50, seed=2)
    assert not df1["Price"].equals(df2["Price"]), "Different seeds should produce different output"


def test_larger_generation():
    df = generate(n_rows=1000, seed=7)
    assert len(df) == 1000
    assert df.isnull().sum().sum() == 0
