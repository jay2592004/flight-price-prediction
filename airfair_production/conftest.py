"""
conftest.py
Shared pytest fixtures available to all test modules.
"""

import sys, os
import pytest
import pandas as pd
import numpy as np

# Make sure src/ is importable from project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def small_synth_df():
    """Generate a tiny 200-row synthetic dataset for fast test runs."""
    from src.data.generator import generate
    return generate(n_rows=200, seed=0)


@pytest.fixture(scope="session")
def fitted_encoders(small_synth_df):
    """Fit encoders on the small synthetic dataset."""
    from src.pipeline.features import fit_encoders
    return fit_encoders(small_synth_df)


@pytest.fixture(scope="session")
def engineered_X(small_synth_df, fitted_encoders):
    """Return feature-engineered X for the small dataset."""
    from src.pipeline.features import engineer
    return engineer(small_synth_df, fitted_encoders)


@pytest.fixture(scope="session")
def sample_input_dict(fitted_encoders):
    """A valid single-row input dict for build_single_row."""
    return {
        "Airline":               "Emirates",
        "Source":                "Dubai",
        "Destination":           "London",
        "Class":                 "Economy",
        "Total_Stops":           "1 stop",
        "Distance_km":           5500,
        "Days_Until_Departure":  45,
        "Journey_Month":         6,
        "Journey_Day":           15,
        "Journey_Weekday":       2,
        "Season":                "off_peak",
        "SAF_Zone":              2,
        "Env_Surcharge_Tier":    3,
        "Fleet_Age_Years":       7.8,
        "Is_Restricted_Airspace": 0,
        "Aircraft_Type":         "wide-body",
        "Airline_Tier":          "premium",
        "Geo_Risk_Score":        0.15,
        "Fuel_Price_Index":      100.0,
        "Seat_Availability":     0.5,
        "Layover_Hours":         3,
    }
