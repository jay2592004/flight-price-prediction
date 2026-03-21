"""
src/data_loader.py
Loads raw CSV, back-fills BRD columns on original data,
merges with synthetic, returns clean combined DataFrame.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import os
from .data_generator import (
    SAF_ZONE, ENV_TIER, FLEET_AGE, AIRLINE_TIER, RESTRICTED, get_season, generate
)
from .config import RAW_CSV, SYNTHETIC_CSV, COMBINED_CSV, DEFAULT_N_ROWS, DEFAULT_SEED


def _parse_date(s):
    try:    return datetime.strptime(str(s), "%d-%m-%Y")
    except: return datetime(2026, 6, 1)


def backfill_brd_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add all BRD + extra columns to the original 20k dataset rows."""
    df = df.copy()
    df["SAF_Zone"]               = df["Destination"].map(lambda x: SAF_ZONE.get(x, 0))
    df["Env_Surcharge_Tier"]     = df.apply(
        lambda r: max(ENV_TIER.get(r["Source"], 0), ENV_TIER.get(r["Destination"], 0)), axis=1)
    df["Fleet_Age_Years"]        = df["Airline"].map(lambda x: FLEET_AGE.get(x, 10.0))
    df["Is_Restricted_Airspace"] = df.apply(
        lambda r: 1 if (r["Source"],r["Destination"]) in RESTRICTED
                    or (r["Destination"],r["Source"]) in RESTRICTED else 0, axis=1)
    dates = df["Journey_Date"].apply(_parse_date)
    df["Journey_Weekday"]   = dates.apply(lambda d: d.weekday())
    df["Is_Weekend"]        = (df["Journey_Weekday"] >= 5).astype(int)
    df["Season"]            = df["Journey_Month"].apply(get_season)
    df["Geo_Risk_Score"]    = (df["Is_Restricted_Airspace"]*0.5 +
                               np.random.beta(1.5,6,len(df))*0.5).round(3)
    df["Fuel_Price_Index"]  = np.random.normal(100,12,len(df)).clip(70,140).round(1)
    df["Seat_Availability"] = np.random.beta(2,3,len(df)).round(3)
    df["Layover_Hours"]     = df["Total_Stops"].apply(
        lambda s: 0 if s=="non-stop" else (np.random.randint(1,8) if s=="1 stop"
                                           else np.random.randint(4,14)))
    df["Aircraft_Type"]     = df["Distance_km"].apply(
        lambda d: "wide-body" if d>4000 else "narrow-body")
    df["Airline_Tier"]      = df["Airline"].map(AIRLINE_TIER)
    return df


def load_combined(n_rows=DEFAULT_N_ROWS, seed=DEFAULT_SEED, force=False) -> pd.DataFrame:
    """
    Returns the combined DataFrame (original + synthetic).
    Generates and saves if not present or force=True.
    """
    if not os.path.exists(COMBINED_CSV) or force:
        # Generate synthetic
        synth = generate(n_rows=n_rows, seed=seed)
        synth.to_csv(SYNTHETIC_CSV, index=False)

        col_order = synth.columns.tolist()
        if os.path.exists(RAW_CSV):
            orig = pd.read_csv(RAW_CSV)
            orig = backfill_brd_columns(orig)
            for col in col_order:
                if col not in orig.columns:
                    orig[col] = 0
            orig = orig[col_order]
            combined = pd.concat([orig, synth], ignore_index=True)
        else:
            combined = synth.copy()

        combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
        combined.to_csv(COMBINED_CSV, index=False)
    else:
        combined = pd.read_csv(COMBINED_CSV)

    combined = combined.drop_duplicates()
    combined["Journey_Date_dt"] = pd.to_datetime(
        combined["Journey_Date"], dayfirst=True, errors="coerce")
    combined = combined.sort_values("Journey_Date_dt").reset_index(drop=True)
    return combined
