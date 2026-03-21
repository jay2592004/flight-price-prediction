"""
src/data/preprocessor.py
Loads raw CSVs, back-fills BRD columns on original data,
merges with synthetic, and returns a clean combined DataFrame.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import os
from .generator import (
    SAF_ZONE, ENV_TIER, FLEET_AGE, AIRLINE_TIER, RESTRICTED, get_season
)


def _parse_date(s: str) -> datetime:
    try:    return datetime.strptime(str(s), "%d-%m-%Y")
    except: return datetime(2026, 6, 1)


def backfill_original(df: pd.DataFrame) -> pd.DataFrame:
    """Add BRD Phase-2 and extra columns to the original 20k dataset."""
    df = df.copy()
    df["SAF_Zone"]               = df["Destination"].map(lambda x: SAF_ZONE.get(x,0))
    df["Env_Surcharge_Tier"]     = df.apply(
        lambda r: max(ENV_TIER.get(r["Source"],0), ENV_TIER.get(r["Destination"],0)), axis=1)
    df["Fleet_Age_Years"]        = df["Airline"].map(lambda x: FLEET_AGE.get(x,10.0))
    df["Is_Restricted_Airspace"] = df.apply(
        lambda r: 1 if (r["Source"],r["Destination"]) in RESTRICTED
                    or (r["Destination"],r["Source"]) in RESTRICTED else 0, axis=1)

    dates = df["Journey_Date"].apply(_parse_date)
    df["Journey_Weekday"]   = dates.apply(lambda d: d.weekday())
    df["Is_Weekend"]        = (df["Journey_Weekday"] >= 5).astype(int)
    df["Season"]            = df["Journey_Month"].apply(get_season)
    df["Geo_Risk_Score"]    = (df["Is_Restricted_Airspace"]*0.5
                               + np.random.beta(1.5,6,len(df))*0.5).round(3)
    df["Fuel_Price_Index"]  = np.random.normal(100,12,len(df)).clip(70,140).round(1)
    df["Seat_Availability"] = np.random.beta(2,3,len(df)).round(3)
    df["Layover_Hours"]     = df["Total_Stops"].apply(
        lambda s: 0 if s=="non-stop" else (np.random.randint(1,8) if s=="1 stop"
                                           else np.random.randint(4,14)))
    df["Aircraft_Type"]     = df["Distance_km"].apply(
        lambda d: "wide-body" if d>4000 else "narrow-body")
    df["Airline_Tier"]      = df["Airline"].map(AIRLINE_TIER)
    return df


def load_and_merge(orig_path: str, synth_df: pd.DataFrame) -> pd.DataFrame:
    """Merge original CSV with synthetic DataFrame. Returns shuffled combined DF."""
    col_order = synth_df.columns.tolist()

    if os.path.exists(orig_path):
        df_orig = pd.read_csv(orig_path)
        df_orig = backfill_original(df_orig)
        for col in col_order:
            if col not in df_orig.columns:
                df_orig[col] = 0
        df_orig = df_orig[col_order]
        combined = pd.concat([df_orig, synth_df], ignore_index=True)
    else:
        combined = synth_df.copy()

    return combined.sample(frac=1, random_state=42).reset_index(drop=True)
