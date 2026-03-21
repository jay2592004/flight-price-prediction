"""
src/pipeline/features.py
All feature engineering logic — encoding, interactions, derived flags.
Used by both the training pipeline and the Streamlit app at prediction time.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, List


# Ordered list of features the model is trained on — DO NOT reorder
FEATURE_COLUMNS: List[str] = [
    "Airline_Enc","Source_Enc","Destination_Enc","Class_Enc","Total_Stops_Enc",
    "Distance_km","Log_Distance","Days_Until_Departure","Log_Days_Until",
    "Journey_Month","Journey_Day","Journey_Weekday","Is_Weekend",
    "Season_Enc",
    "SAF_Zone","Env_Surcharge_Tier","Fleet_Age_Years","Is_Restricted_Airspace",
    "Geo_Risk_Score","Fuel_Price_Index","Seat_Availability","Layover_Hours",
    "Aircraft_Type_Enc","Airline_Tier_Enc",
    "Class_Dist_Interact","Season_BookWin_Interact","Tier_SAF_Interact",
    "Stops_Dist_Interact","Env_Fleet_Interact",
    "Is_Long_Haul","Is_Last_Minute","Is_Advance_Booking",
]

TARGET_COLUMN = "Price"

CAT_COLUMNS = [
    "Airline","Source","Destination","Class","Total_Stops",
    "Season","Aircraft_Type","Airline_Tier",
]


def fit_encoders(df: pd.DataFrame) -> Dict[str, LabelEncoder]:
    """Fit one LabelEncoder per categorical column. Returns dict of encoders."""
    encoders: Dict[str, LabelEncoder] = {}
    for col in CAT_COLUMNS:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    return encoders


def engineer(df: pd.DataFrame, encoders: Dict[str, LabelEncoder]) -> pd.DataFrame:
    """Apply feature engineering. Returns new DataFrame with FEATURE_COLUMNS."""
    df = df.copy()

    # Encode categoricals
    for col in CAT_COLUMNS:
        df[col+"_Enc"] = encoders[col].transform(df[col].astype(str))

    # Log transforms
    df["Log_Distance"]    = np.log1p(df["Distance_km"])
    df["Log_Days_Until"]  = np.log1p(df["Days_Until_Departure"])

    # Interaction features
    df["Class_Dist_Interact"]     = df["Class_Enc"]       * df["Distance_km"]
    df["Season_BookWin_Interact"] = df["Season_Enc"]       * df["Days_Until_Departure"]
    df["Tier_SAF_Interact"]       = df["Airline_Tier_Enc"] * df["SAF_Zone"]
    df["Stops_Dist_Interact"]     = df["Total_Stops_Enc"]  * df["Distance_km"]
    df["Env_Fleet_Interact"]      = df["Env_Surcharge_Tier"] * df["Fleet_Age_Years"]

    # Binary flags
    df["Is_Long_Haul"]       = (df["Distance_km"]          > 5000).astype(int)
    df["Is_Last_Minute"]     = (df["Days_Until_Departure"]  < 7).astype(int)
    df["Is_Advance_Booking"] = (df["Days_Until_Departure"]  > 90).astype(int)

    return df[FEATURE_COLUMNS]


def build_single_row(input_dict: dict, encoders: Dict[str, LabelEncoder]) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame for inference (Streamlit predictor).
    input_dict must contain the raw field values (Airline, Source, etc.).
    """
    d = input_dict
    stops_map  = {"non-stop":0,"1 stop":1,"2 stops":2}
    season_val = d.get("Season","off_peak")

    row = {
        "Airline_Enc":          encoders["Airline"].transform([d["Airline"]])[0],
        "Source_Enc":           encoders["Source"].transform([d["Source"]])[0],
        "Destination_Enc":      encoders["Destination"].transform([d["Destination"]])[0],
        "Class_Enc":            encoders["Class"].transform([d["Class"]])[0],
        "Total_Stops_Enc":      encoders["Total_Stops"].transform([d["Total_Stops"]])[0],
        "Distance_km":          d["Distance_km"],
        "Log_Distance":         np.log1p(d["Distance_km"]),
        "Days_Until_Departure": d["Days_Until_Departure"],
        "Log_Days_Until":       np.log1p(d["Days_Until_Departure"]),
        "Journey_Month":        d["Journey_Month"],
        "Journey_Day":          d["Journey_Day"],
        "Journey_Weekday":      d["Journey_Weekday"],
        "Is_Weekend":           int(d["Journey_Weekday"] >= 5),
        "Season_Enc":           encoders["Season"].transform([season_val])[0],
        "SAF_Zone":             d["SAF_Zone"],
        "Env_Surcharge_Tier":   d["Env_Surcharge_Tier"],
        "Fleet_Age_Years":      d["Fleet_Age_Years"],
        "Is_Restricted_Airspace": d["Is_Restricted_Airspace"],
        "Geo_Risk_Score":       d.get("Geo_Risk_Score", d["Is_Restricted_Airspace"]*0.5+0.1),
        "Fuel_Price_Index":     d.get("Fuel_Price_Index", 100.0),
        "Seat_Availability":    d.get("Seat_Availability", 0.5),
        "Layover_Hours":        d.get("Layover_Hours", 0),
        "Aircraft_Type_Enc":    encoders["Aircraft_Type"].transform([d["Aircraft_Type"]])[0],
        "Airline_Tier_Enc":     encoders["Airline_Tier"].transform([d["Airline_Tier"]])[0],
        "Class_Dist_Interact":  encoders["Class"].transform([d["Class"]])[0] * d["Distance_km"],
        "Season_BookWin_Interact": encoders["Season"].transform([season_val])[0] * d["Days_Until_Departure"],
        "Tier_SAF_Interact":    encoders["Airline_Tier"].transform([d["Airline_Tier"]])[0] * d["SAF_Zone"],
        "Stops_Dist_Interact":  stops_map[d["Total_Stops"]] * d["Distance_km"],
        "Env_Fleet_Interact":   d["Env_Surcharge_Tier"] * d["Fleet_Age_Years"],
        "Is_Long_Haul":         int(d["Distance_km"] > 5000),
        "Is_Last_Minute":       int(d["Days_Until_Departure"] < 7),
        "Is_Advance_Booking":   int(d["Days_Until_Departure"] > 90),
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]
