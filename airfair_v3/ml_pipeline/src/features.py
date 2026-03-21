"""
src/features.py
All feature engineering — encoding, interactions, derived flags.
Shared between training pipeline AND Streamlit app at prediction time.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List

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
CAT_COLUMNS   = ["Airline","Source","Destination","Class","Total_Stops",
                 "Season","Aircraft_Type","Airline_Tier"]


def fit_encoders(df: pd.DataFrame) -> Dict[str, LabelEncoder]:
    encoders = {}
    for col in CAT_COLUMNS:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    return encoders


def engineer(df: pd.DataFrame, encoders: Dict[str, LabelEncoder]) -> pd.DataFrame:
    df = df.copy()
    for col in CAT_COLUMNS:
        df[col+"_Enc"] = encoders[col].transform(df[col].astype(str))
    df["Log_Distance"]            = np.log1p(df["Distance_km"])
    df["Log_Days_Until"]          = np.log1p(df["Days_Until_Departure"])
    df["Class_Dist_Interact"]     = df["Class_Enc"]        * df["Distance_km"]
    df["Season_BookWin_Interact"] = df["Season_Enc"]        * df["Days_Until_Departure"]
    df["Tier_SAF_Interact"]       = df["Airline_Tier_Enc"]  * df["SAF_Zone"]
    df["Stops_Dist_Interact"]     = df["Total_Stops_Enc"]   * df["Distance_km"]
    df["Env_Fleet_Interact"]      = df["Env_Surcharge_Tier"]* df["Fleet_Age_Years"]
    df["Is_Long_Haul"]            = (df["Distance_km"]         > 5000).astype(int)
    df["Is_Last_Minute"]          = (df["Days_Until_Departure"] < 7).astype(int)
    df["Is_Advance_Booking"]      = (df["Days_Until_Departure"] > 90).astype(int)
    return df[FEATURE_COLUMNS]


def build_single_row(inp: dict, encoders: Dict[str, LabelEncoder]) -> pd.DataFrame:
    """Build one prediction row from raw user inputs."""
    stops_map = {"non-stop":0,"1 stop":1,"2 stops":2}
    season    = inp.get("Season","off_peak")
    row = {
        "Airline_Enc":          encoders["Airline"].transform([inp["Airline"]])[0],
        "Source_Enc":           encoders["Source"].transform([inp["Source"]])[0],
        "Destination_Enc":      encoders["Destination"].transform([inp["Destination"]])[0],
        "Class_Enc":            encoders["Class"].transform([inp["Class"]])[0],
        "Total_Stops_Enc":      encoders["Total_Stops"].transform([inp["Total_Stops"]])[0],
        "Distance_km":          inp["Distance_km"],
        "Log_Distance":         np.log1p(inp["Distance_km"]),
        "Days_Until_Departure": inp["Days_Until_Departure"],
        "Log_Days_Until":       np.log1p(inp["Days_Until_Departure"]),
        "Journey_Month":        inp["Journey_Month"],
        "Journey_Day":          inp["Journey_Day"],
        "Journey_Weekday":      inp["Journey_Weekday"],
        "Is_Weekend":           int(inp["Journey_Weekday"] >= 5),
        "Season_Enc":           encoders["Season"].transform([season])[0],
        "SAF_Zone":             inp["SAF_Zone"],
        "Env_Surcharge_Tier":   inp["Env_Surcharge_Tier"],
        "Fleet_Age_Years":      inp["Fleet_Age_Years"],
        "Is_Restricted_Airspace": inp["Is_Restricted_Airspace"],
        "Geo_Risk_Score":       inp.get("Geo_Risk_Score", inp["Is_Restricted_Airspace"]*0.5+0.1),
        "Fuel_Price_Index":     inp.get("Fuel_Price_Index", 100.0),
        "Seat_Availability":    inp.get("Seat_Availability", 0.5),
        "Layover_Hours":        inp.get("Layover_Hours", 0),
        "Aircraft_Type_Enc":    encoders["Aircraft_Type"].transform([inp["Aircraft_Type"]])[0],
        "Airline_Tier_Enc":     encoders["Airline_Tier"].transform([inp["Airline_Tier"]])[0],
        "Class_Dist_Interact":  encoders["Class"].transform([inp["Class"]])[0]*inp["Distance_km"],
        "Season_BookWin_Interact": encoders["Season"].transform([season])[0]*inp["Days_Until_Departure"],
        "Tier_SAF_Interact":    encoders["Airline_Tier"].transform([inp["Airline_Tier"]])[0]*inp["SAF_Zone"],
        "Stops_Dist_Interact":  stops_map[inp["Total_Stops"]]*inp["Distance_km"],
        "Env_Fleet_Interact":   inp["Env_Surcharge_Tier"]*inp["Fleet_Age_Years"],
        "Is_Long_Haul":         int(inp["Distance_km"] > 5000),
        "Is_Last_Minute":       int(inp["Days_Until_Departure"] < 7),
        "Is_Advance_Booking":   int(inp["Days_Until_Departure"] > 90),
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]
