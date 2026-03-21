"""
src/data/generator.py
Generates 100k synthetic flight price records with all 25 features.
Called by the training pipeline — not at app runtime.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# ── Master lookup tables ──────────────────────────────────────────────────────
AIRLINES = [
    "Air France","Air India","AirAsia India","British Airways","Cathay Pacific",
    "Emirates","Etihad Airways","IndiGo","Lufthansa","Qatar Airways",
    "Singapore Airlines","SpiceJet","Thai Airways","Turkish Airlines","Vistara",
]
AIRLINE_TIER = {
    "Singapore Airlines":"premium","Emirates":"premium","Qatar Airways":"premium",
    "British Airways":"premium","Lufthansa":"premium","Cathay Pacific":"premium",
    "Air France":"mid","Etihad Airways":"mid","Thai Airways":"mid",
    "Turkish Airlines":"mid","Air India":"mid","Vistara":"mid",
    "AirAsia India":"budget","IndiGo":"budget","SpiceJet":"budget",
}
TIER_MULT = {"premium":1.30,"mid":1.00,"budget":0.82}
FLEET_AGE = {
    "Singapore Airlines":6.2,"Emirates":7.8,"Qatar Airways":6.5,
    "British Airways":13.4,"Lufthansa":12.1,"Cathay Pacific":9.3,
    "Air France":11.7,"Etihad Airways":8.4,"Thai Airways":16.2,
    "Turkish Airlines":9.8,"Air India":14.1,"Vistara":5.3,
    "AirAsia India":7.1,"IndiGo":5.9,"SpiceJet":10.2,
}
CITIES = [
    "Ahmedabad","Bangalore","Bangkok","Chennai","Delhi","Doha","Dubai",
    "Frankfurt","Hong Kong","Istanbul","London","Mumbai","New York","Paris","Singapore",
]
DIST_MAP = {
    ("Delhi","Mumbai"):1400,("Delhi","Bangalore"):2150,("Delhi","Chennai"):2200,
    ("Delhi","Ahmedabad"):950,("Delhi","London"):6700,("Delhi","Dubai"):2200,
    ("Delhi","Singapore"):4150,("Delhi","Frankfurt"):6200,("Delhi","New York"):11750,
    ("Delhi","Bangkok"):2900,("Delhi","Hong Kong"):3750,("Delhi","Paris"):6600,
    ("Delhi","Istanbul"):4200,("Delhi","Doha"):2350,
    ("Mumbai","Bangalore"):980,("Mumbai","Chennai"):1330,("Mumbai","Ahmedabad"):530,
    ("Mumbai","London"):7200,("Mumbai","Dubai"):1950,("Mumbai","Singapore"):4180,
    ("Mumbai","Frankfurt"):7050,("Mumbai","New York"):12550,("Mumbai","Bangkok"):3080,
    ("Mumbai","Hong Kong"):4250,("Mumbai","Paris"):6950,("Mumbai","Istanbul"):4400,
    ("Mumbai","Doha"):1960,("London","New York"):5540,("London","Dubai"):5500,
    ("London","Singapore"):10850,("London","Hong Kong"):9600,("London","Frankfurt"):650,
    ("London","Paris"):340,("London","Istanbul"):2510,("London","Bangkok"):9540,
    ("London","Doha"):5500,("Dubai","Singapore"):5840,("Dubai","Bangkok"):4900,
    ("Dubai","Hong Kong"):6300,("Dubai","Frankfurt"):5000,("Dubai","New York"):11020,
    ("Dubai","Paris"):5250,("Dubai","Istanbul"):2600,("Dubai","Doha"):350,
    ("Dubai","Ahmedabad"):1950,("Singapore","Hong Kong"):2580,("Singapore","Bangkok"):1450,
    ("Singapore","New York"):15300,("Singapore","Frankfurt"):10360,
    ("Singapore","Paris"):10730,("Singapore","Istanbul"):8000,
    ("Bangkok","Hong Kong"):1730,("Bangkok","New York"):13600,
    ("Bangkok","Frankfurt"):9050,("Bangkok","Istanbul"):7410,
    ("Frankfurt","New York"):6200,("Frankfurt","Istanbul"):2250,
    ("Frankfurt","Hong Kong"):9200,("Frankfurt","Doha"):4770,
    ("Paris","New York"):5840,("Paris","Istanbul"):2240,
    ("Paris","Hong Kong"):9600,("Paris","Doha"):5250,
    ("Istanbul","New York"):8500,("Istanbul","Hong Kong"):8170,
    ("Istanbul","Doha"):3350,("Hong Kong","New York"):12970,
    ("Doha","New York"):11540,("Ahmedabad","Doha"):2000,
    ("Ahmedabad","Dubai"):1950,("Ahmedabad","Bangkok"):3700,
    ("Chennai","Dubai"):2900,("Chennai","Singapore"):3280,
    ("Chennai","Bangkok"):2200,("Bangalore","Dubai"):2900,
    ("Bangalore","Singapore"):3300,("Bangalore","Bangkok"):2150,
}
SAF_ZONE = {
    "London":2,"Frankfurt":2,"Paris":2,"Istanbul":1,"Singapore":1,"New York":1,
    "Dubai":0,"Doha":0,"Bangkok":0,"Hong Kong":0,
    "Delhi":0,"Mumbai":0,"Bangalore":0,"Chennai":0,"Ahmedabad":0,
}
ENV_TIER = {
    "London":3,"Frankfurt":3,"Paris":3,"New York":2,"Istanbul":2,"Singapore":2,
    "Dubai":1,"Doha":1,"Bangkok":1,"Hong Kong":1,
    "Delhi":0,"Mumbai":0,"Bangalore":0,"Chennai":0,"Ahmedabad":0,
}
RESTRICTED = {
    ("Delhi","Frankfurt"),("Delhi","Paris"),("Delhi","London"),
    ("Mumbai","Frankfurt"),("Mumbai","Paris"),("Mumbai","London"),
    ("Bangkok","Frankfurt"),("Dubai","New York"),("Doha","New York"),
}
CLASSES    = ["Economy","Premium Economy","Business","First"]
CLASS_BASE = {"Economy":5000,"Premium Economy":12000,"Business":28000,"First":55000}
CLASS_W    = [0.52, 0.22, 0.18, 0.08]
STOPS      = ["non-stop","1 stop","2 stops"]
STOPS_W    = [0.38, 0.42, 0.20]


def get_season(month: int) -> str:
    if month in [12,1,3,4,5]: return "peak"
    if month in [2,6,10,11]:  return "shoulder"
    return "off_peak"


def get_distance(src: str, dst: str) -> int:
    d = DIST_MAP.get((src,dst)) or DIST_MAP.get((dst,src))
    return d or abs(CITIES.index(src) - CITIES.index(dst)) * 800 + 500


def build_route(src: str, dst: str, stops: str) -> str:
    iv = [c for c in CITIES if c != src and c != dst]
    if stops == "non-stop": return f"{src} - {dst}"
    if stops == "1 stop":   return f"{src} - {np.random.choice(iv)} - {dst}"
    v1,v2 = np.random.choice(iv, 2, replace=False)
    return f"{src} - {v1} - {v2} - {dst}"


def generate(n_rows: int = 100_000, seed: int = 42) -> pd.DataFrame:
    """Generate n_rows synthetic flight records. Returns a DataFrame with 25 columns."""
    np.random.seed(seed)
    N = n_rows

    airlines = np.random.choice(AIRLINES, N)
    sources  = np.random.choice(CITIES,   N)
    dests    = np.array([np.random.choice([c for c in CITIES if c != s]) for s in sources])
    classes  = np.random.choice(CLASSES, N, p=CLASS_W)
    stops_arr= np.random.choice(STOPS,   N, p=STOPS_W)

    base_dt  = datetime(2026,1,1)
    jdates   = [base_dt + timedelta(days=int(d)) for d in np.random.randint(0,540,N)]
    days_dep = np.random.exponential(60, N).clip(1,365).astype(int)

    jmonth = np.array([d.month     for d in jdates])
    jday   = np.array([d.day       for d in jdates])
    jwday  = np.array([d.weekday() for d in jdates])
    wkend  = (jwday >= 5).astype(int)
    season = np.array([get_season(m) for m in jmonth])

    dists  = (np.array([get_distance(s,d) for s,d in zip(sources,dests)])
              * np.random.uniform(0.92,1.10,N)).astype(int)
    routes = [build_route(s,d,st) for s,d,st in zip(sources,dests,stops_arr)]

    saf    = np.array([SAF_ZONE.get(d,0) for d in dests])
    env    = np.array([max(ENV_TIER.get(s,0),ENV_TIER.get(d,0)) for s,d in zip(sources,dests)])
    fage   = (np.array([FLEET_AGE[a] for a in airlines]) + np.random.normal(0,0.5,N)).clip(3,25).round(1)
    restr  = np.array([1 if (s,d) in RESTRICTED or (d,s) in RESTRICTED else 0
                       for s,d in zip(sources,dests)])
    georisk= (restr*0.5 + np.random.beta(1.5,6,N)*0.5).round(3)
    fuel   = np.random.normal(100,12,N).clip(70,140).round(1)
    seatav = np.random.beta(2,3,N).round(3)
    lhours = np.where(stops_arr=="non-stop", 0,
             np.where(stops_arr=="1 stop", np.random.randint(1,8,N),
                                           np.random.randint(4,14,N)))
    actype = np.where(dists>4000,"wide-body","narrow-body")
    tier_c = np.array([AIRLINE_TIER[a] for a in airlines])

    # Price formula
    p = np.array([CLASS_BASE[c] for c in classes], dtype=float)
    p += dists*1.9*(1-np.log1p(dists)*0.018)
    p *= np.where(stops_arr=="non-stop",1.08,np.where(stops_arr=="1 stop",0.95,0.88))
    p *= np.where(days_dep<7,1.40,np.where(days_dep<30,1.20,np.where(days_dep<60,1.05,1.00)))
    p *= np.where(season=="peak",1.18,np.where(season=="shoulder",1.00,0.92))
    p *= np.where(wkend==1,1.07,1.00)
    p *= np.array([TIER_MULT[AIRLINE_TIER[a]] for a in airlines])
    p *= np.where(saf==2,1.06,np.where(saf==1,1.02,1.00))
    p *= (1+env*0.015)
    p *= (1+(fage-8)*0.004).clip(0.96,1.10)
    p *= np.where(restr==1,1.09,1.00)
    p *= (0.70+fuel/333)
    p *= (1.15-seatav*0.30)
    p *= np.where(lhours>8,0.93,np.where(lhours>4,0.97,1.00))
    p *= np.random.normal(1.0,0.08,N)
    p  = p.clip(1500,250000).round(0).astype(int)

    return pd.DataFrame({
        "Airline":airlines,"Source":sources,"Destination":dests,"Route":routes,
        "Journey_Date":[d.strftime("%d-%m-%Y") for d in jdates],
        "Journey_Month":jmonth,"Journey_Day":jday,"Total_Stops":stops_arr,"Class":classes,
        "Days_Until_Departure":days_dep,"Distance_km":dists,
        "SAF_Zone":saf,"Env_Surcharge_Tier":env,"Fleet_Age_Years":fage,
        "Is_Restricted_Airspace":restr,"Journey_Weekday":jwday,"Is_Weekend":wkend,
        "Season":season,"Geo_Risk_Score":georisk,"Fuel_Price_Index":fuel,
        "Seat_Availability":seatav,"Layover_Hours":lhours,
        "Aircraft_Type":actype,"Airline_Tier":tier_c,"Price":p,
    })
