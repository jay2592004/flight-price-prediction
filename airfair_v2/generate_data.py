"""
✈️  AirFair Vista — Synthetic Data Generator
=============================================
Generates realistic flight price data with ALL features needed for:
  - EDA (distributions, correlations, outliers)
  - Feature Engineering (derived columns, interactions)
  - ML Model Training (Phase 3)
  - SHAP Explainability (Phase 4 — BRD macro-factors show up in top-10)

Usage:
    python generate_data.py               # generates 100,000 rows (default)
    python generate_data.py --rows 500000 # custom row count
    python generate_data.py --seed 99     # custom random seed

Output:
    flight_price_synthetic.csv            (synthetic rows only)
    flight_price_combined.csv             (synthetic + original 20k merged)
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings("ignore")

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--rows", type=int, default=100_000, help="Number of rows to generate")
parser.add_argument("--seed", type=int, default=42,      help="Random seed for reproducibility")
args = parser.parse_args()

np.random.seed(args.seed)
N = args.rows
print(f"\n✈  AirFair Vista — Generating {N:,} synthetic rows (seed={args.seed})...\n")

# ══════════════════════════════════════════════════════════════════════════════
# 1. MASTER LOOKUP TABLES
# ══════════════════════════════════════════════════════════════════════════════

AIRLINES = [
    "Air France", "Air India", "AirAsia India", "British Airways",
    "Cathay Pacific", "Emirates", "Etihad Airways", "IndiGo",
    "Lufthansa", "Qatar Airways", "Singapore Airlines",
    "SpiceJet", "Thai Airways", "Turkish Airlines", "Vistara"
]

AIRLINE_TIER = {
    "Singapore Airlines": "premium", "Emirates": "premium",
    "Qatar Airways": "premium",      "British Airways": "premium",
    "Lufthansa": "premium",          "Cathay Pacific": "premium",
    "Air France": "mid",             "Etihad Airways": "mid",
    "Thai Airways": "mid",           "Turkish Airlines": "mid",
    "Air India": "mid",              "Vistara": "mid",
    "AirAsia India": "budget",       "IndiGo": "budget",
    "SpiceJet": "budget"
}
TIER_MULTIPLIER = {"premium": 1.30, "mid": 1.00, "budget": 0.82}

# BRD Phase 2 — Fleet age per airline (years)
FLEET_AGE = {
    "Singapore Airlines": 6.2,  "Emirates": 7.8,   "Qatar Airways": 6.5,
    "British Airways": 13.4,    "Lufthansa": 12.1,  "Cathay Pacific": 9.3,
    "Air France": 11.7,         "Etihad Airways": 8.4, "Thai Airways": 16.2,
    "Turkish Airlines": 9.8,    "Air India": 14.1,  "Vistara": 5.3,
    "AirAsia India": 7.1,       "IndiGo": 5.9,      "SpiceJet": 10.2
}

CITIES = [
    "Ahmedabad", "Bangalore", "Bangkok", "Chennai", "Delhi",
    "Doha", "Dubai", "Frankfurt", "Hong Kong", "Istanbul",
    "London", "Mumbai", "New York", "Paris", "Singapore"
]

# Real approximate distances (km) between city pairs
DISTANCE_MAP = {
    ("Delhi","Mumbai"):1400,      ("Delhi","Bangalore"):2150,   ("Delhi","Chennai"):2200,
    ("Delhi","Ahmedabad"):950,    ("Delhi","London"):6700,      ("Delhi","Dubai"):2200,
    ("Delhi","Singapore"):4150,   ("Delhi","Frankfurt"):6200,   ("Delhi","New York"):11750,
    ("Delhi","Bangkok"):2900,     ("Delhi","Hong Kong"):3750,   ("Delhi","Paris"):6600,
    ("Delhi","Istanbul"):4200,    ("Delhi","Doha"):2350,
    ("Mumbai","Bangalore"):980,   ("Mumbai","Chennai"):1330,    ("Mumbai","Ahmedabad"):530,
    ("Mumbai","London"):7200,     ("Mumbai","Dubai"):1950,      ("Mumbai","Singapore"):4180,
    ("Mumbai","Frankfurt"):7050,  ("Mumbai","New York"):12550,  ("Mumbai","Bangkok"):3080,
    ("Mumbai","Hong Kong"):4250,  ("Mumbai","Paris"):6950,      ("Mumbai","Istanbul"):4400,
    ("Mumbai","Doha"):1960,
    ("London","New York"):5540,   ("London","Dubai"):5500,      ("London","Singapore"):10850,
    ("London","Hong Kong"):9600,  ("London","Frankfurt"):650,   ("London","Paris"):340,
    ("London","Istanbul"):2510,   ("London","Bangkok"):9540,    ("London","Doha"):5500,
    ("Dubai","Singapore"):5840,   ("Dubai","Bangkok"):4900,     ("Dubai","Hong Kong"):6300,
    ("Dubai","Frankfurt"):5000,   ("Dubai","New York"):11020,   ("Dubai","Paris"):5250,
    ("Dubai","Istanbul"):2600,    ("Dubai","Doha"):350,         ("Dubai","Ahmedabad"):1950,
    ("Singapore","Hong Kong"):2580, ("Singapore","Bangkok"):1450,
    ("Singapore","New York"):15300, ("Singapore","Frankfurt"):10360,
    ("Singapore","Paris"):10730,  ("Singapore","Istanbul"):8000,
    ("Bangkok","Hong Kong"):1730, ("Bangkok","New York"):13600,
    ("Bangkok","Frankfurt"):9050, ("Bangkok","Istanbul"):7410,
    ("Frankfurt","New York"):6200,("Frankfurt","Istanbul"):2250,
    ("Frankfurt","Hong Kong"):9200,("Frankfurt","Doha"):4770,
    ("Paris","New York"):5840,    ("Paris","Istanbul"):2240,
    ("Paris","Hong Kong"):9600,   ("Paris","Doha"):5250,
    ("Istanbul","New York"):8500, ("Istanbul","Hong Kong"):8170,
    ("Istanbul","Doha"):3350,     ("Hong Kong","New York"):12970,
    ("Doha","New York"):11540,    ("Ahmedabad","Doha"):2000,
    ("Ahmedabad","Dubai"):1950,   ("Ahmedabad","Bangkok"):3700,
    ("Chennai","Dubai"):2900,     ("Chennai","Singapore"):3280,
    ("Chennai","Bangkok"):2200,   ("Bangalore","Dubai"):2900,
    ("Bangalore","Singapore"):3300,("Bangalore","Bangkok"):2150,
}

# BRD Phase 2 — SAF zone per destination city
# 0=no mandate, 1=voluntary target, 2=mandatory (EU/UK)
SAF_ZONE_MAP = {
    "London":2, "Frankfurt":2, "Paris":2,
    "Istanbul":1, "Singapore":1, "New York":1,
    "Dubai":0, "Doha":0, "Bangkok":0, "Hong Kong":0,
    "Delhi":0, "Mumbai":0, "Bangalore":0, "Chennai":0, "Ahmedabad":0
}

# BRD Phase 2 — Environmental surcharge tier per city
# 0=none, 1=low, 2=medium, 3=high (EU ETS = high)
ENV_TIER_MAP = {
    "London":3, "Frankfurt":3, "Paris":3,
    "New York":2, "Istanbul":2, "Singapore":2,
    "Dubai":1, "Doha":1, "Bangkok":1, "Hong Kong":1,
    "Delhi":0, "Mumbai":0, "Bangalore":0, "Chennai":0, "Ahmedabad":0
}

# BRD Phase 2 — Routes affected by airspace restrictions in 2026
RESTRICTED_PAIRS = {
    ("Delhi","Frankfurt"), ("Delhi","Paris"), ("Delhi","London"),
    ("Mumbai","Frankfurt"), ("Mumbai","Paris"), ("Mumbai","London"),
    ("Bangkok","Frankfurt"), ("Dubai","New York"), ("Doha","New York"),
}

CLASSES       = ["Economy", "Premium Economy", "Business", "First"]
CLASS_BASE    = {"Economy": 5000, "Premium Economy": 12000, "Business": 28000, "First": 55000}
CLASS_WEIGHTS = [0.52, 0.22, 0.18, 0.08]

STOPS         = ["non-stop", "1 stop", "2 stops"]
STOPS_WEIGHTS = [0.38, 0.42, 0.20]

# ══════════════════════════════════════════════════════════════════════════════
# 2. GENERATE CORE COLUMNS
# ══════════════════════════════════════════════════════════════════════════════

print("  [1/5] Sampling airlines, routes, classes, stops...")

airlines     = np.random.choice(AIRLINES, N)
sources      = np.random.choice(CITIES, N)
destinations = np.array([np.random.choice([c for c in CITIES if c != s]) for s in sources])
classes      = np.random.choice(CLASSES, N, p=CLASS_WEIGHTS)
stops        = np.random.choice(STOPS,   N, p=STOPS_WEIGHTS)

# Journey dates: Jan 2026 – Jun 2027 (18 months spread)
base_date     = datetime(2026, 1, 1)
journey_dates = [base_date + timedelta(days=int(d)) for d in np.random.randint(0, 540, N)]

# Days until departure — exponential, skewed toward medium-term bookings
days_until = np.random.exponential(scale=60, size=N).clip(1, 365).astype(int)

# ══════════════════════════════════════════════════════════════════════════════
# 3. DERIVE DATE & ROUTE FEATURES
# ══════════════════════════════════════════════════════════════════════════════

print("  [2/5] Engineering feature columns (date, distance, route, season)...")

journey_month   = np.array([d.month     for d in journey_dates])
journey_day     = np.array([d.day       for d in journey_dates])
journey_weekday = np.array([d.weekday() for d in journey_dates])
is_weekend      = (journey_weekday >= 5).astype(int)

def get_season(month):
    if month in [12, 1, 3, 4, 5]: return "peak"
    elif month in [2, 6, 10, 11]:  return "shoulder"
    else:                           return "off_peak"

season = np.array([get_season(m) for m in journey_month])

def get_distance(src, dst):
    d = DISTANCE_MAP.get((src, dst)) or DISTANCE_MAP.get((dst, src))
    return d if d else abs(CITIES.index(src) - CITIES.index(dst)) * 800 + 500

distances = np.array([get_distance(s, d) for s, d in zip(sources, destinations)])
distances = (distances * np.random.uniform(0.92, 1.10, N)).astype(int)

def build_route(src, dst, n_stops):
    intermediates = [c for c in CITIES if c != src and c != dst]
    if n_stops == "non-stop":
        return f"{src} - {dst}"
    elif n_stops == "1 stop":
        return f"{src} - {np.random.choice(intermediates)} - {dst}"
    else:
        v1, v2 = np.random.choice(intermediates, 2, replace=False)
        return f"{src} - {v1} - {v2} - {dst}"

routes = [build_route(s, d, st) for s, d, st in zip(sources, destinations, stops)]

# ══════════════════════════════════════════════════════════════════════════════
# 4. BRD PHASE 2 MACRO-FACTOR FEATURES
# ══════════════════════════════════════════════════════════════════════════════

print("  [3/5] Adding BRD Phase 2 macro-factor features...")

saf_zone    = np.array([SAF_ZONE_MAP.get(d, 0) for d in destinations])
env_tier    = np.array([max(ENV_TIER_MAP.get(s,0), ENV_TIER_MAP.get(d,0)) for s,d in zip(sources,destinations)])
fleet_age   = (np.array([FLEET_AGE[a] for a in airlines]) + np.random.normal(0, 0.5, N)).clip(3, 25).round(1)
is_restricted = np.array([1 if (s,d) in RESTRICTED_PAIRS or (d,s) in RESTRICTED_PAIRS else 0 for s,d in zip(sources,destinations)])

# Extra real-world features for richer EDA
geo_risk         = (is_restricted * 0.5 + np.random.beta(1.5, 6, N) * 0.5).round(3)
fuel_price_index = np.random.normal(100, 12, N).clip(70, 140).round(1)
seat_availability= np.random.beta(2, 3, N).round(3)
layover_hours    = np.where(stops=="non-stop", 0,
                   np.where(stops=="1 stop",   np.random.randint(1,8,N),
                                               np.random.randint(4,14,N)))
aircraft_type    = np.where(distances > 4000, "wide-body", "narrow-body")
airline_tier_col = np.array([AIRLINE_TIER[a] for a in airlines])

# ══════════════════════════════════════════════════════════════════════════════
# 5. PRICE COMPUTATION
#    Each feature contributes a measurable signal so SHAP works correctly
# ══════════════════════════════════════════════════════════════════════════════

print("  [4/5] Computing prices using multi-factor formula...")

prices = np.array([CLASS_BASE[c] for c in classes], dtype=float)

# Distance — non-linear (diminishing returns on ultra-long haul)
prices += distances * 1.9 * (1 - np.log1p(distances) * 0.018)

# Stops adjustment
prices *= np.where(stops=="non-stop", 1.08, np.where(stops=="1 stop", 0.95, 0.88))

# Booking window premium
prices *= np.where(days_until<7,  1.40,
          np.where(days_until<30, 1.20,
          np.where(days_until<60, 1.05, 1.00)))

# Season premium
prices *= np.where(season=="peak", 1.18, np.where(season=="shoulder", 1.00, 0.92))

# Weekend surcharge
prices *= np.where(is_weekend==1, 1.07, 1.00)

# Airline tier
prices *= np.array([TIER_MULTIPLIER[AIRLINE_TIER[a]] for a in airlines])

# BRD macro-factors ─────────────────────────────────────
prices *= np.where(saf_zone==2, 1.06, np.where(saf_zone==1, 1.02, 1.00))   # SAF
prices *= (1 + env_tier * 0.015)                                             # Env surcharge
prices *= (1 + (fleet_age - 8) * 0.004).clip(0.96, 1.10)                   # Fleet age
prices *= np.where(is_restricted==1, 1.09, 1.00)                            # Restricted airspace
# ────────────────────────────────────────────────────────

# Extra real-world signals
prices *= (0.70 + fuel_price_index / 333)          # fuel cost pass-through
prices *= (1.15 - seat_availability * 0.30)         # demand / seat pressure
prices *= np.where(layover_hours>8, 0.93, np.where(layover_hours>4, 0.97, 1.00))

# Gaussian noise ±8%
prices *= np.random.normal(1.0, 0.08, N)
prices  = prices.clip(1500, 250000).round(0).astype(int)

# ══════════════════════════════════════════════════════════════════════════════
# 6. ASSEMBLE & SAVE
# ══════════════════════════════════════════════════════════════════════════════

print("  [5/5] Assembling DataFrame and saving...")

df_syn = pd.DataFrame({
    # Original schema columns
    "Airline":                airlines,
    "Source":                 sources,
    "Destination":            destinations,
    "Route":                  routes,
    "Journey_Date":           [d.strftime("%d-%m-%Y") for d in journey_dates],
    "Journey_Month":          journey_month,
    "Journey_Day":            journey_day,
    "Total_Stops":            stops,
    "Class":                  classes,
    "Days_Until_Departure":   days_until,
    "Distance_km":            distances,
    # BRD Phase 2 macro-factor features
    "SAF_Zone":               saf_zone,
    "Env_Surcharge_Tier":     env_tier,
    "Fleet_Age_Years":        fleet_age,
    "Is_Restricted_Airspace": is_restricted,
    # Extra real-world features
    "Journey_Weekday":        journey_weekday,
    "Is_Weekend":             is_weekend,
    "Season":                 season,
    "Geo_Risk_Score":         geo_risk,
    "Fuel_Price_Index":       fuel_price_index,
    "Seat_Availability":      seat_availability,
    "Layover_Hours":          layover_hours,
    "Aircraft_Type":          aircraft_type,
    "Airline_Tier":           airline_tier_col,
    # Target
    "Price":                  prices,
})

out_dir  = os.path.dirname(os.path.abspath(__file__))
out_syn  = os.path.join(out_dir, "flight_price_synthetic.csv")
df_syn.to_csv(out_syn, index=False)
print(f"\n  ✓  Saved: flight_price_synthetic.csv")
print(f"     Rows: {len(df_syn):,}  |  Columns: {len(df_syn.columns)}")

# ── Merge with original 20k dataset ─────────────────────────────────────────
orig_path = os.path.join(out_dir, "flight_price_dataset.csv")
if os.path.exists(orig_path):
    print(f"\n  Merging with original 20k dataset...")
    df_orig = pd.read_csv(orig_path)

    # Back-fill new columns on original rows
    df_orig["SAF_Zone"]              = df_orig["Destination"].map(lambda x: SAF_ZONE_MAP.get(x, 0))
    df_orig["Env_Surcharge_Tier"]    = df_orig.apply(lambda r: max(ENV_TIER_MAP.get(r["Source"],0), ENV_TIER_MAP.get(r["Destination"],0)), axis=1)
    df_orig["Fleet_Age_Years"]       = df_orig["Airline"].map(lambda x: FLEET_AGE.get(x, 10.0))
    df_orig["Is_Restricted_Airspace"]= df_orig.apply(lambda r: 1 if (r["Source"],r["Destination"]) in RESTRICTED_PAIRS or (r["Destination"],r["Source"]) in RESTRICTED_PAIRS else 0, axis=1)

    def parse_date(s):
        try:    return datetime.strptime(str(s), "%d-%m-%Y")
        except: return datetime(2026, 6, 1)

    orig_dates = df_orig["Journey_Date"].apply(parse_date)
    df_orig["Journey_Weekday"]   = orig_dates.apply(lambda d: d.weekday())
    df_orig["Is_Weekend"]        = (df_orig["Journey_Weekday"] >= 5).astype(int)
    df_orig["Season"]            = df_orig["Journey_Month"].apply(get_season)
    df_orig["Geo_Risk_Score"]    = (df_orig["Is_Restricted_Airspace"] * 0.5 + np.random.beta(1.5, 6, len(df_orig)) * 0.5).round(3)
    df_orig["Fuel_Price_Index"]  = np.random.normal(100, 12, len(df_orig)).clip(70, 140).round(1)
    df_orig["Seat_Availability"] = np.random.beta(2, 3, len(df_orig)).round(3)
    df_orig["Layover_Hours"]     = df_orig["Total_Stops"].apply(lambda s: 0 if s=="non-stop" else (np.random.randint(1,8) if s=="1 stop" else np.random.randint(4,14)))
    df_orig["Aircraft_Type"]     = df_orig["Distance_km"].apply(lambda d: "wide-body" if d > 4000 else "narrow-body")
    df_orig["Airline_Tier"]      = df_orig["Airline"].map(AIRLINE_TIER)

    col_order = df_syn.columns.tolist()
    for col in col_order:
        if col not in df_orig.columns:
            df_orig[col] = 0
    df_orig = df_orig[col_order]

    df_combined = pd.concat([df_orig, df_syn], ignore_index=True)
    df_combined  = df_combined.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    out_comb = os.path.join(out_dir, "flight_price_combined.csv")
    df_combined.to_csv(out_comb, index=False)
    print(f"  ✓  Saved: flight_price_combined.csv")
    print(f"     Rows: {len(df_combined):,}  |  Columns: {len(df_combined.columns)}")
else:
    print("  ⚠  Original dataset not found — skipping merge step.")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "─"*55)
print("  DATASET SUMMARY")
print("─"*55)
print(f"  Rows generated  : {len(df_syn):>10,}")
print(f"  Total columns   : {len(df_syn.columns):>10}")
print(f"\n  Price distribution (₹):")
print(f"    Min    : {df_syn['Price'].min():>10,.0f}")
print(f"    Median : {df_syn['Price'].median():>10,.0f}")
print(f"    Mean   : {df_syn['Price'].mean():>10,.0f}")
print(f"    Max    : {df_syn['Price'].max():>10,.0f}")
print(f"\n  Class split:")
for cls in CLASSES:
    print(f"    {cls:<18}: {(df_syn['Class']==cls).mean()*100:.1f}%")
print(f"\n  SAF zone split:")
for z in [0,1,2]:
    print(f"    Zone {z}: {(df_syn['SAF_Zone']==z).mean()*100:.1f}%")
print(f"\n  Restricted airspace rows : {df_syn['Is_Restricted_Airspace'].sum():,}")
print(f"  Wide-body routes         : {(df_syn['Aircraft_Type']=='wide-body').sum():,}")
print("─"*55)
print("\n  ✅  Done! Ready for EDA, feature engineering & model training.\n")
