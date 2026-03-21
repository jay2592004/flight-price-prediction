"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          ✈  AirFair Vista — End-to-End ML Pipeline                         ║
║          Flight Price Prediction | B.Tech Final Year Project                ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHAT THIS SCRIPT DOES (top to bottom):
  STEP 0  — Generate synthetic data (calls generate_data.py logic inline)
  STEP 1  — Load & validate combined dataset
  STEP 2  — Exploratory Data Analysis  (Matplotlib + Seaborn + Plotly)
  STEP 3  — Feature Engineering        (encoding, interactions, scaling)
  STEP 4  — Time-Series Cross-Validation split  (BRD requirement)
  STEP 5  — Baseline model             (30-day moving average)
  STEP 6  — Model training & comparison (LinearReg, RandomForest, XGBoost, LightGBM)
  STEP 7  — SHAP Explainability        (BRD Phase 4)
  STEP 8  — Save model artefacts       (pkl files for Streamlit app)

Run:
    pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost lightgbm shap joblib
    python pipeline.py
"""

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, warnings, time
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # headless — no GUI needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express       as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

from datetime  import datetime, timedelta
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost  as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  ⚠  xgboost not installed — skipping XGBoost model")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("  ⚠  lightgbm not installed — skipping LightGBM model")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("  ⚠  shap not installed — skipping SHAP explainability")

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
SYNTH_CSV     = os.path.join(BASE_DIR, "flight_price_synthetic.csv")
ORIG_CSV      = os.path.join(BASE_DIR, "flight_price_dataset.csv")
COMBINED_CSV  = os.path.join(BASE_DIR, "flight_price_combined.csv")
PLOTS_DIR     = os.path.join(BASE_DIR, "pipeline_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      saved → pipeline_plots/{name}")

def saveplotly(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.write_html(path)
    print(f"      saved → pipeline_plots/{name}")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
PALETTE = ["#1565C0","#0D9B6E","#E65100","#7B1FA2","#C62828","#00695C","#F57F17"]

print("\n" + "═"*65)
print("  ✈  AirFair Vista — End-to-End ML Pipeline")
print("═"*65)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 0 — GENERATE SYNTHETIC DATA (inline, no subprocess needed)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 0] Generating synthetic data...")

if os.path.exists(COMBINED_CSV):
    print("  ✓  flight_price_combined.csv already exists — skipping generation.")
else:
    # ── inline generator ─────────────────────────────────────────────────────
    np.random.seed(42)
    N = 100_000

    AIRLINES = ["Air France","Air India","AirAsia India","British Airways",
                "Cathay Pacific","Emirates","Etihad Airways","IndiGo",
                "Lufthansa","Qatar Airways","Singapore Airlines",
                "SpiceJet","Thai Airways","Turkish Airlines","Vistara"]

    AIRLINE_TIER = {
        "Singapore Airlines":"premium","Emirates":"premium","Qatar Airways":"premium",
        "British Airways":"premium","Lufthansa":"premium","Cathay Pacific":"premium",
        "Air France":"mid","Etihad Airways":"mid","Thai Airways":"mid",
        "Turkish Airlines":"mid","Air India":"mid","Vistara":"mid",
        "AirAsia India":"budget","IndiGo":"budget","SpiceJet":"budget"}
    TIER_MULT = {"premium":1.30,"mid":1.00,"budget":0.82}

    FLEET_AGE = {"Singapore Airlines":6.2,"Emirates":7.8,"Qatar Airways":6.5,
                 "British Airways":13.4,"Lufthansa":12.1,"Cathay Pacific":9.3,
                 "Air France":11.7,"Etihad Airways":8.4,"Thai Airways":16.2,
                 "Turkish Airlines":9.8,"Air India":14.1,"Vistara":5.3,
                 "AirAsia India":7.1,"IndiGo":5.9,"SpiceJet":10.2}

    CITIES = ["Ahmedabad","Bangalore","Bangkok","Chennai","Delhi","Doha","Dubai",
              "Frankfurt","Hong Kong","Istanbul","London","Mumbai","New York","Paris","Singapore"]

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

    SAF_ZONE = {"London":2,"Frankfurt":2,"Paris":2,"Istanbul":1,"Singapore":1,
                "New York":1,"Dubai":0,"Doha":0,"Bangkok":0,"Hong Kong":0,
                "Delhi":0,"Mumbai":0,"Bangalore":0,"Chennai":0,"Ahmedabad":0}
    ENV_TIER = {"London":3,"Frankfurt":3,"Paris":3,"New York":2,"Istanbul":2,
                "Singapore":2,"Dubai":1,"Doha":1,"Bangkok":1,"Hong Kong":1,
                "Delhi":0,"Mumbai":0,"Bangalore":0,"Chennai":0,"Ahmedabad":0}
    RESTRICTED = {("Delhi","Frankfurt"),("Delhi","Paris"),("Delhi","London"),
                  ("Mumbai","Frankfurt"),("Mumbai","Paris"),("Mumbai","London"),
                  ("Bangkok","Frankfurt"),("Dubai","New York"),("Doha","New York")}

    CLASSES      = ["Economy","Premium Economy","Business","First"]
    CLASS_BASE   = {"Economy":5000,"Premium Economy":12000,"Business":28000,"First":55000}
    CLASS_W      = [0.52,0.22,0.18,0.08]
    STOPS        = ["non-stop","1 stop","2 stops"]
    STOPS_W      = [0.38,0.42,0.20]

    def get_season(m):
        return "peak" if m in [12,1,3,4,5] else ("shoulder" if m in [2,6,10,11] else "off_peak")

    airlines_s  = np.random.choice(AIRLINES, N)
    sources_s   = np.random.choice(CITIES, N)
    dests_s     = np.array([np.random.choice([c for c in CITIES if c!=s]) for s in sources_s])
    classes_s   = np.random.choice(CLASSES, N, p=CLASS_W)
    stops_s     = np.random.choice(STOPS,   N, p=STOPS_W)
    base        = datetime(2026,1,1)
    jdates      = [base + timedelta(days=int(d)) for d in np.random.randint(0,540,N)]
    days_s      = np.random.exponential(60,N).clip(1,365).astype(int)

    jmonth  = np.array([d.month     for d in jdates])
    jday    = np.array([d.day       for d in jdates])
    jwday   = np.array([d.weekday() for d in jdates])
    weekend = (jwday>=5).astype(int)
    season  = np.array([get_season(m) for m in jmonth])

    def gdist(s,d):
        v=DIST_MAP.get((s,d)) or DIST_MAP.get((d,s))
        return v or abs(CITIES.index(s)-CITIES.index(d))*800+500
    dists = (np.array([gdist(s,d) for s,d in zip(sources_s,dests_s)])*np.random.uniform(0.92,1.10,N)).astype(int)

    def broute(s,d,st):
        iv=[c for c in CITIES if c!=s and c!=d]
        if st=="non-stop":  return f"{s} - {d}"
        if st=="1 stop":    return f"{s} - {np.random.choice(iv)} - {d}"
        v1,v2=np.random.choice(iv,2,replace=False); return f"{s} - {v1} - {v2} - {d}"
    routes_s = [broute(s,d,st) for s,d,st in zip(sources_s,dests_s,stops_s)]

    saf    = np.array([SAF_ZONE.get(d,0) for d in dests_s])
    env    = np.array([max(ENV_TIER.get(s,0),ENV_TIER.get(d,0)) for s,d in zip(sources_s,dests_s)])
    fage   = (np.array([FLEET_AGE[a] for a in airlines_s])+np.random.normal(0,0.5,N)).clip(3,25).round(1)
    restr  = np.array([1 if (s,d) in RESTRICTED or (d,s) in RESTRICTED else 0 for s,d in zip(sources_s,dests_s)])
    georisk= (restr*0.5+np.random.beta(1.5,6,N)*0.5).round(3)
    fuel   = np.random.normal(100,12,N).clip(70,140).round(1)
    seatav = np.random.beta(2,3,N).round(3)
    lhours = np.where(stops_s=="non-stop",0,np.where(stops_s=="1 stop",np.random.randint(1,8,N),np.random.randint(4,14,N)))
    actype = np.where(dists>4000,"wide-body","narrow-body")
    tier_c = np.array([AIRLINE_TIER[a] for a in airlines_s])

    p = np.array([CLASS_BASE[c] for c in classes_s],dtype=float)
    p += dists*1.9*(1-np.log1p(dists)*0.018)
    p *= np.where(stops_s=="non-stop",1.08,np.where(stops_s=="1 stop",0.95,0.88))
    p *= np.where(days_s<7,1.40,np.where(days_s<30,1.20,np.where(days_s<60,1.05,1.00)))
    p *= np.where(season=="peak",1.18,np.where(season=="shoulder",1.00,0.92))
    p *= np.where(weekend==1,1.07,1.00)
    p *= np.array([TIER_MULT[AIRLINE_TIER[a]] for a in airlines_s])
    p *= np.where(saf==2,1.06,np.where(saf==1,1.02,1.00))
    p *= (1+env*0.015)
    p *= (1+(fage-8)*0.004).clip(0.96,1.10)
    p *= np.where(restr==1,1.09,1.00)
    p *= (0.70+fuel/333)
    p *= (1.15-seatav*0.30)
    p *= np.where(lhours>8,0.93,np.where(lhours>4,0.97,1.00))
    p *= np.random.normal(1.0,0.08,N)
    p  = p.clip(1500,250000).round(0).astype(int)

    df_syn = pd.DataFrame({
        "Airline":airlines_s,"Source":sources_s,"Destination":dests_s,"Route":routes_s,
        "Journey_Date":[d.strftime("%d-%m-%Y") for d in jdates],
        "Journey_Month":jmonth,"Journey_Day":jday,"Total_Stops":stops_s,"Class":classes_s,
        "Days_Until_Departure":days_s,"Distance_km":dists,
        "SAF_Zone":saf,"Env_Surcharge_Tier":env,"Fleet_Age_Years":fage,
        "Is_Restricted_Airspace":restr,"Journey_Weekday":jwday,"Is_Weekend":weekend,
        "Season":season,"Geo_Risk_Score":georisk,"Fuel_Price_Index":fuel,
        "Seat_Availability":seatav,"Layover_Hours":lhours,
        "Aircraft_Type":actype,"Airline_Tier":tier_c,"Price":p,
    })

    # Back-fill original dataset with new columns
    if os.path.exists(ORIG_CSV):
        df_o = pd.read_csv(ORIG_CSV)
        df_o["SAF_Zone"]              = df_o["Destination"].map(lambda x: SAF_ZONE.get(x,0))
        df_o["Env_Surcharge_Tier"]    = df_o.apply(lambda r: max(ENV_TIER.get(r["Source"],0),ENV_TIER.get(r["Destination"],0)),axis=1)
        df_o["Fleet_Age_Years"]       = df_o["Airline"].map(lambda x: FLEET_AGE.get(x,10.0))
        df_o["Is_Restricted_Airspace"]= df_o.apply(lambda r: 1 if (r["Source"],r["Destination"]) in RESTRICTED or (r["Destination"],r["Source"]) in RESTRICTED else 0,axis=1)
        def pd_(s):
            try:    return datetime.strptime(str(s),"%d-%m-%Y")
            except: return datetime(2026,6,1)
        od=df_o["Journey_Date"].apply(pd_)
        df_o["Journey_Weekday"]   = od.apply(lambda d: d.weekday())
        df_o["Is_Weekend"]        = (df_o["Journey_Weekday"]>=5).astype(int)
        df_o["Season"]            = df_o["Journey_Month"].apply(get_season)
        df_o["Geo_Risk_Score"]    = (df_o["Is_Restricted_Airspace"]*0.5+np.random.beta(1.5,6,len(df_o))*0.5).round(3)
        df_o["Fuel_Price_Index"]  = np.random.normal(100,12,len(df_o)).clip(70,140).round(1)
        df_o["Seat_Availability"] = np.random.beta(2,3,len(df_o)).round(3)
        df_o["Layover_Hours"]     = df_o["Total_Stops"].apply(lambda s: 0 if s=="non-stop" else (np.random.randint(1,8) if s=="1 stop" else np.random.randint(4,14)))
        df_o["Aircraft_Type"]     = df_o["Distance_km"].apply(lambda d: "wide-body" if d>4000 else "narrow-body")
        df_o["Airline_Tier"]      = df_o["Airline"].map(AIRLINE_TIER)
        for col in df_syn.columns:
            if col not in df_o.columns: df_o[col]=0
        df_o = df_o[df_syn.columns]
        df_comb = pd.concat([df_o, df_syn],ignore_index=True).sample(frac=1,random_state=42).reset_index(drop=True)
    else:
        df_comb = df_syn.copy()

    df_syn.to_csv(SYNTH_CSV, index=False)
    df_comb.to_csv(COMBINED_CSV, index=False)
    print(f"  ✓  Generated {len(df_syn):,} synthetic rows")
    print(f"  ✓  Combined dataset: {len(df_comb):,} rows × {len(df_comb.columns)} columns")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD & VALIDATE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 1] Loading and validating dataset...")

df = pd.read_csv(COMBINED_CSV)
print(f"  Shape        : {df.shape}")
print(f"  Columns      : {list(df.columns)}")
print(f"  Nulls        : {df.isnull().sum().sum()}")
print(f"  Duplicates   : {df.duplicated().sum()}")
print(f"  Price range  : ₹{df['Price'].min():,} – ₹{df['Price'].max():,}")
print(f"  Price mean   : ₹{df['Price'].mean():,.0f}")
print("\n  Data types:")
print(df.dtypes.to_string())

# Drop exact duplicates
df = df.drop_duplicates().reset_index(drop=True)

# Parse journey date — sort by date for time-series split
df["Journey_Date_dt"] = pd.to_datetime(df["Journey_Date"], format="%d-%m-%Y", errors="coerce")
df = df.sort_values("Journey_Date_dt").reset_index(drop=True)

print(f"\n  Date range   : {df['Journey_Date_dt'].min().date()} → {df['Journey_Date_dt'].max().date()}")
print(f"  Clean shape  : {df.shape}")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 2] Exploratory Data Analysis...")

# ── 2.1 Price distribution ───────────────────────────────────────────────────
print("  [2.1] Price distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["Price"], bins=80, color=PALETTE[0], edgecolor="white", linewidth=0.4)
axes[0].set_title("Price Distribution (₹)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Price (₹)")
axes[0].set_ylabel("Count")
axes[0].axvline(df["Price"].median(), color="red", linestyle="--", label=f"Median ₹{df['Price'].median():,.0f}")
axes[0].legend()

axes[1].hist(np.log1p(df["Price"]), bins=80, color=PALETTE[1], edgecolor="white", linewidth=0.4)
axes[1].set_title("Log-Price Distribution (log₁₊ₓ)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("log(1 + Price)")
axes[1].set_ylabel("Count")
plt.tight_layout()
savefig("01_price_distribution.png")

# ── 2.2 Price by Class ───────────────────────────────────────────────────────
print("  [2.2] Price by class...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
class_order = ["Economy","Premium Economy","Business","First"]
class_medians = df.groupby("Class")["Price"].median().reindex(class_order)
axes[0].bar(class_order, class_medians.values, color=PALETTE[:4], edgecolor="white")
axes[0].set_title("Median Price by Cabin Class", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Median Price (₹)")
for i,(k,v) in enumerate(class_medians.items()):
    axes[0].text(i, v+1000, f"₹{v:,.0f}", ha="center", fontsize=9)

df.boxplot(column="Price", by="Class", ax=axes[1], notch=False,
           positions=range(len(class_order)), widths=0.5)
axes[1].set_title("Price Distribution by Class", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Class")
axes[1].set_ylabel("Price (₹)")
plt.suptitle("")
plt.tight_layout()
savefig("02_price_by_class.png")

# ── 2.3 Price by Airline ─────────────────────────────────────────────────────
print("  [2.3] Price by airline (Plotly interactive)...")
airline_med = df.groupby("Airline")["Price"].median().sort_values(ascending=True).reset_index()
fig = px.bar(airline_med, x="Price", y="Airline", orientation="h",
             title="Median Flight Price by Airline",
             color="Price", color_continuous_scale="Blues",
             labels={"Price":"Median Price (₹)"})
fig.update_layout(height=500, showlegend=False)
saveplotly(fig, "03_price_by_airline.html")

# ── 2.4 Price vs Distance scatter ───────────────────────────────────────────
print("  [2.4] Price vs distance...")
sample = df.sample(min(8000, len(df)), random_state=42)
fig = px.scatter(sample, x="Distance_km", y="Price", color="Class",
                 title="Price vs Distance by Cabin Class",
                 opacity=0.5, hover_data=["Airline","Source","Destination"],
                 color_discrete_sequence=PALETTE)
fig.update_layout(height=500)
saveplotly(fig, "04_price_vs_distance.html")

# ── 2.5 Booking window vs Price ─────────────────────────────────────────────
print("  [2.5] Booking window vs price...")
bins = [0,7,14,30,60,90,180,365]
labels = ["0-7","8-14","15-30","31-60","61-90","91-180","181-365"]
df["Booking_Bin"] = pd.cut(df["Days_Until_Departure"], bins=bins, labels=labels)
bw_med = df.groupby("Booking_Bin", observed=True)["Price"].median().reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(bw_med["Booking_Bin"], bw_med["Price"], marker="o", color=PALETTE[2], linewidth=2.5, markersize=8)
ax.fill_between(range(len(bw_med)), bw_med["Price"], alpha=0.15, color=PALETTE[2])
ax.set_title("Median Price vs Booking Window", fontsize=13, fontweight="bold")
ax.set_xlabel("Days Until Departure (bins)")
ax.set_ylabel("Median Price (₹)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("05_booking_window.png")

# ── 2.6 Seasonality ─────────────────────────────────────────────────────────
print("  [2.6] Seasonality...")
monthly = df.groupby("Journey_Month")["Price"].median().reset_index()
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
monthly["Month_Name"] = monthly["Journey_Month"].apply(lambda m: month_names[m-1])

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(monthly["Month_Name"], monthly["Price"], color=[
    "#C62828" if m in [12,1,3,4,5] else ("#E65100" if m in [2,6,10,11] else "#1565C0")
    for m in monthly["Journey_Month"]], edgecolor="white")
ax.set_title("Median Price by Month (Red=Peak, Orange=Shoulder, Blue=Off-peak)", fontsize=12, fontweight="bold")
ax.set_ylabel("Median Price (₹)")
plt.tight_layout()
savefig("06_seasonality.png")

# ── 2.7 BRD Macro-factor distributions ──────────────────────────────────────
print("  [2.7] BRD macro-factor plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# SAF Zone
saf_med = df.groupby("SAF_Zone")["Price"].median()
axes[0,0].bar(["Zone 0\n(No mandate)","Zone 1\n(Voluntary)","Zone 2\n(Mandatory)"],
              [saf_med.get(0,0), saf_med.get(1,0), saf_med.get(2,0)],
              color=["#0D9B6E","#E65100","#C62828"], edgecolor="white")
axes[0,0].set_title("Median Price by SAF Zone (BRD)", fontweight="bold")
axes[0,0].set_ylabel("Median Price (₹)")

# Env Surcharge Tier
env_med = df.groupby("Env_Surcharge_Tier")["Price"].median()
axes[0,1].bar([f"Tier {i}" for i in sorted(env_med.index)], env_med.values,
              color=PALETTE[:len(env_med)], edgecolor="white")
axes[0,1].set_title("Median Price by Env Surcharge Tier (BRD)", fontweight="bold")
axes[0,1].set_ylabel("Median Price (₹)")

# Fleet Age vs Price scatter
axes[1,0].scatter(df["Fleet_Age_Years"].sample(5000, random_state=1),
                  df["Price"].sample(5000, random_state=1),
                  alpha=0.15, color=PALETTE[3], s=10)
axes[1,0].set_title("Fleet Age vs Price (BRD)", fontweight="bold")
axes[1,0].set_xlabel("Fleet Age (Years)")
axes[1,0].set_ylabel("Price (₹)")

# Restricted airspace
ra_med = df.groupby("Is_Restricted_Airspace")["Price"].median()
axes[1,1].bar(["Normal route","Restricted airspace\n(+9% reroute)"],
              [ra_med.get(0,0), ra_med.get(1,0)], color=["#1565C0","#C62828"], edgecolor="white")
axes[1,1].set_title("Price: Normal vs Restricted Airspace (BRD)", fontweight="bold")
axes[1,1].set_ylabel("Median Price (₹)")

plt.suptitle("BRD Phase 2 Macro-Factor Impact on Price", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
savefig("07_brd_macrofactors.png")

# ── 2.8 Correlation heatmap ──────────────────────────────────────────────────
print("  [2.8] Correlation heatmap...")
num_cols = ["Price","Distance_km","Days_Until_Departure","Journey_Month","Journey_Weekday",
            "Is_Weekend","SAF_Zone","Env_Surcharge_Tier","Fleet_Age_Years",
            "Is_Restricted_Airspace","Geo_Risk_Score","Fuel_Price_Index",
            "Seat_Availability","Layover_Hours"]
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, linewidths=0.5, annot_kws={"size":8})
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
savefig("08_correlation_heatmap.png")

# ── 2.9 Airline tier vs price ────────────────────────────────────────────────
print("  [2.9] Airline tier analysis...")
fig = px.box(df, x="Airline_Tier", y="Price",
             category_orders={"Airline_Tier":["budget","mid","premium"]},
             color="Airline_Tier", title="Price Distribution by Airline Tier",
             color_discrete_map={"budget":PALETTE[1],"mid":PALETTE[0],"premium":PALETTE[2]})
saveplotly(fig, "09_airline_tier_price.html")

# ── 2.10 Seat availability vs price ─────────────────────────────────────────
print("  [2.10] Seat availability & fuel index...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
s = df.sample(5000, random_state=42)
axes[0].scatter(s["Seat_Availability"], s["Price"], alpha=0.2, color=PALETTE[4], s=10)
axes[0].set_title("Seat Availability vs Price\n(lower availability = higher price)", fontweight="bold")
axes[0].set_xlabel("Seat Availability (0=full, 1=empty)")
axes[0].set_ylabel("Price (₹)")

axes[1].scatter(s["Fuel_Price_Index"], s["Price"], alpha=0.2, color=PALETTE[5], s=10)
axes[1].set_title("Fuel Price Index vs Ticket Price", fontweight="bold")
axes[1].set_xlabel("Fuel Price Index (USD/barrel equiv.)")
axes[1].set_ylabel("Price (₹)")
plt.tight_layout()
savefig("10_availability_fuel.png")

print("  ✓  EDA complete — 10 plots saved to pipeline_plots/")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 3] Feature engineering...")

df_fe = df.copy()

# ── Label encode categoricals ─────────────────────────────────────────────────
ENCODERS = {}
cat_cols = ["Airline","Source","Destination","Class","Total_Stops",
            "Season","Aircraft_Type","Airline_Tier"]
for col in cat_cols:
    le = LabelEncoder()
    df_fe[col+"_Enc"] = le.fit_transform(df_fe[col].astype(str))
    ENCODERS[col] = le
    print(f"    Encoded {col} → {col}_Enc  ({le.classes_.tolist()[:5]}...)")

# ── Interaction features ───────────────────────────────────────────────────────
print("  Creating interaction features...")
df_fe["Class_Dist_Interact"]    = df_fe["Class_Enc"] * df_fe["Distance_km"]
df_fe["Season_BookWin_Interact"]= df_fe["Season_Enc"] * df_fe["Days_Until_Departure"]
df_fe["Tier_SAF_Interact"]      = df_fe["Airline_Tier_Enc"] * df_fe["SAF_Zone"]
df_fe["Stops_Dist_Interact"]    = df_fe["Total_Stops_Enc"] * df_fe["Distance_km"]
df_fe["Env_Fleet_Interact"]     = df_fe["Env_Surcharge_Tier"] * df_fe["Fleet_Age_Years"]

# ── Derived numeric features ───────────────────────────────────────────────────
print("  Creating derived numeric features...")
df_fe["Log_Distance"]           = np.log1p(df_fe["Distance_km"])
df_fe["Log_Days_Until"]         = np.log1p(df_fe["Days_Until_Departure"])
df_fe["Price_Per_KM"]           = df_fe["Price"] / df_fe["Distance_km"].clip(1)  # for analysis only
df_fe["Is_Long_Haul"]           = (df_fe["Distance_km"] > 5000).astype(int)
df_fe["Is_Last_Minute"]         = (df_fe["Days_Until_Departure"] < 7).astype(int)
df_fe["Is_Advance_Booking"]     = (df_fe["Days_Until_Departure"] > 90).astype(int)

# ── Final feature set for model ────────────────────────────────────────────────
FEATURES = [
    # Core
    "Airline_Enc","Source_Enc","Destination_Enc","Class_Enc","Total_Stops_Enc",
    "Distance_km","Log_Distance","Days_Until_Departure","Log_Days_Until",
    # Date
    "Journey_Month","Journey_Day","Journey_Weekday","Is_Weekend",
    # Season
    "Season_Enc",
    # BRD macro-factors (Phase 2)
    "SAF_Zone","Env_Surcharge_Tier","Fleet_Age_Years","Is_Restricted_Airspace",
    # Extra real-world
    "Geo_Risk_Score","Fuel_Price_Index","Seat_Availability","Layover_Hours",
    "Aircraft_Type_Enc","Airline_Tier_Enc",
    # Interactions
    "Class_Dist_Interact","Season_BookWin_Interact","Tier_SAF_Interact",
    "Stops_Dist_Interact","Env_Fleet_Interact",
    # Flags
    "Is_Long_Haul","Is_Last_Minute","Is_Advance_Booking",
]
TARGET = "Price"

print(f"  Total features selected : {len(FEATURES)}")

X = df_fe[FEATURES]
y = df_fe[TARGET]
print(f"  X shape : {X.shape}")
print(f"  y shape : {y.shape}")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — TIME-SERIES CROSS-VALIDATION SPLIT  (BRD requirement)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 4] Time-Series Cross-Validation split...")

# Data is sorted by Journey_Date — use last 15% as final held-out test set
split_idx     = int(len(X) * 0.85)
X_train_full  = X.iloc[:split_idx]
y_train_full  = y.iloc[:split_idx]
X_test        = X.iloc[split_idx:]
y_test        = y.iloc[split_idx:]

print(f"  Training set : {len(X_train_full):,} rows  ({split_idx/len(X)*100:.1f}%)")
print(f"  Test set     : {len(X_test):,} rows  ({len(X_test)/len(X)*100:.1f}%)")

tscv = TimeSeriesSplit(n_splits=5)
print(f"  TimeSeriesSplit : 5 folds on training set")

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — BASELINE MODEL (30-day moving average) — BRD requirement
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 5] Baseline model — 30-day moving average...")

df_dated = df_fe.copy()
df_dated["Price_MA30"] = df_dated["Price"].rolling(window=30, min_periods=1).mean().shift(1).fillna(df_dated["Price"].mean())

baseline_preds = df_dated["Price_MA30"].iloc[split_idx:].values
baseline_mape  = mape(y_test.values, baseline_preds)
baseline_mae   = mean_absolute_error(y_test, baseline_preds)
baseline_r2    = r2_score(y_test, baseline_preds)

print(f"  Baseline MAPE : {baseline_mape:.2f}%")
print(f"  Baseline MAE  : ₹{baseline_mae:,.0f}")
print(f"  Baseline R²   : {baseline_r2:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — MODEL TRAINING & COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 6] Training and comparing models...")

results    = {}
best_model = None
best_mape  = float("inf")
best_name  = ""

def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    global best_model, best_mape, best_name
    t0 = time.time()
    model.fit(X_tr, y_tr)
    preds   = model.predict(X_te)
    elapsed = time.time() - t0

    m  = mape(y_te.values, preds)
    ma = mean_absolute_error(y_te, preds)
    r2 = r2_score(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))

    results[name] = {"MAPE":round(m,2),"MAE":round(ma,0),"R2":round(r2,4),"RMSE":round(rmse,0),"preds":preds}
    print(f"    {name:<30} MAPE={m:.2f}%  MAE=₹{ma:,.0f}  R²={r2:.4f}  [{elapsed:.1f}s]")

    if m < best_mape:
        best_mape  = m
        best_model = model
        best_name  = name

# Linear regression
print("  Training models...")
evaluate("Linear Regression",   LinearRegression(), X_train_full, y_train_full, X_test, y_test)
evaluate("Ridge Regression",    Ridge(alpha=1.0),   X_train_full, y_train_full, X_test, y_test)
evaluate("Random Forest",
         RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_leaf=4,
                               n_jobs=-1, random_state=42),
         X_train_full, y_train_full, X_test, y_test)
evaluate("Gradient Boosting",
         GradientBoostingRegressor(n_estimators=300, learning_rate=0.08,
                                   max_depth=6, subsample=0.85, random_state=42),
         X_train_full, y_train_full, X_test, y_test)

if HAS_XGB:
    evaluate("XGBoost",
             xgb.XGBRegressor(n_estimators=400, learning_rate=0.07, max_depth=7,
                               subsample=0.85, colsample_bytree=0.85,
                               n_jobs=-1, random_state=42, verbosity=0),
             X_train_full, y_train_full, X_test, y_test)
if HAS_LGB:
    evaluate("LightGBM",
             lgb.LGBMRegressor(n_estimators=400, learning_rate=0.07, max_depth=7,
                                num_leaves=63, subsample=0.85, colsample_bytree=0.85,
                                n_jobs=-1, random_state=42, verbose=-1),
             X_train_full, y_train_full, X_test, y_test)

print(f"\n  ✓  Best model : {best_name}  (MAPE={best_mape:.2f}%)")
print(f"     vs Baseline : {baseline_mape:.2f}% MAPE  →  improvement = {baseline_mape-best_mape:.2f}pp")

# ── Plot model comparison ────────────────────────────────────────────────────
print("  Plotting model comparison...")
comp_df = pd.DataFrame([{"Model":k,"MAPE":v["MAPE"],"R2":v["R2"],"MAE":v["MAE"]}
                         for k,v in results.items()])
comp_df.loc[len(comp_df)] = {"Model":"Baseline (MA30)","MAPE":round(baseline_mape,2),"R2":round(baseline_r2,4),"MAE":round(baseline_mae,0)}
comp_df = comp_df.sort_values("MAPE")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = [PALETTE[0] if m != "Baseline (MA30)" else "#888" for m in comp_df["Model"]]
axes[0].barh(comp_df["Model"], comp_df["MAPE"], color=colors, edgecolor="white")
axes[0].set_title("MAPE (lower = better)", fontweight="bold")
axes[0].set_xlabel("MAPE (%)")
axes[1].barh(comp_df["Model"], comp_df["R2"], color=colors, edgecolor="white")
axes[1].set_title("R² Score (higher = better)", fontweight="bold")
axes[1].set_xlabel("R²")
axes[2].barh(comp_df["Model"], comp_df["MAE"], color=colors, edgecolor="white")
axes[2].set_title("Mean Absolute Error ₹ (lower = better)", fontweight="bold")
axes[2].set_xlabel("MAE (₹)")
plt.suptitle("Model Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
savefig("11_model_comparison.png")

# ── Actual vs Predicted plot for best model ──────────────────────────────────
best_preds = results[best_name]["preds"]
fig, axes  = plt.subplots(1, 2, figsize=(14, 5))
sample_n   = min(3000, len(y_test))
idx        = np.random.choice(len(y_test), sample_n, replace=False)
axes[0].scatter(y_test.values[idx], best_preds[idx], alpha=0.3, color=PALETTE[0], s=8)
mn, mx = y_test.min(), y_test.max()
axes[0].plot([mn,mx],[mn,mx],"r--",linewidth=1.5,label="Perfect prediction")
axes[0].set_title(f"Actual vs Predicted — {best_name}", fontweight="bold")
axes[0].set_xlabel("Actual Price (₹)")
axes[0].set_ylabel("Predicted Price (₹)")
axes[0].legend()

residuals = y_test.values - best_preds
axes[1].scatter(best_preds[idx], residuals[idx], alpha=0.3, color=PALETTE[2], s=8)
axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
axes[1].set_title("Residuals Plot", fontweight="bold")
axes[1].set_xlabel("Predicted Price (₹)")
axes[1].set_ylabel("Residual (₹)")
plt.tight_layout()
savefig("12_actual_vs_predicted.png")

# ── Time-series CV scores ────────────────────────────────────────────────────
print("  Running TimeSeriesSplit CV on best model...")
cv_scores = []
for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train_full)):
    Xtr, Xval = X_train_full.iloc[tr_idx], X_train_full.iloc[val_idx]
    ytr, yval = y_train_full.iloc[tr_idx], y_train_full.iloc[val_idx]
    best_model.fit(Xtr, ytr)
    pval = best_model.predict(Xval)
    score = mape(yval.values, pval)
    cv_scores.append(score)
    print(f"    Fold {fold+1}/5 : MAPE = {score:.2f}%")

best_model.fit(X_train_full, y_train_full)   # refit on full training data
print(f"  CV MAPE mean ± std : {np.mean(cv_scores):.2f}% ± {np.std(cv_scores):.2f}%")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1,6), cv_scores, marker="o", color=PALETTE[0], linewidth=2, markersize=8, label="Fold MAPE")
ax.axhline(np.mean(cv_scores), color="red", linestyle="--", label=f"Mean {np.mean(cv_scores):.2f}%")
ax.set_title("TimeSeriesSplit CV — MAPE per Fold", fontweight="bold")
ax.set_xlabel("Fold")
ax.set_ylabel("MAPE (%)")
ax.legend()
plt.tight_layout()
savefig("13_timeseries_cv.png")

# ── Feature importance (tree models) ─────────────────────────────────────────
if hasattr(best_model, "feature_importances_"):
    print("  Plotting feature importance...")
    fi = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    top20 = fi.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top20.index[::-1], top20.values[::-1], color=PALETTE[0], edgecolor="white")
    ax.set_title(f"Top 20 Feature Importances — {best_name}", fontweight="bold")
    ax.set_xlabel("Importance")
    for i, v in enumerate(top20.values[::-1]):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    savefig("14_feature_importance.png")
    print("  Top 10 features:")
    for feat, imp in fi.head(10).items():
        print(f"    {feat:<35} {imp:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — SHAP EXPLAINABILITY  (BRD Phase 4)
# ══════════════════════════════════════════════════════════════════════════════
if HAS_SHAP:
    print("\n[STEP 7] SHAP explainability (BRD Phase 4)...")
    shap_sample = X_test.sample(min(1000, len(X_test)), random_state=42)

    try:
        explainer   = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(shap_sample)

        # Summary bar plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, shap_sample, plot_type="bar",
                          feature_names=FEATURES, show=False, max_display=20)
        plt.title("SHAP Feature Importance (mean |SHAP value|)", fontweight="bold")
        plt.tight_layout()
        savefig("15_shap_importance.png")

        # Summary beeswarm plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, shap_sample, feature_names=FEATURES, show=False, max_display=15)
        plt.title("SHAP Beeswarm — Feature Impact Direction", fontweight="bold")
        plt.tight_layout()
        savefig("16_shap_beeswarm.png")

        # Verify BRD macro-factors are in top 10
        shap_imp = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES).sort_values(ascending=False)
        brd_feats = ["SAF_Zone","Env_Surcharge_Tier","Fleet_Age_Years","Is_Restricted_Airspace"]
        top10_feats = shap_imp.head(10).index.tolist()
        print(f"\n  SHAP Top 10 features:")
        for i, (feat, val) in enumerate(shap_imp.head(10).items()):
            brd_flag = "  ← BRD macro-factor ✓" if feat in brd_feats else ""
            print(f"    {i+1:>2}. {feat:<35} {val:>8.2f}{brd_flag}")

        brd_in_top10 = [f for f in brd_feats if f in top10_feats]
        print(f"\n  BRD macro-factors in top 10: {len(brd_in_top10)}/4")
        if len(brd_in_top10) >= 2:
            print("  ✓  SHAP requirement MET — BRD macro-factors detectable by model")
        else:
            print("  ⚠  Some BRD features have low SHAP impact — check data generation")

        print("\n  ✓  SHAP complete — plots saved")
    except Exception as e:
        print(f"  ⚠  SHAP failed: {e}")
else:
    print("\n[STEP 7] Skipping SHAP (not installed). Run: pip install shap")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8 — SAVE MODEL ARTEFACTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 8] Saving model artefacts...")

joblib.dump(best_model,  os.path.join(BASE_DIR, "airfair_model.pkl"))
joblib.dump(ENCODERS,    os.path.join(BASE_DIR, "airfair_encoders.pkl"))
joblib.dump(FEATURES,    os.path.join(BASE_DIR, "airfair_features.pkl"))

model_meta = {
    "model_name"   : best_name,
    "mape"         : round(best_mape, 2),
    "r2"           : round(results[best_name]["R2"], 4),
    "mae"          : int(results[best_name]["MAE"]),
    "baseline_mape": round(baseline_mape, 2),
    "cv_mape_mean" : round(float(np.mean(cv_scores)), 2),
    "cv_mape_std"  : round(float(np.std(cv_scores)), 2),
    "n_features"   : len(FEATURES),
    "train_rows"   : len(X_train_full),
    "test_rows"    : len(X_test),
    "features"     : FEATURES,
}
pd.Series(model_meta).to_json(os.path.join(BASE_DIR, "model_meta.json"))

print(f"  ✓  airfair_model.pkl    — {best_name}")
print(f"  ✓  airfair_encoders.pkl — {len(ENCODERS)} label encoders")
print(f"  ✓  airfair_features.pkl — {len(FEATURES)} features")
print(f"  ✓  model_meta.json      — metrics & config")

# ══════════════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  PIPELINE COMPLETE — SUMMARY")
print("═"*65)
print(f"  Dataset rows              : {len(df):,}")
print(f"  Features used             : {len(FEATURES)}")
print(f"  Best model                : {best_name}")
print(f"  Test MAPE                 : {best_mape:.2f}%")
print(f"  Test R²                   : {results[best_name]['R2']:.4f}")
print(f"  Test MAE                  : ₹{results[best_name]['MAE']:,.0f}")
print(f"  Baseline MAPE (MA-30)     : {baseline_mape:.2f}%")
print(f"  Improvement vs baseline   : {baseline_mape - best_mape:.2f}pp")
print(f"  CV MAPE (5-fold TS)       : {np.mean(cv_scores):.2f}% ± {np.std(cv_scores):.2f}%")
print(f"  Plots saved to            : pipeline_plots/  ({len(os.listdir(PLOTS_DIR))} files)")
print("═"*65)
print("\n  ✅  Ready for Streamlit app & SHAP report!\n")
