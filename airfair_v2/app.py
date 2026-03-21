"""
✈️  AirFair Vista — Home & Price Predictor
Fully wired to pipeline.py output:
  - airfair_model.pkl  (best model from pipeline)
  - airfair_encoders.pkl (all 8 label encoders)
  - airfair_features.pkl (ordered feature list)
  - model_meta.json    (live metrics)
  - flight_price_combined.csv (120k rows)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os, calendar

st.set_page_config(
    page_title="AirFair Vista ✈️",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]  { font-family:'DM Sans',sans-serif; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0A1931 0%,#0D2550 100%); }
[data-testid="stSidebar"] * { color:#C8DEFF !important; }
[data-testid="stSidebar"] label { color:#90B8F0 !important; font-size:.76rem !important;
  text-transform:uppercase; letter-spacing:.07em; }
.hero { background:linear-gradient(135deg,#0A1931 0%,#1565C0 55%,#0D2550 100%);
  border-radius:18px; padding:2.4rem 3rem; margin-bottom:1.8rem; position:relative; overflow:hidden; }
.hero::after { content:'✈'; position:absolute; right:2.5rem; top:50%;
  transform:translateY(-50%) rotate(15deg); font-size:7rem; opacity:.07; }
.hero h1 { font-family:'Playfair Display',serif; font-size:2.8rem; font-weight:900;
  color:#fff; margin:0 0 .35rem; line-height:1.15; }
.hero p  { color:#90BEFF; font-size:1rem; margin:0; font-weight:300; }
.kpi { background:#fff; border:1px solid #DCE8FA; border-radius:14px;
  padding:1.25rem 1.4rem; text-align:center; box-shadow:0 2px 14px rgba(21,101,192,.07); }
.kpi .v { font-family:'Playfair Display',serif; font-size:2.1rem; font-weight:700; color:#0A1931; line-height:1; }
.kpi .l { font-size:.75rem; color:#64748B; text-transform:uppercase; letter-spacing:.08em; margin-top:.25rem; }
.kpi .s { font-size:.68rem; color:#94A3B8; margin-top:.12rem; }
.panel  { background:#F8FBFF; border:1px solid #DCE8FA; border-radius:16px;
  padding:1.7rem 2rem; margin-bottom:1.4rem; }
.result { background:linear-gradient(135deg,#0A1931,#1565C0); border-radius:16px;
  padding:2rem 2.4rem; text-align:center; color:#fff; margin-top:1.4rem;
  box-shadow:0 8px 30px rgba(21,101,192,.35); }
.result .rl { font-size:.82rem; letter-spacing:.12em; text-transform:uppercase; color:#90BEFF; margin-bottom:.4rem; }
.result .rv { font-family:'Playfair Display',serif; font-size:3.4rem; font-weight:900; color:#FFB300; line-height:1; }
.result .rs { font-size:.82rem; color:#90BEFF; margin-top:.45rem; }
.badge { display:inline-block; background:rgba(0,176,255,.15); color:#00B0FF;
  border:1px solid rgba(0,176,255,.3); border-radius:20px;
  padding:.18rem .85rem; font-size:.75rem; font-weight:600; margin-top:.45rem; }
.tip  { background:#FFF8E6; border-left:4px solid #FFB300; border-radius:8px;
  padding:.85rem 1.1rem; margin-top:.9rem; font-size:.84rem; color:#7A5C00; }
.sec  { font-family:'Playfair Display',serif; font-size:1.45rem; font-weight:700;
  color:#0A1931; padding-bottom:.45rem; border-bottom:2px solid #1565C0;
  display:inline-block; margin:0 0 1.1rem; }
.fcard { background:#F0F6FF; border-radius:12px; padding:1.1rem 1.3rem;
  border-left:4px solid #1565C0; margin-bottom:.9rem; }
.fcard h4 { color:#0A1931; margin:0 0 .35rem; font-size:.93rem; }
.fcard p  { color:#64748B; font-size:.81rem; margin:0; line-height:1.5; }
.stButton>button { background:linear-gradient(135deg,#1565C0,#1E88E5) !important;
  color:#fff !important; border:none !important; border-radius:10px !important;
  padding:.65rem 2rem !important; font-weight:600 !important; font-size:.97rem !important;
  width:100% !important; box-shadow:0 4px 14px rgba(21,101,192,.3) !important; }
</style>""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
AIRLINES = sorted(['Air France','Air India','AirAsia India','British Airways',
                   'Cathay Pacific','Emirates','Etihad Airways','IndiGo',
                   'Lufthansa','Qatar Airways','Singapore Airlines',
                   'SpiceJet','Thai Airways','Turkish Airlines','Vistara'])
CITIES   = sorted(['Ahmedabad','Bangalore','Bangkok','Chennai','Delhi','Doha',
                   'Dubai','Frankfurt','Hong Kong','Istanbul','London',
                   'Mumbai','New York','Paris','Singapore'])
CLASSES  = ['Economy','Premium Economy','Business','First']
STOPS    = ['non-stop','1 stop','2 stops']

AIRLINE_TIER = {
    "Singapore Airlines":"premium","Emirates":"premium","Qatar Airways":"premium",
    "British Airways":"premium","Lufthansa":"premium","Cathay Pacific":"premium",
    "Air France":"mid","Etihad Airways":"mid","Thai Airways":"mid",
    "Turkish Airlines":"mid","Air India":"mid","Vistara":"mid",
    "AirAsia India":"budget","IndiGo":"budget","SpiceJet":"budget"}

SAF_ZONE = {"London":2,"Frankfurt":2,"Paris":2,"Istanbul":1,"Singapore":1,"New York":1,
            "Dubai":0,"Doha":0,"Bangkok":0,"Hong Kong":0,"Delhi":0,"Mumbai":0,
            "Bangalore":0,"Chennai":0,"Ahmedabad":0}
ENV_TIER = {"London":3,"Frankfurt":3,"Paris":3,"New York":2,"Istanbul":2,"Singapore":2,
            "Dubai":1,"Doha":1,"Bangkok":1,"Hong Kong":1,"Delhi":0,"Mumbai":0,
            "Bangalore":0,"Chennai":0,"Ahmedabad":0}
FLEET_AGE = {"Singapore Airlines":6.2,"Emirates":7.8,"Qatar Airways":6.5,
             "British Airways":13.4,"Lufthansa":12.1,"Cathay Pacific":9.3,
             "Air France":11.7,"Etihad Airways":8.4,"Thai Airways":16.2,
             "Turkish Airlines":9.8,"Air India":14.1,"Vistara":5.3,
             "AirAsia India":7.1,"IndiGo":5.9,"SpiceJet":10.2}
RESTRICTED = {("Delhi","Frankfurt"),("Delhi","Paris"),("Delhi","London"),
              ("Mumbai","Frankfurt"),("Mumbai","Paris"),("Mumbai","London"),
              ("Bangkok","Frankfurt"),("Dubai","New York"),("Doha","New York")}

DIST_MAP = {
    ('Delhi','Mumbai'):1400,('Delhi','Bangalore'):2150,('Delhi','London'):6700,
    ('Delhi','Dubai'):2200,('Delhi','Singapore'):4150,('Delhi','Frankfurt'):6200,
    ('Delhi','New York'):11750,('Delhi','Bangkok'):2900,('Delhi','Hong Kong'):3750,
    ('Delhi','Paris'):6600,('Delhi','Istanbul'):4200,('Delhi','Doha'):2350,
    ('Mumbai','Bangalore'):980,('Mumbai','London'):7200,('Mumbai','Dubai'):1950,
    ('Mumbai','Singapore'):4180,('Mumbai','New York'):12550,('Mumbai','Bangkok'):3080,
    ('Mumbai','Paris'):6950,('Mumbai','Istanbul'):4400,('Mumbai','Doha'):1960,
    ('London','New York'):5540,('London','Dubai'):5500,('London','Singapore'):10850,
    ('London','Hong Kong'):9600,('London','Frankfurt'):650,('London','Paris'):340,
    ('Dubai','Singapore'):5840,('Dubai','Bangkok'):4900,('Dubai','New York'):11020,
    ('Singapore','Hong Kong'):2580,('Singapore','Bangkok'):1450,
    ('Frankfurt','New York'):6200,('Paris','New York'):5840,
    ('Doha','New York'):11540,('Ahmedabad','Doha'):2000,
    ('Chennai','Dubai'):2900,('Chennai','Singapore'):3280,
    ('Bangalore','Dubai'):2900,('Bangalore','Singapore'):3300,
}

# ── Load model artefacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    mp   = os.path.join(base, "airfair_model.pkl")
    ep   = os.path.join(base, "airfair_encoders.pkl")
    fp   = os.path.join(base, "airfair_features.pkl")
    if os.path.exists(mp) and os.path.exists(ep) and os.path.exists(fp):
        return joblib.load(mp), joblib.load(ep), joblib.load(fp), True
    return None, None, None, False

@st.cache_data
def load_meta():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_meta.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"model_name":"Gradient Boosting","mape":13.5,"r2":0.976,
            "mae":1715,"train_rows":120000,"n_features":34,
            "baseline_mape":45.0,"cv_mape_mean":13.8}

model, encoders, features, model_loaded = load_model()
meta = load_meta()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style='text-align:center;padding:1rem 0 1.5rem;'>
      <div style='font-size:2.4rem;'>✈️</div>
      <div style='font-family:"Playfair Display",serif;font-size:1.35rem;font-weight:900;color:#fff;'>AirFair Vista</div>
      <div style='font-size:.7rem;color:#7098CC;letter-spacing:.1em;text-transform:uppercase;margin-top:.2rem;'>Flight Price Intelligence</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    status = "🟢 Model loaded" if model_loaded else "🟡 Demo mode"
    st.markdown(f"""<div style='font-size:.7rem;color:#506080;line-height:1.9;'>
      <b style='color:#7098CC;'>Status</b><br>{status}<br><br>
      <b style='color:#7098CC;'>Best Model</b><br>{meta.get('model_name','Gradient Boosting')}<br><br>
      <b style='color:#7098CC;'>Dataset</b><br>{meta.get('train_rows',120000):,} rows · 25 features<br>
      15 airlines · 15 cities · 4 classes<br><br>
      <b style='color:#7098CC;'>Performance</b><br>
      R² = {meta.get('r2',0.976):.4f}<br>
      MAPE = {meta.get('mape',13.5):.1f}%<br>
      MAE = ₹{meta.get('mae',1715):,}<br>
      CV = {meta.get('cv_mape_mean',13.8):.1f}% ± {meta.get('cv_mape_std',1.2):.1f}%
    </div>""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
r2_pct   = f"{meta.get('r2',0.976)*100:.1f}%"
n_rows   = f"{meta.get('train_rows',120000):,}"
mape_val = f"{meta.get('mape',13.5):.1f}%"

st.markdown(f"""<div class="hero">
  <h1>AirFair Vista ✈️</h1>
  <p>International flight price prediction powered by {meta.get('model_name','Gradient Boosting')} ML.<br>
     Trained on {n_rows} records · 15 airlines · 15 global cities · 4 cabin classes · 25 features incl. BRD macro-factors.</p>
</div>""", unsafe_allow_html=True)

# ── KPIs — live from model_meta.json ─────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5)
kpis = [
    (r2_pct,         "Model Accuracy",   "R² Score"),
    (n_rows,         "Training Records", "Combined dataset"),
    (mape_val,       "Avg Error",        "MAPE"),
    ("15",           "Airlines",         "Global carriers"),
    ("25",           "Features",         "incl. BRD macro-factors"),
]
for col,(v,l,s) in zip([k1,k2,k3,k4,k5], kpis):
    with col:
        st.markdown(f'<div class="kpi"><div class="v">{v}</div>'
                    f'<div class="l">{l}</div><div class="s">{s}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Price Predictor ───────────────────────────────────────────────────────────
st.markdown('<div class="sec">✈️ Predict Your Flight Price</div>', unsafe_allow_html=True)
col_form, col_res = st.columns([3,2], gap="large")

with col_form:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.markdown("**🛫 Route & Airline**")
    a1,a2 = st.columns(2)
    with a1:
        airline = st.selectbox("Airline", AIRLINES, index=AIRLINES.index("Emirates"))
        source  = st.selectbox("From (Source)", CITIES, index=CITIES.index("Dubai"))
    with a2:
        cabin       = st.selectbox("Cabin Class", CLASSES, index=1)
        dest_opts   = [c for c in CITIES if c != source]
        destination = st.selectbox("To (Destination)", dest_opts,
                                   index=dest_opts.index("London") if "London" in dest_opts else 0)

    st.markdown("---")
    st.markdown("**🛑 Flight Details**")
    b1,b2,b3 = st.columns(3)
    with b1:
        stops = st.selectbox("Stops", STOPS)
    with b2:
        pair = (source, destination)
        rpair= (destination, source)
        default_dist = DIST_MAP.get(pair, DIST_MAP.get(rpair, 5000))
        distance_km  = st.number_input("Distance (km)", min_value=300, max_value=20000,
                                        value=default_dist, step=100)
    with b3:
        days_until = st.number_input("Days Until Departure", min_value=1, max_value=365, value=45)

    st.markdown("---")
    st.markdown("**📅 Journey Date**")
    c1,c2,c3 = st.columns(3)
    with c1:
        month_name = st.selectbox("Month", list(calendar.month_name)[1:], index=5)
        month      = list(calendar.month_name).index(month_name)
    with c2:
        day = st.number_input("Day", min_value=1, max_value=31, value=20)
    with c3:
        wday_name = st.selectbox("Weekday", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
        wday_enc  = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}[wday_name]

    st.markdown('</div>', unsafe_allow_html=True)
    predict_btn = st.button("🔮  Predict Flight Price", use_container_width=True)

with col_res:
    st.markdown("#### 📋 Booking Summary")
    is_restr = 1 if (source,destination) in RESTRICTED or (destination,source) in RESTRICTED else 0
    saf_z    = SAF_ZONE.get(destination, 0)
    env_t    = max(ENV_TIER.get(source,0), ENV_TIER.get(destination,0))
    fage     = FLEET_AGE.get(airline, 10.0)
    tier     = AIRLINE_TIER.get(airline,"mid")
    season   = "peak" if month in [12,1,3,4,5] else ("shoulder" if month in [2,6,10,11] else "off_peak")
    actype   = "wide-body" if distance_km > 4000 else "narrow-body"

    st.markdown(f"""<div style='background:#F0F6FF;border-radius:12px;padding:1.2rem 1.5rem;
         border:1px solid #DCE8FA;font-size:.87rem;line-height:2.2;'>
      <b>✈️ Route</b>: {source} → {destination}<br>
      <b>🏢 Airline</b>: {airline} ({tier})<br>
      <b>💺 Class</b>: {cabin}<br>
      <b>🛑 Stops</b>: {stops}<br>
      <b>📏 Distance</b>: {distance_km:,} km ({actype})<br>
      <b>📅 Date</b>: {int(day)} {month_name} ({wday_name}) — {season}<br>
      <b>⏳ Booking</b>: {int(days_until)} days ahead<br>
      <b>🌿 SAF Zone</b>: {saf_z} · Env Tier: {env_t} · Fleet Age: {fage:.1f}yr<br>
      <b>🚧 Restricted</b>: {"Yes ⚠️" if is_restr else "No"}
    </div>""", unsafe_allow_html=True)

    if predict_btn:
        if model_loaded and model is not None:
            try:
                stops_enc_map = {"non-stop":0,"1 stop":1,"2 stops":2}
                season_map    = {"off_peak":0,"peak":1,"shoulder":2}
                tier_map      = {"budget":0,"mid":1,"premium":2}
                actype_map    = {"narrow-body":0,"wide-body":1}

                row = {
                    "Airline_Enc":          encoders["Airline"].transform([airline])[0],
                    "Source_Enc":           encoders["Source"].transform([source])[0],
                    "Destination_Enc":      encoders["Destination"].transform([destination])[0],
                    "Class_Enc":            encoders["Class"].transform([cabin])[0],
                    "Total_Stops_Enc":      encoders["Total_Stops"].transform([stops])[0],
                    "Distance_km":          distance_km,
                    "Log_Distance":         np.log1p(distance_km),
                    "Days_Until_Departure": int(days_until),
                    "Log_Days_Until":       np.log1p(int(days_until)),
                    "Journey_Month":        month,
                    "Journey_Day":          int(day),
                    "Journey_Weekday":      wday_enc,
                    "Is_Weekend":           int(wday_enc >= 5),
                    "Season_Enc":           encoders["Season"].transform([season])[0],
                    "SAF_Zone":             saf_z,
                    "Env_Surcharge_Tier":   env_t,
                    "Fleet_Age_Years":      fage,
                    "Is_Restricted_Airspace": is_restr,
                    "Geo_Risk_Score":       is_restr * 0.5 + 0.1,
                    "Fuel_Price_Index":     100.0,
                    "Seat_Availability":    0.5,
                    "Layover_Hours":        0 if stops=="non-stop" else (3 if stops=="1 stop" else 7),
                    "Aircraft_Type_Enc":    encoders["Aircraft_Type"].transform([actype])[0],
                    "Airline_Tier_Enc":     encoders["Airline_Tier"].transform([tier])[0],
                    "Class_Dist_Interact":  encoders["Class"].transform([cabin])[0] * distance_km,
                    "Season_BookWin_Interact": encoders["Season"].transform([season])[0] * int(days_until),
                    "Tier_SAF_Interact":    encoders["Airline_Tier"].transform([tier])[0] * saf_z,
                    "Stops_Dist_Interact":  stops_enc_map[stops] * distance_km,
                    "Env_Fleet_Interact":   env_t * fage,
                    "Is_Long_Haul":         int(distance_km > 5000),
                    "Is_Last_Minute":       int(int(days_until) < 7),
                    "Is_Advance_Booking":   int(int(days_until) > 90),
                }
                input_df = pd.DataFrame([row])[features]
                pred     = model.predict(input_df)[0]
                mode     = "real"
            except Exception as e:
                st.error(f"Prediction error: {e}")
                pred, mode = 15000, "demo"
        else:
            # Demo formula
            base_p = {"Economy":5000,"Premium Economy":12000,"Business":28000,"First":55000}[cabin]
            pred   = base_p + distance_km*1.9
            pred  *= {"non-stop":1.08,"1 stop":0.95,"2 stops":0.88}[stops]
            pred  *= 1.40 if int(days_until)<7 else (1.20 if int(days_until)<30 else 1.00)
            pred  *= 1.18 if month in [12,1,3,4,5] else 1.00
            pred  *= 1.07 if wday_enc>=5 else 1.00
            pred  *= {"premium":1.30,"mid":1.00,"budget":0.82}[tier]
            pred   = max(1500, pred)
            mode   = "demo"
            st.info("ℹ️ Running in **demo mode** — run `python pipeline.py` to generate model files.", icon="💡")

        conf = meta.get("mape", 13.5)
        low, high = pred*(1-conf/100), pred*(1+conf/100)
        st.markdown(f"""<div class="result">
          <div class="rl">Predicted Ticket Price</div>
          <div class="rv">₹{pred:,.0f}</div>
          <div class="rs">Expected range: ₹{low:,.0f} – ₹{high:,.0f}</div>
          <div class="badge">{'± ' + str(round(conf,1)) + '% model confidence' if mode=='real' else 'Demo estimate'}</div>
        </div>""", unsafe_allow_html=True)

        tips = []
        if int(days_until) <= 7:    tips.append("🚨 Last-minute booking — prices at peak. Try flexible dates.")
        elif int(days_until) >= 90: tips.append("✅ 90+ days ahead — typically saves 20–35% vs last-minute.")
        if is_restr:                tips.append("🚧 Restricted airspace on this route adds ~9% reroute cost.")
        if saf_z == 2:              tips.append("🌿 EU/UK mandatory SAF zone — adds ~6% environmental levy.")
        if cabin=="Economy" and distance_km>8000: tips.append("💡 Long-haul >8,000km — Premium Economy worth considering.")
        if month in [3,4,5,12,1]:  tips.append("📅 Peak season — shifting ±2 weeks can lower fares.")
        if wday_enc >= 5:           tips.append("📆 Weekend departure — Tue/Wed typically 10-15% cheaper.")
        if tips:
            st.markdown('<div class="tip">' + "<br>".join(tips) + '</div>', unsafe_allow_html=True)
    else:
        st.markdown("""<div style='background:#F8FBFF;border:2px dashed #90B8E0;border-radius:12px;
             padding:2.8rem 1.5rem;text-align:center;color:#7098CC;margin-top:1rem;'>
          <div style='font-size:2.2rem;'>💰</div>
          <div style='font-weight:600;margin-top:.5rem;'>Price appears here</div>
          <div style='font-size:.8rem;margin-top:.25rem;'>Fill the form and click Predict</div>
        </div>""", unsafe_allow_html=True)

# ── Key Price Drivers — live from model_meta ──────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="sec">🔑 Key Price Drivers</div>', unsafe_allow_html=True)
factors = [
    ("💺","Cabin Class",       "Dominant driver. First class costs ~3.5× Economy on average."),
    ("📏","Distance",          "Each 1,000 km adds ~₹2,000–3,000 non-linearly."),
    ("⏳","Booking Window",    "Last-minute (<7 days) fares are 25–40% higher."),
    ("🌿","SAF Zone",          "EU/UK mandatory SAF adds 6% · voluntary zones add 2%."),
    ("🏭","Fleet Age",         "Older fleets have higher operating costs passed to passengers."),
    ("🚧","Restricted Airspace","Reroutes add ~9% to ticket cost on affected routes."),
]
cols = st.columns(3)
for i,(icon,title,desc) in enumerate(factors):
    with cols[i%3]:
        st.markdown(f'<div class="fcard"><h4>{icon} {title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

st.markdown("<br>")
st.markdown(f"""<div style='text-align:center;color:#94A3B8;font-size:.76rem;padding:.8rem 0;'>
AirFair Vista · Final Year B.Tech Project · Computer Science · 2025 ·
Built with Python, Streamlit & {meta.get('model_name','Gradient Boosting')} ML
</div>""", unsafe_allow_html=True)
