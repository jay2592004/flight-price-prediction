"""
src/frontend/pages/home.py  —  Price Predictor page
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, sys, calendar

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.frontend.pages.loader import load_artefacts
from src.pipeline.features import build_single_row
from src.data.generator import (
    AIRLINES, CITIES, CLASSES, STOPS,
    AIRLINE_TIER, SAF_ZONE, ENV_TIER, FLEET_AGE, RESTRICTED,
)

DIST_MAP = {
    ("Delhi","Mumbai"):1400,("Delhi","Bangalore"):2150,("Delhi","London"):6700,
    ("Delhi","Dubai"):2200,("Delhi","Singapore"):4150,("Delhi","Frankfurt"):6200,
    ("Delhi","New York"):11750,("Delhi","Bangkok"):2900,("Delhi","Paris"):6600,
    ("Mumbai","Bangalore"):980,("Mumbai","London"):7200,("Mumbai","Dubai"):1950,
    ("Mumbai","Singapore"):4180,("Mumbai","New York"):12550,("Mumbai","Paris"):6950,
    ("London","New York"):5540,("London","Dubai"):5500,("London","Singapore"):10850,
    ("Dubai","Singapore"):5840,("Dubai","New York"):11020,
    ("Singapore","Hong Kong"):2580,("Singapore","Bangkok"):1450,
    ("Doha","New York"):11540,("Ahmedabad","Dubai"):1950,
    ("Chennai","Dubai"):2900,("Bangalore","Dubai"):2900,
}

CSS = """
<style>
.hero{background:linear-gradient(135deg,#0A1931 0%,#1565C0 55%,#0D2550 100%);
  border-radius:18px;padding:2.4rem 3rem;margin-bottom:1.8rem;position:relative;overflow:hidden;}
.hero::after{content:'✈';position:absolute;right:2.5rem;top:50%;
  transform:translateY(-50%) rotate(15deg);font-size:7rem;opacity:.07;}
.hero h1{font-family:'Playfair Display',serif;font-size:2.8rem;font-weight:900;
  color:#fff;margin:0 0 .35rem;line-height:1.15;}
.hero p{color:#90BEFF;font-size:1rem;margin:0;font-weight:300;}
.kpi{background:#fff;border:1px solid #DCE8FA;border-radius:14px;padding:1.25rem 1.4rem;
  text-align:center;box-shadow:0 2px 14px rgba(21,101,192,.07);}
.kpi .v{font-family:'Playfair Display',serif;font-size:2.1rem;font-weight:700;color:#0A1931;line-height:1;}
.kpi .l{font-size:.75rem;color:#64748B;text-transform:uppercase;letter-spacing:.08em;margin-top:.25rem;}
.kpi .s{font-size:.68rem;color:#94A3B8;margin-top:.12rem;}
.panel{background:#F8FBFF;border:1px solid #DCE8FA;border-radius:16px;padding:1.7rem 2rem;margin-bottom:1.4rem;}
.result{background:linear-gradient(135deg,#0A1931,#1565C0);border-radius:16px;padding:2rem 2.4rem;
  text-align:center;color:#fff;margin-top:1.4rem;box-shadow:0 8px 30px rgba(21,101,192,.35);}
.result .rl{font-size:.82rem;letter-spacing:.12em;text-transform:uppercase;color:#90BEFF;margin-bottom:.4rem;}
.result .rv{font-family:'Playfair Display',serif;font-size:3.4rem;font-weight:900;color:#FFB300;line-height:1;}
.result .rs{font-size:.82rem;color:#90BEFF;margin-top:.45rem;}
.badge{display:inline-block;background:rgba(0,176,255,.15);color:#00B0FF;
  border:1px solid rgba(0,176,255,.3);border-radius:20px;
  padding:.18rem .85rem;font-size:.75rem;font-weight:600;margin-top:.45rem;}
.tip{background:#FFF8E6;border-left:4px solid #FFB300;border-radius:8px;
  padding:.85rem 1.1rem;margin-top:.9rem;font-size:.84rem;color:#7A5C00;}
.sec{font-family:'Playfair Display',serif;font-size:1.45rem;font-weight:700;
  color:#0A1931;padding-bottom:.45rem;border-bottom:2px solid #1565C0;
  display:inline-block;margin:0 0 1.1rem;}
.fcard{background:#F0F6FF;border-radius:12px;padding:1.1rem 1.3rem;
  border-left:4px solid #1565C0;margin-bottom:.9rem;}
.fcard h4{color:#0A1931;margin:0 0 .35rem;font-size:.93rem;}
.fcard p{color:#64748B;font-size:.81rem;margin:0;line-height:1.5;}
</style>
"""

def _demo_price(airline, cabin, stops, distance_km, days_until, month, wday_enc):
    """Fallback formula when model.pkl not present."""
    base  = {"Economy":5000,"Premium Economy":12000,"Business":28000,"First":55000}[cabin]
    price = base + distance_km * 1.9
    price *= {"non-stop":1.08,"1 stop":0.95,"2 stops":0.88}[stops]
    price *= 1.40 if days_until<7 else (1.20 if days_until<30 else 1.00)
    price *= 1.18 if month in [12,1,3,4,5] else 1.00
    price *= 1.07 if wday_enc>=5 else 1.00
    price *= {"premium":1.30,"mid":1.00,"budget":0.82}[AIRLINE_TIER.get(airline,"mid")]
    return max(1500, price)


def render():
    st.markdown(CSS, unsafe_allow_html=True)
    model, encoders, features, meta, loaded = load_artefacts()

    # Hero
    n_rows  = meta.get("total_rows", 120000)
    r2_pct  = f"{meta['r2']*100:.1f}%" if loaded else "—"
    mape_v  = f"{meta['mape']:.1f}%" if loaded else "—"
    model_n = meta.get("model_name","—")

    st.markdown(f"""<div class="hero">
      <h1>AirFair Vista ✈️</h1>
      <p>Flight price prediction powered by <b>{model_n}</b>.<br>
         {n_rows:,} training records · 15 airlines · 15 cities · 4 cabin classes · BRD macro-factors.</p>
    </div>""", unsafe_allow_html=True)

    # KPI cards
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(v,l,s) in zip([c1,c2,c3,c4,c5],[
        (r2_pct,        "Model Accuracy",   "R² Score"),
        (f"{n_rows:,}", "Training Records", "Combined dataset"),
        (mape_v,        "Avg Error",        "MAPE"),
        ("15",          "Airlines",         "Global carriers"),
        ("32",          "Features",         "incl. BRD macro-factors"),
    ]):
        with col:
            st.markdown(f'<div class="kpi"><div class="v">{v}</div>'
                        f'<div class="l">{l}</div><div class="s">{s}</div></div>',
                        unsafe_allow_html=True)

    if not loaded:
        st.warning("⚠️ Model not trained yet. Run `python train_model.py` then restart the app. "
                   "Predictions below use a demo formula.", icon="⚙️")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec">✈️ Predict Your Flight Price</div>', unsafe_allow_html=True)

    col_form, col_res = st.columns([3,2], gap="large")

    with col_form:
        st.markdown('<div class="panel">', unsafe_allow_html=True)

        st.markdown("**🛫 Route & Airline**")
        r1,r2 = st.columns(2)
        with r1:
            airline = st.selectbox("Airline", sorted(AIRLINES), index=sorted(AIRLINES).index("Emirates"))
            source  = st.selectbox("From", sorted(CITIES), index=sorted(CITIES).index("Dubai"))
        with r2:
            cabin     = st.selectbox("Cabin Class", CLASSES, index=1)
            dest_opts = [c for c in sorted(CITIES) if c != source]
            dest      = st.selectbox("To", dest_opts,
                                     index=dest_opts.index("London") if "London" in dest_opts else 0)

        st.markdown("---")
        st.markdown("**🛑 Flight Details**")
        d1,d2,d3 = st.columns(3)
        with d1:
            stops = st.selectbox("Stops", STOPS)
        with d2:
            pair  = (source,dest)
            default_dist = DIST_MAP.get(pair, DIST_MAP.get((dest,source), 5000))
            distance_km  = st.number_input("Distance (km)", 300, 20000, default_dist, 100)
        with d3:
            days_until = st.number_input("Days Until Departure", 1, 365, 45)

        st.markdown("---")
        st.markdown("**📅 Journey Date**")
        t1,t2,t3 = st.columns(3)
        with t1:
            month_name = st.selectbox("Month", list(calendar.month_name)[1:], index=5)
            month      = list(calendar.month_name).index(month_name)
        with t2:
            day = st.number_input("Day", 1, 31, 15)
        with t3:
            wday_name = st.selectbox("Weekday", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
            wday_enc  = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].index(wday_name)

        st.markdown('</div>', unsafe_allow_html=True)
        predict_btn = st.button("🔮  Predict Flight Price", use_container_width=True)

    # Derived BRD values
    tier   = AIRLINE_TIER.get(airline,"mid")
    saf_z  = SAF_ZONE.get(dest, 0)
    env_t  = max(ENV_TIER.get(source,0), ENV_TIER.get(dest,0))
    fage   = FLEET_AGE.get(airline, 10.0)
    restr  = 1 if (source,dest) in RESTRICTED or (dest,source) in RESTRICTED else 0
    season = "peak" if month in [12,1,3,4,5] else ("shoulder" if month in [2,6,10,11] else "off_peak")
    actype = "wide-body" if distance_km > 4000 else "narrow-body"
    lhours = 0 if stops=="non-stop" else (3 if stops=="1 stop" else 7)

    with col_res:
        st.markdown("#### 📋 Booking Summary")
        st.markdown(f"""<div style='background:#F0F6FF;border-radius:12px;padding:1.2rem 1.5rem;
             border:1px solid #DCE8FA;font-size:.87rem;line-height:2.2;'>
          <b>✈️ Route</b>: {source} → {dest}<br>
          <b>🏢 Airline</b>: {airline} <i>({tier})</i><br>
          <b>💺 Class</b>: {cabin}<br>
          <b>🛑 Stops</b>: {stops} · {actype}<br>
          <b>📏 Distance</b>: {distance_km:,} km<br>
          <b>📅 Date</b>: {int(day)} {month_name} ({wday_name}) — {season}<br>
          <b>⏳ Booking</b>: {int(days_until)} days ahead<br>
          <b>🌿 SAF</b>: Zone {saf_z} · Env Tier {env_t} · Fleet {fage:.1f}yr<br>
          <b>🚧 Airspace</b>: {"Restricted ⚠️" if restr else "Normal"}
        </div>""", unsafe_allow_html=True)

        if predict_btn:
            if loaded:
                try:
                    input_dict = {
                        "Airline":airline,"Source":source,"Destination":dest,
                        "Class":cabin,"Total_Stops":stops,
                        "Distance_km":distance_km,"Days_Until_Departure":int(days_until),
                        "Journey_Month":month,"Journey_Day":int(day),"Journey_Weekday":wday_enc,
                        "Season":season,"SAF_Zone":saf_z,"Env_Surcharge_Tier":env_t,
                        "Fleet_Age_Years":fage,"Is_Restricted_Airspace":restr,
                        "Aircraft_Type":actype,"Airline_Tier":tier,
                        "Geo_Risk_Score":restr*0.5+0.1,"Fuel_Price_Index":100.0,
                        "Seat_Availability":0.5,"Layover_Hours":lhours,
                    }
                    X_pred = build_single_row(input_dict, encoders)
                    pred   = float(model.predict(X_pred)[0])
                    mode   = "real"
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    pred, mode = _demo_price(airline,cabin,stops,distance_km,int(days_until),month,wday_enc), "demo"
            else:
                pred  = _demo_price(airline,cabin,stops,distance_km,int(days_until),month,wday_enc)
                mode  = "demo"

            conf = meta.get("mape",13.5)
            low, high = pred*(1-conf/100), pred*(1+conf/100)
            st.markdown(f"""<div class="result">
              <div class="rl">Predicted Ticket Price</div>
              <div class="rv">₹{pred:,.0f}</div>
              <div class="rs">Range: ₹{low:,.0f} – ₹{high:,.0f}</div>
              <div class="badge">{"±"+str(round(conf,1))+"% confidence" if mode=="real" else "Demo estimate"}</div>
            </div>""", unsafe_allow_html=True)

            tips = []
            if int(days_until)<=7:        tips.append("🚨 Last-minute — prices at peak. Try flexible dates.")
            elif int(days_until)>=90:     tips.append("✅ 90+ days ahead saves 20–35% vs last-minute.")
            if restr:                     tips.append("🚧 Restricted airspace adds ~9% reroute cost.")
            if saf_z==2:                  tips.append("🌿 EU/UK mandatory SAF zone adds ~6% levy.")
            if cabin=="Economy" and distance_km>8000: tips.append("💡 Long-haul >8,000km — consider Premium Economy.")
            if month in [3,4,5,12,1]:    tips.append("📅 Peak season — shift ±2 weeks to save.")
            if wday_enc>=5:               tips.append("📆 Tue/Wed departures typically 10–15% cheaper.")
            if tips:
                st.markdown('<div class="tip">'+"<br>".join(tips)+'</div>', unsafe_allow_html=True)
        else:
            st.markdown("""<div style='background:#F8FBFF;border:2px dashed #90B8E0;border-radius:12px;
                 padding:2.8rem 1.5rem;text-align:center;color:#7098CC;margin-top:1rem;'>
              <div style='font-size:2.2rem;'>💰</div>
              <div style='font-weight:600;margin-top:.5rem;'>Price appears here</div>
              <div style='font-size:.8rem;margin-top:.25rem;'>Fill the form and click Predict</div>
            </div>""", unsafe_allow_html=True)

    # Key drivers section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec">🔑 Key Price Drivers</div>', unsafe_allow_html=True)
    cols3 = st.columns(3)
    for i,(icon,title,desc) in enumerate([
        ("💺","Cabin Class",         "Dominant driver. First class ~3.5× Economy."),
        ("📏","Distance",            "Each 1,000 km adds ~₹2,000–3,000 non-linearly."),
        ("⏳","Booking Window",      "Last-minute (<7 days) fares 25–40% higher."),
        ("🌿","SAF Zone (BRD)",      "EU mandatory zones add 6% · voluntary 2%."),
        ("🏭","Fleet Age (BRD)",     "Older fleets have higher operating costs."),
        ("🚧","Restricted Airspace", "Reroute penalty adds ~9% on affected routes."),
    ]):
        with cols3[i%3]:
            st.markdown(f'<div class="fcard"><h4>{icon} {title}</h4><p>{desc}</p></div>',
                        unsafe_allow_html=True)
