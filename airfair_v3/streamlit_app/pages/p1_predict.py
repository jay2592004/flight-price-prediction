"""
pages/p1_predict.py  —  Price Predictor page
"""
import streamlit as st
import numpy as np
import sys, os, calendar
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.loader import load_model_artefacts, load_insights

# ── Lookup tables (keep in sync with ml_pipeline/src/data_generator.py) ──────
AIRLINES = sorted(["Air France","Air India","AirAsia India","British Airways",
    "Cathay Pacific","Emirates","Etihad Airways","IndiGo","Lufthansa",
    "Qatar Airways","Singapore Airlines","SpiceJet","Thai Airways",
    "Turkish Airlines","Vistara"])
CITIES = sorted(["Ahmedabad","Bangalore","Bangkok","Chennai","Delhi","Doha",
    "Dubai","Frankfurt","Hong Kong","Istanbul","London","Mumbai",
    "New York","Paris","Singapore"])
CLASSES = ["Economy","Premium Economy","Business","First"]
STOPS   = ["non-stop","1 stop","2 stops"]

AIRLINE_TIER = {
    "Singapore Airlines":"premium","Emirates":"premium","Qatar Airways":"premium",
    "British Airways":"premium","Lufthansa":"premium","Cathay Pacific":"premium",
    "Air France":"mid","Etihad Airways":"mid","Thai Airways":"mid",
    "Turkish Airlines":"mid","Air India":"mid","Vistara":"mid",
    "AirAsia India":"budget","IndiGo":"budget","SpiceJet":"budget",
}
SAF_ZONE = {"London":2,"Frankfurt":2,"Paris":2,"Istanbul":1,"Singapore":1,
            "New York":1,"Dubai":0,"Doha":0,"Bangkok":0,"Hong Kong":0,
            "Delhi":0,"Mumbai":0,"Bangalore":0,"Chennai":0,"Ahmedabad":0}
ENV_TIER = {"London":3,"Frankfurt":3,"Paris":3,"New York":2,"Istanbul":2,
            "Singapore":2,"Dubai":1,"Doha":1,"Bangkok":1,"Hong Kong":1,
            "Delhi":0,"Mumbai":0,"Bangalore":0,"Chennai":0,"Ahmedabad":0}
FLEET_AGE = {"Singapore Airlines":6.2,"Emirates":7.8,"Qatar Airways":6.5,
             "British Airways":13.4,"Lufthansa":12.1,"Cathay Pacific":9.3,
             "Air France":11.7,"Etihad Airways":8.4,"Thai Airways":16.2,
             "Turkish Airlines":9.8,"Air India":14.1,"Vistara":5.3,
             "AirAsia India":7.1,"IndiGo":5.9,"SpiceJet":10.2}
RESTRICTED = {("Delhi","Frankfurt"),("Delhi","Paris"),("Delhi","London"),
              ("Mumbai","Frankfurt"),("Mumbai","Paris"),("Mumbai","London"),
              ("Bangkok","Frankfurt"),("Dubai","New York"),("Doha","New York")}
DIST_MAP = {
    ("Delhi","Mumbai"):1400,("Delhi","London"):6700,("Delhi","Dubai"):2200,
    ("Delhi","Singapore"):4150,("Delhi","New York"):11750,("Delhi","Bangkok"):2900,
    ("Mumbai","London"):7200,("Mumbai","Dubai"):1950,("Mumbai","Singapore"):4180,
    ("Mumbai","New York"):12550,("London","New York"):5540,("London","Dubai"):5500,
    ("Dubai","Singapore"):5840,("Dubai","New York"):11020,
    ("Singapore","Hong Kong"):2580,("Singapore","Bangkok"):1450,
    ("Doha","New York"):11540,("Chennai","Dubai"):2900,("Bangalore","Dubai"):2900,
}


def _demo_price(airline, cabin, stops, dist, days, month, wday):
    base  = {"Economy":5000,"Premium Economy":12000,"Business":28000,"First":55000}[cabin]
    price = base + dist * 1.9
    price *= {"non-stop":1.08,"1 stop":0.95,"2 stops":0.88}[stops]
    price *= 1.40 if days<7 else (1.20 if days<30 else 1.00)
    price *= 1.18 if month in [12,1,3,4,5] else 1.00
    price *= 1.07 if wday >= 5 else 1.00
    price *= {"premium":1.30,"mid":1.00,"budget":0.82}[AIRLINE_TIER.get(airline,"mid")]
    return max(1500, price)


def render():
    model, encoders, features, meta, loaded = load_model_artefacts()
    insights = load_insights()
    n_rows = meta.get("total_rows", 120000)
    model_n = meta.get("model_name","—")
    r2_pct  = f"{meta['r2']*100:.1f}%" if loaded else "—"
    mape_v  = f"{meta['mape']:.1f}%" if loaded else "—"

    # Hero
    st.markdown(f"""<div class="hero">
      <h1>AirFair Vista ✈️</h1>
      <p>Powered by <b>{model_n}</b> · {n_rows:,} training records ·
         15 airlines · 15 cities · 4 cabin classes · BRD macro-factors.</p>
    </div>""", unsafe_allow_html=True)

    # KPI row
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(v,l,s) in zip([c1,c2,c3,c4,c5],[
        (r2_pct,        "Model Accuracy",  "R² Score"),
        (f"{n_rows:,}", "Training Rows",   "Combined dataset"),
        (mape_v,        "Error Rate",      "MAPE"),
        ("15",          "Airlines",        "Global carriers"),
        ("32",          "Features",        "incl. BRD macro-factors"),
    ]):
        with col:
            st.markdown(f'<div class="kpi"><div class="v">{v}</div>'
                        f'<div class="l">{l}</div><div class="s">{s}</div></div>',
                        unsafe_allow_html=True)

    if not loaded:
        st.warning("⚠️ Model not trained. Run `cd ml_pipeline && python train.py` first.", icon="⚙️")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec">✈️ Predict Your Flight Price</div>', unsafe_allow_html=True)

    col_form, col_res = st.columns([3, 2], gap="large")

    with col_form:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**🛫 Route & Airline**")
        r1, r2 = st.columns(2)
        with r1:
            airline = st.selectbox("Airline", AIRLINES, index=AIRLINES.index("Emirates"))
            source  = st.selectbox("From", CITIES, index=CITIES.index("Dubai"))
        with r2:
            cabin     = st.selectbox("Cabin Class", CLASSES, index=1)
            dest_opts = [c for c in CITIES if c != source]
            dest      = st.selectbox("To", dest_opts,
                                     index=dest_opts.index("London") if "London" in dest_opts else 0)

        st.markdown("---")
        st.markdown("**🛑 Flight Details**")
        d1, d2, d3 = st.columns(3)
        with d1:
            stops = st.selectbox("Stops", STOPS)
        with d2:
            pair = (source, dest)
            default_dist = DIST_MAP.get(pair, DIST_MAP.get((dest, source), 5000))
            distance_km  = st.number_input("Distance (km)", 300, 20000, default_dist, 100)
        with d3:
            days_until = st.number_input("Days Until Departure", 1, 365, 45)

        st.markdown("---")
        st.markdown("**📅 Journey Date**")
        t1, t2, t3 = st.columns(3)
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
    tier   = AIRLINE_TIER.get(airline, "mid")
    saf_z  = SAF_ZONE.get(dest, 0)
    env_t  = max(ENV_TIER.get(source, 0), ENV_TIER.get(dest, 0))
    fage   = FLEET_AGE.get(airline, 10.0)
    restr  = 1 if (source,dest) in RESTRICTED or (dest,source) in RESTRICTED else 0
    season = "peak" if month in [12,1,3,4,5] else ("shoulder" if month in [2,6,10,11] else "off_peak")
    actype = "wide-body" if distance_km > 4000 else "narrow-body"
    lhours = 0 if stops=="non-stop" else (3 if stops=="1 stop" else 7)

    with col_res:
        st.markdown("#### 📋 Booking Summary")
        st.markdown(f"""<div style='background:#F0F6FF;border-radius:12px;
             padding:1.2rem 1.5rem;border:1px solid #DCE8FA;font-size:.87rem;line-height:2.2;'>
          <b>✈️ Route</b>: {source} → {dest}<br>
          <b>🏢 Airline</b>: {airline} <i>({tier})</i><br>
          <b>💺 Class</b>: {cabin}<br>
          <b>🛑 Stops</b>: {stops} · {actype}<br>
          <b>📏 Distance</b>: {distance_km:,} km<br>
          <b>📅 Date</b>: {int(day)} {month_name} ({wday_name}) — {season}<br>
          <b>⏳ Booking</b>: {int(days_until)} days ahead<br>
          <b>🌿 SAF Zone</b>: {saf_z} · Env Tier: {env_t}<br>
          <b>🏭 Fleet Age</b>: {fage:.1f} yr<br>
          <b>🚧 Airspace</b>: {"Restricted ⚠️" if restr else "Normal ✓"}
        </div>""", unsafe_allow_html=True)

        if predict_btn:
            if loaded:
                try:
                    import sys as _sys
                    _ml = os.environ.get("ML_PIPELINE_PATH",
                          os.path.abspath(os.path.join(
                              os.path.dirname(__file__), "..", "..", "ml_pipeline")))
                    if _ml not in _sys.path: _sys.path.insert(0, _ml)
                    from src.features import build_single_row
                    inp = {
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
                    X_pred = build_single_row(inp, encoders)
                    pred   = float(model.predict(X_pred)[0])
                    mode   = "real"
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    pred = _demo_price(airline,cabin,stops,distance_km,int(days_until),month,wday_enc)
                    mode = "demo"
            else:
                pred = _demo_price(airline,cabin,stops,distance_km,int(days_until),month,wday_enc)
                mode = "demo"

            conf = meta.get("mape", 13.5)
            low, high = pred*(1-conf/100), pred*(1+conf/100)
            st.markdown(f"""<div class="result">
              <div class="rl">Predicted Ticket Price</div>
              <div class="rv">₹{pred:,.0f}</div>
              <div class="rs">Range: ₹{low:,.0f} – ₹{high:,.0f}</div>
              <div class="badge">{"±"+str(round(conf,1))+"% confidence" if mode=="real" else "Demo estimate"}</div>
            </div>""", unsafe_allow_html=True)

            tips = []
            if int(days_until) <= 7:             tips.append("🚨 Last-minute — prices at peak. Try flexible dates.")
            elif int(days_until) >= 90:          tips.append("✅ 90+ days ahead saves 20–35% vs last-minute.")
            if restr:                            tips.append("🚧 Restricted airspace adds ~9% reroute cost.")
            if saf_z == 2:                       tips.append("🌿 EU/UK SAF mandate adds ~6% environmental levy.")
            if cabin=="Economy" and distance_km>8000: tips.append("💡 Long-haul >8,000km — consider Premium Economy.")
            if month in [3,4,5,12,1]:            tips.append("📅 Peak season — shifting ±2 weeks can save.")
            if wday_enc >= 5:                    tips.append("📆 Tue/Wed departures are 10–15% cheaper on average.")
            if tips:
                st.markdown('<div class="tip">'+"<br>".join(tips)+'</div>', unsafe_allow_html=True)
        else:
            st.markdown("""<div style='background:#F8FBFF;border:2px dashed #90B8E0;
                 border-radius:12px;padding:2.8rem 1.5rem;text-align:center;color:#7098CC;margin-top:1rem;'>
              <div style='font-size:2.2rem;'>💰</div>
              <div style='font-weight:600;margin-top:.5rem;'>Price appears here</div>
              <div style='font-size:.8rem;margin-top:.25rem;'>Fill the form and click Predict</div>
            </div>""", unsafe_allow_html=True)

    # Key drivers
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec">🔑 Key Price Drivers</div>', unsafe_allow_html=True)
    cols3 = st.columns(3)
    mult = insights.get("first_vs_economy","~3.5")
    lm_p = insights.get("lastminute_premium","~1.4")
    saf_p = insights.get("saf_zone2_premium_pct","~6")
    for i, (icon,title,desc) in enumerate([
        ("💺","Cabin Class",          f"#1 driver. First class is {mult}× Economy on average."),
        ("📏","Distance",             "Each 1,000 km adds ~₹2,000–3,000 (non-linear)."),
        ("⏳","Booking Window",       f"Last-minute fares are {lm_p}× early-bird prices."),
        ("🌿","SAF Zone (BRD)",       f"EU mandatory zones add ~{saf_p}% to ticket price."),
        ("🏭","Fleet Age (BRD)",      "Older fleets have higher ops costs passed to passengers."),
        ("🚧","Restricted Airspace",  "Reroute penalty adds ~9% on affected routes."),
    ]):
        with cols3[i % 3]:
            st.markdown(f'<div class="fcard"><h4>{icon} {title}</h4><p>{desc}</p></div>',
                        unsafe_allow_html=True)
