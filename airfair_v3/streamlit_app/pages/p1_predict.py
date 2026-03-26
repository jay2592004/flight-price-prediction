"""
pages/p1_predict.py  —  P1: Predict
Dynamic travel time recalculates instantly whenever stops, distance,
source or destination change — no button press required.
"""
import streamlit as st
import numpy as np
import sys, os, calendar
from datetime import date, timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.loader   import load_model_artefacts, load_insights
from utils.distance import (
    CITY_COORDS, calculate_distance, flight_duration_estimate,
)

# ── Reference data ────────────────────────────────────────────────────────────
AIRLINES = sorted([
    "Air France","Air India","AirAsia India","British Airways","Cathay Pacific",
    "Emirates","Etihad Airways","IndiGo","Lufthansa","Qatar Airways",
    "Singapore Airlines","SpiceJet","Thai Airways","Turkish Airlines","Vistara",
])
CITIES = sorted([
    "Ahmedabad","Bangalore","Bangkok","Chennai","Delhi","Doha","Dubai",
    "Frankfurt","Hong Kong","Istanbul","London","Mumbai","New York","Paris","Singapore",
])
CLASSES = ["Economy","Premium Economy","Business","First"]
STOPS   = ["non-stop","1 stop","2 stops"]

AIRLINE_TIER = {
    "Singapore Airlines":"premium","Emirates":"premium","Qatar Airways":"premium",
    "British Airways":"premium","Lufthansa":"premium","Cathay Pacific":"premium",
    "Air France":"mid","Etihad Airways":"mid","Thai Airways":"mid",
    "Turkish Airlines":"mid","Air India":"mid","Vistara":"mid",
    "AirAsia India":"budget","IndiGo":"budget","SpiceJet":"budget",
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
FLEET_AGE = {
    "Singapore Airlines":6.2,"Emirates":7.8,"Qatar Airways":6.5,
    "British Airways":13.4,"Lufthansa":12.1,"Cathay Pacific":9.3,
    "Air France":11.7,"Etihad Airways":8.4,"Thai Airways":16.2,
    "Turkish Airlines":9.8,"Air India":14.1,"Vistara":5.3,
    "AirAsia India":7.1,"IndiGo":5.9,"SpiceJet":10.2,
}
RESTRICTED = {
    ("Delhi","Frankfurt"),("Delhi","Paris"),("Delhi","London"),
    ("Mumbai","Frankfurt"),("Mumbai","Paris"),("Mumbai","London"),
    ("Bangkok","Frankfurt"),("Dubai","New York"),("Doha","New York"),
}
SAF_LABEL  = {0:"No mandate",   1:"Voluntary (Zone 1)", 2:"EU Mandatory (Zone 2)"}
ENV_LABEL  = {0:"None",         1:"Low",                2:"Medium",  3:"High (EU ETS)"}
TIER_LABEL = {"budget":"Budget","mid":"Mid-tier",       "premium":"Premium"}


def _weekday_from_date(month: int, day: int, year: int):
    safe_day = min(day, calendar.monthrange(year, month)[1])
    weekday_index = calendar.weekday(year, month, safe_day)
    weekday_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][weekday_index]
    return safe_day, weekday_index, weekday_name


def _sync_journey_details():
    journey_date = st.session_state.get("journey_date")
    if not journey_date:
        st.session_state["journey_weekday_label"] = ""
        st.session_state["journey_days_until_label"] = ""
        return

    today = date.today()
    weekday_index = journey_date.weekday()
    weekday_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][weekday_index]
    days_until = (journey_date - today).days

    st.session_state["journey_weekday_label"] = weekday_name
    st.session_state["journey_days_until_label"] = str(max(days_until, 0))


def _demo_price(airline, cabin, stops, dist, days, month, wday):
    base  = {"Economy":5000,"Premium Economy":12000,"Business":28000,"First":55000}[cabin]
    price = base + dist * 1.9
    price *= {"non-stop":1.08,"1 stop":0.95,"2 stops":0.88}[stops]
    price *= 1.40 if days<7  else (1.20 if days<30 else 1.00)
    price *= 1.18 if month in [12,1,3,4,5] else 1.00
    price *= 1.07 if wday >= 5 else 1.00
    price *= {"premium":1.30,"mid":1.00,"budget":0.82}[AIRLINE_TIER.get(airline,"mid")]
    return max(1500, price)


def render():
    model, encoders, features, meta, loaded = load_model_artefacts()
    insights = load_insights()
    min_journey_date = date.today() + timedelta(days=1)

    st.session_state.setdefault("journey_weekday_label", "")
    st.session_state.setdefault("journey_days_until_label", "")

    n_rows  = meta.get("total_rows", 120000)
    model_n = meta.get("model_name", "—")
    r2_pct  = f"{meta['r2']*100:.1f}%" if loaded else "—"
    mape_v  = f"{meta['mape']:.1f}%"   if loaded else "—"
    mae_v   = f"₹{meta['mae']:,}"      if loaded else "—"

    # ── Hero ───────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="af-hero">
      <div class="af-hero-eyebrow">AirFair Vista · Flight Intelligence Platform</div>
      <h1>Flight Price Predictor</h1>
      <p>Powered by <strong style='color:#C5A028;'>{model_n}</strong> &nbsp;·&nbsp;
         {n_rows:,} training records &nbsp;·&nbsp;
         15 airlines &nbsp;·&nbsp; 15 global cities &nbsp;·&nbsp;
         BRD macro-economic factors</p>
    </div>""", unsafe_allow_html=True)

    # ── KPI strip ───────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="af-kpi-strip">
      <div class="af-kpi">
        <div class="af-kpi-value gold">{r2_pct}</div>
        <div class="af-kpi-label">Model Accuracy</div>
        <div class="af-kpi-sub">R² Score</div>
      </div>
      <div class="af-kpi">
        <div class="af-kpi-value">{n_rows:,}</div>
        <div class="af-kpi-label">Training Records</div>
        <div class="af-kpi-sub">Combined dataset</div>
      </div>
      <div class="af-kpi">
        <div class="af-kpi-value gold">{mape_v}</div>
        <div class="af-kpi-label">Error Rate</div>
        <div class="af-kpi-sub">MAPE</div>
      </div>
      <div class="af-kpi">
        <div class="af-kpi-value">{mae_v}</div>
        <div class="af-kpi-label">Mean Abs Error</div>
        <div class="af-kpi-sub">MAE</div>
      </div>
      <div class="af-kpi">
        <div class="af-kpi-value">15</div>
        <div class="af-kpi-label">Airlines</div>
        <div class="af-kpi-sub">Global carriers</div>
      </div>
      <div class="af-kpi">
        <div class="af-kpi-value">32</div>
        <div class="af-kpi-label">Features</div>
        <div class="af-kpi-sub">incl. BRD factors</div>
      </div>
    </div>""", unsafe_allow_html=True)

    if not loaded:
        st.warning(
            "Model not trained yet. Run `cd ml_pipeline && python train.py` first.",
            icon="⚙️",
        )

    st.markdown('<div class="af-section-title">Predict Your Flight Price</div>',
                unsafe_allow_html=True)
    st.caption(
        "Recommended sequence: choose route and airline, set the journey date, "
        "confirm booking details, review distance, then calculate the fare."
    )

    col_form, col_res = st.columns([3, 2], gap="large")

    # ══════════════════════════════════════════════════════════════════════════
    # LEFT COLUMN — inputs
    # ══════════════════════════════════════════════════════════════════════════
    with col_form:

        # Panel 1 — Route & Airline
        st.markdown('<div class="af-panel">', unsafe_allow_html=True)
        st.markdown('<div class="af-panel-title">1. Route &amp; Airline</div>',
                    unsafe_allow_html=True)
        r1, r2_ = st.columns(2)
        with r1:
            airline = st.selectbox(
                "Airline",
                AIRLINES,
                index=None,
                placeholder="Select airline",
                key="airline",
            )
            source  = st.selectbox(
                "Departure city",
                CITIES,
                index=None,
                placeholder="Select departure city",
                key="source",
            )
        with r2_:
            cabin = st.selectbox(
                "Cabin class",
                CLASSES,
                index=None,
                placeholder="Select cabin class",
                key="cabin",
            )
            dest_opts = [c for c in CITIES if c != source] if source else CITIES
            dest      = st.selectbox(
                "Arrival city", dest_opts,
                index=None,
                placeholder="Select arrival city",
                key="dest",
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # Panel 2 — Journey Date
        st.markdown('<div class="af-panel">', unsafe_allow_html=True)
        st.markdown('<div class="af-panel-title">2. Journey Date</div>',
                    unsafe_allow_html=True)
        jd1, jd2 = st.columns(2)
        with jd1:
            journey_date = st.date_input(
                "Flight date",
                value=None,
                min_value=min_journey_date,
                format="DD/MM/YYYY",
                key="journey_date",
                on_change=_sync_journey_details,
            )

        if journey_date:
            month = journey_date.month
            day = journey_date.day
            month_name = calendar.month_name[month]
            wday_enc = journey_date.weekday()
            wday_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][wday_enc]
            days_until = (journey_date - date.today()).days
        else:
            month = None
            day = None
            month_name = None
            wday_enc = None
            wday_name = ""
            days_until = None

        with jd2:
            st.text_input(
                "Day of week",
                value=st.session_state["journey_weekday_label"] or wday_name,
                disabled=True,
                help="Calculated automatically after you confirm the flight date",
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # Panel 3 — Flight Details (stops + days)
        st.markdown('<div class="af-panel">', unsafe_allow_html=True)
        st.markdown('<div class="af-panel-title">3. Flight Details</div>',
                    unsafe_allow_html=True)
        fd1, fd2 = st.columns(2)
        with fd1:
            stops = st.selectbox(
                "Number of stops",
                STOPS,
                index=None,
                placeholder="Select number of stops",
                key="stops",
            )
        with fd2:
            st.text_input(
                "Days until departure",
                value=(
                    st.session_state["journey_days_until_label"]
                    if st.session_state["journey_days_until_label"]
                    else ("" if days_until is None else str(days_until))
                ),
                disabled=True,
                help="Calculated automatically from the flight date",
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # Panel 4 — Distance
        st.markdown('<div class="af-panel">', unsafe_allow_html=True)
        st.markdown('<div class="af-panel-title">4. Route Distance</div>',
                    unsafe_allow_html=True)

        dist_mode = st.radio(
            "Distance method",
            ["Auto-calculate from cities", "Enter manually (km)"],
            horizontal=True,
            label_visibility="collapsed",
            key="dist_mode",
        )

        # ── Auto or manual distance ────────────────────────────────────────
        distance_km = None
        if not source or not dest:
            st.info("Select departure and arrival city to continue.", icon="ðŸ“")
        elif dist_mode == "Auto-calculate from cities":
            auto_dist, method = calculate_distance(source, dest)
            distance_km = auto_dist

            method_label = {
                "lookup":    "Exact route data",
                "haversine": "Haversine great-circle",
                "estimate":  "Approximate estimate",
            }[method]

            # Dynamic travel time — recalculates on every widget change
            dur = flight_duration_estimate(distance_km, stops)

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric("Distance", f"{distance_km:,} km")
            with mc2:
                st.metric("Est. travel time", dur,
                          help="Flight time + layover. Updates when stops change.")
            with mc3:
                st.metric("Source", method_label)

            st.markdown(
                f'<div class="af-dist-result">'
                f'📐 {source} → {dest} &nbsp;·&nbsp; {distance_km:,} km'
                f' &nbsp;·&nbsp; {dur} &nbsp;·&nbsp; {method_label}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Haversine breakdown
            if method == "haversine" and source in CITY_COORDS and dest in CITY_COORDS:
                with st.expander("Haversine calculation details"):
                    lat1, lon1 = CITY_COORDS[source]
                    lat2, lon2 = CITY_COORDS[dest]
                    st.markdown(f"""
| | {source} | {dest} |
|---|---|---|
| Latitude  | {lat1:.4f}° | {lat2:.4f}° |
| Longitude | {lon1:.4f}° | {lon2:.4f}° |

**Formula:** d = 2R · arcsin(√(sin²(Δφ/2) + cos φ₁ · cos φ₂ · sin²(Δλ/2)))
&nbsp;&nbsp;→ **{distance_km:,} km** &nbsp;*(R = 6,371 km)*
                    """)

        else:  # Manual entry
            auto_ref, _ = calculate_distance(source, dest)
            distance_km = st.number_input(
                f"Distance (km)  — auto-calculated reference: {auto_ref:,} km",
                min_value=100, max_value=20_000,
                value=auto_ref, step=50, key="dist_manual",
            )
            # Dynamic travel time — still updates on stops change
            dur = flight_duration_estimate(distance_km, stops)
            st.markdown(
                f'<div class="af-traveltime">'
                f'🕐 Estimated travel time: <strong>{dur}</strong>'
                f'<span class="af-traveltime-label">Updates with stops &amp; distance</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if abs(distance_km - auto_ref) > 500:
                st.info(
                    f"Your value ({distance_km:,} km) differs from the "
                    f"auto-calculated reference ({auto_ref:,} km). "
                    "This is fine for custom routes.",
                    icon="📐",
                )

        st.markdown('</div>', unsafe_allow_html=True)

        # Predict button
        form_ready = all([
            airline,
            source,
            dest,
            cabin,
            journey_date,
            stops,
            distance_km is not None,
            days_until is not None,
        ])
        predict_btn = st.button(
            "Calculate Flight Price",
            use_container_width=True,
            disabled=not form_ready,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Derived values (computed from current widget state every render)
    # ══════════════════════════════════════════════════════════════════════════
    tier    = AIRLINE_TIER.get(airline, "mid") if airline else None
    saf_z   = SAF_ZONE.get(dest, 0) if dest else 0
    env_t   = max(ENV_TIER.get(source, 0), ENV_TIER.get(dest, 0)) if source and dest else 0
    fage    = FLEET_AGE.get(airline, 10.0) if airline else None
    restr   = 1 if source and dest and ((source,dest) in RESTRICTED or (dest,source) in RESTRICTED) else 0
    season  = (("peak"     if month in [12,1,3,4,5] else
                "shoulder" if month in [2,6,10,11]  else "off_peak")
               if month else "off_peak")
    actype  = ("wide-body" if distance_km > 4000 else "narrow-body") if distance_km is not None else None
    lhours  = 0 if stops=="non-stop" else (3 if stops=="1 stop" else 7) if stops else 0
    # Dynamic travel time (same formula, always in sync with current stops+distance)
    dur_display = flight_duration_estimate(distance_km, stops) if distance_km is not None and stops else "-"
    season_disp = {"peak":"Peak","shoulder":"Shoulder","off_peak":"Off-peak"}[season] if journey_date else "-"

    # ══════════════════════════════════════════════════════════════════════════
    # RIGHT COLUMN — summary + result
    # ══════════════════════════════════════════════════════════════════════════
    with col_res:
        st.markdown(
            "<div style='font-size:.7rem;font-weight:700;color:#718096;"
            "text-transform:uppercase;letter-spacing:.1em;margin-bottom:.45rem;'>"
            "Booking Summary</div>",
            unsafe_allow_html=True,
        )

        restr_class = "restricted" if restr else "normal"
        restr_val   = "Restricted ⚠" if restr else "Normal ✓"

        st.markdown(f"""
        <div class="af-summary">
          <div class="af-summary-row">
            <span class="af-summary-key">Route</span>
            <span class="af-summary-val">{f"{source} -> {dest}" if source and dest else "-"}</span>
          </div>
          <div class="af-summary-row">
            <span class="af-summary-key">Airline</span>
            <span class="af-summary-val">{airline or "-"}</span>
          </div>
          <div class="af-summary-row">
            <span class="af-summary-key">Tier</span>
            <span class="af-summary-val">{TIER_LABEL[tier] if tier else "-"}</span>
          </div>
          <div class="af-summary-row">
            <span class="af-summary-key">Cabin</span>
            <span class="af-summary-val">{cabin or "-"}</span>
          </div>
          <div class="af-summary-row">
            <span class="af-summary-key">Stops</span>
            <span class="af-summary-val">{stops or "-"}</span>
          </div>
          <div class="af-summary-row">
            <span class="af-summary-key">Aircraft</span>
            <span class="af-summary-val muted">{actype or "-"}</span>
          </div>
          <div class="af-summary-row">
            <span class="af-summary-key">Distance</span>
            <span class="af-summary-val">{"-" if distance_km is None else f"{distance_km:,} km"}</span>
          </div>
          <div class="af-summary-row">
            <span class="af-summary-key">Travel time</span>
            <span class="af-summary-val gold">{dur_display if journey_date else "-"}</span>
          </div>
          <div class="af-summary-row">
            <span class="af-summary-key">Departure</span>
            <span class="af-summary-val">{"-" if journey_date is None else f"{int(day)} {month_name} ({wday_name})"}</span>
          </div>
          <div class="af-summary-row">
            <span class="af-summary-key">Season</span>
            <span class="af-summary-val">{season_disp}</span>
          </div>
          <div class="af-summary-row">
            <span class="af-summary-key">Booking lead</span>
            <span class="af-summary-val">{"-" if days_until is None else f"{int(days_until)} days"}</span>
          </div>
          <div class="af-summary-row">
            <span class="af-summary-key">SAF Zone</span>
            <span class="af-summary-val gold">{SAF_LABEL[saf_z]}</span>
          </div>
          <div class="af-summary-row">
            <span class="af-summary-key">Env Tier</span>
            <span class="af-summary-val muted">{ENV_LABEL[env_t]}</span>
          </div>
          <div class="af-summary-row">
            <span class="af-summary-key">Fleet age</span>
            <span class="af-summary-val muted">{"-" if fage is None else f"{fage:.1f} yr"}</span>
          </div>
          <div class="af-summary-row">
            <span class="af-summary-key">Airspace</span>
            <span class="af-summary-val {restr_class}">{restr_val}</span>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Prediction / placeholder ───────────────────────────────────────
        if predict_btn:
            if loaded:
                try:
                    _ml = os.environ.get(
                        "ML_PIPELINE_PATH",
                        os.path.abspath(os.path.join(
                            os.path.dirname(__file__), "..", "..", "ml_pipeline"))
                    )
                    if _ml not in sys.path:
                        sys.path.insert(0, _ml)
                    from src.features import build_single_row
                    inp = {
                        "Airline": airline, "Source": source, "Destination": dest,
                        "Class": cabin, "Total_Stops": stops,
                        "Distance_km": distance_km,
                        "Days_Until_Departure": int(days_until),
                        "Journey_Month": month, "Journey_Day": int(day),
                        "Journey_Weekday": wday_enc,
                        "Season": season,
                        "SAF_Zone": saf_z, "Env_Surcharge_Tier": env_t,
                        "Fleet_Age_Years": fage, "Is_Restricted_Airspace": restr,
                        "Aircraft_Type": actype,
                        "Airline_Tier": tier,
                        "Geo_Risk_Score": restr * 0.5 + 0.1,
                        "Fuel_Price_Index": 100.0,
                        "Seat_Availability": 0.5,
                        "Layover_Hours": lhours,
                    }
                    X_pred = build_single_row(inp, encoders)
                    pred   = float(model.predict(X_pred)[0])
                    mode   = "real"
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    pred = _demo_price(airline, cabin, stops, distance_km,
                                       int(days_until), month, wday_enc)
                    mode = "demo"
            else:
                pred = _demo_price(airline, cabin, stops, distance_km,
                                   int(days_until), month, wday_enc)
                mode = "demo"

            conf  = meta.get("mape", 13.5)
            low   = pred * (1 - conf / 100)
            high  = pred * (1 + conf / 100)
            badge = (f"± {round(conf,1)}% model confidence"
                     if mode == "real"
                     else "Demo estimate — train model for live prediction")

            st.markdown(f"""
            <div class="af-result">
              <div class="af-result-eyebrow">Predicted Ticket Price</div>
              <div class="af-result-price">₹{pred:,.0f}</div>
              <div class="af-result-range">
                Likely range &nbsp;·&nbsp; ₹{low:,.0f} – ₹{high:,.0f}
              </div>
              <div class="af-result-badge">{badge}</div>
              <div class="af-result-model">
                {model_n} &nbsp;·&nbsp; {distance_km:,} km &nbsp;·&nbsp; {dur_display}
              </div>
            </div>""", unsafe_allow_html=True)

            # Smart tips
            tips = []
            if int(days_until) <= 3:
                tips.append("<strong>Urgent booking</strong> — fares at maximum. Very little flexibility.")
            elif int(days_until) <= 7:
                tips.append("<strong>Last-minute</strong> — prices at peak. Try ±2 days for savings.")
            elif int(days_until) >= 90:
                tips.append("<strong>Early-bird window</strong> — booking 90+ days ahead saves 20–35%.")
            if restr:
                tips.append("<strong>Restricted airspace</strong> on this route adds ~9% reroute cost.")
            if saf_z == 2:
                tips.append("<strong>EU/UK SAF mandate</strong> (Zone 2) adds ~6% environmental levy.")
            if cabin == "Economy" and distance_km > 8000:
                tips.append("Long-haul >8,000 km — <strong>Premium Economy</strong> is often worth considering.")
            if month in [3, 4, 5, 12, 1]:
                tips.append("<strong>Peak season</strong> — shifting ±2 weeks could reduce fares.")
            if wday_enc >= 5:
                tips.append("<strong>Weekend departure</strong> — Tuesday/Wednesday is typically 10–15% cheaper.")
            if stops == "2 stops":
                tips.append("<strong>2 stops</strong> adds significant layover time — check if 1-stop is available.")
            if tips:
                st.markdown(
                    '<div class="af-tip">'
                    + "<br>".join(f"· {t}" for t in tips)
                    + '</div>',
                    unsafe_allow_html=True,
                )

        else:
            st.markdown("""
            <div style='background:#FFFFFF;border:1px dashed #D4C9B0;
                 border-radius:8px;padding:2.6rem 1.4rem;
                 text-align:center;color:#A0AEC0;margin-top:1rem;'>
              <div style='font-size:1.8rem;margin-bottom:.45rem;'>₹</div>
              <div style='font-family:Georgia,serif;font-size:1.1rem;
                   font-weight:600;color:#718096;'>
                Price appears here
              </div>
              <div style='font-size:.78rem;margin-top:.25rem;'>
                Complete the form and click Calculate
              </div>
            </div>""", unsafe_allow_html=True)

    # ── Key price drivers ──────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="af-section-title">Key Price Drivers</div>',
                unsafe_allow_html=True)

    mult  = insights.get("first_vs_economy", "~3.5")
    lm_p  = insights.get("lastminute_premium", "~1.4")
    saf_p = insights.get("saf_zone2_premium_pct", "~6")
    dcorr = insights.get("distance_corr", "0.78")

    drv_cols = st.columns(3)
    for i, (icon, title, desc) in enumerate([
        ("💺", "Cabin Class",         f"Dominant driver. First class is {mult}× Economy on average."),
        ("📐", "Route Distance",       f"Correlation r = {dcorr}. Each 1,000 km adds ~₹2,000–3,000."),
        ("⏳", "Booking Window",       f"Last-minute fares are {lm_p}× early-bird. Book 90+ days ahead."),
        ("🌿", "SAF Zone",            f"EU mandatory zones (Zone 2) add ~{saf_p}% to ticket price."),
        ("🏭", "Fleet Age",           "Older fleets carry higher operating costs passed to passengers."),
        ("🚧", "Restricted Airspace", "Reroute penalty adds ~9% on Delhi/Mumbai → Europe routes."),
    ]):
        with drv_cols[i % 3]:
            st.markdown(f"""
            <div class="af-fcard">
              <div class="af-fcard-title">{icon}&nbsp; {title}</div>
              <div class="af-fcard-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    # ── Distance reference table ───────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Distance reference — all city pairs"):
        import pandas as pd
        rows = []
        for src in CITIES:
            for dst in CITIES:
                if src >= dst:
                    continue
                d, method = calculate_distance(src, dst)
                rows.append({
                    "From": src, "To": dst,
                    "Distance (km)": f"{d:,}",
                    "Non-stop": flight_duration_estimate(d, "non-stop"),
                    "1 stop":   flight_duration_estimate(d, "1 stop"),
                    "2 stops":  flight_duration_estimate(d, "2 stops"),
                    "Method": method.title(),
                })
        dist_df = pd.DataFrame(rows)
        fc1, fc2 = st.columns(2)
        with fc1:
            fs = st.selectbox("Filter departure", ["All"] + CITIES, key="dist_filter_src")
        with fc2:
            fd_ = st.selectbox("Filter arrival",   ["All"] + CITIES, key="dist_filter_dst")
        if fs  != "All": dist_df = dist_df[(dist_df["From"]==fs) | (dist_df["To"]==fs)]
        if fd_ != "All": dist_df = dist_df[(dist_df["From"]==fd_) | (dist_df["To"]==fd_)]
        st.dataframe(dist_df, use_container_width=True, hide_index=True)
        st.caption(
            f"{len(dist_df)} route pairs · "
            "Travel times include layovers (1 stop +2h, 2 stops +5h)"
        )
