"""
pages/p5_about.py  —  About  (Classic Edition)
"""
import streamlit as st
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.loader import load_model_artefacts


def render():
    _, __, ___, meta, loaded = load_model_artefacts()
    r2    = meta.get("r2",    0.980)
    mape  = meta.get("mape",  12.1)
    mae   = meta.get("mae",   1580)
    rows  = meta.get("total_rows", 120000)
    nf    = meta.get("n_features", 32)
    mname = meta.get("model_name", "LightGBM")
    cvm   = meta.get("cv_mape_mean", 12.3)
    cvs   = meta.get("cv_mape_std",  1.1)
    bline = meta.get("baseline_mape", 45.0)

    st.markdown("""<div class="af-page-hdr">
      <h2>ℹ️  About AirFair Vista</h2>
      <p>Final Year B.Tech Project · Department of Computer Science · 2025–2026</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        # Abstract
        st.markdown('<div class="af-card"><div class="af-card-title">Project Abstract</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
**AirFair Vista** is a production-ready ML system for predicting international flight prices.

A combined dataset of **{rows:,} records** (20k real + 100k synthetic) trains a
**{mname}** model achieving **R²={r2:.4f}** with **MAPE={mape:.1f}%** — validated
using TimeSeriesSplit cross-validation to prevent data leakage.

BRD Phase-2 macro-economic features (SAF mandate zones, environmental surcharge tiers,
fleet age, restricted airspace) are incorporated and confirmed via SHAP to appear in the
top-10 feature contributors.

The project ships as a two-service Docker stack:
`ml_pipeline` handles all training and EDA; `streamlit_app` is a pure frontend
that reads trained artefacts at runtime.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        # Architecture
        st.markdown('<div class="af-card"><div class="af-card-title">Two-Service Architecture</div>',
                    unsafe_allow_html=True)
        st.code("""airfair_v3/
├── ml_pipeline/              ← Training service
│   ├── src/
│   │   ├── config.py         all paths & constants
│   │   ├── data_generator.py 100k synthetic rows, 25 features
│   │   ├── data_loader.py    load raw + merge + BRD backfill
│   │   ├── features.py       ★ SHARED — encoding + build_single_row
│   │   ├── eda.py            16 EDA plots + insights.json
│   │   └── trainer.py        6 models, TimeSeriesSplit, SHAP
│   ├── data/raw/             flight_price_dataset.csv
│   ├── data/processed/       generated CSVs
│   ├── models/               model.pkl · encoders.pkl · features.pkl
│   │                         model_meta.json
│   ├── reports/              01–16 PNG plots + insights.json
│   ├── train.py              entry point
│   └── Dockerfile
│
├── streamlit_app/            ← Frontend service (reads models/ + reports/)
│   ├── app.py                entry point — 5-page sidebar nav
│   ├── pages/
│   │   ├── p1_predict.py     Price Predictor + distance calculator
│   │   ├── p2_eda.py         EDA charts + text insights
│   │   ├── p3_features.py    Feature engineering visuals
│   │   ├── p4_models.py      Model comparison + SHAP
│   │   └── p5_about.py       This page
│   ├── utils/
│   │   ├── loader.py         @st.cache_resource PKL loader
│   │   ├── style.py          Classic design system CSS
│   │   └── distance.py       Haversine calculator + city coords
│   └── Dockerfile
│
├── docker-compose.yml        one command deploys both services
└── Makefile""", language="text")
        st.markdown('</div>', unsafe_allow_html=True)

        # Pipeline steps
        st.markdown('<div class="af-card"><div class="af-card-title">ML Pipeline Steps</div>',
                    unsafe_allow_html=True)
        steps = [
            ("Data Generation",     f"100k synthetic rows via `data_generator.py` — all 25 features, BRD macro-factors, realistic price formula"),
            ("Data Loading",        "Original 20k CSV back-filled with BRD columns, merged with synthetic → 120k combined, sorted chronologically"),
            ("EDA",                 "16 Matplotlib/Seaborn plots + `insights.json` saved to `reports/` — read by Streamlit at runtime"),
            ("Feature Engineering", f"{nf} engineered features: 8 label-encoded, 2 log-transforms, 5 interactions, 3 binary flags, 4 BRD macro-factors"),
            ("TimeSeriesSplit",     "85/15 chronological split · 5-fold TS-CV on training set · no future data leakage (BRD requirement)"),
            ("Baseline",            "30-day moving average benchmark — model must beat this MAPE"),
            ("Model Training",      "6 models compared: Linear, Ridge, Random Forest, Gradient Boosting, XGBoost, LightGBM · best auto-selected by MAPE"),
            ("SHAP",                "TreeExplainer on best model · BRD macro-factors verified in top-10 · bar + beeswarm plots saved to reports/"),
            ("Artefact Save",       "`model.pkl` · `encoders.pkl` · `features.pkl` · `model_meta.json` → written to `models/`"),
        ]
        for i, (title, desc) in enumerate(steps):
            st.markdown(f"""
            <div style='display:flex;gap:.8rem;align-items:flex-start;
                 padding:.5rem 0;border-bottom:1px solid #EDE8DC;'>
              <div style='background:#0D1B2A;color:#C5A028;border-radius:4px;
                   padding:.15rem .55rem;font-size:.75rem;font-weight:700;
                   flex-shrink:0;margin-top:.1rem;'>{i+1}</div>
              <div>
                <div style='font-weight:600;font-size:.88rem;color:#0D1B2A;'>{title}</div>
                <div style='font-size:.8rem;color:#4A5568;margin-top:.1rem;'>{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Live metrics
        st.markdown('<div class="af-card"><div class="af-card-title">Live Model Performance</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="af-metric-strip">
          <div class="af-metric">
            <div class="af-metric-val">{r2*100:.1f}%</div>
            <div class="af-metric-label">R² Score</div>
          </div>
          <div class="af-metric">
            <div class="af-metric-val">{mape:.1f}%</div>
            <div class="af-metric-label">MAPE</div>
          </div>
          <div class="af-metric">
            <div class="af-metric-val">₹{mae:,}</div>
            <div class="af-metric-label">MAE</div>
          </div>
          <div class="af-metric">
            <div class="af-metric-val">{cvm:.1f}%</div>
            <div class="af-metric-label">CV MAPE</div>
          </div>
        </div>
        <div style='font-size:.8rem;color:#718096;margin-top:.3rem;'>
          Baseline (MA-30): {bline:.1f}% →
          <strong style='color:#0D1B2A;'>+{bline-mape:.1f}pp improvement</strong>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Quick start
        st.markdown('<div class="af-card"><div class="af-card-title">Quick Start</div>',
                    unsafe_allow_html=True)
        st.markdown("**Local development**")
        st.code("""# Install & train
cd ml_pipeline
pip install -r requirements.txt
python train.py          # ~5–8 min

# Launch app
cd ../streamlit_app
pip install -r requirements.txt
streamlit run app.py
# → http://localhost:8501""", language="bash")

        st.markdown("**Docker (production)**")
        st.code("""# One command — builds, trains, serves
docker-compose up --build
# → http://localhost:8501

# Re-train with more data
docker-compose run ml_pipeline python train.py --rows 500000""",
                language="bash")
        st.markdown('</div>', unsafe_allow_html=True)

        # Tech stack
        st.markdown('<div class="af-card"><div class="af-card-title">Tech Stack</div>',
                    unsafe_allow_html=True)

        stack = [
            ("ML Models",       ["LightGBM","XGBoost","Gradient Boosting","Random Forest"]),
            ("Explainability",  ["SHAP TreeExplainer"]),
            ("BRD Features",    ["SAF Zones","Env Tiers","Fleet Age","Airspace Flags"]),
            ("Validation",      ["TimeSeriesSplit 5-fold CV"]),
            ("Data",            ["Pandas","NumPy","SciPy"]),
            ("Visualisation",   ["Matplotlib","Seaborn"]),
            ("Frontend",        ["Streamlit 1.36+","Plotly"]),
            ("Distance Calc",   ["Haversine formula","City coord map"]),
            ("Deploy",          ["Docker","docker-compose","multi-stage build"]),
        ]
        for cat, items in stack:
            st.markdown(
                f"<div style='font-size:.65rem;font-weight:700;color:#718096;"
                f"text-transform:uppercase;letter-spacing:.08em;"
                f"margin-top:.55rem;'>{cat}</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                " &nbsp;·&nbsp; ".join([f"`{x}`" for x in items]),
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown(f"""
    <div style='text-align:center;color:#A0AEC0;font-size:.75rem;
         padding:1.5rem 0 .5rem;border-top:1px solid #EDE8DC;margin-top:1rem;'>
      AirFair Vista &nbsp;·&nbsp; Final Year B.Tech Project &nbsp;·&nbsp;
      Computer Science &nbsp;·&nbsp; 2025–2026<br>
      {mname} &nbsp;·&nbsp; Docker two-service stack &nbsp;·&nbsp; BRD compliant
    </div>""", unsafe_allow_html=True)
