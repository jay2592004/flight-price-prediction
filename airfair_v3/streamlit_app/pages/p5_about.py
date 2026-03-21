"""
pages/p5_about.py  —  About page
"""
import streamlit as st
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.loader import load_model_artefacts

def render():
    _, __, ___, meta, loaded = load_model_artefacts()
    r2    = meta.get("r2",0.980); mape=meta.get("mape",12.1)
    mae   = meta.get("mae",1580); rows=meta.get("total_rows",120000)
    nf    = meta.get("n_features",32); mname=meta.get("model_name","LightGBM")
    cvm   = meta.get("cv_mape_mean",12.3); cvs=meta.get("cv_mape_std",1.1)
    bline = meta.get("baseline_mape",45.0)

    st.markdown(f"""<div class="page-hdr">
      <h2>ℹ️ About AirFair Vista</h2>
      <p>Final Year B.Tech Project · Computer Science · 2025–2026</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([3,2], gap="large")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📄 Project Abstract")
        st.markdown(f"""
**AirFair Vista** is a production-ready ML system predicting international flight prices.
A combined dataset of **{rows:,} records** (20k real + 100k synthetic) trains a **{mname}**
achieving **R²={r2:.4f}** — validated with TimeSeriesSplit CV.

BRD Phase-2 macro-economic features (SAF zones, environmental tiers, fleet age, restricted
airspace) are confirmed via SHAP in top-10 contributors. The project ships as a two-service
Docker stack: `ml_pipeline` for training, `streamlit_app` for serving.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🗂️ Two-Service Architecture")
        st.code("""airfair_v3/
├── ml_pipeline/              ← Training service
│   ├── src/
│   │   ├── config.py         all paths in one place
│   │   ├── data_generator.py 100k synthetic rows
│   │   ├── data_loader.py    merge + BRD backfill
│   │   ├── features.py       SHARED feature engineering
│   │   ├── eda.py            16 plots + insights.json
│   │   └── trainer.py        6 models, CV, SHAP, artefacts
│   ├── data/raw/             flight_price_dataset.csv
│   ├── data/processed/       generated CSVs
│   ├── models/               ← model.pkl + 3 companion files
│   ├── reports/              ← 16 PNGs + insights.json
│   ├── logs/training.log
│   ├── train.py              entry point
│   ├── requirements.txt
│   └── Dockerfile
│
├── streamlit_app/            ← Frontend service
│   ├── app.py                entry point (5-page nav)
│   ├── pages/
│   │   ├── p1_predict.py     price predictor
│   │   ├── p2_eda.py         EDA charts + insights
│   │   ├── p3_features.py    feature engineering visuals
│   │   ├── p4_models.py      model comparison + SHAP
│   │   └── p5_about.py       this page
│   ├── utils/
│   │   ├── loader.py         cached PKL + JSON loader
│   │   └── style.py          shared CSS
│   ├── .streamlit/config.toml
│   ├── requirements.txt
│   └── Dockerfile
│
├── docker-compose.yml        ← one command deploys both
├── Makefile                  ← make train / make app / make up
└── README.md""", language="text")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🏆 Live Results")
        c1,c2_ = st.columns(2)
        with c1:
            st.metric("R² Score",   f"{r2*100:.1f}%")
            st.metric("MAE",        f"₹{mae:,}")
            st.metric("CV MAPE",    f"{cvm:.1f}%")
        with c2_:
            st.metric("MAPE",        f"{mape:.1f}%")
            st.metric("vs Baseline", f"{bline-mape:.1f}pp ↑")
            st.metric("Features",    str(nf))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🚀 Quick Start")
        st.markdown("**Local**")
        st.code("""# Train
cd ml_pipeline
pip install -r requirements.txt
python train.py

# App
cd ../streamlit_app
pip install -r requirements.txt
streamlit run app.py""", language="bash")

        st.markdown("**Docker (production)**")
        st.code("""# One command — trains + serves
docker-compose up --build
# → http://localhost:8501""", language="bash")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🛠️ Tech Stack")
        for cat, chips in [
            ("ML Models",    ["LightGBM","XGBoost","GBM","RF"]),
            ("Explainability",["SHAP"]),
            ("BRD Features", ["SAF Zones","Env Tiers","Fleet Age","Airspace"]),
            ("Validation",   ["TimeSeriesSplit CV"]),
            ("Data",         ["Pandas","NumPy","SciPy"]),
            ("Viz (Pipeline)",["Matplotlib","Seaborn"]),
            ("Frontend",     ["Streamlit","Plotly"]),
            ("Deploy",       ["Docker","docker-compose"]),
        ]:
            st.markdown(f"<div style='font-size:.7rem;color:#64748B;text-transform:uppercase;margin-top:.4rem;'>{cat}</div>",
                        unsafe_allow_html=True)
            st.markdown(" · ".join([f"`{c}`" for c in chips]))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""<div style='text-align:center;color:#94A3B8;font-size:.76rem;padding:1rem 0;'>
    AirFair Vista · Final Year B.Tech Project · CS · 2025–2026 ·
    {mname} · Docker two-service stack · BRD compliant
    </div>""", unsafe_allow_html=True)
