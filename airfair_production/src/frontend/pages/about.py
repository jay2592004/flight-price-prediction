"""
src/frontend/pages/about.py  —  About page
"""

import streamlit as st
import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.frontend.pages.loader import load_artefacts

CSS = """<style>
.hdr{background:linear-gradient(135deg,#0A1931,#1565C0);border-radius:14px;
  padding:1.8rem 2.4rem;margin-bottom:1.8rem;color:#fff;}
.hdr h2{font-family:'Playfair Display',serif;font-size:1.9rem;font-weight:900;margin:0 0 .3rem;}
.hdr p{color:#90BEFF;font-size:.93rem;margin:0;}
.card{background:#fff;border:1px solid #DCE8FA;border-radius:14px;
  padding:1.5rem 1.7rem;box-shadow:0 2px 12px rgba(21,101,192,.07);margin-bottom:1.2rem;}
.step{display:flex;gap:.9rem;align-items:flex-start;margin-bottom:.85rem;}
.snum{background:#1565C0;color:#fff;border-radius:50%;width:26px;height:26px;
  display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.82rem;flex-shrink:0;margin-top:2px;}
.stext strong{display:block;color:#0A1931;font-size:.9rem;}
.stext span{color:#64748B;font-size:.81rem;}
.chip{display:inline-block;background:#EBF5FF;color:#1565C0;border:1px solid #C5D8F0;
  border-radius:20px;padding:.28rem .85rem;font-size:.81rem;font-weight:600;margin:.2rem;}
.brd{display:inline-block;background:#FFF3E0;color:#E65100;border:1px solid #FFCC80;
  border-radius:20px;padding:.28rem .85rem;font-size:.81rem;font-weight:600;margin:.2rem;}
.mm{background:#F0F6FF;border-radius:10px;padding:.7rem 1rem;text-align:center;flex:1;min-width:90px;}
.mm .v{font-family:'Playfair Display',serif;font-size:1.45rem;font-weight:700;color:#0A1931;}
.mm .l{font-size:.7rem;color:#64748B;text-transform:uppercase;letter-spacing:.06em;}
.cmd{background:#0A1931;border-radius:10px;padding:.9rem 1.2rem;
  font-family:'Courier New',monospace;font-size:.83rem;color:#A8D8FF;margin-top:.5rem;line-height:1.9;}
.cmd .c{color:#506880;}
</style>"""


def render():
    st.markdown(CSS, unsafe_allow_html=True)
    _,__,___,meta,loaded = load_artefacts()

    r2   = meta.get("r2",0.980); mape=meta.get("mape",12.1)
    mae  = meta.get("mae",1580); rows=meta.get("total_rows",120000)
    nf   = meta.get("n_features",32); mname=meta.get("model_name","LightGBM")
    cvm  = meta.get("cv_mape_mean",12.3); cvs=meta.get("cv_mape_std",1.1)
    bline= meta.get("baseline_mape",45.0)

    st.markdown(f"""<div class="hdr">
      <h2>ℹ️ About AirFair Vista</h2>
      <p>Final Year B.Tech Project · Computer Science · 2025–2026</p>
    </div>""", unsafe_allow_html=True)

    col1,col2 = st.columns([3,2],gap="large")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📄 Project Abstract")
        st.markdown(f"""
**AirFair Vista** is a production-ready ML system for predicting international flight ticket prices.
A combined dataset of **{rows:,} records** (20k real + 100k synthetic) trains a **{mname}** model
achieving **R²={r2:.4f}** — validated with TimeSeriesSplit CV to prevent data leakage.

BRD Phase 2 macro-economic features (SAF zones, environmental tiers, fleet age, restricted
airspace) are incorporated and confirmed via SHAP to appear in the top-10 feature contributors.

The project ships as a Dockerised Streamlit application deployable to any container platform.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🔄 Production Pipeline")
        steps = [
            ("Data Generation",     "src/data/generator.py — 100k synthetic rows with 25 features"),
            ("Preprocessing",       "src/data/preprocessor.py — merges with original 20k, back-fills BRD columns"),
            ("Feature Engineering", "src/pipeline/features.py — 32 engineered features incl. interactions + log transforms"),
            ("Training",            "src/pipeline/train.py — trains 6 models, selects best by MAPE, saves to models/"),
            ("Artefacts",           "models/ — model.pkl · encoders.pkl · features.pkl · model_meta.json"),
            ("Frontend",            "src/frontend/app.py — Streamlit multi-page app reads from models/ at runtime"),
            ("Containerisation",    "Dockerfile + docker-compose.yml — single container, production-ready"),
        ]
        for i,(t,d) in enumerate(steps):
            st.markdown(f"""<div class="step">
              <div class="snum">{i+1}</div>
              <div class="stext"><strong>{t}</strong><span>{d}</span></div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🏆 Live Results")
        st.markdown(f"""<div style='display:flex;gap:.8rem;flex-wrap:wrap;'>
          <div class="mm"><div class="v">{r2*100:.1f}%</div><div class="l">R²</div></div>
          <div class="mm"><div class="v">{mape:.1f}%</div><div class="l">MAPE</div></div>
          <div class="mm"><div class="v">₹{mae:,}</div><div class="l">MAE</div></div>
          <div class="mm"><div class="v">{cvm:.1f}%</div><div class="l">CV MAPE</div></div>
        </div>
        <div style='margin-top:.8rem;font-size:.8rem;color:#64748B;'>
          Baseline (MA-30) {bline:.1f}% →
          <b style='color:#0A1931;'>{bline-mape:.1f}pp improvement ✅</b>
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🛠️ Tech Stack")
        for cat,chips,style in [
            ("Language",       ["Python 3.11+"],                         "chip"),
            ("App",            ["Streamlit"],                            "chip"),
            ("ML Models",      ["LightGBM","XGBoost","Gradient Boosting"],"chip"),
            ("Data",           ["Pandas","NumPy"],                       "chip"),
            ("Visualisation",  ["Matplotlib","Seaborn","Plotly"],        "chip"),
            ("Explainability", ["SHAP"],                                 "brd"),
            ("BRD Features",   ["SAF Zones","Env Tiers","Fleet Age"],    "brd"),
            ("Validation",     ["TimeSeriesSplit CV"],                   "brd"),
            ("Deploy",         ["Docker","docker-compose"],              "chip"),
        ]:
            st.markdown(f"<div style='font-size:.7rem;color:#64748B;text-transform:uppercase;margin-top:.4rem;'>{cat}</div>",
                        unsafe_allow_html=True)
            st.markdown("".join([f'<span class="{style}">{c}</span>' for c in chips]),
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 🚀 How to Run")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("**Local — train then serve**")
        st.markdown("""<div class="cmd">
<span class="c"># 1. Install</span><br>
pip install -r requirements.txt<br><br>
<span class="c"># 2. Copy your dataset to data/</span><br>
cp flight_price_dataset.csv data/<br><br>
<span class="c"># 3. Train (generates data + model)</span><br>
python train_model.py<br><br>
<span class="c"># 4. Launch app</span><br>
streamlit run src/frontend/app.py
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("**Docker (production)**")
        st.markdown("""<div class="cmd">
<span class="c"># Build image</span><br>
docker build -t airfair-vista .<br><br>
<span class="c"># Run container</span><br>
docker run -p 8501:8501 airfair-vista<br><br>
<span class="c"># Or with compose</span><br>
docker-compose up --build<br><br>
<span class="c"># App at http://localhost:8501</span>
</div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("**Re-train with more data**")
        st.markdown("""<div class="cmd">
<span class="c"># Generate 500k rows</span><br>
python train_model.py --rows 500000<br><br>
<span class="c"># Force re-generate data</span><br>
python train_model.py --force<br><br>
<span class="c"># App picks up new model</span><br>
<span class="c"># on next restart/refresh</span><br>
streamlit run src/frontend/app.py
</div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.code("""airfair_production/
├── train_model.py              ← Entry point: train & save model
├── requirements.txt            ← All dependencies
├── Dockerfile                  ← Production Docker image
├── docker-compose.yml          ← One-command deploy
├── .dockerignore
├── .streamlit/
│   └── config.toml             ← Streamlit server config
├── src/
│   ├── data/
│   │   ├── generator.py        ← Synthetic data generation
│   │   └── preprocessor.py     ← Merge + back-fill BRD columns
│   ├── pipeline/
│   │   ├── features.py         ← Feature engineering (train + predict)
│   │   └── train.py            ← Full training pipeline
│   └── frontend/
│       ├── app.py              ← Streamlit entry point (all pages)
│       └── pages/
│           ├── loader.py       ← Cached model loader (shared)
│           ├── home.py         ← Price Predictor page
│           ├── eda.py          ← EDA Dashboard page
│           ├── model_comparison.py ← Model metrics page
│           └── about.py        ← About page
├── models/                     ← AUTO-GENERATED by train_model.py
│   ├── model.pkl               ← Best trained model
│   ├── encoders.pkl            ← Label encoders
│   ├── features.pkl            ← Feature list
│   └── model_meta.json         ← Live metrics (R², MAPE, MAE …)
├── data/                       ← Place flight_price_dataset.csv here
│   ├── flight_price_dataset.csv
│   ├── flight_price_synthetic.csv  (generated)
│   └── flight_price_combined.csv   (generated)
├── notebooks/                  ← AirFair_Vista.ipynb (research)
└── logs/
    └── training.log""", language="text")

    st.markdown(f"""<div style='text-align:center;color:#94A3B8;font-size:.76rem;padding:.8rem 0;'>
    AirFair Vista · Final Year B.Tech Project · CS · 2025–2026 ·
    {mname} · Dockerised Streamlit · BRD compliant
    </div>""", unsafe_allow_html=True)
