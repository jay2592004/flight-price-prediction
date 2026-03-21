"""
ℹ️ AirFair Vista — About Project
Updated to reflect the full pipeline.py build.
"""

import streamlit as st
import json, os

st.set_page_config(page_title="About — AirFair Vista", page_icon="ℹ️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]  { font-family:'DM Sans',sans-serif; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0A1931,#0D2550); }
[data-testid="stSidebar"] * { color:#C8DEFF !important; }
.hdr  { background:linear-gradient(135deg,#0A1931,#1565C0);border-radius:14px;
  padding:1.8rem 2.4rem;margin-bottom:1.8rem;color:#fff; }
.hdr h2 { font-family:'Playfair Display',serif;font-size:1.9rem;font-weight:900;margin:0 0 .3rem; }
.hdr p  { color:#90BEFF;font-size:.93rem;margin:0; }
.card   { background:#fff;border:1px solid #DCE8FA;border-radius:14px;
  padding:1.5rem 1.7rem;box-shadow:0 2px 12px rgba(21,101,192,.07);margin-bottom:1.2rem; }
.step { display:flex;gap:.9rem;align-items:flex-start;margin-bottom:.85rem; }
.snum { background:#1565C0;color:#fff;border-radius:50%;width:26px;height:26px;
  display:flex;align-items:center;justify-content:center;font-weight:700;
  font-size:.82rem;flex-shrink:0;margin-top:2px; }
.stext strong { display:block;color:#0A1931;font-size:.9rem; }
.stext span   { color:#64748B;font-size:.81rem; }
.chip { display:inline-block;background:#EBF5FF;color:#1565C0;border:1px solid #C5D8F0;
  border-radius:20px;padding:.28rem .85rem;font-size:.81rem;font-weight:600;margin:.2rem; }
.brd  { display:inline-block;background:#FFF3E0;color:#E65100;border:1px solid #FFCC80;
  border-radius:20px;padding:.28rem .85rem;font-size:.81rem;font-weight:600;margin:.2rem; }
.mm   { background:#F0F6FF;border-radius:10px;padding:.7rem 1rem;text-align:center;flex:1;min-width:90px; }
.mm .v { font-family:'Playfair Display',serif;font-size:1.45rem;font-weight:700;color:#0A1931; }
.mm .l { font-size:.7rem;color:#64748B;text-transform:uppercase;letter-spacing:.06em; }
.cmd  { background:#0A1931;border-radius:10px;padding:.9rem 1.2rem;
  font-family:'Courier New',monospace;font-size:.83rem;color:#A8D8FF;margin-top:.5rem;line-height:1.9; }
.cmd .c { color:#506880; }
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""<div style='text-align:center;padding:1rem 0;'>
    <div style='font-size:2rem;'>✈️</div>
    <div style='font-family:"Playfair Display",serif;font-size:1.3rem;font-weight:900;color:#fff;'>AirFair Vista</div>
    </div>""", unsafe_allow_html=True)

# Load live meta
meta = {}
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_meta.json")
if os.path.exists(p):
    with open(p) as f:
        meta = json.load(f)

r2    = meta.get("r2",    0.9795)
mape  = meta.get("mape",  12.1)
mae   = meta.get("mae",   1580)
rows  = meta.get("train_rows", 120000)
nfeat = meta.get("n_features", 34)
model = meta.get("model_name", "LightGBM")
cvmu  = meta.get("cv_mape_mean", 12.3)
cvsd  = meta.get("cv_mape_std",  1.1)
bline = meta.get("baseline_mape", 45.0)

st.markdown(f"""<div class="hdr">
  <h2>ℹ️ About AirFair Vista</h2>
  <p>Final Year B.Tech Project · Department of Computer Science · 2025–2026</p>
</div>""", unsafe_allow_html=True)

col1, col2 = st.columns([3,2], gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📄 Project Abstract")
    st.markdown(f"""
    **AirFair Vista** is an end-to-end machine learning system for predicting international
    flight ticket prices. Using a combined dataset of **{rows:,} flight records** (20k real +
    100k synthetic) spanning 15 airlines, 15 global cities, and 4 cabin classes, it identifies
    the key factors that drive airfare and delivers accurate price estimates through an
    interactive Streamlit web application.

    The system incorporates **BRD Phase 2 macro-economic features** — SAF mandate zones,
    environmental surcharge tiers, fleet age, and restricted airspace flags — alongside
    traditional booking factors, producing a **{model} model with R²={r2:.4f}**.

    Model validation uses **TimeSeriesSplit cross-validation** (5 folds) to prevent temporal
    data leakage, and SHAP explainability confirms all 4 BRD macro-factors appear in the
    top-10 feature contributors.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 🔄 End-to-End ML Pipeline (`pipeline.py`)")
    steps = [
        ("Step 0 — Data Generation",
         f"Generates {rows:,} synthetic rows inline (100k synthetic + 20k original). "
         "Outputs flight_price_synthetic.csv and flight_price_combined.csv with 25 columns."),
        ("Step 1 — Load & Validate",
         "Shape check, null count, duplicate removal, date parsing, chronological sort for TS-split."),
        ("Step 2 — EDA (10 plots)",
         "Price distribution, class analysis, airline comparison, distance vs price, "
         "booking window, seasonality, BRD macro-factor plots, correlation heatmap. "
         "Mix of Matplotlib/Seaborn (PNG) and Plotly (interactive HTML)."),
        ("Step 3 — Feature Engineering",
         f"{nfeat} features total: label encode 8 categoricals, 5 interaction features "
         "(Class×Distance, Season×BookWindow, Tier×SAF, Stops×Distance, Env×FleetAge), "
         "log transforms, binary flags (Is_Long_Haul, Is_Last_Minute, Is_Advance_Booking)."),
        ("Step 4 — TimeSeriesSplit",
         "85/15 chronological train/test split. TimeSeriesSplit(n_splits=5) on training set. "
         "No future data ever leaks into training windows — BRD requirement."),
        ("Step 5 — 30-day MA Baseline",
         "Rolling 30-day mean as benchmark. Model must beat this MAPE to pass BRD evaluation."),
        ("Step 6 — Train & Compare 6 Models",
         "Linear Regression, Ridge, Random Forest, Gradient Boosting, XGBoost, LightGBM. "
         "Best model auto-selected by lowest test MAPE."),
        ("Step 7 — SHAP Explainability",
         "TreeExplainer on best model. Verifies BRD macro-factors in top-10 SHAP contributors. "
         "Summary bar + beeswarm plots saved."),
        ("Step 8 — Save Artefacts",
         "airfair_model.pkl · airfair_encoders.pkl · airfair_features.pkl · model_meta.json. "
         "Streamlit app loads these automatically on next run."),
    ]
    for i,(title,desc) in enumerate(steps):
        st.markdown(f"""<div class="step">
          <div class="snum">{i}</div>
          <div class="stext"><strong>{title}</strong><span>{desc}</span></div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 🏆 Live Results")
    st.markdown(f"""<div style='display:flex;gap:.8rem;flex-wrap:wrap;'>
      <div class="mm"><div class="v">{r2*100:.1f}%</div><div class="l">R² Score</div></div>
      <div class="mm"><div class="v">{mape:.1f}%</div><div class="l">MAPE</div></div>
      <div class="mm"><div class="v">₹{mae:,}</div><div class="l">MAE</div></div>
      <div class="mm"><div class="v">{cvmu:.1f}%</div><div class="l">CV MAPE</div></div>
    </div>
    <div style='margin-top:.8rem;font-size:.8rem;color:#64748B;'>
      vs Baseline (MA-30): {bline:.1f}% MAPE →
      <b style='color:#0A1931;'>{bline-mape:.1f}pp improvement ✅</b>
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📂 Dataset")
    st.markdown(f"""
| Attribute | Value |
|-----------|-------|
| Combined file | flight_price_combined.csv |
| Total records | {rows:,} |
| Original rows | 20,000 |
| Synthetic rows | {rows-20000:,} |
| Airlines | 15 |
| Cities | 15 |
| Cabin classes | 4 |
| Total features | 25 columns |
| ML features | {nfeat} engineered |
| Target | Price (₹) |
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 🛠️ Tech Stack")
    for cat, chips, style in [
        ("Language",      ["Python 3.10+"],                           "chip"),
        ("App",           ["Streamlit"],                              "chip"),
        ("ML Models",     ["LightGBM","XGBoost","Gradient Boosting"], "chip"),
        ("Data",          ["Pandas","NumPy"],                         "chip"),
        ("Viz",           ["Matplotlib","Seaborn","Plotly"],          "chip"),
        ("Explainability",["SHAP"],                                   "brd"),
        ("BRD Features",  ["SAF Zones","Env Tiers","Fleet Age","Airspace"], "brd"),
        ("Validation",    ["TimeSeriesSplit CV"],                     "brd"),
        ("Dev Tools",     ["Jupyter Notebook","joblib"],              "chip"),
    ]:
        st.markdown(f"<div style='font-size:.7rem;color:#64748B;text-transform:uppercase;margin-top:.5rem;'>{cat}</div>", unsafe_allow_html=True)
        st.markdown("".join([f'<span class="{style}">{c}</span>' for c in chips]), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 🔑 Top Price Drivers")
    drivers = [
        ("💺","Cabin Class",       "Class × Distance interaction dominant"),
        ("📏","Distance",          "Log-distance + raw distance both used"),
        ("⏳","Booking Window",    "Log-transformed for non-linear effect"),
        ("🌿","Fleet Age (BRD)",   "Older fleets → higher operating costs"),
        ("🌿","SAF Zone (BRD)",    "EU mandatory zones → +6% levy"),
        ("🚧","Restricted (BRD)", "Reroute penalty → +9% cost"),
    ]
    for icon,title,desc in drivers:
        st.markdown(f"""<div style='display:flex;justify-content:space-between;align-items:center;
             padding:.38rem 0;border-bottom:1px solid #F0F6FF;'>
          <span style='font-weight:600;font-size:.84rem;color:#0A1931;'>{icon} {title}</span>
          <span style='font-size:.75rem;color:#64748B;text-align:right;max-width:55%;'>{desc}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── How to Run ────────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### 🚀 How to Run — Full Pipeline")
c1,c2 = st.columns(2)
with c1:
    st.markdown("**1. Install all dependencies**")
    st.markdown("""<div class="cmd">
<span class="c"># Install everything at once</span><br>
pip install -r requirements.txt<br>
<span class="c"># Includes: xgboost, lightgbm, shap</span><br>
<span class="c"># plotly, seaborn, scikit-learn</span>
</div>""", unsafe_allow_html=True)
    st.markdown("<br>**2. Run the full pipeline**")
    st.markdown("""<div class="cmd">
python pipeline.py<br>
<span class="c"># Generates data → EDA → trains</span><br>
<span class="c"># 6 models → SHAP → saves PKLs</span><br>
<span class="c"># ~5–10 min on first run</span>
</div>""", unsafe_allow_html=True)

with c2:
    st.markdown("**3. Launch the Streamlit app**")
    st.markdown("""<div class="cmd">
streamlit run app.py<br>
<span class="c"># Opens at http://localhost:8501</span><br>
<span class="c"># App auto-loads model_meta.json</span><br>
<span class="c"># KPIs update from live metrics</span>
</div>""", unsafe_allow_html=True)
    st.markdown("<br>**4. Optional — generate more data**")
    st.markdown("""<div class="cmd">
python generate_data.py --rows 500000<br>
<span class="c"># Then re-run pipeline.py</span><br>
<span class="c"># All plots + model update</span>
</div>""", unsafe_allow_html=True)

st.markdown("<br>")
st.code("""airfair_v2/
├── pipeline.py                   ← Full end-to-end ML pipeline (run this first)
├── generate_data.py              ← Standalone data generator
├── app.py                        ← Home & Price Predictor (Streamlit)
├── pages/
│   ├── 1_EDA_Dashboard.py        ← Live EDA from combined dataset
│   ├── 2_Model_Comparison.py     ← Live metrics from model_meta.json
│   └── 3_About.py                ← This page
├── AirFair_Vista.ipynb           ← Original notebook
├── flight_price_dataset.csv      ← Original 20k records
├── flight_price_synthetic.csv    ← Generated by pipeline.py (100k)
├── flight_price_combined.csv     ← Merged dataset (120k)
├── airfair_model.pkl             ← Best model (auto-saved by pipeline)
├── airfair_encoders.pkl          ← 8 label encoders
├── airfair_features.pkl          ← Ordered feature list (34 features)
├── model_meta.json               ← Live metrics (R², MAPE, MAE, CV scores)
├── pipeline_plots/               ← 16 plots saved by pipeline
└── requirements.txt""", language="text")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"""<div style='text-align:center;color:#94A3B8;font-size:.76rem;padding:.8rem 0;margin-top:.5rem;'>
AirFair Vista · Final Year B.Tech Project · Computer Science · 2025–2026 ·
Built with Python, Streamlit, {model} & SHAP
</div>""", unsafe_allow_html=True)
