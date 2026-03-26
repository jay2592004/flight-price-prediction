# ✈ AirFair Vista — Flight Price Prediction

**Final Year B.Tech Project · Computer Science · 2025–2026**

End-to-end flight price prediction system with a clean two-service architecture:
`ml_pipeline` handles all data generation, EDA, and model training.
`streamlit_app` is a pure frontend that reads the trained artefacts at runtime.

---

## Architecture

```
airfair_v3/
├── ml_pipeline/          ← Training service
│   ├── src/
│   │   ├── config.py         all paths & constants in one place
│   │   ├── data_generator.py 100k synthetic rows, 25 features
│   │   ├── data_loader.py    load raw + merge + BRD backfill
│   │   ├── features.py       ★ SHARED — encoding + build_single_row
│   │   ├── eda.py            16 EDA plots + insights.json → reports/
│   │   └── trainer.py        6 models, TimeSeriesSplit, SHAP, artefacts
│   ├── data/raw/             flight_price_dataset.csv
│   ├── data/processed/       generated CSVs (gitignored)
│   ├── models/               model.pkl · encoders.pkl · features.pkl · model_meta.json
│   ├── reports/              16 PNG plots + insights.json (read by Streamlit)
│   ├── logs/training.log
│   ├── train.py              ← entry point
│   ├── requirements.txt
│   └── Dockerfile
│
├── streamlit_app/        ← Frontend service (reads models/ and reports/)
│   ├── app.py                ← entry point — 5-page sidebar nav
│   ├── pages/
│   │   ├── p1_predict.py     Price Predictor
│   │   ├── p2_eda.py         EDA & Insights (shows all 16 plots + text)
│   │   ├── p3_features.py    Feature Engineering (visual explanations)
│   │   ├── p4_models.py      Model Comparison + SHAP plots
│   │   └── p5_about.py       About & Quick Start
│   ├── utils/
│   │   ├── loader.py         @st.cache_resource — PKLs loaded once per session
│   │   └── style.py          shared CSS for all pages
│   ├── .streamlit/config.toml
│   ├── requirements.txt
│   └── Dockerfile
│
├── docker-compose.yml    ← one command deploys both services
├── Makefile              ← make train / make app / make up
├── .gitignore
└── README.md
```

---

## Quick Start

### Option A — Local (recommended for development)

```bash
cd airfair_v3

# 1. Install all dependencies
make install
# or separately:
# cd ml_pipeline  && pip install -r requirements.txt
# cd streamlit_app && pip install -r requirements.txt

# 2. Train — generates data, 16 EDA plots, trains 6 models, saves best
make train
# (first run ~5–8 min; subsequent runs faster if data already exists)

# 3. Launch the app
make app
# → http://localhost:8501
```

### Windows PowerShell

Windows does not include `make` by default. From the project root, use the bundled
PowerShell task runner instead:

```powershell
cd airfair_v3

# 1. Install all dependencies
.\tasks.ps1 install

# 2. Train locally
.\tasks.ps1 train

# 3. Launch the app
.\tasks.ps1 app
```

It supports the same targets as the `Makefile`, including `train-large`,
`train-force`, `train-skip-eda`, `up`, `down`, and `retrain`.

### Option B — Docker (production)

```bash
cd airfair_v3

# Build + train + serve — one command
make up
# or: docker-compose up --build
# → http://localhost:8501

# The ml_pipeline container trains first, then exits.
# The streamlit_app container waits for training to finish, then serves.
```

### Re-train with more data

```bash
make train-large          # 500k rows
make train-force          # force re-generate data + retrain
make train-skip-eda       # retrain without re-running EDA (faster)

# Docker re-train + restart app:
make retrain
```

---

## How the Two Services Connect

```
ml_pipeline/train.py
    ↓ writes
ml_pipeline/models/       model.pkl · encoders.pkl · features.pkl · model_meta.json
ml_pipeline/reports/      01_price_distribution.png … 16_feature_importance_eda.png
                          insights.json
    ↑ mounts as read-only volume
streamlit_app/utils/loader.py
    → loads PKLs once via @st.cache_resource
    → loads insights.json via @st.cache_data
    → pages read from loader — never import ml_pipeline directly
```

The shared volume (`ml_pipeline/models/` and `ml_pipeline/reports/`) is the **only
coupling point** between the two services. Retrain anytime — restart the Streamlit
container and it picks up the new model automatically.

The only exception is `ml_pipeline/src/features.py` which is **imported at prediction
time** by `p1_predict.py` via `ML_PIPELINE_PATH` env var. This ensures training-serving
feature parity — the exact same `build_single_row()` that was used during training
is called at inference time, eliminating training-serving skew.

---

## Streamlit Pages

| Page | What it shows |
|------|--------------|
| 🏠 Predict Price | Live predictor — fill form → get ₹ estimate with tips |
| 📊 EDA & Insights | All 16 EDA plots from ml_pipeline/reports/ with text insights |
| ⚙️ Feature Engineering | Visual explanation of every engineered feature |
| 🤖 Model Comparison | Model metrics, actual vs predicted, SHAP plots |
| ℹ️ About | Architecture, tech stack, quick-start commands |

---

## Model Performance

| Metric | Value |
|--------|-------|
| Best model | LightGBM (auto-selected by lowest MAPE) |
| R² Score | ~0.980 |
| MAPE | ~12.1% |
| MAE | ~₹1,580 |
| CV MAPE (5-fold TS) | ~12.3% ± 1.1% |
| Baseline (MA-30) | ~45.0% → **32.9pp improvement** |

---

## BRD Phase-2 Macro-Factors

| Feature | Values | Price Effect | Verified by SHAP |
|---------|--------|-------------|-----------------|
| `SAF_Zone` | 0/1/2 | +0% / +2% / +6% | ✅ |
| `Env_Surcharge_Tier` | 0/1/2/3 | +0–4.5% | ✅ |
| `Fleet_Age_Years` | 3–25 yr | +0.4%/yr above 8yr | ✅ |
| `Is_Restricted_Airspace` | 0/1 | +9% | ✅ |

---

## EDA Reports Generated

16 plots saved to `ml_pipeline/reports/` after training:

```
01_price_distribution.png      09_correlation_heatmap.png
02_price_by_class.png          10_availability_fuel.png
03_price_by_airline.png        11_source_destination.png
04_price_vs_distance.png       12_class_distance_heatmap.png
05_booking_window.png          13_layover_aircraft.png
06_seasonality.png             14_price_over_time.png
07_stops_analysis.png          15_outlier_analysis.png
08_brd_macrofactors.png        16_feature_importance_eda.png
insights.json                  (auto-generated text insights)
```

Plus training artefacts:
```
model_comparison.png           actual_vs_predicted.png
feature_importance.png         shap_importance.png
shap_beeswarm.png
```

---

## Tech Stack

- **ML**: scikit-learn, XGBoost, LightGBM, SHAP
- **Data**: Pandas, NumPy, SciPy
- **Visualisation (pipeline)**: Matplotlib, Seaborn
- **Frontend**: Streamlit, Plotly
- **Deploy**: Docker, docker-compose
- **Validation**: TimeSeriesSplit CV (BRD requirement)
