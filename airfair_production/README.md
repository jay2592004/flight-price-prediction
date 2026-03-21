# ✈ AirFair Vista — Flight Price Prediction

**Final Year B.Tech Project · Computer Science · 2025–2026**

End-to-end ML system for predicting international flight ticket prices, featuring BRD Phase 2
macro-economic factors (SAF mandates, environmental tiers, fleet age, restricted airspace),
TimeSeriesSplit cross-validation, SHAP explainability, and a Dockerised Streamlit frontend.

---

## Quick Start

### Option A — Local

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your dataset (already in data/ if copied)
#    data/flight_price_dataset.csv

# 3. Train — generates data, trains 6 models, saves best to models/
python train_model.py

# 4. Launch app
streamlit run src/frontend/app.py
# → http://localhost:8501
```

### Option B — Docker (production)

```bash
# Build and run in one command
docker-compose up --build

# → http://localhost:8501
# Container auto-trains if models/model.pkl is missing
```

### Re-train with more data

```bash
python train_model.py --rows 500000   # 500k synthetic rows
python train_model.py --force         # force re-generate data
```

---

## Project Structure

```
airfair_production/
├── train_model.py              ← Training entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .streamlit/config.toml      ← Streamlit server config
├── src/
│   ├── data/
│   │   ├── generator.py        ← Synthetic data (100k rows, 25 features)
│   │   └── preprocessor.py     ← Merge + BRD column back-fill
│   ├── pipeline/
│   │   ├── features.py         ← Feature engineering (train + runtime)
│   │   └── train.py            ← 6-model training, TimeSeriesSplit, SHAP
│   └── frontend/
│       ├── app.py              ← Streamlit app (all pages)
│       └── pages/
│           ├── loader.py       ← Cached model loader
│           ├── home.py         ← Price Predictor
│           ├── eda.py          ← EDA Dashboard
│           ├── model_comparison.py
│           └── about.py
├── models/                     ← AUTO-GENERATED — do not edit
│   ├── model.pkl
│   ├── encoders.pkl
│   ├── features.pkl
│   └── model_meta.json
├── data/
│   └── flight_price_dataset.csv
├── notebooks/
│   └── AirFair_Vista.ipynb
└── logs/
    └── training.log
```

---

## Model Performance (after training)

| Metric | Value |
|--------|-------|
| Best model | LightGBM / XGBoost (auto-selected) |
| R² Score | ~0.98 |
| MAPE | ~12% |
| MAE | ~₹1,580 |
| CV MAPE (5-fold TS) | ~12.3% ± 1.1% |
| vs Baseline (MA-30) | ~33pp improvement |

---

## BRD Phase 2 Macro-Factors

| Feature | Description | Price Impact |
|---------|-------------|--------------|
| `SAF_Zone` | 0=none, 1=voluntary, 2=EU mandatory | +2% / +6% |
| `Env_Surcharge_Tier` | 0–3 environmental levy tier | +1.5% per tier |
| `Fleet_Age_Years` | Avg fleet age per airline | +0.4% per year above 8yr |
| `Is_Restricted_Airspace` | Reroute required (0/1) | +9% |

All 4 features verified in SHAP top-10 contributors.

---

## Tech Stack

- **ML**: scikit-learn, XGBoost, LightGBM, SHAP
- **Data**: Pandas, NumPy
- **Frontend**: Streamlit
- **Visualisation**: Matplotlib, Seaborn, Plotly
- **Deploy**: Docker, docker-compose
