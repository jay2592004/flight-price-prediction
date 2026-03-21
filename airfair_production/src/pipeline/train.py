"""
src/pipeline/train.py
Full training pipeline:
  1. Generate or load combined dataset
  2. Feature engineering
  3. TimeSeriesSplit train/test
  4. Train 6 models, pick best by MAPE
  5. Save best model + artefacts to models/
  6. Log all metrics
"""

import os, json, time, logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

from .features import (
    FEATURE_COLUMNS, TARGET_COLUMN, CAT_COLUMNS,
    fit_encoders, engineer,
)
from ..data.generator    import generate
from ..data.preprocessor import load_and_merge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(os.path.dirname(__file__), "..", "..", "logs", "training.log"),
            mode="a"
        ),
    ],
)
log = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100)


def _try_import_boosters():
    models = {}
    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.07, max_depth=7,
            subsample=0.85, colsample_bytree=0.85,
            n_jobs=-1, random_state=42, verbosity=0)
    except ImportError:
        log.warning("xgboost not installed — skipping")
    try:
        import lightgbm as lgb
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.07, max_depth=7,
            num_leaves=63, subsample=0.85, colsample_bytree=0.85,
            n_jobs=-1, random_state=42, verbose=-1)
    except ImportError:
        log.warning("lightgbm not installed — skipping")
    return models


def run(n_rows: int = 100_000, seed: int = 42, force_regen: bool = False):
    """
    Full training pipeline. Call from train_model.py or CLI.
    Saves artefacts to models/ and returns the metrics dict.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,   exist_ok=True)

    combined_path = os.path.join(DATA_DIR, "flight_price_combined.csv")
    orig_path     = os.path.join(DATA_DIR, "flight_price_dataset.csv")

    # ── Step 1: Data ────────────────────────────────────────────────────────
    if not os.path.exists(combined_path) or force_regen:
        log.info(f"Generating {n_rows:,} synthetic rows...")
        synth_df = generate(n_rows=n_rows, seed=seed)
        synth_df.to_csv(os.path.join(DATA_DIR, "flight_price_synthetic.csv"), index=False)
        combined = load_and_merge(orig_path, synth_df)
        combined.to_csv(combined_path, index=False)
        log.info(f"Combined dataset: {len(combined):,} rows")
    else:
        log.info(f"Loading existing combined dataset from {combined_path}")
        combined = pd.read_csv(combined_path)

    combined = combined.drop_duplicates()
    combined["Journey_Date_dt"] = pd.to_datetime(
        combined["Journey_Date"], dayfirst=True, errors="coerce")
    combined = combined.sort_values("Journey_Date_dt").reset_index(drop=True)
    log.info(f"Dataset shape after dedup+sort: {combined.shape}")

    # ── Step 2: Feature engineering ─────────────────────────────────────────
    log.info("Fitting encoders and engineering features...")
    encoders = fit_encoders(combined)
    X = engineer(combined, encoders)
    y = combined[TARGET_COLUMN]

    # ── Step 3: Time-series split ────────────────────────────────────────────
    split_idx    = int(len(X) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    tscv = TimeSeriesSplit(n_splits=5)
    log.info(f"Train: {len(X_train):,}  Test: {len(X_test):,}  (TimeSeriesSplit 5-fold)")

    # ── Step 4: Baseline (30-day MA) ─────────────────────────────────────────
    ma30 = combined["Price"].rolling(30, min_periods=1).mean().shift(1).fillna(
        combined["Price"].mean())
    baseline_preds = ma30.iloc[split_idx:].values
    baseline_mape  = mape(y_test.values, baseline_preds)
    log.info(f"Baseline (MA-30) MAPE: {baseline_mape:.2f}%")

    # ── Step 5: Model comparison ──────────────────────────────────────────────
    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=1.0),
        "Random Forest":     RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_leaf=4,
            n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.08, max_depth=6,
            subsample=0.85, random_state=42),
    }
    candidate_models.update(_try_import_boosters())

    all_results = {}
    best_model, best_name, best_mape_val = None, "", float("inf")

    for name, model in candidate_models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        elapsed = time.time() - t0

        m    = mape(y_test.values, preds)
        mae_ = mean_absolute_error(y_test, preds)
        r2_  = r2_score(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

        all_results[name] = {"mape":round(m,2),"mae":round(mae_,0),
                              "r2":round(r2_,4),"rmse":round(rmse,0)}
        log.info(f"  {name:<28}  MAPE={m:.2f}%  R²={r2_:.4f}  MAE=₹{mae_:,.0f}  [{elapsed:.1f}s]")

        if m < best_mape_val:
            best_mape_val = m
            best_model    = model
            best_name     = name

    log.info(f"Best model: {best_name}  MAPE={best_mape_val:.2f}%")
    log.info(f"Improvement vs baseline: {baseline_mape - best_mape_val:.2f}pp")

    # ── Step 6: TimeSeriesSplit CV on best model ──────────────────────────────
    cv_scores = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        best_model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        pv = best_model.predict(X_train.iloc[val_idx])
        cv_scores.append(mape(y_train.iloc[val_idx].values, pv))
    best_model.fit(X_train, y_train)   # refit on full training data

    cv_mean = float(np.mean(cv_scores))
    cv_std  = float(np.std(cv_scores))
    log.info(f"CV MAPE: {cv_mean:.2f}% ± {cv_std:.2f}%")

    # ── Step 7: Save artefacts to models/ ────────────────────────────────────
    joblib.dump(best_model,             os.path.join(MODELS_DIR, "model.pkl"))
    joblib.dump(encoders,               os.path.join(MODELS_DIR, "encoders.pkl"))
    joblib.dump(FEATURE_COLUMNS,        os.path.join(MODELS_DIR, "features.pkl"))

    meta = {
        "model_name":    best_name,
        "trained_at":    datetime.now().isoformat(),
        "mape":          round(best_mape_val, 2),
        "r2":            all_results[best_name]["r2"],
        "mae":           int(all_results[best_name]["mae"]),
        "rmse":          int(all_results[best_name]["rmse"]),
        "baseline_mape": round(baseline_mape, 2),
        "cv_mape_mean":  round(cv_mean, 2),
        "cv_mape_std":   round(cv_std, 2),
        "n_features":    len(FEATURE_COLUMNS),
        "train_rows":    len(X_train),
        "test_rows":     len(X_test),
        "total_rows":    len(combined),
        "all_models":    all_results,
    }
    with open(os.path.join(MODELS_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Artefacts saved to models/")
    log.info(f"  model.pkl · encoders.pkl · features.pkl · model_meta.json")
    return meta
