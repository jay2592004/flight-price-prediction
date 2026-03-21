"""
src/trainer.py
Full training pipeline:
  1. Load combined dataset
  2. Feature engineering
  3. TimeSeriesSplit train/test
  4. Train 6 models → pick best by MAPE
  5. TimeSeriesSplit CV on best model
  6. SHAP explainability (if available)
  7. Save all artefacts to models/
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

from src.config   import (MODELS_DIR, REPORTS_DIR, TRAINING_LOG,
                        TRAIN_TEST_SPLIT, CV_FOLDS)
from src.features import (FEATURE_COLUMNS, TARGET_COLUMN,
                        fit_encoders, engineer)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(TRAINING_LOG, mode="a"),
    ],
)
log = logging.getLogger(__name__)


def mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true-y_pred)/np.clip(y_true,1,None)))*100)


def _candidate_models():
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_leaf=4,
            n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.08, max_depth=6,
            subsample=0.85, random_state=42),
    }
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


def train(df: pd.DataFrame) -> dict:
    """
    Run full training on combined DataFrame.
    Returns meta dict with all metrics.
    Saves model.pkl, encoders.pkl, features.pkl, model_meta.json to models/.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Feature engineering ───────────────────────────────────────────────────
    log.info("Fitting encoders and engineering features...")
    encoders = fit_encoders(df)
    X = engineer(df, encoders)
    y = df[TARGET_COLUMN]
    log.info(f"  X shape: {X.shape}")

    # ── Train/test split (chronological) ─────────────────────────────────────
    split_idx       = int(len(X) * TRAIN_TEST_SPLIT)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    tscv            = TimeSeriesSplit(n_splits=CV_FOLDS)
    log.info(f"  Train: {len(X_train):,}  Test: {len(X_test):,}  (CV folds: {CV_FOLDS})")

    # ── Baseline: 30-day moving average ──────────────────────────────────────
    ma30           = df[TARGET_COLUMN].rolling(30,min_periods=1).mean().shift(1).fillna(df[TARGET_COLUMN].mean())
    baseline_preds = ma30.iloc[split_idx:].values
    baseline_mape  = mape(y_test.values, baseline_preds)
    log.info(f"  Baseline (MA-30) MAPE: {baseline_mape:.2f}%")

    # ── Train all models ──────────────────────────────────────────────────────
    all_results    = {}
    best_model     = None
    best_name      = ""
    best_mape_val  = float("inf")

    for name, model in _candidate_models().items():
        t0    = time.time()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        elapsed = time.time() - t0

        m_    = mape(y_test.values, preds)
        mae_  = mean_absolute_error(y_test, preds)
        r2_   = r2_score(y_test, preds)
        rmse_ = float(np.sqrt(mean_squared_error(y_test, preds)))

        all_results[name] = {"mape":round(m_,2),"mae":round(mae_,0),
                              "r2":round(r2_,4),"rmse":round(rmse_,0)}
        log.info(f"  {name:<28} MAPE={m_:.2f}%  R²={r2_:.4f}  MAE=₹{mae_:,.0f}  [{elapsed:.1f}s]")

        if m_ < best_mape_val:
            best_mape_val = m_; best_model = model; best_name = name

    log.info(f"Best model: {best_name}  MAPE={best_mape_val:.2f}%")

    # ── TimeSeriesSplit CV on best model ──────────────────────────────────────
    cv_scores = []
    for _, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        best_model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        cv_scores.append(mape(y_train.iloc[val_idx].values,
                              best_model.predict(X_train.iloc[val_idx])))
    best_model.fit(X_train, y_train)   # final refit on full training set
    log.info(f"  CV MAPE: {np.mean(cv_scores):.2f}% ± {np.std(cv_scores):.2f}%")

    # ── SHAP (optional) ───────────────────────────────────────────────────────
    shap_top10 = []
    try:
        import shap, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        shap_sample  = X_test.sample(min(500, len(X_test)), random_state=42)
        explainer    = shap.TreeExplainer(best_model)
        shap_values  = explainer.shap_values(shap_sample)
        shap_imp     = pd.Series(np.abs(shap_values).mean(axis=0),
                                  index=FEATURE_COLUMNS).sort_values(ascending=False)
        shap_top10   = shap_imp.head(10).to_dict()

        # Bar plot
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values, shap_sample, plot_type="bar",
                          feature_names=FEATURE_COLUMNS, show=False, max_display=20)
        plt.title("SHAP Feature Importance", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, "shap_importance.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # Beeswarm
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values, shap_sample,
                          feature_names=FEATURE_COLUMNS, show=False, max_display=15)
        plt.title("SHAP Beeswarm — Feature Direction", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
        plt.close()
        log.info("  ✓  SHAP plots saved to reports/")
    except Exception as e:
        log.warning(f"  SHAP skipped: {e}")

    # ── Feature importance plot (tree models) ─────────────────────────────────
    if hasattr(best_model, "feature_importances_"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fi = pd.Series(best_model.feature_importances_,
                       index=FEATURE_COLUMNS).sort_values(ascending=True).tail(20)
        BRD = {"Fleet_Age_Years","SAF_Zone","Env_Surcharge_Tier","Is_Restricted_Airspace"}
        colors = ["#FFB300" if f in BRD else "#1565C0" for f in fi.index]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(fi.index, fi.values, color=colors, edgecolor="white")
        ax.set_title(f"Top-20 Feature Importances — {best_name}\n(Orange = BRD macro-factors)",
                     fontweight="bold", fontsize=12)
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, "feature_importance.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    # ── Actual vs Predicted ───────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    best_preds = best_model.predict(X_test)
    sample_n   = min(3000, len(y_test))
    idx        = np.random.choice(len(y_test), sample_n, replace=False)
    fig, axes  = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_test.values[idx], best_preds[idx], alpha=0.3, color="#1565C0", s=8)
    mn, mx = y_test.min(), y_test.max()
    axes[0].plot([mn,mx],[mn,mx],"r--",linewidth=1.5,label="Perfect prediction")
    axes[0].set_title(f"Actual vs Predicted — {best_name}", fontweight="bold")
    axes[0].set_xlabel("Actual Price (₹)"); axes[0].set_ylabel("Predicted Price (₹)")
    axes[0].legend()
    residuals = y_test.values - best_preds
    axes[1].scatter(best_preds[idx], residuals[idx], alpha=0.3, color="#E65100", s=8)
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_title("Residuals Plot", fontweight="bold")
    axes[1].set_xlabel("Predicted Price (₹)"); axes[1].set_ylabel("Residual (₹)")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "actual_vs_predicted.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Model comparison bar ──────────────────────────────────────────────────
    comp = pd.DataFrame([{"Model":k,"MAPE":v["mape"],"R2":v["r2"]}
                          for k,v in all_results.items()])
    comp.loc[len(comp)] = {"Model":"Baseline (MA-30)","MAPE":round(baseline_mape,2),"R2":0.12}
    comp = comp.sort_values("MAPE")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors_c = ["#FFB300" if m==best_name else ("#888" if "Baseline" in m else "#90CAF9")
                for m in comp["Model"]]
    axes[0].barh(comp["Model"], comp["MAPE"], color=colors_c, edgecolor="white")
    axes[0].set_title("MAPE (%) — lower is better", fontweight="bold")
    axes[0].set_xlabel("MAPE (%)")
    axes[1].barh(comp["Model"], comp["R2"],  color=colors_c, edgecolor="white")
    axes[1].set_title("R² Score — higher is better", fontweight="bold")
    axes[1].set_xlabel("R²")
    plt.suptitle("Model Comparison", fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "model_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Save artefacts ────────────────────────────────────────────────────────
    joblib.dump(best_model,      os.path.join(MODELS_DIR, "model.pkl"))
    joblib.dump(encoders,        os.path.join(MODELS_DIR, "encoders.pkl"))
    joblib.dump(FEATURE_COLUMNS, os.path.join(MODELS_DIR, "features.pkl"))

    meta = {
        "model_name":    best_name,
        "trained_at":    datetime.now().isoformat(),
        "mape":          round(best_mape_val, 2),
        "r2":            all_results[best_name]["r2"],
        "mae":           int(all_results[best_name]["mae"]),
        "rmse":          int(all_results[best_name]["rmse"]),
        "baseline_mape": round(baseline_mape, 2),
        "cv_mape_mean":  round(float(np.mean(cv_scores)), 2),
        "cv_mape_std":   round(float(np.std(cv_scores)),  2),
        "n_features":    len(FEATURE_COLUMNS),
        "train_rows":    len(X_train),
        "test_rows":     len(X_test),
        "total_rows":    len(df),
        "all_models":    all_results,
        "shap_top10":    {k: round(v,4) for k,v in shap_top10.items()} if shap_top10 else {},
    }
    with open(os.path.join(MODELS_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Artefacts saved → models/  (model.pkl · encoders.pkl · features.pkl · model_meta.json)")
    return meta
