"""
utils/loader.py
Single cached loader for all ML artefacts and reports.
All Streamlit pages import from here — PKLs loaded once per session.
"""

import os, json
import joblib
import streamlit as st

# Path to ml_pipeline/ — resolved via env var for Docker, fallback for local
ML_ROOT     = os.environ.get(
    "ML_PIPELINE_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ml_pipeline"))
)
MODELS_DIR  = os.path.join(ML_ROOT, "models")
REPORTS_DIR = os.path.join(ML_ROOT, "reports")


@st.cache_resource(show_spinner="Loading model artefacts…")
def load_model_artefacts():
    """Returns (model, encoders, features, meta, loaded:bool)."""
    mp = os.path.join(MODELS_DIR, "model.pkl")
    ep = os.path.join(MODELS_DIR, "encoders.pkl")
    fp = os.path.join(MODELS_DIR, "features.pkl")
    if not (os.path.exists(mp) and os.path.exists(ep) and os.path.exists(fp)):
        return None, None, None, _default_meta(), False
    return (joblib.load(mp), joblib.load(ep), joblib.load(fp),
            _load_meta(os.path.join(MODELS_DIR, "model_meta.json")), True)


@st.cache_data(show_spinner=False)
def load_insights():
    """Returns insights dict from reports/insights.json."""
    p = os.path.join(REPORTS_DIR, "insights.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def get_report_path(filename: str) -> str:
    return os.path.join(REPORTS_DIR, filename)


def report_exists(filename: str) -> bool:
    return os.path.exists(get_report_path(filename))


def _load_meta(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return _default_meta()


def _default_meta():
    return {
        "model_name":"Not trained yet","mape":0.0,"r2":0.0,"mae":0,
        "baseline_mape":0.0,"cv_mape_mean":0.0,"cv_mape_std":0.0,
        "n_features":32,"train_rows":0,"test_rows":0,"total_rows":0,
        "all_models":{},"shap_top10":{},
    }
