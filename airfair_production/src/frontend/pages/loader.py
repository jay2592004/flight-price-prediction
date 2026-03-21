"""
src/frontend/pages/loader.py
Shared model loader — cached so artefacts are loaded ONCE per Streamlit session.
All pages import from here. Never import joblib directly in page files.
"""

import os, json
import joblib
import streamlit as st

MODELS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "models"
)


@st.cache_resource(show_spinner="Loading model artefacts...")
def load_artefacts():
    """
    Returns (model, encoders, features, meta, loaded:bool).
    loaded=False means models/ folder has no pkl yet — app runs in demo mode.
    """
    model_path    = os.path.join(MODELS_DIR, "model.pkl")
    encoders_path = os.path.join(MODELS_DIR, "encoders.pkl")
    features_path = os.path.join(MODELS_DIR, "features.pkl")
    meta_path     = os.path.join(MODELS_DIR, "model_meta.json")

    if not (os.path.exists(model_path) and os.path.exists(encoders_path)):
        return None, None, None, _default_meta(), False

    model    = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    features = joblib.load(features_path)
    meta     = _load_meta(meta_path)
    return model, encoders, features, meta, True


def _load_meta(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return _default_meta()


def _default_meta() -> dict:
    return {
        "model_name": "Not trained yet",
        "mape": 0.0, "r2": 0.0, "mae": 0,
        "baseline_mape": 0.0, "cv_mape_mean": 0.0, "cv_mape_std": 0.0,
        "n_features": 32, "train_rows": 0, "test_rows": 0, "total_rows": 0,
        "all_models": {},
    }
