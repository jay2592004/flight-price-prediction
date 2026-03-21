"""
tests/test_pipeline.py
Integration tests for src/pipeline/train.py
Runs on a tiny 500-row dataset so the CI stays fast (<60s).
"""

import pytest
import os, json
import numpy as np
import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


@pytest.fixture(scope="module")
def trained_meta(tmp_path_factory):
    """
    Runs the full training pipeline on 500 rows into a temp directory.
    Returns the meta dict. Runs once per test module.
    """
    import shutil
    from src.pipeline import train as train_mod

    # Redirect models/ and data/ to tmp dirs for test isolation
    tmp = tmp_path_factory.mktemp("pipeline")
    orig_models = train_mod.MODELS_DIR
    orig_data   = train_mod.DATA_DIR

    train_mod.MODELS_DIR = str(tmp / "models")
    train_mod.DATA_DIR   = str(tmp / "data")
    os.makedirs(train_mod.MODELS_DIR, exist_ok=True)
    os.makedirs(train_mod.DATA_DIR,   exist_ok=True)

    # Copy original dataset if available
    orig_csv = os.path.join(os.path.dirname(__file__), "..", "data", "flight_price_dataset.csv")
    if os.path.exists(orig_csv):
        shutil.copy(orig_csv, os.path.join(train_mod.DATA_DIR, "flight_price_dataset.csv"))

    meta = train_mod.run(n_rows=500, seed=42, force_regen=True)

    # Restore paths
    train_mod.MODELS_DIR = orig_models
    train_mod.DATA_DIR   = orig_data

    return meta, str(tmp)


def test_pipeline_returns_meta(trained_meta):
    meta, _ = trained_meta
    assert isinstance(meta, dict)


def test_meta_has_required_keys(trained_meta):
    meta, _ = trained_meta
    required = ["model_name","mape","r2","mae","rmse",
                "baseline_mape","cv_mape_mean","cv_mape_std",
                "n_features","train_rows","test_rows","total_rows","all_models"]
    for key in required:
        assert key in meta, f"Missing meta key: {key}"


def test_r2_is_positive(trained_meta):
    meta, _ = trained_meta
    assert meta["r2"] > 0, "R² should be positive"


def test_mape_below_baseline(trained_meta):
    meta, _ = trained_meta
    assert meta["mape"] < meta["baseline_mape"], \
        f"Model MAPE ({meta['mape']}) should beat baseline ({meta['baseline_mape']})"


def test_mae_positive(trained_meta):
    meta, _ = trained_meta
    assert meta["mae"] > 0


def test_n_features_correct(trained_meta):
    meta, _ = trained_meta
    assert meta["n_features"] == 32


def test_artefacts_saved(trained_meta):
    _, tmp = trained_meta
    models_dir = os.path.join(tmp, "models")
    for fname in ["model.pkl", "encoders.pkl", "features.pkl", "model_meta.json"]:
        path = os.path.join(models_dir, fname)
        assert os.path.exists(path), f"Missing artefact: {fname}"


def test_model_pkl_loadable(trained_meta):
    import joblib
    _, tmp = trained_meta
    model = joblib.load(os.path.join(tmp, "models", "model.pkl"))
    assert hasattr(model, "predict")


def test_encoders_pkl_loadable(trained_meta):
    import joblib
    _, tmp = trained_meta
    encoders = joblib.load(os.path.join(tmp, "models", "encoders.pkl"))
    assert isinstance(encoders, dict)
    assert len(encoders) > 0


def test_features_pkl_loadable(trained_meta):
    import joblib
    _, tmp = trained_meta
    features = joblib.load(os.path.join(tmp, "models", "features.pkl"))
    assert isinstance(features, list)
    assert len(features) == 32


def test_model_meta_json_valid(trained_meta):
    _, tmp = trained_meta
    with open(os.path.join(tmp, "models", "model_meta.json")) as f:
        meta = json.load(f)
    assert "model_name" in meta
    assert "mape" in meta


def test_end_to_end_single_prediction(trained_meta, sample_input_dict):
    """Load saved artefacts and make one prediction — full runtime path."""
    import joblib
    from src.pipeline.features import build_single_row

    _, tmp = trained_meta
    model    = joblib.load(os.path.join(tmp, "models", "model.pkl"))
    encoders = joblib.load(os.path.join(tmp, "models", "encoders.pkl"))
    features = joblib.load(os.path.join(tmp, "models", "features.pkl"))

    X = build_single_row(sample_input_dict, encoders)
    assert list(X.columns) == features

    pred = model.predict(X)[0]
    assert isinstance(float(pred), float)
    assert pred > 0, "Prediction must be positive"
    assert pred < 500_000, "Prediction unreasonably large"


def test_all_models_compared(trained_meta):
    meta, _ = trained_meta
    # At minimum Linear, Ridge, RF, GBM should always run
    always_present = ["Linear Regression", "Ridge Regression",
                      "Random Forest", "Gradient Boosting"]
    for name in always_present:
        assert name in meta["all_models"], f"Model missing from comparison: {name}"


def test_best_model_is_in_all_models(trained_meta):
    meta, _ = trained_meta
    assert meta["model_name"] in meta["all_models"]


def test_cv_scores_reasonable(trained_meta):
    meta, _ = trained_meta
    assert 0 < meta["cv_mape_mean"] < 100
    assert meta["cv_mape_std"] >= 0
