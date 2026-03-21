"""
src/config.py
Single place for every path and constant used across the pipeline.
Import this in every other module instead of hardcoding paths.
"""

import os

# ── Root of ml_pipeline/ ──────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Directories ───────────────────────────────────────────────
DATA_RAW_DIR       = os.path.join(ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR         = os.path.join(ROOT, "models")
REPORTS_DIR        = os.path.join(ROOT, "reports")
LOGS_DIR           = os.path.join(ROOT, "logs")

# ── File paths ────────────────────────────────────────────────
RAW_CSV            = os.path.join(DATA_RAW_DIR,       "flight_price_dataset.csv")
SYNTHETIC_CSV      = os.path.join(DATA_PROCESSED_DIR, "flight_price_synthetic.csv")
COMBINED_CSV       = os.path.join(DATA_PROCESSED_DIR, "flight_price_combined.csv")

MODEL_PKL          = os.path.join(MODELS_DIR, "model.pkl")
ENCODERS_PKL       = os.path.join(MODELS_DIR, "encoders.pkl")
FEATURES_PKL       = os.path.join(MODELS_DIR, "features.pkl")
META_JSON          = os.path.join(MODELS_DIR, "model_meta.json")

TRAINING_LOG       = os.path.join(LOGS_DIR,   "training.log")

# ── Data generation defaults ──────────────────────────────────
DEFAULT_N_ROWS = 100_000
DEFAULT_SEED   = 42

# ── Model training ────────────────────────────────────────────
TRAIN_TEST_SPLIT = 0.85   # 85 % train, 15 % test (chronological)
CV_FOLDS         = 5

# ── Ensure all directories exist when this module is imported ─
for _d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    os.makedirs(_d, exist_ok=True)
