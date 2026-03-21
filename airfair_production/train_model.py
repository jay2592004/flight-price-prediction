"""
train_model.py  —  Entry point for training.
Run from project root:
    python train_model.py
    python train_model.py --rows 500000 --seed 99 --force
"""

import argparse
from src.pipeline.train import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AirFair Vista — Training Pipeline")
    parser.add_argument("--rows",  type=int,  default=100_000, help="Synthetic rows to generate")
    parser.add_argument("--seed",  type=int,  default=42,      help="Random seed")
    parser.add_argument("--force", action="store_true",        help="Force re-generation of data")
    args = parser.parse_args()

    print(f"\n✈  AirFair Vista — Training Pipeline")
    print(f"   rows={args.rows:,}  seed={args.seed}  force_regen={args.force}\n")

    meta = run(n_rows=args.rows, seed=args.seed, force_regen=args.force)

    print("\n" + "═"*55)
    print("  TRAINING COMPLETE")
    print("═"*55)
    print(f"  Best model  : {meta['model_name']}")
    print(f"  MAPE        : {meta['mape']:.2f}%")
    print(f"  R²          : {meta['r2']:.4f}")
    print(f"  MAE         : ₹{meta['mae']:,}")
    print(f"  CV MAPE     : {meta['cv_mape_mean']:.2f}% ± {meta['cv_mape_std']:.2f}%")
    print(f"  vs Baseline : {meta['baseline_mape']:.2f}% → improvement "
          f"{meta['baseline_mape']-meta['mape']:.2f}pp")
    print(f"  Artefacts   : models/model.pkl · encoders.pkl · features.pkl · model_meta.json")
    print("═"*55 + "\n")
