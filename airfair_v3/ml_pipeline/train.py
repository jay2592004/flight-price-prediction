"""
train.py  —  ML Pipeline entry point
Run from ml_pipeline/:
    python train.py
    python train.py --rows 500000 --seed 99 --force
"""

import argparse, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_combined
from src.eda         import run_eda
from src.trainer     import train


def main():
    parser = argparse.ArgumentParser(description="AirFair Vista — ML Pipeline")
    parser.add_argument("--rows",  type=int,  default=100_000)
    parser.add_argument("--seed",  type=int,  default=42)
    parser.add_argument("--force", action="store_true",
                        help="Force re-generation of data even if combined CSV exists")
    parser.add_argument("--skip-eda", action="store_true",
                        help="Skip EDA plots (faster if already generated)")
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  ✈  AirFair Vista — ML Training Pipeline")
    print("═"*60)
    t0 = time.time()

    # ── Step 1: Load / generate data ─────────────────────────────────────────
    print(f"\n[1/3] Loading data  (rows={args.rows:,}  seed={args.seed}  force={args.force})")
    df = load_combined(n_rows=args.rows, seed=args.seed, force=args.force)
    print(f"      Dataset: {len(df):,} rows × {len(df.columns)} columns")

    # ── Step 2: EDA ───────────────────────────────────────────────────────────
    if not args.skip_eda:
        print("\n[2/3] Running EDA — generating 16 plots + insights.json")
        insights = run_eda(df)
        print(f"      Price range: ₹{insights['price_min']:,} – ₹{insights['price_max']:,}")
        print(f"      First vs Economy: {insights['first_vs_economy']}×")
        print(f"      Last-minute premium: {insights['lastminute_premium']}×")
    else:
        print("\n[2/3] EDA skipped (--skip-eda)")

    # ── Step 3: Train ─────────────────────────────────────────────────────────
    print("\n[3/3] Training models...")
    meta = train(df)

    elapsed = time.time() - t0
    print("\n" + "═"*60)
    print("  PIPELINE COMPLETE")
    print("═"*60)
    print(f"  Best model     : {meta['model_name']}")
    print(f"  MAPE           : {meta['mape']:.2f}%")
    print(f"  R²             : {meta['r2']:.4f}")
    print(f"  MAE            : ₹{meta['mae']:,}")
    print(f"  CV MAPE        : {meta['cv_mape_mean']:.2f}% ± {meta['cv_mape_std']:.2f}%")
    print(f"  vs Baseline    : {meta['baseline_mape']:.2f}% → "
          f"+{meta['baseline_mape']-meta['mape']:.2f}pp improvement")
    print(f"  Elapsed        : {elapsed:.1f}s")
    print(f"  Artefacts      : models/  (4 files)")
    print(f"  EDA plots      : reports/ (16+ files)")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()
