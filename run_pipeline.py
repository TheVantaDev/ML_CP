"""
run_pipeline.py
Master orchestration script: runs all pipeline steps end-to-end.

Steps:
  1. Generate synthetic data
  2. Attempt to download DeepShip + NOAA (optional)
  3. Organize raw data -> processed/ + master_metadata.csv
  4. Build feature dataset -> features.parquet
  5. Feature selection -> selected_features.json
  6. Train + evaluate all classical ML models

Usage:
  python run_pipeline.py                   # full pipeline
  python run_pipeline.py --skip-download   # skip download step (use existing)
  python run_pipeline.py --synthetic-only  # synthetic data only (fastest)
  python run_pipeline.py --step train      # run only a specific step
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve()))


def step1_synthetic(vessel=200, ambient=200, unknown=100):
    print("\n" + "="*60)
    print("  STEP 1: Generating Synthetic Data")
    print("="*60)
    from src.data.synthetic import generate_dataset
    generate_dataset(counts={
        "vessel_like": vessel,
        "ambient":     ambient,
        "unknown":     unknown,
    })


def step2_download():
    print("\n" + "="*60)
    print("  STEP 2: Downloading Public Datasets")
    print("="*60)
    from src.data.download import download_deepship, download_noaa
    download_deepship()
    download_noaa()


def step3_organize():
    print("\n" + "="*60)
    print("  STEP 3: Organizing Data + Writing Metadata")
    print("="*60)
    from src.data.organizer import organize
    organize(copy_files=True)


def step4_features():
    print("\n" + "="*60)
    print("  STEP 4: Extracting Features")
    print("="*60)
    from src.features.build_dataset import build_dataset
    build_dataset(workers=1)


def step5_selection():
    print("\n" + "="*60)
    print("  STEP 5: Feature Selection")
    print("="*60)
    from src.features.selection import select_features
    select_features()


def step6_train():
    print("\n" + "="*60)
    print("  STEP 6: Training & Evaluating ML Models")
    print("="*60)
    from src.models.train import train
    train(model_filter="all")


def main():
    parser = argparse.ArgumentParser(description="Sonobuoy Sound Detective Pipeline")
    parser.add_argument("--skip-download",  action="store_true")
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--step", choices=[
        "synthetic", "download", "organize",
        "features", "selection", "train"
    ], default=None)
    parser.add_argument("--vessel",  type=int, default=200)
    parser.add_argument("--ambient", type=int, default=200)
    parser.add_argument("--unknown", type=int, default=100)
    args = parser.parse_args()

    if args.step:
        dispatch = {
            "synthetic": lambda: step1_synthetic(args.vessel, args.ambient, args.unknown),
            "download":  step2_download,
            "organize":  step3_organize,
            "features":  step4_features,
            "selection": step5_selection,
            "train":     step6_train,
        }
        dispatch[args.step]()
        return

    # Full pipeline
    step1_synthetic(args.vessel, args.ambient, args.unknown)

    if not args.skip_download and not args.synthetic_only:
        step2_download()

    step3_organize()
    step4_features()
    step5_selection()
    step6_train()

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("  Check reports/plots/ for confusion matrices and charts.")
    print("  Check models/ for saved model artifacts.")
    print("="*60)


if __name__ == "__main__":
    main()
