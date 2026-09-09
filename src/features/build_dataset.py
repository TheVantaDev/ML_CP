"""
build_dataset.py
Run the full feature extraction pipeline over all processed audio files
and produce a feature matrix (features.csv + features.parquet).

Usage:
  python -m src.features.build_dataset
  python -m src.features.build_dataset --workers 4
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    PROCESSED_DIR, FEATURES_CSV, FEATURES_PARQ, FEATURES_DIR,
    MASTER_METADATA, SAMPLE_RATE, SEGMENT_DURATION, OVERLAP_FRACTION
)


def _process_one_file(args_tuple):
    """Worker function: preprocess + segment + extract features for one file."""
    row_dict, seg_duration, overlap, sr = args_tuple
    path     = Path(row_dict["processed_file"])
    label    = row_dict["label"]
    split    = row_dict.get("split", "")
    clip_id  = row_dict["clip_id"]

    if not path.exists():
        return []

    try:
        from src.preprocessing.segmenter import segment_file
        from src.features.extractor import extract_features

        segments = segment_file(path, sr=sr, duration=seg_duration, overlap=overlap)
        results = []
        for seg_idx, (start_sec, end_sec, window) in enumerate(segments):
            seg_id = f"{clip_id}_s{seg_idx:03d}"
            feat = extract_features(window, sr=sr, clip_id=seg_id, label=label, split=split)
            feat["source_clip_id"] = clip_id
            feat["start_sec"]      = round(start_sec, 3)
            feat["end_sec"]        = round(end_sec, 3)
            results.append(feat)
        return results
    except Exception as e:
        print(f"  [ERROR] {path.name}: {e}")
        return []


def build_dataset(
    workers: int = 1,
    seg_duration: float = SEGMENT_DURATION,
    overlap: float = OVERLAP_FRACTION,
    sr: int = SAMPLE_RATE,
) -> pd.DataFrame:
    """
    Extract features from all files listed in master_metadata.csv.
    Saves features.csv and features.parquet to data/features/.
    """
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    if not MASTER_METADATA.exists():
        raise FileNotFoundError(
            f"master_metadata.csv not found at {MASTER_METADATA}.\n"
            "Run src/data/organizer.py first."
        )

    meta_df = pd.read_csv(MASTER_METADATA)
    print(f"Processing {len(meta_df)} audio files ...")
    print(meta_df.groupby("label").size().to_string())

    all_rows = meta_df.to_dict("records")
    tasks = [(row, seg_duration, overlap, sr) for row in all_rows]

    all_features = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_process_one_file, t): t for t in tasks}
            for fut in tqdm(as_completed(futures), total=len(tasks), unit="file"):
                all_features.extend(fut.result())
    else:
        for task in tqdm(tasks, unit="file"):
            all_features.extend(_process_one_file(task))

    if not all_features:
        print("[WARN] No features extracted. Check that processed/ contains audio files.")
        return pd.DataFrame()

    df = pd.DataFrame(all_features)

    # Move label/split to front
    cols_first = ["clip_id", "source_clip_id", "label", "split", "start_sec", "end_sec"]
    cols_rest  = [c for c in df.columns if c not in cols_first]
    df = df[cols_first + cols_rest]

    df.to_csv(FEATURES_CSV, index=False)
    df.to_parquet(FEATURES_PARQ, index=False)
    print(f"\nFeature dataset: {len(df)} rows x {len(df.columns)} columns")
    print(f"  -> {FEATURES_CSV}")
    print(f"  -> {FEATURES_PARQ}")
    print(df.groupby(["label", "split"]).size().unstack(fill_value=0).to_string())
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build feature dataset")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    args = parser.parse_args()
    build_dataset(workers=args.workers)
