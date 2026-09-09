"""
organizer.py
Scans raw downloaded data, maps labels to broad project classes,
copies/links files into processed/, and writes master_metadata.csv.

Label mapping logic:
  DeepShip: Cargo / Passengership / Tanker / Tug -> vessel_like
  NOAA:     ambient ocean recordings             -> ambient
  Synthetic: already labelled by generator       -> vessel_like / ambient / unknown

Usage:
  python -m src.data.organizer
"""

import hashlib
import shutil
import sys
import uuid
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    RAW_DIR, PROCESSED_DIR, METADATA_DIR, RAW_DEEPSHIP,
    RAW_NOAA, RAW_SYNTHETIC, PROC_VESSEL, PROC_AMBIENT,
    PROC_UNKNOWN, PROC_MARINE, MASTER_METADATA, SAMPLE_RATE,
    TRAIN_RATIO, VAL_RATIO, RANDOM_SEED
)

import numpy as np

rng = np.random.default_rng(RANDOM_SEED)


# ─── Label map ─────────────────────────────────────────────────────────────────

# Each entry: (source_keyword_in_path_or_label, broad_class, subclass)
DEEPSHIP_LABEL_MAP = {
    "cargo":         ("vessel_like", "cargo"),
    "passengership": ("vessel_like", "passengership"),
    "tanker":        ("vessel_like", "tanker"),
    "tug":           ("vessel_like", "tug"),
}

NOAA_LABEL_MAP = {
    # Default all NOAA files to ambient; adjust if you have labelled bioacoustic
    "default": ("ambient", "ocean_ambient"),
}

SYNTHETIC_LABEL_MAP = {
    "vessel_like": ("vessel_like", "synthetic_vessel"),
    "ambient":     ("ambient",     "synthetic_ambient"),
    "unknown":     ("unknown",     "synthetic_unknown"),
}


BROAD_CLASS_DIRS = {
    "vessel_like": PROC_VESSEL,
    "marine_life": PROC_MARINE,
    "ambient":     PROC_AMBIENT,
    "unknown":     PROC_UNKNOWN,
}

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _sha256(path: Path, chunk=65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _get_audio_duration(path: Path) -> float:
    """Get duration in seconds using soundfile (fast, no decode needed)."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return info.duration
    except Exception:
        return -1.0


def _get_audio_sr(path: Path) -> int:
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return info.samplerate
    except Exception:
        return -1


# ─── Source scanners ───────────────────────────────────────────────────────────

def _scan_deepship() -> list[dict]:
    """Walk DeepShip directory tree and infer label from parent folder name."""
    records = []
    for audio_path in RAW_DEEPSHIP.rglob("*"):
        if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        # DeepShip folder structure: raw/deepship/Cargo/shipXXX/*.wav
        # Find the class folder (first subfolder under deepship)
        parts = audio_path.relative_to(RAW_DEEPSHIP).parts
        if len(parts) < 2:
            continue
        folder_label = parts[0].lower()
        if folder_label in DEEPSHIP_LABEL_MAP:
            broad, subclass = DEEPSHIP_LABEL_MAP[folder_label]
        else:
            broad, subclass = "unknown", folder_label
        records.append({
            "source_dataset": "DeepShip",
            "original_file":  str(audio_path),
            "raw_label":      parts[0],
            "label":          broad,
            "subclass":       subclass,
        })
    return records


def _scan_noaa() -> list[dict]:
    """Treat all NOAA audio files as ambient ocean noise."""
    records = []
    for audio_path in RAW_NOAA.rglob("*"):
        if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        records.append({
            "source_dataset": "NOAA",
            "original_file":  str(audio_path),
            "raw_label":      "ocean_ambient",
            "label":          "ambient",
            "subclass":       "ocean_ambient",
        })
    return records


def _scan_synthetic() -> list[dict]:
    """Infer synthetic label from parent folder name."""
    records = []
    for audio_path in RAW_SYNTHETIC.rglob("*.wav"):
        parts = audio_path.relative_to(RAW_SYNTHETIC).parts
        if len(parts) < 2:
            continue
        folder_label = parts[0].lower()
        if folder_label in SYNTHETIC_LABEL_MAP:
            broad, subclass = SYNTHETIC_LABEL_MAP[folder_label]
        else:
            broad, subclass = "unknown", folder_label
        records.append({
            "source_dataset": "Synthetic",
            "original_file":  str(audio_path),
            "raw_label":      folder_label,
            "label":          broad,
            "subclass":       subclass,
        })
    return records


# ─── Main organizer ────────────────────────────────────────────────────────────

def organize(copy_files: bool = True) -> pd.DataFrame:
    """
    Scan all raw sources, build metadata, copy files to processed/<class>/,
    assign train/val/test splits, and write master_metadata.csv.

    Args:
        copy_files: if True, copy audio to processed/; if False, only write metadata.
    Returns:
        DataFrame of master_metadata.
    """
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    for d in BROAD_CLASS_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    print("Scanning raw data sources ...")
    records = []
    records.extend(_scan_deepship())
    records.extend(_scan_noaa())
    records.extend(_scan_synthetic())

    if not records:
        print("[WARN] No audio files found in raw/. Run download.py or synthetic.py first.")
        return pd.DataFrame()

    print(f"Found {len(records)} audio files total.")

    # Enrich each record
    final_rows = []
    for rec in tqdm(records, desc="Processing file metadata"):
        path = Path(rec["original_file"])
        if not path.exists():
            continue
        try:
            duration  = _get_audio_duration(path)
            orig_sr   = _get_audio_sr(path)
            file_hash = _sha256(path)
        except Exception as e:
            print(f"  [WARN] Skipping {path.name}: {e}")
            continue

        clip_id     = str(uuid.uuid4())[:12]
        proc_dir    = BROAD_CLASS_DIRS.get(rec["label"], PROC_UNKNOWN)
        new_filename= f"{rec['label']}_{clip_id}{path.suffix}"
        new_path    = proc_dir / new_filename

        if copy_files and not new_path.exists():
            shutil.copy2(str(path), str(new_path))

        final_rows.append({
            "clip_id":               clip_id,
            "source_dataset":        rec["source_dataset"],
            "original_file":         rec["original_file"],
            "processed_file":        str(new_path),
            "raw_label":             rec["raw_label"],
            "label":                 rec["label"],
            "subclass":              rec["subclass"],
            "duration_seconds":      round(duration, 3),
            "sample_rate":           orig_sr,
            "processed_sample_rate": SAMPLE_RATE,
            "file_hash":             file_hash,
            "split":                 "",   # assigned below
            "noise_condition":       "unknown",
            "license_or_attribution":"",
        })

    df = pd.DataFrame(final_rows)

    # ── Source-aware train/val/test split ───────────────────────────────────────
    # Group by (source_dataset, label) so entire source groups stay together.
    df["split"] = ""
    for _, grp in df.groupby(["source_dataset", "label"]):
        idx = grp.index.tolist()
        rng.shuffle(idx)
        n = len(idx)
        n_train = max(1, int(n * TRAIN_RATIO))
        n_val   = max(1, int(n * VAL_RATIO))
        df.loc[idx[:n_train],       "split"] = "train"
        df.loc[idx[n_train:n_train + n_val], "split"] = "val"
        df.loc[idx[n_train + n_val:],        "split"] = "test"

    df.to_csv(MASTER_METADATA, index=False)
    print(f"\nMaster metadata saved -> {MASTER_METADATA}")
    print(df.groupby(["label", "split"]).size().unstack(fill_value=0).to_string())
    return df


if __name__ == "__main__":
    organize(copy_files=True)
