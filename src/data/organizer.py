"""
organizer.py
Scans raw downloaded data, maps labels to broad project classes,
copies/links files into processed/, and writes master_metadata.csv.

Label mapping logic:
  ShipsEar: Classes A, B, C, D (or 0, 1, 2, 3) -> vessel_like, Class E (or 4) -> ambient
  DeepShip: Cargo / Passengership / Tanker / Tug -> vessel_like
  NOAA:     ambient ocean recordings             -> ambient
  Watkins:  whale / dolphin / seal / fish        -> marine_life
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
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    RAW_DIR, PROCESSED_DIR, METADATA_DIR, RAW_SHIPSEAR,
    RAW_DEEPSHIP, RAW_NOAA, RAW_SYNTHETIC, RAW_WATKINS, RAW_MARINE,
    PROC_VESSEL, PROC_AMBIENT, PROC_UNKNOWN, PROC_MARINE,
    MASTER_METADATA, SAMPLE_RATE, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED
)

rng = np.random.default_rng(RANDOM_SEED)

BROAD_CLASS_DIRS = {
    "vessel_like": PROC_VESSEL,
    "marine_life": PROC_MARINE,
    "ambient":     PROC_AMBIENT,
    "unknown":     PROC_UNKNOWN,
}

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}

DEEPSHIP_LABEL_MAP = {
    "cargo":         ("vessel_like", "cargo"),
    "passengership": ("vessel_like", "passengership"),
    "tanker":        ("vessel_like", "tanker"),
    "tug":           ("vessel_like", "tug"),
}

SHIPSEAR_NUMERIC_MAP = {
    "0": ("vessel_like", "fishing_trawler"),  # Class A
    "1": ("vessel_like", "passenger_ferry"),  # Class B
    "2": ("vessel_like", "ocean_liner"),      # Class C
    "3": ("vessel_like", "motorboat"),        # Class D
    "4": ("ambient",     "ocean_ambient"),    # Class E
}

NOAA_LABEL_MAP = {
    "default": ("ambient", "ocean_ambient"),
}

SYNTHETIC_LABEL_MAP = {
    "vessel_like": ("vessel_like", "synthetic_vessel"),
    "ambient":     ("ambient",     "synthetic_ambient"),
    "unknown":     ("unknown",     "synthetic_unknown"),
}


def _sha256(path: Path, chunk=65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _get_audio_duration(path: Path) -> float:
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


def _scan_deepship() -> list[dict]:
    records = []
    if not RAW_DEEPSHIP.exists():
        return records
    for audio_path in RAW_DEEPSHIP.rglob("*"):
        if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        parts = [p.lower() for p in audio_path.relative_to(RAW_DEEPSHIP).parts]
        
        broad = "vessel_like"
        subclass = "deepship_vessel"
        for p in parts:
            if p in DEEPSHIP_LABEL_MAP:
                broad, subclass = DEEPSHIP_LABEL_MAP[p]
                break
                
        records.append({
            "source_dataset": "DeepShip",
            "original_file":  str(audio_path),
            "raw_label":      subclass,
            "label":          broad,
            "subclass":       subclass,
        })
    return records


def _scan_shipsear() -> list[dict]:
    records = []
    if not RAW_SHIPSEAR.exists():
        return records
    for audio_path in RAW_SHIPSEAR.rglob("*"):
        if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        parts = [p.lower() for p in audio_path.relative_to(RAW_SHIPSEAR).parts]
        combined = " ".join(parts)
        
        broad = "vessel_like"
        subclass = "shipsear_vessel"
        
        # Check numeric folder classes (0, 1, 2, 3, 4) in path
        matched_num = False
        for p in parts:
            if p in SHIPSEAR_NUMERIC_MAP:
                broad, subclass = SHIPSEAR_NUMERIC_MAP[p]
                matched_num = True
                break
                
        if not matched_num:
            if any(k in combined for k in ["classe", "class_e", "ambient", "background", "natural"]):
                broad, subclass = "ambient", "natural_background"
            elif any(k in combined for k in ["classa", "class_a", "fishing", "trawler"]):
                broad, subclass = "vessel_like", "fishing_vessel"
            elif any(k in combined for k in ["classb", "class_b", "passenger", "ferry"]):
                broad, subclass = "vessel_like", "passenger_ferry"
            elif any(k in combined for k in ["classc", "class_c", "ocean_liner", "cargo", "ro-ro"]):
                broad, subclass = "vessel_like", "ocean_liner"
            elif any(k in combined for k in ["classd", "class_d", "motorboat", "pilot", "sailboat"]):
                broad, subclass = "vessel_like", "motorboat"
                
        records.append({
            "source_dataset": "ShipsEar",
            "original_file":  str(audio_path),
            "raw_label":      subclass,
            "label":          broad,
            "subclass":       subclass,
        })
    return records


def _scan_noaa() -> list[dict]:
    records = []
    if not RAW_NOAA.exists():
        return records
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


def _scan_marine_life() -> list[dict]:
    records = []
    for raw_dir, name in [(RAW_WATKINS, "Watkins"), (RAW_MARINE, "MarineLife")]:
        if not raw_dir.exists():
            continue
        for audio_path in raw_dir.rglob("*"):
            if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            parts = audio_path.relative_to(raw_dir).parts
            subclass = parts[0] if len(parts) >= 2 else audio_path.stem
            records.append({
                "source_dataset": name,
                "original_file":  str(audio_path),
                "raw_label":      subclass,
                "label":          "marine_life",
                "subclass":       subclass,
            })
    return records


def _scan_synthetic() -> list[dict]:
    records = []
    if not RAW_SYNTHETIC.exists():
        return records
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


def organize(copy_files: bool = True, max_per_source: int = 400) -> pd.DataFrame:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    for d in BROAD_CLASS_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    print("Scanning raw data sources ...")
    raw_deepship = _scan_deepship()
    raw_shipsear = _scan_shipsear()
    raw_noaa     = _scan_noaa()
    raw_marine   = _scan_marine_life()
    raw_synthetic= _scan_synthetic()

    print(f"  ShipsEar files found : {len(raw_shipsear)}")
    print(f"  DeepShip files found : {len(raw_deepship)}")
    print(f"  Synthetic files found: {len(raw_synthetic)}")
    print(f"  NOAA files found     : {len(raw_noaa)}")
    print(f"  MarineLife files     : {len(raw_marine)}")

    # Balance/sample from large datasets so ShipsEar doesn't drown everything
    records = []
    for src_records in [raw_shipsear, raw_deepship, raw_noaa, raw_marine, raw_synthetic]:
        if len(src_records) > max_per_source:
            rng.shuffle(src_records)
            records.extend(src_records[:max_per_source])
        else:
            records.extend(src_records)

    if not records:
        print("[WARN] No audio files found in raw/.")
        return pd.DataFrame()

    print(f"Indexed {len(records)} balanced audio files total.")

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
            "split":                 "",
            "noise_condition":       "unknown",
            "license_or_attribution":"",
        })

    df = pd.DataFrame(final_rows)

    # Source-aware train/val/test split
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
