"""
download.py
Helpers to download public datasets for the Sonobuoy Sound Detective project.

Supported sources:
  - DeepShip  (Zenodo / direct links — 4 vessel classes)
  - NOAA SanctSound (ERDDAP API — ambient ocean recordings)

Usage:
  python -m src.data.download --source deepship
  python -m src.data.download --source noaa
  python -m src.data.download --source all
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import RAW_DEEPSHIP, RAW_NOAA


# ─── Progress-bar download helper ──────────────────────────────────────────────

class _ProgressHook:
    """tqdm progress hook for urllib."""
    def __init__(self, desc: str):
        self._pbar = None
        self._desc = desc

    def __call__(self, block_num, block_size, total_size):
        if self._pbar is None:
            self._pbar = tqdm(
                total=total_size, unit="B", unit_scale=True,
                unit_divisor=1024, desc=self._desc
            )
        downloaded = block_num * block_size
        self._pbar.update(min(block_size, total_size - downloaded + block_size))

    def close(self):
        if self._pbar:
            self._pbar.close()


def _download_file(url: str, dest: Path, expected_sha256: str = None):
    """Download url to dest, with optional SHA-256 verification."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [SKIP] Already downloaded: {dest.name}")
        return True

    print(f"  Downloading {dest.name} ...")
    hook = _ProgressHook(dest.name)
    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=hook)
    finally:
        hook.close()

    if expected_sha256:
        sha = hashlib.sha256(dest.read_bytes()).hexdigest()
        if sha != expected_sha256:
            dest.unlink()
            raise ValueError(f"SHA-256 mismatch for {dest.name}. Got {sha}")
        print(f"  SHA-256 verified OK.")
    return True


# ─── DeepShip ──────────────────────────────────────────────────────────────────
# DeepShip: "A Large Underwater Acoustic Benchmark Dataset and Study for
#            Identification of Merchant Marine Vessels"
# 4 classes: cargo, passengership, tanker, tug
# Published on IEEE DataPort / Zenodo.
#
# NOTE: The full dataset is ~8 GB. Below we list the Zenodo DOI record.
# If the links expire, go to: https://doi.org/10.5281/zenodo.6561252
# and download the zip files manually into data/raw/deepship/

DEEPSHIP_FILES = [
    # (filename, url, sha256_or_None)
    # Subset ZIP files from the Zenodo record (parts may change; update if 404)
    (
        "DeepShip.zip",
        "https://zenodo.org/record/6561252/files/DeepShip.zip?download=1",
        None,   # SHA-256 not verified automatically (large file)
    ),
]

DEEPSHIP_LABEL_MAP = {
    # original DeepShip folder names -> project broad class
    "Cargo":         "vessel_like",
    "Passengership": "vessel_like",
    "Tanker":        "vessel_like",
    "Tug":           "vessel_like",
}


def download_deepship(dest_dir: Path = RAW_DEEPSHIP):
    """Download DeepShip dataset archives."""
    print("\n=== DeepShip Download ===")
    dest_dir.mkdir(parents=True, exist_ok=True)

    any_downloaded = False
    for filename, url, sha in DEEPSHIP_FILES:
        dest = dest_dir / filename
        try:
            _download_file(url, dest, sha)
            any_downloaded = True
        except Exception as e:
            print(f"  [ERROR] {e}")
            print(f"  Please download manually from:")
            print(f"  https://doi.org/10.5281/zenodo.6561252")
            print(f"  and place the ZIP in: {dest_dir}")

    if any_downloaded:
        print("\n  Extracting archives ...")
        import zipfile
        for zf_path in dest_dir.glob("*.zip"):
            print(f"  Extracting {zf_path.name} ...")
            with zipfile.ZipFile(zf_path, "r") as zf:
                zf.extractall(dest_dir)
        print("  Done.")

    return dest_dir


# ─── NOAA SanctSound ───────────────────────────────────────────────────────────
# SanctSound is a NOAA–Navy passive acoustic monitoring project.
# Public data browser: https://sanctsound.ioos.us/
# ERDDAP API: https://erddap.sanctsound.ioos.us/erddap/
#
# We download a small subset of publicly available sound-level data.
# Raw audio is very large; here we fetch a manageable sample.

NOAA_SAMPLE_FILES = [
    # Small publicly available demonstration WAV files from NOAA archives.
    # These are ambient ocean recordings suitable for the "ambient" class.
    (
        "NOAA_ONC_ambient_sample_01.wav",
        "https://www.ncei.noaa.gov/data/oceans/passive-acoustic/data/sample_audio/sample_ocean_ambient_01.wav",
        None,
    ),
]

# Fallback: list of direct MBARI / NOAA freely available ocean ambient samples
NOAA_FALLBACK_URLS = [
    # If the above fails, use these open OGG/WAV files from public sources:
    (
        "mbari_ambient_deepwater_01.wav",
        "https://www.mbari.org/wp-content/uploads/2015/10/ambient_ocean.wav",
        None,
    ),
]


def download_noaa(dest_dir: Path = RAW_NOAA):
    """
    Attempt to download a small NOAA ambient acoustic sample set.
    Falls back to MBARI/public samples if NOAA links are unavailable.
    """
    print("\n=== NOAA Ambient Samples Download ===")
    dest_dir.mkdir(parents=True, exist_ok=True)

    success = False
    for filename, url, sha in NOAA_SAMPLE_FILES:
        dest = dest_dir / filename
        try:
            _download_file(url, dest, sha)
            success = True
        except Exception as e:
            print(f"  [WARN] Primary URL failed: {e}")

    if not success:
        print("  Trying fallback MBARI sources ...")
        for filename, url, sha in NOAA_FALLBACK_URLS:
            dest = dest_dir / filename
            try:
                _download_file(url, dest, sha)
                success = True
            except Exception as e:
                print(f"  [WARN] Fallback also failed: {e}")

    if not success:
        print(
            "\n  [INFO] Could not auto-download NOAA samples.\n"
            "  You can manually place ambient ocean WAV files into:\n"
            f"  {dest_dir}\n"
            "  Good free sources:\n"
            "    - https://sanctsound.ioos.us/   (SanctSound data portal)\n"
            "    - https://www.ncei.noaa.gov/products/passive-acoustic-data\n"
        )
    return dest_dir


# ─── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download public acoustic datasets")
    parser.add_argument(
        "--source",
        choices=["deepship", "noaa", "all"],
        default="all",
        help="Which dataset to download",
    )
    args = parser.parse_args()

    if args.source in ("deepship", "all"):
        download_deepship()
    if args.source in ("noaa", "all"):
        download_noaa()
