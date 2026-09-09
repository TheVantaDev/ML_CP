"""
synthetic.py
Generate synthetic underwater acoustic audio clips for training/testing.

Classes produced:
  vessel_like  - propeller harmonics + AM modulation + broadband machinery
  ambient      - filtered Gaussian noise with wave-like low-frequency modulation
  unknown      - mixed, low-SNR, or randomly generated clips

Usage:
  python -m src.data.synthetic            # generate default set
  python -m src.data.synthetic --count 200 --sr 22050
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    RAW_SYNTHETIC, SAMPLE_RATE, RANDOM_SEED, METADATA_DIR
)


# ─── Random seed ───────────────────────────────────────────────────────────────
rng = np.random.default_rng(RANDOM_SEED)


# ─── Audio generators ──────────────────────────────────────────────────────────

def _sine(freq, duration, sr, amplitude=1.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t)


def _gaussian_noise(duration, sr, amplitude=0.1):
    n = int(sr * duration)
    return rng.standard_normal(n) * amplitude


def _butterworth_lowpass(signal, cutoff_hz, sr, order=4):
    from scipy.signal import butter, sosfilt
    sos = butter(order, cutoff_hz / (sr / 2), btype='low', output='sos')
    return sosfilt(sos, signal)


def _butterworth_bandpass(signal, low_hz, high_hz, sr, order=4):
    from scipy.signal import butter, sosfilt
    nyq = sr / 2
    sos = butter(order, [low_hz / nyq, high_hz / nyq], btype='band', output='sos')
    return sosfilt(sos, signal)


def generate_vessel_like(duration=4.0, sr=SAMPLE_RATE):
    """
    Simulates radiated noise from a vessel:
    - Fundamental propeller shaft frequency + harmonics
    - Amplitude-modulated broadband machinery noise
    - Low-level background noise
    """
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    # Propeller fundamental: 50–200 Hz, random for each clip
    f0 = float(rng.uniform(50, 200))
    n_harmonics = int(rng.integers(3, 8))
    signal = np.zeros(n)
    for k in range(1, n_harmonics + 1):
        amp = 1.0 / k  # harmonic series roll-off
        signal += amp * np.sin(2 * np.pi * f0 * k * t)

    # AM modulation (blade-rate modulation, slow)
    blade_rate = float(rng.uniform(1.0, 5.0))
    mod = 0.5 * (1 + np.sin(2 * np.pi * blade_rate * t))
    signal *= mod

    # Broadband machinery noise (bandpassed)
    machinery = _gaussian_noise(duration, sr, amplitude=0.3)
    machinery = _butterworth_bandpass(machinery, 20, 3000, sr)
    signal += machinery

    # Low background noise
    signal += _gaussian_noise(duration, sr, amplitude=0.05)

    # Normalize
    signal /= (np.max(np.abs(signal)) + 1e-9)
    signal *= float(rng.uniform(0.6, 0.95))

    meta = {
        "fundamental_hz": round(f0, 2),
        "n_harmonics": n_harmonics,
        "blade_rate_hz": round(blade_rate, 2),
    }
    return signal.astype(np.float32), meta


def generate_ambient(duration=4.0, sr=SAMPLE_RATE):
    """
    Simulates ocean ambient noise:
    - Broadband Gaussian noise (wind/wave)
    - Low-frequency wave modulation
    - Optional distant shipping hum (very low level)
    """
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    # Broadband noise filtered to ocean-like spectrum (more energy at low freq)
    noise = _gaussian_noise(duration, sr, amplitude=1.0)
    noise = _butterworth_lowpass(noise, 1500, sr)

    # Wave-like modulation (0.1–0.5 Hz)
    wave_freq = float(rng.uniform(0.1, 0.5))
    modulation = 0.5 * (1 + np.sin(2 * np.pi * wave_freq * t))
    signal = noise * modulation

    # Very faint distant ship (optional, 30% of clips)
    if rng.random() < 0.3:
        f_ship = float(rng.uniform(30, 80))
        faint_ship = 0.05 * np.sin(2 * np.pi * f_ship * t)
        signal += faint_ship

    signal /= (np.max(np.abs(signal)) + 1e-9)
    signal *= float(rng.uniform(0.4, 0.8))

    meta = {"wave_freq_hz": round(wave_freq, 3)}
    return signal.astype(np.float32), meta


def generate_unknown(duration=4.0, sr=SAMPLE_RATE):
    """
    Low-SNR, mixed, or random signals — difficult to classify.
    Represents the 'unknown/other' class.
    """
    choice = int(rng.integers(0, 3))

    if choice == 0:
        # Very low SNR vessel-like signal
        signal, meta = generate_vessel_like(duration, sr)
        noise_amp = float(rng.uniform(0.5, 1.5))
        signal += _gaussian_noise(duration, sr, amplitude=noise_amp)
        meta["type"] = "low_snr_vessel"
    elif choice == 1:
        # Mixed vessel + ambient
        v, mv = generate_vessel_like(duration, sr)
        a, ma = generate_ambient(duration, sr)
        alpha = float(rng.uniform(0.3, 0.7))
        signal = alpha * v + (1 - alpha) * a
        meta = {"type": "mixed", "vessel_weight": round(alpha, 2)}
    else:
        # Broadband noise with random tones (interference / unknown source)
        signal = _gaussian_noise(duration, sr, amplitude=0.8)
        f_random = float(rng.uniform(200, 4000))
        signal += 0.3 * np.sin(
            2 * np.pi * f_random * np.linspace(0, duration, int(sr * duration))
        )
        meta = {"type": "broadband_plus_tone", "tone_hz": round(f_random, 1)}

    signal /= (np.max(np.abs(signal)) + 1e-9)
    signal *= float(rng.uniform(0.3, 0.7))
    return signal.astype(np.float32), meta


# ─── Batch generator ───────────────────────────────────────────────────────────

GENERATORS = {
    "vessel_like": generate_vessel_like,
    "ambient":     generate_ambient,
    "unknown":     generate_unknown,
}


def generate_dataset(
    counts: dict = None,
    duration: float = 4.0,
    sr: int = SAMPLE_RATE,
    output_dir: Path = RAW_SYNTHETIC,
):
    """
    Generate synthetic audio clips and save them as WAV files.

    Args:
        counts: dict mapping class name -> number of clips to generate.
                Default: {vessel_like:200, ambient:200, unknown:100}
        duration: clip duration in seconds
        sr: sample rate
        output_dir: root directory; clips saved to output_dir/<class>/
    """
    if counts is None:
        counts = {"vessel_like": 200, "ambient": 200, "unknown": 100}

    records = []

    for label, n_clips in counts.items():
        if label not in GENERATORS:
            print(f"[WARN] Unknown label '{label}', skipping.")
            continue

        gen_fn = GENERATORS[label]
        class_dir = output_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating {n_clips} '{label}' clips ...")
        for i in tqdm(range(n_clips), unit="clip"):
            signal, meta = gen_fn(duration=duration, sr=sr)
            filename = f"syn_{label}_{i:04d}.wav"
            filepath = class_dir / filename
            sf.write(str(filepath), signal, sr, subtype="PCM_16")

            records.append({
                "filename": filename,
                "filepath": str(filepath),
                "label": label,
                "source": "synthetic",
                "duration_sec": duration,
                "sample_rate": sr,
                "meta": json.dumps(meta),
            })

    # Save manifest
    import pandas as pd
    df = pd.DataFrame(records)
    manifest_path = output_dir / "synthetic_manifest.csv"
    df.to_csv(manifest_path, index=False)
    print(f"\nSaved {len(records)} clips. Manifest -> {manifest_path}")
    return df


# ─── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic underwater audio")
    parser.add_argument("--vessel", type=int, default=200, help="Vessel-like clip count")
    parser.add_argument("--ambient", type=int, default=200, help="Ambient clip count")
    parser.add_argument("--unknown", type=int, default=100, help="Unknown clip count")
    parser.add_argument("--duration", type=float, default=4.0, help="Clip duration (seconds)")
    parser.add_argument("--sr", type=int, default=SAMPLE_RATE, help="Sample rate (Hz)")
    args = parser.parse_args()

    generate_dataset(
        counts={
            "vessel_like": args.vessel,
            "ambient":     args.ambient,
            "unknown":     args.unknown,
        },
        duration=args.duration,
        sr=args.sr,
    )
