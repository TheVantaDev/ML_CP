import json
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def make_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.13"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

# ==============================================================================
# NOTEBOOK 1: Audio EDA and Visualization
# ==============================================================================
nb1_cells = [
    md("""# 01. Audio Data Exploration & Exploratory Data Analysis (EDA)
### Sonobuoy Sound Detective: Offline Underwater Acoustic Event Detection
**ML Philosophy:** Classical ML & Digital Signal Processing (DSP) only — No Deep Learning.

This notebook performs first-pass exploratory analysis on underwater acoustic recordings:
1. Ingesting raw and processed audio recordings.
2. Checking class distribution, sample rates, and file durations.
3. Visualizing Time-Domain Waveforms for all 4 acoustic classes (`vessel_like`, `marine_life`, `ambient`, `unknown`).
4. Analyzing Frequency-Domain Power Spectra (FFT) to spot narrowband engine tones vs broadband sea noise.
5. Visualizing LOFAR / Spectrograms (Short-Time Fourier Transform).
6. Estimating Signal-to-Noise Ratio (SNR) and assessing audio quality.
"""),
    code("""import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path("..").resolve() if Path(".").resolve().name == "notebooks" else Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf
import librosa
import librosa.display
from scipy import signal

from src.utils.config import (
    DATA_DIR, RAW_DIR, PROCESSED_DIR, MASTER_METADATA,
    SAMPLE_RATE, CLASSES, CLASS_NAMES
)

print(f"Project Root : {PROJECT_ROOT}")
print(f"Sample Rate  : {SAMPLE_RATE} Hz")
print(f"Target Classes: {CLASS_NAMES}")
"""),
    md("""## 1. Inspect Master Metadata and Class Distribution
Let's inspect the `master_metadata.csv` to see how many files we have per class and per dataset source.
"""),
    code("""if MASTER_METADATA.exists():
    df_meta = pd.read_csv(MASTER_METADATA)
    print(f"Total registered audio files: {len(df_meta)}")
    display(df_meta.head(5))
    
    # Class breakdown
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    df_meta['label'].value_counts().plot(kind='bar', color=['#2b5c8f', '#2a9d8f', '#e76f51', '#e9c46a'])
    plt.title("Clips per Class")
    plt.xlabel("Class")
    plt.ylabel("Count")
    
    plt.subplot(1, 2, 2)
    df_meta['source_dataset'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette("Set2"))
    plt.title("Clips by Source Dataset")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()
else:
    print("master_metadata.csv not found yet. Searching processed/ and raw/ directories...")
    # Scan processed dir
    sample_files = list(PROCESSED_DIR.rglob("*.wav"))
    print(f"Found {len(sample_files)} processed wav files.")
"""),
    md("""## 2. Load Representative Audio Clips for Each Class
Let's select one representative audio sample from each class (`vessel_like`, `marine_life`, `ambient`, `unknown`) and load the raw audio waveform.
"""),
    code("""def get_sample_for_class(cls_name):
    # Try processed first, then raw
    cls_dir = PROCESSED_DIR / cls_name
    files = list(cls_dir.glob("*.wav")) if cls_dir.exists() else []
    if not files:
        files = list((RAW_DIR / "synthetic" / cls_name).glob("*.wav"))
    if files:
        return files[0]
    return None

sample_paths = {cls: get_sample_for_class(cls) for cls in CLASS_NAMES}
audio_data = {}

for cls, path in sample_paths.items():
    if path and path.exists():
        y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        audio_data[cls] = (y, sr, path.name)
        print(f"[{cls:<12}] Loaded: {path.name} | Duration: {len(y)/sr:.2f}s | SR: {sr}Hz")
    else:
        print(f"[{cls:<12}] No sample found yet.")
"""),
    md("""## 3. Time-Domain Waveform Comparison
Notice the difference in envelope:
- **Vessel-like sound:** Continuous amplitude envelope driven by propeller cavitation and reciprocating diesel machinery.
- **Ambient noise:** Uniform Gaussian/Rayleigh random fluctuation from wave and wind turbulence.
- **Marine life:** Intermittent pulsed clicks or sweeping tonal whistles.
"""),
    code("""fig, axes = plt.subplots(len(audio_data), 1, figsize=(12, 2.5 * len(audio_data)), sharex=True)
if len(audio_data) == 1:
    axes = [axes]

for ax, (cls, (y, sr, fname)) in zip(axes, audio_data.items()):
    t = np.linspace(0, len(y) / sr, len(y))
    ax.plot(t, y, color='#1f77b4', lw=0.8)
    ax.set_title(f"Class: {cls.upper()}  ({fname})", fontsize=11, fontweight='bold')
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (seconds)")
plt.tight_layout()
plt.show()
"""),
    md("""## 4. Frequency-Domain Power Spectral Density (FFT)
Vessel noise is characterized by **discrete tonals** (narrowband spectral lines corresponding to engine cylinder firing rates, shaft revolutions, and propeller blade pass frequency). Ocean ambient noise is broadband.
"""),
    code("""plt.figure(figsize=(12, 6))

colors = {'vessel_like': 'red', 'marine_life': 'green', 'ambient': 'blue', 'unknown': 'gray'}

for cls, (y, sr, fname) in audio_data.items():
    # Compute power spectrum using Welch method
    freqs, psd = signal.welch(y, fs=sr, nperseg=2048)
    # Filter up to 3000 Hz where acoustic target signatures dominate
    mask = freqs <= 3000
    plt.semilogy(freqs[mask], psd[mask], label=f"{cls} ({fname})", color=colors.get(cls, 'black'), alpha=0.8)

plt.title("Power Spectral Density (0 - 3000 Hz) — Narrowband Tonals vs Broadband Noise", fontsize=12)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (V²/Hz)")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.show()
"""),
    md("""## 5. LOFAR / Time-Frequency Spectrograms (STFT)
LOFAR (Low-Frequency Analysis and Recording) displays are standard in passive acoustic sonar. Narrowband tonals appear as distinct horizontal lines over time.
"""),
    code("""fig, axes = plt.subplots(len(audio_data), 1, figsize=(12, 3 * len(audio_data)))
if len(audio_data) == 1:
    axes = [axes]

for ax, (cls, (y, sr, fname)) in zip(axes, audio_data.items()):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=512)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, hop_length=512, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
    ax.set_ylim(0, 4000)
    ax.set_title(f"Spectrogram / LOFAR: {cls.upper()}", fontsize=11, fontweight='bold')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

plt.tight_layout()
plt.show()
"""),
    md("""## 6. Summary & Key Takeaways
1. **Vessel signatures:** Possess distinct narrowband spectral peaks below 1000 Hz (shaft/propeller tones) and persistent tonal lines in spectrograms.
2. **Ambient noise:** Has decaying spectral roll-off with no persistent tonal peaks.
3. **Marine life:** Displays localized bursts or frequency-modulated chirps.
4. Next Step: **02. Signal Preprocessing & Handcrafted DSP Feature Extraction**.
""")
]

with open(NOTEBOOKS_DIR / "01_audio_eda_and_visualization.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_nb(nb1_cells), f, indent=2)

print("Created 01_audio_eda_and_visualization.ipynb")
