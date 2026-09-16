import json
from pathlib import Path

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
# 00. Data Prep & Feature Extraction Notebook
# ==============================================================================
nb00_cells = [
    md("""# 00. Data Ingestion, Signal Preprocessing & Handcrafted Feature Extraction
### Sonobuoy Sound Detective: Offline Underwater Acoustic Event Detection
**Role in Project:** Shared Data Preparation Step (Produces the benchmark dataset for all 5 algorithms)

This notebook prepares the common tabular acoustic feature dataset that will be used by all team members to train their respective classical ML models:
1. **Explore Ingested Audio:** ShipsEar, DeepShip, and Synthetic audio files.
2. **Audio Conditioning:** Mono conversion, resampling (22,050 Hz), DC-offset removal, 20–5,000 Hz Butterworth bandpass filtering.
3. **Sliding-Window Segmentation:** 4.0-second windows with 50% overlap.
4. **Candidate Event Detection:** Energy thresholding, FFT peak detection, tonal persistence.
5. **Handcrafted Feature Extraction (~100 features):**
   - 13 MFCCs + Statistics (mean, std, delta)
   - Spectral Centroid, Bandwidth, Rolloff, Flatness, Contrast
   - RMS Energy, Zero Crossing Rate (ZCR), Dynamic Range
   - Tonal features (dominant frequency, harmonic ratio, tonal persistence)
6. **Feature Selection:** Collinearity filtering ($r > 0.90$) + ANOVA F-test, Mutual Information, and Random Forest ranking to select top 35 features.
7. **Export Shared Data:** Writes `data/features/features.parquet` and `data/features/selected_features.json`.
"""),
    code("""import os
import sys
import json
from pathlib import Path

# Set paths
PROJECT_ROOT = Path("..").resolve()
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
    DATA_DIR, RAW_DIR, PROCESSED_DIR, FEATURES_DIR, METADATA_DIR,
    MASTER_METADATA, FEATURES_PARQ, FEATURES_CSV, SEL_FEATURES,
    SAMPLE_RATE, SEGMENT_DURATION, OVERLAP_FRACTION, BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ,
    RANDOM_SEED
)
from src.features.build_dataset import build_dataset
from src.features.selection import select_features

print("Environment configured.")
print(f"Data directory: {DATA_DIR}")
"""),
    md("""## 1. Check Master Metadata & Audio Class Distribution"""),
    code("""if MASTER_METADATA.exists():
    df_meta = pd.read_csv(MASTER_METADATA)
    print(f"Total audio files registered: {len(df_meta)}")
    display(df_meta.head(5))
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    df_meta['label'].value_counts().plot(kind='bar', color=['#2b5c8f', '#2a9d8f', '#e76f51'])
    plt.title("Clips per Target Class")
    plt.xlabel("Class")
    plt.ylabel("Audio Count")
    
    plt.subplot(1, 2, 2)
    df_meta['source_dataset'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette("Set2"))
    plt.title("Audio by Source Dataset")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("plots/dataset_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()
else:
    print("Run python -m src.data.organizer to register audio.")
"""),
    md("""## 2. Inspect Waveforms and Spectrograms (Vessel vs Ambient)"""),
    code("""vessel_files = list((PROCESSED_DIR / "vessel_like").glob("*.wav"))
ambient_files = list((PROCESSED_DIR / "ambient").glob("*.wav"))

if vessel_files and ambient_files:
    fig, axes = plt.subplots(2, 2, figsize=(14, 6))
    
    # Vessel audio
    y_vessel, sr = librosa.load(str(vessel_files[0]), sr=SAMPLE_RATE)
    axes[0, 0].plot(np.linspace(0, len(y_vessel)/sr, len(y_vessel)), y_vessel, color='#2b5c8f', lw=0.8)
    axes[0, 0].set_title(f"Vessel Waveform: {vessel_files[0].name}", fontsize=10, fontweight='bold')
    axes[0, 0].set_ylabel("Amplitude")
    
    D_vessel = librosa.amplitude_to_db(np.abs(librosa.stft(y_vessel)), ref=np.max)
    librosa.display.specshow(D_vessel, sr=sr, x_axis='time', y_axis='hz', ax=axes[0, 1], cmap='magma')
    axes[0, 1].set_title("Vessel Spectrogram (LOFAR Tonals)", fontsize=10, fontweight='bold')
    axes[0, 1].set_ylim(0, 3000)
    
    # Ambient audio
    y_ambient, sr = librosa.load(str(ambient_files[0]), sr=SAMPLE_RATE)
    axes[1, 0].plot(np.linspace(0, len(y_ambient)/sr, len(y_ambient)), y_ambient, color='#2a9d8f', lw=0.8)
    axes[1, 0].set_title(f"Ambient Ocean Waveform: {ambient_files[0].name}", fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].set_xlabel("Time (s)")
    
    D_ambient = librosa.amplitude_to_db(np.abs(librosa.stft(y_ambient)), ref=np.max)
    librosa.display.specshow(D_ambient, sr=sr, x_axis='time', y_axis='hz', ax=axes[1, 1], cmap='magma')
    axes[1, 1].set_title("Ambient Spectrogram (Broadband Ocean Noise)", fontsize=10, fontweight='bold')
    axes[1, 1].set_ylim(0, 3000)
    axes[1, 1].set_xlabel("Time (s)")
    
    plt.tight_layout()
    plt.savefig("plots/waveform_spectrogram_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
"""),
    md("""## 3. Extract Features and Build the Shared Dataset
This executes the batch DSP feature extraction pipeline, producing `data/features/features.parquet`.
"""),
    code("""print("Extracting features from processed audio segments...")
df_features = build_dataset(workers=1)
print(f"Extracted feature dataset shape: {df_features.shape}")
display(df_features.head(3))
"""),
    md("""## 4. Run Multi-Criteria Feature Selection
Identifies and saves the top 35 most discriminative features to `data/features/selected_features.json`.
"""),
    code("""selected_cols = select_features()
print(f"Top 35 Selected Features for All Algorithm Models:")
for i, col in enumerate(selected_cols[:15], 1):
    print(f"  {i:>2}. {col}")
print(f"... and {len(selected_cols)-15} more. Saved to {SEL_FEATURES}")
"""),
    md("""## 5. Dataset Ready for Team Members!
All team members can now proceed to their individual algorithm folders:
- `01_logistic_regression/`
- `02_k_nearest_neighbors/`
- `03_support_vector_machine/`
- `04_random_forest/`
- `05_xgboost/`
""")
]

with open(Path("00_data_prep_and_features/data_prep_and_feature_extraction.ipynb"), "w", encoding="utf-8") as f:
    json.dump(make_nb(nb00_cells), f, indent=2)

print("Created 00_data_prep_and_features/data_prep_and_feature_extraction.ipynb")
