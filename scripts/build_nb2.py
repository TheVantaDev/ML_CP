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
# NOTEBOOK 2: Signal Preprocessing and DSP Feature Extraction
# ==============================================================================
nb2_cells = [
    md("""# 02. Signal Preprocessing & Handcrafted DSP Feature Extraction
### Sonobuoy Sound Detective: Offline Underwater Acoustic Event Detection
**ML Philosophy:** Classical ML & Digital Signal Processing (DSP) only — No Deep Learning.

In this notebook, we transform raw underwater acoustic signals into structured, explainable numeric feature vectors:
1. Audio conditioning: Mono conversion, standardizing sample rate (22,050 Hz), DC offset removal.
2. 5th-order Butterworth Bandpass Filtering (20 Hz - 5,000 Hz) to isolate acoustic cavitation and machinery frequencies.
3. Sliding-window segmentation (4.0s windows, 50% overlap).
4. Candidate event detection: Short-time energy thresholding, FFT peak detection, and tonal persistence tracking.
5. Handcrafted feature extraction (~100 features covering cepstral, spectral, energy, temporal, and tonal domains).
6. Building and validating the structured dataset (`features.parquet`).
"""),
    code("""import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("..").resolve() if Path(".").resolve().name == "notebooks" else Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf
import librosa

from src.utils.config import (
    SAMPLE_RATE, SEGMENT_DURATION, OVERLAP_FRACTION,
    BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ, PROCESSED_DIR,
    FEATURES_PARQ, FEATURES_CSV, CLASS_NAMES
)
from src.preprocessing.audio_utils import (
    load_and_resample, to_mono, remove_dc_offset,
    apply_bandpass_filter, rms_normalize
)
from src.preprocessing.segmenter import segment_audio
from src.features.extractor import extract_all_features
from src.features.event_detection import detect_candidate_event

print("Modules and configuration imported successfully.")
"""),
    md("""## 1. Demonstrate the Audio Preprocessing Pipeline on a Sample Clip
Let's trace how a raw hydrophone signal is cleaned and normalized.
"""),
    code("""# Find a sample audio file
sample_files = list(PROCESSED_DIR.rglob("*.wav"))
if not sample_files:
    sample_files = list(Path("data/raw/synthetic").rglob("*.wav"))

if sample_files:
    sample_path = sample_files[0]
    print(f"Testing pipeline on: {sample_path.name}")
    
    # Step 1: Load and resample
    raw_audio, orig_sr = sf.read(str(sample_path))
    audio_mono = to_mono(raw_audio)
    
    # Step 2: DC Offset removal
    audio_no_dc = remove_dc_offset(audio_mono)
    
    # Step 3: Butterworth bandpass filter (20 - 5000 Hz)
    audio_filtered = apply_bandpass_filter(audio_no_dc, sr=SAMPLE_RATE, low_hz=BANDPASS_LOW_HZ, high_hz=BANDPASS_HIGH_HZ)
    
    # Step 4: RMS Normalization
    audio_norm = rms_normalize(audio_filtered, target_rms=0.1)
    
    # Plot before vs after
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(audio_mono[:int(SAMPLE_RATE*2)], color='gray', lw=0.8)
    plt.title("Original Waveform (First 2 seconds)")
    plt.ylabel("Raw Amplitude")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(audio_norm[:int(SAMPLE_RATE*2)], color='#1f77b4', lw=0.8)
    plt.title(f"Preprocessed & Filtered ({BANDPASS_LOW_HZ}-{BANDPASS_HIGH_HZ} Hz Butterworth) & RMS Normalized")
    plt.ylabel("Normalized Amplitude")
    plt.xlabel("Sample Index")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("No audio files found. Please generate synthetic data or add datasets.")
"""),
    md("""## 2. Sliding-Window Segmentation & Silence Removal
Long continuous underwater recordings are sliced into fixed **4.0-second windows** with **50% overlap** (2.0s hop). Windows consisting purely of quiet ocean background without signal energy are filtered out.
"""),
    code("""if sample_files:
    segments = segment_audio(
        audio_norm, sr=SAMPLE_RATE,
        window_sec=SEGMENT_DURATION,
        overlap=OVERLAP_FRACTION
    )
    print(f"Total valid segments extracted: {len(segments)}")
    if segments:
        seg0 = segments[0]
        print(f"Segment 0 shape: {seg0['audio'].shape} ({seg0['duration_sec']}s, time: {seg0['start_sec']}s - {seg0['end_sec']}s)")
"""),
    md("""## 3. Candidate Event Detection (Tonal Persistence & Energy)
Before feature extraction, a lightweight signal detector checks whether the segment contains high energy or persistent tonal lines (candidate vessel signature).
"""),
    code("""if sample_files and segments:
    event_info = detect_candidate_event(segments[0]['audio'], sr=SAMPLE_RATE)
    print("Candidate Event Detection Output:")
    for k, v in event_info.items():
        print(f"  {k:<25}: {v}")
"""),
    md("""## 4. Handcrafted Feature Extraction (~100 Acoustic Descriptors)
Each 4-second segment is transformed into an explainable numeric feature vector:
- **Cepstral:** MFCC 1-13 (mean, std, delta mean, delta std)
- **Spectral:** Centroid, Bandwidth, Rolloff, Flatness, Contrast
- **Temporal & Energy:** RMS energy, Zero Crossing Rate (ZCR), Dynamic Range
- **Tonal & Peak:** Top FFT peak frequencies, peak amplitudes, tonal persistence
- **Signal Quality:** Estimated SNR (dB), clipping %, silence %
"""),
    code("""if sample_files and segments:
    features = extract_all_features(segments[0]['audio'], sr=SAMPLE_RATE, clip_id="sample_01")
    print(f"Extracted {len(features)} handcrafted features:")
    df_feat_sample = pd.DataFrame([features])
    display(df_feat_sample.iloc[:, :12])
"""),
    md("""## 5. Load and Inspect Full Extracted Feature Matrix (`features.parquet`)
Let's inspect the complete feature table extracted across all recordings.
"""),
    code("""if FEATURES_PARQ.exists():
    df_feats = pd.read_parquet(FEATURES_PARQ)
    print(f"Loaded features from {FEATURES_PARQ.name}: Shape = {df_feats.shape}")
elif FEATURES_CSV.exists():
    df_feats = pd.read_csv(FEATURES_CSV)
    print(f"Loaded features from {FEATURES_CSV.name}: Shape = {df_feats.shape}")
else:
    print("Feature dataset not yet built. Building dataset now via build_dataset()...")
    from src.features.build_dataset import build_dataset
    df_feats = build_dataset(workers=1)

print(f"Total samples: {len(df_feats)}")
print(f"Class distribution in feature table:")
print(df_feats['label'].value_counts())
display(df_feats.head(3))
"""),
    md("""## 6. Distribution of Key Acoustic Features Across Classes
Let's compare feature distributions across the classes:
- **Spectral Centroid:** Vessel sounds typically concentrate power at lower frequencies compared to ambient noise.
- **Tonal Persistence:** Vessel sounds have sustained narrowband machinery tones.
"""),
    code("""if 'df_feats' in locals() and not df_feats.empty:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    key_features = ['spectral_centroid_mean', 'tonal_persistence', 'mfcc_1_mean']
    available_keys = [f for f in key_features if f in df_feats.columns]
    
    for ax, feat in zip(axes, available_keys):
        sns.boxplot(data=df_feats, x='label', y=feat, ax=ax, palette='Set2')
        ax.set_title(f"Distribution: {feat}", fontsize=10, fontweight='bold')
        ax.set_xlabel("Class")
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.show()
""")
]

with open(NOTEBOOKS_DIR / "02_signal_preprocessing_and_dsp_features.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_nb(nb2_cells), f, indent=2)

print("Created 02_signal_preprocessing_and_dsp_features.ipynb")
