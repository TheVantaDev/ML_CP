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
# NOTEBOOK 5: Offline Inference & Operator Decision Support
# ==============================================================================
nb5_cells = [
    md("""# 05. Offline Ground-Station Inference & Decision-Support System
### Sonobuoy Sound Detective: Offline Underwater Acoustic Event Detection
**System Layer:** End-to-End Ground-Station Pipeline with Hybrid Intelligence (ML + Acoustic Rules).

The project is not just a standalone classifier; it is a **decision-support application** designed to help operators analyze long sonobuoy audio streams without manual listening:
1. Ingesting a test sonobuoy hydrophone recording.
2. Loading the best trained model package from `models/`.
3. Preprocessing, segmenting, and running candidate event detection.
4. Extracting the registered handcrafted feature vector.
5. Computing class probabilities and confidence scores.
6. Applying the **Rule-Based Acoustic Cross-Check Engine**:
   - Highlighting low confidence (< 0.60) $\rightarrow$ *Review Required*.
   - Low SNR warning (< 3 dB) $\rightarrow$ *Low Quality Segment*.
   - Tonal consistency check: Flagging vessel predictions lacking persistent tonal machinery peaks.
   - Ambient contradiction check: Flagging ambient predictions that contain strong persistent tonals.
7. Generating the **Operator Decision Card** and simulating feedback capture for offline retraining.
"""),
    code("""import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path("..").resolve() if Path(".").resolve().name == "notebooks" else Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import joblib

from src.utils.config import (
    MODELS_DIR, PROCESSED_DIR, RAW_DIR, SAMPLE_RATE,
    SEGMENT_DURATION, OVERLAP_FRACTION, BANDPASS_LOW_HZ,
    BANDPASS_HIGH_HZ, LOW_CONFIDENCE_THRESHOLD, MIN_SNR_DB
)
from src.preprocessing.audio_utils import (
    to_mono, remove_dc_offset, apply_bandpass_filter, rms_normalize
)
from src.preprocessing.segmenter import segment_audio
from src.features.extractor import extract_all_features
from src.features.event_detection import detect_candidate_event

print("Pipeline modules loaded.")
"""),
    md("""## 1. Load Registered Model Pipeline
We locate and load the highest-performing registered model artifact from `models/`.
"""),
    code("""# Find available model directories
available_models = [d for d in MODELS_DIR.iterdir() if d.is_dir() and (d / "model.joblib").exists()]
print(f"Available registered models: {[d.name for d in available_models]}")

# Prefer random_forest or xgboost if present
target_dir = None
for pref in ["random_forest_v1", "xgboost_v1", "svm_v1"]:
    for d in available_models:
        if pref in d.name:
            target_dir = d
            break
    if target_dir:
        break

if not target_dir and available_models:
    target_dir = available_models[0]

if not target_dir:
    raise FileNotFoundError("No registered model found in models/. Run Notebook 04 or python -m src.models.train first.")

print(f"\nActive Production Model: {target_dir.name}")
model  = joblib.load(target_dir / "model.joblib")
scaler = joblib.load(target_dir / "scaler.joblib") if (target_dir / "scaler.joblib").exists() else None
le     = joblib.load(target_dir / "label_encoder.joblib")

with open(target_dir / "selected_features.json") as f:
    selected_features = json.load(f)

print(f"Model Type       : {type(model).__name__}")
print(f"Features Required: {len(selected_features)}")
print(f"Target Classes   : {list(le.classes_)}")
"""),
    md("""## 2. Ingest Test Sonobuoy Audio Recording
Let's select a recording to simulate an incoming hydrophone mission log.
"""),
    code("""# Look for an audio clip
test_files = list(PROCESSED_DIR.rglob("*.wav"))
if not test_files:
    test_files = list(RAW_DIR.rglob("*.wav"))

test_audio_path = test_files[0]
print(f"Ingesting Sonobuoy Recording: {test_audio_path.name}")

raw_audio, orig_sr = sf.read(str(test_audio_path))
audio_mono = to_mono(raw_audio)
if orig_sr != SAMPLE_RATE:
    audio_mono = librosa.resample(audio_mono, orig_sr=orig_sr, target_sr=SAMPLE_RATE)

# Conditioning
audio_clean = remove_dc_offset(audio_mono)
audio_clean = apply_bandpass_filter(audio_clean, sr=SAMPLE_RATE, low_hz=BANDPASS_LOW_HZ, high_hz=BANDPASS_HIGH_HZ)
audio_clean = rms_normalize(audio_clean)

duration_sec = len(audio_clean) / SAMPLE_RATE
print(f"Audio Duration: {duration_sec:.2f} seconds ({len(audio_clean)} samples at {SAMPLE_RATE} Hz)")
"""),
    md("""## 3. Sliding-Window Segmentation & Event Analysis
"""),
    code("""segments = segment_audio(
    audio_clean, sr=SAMPLE_RATE,
    window_sec=SEGMENT_DURATION,
    overlap=OVERLAP_FRACTION
)
print(f"Generated {len(segments)} analysis windows (4.0s duration, 2.0s step).")
"""),
    md("""## 4. Full Ground-Station Inference & Rule-Engine Cross-Check
For each segment:
1. Extract candidate event telemetry (tonal peak count, dominant frequency, SNR).
2. Extract the exact feature vector and scale.
3. Compute ML model probability distribution.
4. Run deterministic rule-based cross-checks.
"""),
    code("""def rule_based_cross_check(predicted_class, max_prob, event_telemetry, snr_est):
    warnings = []
    final_status = "Consistent"
    
    # Rule 1: Low Confidence Check
    if max_prob < LOW_CONFIDENCE_THRESHOLD:
        warnings.append(f"Low model confidence ({max_prob:.1%} < {LOW_CONFIDENCE_THRESHOLD:.0%}).")
        final_status = "Review Required"
        
    # Rule 2: Low SNR Check
    if snr_est < MIN_SNR_DB:
        warnings.append(f"Estimated SNR ({snr_est:.1f} dB) below minimum threshold ({MIN_SNR_DB} dB).")
        final_status = "Low Quality / Review Required"
        
    # Rule 3: Vessel Contradiction Check (Vessel predicted but 0 persistent tonal peaks)
    if predicted_class == "vessel_like" and event_telemetry.get("tonal_persistence", 0) < 0.15:
        warnings.append("Vessel predicted, but weak/no persistent narrowband tonal peaks detected.")
        final_status = "Review Required (Acoustic Anomaly)"
        
    # Rule 4: Ambient Contradiction Check (Ambient predicted with high persistent tonals)
    if predicted_class == "ambient" and event_telemetry.get("tonal_persistence", 0) > 0.60:
        warnings.append("Ambient predicted, but strong persistent machinery tone detected.")
        final_status = "Review Required (Possible Missed Target)"
        
    return final_status, warnings

detection_results = []

for i, seg in enumerate(segments):
    audio_seg = seg["audio"]
    
    # Event detection
    evt = detect_candidate_event(audio_seg, sr=SAMPLE_RATE)
    
    # Handcrafted feature extraction
    feats = extract_all_features(audio_seg, sr=SAMPLE_RATE, clip_id=f"seg_{i}")
    snr_est = feats.get("snr_estimate", 10.0)
    
    # Align features with training column order
    x_vec = np.array([feats.get(col, 0.0) for col in selected_features]).reshape(1, -1)
    x_vec = np.nan_to_num(x_vec, nan=0.0, posinf=0.0, neginf=0.0)
    
    if scaler:
        x_vec = scaler.transform(x_vec)
        
    # ML Prediction
    probs = model.predict_proba(x_vec)[0]
    pred_idx = np.argmax(probs)
    pred_class = le.inverse_transform([pred_idx])[0]
    conf = probs[pred_idx]
    
    # Rule-engine validation
    status, warns = rule_based_cross_check(pred_class, conf, evt, snr_est)
    
    detection_results.append({
        "seg_index": i,
        "time_window": f"{seg['start_sec']:.1f}s - {seg['end_sec']:.1f}s",
        "predicted_class": pred_class,
        "confidence": conf,
        "dominant_freq_hz": round(evt.get("dominant_freq_hz", 0)),
        "snr_db": round(snr_est, 1),
        "rule_status": status,
        "warnings": warns,
        "probs": dict(zip(le.classes_, probs))
    })

df_dets = pd.DataFrame(detection_results)
display(df_dets[["seg_index", "time_window", "predicted_class", "confidence", "dominant_freq_hz", "snr_db", "rule_status"]])
"""),
    md("""## 5. Operator Decision Card Display
Mocking up the exact telemetry card displayed to an acoustic operator in the ground-station dashboard.
"""),
    code("""def print_decision_card(det):
    print("="*60)
    print(f"  OPERATOR DECISION CARD — WINDOW: {det['time_window']}")
    print("="*60)
    print(f"  Predicted Classification: {det['predicted_class'].upper()}")
    print(f"  Model Confidence        : {det['confidence']:.1%}")
    print(f"  Dominant Frequency      : {det['dominant_freq_hz']} Hz")
    print(f"  Estimated SNR           : {det['snr_db']} dB")
    print(f"  Rule-Engine Status      : {det['rule_status']}")
    if det['warnings']:
        print("  Warnings & Flags        :")
        for w in det['warnings']:
            print(f"    ⚠️  {w}")
    else:
        print("  Warnings & Flags        : None (Acoustic evidence consistent)")
    print("\n  Class Probabilities     :")
    for cls, p in det['probs'].items():
        bar = "█" * int(p * 25)
        print(f"    {cls:<14} [{p:>5.1%}] {bar}")
    print("="*60)

# Display card for segment 0
if detection_results:
    print_decision_card(detection_results[0])
"""),
    md("""## 6. Human-in-the-Loop Operator Feedback Simulation
The operator reviews the detection, accepts, rejects, or enters a corrected label, which is logged to an audit trail for future controlled model retraining.
"""),
    code("""# Simulate operator action
simulated_feedback = {
    "detection_id": "DET-001",
    "file_analyzed": test_audio_path.name,
    "time_window": detection_results[0]["time_window"],
    "predicted_label": detection_results[0]["predicted_class"],
    "operator_decision": "ACCEPT", # or "REJECT" / "CORRECT"
    "corrected_label": detection_results[0]["predicted_class"],
    "operator_notes": "Continuous diesel machinery signature verified on LOFAR.",
    "reviewed_by": "operator_01"
}

print("Operator Feedback Audit Entry Recorded:")
print(json.dumps(simulated_feedback, indent=2))
""")
]

with open(NOTEBOOKS_DIR / "05_inference_and_decision_support.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_nb(nb5_cells), f, indent=2)

print("Created 05_inference_and_decision_support.ipynb")
