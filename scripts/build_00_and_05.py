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
# 00. Data Prep Notebook
# ==============================================================================
nb00_cells = [
    md("""# 00. Data Ingestion, Signal Preprocessing & Handcrafted Feature Extraction
### Sonobuoy Sound Detective: Offline Underwater Acoustic Event Detection
**Role in Project:** Shared Data Preparation Step (Produces the benchmark dataset for all 8 algorithms)

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

from src.utils.config import (
    DATA_DIR, RAW_DIR, PROCESSED_DIR, FEATURES_DIR, METADATA_DIR,
    MASTER_METADATA, FEATURES_PARQ, FEATURES_CSV, SEL_FEATURES,
    SAMPLE_RATE, SEGMENT_DURATION, OVERLAP_FRACTION, BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ,
    RANDOM_SEED
)
from src.features.build_dataset import build_dataset
from src.features.selection import select_features

Path("plots").mkdir(parents=True, exist_ok=True)
print("Environment configured.")
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
"""),
    md("""## 2. Inspect Waveforms and Spectrograms (Vessel vs Ambient)"""),
    code("""vessel_files = list((PROCESSED_DIR / "vessel_like").glob("*.wav"))
ambient_files = list((PROCESSED_DIR / "ambient").glob("*.wav"))

if vessel_files and ambient_files:
    fig, axes = plt.subplots(2, 2, figsize=(14, 6))
    
    y_vessel, sr = librosa.load(str(vessel_files[0]), sr=SAMPLE_RATE)
    axes[0, 0].plot(np.linspace(0, len(y_vessel)/sr, len(y_vessel)), y_vessel, color='#2b5c8f', lw=0.8)
    axes[0, 0].set_title(f"Vessel Waveform: {vessel_files[0].name}", fontsize=10, fontweight='bold')
    axes[0, 0].set_ylabel("Amplitude")
    
    D_vessel = librosa.amplitude_to_db(np.abs(librosa.stft(y_vessel)), ref=np.max)
    librosa.display.specshow(D_vessel, sr=sr, x_axis='time', y_axis='hz', ax=axes[0, 1], cmap='magma')
    axes[0, 1].set_title("Vessel Spectrogram (LOFAR Tonals)", fontsize=10, fontweight='bold')
    axes[0, 1].set_ylim(0, 3000)
    
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
    md("""## 3. Extract Features and Build the Shared Dataset"""),
    code("""print("Extracting features from processed audio segments...")
df_features = build_dataset(workers=1)
print(f"Extracted feature dataset shape: {df_features.shape}")
display(df_features.head(3))
"""),
    md("""## 4. Run Multi-Criteria Feature Selection"""),
    code("""selected_cols = select_features()
print(f"Top 35 Selected Features for All Algorithm Models:")
for i, col in enumerate(selected_cols[:15], 1):
    print(f"  {i:>2}. {col}")
print(f"... and {len(selected_cols)-15} more. Saved to {SEL_FEATURES}")
"""),
    md("""## 5. Dataset Ready for Team Members!
All 4 team members can now proceed to their individual algorithm folders:
- Person 1: `01_person1_lr_and_knn/` (Logistic Regression + KNN)
- Person 2: `02_person2_svm_and_naive_bayes/` (SVM + Naive Bayes)
- Person 3: `03_person3_decision_tree_and_random_forest/` (Decision Tree + Random Forest)
- Person 4: `04_person4_adaboost_and_xgboost/` (AdaBoost + XGBoost)
""")
]

with open("00_data_prep_and_features/data_prep_and_feature_extraction.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_nb(nb00_cells), f, indent=2)

# ==============================================================================
# 05. Master Comparison Notebook (8 Models)
# ==============================================================================
nb_comp_cells = [
    md("""# 05A. Master Classical ML Benchmark: 8 Models Comparison
### Sonobuoy Sound Detective: Offline Underwater Acoustic Event Detection
**Role in Project:** The Final Showdown — Benchmarking all 8 algorithms across 4 teammates side-by-side.

### Models Benchmarked:
1. **Person 1:** Logistic Regression (`01_person1_lr_and_knn`)
2. **Person 1:** K-Nearest Neighbors (`01_person1_lr_and_knn`)
3. **Person 2:** Support Vector Machine (RBF) (`02_person2_svm_and_naive_bayes`)
4. **Person 2:** Gaussian Naive Bayes (`02_person2_svm_and_naive_bayes`)
5. **Person 3:** Decision Tree (`03_person3_decision_tree_and_random_forest`)
6. **Person 3:** Random Forest (`03_person3_decision_tree_and_random_forest`)
7. **Person 4:** AdaBoost (`04_person4_adaboost_and_xgboost`)
8. **Person 4:** XGBoost (`04_person4_adaboost_and_xgboost`)

We evaluate all 8 models on the identical held-out test set and produce:
- **Master Comparison Table:** Test Accuracy, Macro Precision, Macro Recall, Macro F1, Multi-Class ROC-AUC, Inference Latency.
- **Side-by-Side Confusion Matrices:** Direct visual comparison across all 8 algorithms.
- **Unified Multi-Model ROC Curves:** All 8 models plotted on the same figure.
- **Latency vs Accuracy Tradeoff Scatter Plot.**
- **Winner Declaration:** Declaring the top candidate for ground-station deployment.
"""),
    code("""import os
import sys
import time
import json
from pathlib import Path

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)

Path("plots").mkdir(parents=True, exist_ok=True)
Path("reports").mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
print("Final 8-model comparison suite loaded.")
"""),
    md("""## 1. Load Test Dataset and Standardized Features"""),
    code("""parq_path = Path("../data/features/features.parquet")
csv_path  = Path("../data/features/features.csv")
sel_path  = Path("../data/features/selected_features.json")

df = pd.read_parquet(parq_path) if parq_path.exists() else pd.read_csv(csv_path)

with open(sel_path) as f:
    raw_sel = json.load(f)
    feature_cols = raw_sel["selected_features"] if isinstance(raw_sel, dict) and "selected_features" in raw_sel else raw_sel

test_mask = df["split"] == "test" if "split" in df.columns else df.index.isin(df.sample(frac=0.15, random_state=RANDOM_SEED).index)
df_test = df[test_mask].copy()

classes = sorted(df["label"].unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
y_test = df_test["label"].map(class_to_idx).values
X_test_raw = np.nan_to_num(df_test[feature_cols].values, nan=0.0)

print(f"Loaded held-out test set: {len(df_test)} samples across classes: {classes}")
"""),
    md("""## 2. Load All 8 Models from Teammate Folders"""),
    code("""model_registry_list = [
    ("Logistic Regression", Path("../01_person1_lr_and_knn/models"), "logistic_regression"),
    ("K-Nearest Neighbors", Path("../01_person1_lr_and_knn/models"), "knn"),
    ("Support Vector Machine", Path("../02_person2_svm_and_naive_bayes/models"), "svm_rbf"),
    ("Gaussian Naive Bayes", Path("../02_person2_svm_and_naive_bayes/models"), "naive_bayes"),
    ("Decision Tree", Path("../03_person3_decision_tree_and_random_forest/models"), "decision_tree"),
    ("Random Forest", Path("../03_person3_decision_tree_and_random_forest/models"), "random_forest"),
    ("AdaBoost", Path("../04_person4_adaboost_and_xgboost/models"), "adaboost"),
    ("XGBoost", Path("../04_person4_adaboost_and_xgboost/models"), "xgboost"),
]

loaded_models = {}
loaded_scalers = {}

for name, m_dir, slug in model_registry_list:
    m_file = m_dir / f"model_{slug}.joblib"
    s_file = m_dir / f"scaler_{slug}.joblib"
    
    if not m_file.exists():
        m_file = m_dir / "model.joblib"
        s_file = m_dir / "scaler.joblib"

    if m_file.exists():
        loaded_models[name] = joblib.load(m_file)
        loaded_scalers[name] = joblib.load(s_file) if s_file.exists() else None
        print(f"  [OK] Loaded: {name:<25} from {m_dir.parent.name}")
    else:
        print(f"  [MISSING] {name} not found in {m_dir}!")

print(f"\\nSuccessfully loaded {len(loaded_models)} / 8 classical algorithms.")
"""),
    md("""## 3. Run Comprehensive 8-Model Benchmark on Held-Out Test Set"""),
    code("""records = []
predictions = {}
probabilities = {}

for name, model in loaded_models.items():
    scaler = loaded_scalers.get(name)
    X_test_scaled = scaler.transform(X_test_raw) if scaler else X_test_raw
    
    t_start = time.perf_counter()
    n_iters = 50
    for _ in range(n_iters):
        _ = model.predict(X_test_scaled[:1])
    latency_ms = ((time.perf_counter() - t_start) / n_iters) * 1000.0
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled) if hasattr(model, "predict_proba") else None
    
    predictions[name] = y_pred
    probabilities[name] = y_prob
    
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
    
    auc_val = None
    if y_prob is not None:
        try:
            auc_val = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        except Exception:
            auc_val = None
            
    records.append({
        "Model": name,
        "Test Accuracy": acc,
        "Macro Precision": prec,
        "Macro Recall": rec,
        "Macro F1": f1,
        "ROC-AUC (Macro)": auc_val if auc_val is not None else 0.0,
        "Latency (ms/segment)": latency_ms
    })

df_results = pd.DataFrame(records).sort_values("Macro F1", ascending=False).reset_index(drop=True)
df_results.to_csv("reports/master_8_models_comparison.csv", index=False)

print("="*85)
print("  MASTER 8-MODEL CLASSICAL ALGORITHM BENCHMARK TABLE")
print("="*85)
display(df_results.style.format({
    "Test Accuracy": "{:.2%}",
    "Macro Precision": "{:.2%}",
    "Macro Recall": "{:.2%}",
    "Macro F1": "{:.2%}",
    "ROC-AUC (Macro)": "{:.3f}",
    "Latency (ms/segment)": "{:.2f} ms"
}))
"""),
    md("""## 4. Side-by-Side Confusion Matrices (All 8 Algorithms)"""),
    code("""n_models = len(predictions)
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for ax, (name, y_pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, y_pred, labels=range(len(classes)))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm, nan=0.0)
    
    sns.heatmap(cm_norm, annot=True, fmt=".0%", cmap="Blues", cbar=False, ax=ax,
                xticklabels=classes, yticklabels=classes)
    ax.set_title(name, fontsize=10, fontweight='bold')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

plt.tight_layout()
plt.savefig("plots/all_8_confusion_matrices.png", dpi=300, bbox_inches="tight")
plt.show()
"""),
    md("""## 5. Unified ROC Curves (All 8 Algorithms Comparison)"""),
    code("""plt.figure(figsize=(9, 6))
palette = sns.color_palette("tab10", len(probabilities))

for (name, y_prob), color in zip(probabilities.items(), palette):
    if y_prob is not None:
        all_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(classes)):
            y_bin = (y_test == i).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr /= len(classes)
        macro_auc = auc(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr, lw=2, color=color, label=f"{name} (AUC = {macro_auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Comparison of All 8 Classical ML Models (Macro ROC Curves)", fontsize=12, fontweight='bold')
plt.legend(loc="lower right", fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/all_8_models_roc_curves.png", dpi=300, bbox_inches="tight")
plt.show()
"""),
    md("""## 6. Latency vs Macro-F1 Trade-off Analysis"""),
    code("""plt.figure(figsize=(9, 5))
for _, row in df_results.iterrows():
    plt.scatter(row['Latency (ms/segment)'], row['Macro F1'], s=160, alpha=0.85, label=row['Model'])
    plt.annotate(row['Model'], (row['Latency (ms/segment)'] + 0.05, row['Macro F1']), fontsize=9)

plt.xlabel("Inference Latency (ms / 4s segment) [Lower is Better]")
plt.ylabel("Macro F1-Score [Higher is Better]")
plt.title("Deployment Efficiency: Classification Quality vs Processing Speed", fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/latency_vs_f1_tradeoff.png", dpi=300, bbox_inches="tight")
plt.show()
"""),
    md("""## 7. Official Winner Selection
The winning model among all 8 algorithms is selected for deployment in the Ground-Station Decision Support System.
""")
]

with open("05_final_comparison_and_system/final_model_comparison.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_nb(nb_comp_cells), f, indent=2)

# ==============================================================================
# Decision Support System Notebook
# ==============================================================================
nb_dss_cells = [
    md("""# 05B. Ground-Station Decision Support & Acoustic Cross-Check Engine
### Sonobuoy Sound Detective: Offline Underwater Acoustic Event Detection
**Role in Project:** Operational Deployment Prototype (Hybrid ML + Deterministic Signal Rules)

This notebook simulates the offline sonobuoy ground-station operator workstation:
1. Ingesting raw acoustic audio streams.
2. Slicing into 4.0s windows & running event candidate detection.
3. Feature extraction & model inference using the winning classifier.
4. **Deterministic Acoustic Cross-Check Rules:**
   - Confidence threshold check (< 0.60 $\rightarrow$ *Review Required*).
   - SNR estimate check (< 3.0 dB $\rightarrow$ *Low Quality Warning*).
   - Tonal consistency validation: Flags vessel predictions lacking narrowband persistent tonals.
   - Ambient contradiction validation: Flags ambient predictions containing strong machinery tones.
5. Interactive Operator Decision Cards and feedback audit logging.
"""),
    code("""import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import joblib

from src.utils.config import (
    PROCESSED_DIR, RAW_DIR, SAMPLE_RATE, SEGMENT_DURATION,
    OVERLAP_FRACTION, BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ,
    LOW_CONFIDENCE_THRESHOLD, MIN_SNR_DB
)
from src.preprocessing.audio_utils import to_mono, remove_dc_offset, apply_bandpass_filter, rms_normalize
from src.preprocessing.segmenter import segment_audio
from src.features.extractor import extract_all_features
from src.features.event_detection import detect_candidate_event

print("Decision support system loaded.")
"""),
    md("""## 1. Load Winning Model from Final Comparison"""),
    code("""model_search_paths = [
    Path("../04_person4_adaboost_and_xgboost/models/model_xgboost.joblib"),
    Path("../03_person3_decision_tree_and_random_forest/models/model_random_forest.joblib"),
    Path("../02_person2_svm_and_naive_bayes/models/model_svm_rbf.joblib"),
    Path("../01_person1_lr_and_knn/models/model_logistic_regression.joblib"),
]

model_path = None
for p in model_search_paths:
    if p.exists():
        model_path = p
        break

if not model_path:
    # Any available model
    any_models = list(Path("..").rglob("model*.joblib"))
    if any_models:
        model_path = any_models[0]

if not model_path:
    raise FileNotFoundError("No trained model found! Please execute an algorithm notebook first.")

print(f"Active Production Model: {model_path}")
model = joblib.load(model_path)

# Scaler
scaler_path = model_path.parent / model_path.name.replace("model_", "scaler_")
scaler = joblib.load(scaler_path) if scaler_path.exists() else None

with open(Path("../data/features/selected_features.json")) as f:
    raw_sel = json.load(f)
    selected_features = raw_sel["selected_features"] if isinstance(raw_sel, dict) and "selected_features" in raw_sel else raw_sel

classes = ["ambient", "unknown", "vessel_like"]
print(f"Model Type: {type(model).__name__} | Features: {len(selected_features)}")
"""),
    md("""## 2. Ingest Sample Mission Log Audio Recording"""),
    code("""audio_files = list(PROCESSED_DIR.rglob("*.wav"))
if not audio_files:
    audio_files = list(RAW_DIR.rglob("*.wav"))

mission_audio_path = audio_files[0]
print(f"Processing Hydrophone Stream: {mission_audio_path.name}")

y_raw, orig_sr = sf.read(str(mission_audio_path))
y_mono = to_mono(y_raw)
if orig_sr != SAMPLE_RATE:
    y_mono = librosa.resample(y_mono, orig_sr=orig_sr, target_sr=SAMPLE_RATE)

y_clean = remove_dc_offset(y_mono)
y_clean = apply_bandpass_filter(y_clean, sr=SAMPLE_RATE, low_hz=BANDPASS_LOW_HZ, high_hz=BANDPASS_HIGH_HZ)
y_clean = rms_normalize(y_clean)

segments = segment_audio(y_clean, sr=SAMPLE_RATE, window_sec=SEGMENT_DURATION, overlap=OVERLAP_FRACTION)
print(f"Audio Duration: {len(y_clean)/SAMPLE_RATE:.2f}s | Sliced into {len(segments)} analysis windows.")
"""),
    md("""## 3. Run Inference Pipeline with Deterministic Acoustic Cross-Check Rules"""),
    code("""def evaluate_acoustic_rules(pred_class, conf, evt, snr_est):
    warnings = []
    status = "Consistent"
    
    if conf < LOW_CONFIDENCE_THRESHOLD:
        warnings.append(f"Model confidence ({conf:.1%}) below required threshold ({LOW_CONFIDENCE_THRESHOLD:.0%}).")
        status = "Review Required"
        
    if snr_est < MIN_SNR_DB:
        warnings.append(f"Estimated SNR ({snr_est:.1f} dB) below minimum threshold ({MIN_SNR_DB} dB).")
        status = "Low Quality / Review Required"
        
    if pred_class == "vessel_like" and evt.get("tonal_persistence", 0) < 0.15:
        warnings.append("Vessel predicted, but weak/no persistent narrowband tonal peaks detected.")
        status = "Review Required (Acoustic Anomaly)"
        
    if pred_class == "ambient" and evt.get("tonal_persistence", 0) > 0.60:
        warnings.append("Ambient predicted, but strong persistent machinery tone detected.")
        status = "Review Required (Possible Missed Target)"
        
    return status, warnings

results = []
for i, seg in enumerate(segments):
    audio_seg = seg["audio"]
    evt = detect_candidate_event(audio_seg, sr=SAMPLE_RATE)
    feats = extract_all_features(audio_seg, sr=SAMPLE_RATE, clip_id=f"seg_{i}")
    snr_est = feats.get("snr_estimate", 10.0)
    
    x_vec = np.array([feats.get(col, 0.0) for col in selected_features]).reshape(1, -1)
    x_vec = np.nan_to_num(x_vec, nan=0.0)
    if scaler:
        x_vec = scaler.transform(x_vec)
        
    probs = model.predict_proba(x_vec)[0]
    pred_idx = np.argmax(probs)
    pred_class = classes[pred_idx] if pred_idx < len(classes) else "unknown"
    conf = probs[pred_idx]
    
    status, warns = evaluate_acoustic_rules(pred_class, conf, evt, snr_est)
    
    results.append({
        "Detection ID": f"DET-{i+1:03d}",
        "Window": f"{seg['start_sec']:.1f}s - {seg['end_sec']:.1f}s",
        "Prediction": pred_class,
        "Confidence": f"{conf:.1%}",
        "Dominant Freq": f"{evt.get('dominant_freq_hz', 0):.0f} Hz",
        "SNR (dB)": f"{snr_est:.1f}",
        "Rule Status": status,
        "Warnings": "; ".join(warns) if warns else "None",
        "_probs": dict(zip(classes, probs))
    })

df_detections = pd.DataFrame(results)
display(df_detections[["Detection ID", "Window", "Prediction", "Confidence", "Dominant Freq", "SNR (dB)", "Rule Status", "Warnings"]])
"""),
    md("""## 4. Operator Telemetry Card & Feedback Logging"""),
    code("""def show_telemetry_card(det):
    print("="*65)
    print(f"  OPERATOR DECISION TELEMETRY — {det['Detection ID']} ({det['Window']})")
    print("="*65)
    print(f"  Class Classification  : {det['Prediction'].upper()}")
    print(f"  Confidence            : {det['Confidence']}")
    print(f"  Dominant Frequency    : {det['Dominant Freq']}")
    print(f"  Estimated SNR         : {det['SNR (dB)']} dB")
    print(f"  Rule Engine Status    : {det['Rule Status']}")
    print(f"  Acoustic Warnings     : {det['Warnings']}")
    print("\\n  Posterior Probabilities:")
    for cls, p in det['_probs'].items():
        bar = "█" * int(p * 20)
        print(f"    {cls:<14} [{p:>5.1%}] {bar}")
    print("="*65)

if results:
    show_telemetry_card(results[0])
    
    audit_entry = {
        "detection_id": results[0]["Detection ID"],
        "window": results[0]["Window"],
        "predicted_label": results[0]["Prediction"],
        "operator_decision": "ACCEPT",
        "corrected_label": results[0]["Prediction"],
        "notes": "Verified narrow-band shaft line at 180 Hz on LOFAR.",
        "operator": "acoustic_analyst_01"
    }
    print("\\nAudit Log Record:")
    print(json.dumps(audit_entry, indent=2))
""")
]

with open("05_final_comparison_and_system/decision_support_system.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_nb(nb_dss_cells), f, indent=2)

print("Created 00 and 05 notebooks successfully!")
