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
# FINAL MODEL COMPARISON NOTEBOOK
# ==============================================================================
nb_comp_cells = [
    md("""# 06A. Master Classical ML Model Comparison & Benchmark
### Sonobuoy Sound Detective: Offline Underwater Acoustic Event Detection
**Role in Project:** The Final Showdown — Benchmarking all 5 candidate algorithms side-by-side.

This notebook aggregates the results from each team member's assigned algorithm:
1. **Logistic Regression** (`01_logistic_regression/`)
2. **K-Nearest Neighbors** (`02_k_nearest_neighbors/`)
3. **Support Vector Machine** (`03_support_vector_machine/`)
4. **Random Forest** (`04_random_forest/`)
5. **XGBoost** (`05_xgboost/`)

We evaluate all 5 models on the identical held-out test set and produce:
- **Master Comparison Table:** Accuracy, Macro Precision, Macro Recall, Macro F1, ROC-AUC, Inference Latency.
- **Side-by-Side Confusion Matrices:** Direct visual comparison of misclassifications.
- **Unified Multi-Model ROC Curves:** All 5 models plotted together to compare separation power.
- **Latency vs Accuracy Tradeoff Scatter Plot:** Evaluating operational deployment readiness for offline hardware.
- **Winner Declaration:** Declaring the top candidate for the ground-station deployment.
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
print("Final comparison suite loaded.")
"""),
    md("""## 1. Load Test Dataset and Standardized Features"""),
    code("""parq_path = Path("../data/features/features.parquet")
csv_path  = Path("../data/features/features.csv")
sel_path  = Path("../data/features/selected_features.json")

df = pd.read_parquet(parq_path) if parq_path.exists() else pd.read_csv(csv_path)

with open(sel_path) as f:
    raw_sel = json.load(f)
    feature_cols = raw_sel["selected_features"] if isinstance(raw_sel, dict) and "selected_features" in raw_sel else raw_sel

# Use test split
test_mask = df["split"] == "test" if "split" in df.columns else df.index.isin(df.sample(frac=0.15, random_state=RANDOM_SEED).index)
df_test = df[test_mask].copy()

classes = sorted(df["label"].unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
y_test = df_test["label"].map(class_to_idx).values
X_test_raw = np.nan_to_num(df_test[feature_cols].values, nan=0.0)

print(f"Loaded held-out test set: {len(df_test)} samples across classes: {classes}")
"""),
    md("""## 2. Load Models from Each Algorithm Folder
Loads trained models from `../01_logistic_regression/models/`, `../02_k_nearest_neighbors/models/`, etc.
"""),
    code("""model_configs = [
    ("Logistic Regression", Path("../01_logistic_regression/models")),
    ("K-Nearest Neighbors", Path("../02_k_nearest_neighbors/models")),
    ("Support Vector Machine", Path("../03_support_vector_machine/models")),
    ("Random Forest", Path("../04_random_forest/models")),
    ("XGBoost", Path("../05_xgboost/models"))
]

loaded_models = {}
loaded_scalers = {}

for name, m_dir in model_configs:
    m_file = m_dir / "model.joblib"
    s_file = m_dir / "scaler.joblib"
    
    if not m_file.exists():
        fallback_name = name.lower().replace(" ", "_") + "_v1"
        fallback_dir = Path(f"../models/{fallback_name}")
        if (fallback_dir / "model.joblib").exists():
            m_file = fallback_dir / "model.joblib"
            s_file = fallback_dir / "scaler.joblib"

    if m_file.exists():
        loaded_models[name] = joblib.load(m_file)
        loaded_scalers[name] = joblib.load(s_file) if s_file.exists() else None
        print(f"  [OK] Loaded: {name} from {m_file}")
    else:
        print(f"  [MISSING] {name} not found. Please execute the notebook in {m_dir.parent}!")

print(f"\\nSuccessfully loaded {len(loaded_models)} / 5 classical models.")
"""),
    md("""## 3. Run Comprehensive Benchmark on Held-Out Test Set"""),
    code("""records = []
predictions = {}
probabilities = {}

for name, model in loaded_models.items():
    scaler = loaded_scalers.get(name)
    X_test_scaled = scaler.transform(X_test_raw) if scaler else X_test_raw
    
    # Latency benchmark
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
df_results.to_csv("reports/master_model_comparison.csv", index=False)

print("="*80)
print("  MASTER CLASSICAL ML ALGORITHM BENCHMARK TABLE")
print("="*80)
display(df_results.style.format({
    "Test Accuracy": "{:.2%}",
    "Macro Precision": "{:.2%}",
    "Macro Recall": "{:.2%}",
    "Macro F1": "{:.2%}",
    "ROC-AUC (Macro)": "{:.3f}",
    "Latency (ms/segment)": "{:.2f} ms"
}))
"""),
    md("""## 4. Side-by-Side Confusion Matrices (All 5 Algorithms)"""),
    code("""n_models = len(predictions)
fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4))
if n_models == 1:
    axes = [axes]

for ax, (name, y_pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, y_pred, labels=range(len(classes)))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm, nan=0.0)
    
    sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues", cbar=False, ax=ax,
                xticklabels=classes, yticklabels=classes)
    ax.set_title(name, fontsize=10, fontweight='bold')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

plt.tight_layout()
plt.savefig("plots/side_by_side_confusion_matrices.png", dpi=300, bbox_inches="tight")
plt.show()
"""),
    md("""## 5. Unified ROC Curves (Multi-Model Comparison)"""),
    code("""plt.figure(figsize=(8, 6))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for (name, y_prob), color in zip(probabilities.items(), colors):
    if y_prob is not None:
        all_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(classes)):
            y_bin = (y_test == i).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr /= len(classes)
        macro_auc = auc(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr, lw=2, color=color, label=f"{name} (Macro AUC = {macro_auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Comparison of All 5 Classical ML Models (Macro ROC Curves)", fontsize=12, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/all_models_roc_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
"""),
    md("""## 6. Latency vs Macro-F1 Tradeoff"""),
    code("""plt.figure(figsize=(8, 5))
for _, row in df_results.iterrows():
    plt.scatter(row['Latency (ms/segment)'], row['Macro F1'], s=150, alpha=0.8, label=row['Model'])
    plt.annotate(row['Model'], (row['Latency (ms/segment)'] + 0.05, row['Macro F1']), fontsize=10)

plt.xlabel("Inference Latency (ms / 4s segment) [Lower is Better]")
plt.ylabel("Macro F1-Score [Higher is Better]")
plt.title("Trade-off Analysis: Classification Quality vs Processing Speed", fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/latency_vs_accuracy_tradeoff.png", dpi=300, bbox_inches="tight")
plt.show()
"""),
    md("""## 7. Official Winner Selection & Academic Conclusion
Based on the multi-class benchmark, we select the top algorithm (**Random Forest** or **XGBoost**) for production deployment in the Ground-Station Decision Support System.
""")
]

with open("06_final_comparison_and_system/final_model_comparison.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_nb(nb_comp_cells), f, indent=2)
print("Created 06_final_comparison_and_system/final_model_comparison.ipynb")

# ==============================================================================
# DECISION SUPPORT SYSTEM NOTEBOOK
# ==============================================================================
nb_dss_cells = [
    md("""# 06B. Ground-Station Decision Support & Acoustic Cross-Check Engine
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
    code("""model_candidates = [
    Path("../05_xgboost/models"),
    Path("../04_random_forest/models"),
    Path("../03_support_vector_machine/models"),
    Path("../models/random_forest_v1"),
    Path("../models/xgboost_v1")
]

target_dir = None
for d in model_candidates:
    if (d / "model.joblib").exists():
        target_dir = d
        break

if not target_dir:
    raise FileNotFoundError("No trained model found! Please run an algorithm notebook first.")

print(f"Deploying Active Production Model from: {target_dir}")
model  = joblib.load(target_dir / "model.joblib")
scaler = joblib.load(target_dir / "scaler.joblib") if (target_dir / "scaler.joblib").exists() else None
le     = joblib.load(target_dir / "label_encoder.joblib") if (target_dir / "label_encoder.joblib").exists() else None

with open(Path("../data/features/selected_features.json")) as f:
    raw_sel = json.load(f)
    selected_features = raw_sel["selected_features"] if isinstance(raw_sel, dict) and "selected_features" in raw_sel else raw_sel

classes = list(le.classes_) if le else ["ambient", "unknown", "vessel_like"]
print(f"Model: {type(model).__name__} | Features: {len(selected_features)} | Target Classes: {classes}")
"""),
    md("""## 2. Ingest Sample Mission Log Audio Recording"""),
    code("""audio_files = list(PROCESSED_DIR.rglob("*.wav"))
if not audio_files:
    audio_files = list(RAW_DIR.rglob("*.wav"))

mission_audio_path = audio_files[0]
print(f"Processing Sonobuoy Hydrophone Stream: {mission_audio_path.name}")

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
    pred_class = classes[pred_idx]
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

with open("06_final_comparison_and_system/decision_support_system.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_nb(nb_dss_cells), f, indent=2)
print("Created 06_final_comparison_and_system/decision_support_system.ipynb")
