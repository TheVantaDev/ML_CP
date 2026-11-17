"""
Flask backend for Sonobuoy Sound Detective
==========================================
Dual-Engine System:
  Branch A — Random Forest (Discriminative, 99.37% accuracy)
  Branch B — GMM Generative Model (Probabilistic, anomaly detection)

Both models are loaded at startup and run on every uploaded audio file.
The /predict endpoint returns results from BOTH models.
"""

import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import librosa
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Branch A — Random Forest
RF_MODEL_DIR = ROOT / "03_person3_decision_tree_and_random_forest" / "models"
RF_MODEL_PATH  = RF_MODEL_DIR / "model_random_forest.joblib"
RF_SCALER_PATH = RF_MODEL_DIR / "scaler_random_forest.joblib"
RF_LE_PATH     = RF_MODEL_DIR / "label_encoder_random_forest.joblib"

# Branch B — GMM
GMM_DIR           = ROOT / "06_gmm_model" / "models"
GMM_SCALER_PATH   = GMM_DIR / "gmm_scaler.joblib"
GMM_METADATA_PATH = GMM_DIR / "gmm_metadata.json"

# Shared feature list
SEL_FEAT = ROOT / "data" / "features" / "selected_features.json"


# ── Load models once at startup ───────────────────────────────────────────────
print("=" * 55, flush=True)
print("  Loading Branch A: Random Forest …", flush=True)
rf_model  = joblib.load(RF_MODEL_PATH)
rf_scaler = joblib.load(RF_SCALER_PATH)
rf_le     = joblib.load(RF_LE_PATH)
print(f"  ✓ RF ready | Classes: {list(rf_le.classes_)}", flush=True)

print("  Loading Branch B: GMM …", flush=True)
with open(GMM_METADATA_PATH) as f:
    gmm_meta = json.load(f)

GMM_CLASSES         = gmm_meta["classes"]
ANOMALY_THRESHOLD   = gmm_meta["anomaly_threshold"]
# GMM uses its own scaler — trained on live-extracted features, same pipeline as Flask
gmm_scaler          = joblib.load(GMM_SCALER_PATH)
gmm_models = {}
for label in GMM_CLASSES:
    path = GMM_DIR / f"gmm_{label}.joblib"
    gmm_models[label] = joblib.load(path)
print(f"  ✓ GMM ready | Classes: {GMM_CLASSES} | Anomaly threshold: {ANOMALY_THRESHOLD:.4f}", flush=True)

with open(SEL_FEAT) as f:
    raw = json.load(f)
    FEATURES = raw["selected_features"] if isinstance(raw, dict) else raw
print(f"  ✓ Features: {len(FEATURES)}", flush=True)
print("=" * 55, flush=True)


# ── Feature extraction & Preprocessing ─────────────────────────────────────────
from src.preprocessing.audio_utils import preprocess
from src.preprocessing.segmenter import segment_signal
from src.features.extractor import extract_features

SAMPLE_RATE = 22050
SEGMENT_SEC = 4.0


def process_and_extract_features(file_path: Path | str) -> tuple[np.ndarray, float, int]:
    """
    Runs the exact preprocessing pipeline from the project DSP stack:
      1. Load audio & convert to mono
      2. Resample to 22,050 Hz
      3. Remove DC offset
      4. Apply Butterworth bandpass filter
      5. Apply spectral noise gating
      6. RMS energy normalization (target_rms = 0.1)
      7. Segment to 4.0s window
      8. Extract 66 audio features
    """
    signal, sr = preprocess(file_path, sr=SAMPLE_RATE)
    duration_s = round(len(signal) / sr, 2)
    
    segments = segment_signal(signal, sr=sr, duration=SEGMENT_SEC, overlap=0.0)
    if segments:
        window = segments[0][2]
    else:
        target_len = int(SAMPLE_RATE * SEGMENT_SEC)
        if len(signal) < target_len:
            window = np.pad(signal, (0, target_len - len(signal)))
        else:
            window = signal[:target_len]

    feats = extract_features(window, sr=sr, clip_id="upload")
    vec = np.array([feats.get(col, 0.0) for col in FEATURES], dtype=float)
    vec = np.nan_to_num(vec, nan=0.0)
    return vec, duration_s, sr


# ── Label metadata ────────────────────────────────────────────────────────────
LABEL_INFO = {
    "ambient":     {"icon": "🌊", "color": "#3b82f6", "desc": "Natural background ocean noise – no vessel detected."},
    "unknown":     {"icon": "❓", "color": "#f59e0b", "desc": "Unrecognised acoustic signature – manual review recommended."},
    "vessel_like": {"icon": "🚢", "color": "#ef4444", "desc": "Vessel-like machinery tones detected – possible surface contact."},
    "anomaly":     {"icon": "⚠️", "color": "#a855f7", "desc": "Sound does not match any known acoustic profile. Flagged for operator review."},
}


# ── Branch B: GMM prediction helper ──────────────────────────────────────────
def predict_gmm(vec_scaled: np.ndarray):
    """
    Computes log-likelihoods and converts to calibrated posterior probabilities P(Class | Sound)
    using Bayes rule with class priors.
    """
    raw_scores = {
        label: float(gmm_models[label].score_samples(vec_scaled)[0])
        for label in GMM_CLASSES
    }
    
    best_label = max(raw_scores, key=raw_scores.get)
    best_raw_score = raw_scores[best_label]
    
    # Anomaly check: log-likelihood is below anomaly threshold
    is_anomaly = bool(best_raw_score < ANOMALY_THRESHOLD)

    # Softmax on log-likelihoods to get relative confidence %
    max_log = max(raw_scores.values())
    exp_scores = {k: np.exp(v - max_log) for k, v in raw_scores.items()}
    total_exp = sum(exp_scores.values())
    norm_scores = {
        k: round(float((v / total_exp) * 100), 1)
        for k, v in exp_scores.items()
    }

    return best_label, is_anomaly, best_raw_score, raw_scores, norm_scores


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


@app.route("/ui")
@app.route("/")
def serve_ui():
    return send_from_directory(Path(__file__).parent, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        vec, duration_s, sr = process_and_extract_features(tmp_path)
    except Exception as e:
        return jsonify({"error": f"Could not process audio: {e}"}), 422
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # ── Branch A: Random Forest ───────────────────────────────────
    vec_rf = rf_scaler.transform(vec.reshape(1, -1))
    rf_probs  = rf_model.predict_proba(vec_rf)[0]
    rf_pred_i = int(np.argmax(rf_probs))
    rf_label  = rf_le.inverse_transform([rf_pred_i])[0]
    rf_conf   = float(rf_probs[rf_pred_i])
    rf_all_probs = {
        rf_le.inverse_transform([i])[0]: round(float(p) * 100, 2)
        for i, p in enumerate(rf_probs)
    }
    rf_info = LABEL_INFO.get(rf_label, LABEL_INFO["unknown"])

    # ── Branch B: GMM ─────────────────────────────────────────────
    vec_gmm = gmm_scaler.transform(vec.reshape(1, -1))
    gmm_label, is_anomaly, best_score, raw_scores, norm_scores = predict_gmm(vec_gmm)
    display_label = "anomaly" if is_anomaly else gmm_label
    gmm_info = LABEL_INFO.get(display_label, LABEL_INFO["unknown"])

    # Agreement flag
    models_agree = (rf_label == gmm_label) and not is_anomaly

    return jsonify({
        # ── Branch A result ──────────────────────────────────────
        "branch_a": {
            "model"       : "Random Forest",
            "accuracy"    : "99.37%",
            "label"       : rf_label,
            "confidence"  : round(rf_conf * 100, 2),
            "icon"        : rf_info["icon"],
            "color"       : rf_info["color"],
            "description" : rf_info["desc"],
            "all_probs"   : rf_all_probs,
        },
        # ── Branch B result ──────────────────────────────────────
        "branch_b": {
            "model"           : "Gaussian Mixture Model",
            "paradigm"        : "Generative / Probabilistic",
            "label"           : gmm_label,
            "display_label"   : display_label,
            "is_anomaly"      : is_anomaly,
            "best_score"      : round(best_score, 4),
            "anomaly_threshold": round(ANOMALY_THRESHOLD, 4),
            "icon"            : gmm_info["icon"],
            "color"           : gmm_info["color"],
            "description"     : gmm_info["desc"],
            "norm_scores"     : norm_scores,
        },
        # ── Agreement ────────────────────────────────────────────
        "models_agree"  : models_agree,
        "duration_s"    : duration_s,
        "sample_rate"   : sr,
    })


@app.route("/health")
def health():
    return jsonify({
        "status"   : "ok",
        "branch_a" : {"model": "RandomForest", "accuracy": "99.37%"},
        "branch_b" : {"model": "GMM", "classes": GMM_CLASSES, "anomaly_threshold": ANOMALY_THRESHOLD},
    })


if __name__ == "__main__":
    import webbrowser, threading
    threading.Timer(1.5, lambda: webbrowser.open("http://localhost:5000/ui")).start()
    app.run(host="0.0.0.0", port=5000, debug=False)
