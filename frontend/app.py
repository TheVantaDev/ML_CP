"""
Flask backend for Sonobuoy Sound Detective
Loads the best model (Random Forest, 99.37% accuracy) and
classifies uploaded audio as: ambient / unknown / vessel_like
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

MODEL_DIR = ROOT / "03_person3_decision_tree_and_random_forest" / "models"
MODEL_PATH  = MODEL_DIR / "model_random_forest.joblib"
SCALER_PATH = MODEL_DIR / "scaler_random_forest.joblib"
LE_PATH     = MODEL_DIR / "label_encoder_random_forest.joblib"
SEL_FEAT    = ROOT / "data" / "features" / "selected_features.json"

# ── Load model once at startup ─────────────────────────────────────────────────
print("Loading Random Forest model …", flush=True)
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le     = joblib.load(LE_PATH)
with open(SEL_FEAT) as f:
    raw = json.load(f)
    FEATURES = raw["selected_features"] if isinstance(raw, dict) else raw
print(f"Model ready! Classes: {list(le.classes_)}, Features: {len(FEATURES)}", flush=True)

# ── Feature extraction (mirrors src/features/extractor.py logic) ────────────
from src.features.extractor import extract_features

SAMPLE_RATE = 22050
SEGMENT_SEC = 4.0

def extract_features_from_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """Extract features from a single audio array and return feature vector."""
    # Resample if needed
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    # Mono
    if y.ndim > 1:
        y = librosa.to_mono(y.T)

    # Pad/trim to segment length
    target_len = int(SAMPLE_RATE * SEGMENT_SEC)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # Extract via project extractor
    feats = extract_features(y, sr=sr, clip_id="upload")
    vec = np.array([feats.get(col, 0.0) for col in FEATURES], dtype=float)
    vec = np.nan_to_num(vec, nan=0.0)
    return vec

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

LABEL_INFO = {
    "ambient":     {"icon": "🌊", "color": "#3b82f6", "desc": "Natural background ocean noise – no vessel detected."},
    "unknown":     {"icon": "❓", "color": "#f59e0b", "desc": "Unrecognised acoustic signature – manual review recommended."},
    "vessel_like": {"icon": "🚢", "color": "#ef4444", "desc": "Vessel-like machinery tones detected – possible surface contact."},
}

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

    # Save to temp file
    suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        y, sr = librosa.load(tmp_path, sr=None, mono=True)
    except Exception as e:
        return jsonify({"error": f"Could not read audio: {e}"}), 422
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    vec = extract_features_from_audio(y, sr)
    vec_scaled = scaler.transform(vec.reshape(1, -1))
    probs  = model.predict_proba(vec_scaled)[0]
    pred_i = int(np.argmax(probs))
    label  = le.inverse_transform([pred_i])[0]
    conf   = float(probs[pred_i])

    all_probs = {le.inverse_transform([i])[0]: round(float(p) * 100, 2)
                 for i, p in enumerate(probs)}

    info = LABEL_INFO.get(label, {"icon": "❓", "color": "#6b7280", "desc": ""})
    return jsonify({
        "label":       label,
        "confidence":  round(conf * 100, 2),
        "icon":        info["icon"],
        "color":       info["color"],
        "description": info["desc"],
        "all_probs":   all_probs,
        "duration_s":  round(len(y) / sr, 2),
        "sample_rate": sr,
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "RandomForest", "accuracy": "99.37%"})

if __name__ == "__main__":
    import webbrowser, threading
    threading.Timer(1.5, lambda: webbrowser.open("http://localhost:5000/ui")).start()
    app.run(host="0.0.0.0", port=5000, debug=False)
