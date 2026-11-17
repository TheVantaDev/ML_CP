"""
06_gmm_model/train_gmm.py
=========================
Branch B: GMM Generative Classifier

KEY FIX: Trains on features extracted LIVE from wav files using the SAME 
extract_features() pipeline that Flask uses at inference time.
This guarantees training distribution = inference distribution.
"""

import sys
import json
from pathlib import Path

import joblib
import librosa
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features.extractor import extract_features

SEL_FEAT_JSON = ROOT / "data" / "features" / "selected_features.json"
MODEL_DIR     = Path(__file__).parent / "models"
PLOT_DIR      = Path(__file__).parent / "plots"
MODEL_DIR.mkdir(exist_ok=True, parents=True)
PLOT_DIR.mkdir(exist_ok=True, parents=True)

CLASSES = ["vessel_like", "ambient", "unknown"]
SAMPLE_RATE = 22050
SEGMENT_SEC = 4.0
N_COMPONENTS = 4
COVARIANCE_TYPE = "diag"
RANDOM_STATE = 42


def extract_live(wav_path, feature_cols):
    """Extract features from a wav file exactly like Flask does."""
    y, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    target_len = int(SAMPLE_RATE * SEGMENT_SEC)
    if len(y) < SAMPLE_RATE:  # skip very short
        return None
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    feats = extract_features(y, sr=SAMPLE_RATE, clip_id=wav_path.stem)
    vec = np.array([feats.get(c, 0.0) for c in feature_cols], dtype=float)
    vec = np.nan_to_num(vec)
    return vec


def load_data_live(feature_cols):
    """Load features by extracting directly from wav files."""
    processed_dir = ROOT / "data" / "processed"
    rows, labels = [], []

    for label in CLASSES:
        folder = processed_dir / label
        if not folder.exists():
            print(f"  WARNING: {folder} not found, skipping")
            continue
        wavs = sorted(folder.glob("*.wav"))
        ok, skip = 0, 0
        for wav in wavs:
            try:
                vec = extract_live(wav, feature_cols)
                if vec is not None:
                    rows.append(vec)
                    labels.append(label)
                    ok += 1
                else:
                    skip += 1
            except Exception:
                skip += 1
        print(f"  {label:15s}: {ok} loaded, {skip} skipped")

    return np.array(rows), np.array(labels)


def main():
    print("\n" + "=" * 60)
    print("  Branch B: GMM (Live Feature Extraction)")
    print("=" * 60 + "\n")

    # Load feature column names
    with open(SEL_FEAT_JSON) as f:
        sel = json.load(f)
    feature_cols = sel["selected_features"] if isinstance(sel, dict) else sel
    print(f"Features: {len(feature_cols)}\n")

    # Extract features live from wav files
    print("Extracting features from wav files (same pipeline as Flask)...")
    X, y = load_data_live(feature_cols)
    print(f"\nTotal samples: {len(X)}")
    from collections import Counter
    dist = Counter(y)
    print(f"Distribution: {dict(dist)}\n")

    if len(X) == 0:
        print("ERROR: No samples loaded!")
        return

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    # Fit scaler on live-extracted training data
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train one GMM per class
    print("Training GMMs...")
    gmm_models = {}
    for label in CLASSES:
        X_c = X_train_s[y_train == label]
        n_comp = min(N_COMPONENTS, max(1, len(X_c) - 1))
        gmm = GaussianMixture(
            n_components=n_comp,
            covariance_type=COVARIANCE_TYPE,
            max_iter=300,
            reg_covar=1e-3,
            random_state=RANDOM_STATE
        )
        gmm.fit(X_c)
        gmm_models[label] = gmm
        print(f"  {label:15s}: {len(X_c)} samples, {n_comp} components, converged={gmm.converged_}")

    # Evaluate on test set
    print("\nEvaluating...")
    test_scores = np.column_stack([
        gmm_models[lbl].score_samples(X_test_s) for lbl in CLASSES
    ])
    y_pred = [CLASSES[i] for i in np.argmax(test_scores, axis=1)]

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {acc*100:.2f}%  |  Balanced: {bal_acc*100:.2f}%")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Calibrate anomaly threshold
    train_scores = np.column_stack([
        gmm_models[lbl].score_samples(X_train_s) for lbl in CLASSES
    ])
    best_train = train_scores.max(axis=1)
    anomaly_threshold = float(np.percentile(best_train, 1.0))
    print(f"Anomaly threshold (p1): {anomaly_threshold:.2f}\n")

    # Per-class analysis
    print("Per-class analysis on test set:")
    for label in CLASSES:
        mask = y_test == label
        sub_preds = [y_pred[i] for i in range(len(y_pred)) if mask[i]]
        correct = sum(1 for p in sub_preds if p == label)
        total = mask.sum()
        print(f"  {label:15s}: {correct}/{total} correct ({100*correct/total:.1f}%)" if total > 0 else f"  {label}: no samples")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("GMM Confusion Matrix (Live Features)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "gmm_confusion_matrix.png", dpi=120)
    plt.close()
    print(f"\nPlot saved: {PLOT_DIR / 'gmm_confusion_matrix.png'}")

    # Save artifacts
    print("\nSaving models...")
    for lbl, gmm in gmm_models.items():
        joblib.dump(gmm, MODEL_DIR / f"gmm_{lbl}.joblib")
        print(f"  Saved: gmm_{lbl}.joblib")
    joblib.dump(scaler, MODEL_DIR / "gmm_scaler.joblib")
    print("  Saved: gmm_scaler.joblib")

    metadata = {
        "model_type": "GaussianMixtureModel",
        "paradigm": "Generative / Probabilistic",
        "trained_on": "live wav extraction (same pipeline as Flask)",
        "classes": CLASSES,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "n_components": N_COMPONENTS,
        "covariance_type": COVARIANCE_TYPE,
        "anomaly_threshold": anomaly_threshold,
        "metrics": {
            "accuracy": round(float(acc), 6),
            "balanced_accuracy": round(float(bal_acc), 6),
        },
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
    }
    with open(MODEL_DIR / "gmm_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("  Saved: gmm_metadata.json")

    print("\n" + "=" * 60)
    print(f"  ✅ GMM TRAINING COMPLETE (Live Features)")
    print(f"  Accuracy: {acc*100:.2f}%  |  Balanced: {bal_acc*100:.2f}%")
    print(f"  Anomaly threshold: {anomaly_threshold:.2f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
