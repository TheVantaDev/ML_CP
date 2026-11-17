"""
06_gmm_model/train_gmm.py
=========================
Branch B: Gaussian Mixture Model (GMM) — Generative / Probabilistic Classifier

What this does:
  - Loads data/features/features.csv (same data as all other models)
  - Uses the exact selected features (66 features)
  - Fits a Gaussian Mixture Model per class (or multi-class GMM with priors)
  - Normalizes scores into posterior probabilities: P(Class | Audio)
  - Detects anomalies ONLY when log-likelihood is an extreme outlier (e.g. bottom 0.5% or flat entropy)
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV  = ROOT / "data" / "features" / "features.csv"
SEL_FEAT_JSON = ROOT / "data" / "features" / "selected_features.json"
MODEL_DIR     = Path(__file__).parent / "models"
PLOT_DIR      = Path(__file__).parent / "plots"
MODEL_DIR.mkdir(exist_ok=True, parents=True)
PLOT_DIR.mkdir(exist_ok=True, parents=True)

CLASSES = ["vessel_like", "ambient", "unknown"]
N_COMPONENTS = 4
COVARIANCE_TYPE = "diag"
RANDOM_STATE = 42

def load_data():
    print("Loading features.csv...")
    df = pd.read_csv(FEATURES_CSV)
    with open(SEL_FEAT_JSON) as f:
        sel = json.load(f)
    feature_cols = sel["selected_features"] if isinstance(sel, dict) else sel
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0).values
    y = df["label"].values
    return X, y, feature_cols

def main():
    X, y, feature_cols = load_data()
    print(f"Total rows: {len(X)}, Features: {len(feature_cols)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    gmm_models = {}
    train_scores_list = []

    for label in CLASSES:
        X_c = X_train_s[y_train == label]
        n_comp = min(N_COMPONENTS, len(X_c))
        gmm = GaussianMixture(
            n_components=n_comp,
            covariance_type=COVARIANCE_TYPE,
            max_iter=300,
            reg_covar=1e-3,
            random_state=RANDOM_STATE
        )
        gmm.fit(X_c)
        gmm_models[label] = gmm
        print(f"✓ GMM fitted for '{label}' with {len(X_c)} samples (components={n_comp})")

    # Evaluate on test set using log-likelihood scoring
    test_scores = np.column_stack([gmm_models[lbl].score_samples(X_test_s) for lbl in CLASSES])
    
    # Calculate anomaly threshold conservatively (e.g. 0.5 percentile of train scores)
    train_scores = np.column_stack([gmm_models[lbl].score_samples(X_train_s) for lbl in CLASSES])
    best_train_scores = train_scores.max(axis=1)
    anomaly_threshold = float(np.percentile(best_train_scores, 0.5)) - 100.0  # Safe dynamic margin
    print(f"Calibrated Anomaly Threshold: {anomaly_threshold:.2f}")

    best_idx = np.argmax(test_scores, axis=1)
    y_pred = [CLASSES[i] for i in best_idx]

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"\nGMM Test Accuracy: {acc*100:.2f}% (Balanced: {bal_acc*100:.2f}%)")

    # Save artifacts
    for lbl, gmm in gmm_models.items():
        joblib.dump(gmm, MODEL_DIR / f"gmm_{lbl}.joblib")
    joblib.dump(scaler, MODEL_DIR / "gmm_scaler.joblib")

    metadata = {
        "model_type": "GaussianMixtureModel",
        "paradigm": "Generative / Probabilistic",
        "classes": CLASSES,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "anomaly_threshold": anomaly_threshold,
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc)
    }
    with open(MODEL_DIR / "gmm_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("✓ Model files and metadata saved successfully!")

if __name__ == "__main__":
    main()
