"""
registry.py
Save and load versioned ML model artifacts.

Saves:
  models/<model_name>_v<version>/
    model.joblib
    scaler.joblib          (if used)
    label_encoder.joblib
    selected_features.json
    model_metadata.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import MODELS_DIR, SEL_FEATURES


def _next_version(model_dir: Path, model_name: str) -> int:
    """Find the next version number for a model."""
    existing = sorted(model_dir.glob(f"{model_name}_v*/"))
    if not existing:
        return 1
    last = existing[-1].name  # e.g. "random_forest_v3"
    try:
        return int(last.split("_v")[-1]) + 1
    except ValueError:
        return 1


def save_model(
    model,
    scaler,
    feature_names:  list,
    label_encoder,
    model_name:     str,
    metrics:        dict,
    model_dir:      Path = MODELS_DIR,
) -> Path:
    """
    Save a trained model and all associated artifacts.
    Returns the path to the model directory.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    version  = _next_version(model_dir, model_name)
    save_dir = model_dir / f"{model_name}_v{version}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Model
    joblib.dump(model, save_dir / "model.joblib")

    # Scaler (optional)
    if scaler is not None:
        joblib.dump(scaler, save_dir / "scaler.joblib")

    # Label encoder
    joblib.dump(label_encoder, save_dir / "label_encoder.joblib")

    # Feature list
    feat_info = {"selected_features": feature_names, "n_features": len(feature_names)}
    with open(save_dir / "selected_features.json", "w") as f:
        json.dump(feat_info, f, indent=2)

    # Metadata
    meta = {
        "model_name":   model_name,
        "version":      version,
        "timestamp":    datetime.now().isoformat(),
        "metrics":      metrics,
        "n_features":   len(feature_names),
        "classes":      label_encoder.classes_.tolist(),
        "model_params": str(model.get_params()) if hasattr(model, "get_params") else {},
    }
    with open(save_dir / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Model saved -> {save_dir}")
    return save_dir


def load_model(model_dir: Path):
    """
    Load a saved model bundle from a version directory.
    Returns (model, scaler_or_None, label_encoder, feature_names).
    """
    model        = joblib.load(model_dir / "model.joblib")
    label_encoder= joblib.load(model_dir / "label_encoder.joblib")
    scaler_path  = model_dir / "scaler.joblib"
    scaler       = joblib.load(scaler_path) if scaler_path.exists() else None

    with open(model_dir / "selected_features.json") as f:
        feat_info = json.load(f)
    feature_names = feat_info["selected_features"]

    return model, scaler, label_encoder, feature_names


def get_latest_model(model_name: str, model_dir: Path = MODELS_DIR):
    """Return path to the latest version of a model."""
    versions = sorted(model_dir.glob(f"{model_name}_v*/"))
    if not versions:
        raise FileNotFoundError(f"No saved models found for '{model_name}' in {model_dir}")
    return versions[-1]
