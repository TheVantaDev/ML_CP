"""
train.py
Train all 5 classical ML classifiers and compare them.

Models:
  1. Logistic Regression  (baseline)
  2. K-Nearest Neighbours (baseline)
  3. SVM                  (main comparison)
  4. Random Forest        (main candidate)
  5. XGBoost              (main candidate)

Workflow:
  - Load feature matrix (features.parquet)
  - Load selected features from selected_features.json
  - Scale features
  - Train each model with cross-validation
  - Hyper-parameter search for RF and XGBoost
  - Evaluate on held-out test set
  - Save best model via model registry

Usage:
  python -m src.models.train
  python -m src.models.train --model rf   # train only Random Forest
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    FEATURES_PARQ, FEATURES_CSV, SEL_FEATURES, RANDOM_SEED, CV_FOLDS
)
from src.models.evaluate import evaluate_model
from src.models.registry import save_model


NON_FEATURE_COLS = {
    "clip_id", "source_clip_id", "label", "split",
    "start_sec", "end_sec", "is_poor_quality",
}


# ─── Data loading ───────────────────────────────────────────────────────────────

def load_data():
    path = FEATURES_PARQ if FEATURES_PARQ.exists() else FEATURES_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found at {path}.\n"
            "Run: python -m src.features.build_dataset"
        )
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)

    # Load selected features
    if SEL_FEATURES.exists():
        with open(SEL_FEATURES) as f:
            sel = json.load(f)
        feature_cols = sel["selected_features"]
        # Filter to only those present in the dataframe
        feature_cols = [c for c in feature_cols if c in df.columns]
        print(f"Using {len(feature_cols)} selected features.")
    else:
        feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
        print(f"No selection file found. Using all {len(feature_cols)} features.")

    # Split
    train_df = df[df["split"] == "train"]
    val_df   = df[df["split"] == "val"]
    test_df  = df[df["split"] == "test"]

    def _prepare(subset):
        X = subset[feature_cols].copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.median(), inplace=True)
        return X.values, subset["label"].values

    X_train, y_train_raw = _prepare(train_df)
    X_val,   y_val_raw   = _prepare(val_df)
    X_test,  y_test_raw  = _prepare(test_df)

    le = LabelEncoder()
    le.fit(y_train_raw)
    y_train = le.transform(y_train_raw)
    y_val   = le.transform(y_val_raw)
    y_test  = le.transform(y_test_raw)

    print(f"\nDataset sizes:")
    print(f"  Train: {len(X_train)} samples | {np.bincount(y_train)}")
    print(f"  Val:   {len(X_val)} samples   | {np.bincount(y_val)}")
    print(f"  Test:  {len(X_test)} samples  | {np.bincount(y_test)}")
    print(f"  Classes: {le.classes_}")

    return X_train, X_val, X_test, y_train, y_val, y_test, le, feature_cols


# ─── Model definitions ──────────────────────────────────────────────────────────

def get_models(class_weights=None):
    """Return dict of {name: (model, needs_scaling)}."""
    n_classes = len(class_weights) if class_weights else 3
    return {
        "logistic_regression": (
            LogisticRegression(
                max_iter=2000, C=1.0,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
            True,   # needs scaling
        ),
        "knn": (
            KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1),
            True,
        ),
        "svm": (
            SVC(
                kernel="rbf", C=10, gamma="scale",
                probability=True, class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
            True,
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=300, max_depth=None,
                min_samples_leaf=2, class_weight="balanced",
                random_state=RANDOM_SEED, n_jobs=-1,
            ),
            False,  # tree-based: no scaling needed
        ),
        "xgboost": (
            XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="mlogloss",
                random_state=RANDOM_SEED, n_jobs=-1,
            ),
            False,
        ),
    }


# ─── Hyper-parameter search ─────────────────────────────────────────────────────

RF_PARAM_DIST = {
    "n_estimators":     [100, 200, 300, 500],
    "max_depth":        [None, 10, 20, 30],
    "min_samples_leaf": [1, 2, 4],
    "max_features":     ["sqrt", "log2", 0.5],
}

XGB_PARAM_DIST = {
    "n_estimators":    [100, 200, 300],
    "max_depth":       [4, 6, 8],
    "learning_rate":   [0.01, 0.05, 0.1],
    "subsample":       [0.7, 0.8, 0.9],
    "colsample_bytree":[0.7, 0.8, 1.0],
    "reg_alpha":       [0, 0.1, 1.0],
}


def tune_model(model, param_dist, X_train, y_train, n_iter=20):
    """RandomizedSearchCV with StratifiedKFold."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    search = RandomizedSearchCV(
        model, param_dist,
        n_iter=n_iter, scoring="f1_macro",
        cv=cv, random_state=RANDOM_SEED, n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"  Best params: {search.best_params_}")
    print(f"  CV macro-F1: {search.best_score_:.4f}")
    return search.best_estimator_


# ─── Main training ──────────────────────────────────────────────────────────────

def train(model_filter: str = "all"):
    X_train, X_val, X_test, y_train, y_val, y_test, le, feature_cols = load_data()

    # Scale for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    models_to_train = get_models()
    if model_filter != "all":
        models_to_train = {k: v for k, v in models_to_train.items() if k == model_filter}

    results = {}
    best_f1   = -1.0
    best_name = None
    best_clf  = None

    for name, (clf, needs_scale) in models_to_train.items():
        print(f"\n{'='*60}")
        print(f"  Training: {name.upper()}")
        print(f"{'='*60}")

        X_tr = X_train_scaled if needs_scale else X_train
        X_vl = X_val_scaled   if needs_scale else X_val
        X_te = X_test_scaled  if needs_scale else X_test

        # Hyper-parameter tuning for RF and XGBoost
        if name == "random_forest":
            rf_base = RandomForestClassifier(class_weight="balanced",
                                              random_state=RANDOM_SEED, n_jobs=-1)
            clf = tune_model(rf_base, RF_PARAM_DIST, X_tr, y_train)
        elif name == "xgboost":
            xgb_base = XGBClassifier(eval_metric="mlogloss",
                                  random_state=RANDOM_SEED, n_jobs=-1)
            clf = tune_model(xgb_base, XGB_PARAM_DIST, X_tr, y_train)
        else:
            clf.fit(X_tr, y_train)

        metrics = evaluate_model(
            clf, X_te, y_te=y_test,
            X_val=X_vl, y_val=y_val,
            class_names=le.classes_.tolist(),
            model_name=name,
        )
        results[name] = metrics

        f1 = metrics["test"]["macro_f1"]
        print(f"  Test macro-F1: {f1:.4f}  |  Accuracy: {metrics['test']['accuracy']:.4f}")

        if f1 > best_f1:
            best_f1   = f1
            best_name = name
            best_clf  = clf

        # Save every model
        save_model(
            model=clf,
            scaler=scaler if needs_scale else None,
            feature_names=feature_cols,
            label_encoder=le,
            model_name=name,
            metrics=metrics,
        )

    # ── Summary table ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    rows = []
    for name, m in results.items():
        rows.append({
            "model":      name,
            "val_f1":     round(m["val"]["macro_f1"], 4),
            "test_f1":    round(m["test"]["macro_f1"], 4),
            "test_acc":   round(m["test"]["accuracy"], 4),
        })
    summary = pd.DataFrame(rows).sort_values("test_f1", ascending=False)
    print(summary.to_string(index=False))

    print(f"\nBest model: {best_name}  (test macro-F1 = {best_f1:.4f})")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classical ML models")
    parser.add_argument(
        "--model",
        choices=["all", "logistic_regression", "knn", "svm", "random_forest", "xgboost"],
        default="all",
    )
    args = parser.parse_args()
    train(model_filter=args.model)
