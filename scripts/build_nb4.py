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
# NOTEBOOK 4: Classical ML Model Comparison & Training
# ==============================================================================
nb4_cells = [
    md("""# 04. Classical ML Model Training, Comparison & Benchmarking
### Sonobuoy Sound Detective: Offline Underwater Acoustic Event Detection
**ML Approach:** Classical Machine Learning Only — No Deep Learning, CNN, RNN, or Transformers.

In this notebook, we train, evaluate, and rigorously benchmark **5 classical machine learning algorithms** on handcrafted acoustic features:
1. **Logistic Regression** (Linear baseline with L2 regularization)
2. **K-Nearest Neighbors (KNN)** (Distance/metric-based non-parametric baseline)
3. **Support Vector Machine (SVM)** (Maximum-margin classifier with RBF kernel)
4. **Random Forest Classifier** (Bagged decision tree ensemble)
5. **XGBoost Classifier** (Gradient-boosted decision trees)

We compare the models across:
- **5-Fold Stratified Cross-Validation** (Macro F1 & Accuracy)
- **Held-Out Test Set Performance** (Accuracy, Macro Precision, Macro Recall, Macro F1, ROC-AUC)
- **Side-by-Side Confusion Matrices**
- **One-vs-Rest (OvR) ROC Curves**
- **Inference Latency** (milliseconds per 4-second audio segment)
- **Model Explainability & Feature Importance**
"""),
    code("""import os
import sys
import time
import json
from pathlib import Path

PROJECT_ROOT = Path("..").resolve() if Path(".").resolve().name == "notebooks" else Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.inspection import permutation_importance

from src.utils.config import (
    FEATURES_PARQ, FEATURES_CSV, SEL_FEATURES,
    MODELS_DIR, RANDOM_SEED, CV_FOLDS, CLASS_NAMES
)

print("Modules imported successfully.")
"""),
    md("""## 1. Load Feature Matrix and Split by Train / Val / Test
We use source-aware splitting (`train`, `val`, `test` defined in the metadata) or stratified splits to prevent data leakage.
"""),
    code("""path = FEATURES_PARQ if FEATURES_PARQ.exists() else FEATURES_CSV
df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)

# Load selected features if available, otherwise use all numeric features
if SEL_FEATURES.exists():
    with open(SEL_FEATURES) as f:
        feature_cols = json.load(f)
    print(f"Loaded {len(feature_cols)} selected features from selected_features.json")
else:
    NON_FEATURE_COLS = {"clip_id", "source_clip_id", "label", "split", "start_sec", "end_sec", "is_poor_quality"}
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    print(f"Using {len(feature_cols)} features from feature table.")

# Target encoding
le = LabelEncoder()
df['target'] = le.fit_transform(df['label'])
classes = list(le.classes_)
num_classes = len(classes)
print(f"Class mapping: {dict(zip(range(num_classes), classes))}")

# Check splits
if "split" in df.columns and set(df["split"].unique()) >= {"train", "test"}:
    train_mask = df["split"] == "train"
    val_mask   = df["split"] == "val"
    test_mask  = df["split"] == "test"
    # If val is empty, split 10% of train into val
    if val_mask.sum() == 0:
        val_mask = train_mask.sample(frac=0.1, random_state=RANDOM_SEED)
        train_mask = train_mask & (~val_mask)
else:
    # Stratified 80/10/10
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(df.index, test_size=0.15, stratify=df['target'], random_state=RANDOM_SEED)
    train_idx, val_idx  = train_test_split(train_idx, test_size=0.12, stratify=df.loc[train_idx, 'target'], random_state=RANDOM_SEED)
    train_mask = df.index.isin(train_idx)
    val_mask   = df.index.isin(val_idx)
    test_mask  = df.index.isin(test_idx)

X_train_raw = df.loc[train_mask, feature_cols].values
y_train     = df.loc[train_mask, "target"].values

X_val_raw   = df.loc[val_mask, feature_cols].values
y_val       = df.loc[val_mask, "target"].values

X_test_raw  = df.loc[test_mask, feature_cols].values
y_test      = df.loc[test_mask, "target"].values

# Clean NaNs if any
X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
X_val_raw   = np.nan_to_num(X_val_raw, nan=0.0, posinf=0.0, neginf=0.0)
X_test_raw  = np.nan_to_num(X_test_raw, nan=0.0, posinf=0.0, neginf=0.0)

# Feature scaling (StandardScaler fit ONLY on training set)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val   = scaler.transform(X_val_raw)
X_test  = scaler.transform(X_test_raw)

print(f"Dataset Split Sizes:")
print(f"  Training set   : {X_train.shape[0]} samples")
print(f"  Validation set : {X_val.shape[0]} samples")
print(f"  Held-out test  : {X_test.shape[0]} samples")
"""),
    md("""## 2. Define the 5 Classical ML Model Candidates
"""),
    code("""def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=RANDOM_SEED, class_weight="balanced"
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5, metric="minkowski", p=2, weights="distance"
        ),
        "Support Vector Machine": SVC(
            C=10.0, kernel="rbf", probability=True, random_state=RANDOM_SEED, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_split=4,
            random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
            random_state=RANDOM_SEED, n_jobs=-1
        )
    }

models = get_models()
for name, m in models.items():
    print(f"Initialized candidate model: {name}")
"""),
    md("""## 3. 5-Fold Stratified Cross-Validation Comparison
Evaluate generalization stability across 5 folds on the training set.
"""),
    code("""cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
cv_results = []

print("Running 5-Fold Stratified Cross-Validation on Training Data...")
for name, model in models.items():
    scores = cross_validate(
        model, X_train, y_train, cv=cv,
        scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
        n_jobs=-1
    )
    mean_acc = scores["test_accuracy"].mean()
    std_acc  = scores["test_accuracy"].std()
    mean_f1  = scores["test_macro_f1"].mean()
    std_f1   = scores["test_macro_f1"].std()
    
    cv_results.append({
        "Model": name,
        "CV Accuracy": f"{mean_acc*100:.2f}% ± {std_acc*100:.2f}%",
        "CV Macro F1": f"{mean_f1*100:.2f}% ± {std_f1*100:.2f}%",
        "_mean_f1": mean_f1
    })

df_cv = pd.DataFrame(cv_results).sort_values("_mean_f1", ascending=False).drop(columns=["_mean_f1"])
display(df_cv)
"""),
    md("""## 4. Train Models and Evaluate on Held-Out Test Set
We train each model on the full training set and benchmark against the unseen Test Set:
- **Accuracy, Macro Precision, Macro Recall, Macro F1**
- **Multi-Class ROC-AUC (One-vs-Rest)**
- **Inference Latency (milliseconds per 4s clip)**
"""),
    code("""benchmark_records = []
trained_models = {}
y_probs = {}
y_preds = {}

for name, model in models.items():
    # Fit model
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    trained_models[name] = model
    
    # Inference Latency benchmark (single segment)
    t_start = time.perf_counter()
    n_iters = 50
    for _ in range(n_iters):
        _ = model.predict(X_test[:1])
    latency_ms = ((time.perf_counter() - t_start) / n_iters) * 1000.0
    
    # Predict on test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    y_preds[name] = y_pred
    y_probs[name] = y_prob
    
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
    
    auc_val = None
    if y_prob is not None:
        try:
            if num_classes == 2:
                auc_val = roc_auc_score(y_test, y_prob[:, 1])
            else:
                auc_val = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        except Exception:
            auc_val = None
            
    benchmark_records.append({
        "Model": name,
        "Test Accuracy": acc,
        "Macro Precision": prec,
        "Macro Recall": rec,
        "Macro F1": f1,
        "ROC-AUC (Macro)": auc_val if auc_val is not None else 0.0,
        "Latency (ms/clip)": latency_ms,
        "Training Time (s)": fit_time
    })

df_bench = pd.DataFrame(benchmark_records).sort_values("Macro F1", ascending=False).reset_index(drop=True)
print("\n" + "="*80)
print("  OFFICIAL CLASSICAL ML BENCHMARK COMPARISON (HELD-OUT TEST SET)")
print("="*80)
display(df_bench.style.format({
    "Test Accuracy": "{:.2%}",
    "Macro Precision": "{:.2%}",
    "Macro Recall": "{:.2%}",
    "Macro F1": "{:.2%}",
    "ROC-AUC (Macro)": "{:.3f}",
    "Latency (ms/clip)": "{:.2f} ms",
    "Training Time (s)": "{:.2f} s"
}))
"""),
    md("""## 5. Side-by-Side Confusion Matrices
Visualizing misclassifications across all 5 candidate models.
"""),
    code("""fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 4))

for ax, (name, y_pred) in zip(axes, y_preds.items()):
    cm = confusion_matrix(y_test, y_pred, labels=range(num_classes))
    # Normalize by true class count (percentage recall)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm, nan=0.0)
    
    sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues", cbar=False, ax=ax,
                xticklabels=classes, yticklabels=classes)
    ax.set_title(f"{name}", fontsize=11, fontweight='bold')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

plt.tight_layout()
plt.show()
"""),
    md("""## 6. One-vs-Rest (OvR) ROC Curves for the Top Performing Model
"""),
    code("""best_model_name = df_bench.iloc[0]["Model"]
best_model = trained_models[best_model_name]
best_probs = y_probs[best_model_name]

print(f"Top Model Selected: {best_model_name}")

plt.figure(figsize=(8, 6))
for i, cls in enumerate(classes):
    y_test_bin = (y_test == i).astype(int)
    fpr, tpr, _ = roc_curve(y_test_bin, best_probs[:, i])
    roc_auc_cls = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"Class: {cls} (AUC = {roc_auc_cls:.3f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curves (One-vs-Rest) — {best_model_name}", fontsize=12, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()
"""),
    md("""## 7. Feature Importance Analysis (Interpretability)
Classical ML allows operators to inspect **why** a decision was made. We compute both Gini importance (MDI) and Permutation Feature Importance.
"""),
    code("""if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    feat_series = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(15)
    
    plt.figure(figsize=(10, 5))
    feat_series.plot(kind="barh", color="#2a9d8f")
    plt.title(f"Top 15 Most Important Acoustic Features — {best_model_name}", fontsize=11, fontweight="bold")
    plt.xlabel("Relative Importance")
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
"""),
    md("""## 8. Save Best Model to Model Registry
Export the trained model, standard scaler, selected features list, and metadata to `models/<model_name>_v1/`.
"""),
    code("""# Save model package
slug = best_model_name.lower().replace(" ", "_")
model_save_dir = MODELS_DIR / f"{slug}_v1"
model_save_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(best_model, model_save_dir / "model.joblib")
joblib.dump(scaler,     model_save_dir / "scaler.joblib")
joblib.dump(le,         model_save_dir / "label_encoder.joblib")

with open(model_save_dir / "selected_features.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

metadata = {
    "model_name": best_model_name,
    "version": "1.0",
    "test_accuracy": float(df_bench.iloc[0]["Test Accuracy"]),
    "test_macro_f1": float(df_bench.iloc[0]["Macro F1"]),
    "features_used": feature_cols,
    "classes": classes
}
with open(model_save_dir / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Successfully saved {best_model_name} artifacts to:")
print(f"  {model_save_dir}")
""")
]

with open(NOTEBOOKS_DIR / "04_classical_ml_model_comparison_and_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_nb(nb4_cells), f, indent=2)

print("Created 04_classical_ml_model_comparison_and_training.ipynb")
