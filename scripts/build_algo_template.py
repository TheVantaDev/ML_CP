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

# Template builder for algorithm notebooks
def build_algo_notebook(algo_name, folder_name, theory_md, model_init_code, tuning_code, interp_code):
    cells = [
        md(f"""# {algo_name} — Acoustic Event Classification
### Sonobuoy Sound Detective: Offline Underwater Acoustic Event Detection
**Assigned Algorithm Folder:** `{folder_name}/`

---

## 1. Algorithm Overview & Theoretical Foundation
{theory_md}

This self-contained notebook performs:
1. Loading the shared acoustic feature matrix (`../data/features/features.parquet`).
2. Stratified Train / Validation / Test splitting (80% / 10% / 10%).
3. Feature standardization with `StandardScaler`.
4. Hyperparameter tuning & 5-Fold Stratified Cross-Validation.
5. Held-out Test Set evaluation (Accuracy, Macro Precision, Macro Recall, Macro F1, Multi-Class ROC-AUC).
6. Generating and saving publication-quality **Confusion Matrix**, **ROC Curves**, and **Interpretability Plots** into `plots/`.
7. Exporting the final trained model, scaler, and evaluation metrics into `models/` for the final comparison showdown!
"""),
        code(f"""import os
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

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)

Path("plots").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
print("Environment configured for {algo_name}.")
"""),
        md("""## 2. Load Shared Acoustic Features and Selected Descriptors
All models share the identical benchmark dataset created in `00_data_prep_and_features`.
"""),
        code("""# Locate feature table
parq_path = Path("../data/features/features.parquet")
csv_path  = Path("../data/features/features.csv")
sel_path  = Path("../data/features/selected_features.json")

if parq_path.exists():
    df = pd.read_parquet(parq_path)
elif csv_path.exists():
    df = pd.read_csv(csv_path)
else:
    raise FileNotFoundError("Feature matrix not found! Run notebook 00 first.")

if sel_path.exists():
    with open(sel_path) as f:
        data = json.load(f)
        feature_cols = data["selected_features"] if isinstance(data, dict) and "selected_features" in data else data
    print(f"Loaded {len(feature_cols)} selected features from selected_features.json")
else:
    NON_FEATURE_COLS = {"clip_id", "source_clip_id", "label", "split", "start_sec", "end_sec", "is_poor_quality"}
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    print(f"Using all {len(feature_cols)} numeric features.")

print(f"Total samples: {len(df)}")
print(f"Target class distribution:\\n{df['label'].value_counts()}")
"""),
        md("""## 3. Stratified Train / Validation / Test Splitting & Feature Scaling"""),
        code("""le = LabelEncoder()
df['target'] = le.fit_transform(df['label'])
classes = list(le.classes_)
num_classes = len(classes)
print(f"Classes: {classes}")

if "split" in df.columns and set(df["split"].unique()) >= {"train", "test"}:
    train_mask = df["split"] == "train"
    val_mask   = df["split"] == "val"
    test_mask  = df["split"] == "test"
    if val_mask.sum() == 0:
        val_mask = train_mask.sample(frac=0.1, random_state=RANDOM_SEED)
        train_mask = train_mask & (~val_mask)
else:
    train_idx, test_idx = train_test_split(df.index, test_size=0.15, stratify=df['target'], random_state=RANDOM_SEED)
    train_idx, val_idx  = train_test_split(train_idx, test_size=0.12, stratify=df.loc[train_idx, 'target'], random_state=RANDOM_SEED)
    train_mask = df.index.isin(train_idx)
    val_mask   = df.index.isin(val_idx)
    test_mask  = df.index.isin(test_idx)

X_train_raw = np.nan_to_num(df.loc[train_mask, feature_cols].values, nan=0.0)
y_train     = df.loc[train_mask, "target"].values

X_val_raw   = np.nan_to_num(df.loc[val_mask, feature_cols].values, nan=0.0)
y_val       = df.loc[val_mask, "target"].values

X_test_raw  = np.nan_to_num(df.loc[test_mask, feature_cols].values, nan=0.0)
y_test      = df.loc[test_mask, "target"].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val   = scaler.transform(X_val_raw)
X_test  = scaler.transform(X_test_raw)

print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")
"""),
        md(f"## 4. Hyperparameter Tuning & Cross-Validation for {algo_name}"),
        code(tuning_code),
        md(f"## 5. Train Optimal {algo_name} & Evaluate on Held-Out Test Set"),
        code(model_init_code),
        md("## 6. Confusion Matrix & Multi-Class ROC Curves"),
        code("""# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred, labels=range(num_classes))
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_norm = np.nan_to_num(cm_norm, nan=0.0)

sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.title(f"Confusion Matrix (Normalized) — {best_name}", fontsize=11, fontweight='bold')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot Multi-Class ROC Curves
plt.figure(figsize=(7, 5))
for i, cls in enumerate(classes):
    y_bin = (y_test == i).astype(int)
    fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
    roc_auc_cls = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC = {roc_auc_cls:.3f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curves (One-vs-Rest) — {best_name}", fontsize=11, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/roc_curves.png", dpi=300, bbox_inches="tight")
plt.show()
"""),
        md("## 7. Feature Interpretability & Learned Representations"),
        code(interp_code),
        md("## 8. Export Model Artifacts & Metrics for Final Comparison"),
        code(f"""# Export model and scaler
joblib.dump(best_model, "models/model.joblib")
joblib.dump(scaler,     "models/scaler.joblib")
joblib.dump(le,         "models/label_encoder.joblib")

metrics = {{
    "algorithm": "{algo_name}",
    "test_accuracy": float(acc),
    "macro_precision": float(prec),
    "macro_recall": float(rec),
    "macro_f1": float(f1),
    "roc_auc_macro": float(auc_val) if auc_val is not None else 0.0,
    "latency_ms": float(latency_ms),
    "classes": classes,
    "feature_count": len(feature_cols)
}}

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved model artifacts to models/model.joblib")
print("Saved metrics to models/metrics.json:")
print(json.dumps(metrics, indent=2))
""")
    ]
    return make_nb(cells)

print("Updated template builder.")
