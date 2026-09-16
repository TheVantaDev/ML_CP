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

def build_single_model_nb(algo_name, file_slug, folder_name, theory_md, tuning_code, interp_code):
    cells = [
        md(f"""# {algo_name} — Acoustic Event Classification
### Sonobuoy Sound Detective: Offline Underwater Acoustic Event Detection
**Workspace Folder:** `{folder_name}/`  
**Algorithm Identifier:** `{file_slug}`

---

## 1. Algorithm Overview & Theoretical Foundation
{theory_md}

This self-contained notebook performs:
1. Loading the shared acoustic feature matrix (`../data/features/features.parquet`).
2. Stratified Train / Validation / Test splitting (80% / 10% / 10%).
3. Feature standardization with `StandardScaler`.
4. Hyperparameter tuning & 5-Fold Stratified Cross-Validation.
5. Held-out Test Set evaluation (Accuracy, Macro Precision, Macro Recall, Macro F1, Multi-Class ROC-AUC).
6. Generating and saving **Confusion Matrix**, **ROC Curves**, and **Interpretability Plots** into `plots/`.
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
ALGO_NAME = "{algo_name}"
FILE_SLUG = "{file_slug}"
print(f"Configured notebook for {{ALGO_NAME}} ({{FILE_SLUG}}).")
"""),
        md("""## 2. Load Shared Acoustic Features and Selected Descriptors"""),
        code("""parq_path = Path("../data/features/features.parquet")
csv_path  = Path("../data/features/features.csv")
sel_path  = Path("../data/features/selected_features.json")

df = pd.read_parquet(parq_path) if parq_path.exists() else pd.read_csv(csv_path)

if sel_path.exists():
    with open(sel_path) as f:
        raw_sel = json.load(f)
        feature_cols = raw_sel["selected_features"] if isinstance(raw_sel, dict) and "selected_features" in raw_sel else raw_sel
    print(f"Loaded {len(feature_cols)} selected features.")
else:
    NON_FEATURE_COLS = {"clip_id", "source_clip_id", "label", "split", "start_sec", "end_sec", "is_poor_quality"}
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    print(f"Using {len(feature_cols)} features.")

print(f"Total samples: {len(df)}")
print(f"Class distribution:\\n{df['label'].value_counts()}")
"""),
        md("""## 3. Stratified Train / Validation / Test Splitting & Feature Scaling"""),
        code("""le = LabelEncoder()
df['target'] = le.fit_transform(df['label'])
classes = list(le.classes_)
num_classes = len(classes)

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

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
"""),
        md(f"## 4. Hyperparameter Tuning & Cross-Validation for {algo_name}"),
        code(tuning_code),
        md(f"## 5. Evaluate Optimal {algo_name} on Held-Out Test Set"),
        code("""t0 = time.time()
best_model.fit(X_train, y_train)
train_time = time.time() - t0

# Latency
t_start = time.perf_counter()
for _ in range(100):
    _ = best_model.predict(X_test[:1])
latency_ms = ((time.perf_counter() - t_start) / 100) * 1000.0

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test) if hasattr(best_model, "predict_proba") else None

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
auc_val = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro") if y_prob is not None else 0.0

print(f"=== {ALGO_NAME} TEST RESULTS ===")
print(f"Accuracy        : {acc:.2%}")
print(f"Macro Precision : {prec:.2%}")
print(f"Macro Recall    : {rec:.2%}")
print(f"Macro F1-Score  : {f1:.2%}")
print(f"Macro ROC-AUC   : {auc_val:.3f}")
print(f"Latency         : {latency_ms:.3f} ms / segment")
"""),
        md("## 6. Confusion Matrix & Multi-Class ROC Curves"),
        code(f"""# Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred, labels=range(num_classes))
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_norm = np.nan_to_num(cm_norm, nan=0.0)

sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.title(f"Confusion Matrix (Normalized) — {{ALGO_NAME}}", fontsize=11, fontweight='bold')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(f"plots/confusion_matrix_{{FILE_SLUG}}.png", dpi=300, bbox_inches="tight")
plt.show()

# ROC Curves
if y_prob is not None:
    plt.figure(figsize=(7, 5))
    for i, cls in enumerate(classes):
        y_bin = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        roc_auc_cls = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{{cls}} (AUC = {{roc_auc_cls:.3f}})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves (One-vs-Rest) — {{ALGO_NAME}}", fontsize=11, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/roc_curves_{{FILE_SLUG}}.png", dpi=300, bbox_inches="tight")
    plt.show()
"""),
        md("## 7. Model Interpretability & Feature Importance"),
        code(interp_code),
        md("## 8. Export Model Artifacts for Final Showdown"),
        code(f"""joblib.dump(best_model, f"models/model_{{FILE_SLUG}}.joblib")
joblib.dump(scaler,     f"models/scaler_{{FILE_SLUG}}.joblib")
joblib.dump(le,         f"models/label_encoder_{{FILE_SLUG}}.joblib")

metrics = {{
    "algorithm": ALGO_NAME,
    "slug": FILE_SLUG,
    "test_accuracy": float(acc),
    "macro_precision": float(prec),
    "macro_recall": float(rec),
    "macro_f1": float(f1),
    "roc_auc_macro": float(auc_val),
    "latency_ms": float(latency_ms),
    "classes": classes,
    "feature_count": len(feature_cols)
}}

with open(f"models/metrics_{{FILE_SLUG}}.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Saved model artifacts for {{ALGO_NAME}} in models/!")
""")
    ]
    return make_nb(cells)

print("Single model builder defined.")
