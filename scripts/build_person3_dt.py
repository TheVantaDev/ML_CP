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
# MASTERPIECE: DECISION TREE (PERSON 3 - MODEL 1)
# ==============================================================================
dt_cells = [
    md("""# Decision Tree Classifier (CART) — Underwater Acoustic Event Classification
### Sonobuoy Sound Detective: Offline Ground-Station Acoustic System
**Assigned Team Member:** Person 3 (Tree & Ensemble Specialist)  
**Algorithm:** Classification and Regression Trees (CART)  
**Folder:** `03_person3_decision_tree_and_random_forest/`

---

## 1. Theoretical Foundation & Mathematical Formulation

The **Decision Tree** is a non-parametric supervised learning algorithm that segments the acoustic feature space $\\mathcal{X} \\subset \\mathbb{R}^D$ into a collection of mutually exclusive, axis-aligned hyper-rectangles:
$$\\mathcal{X} = \\bigcup_{m=1}^M R_m, \\quad R_i \\cap R_j = \\emptyset \\; (i \\neq j)$$

### Splitting Criterion: Gini Impurity
At any internal node $m$ containing subset $S_m$ with class proportions $p_{mk} = \\frac{1}{|S_m|} \\sum_{\\mathbf{x}_i \\in S_m} \\mathbb{I}(y_i = k)$, the **Gini Impurity** measures label heterogeneity:
$$I_G(S_m) = 1 - \\sum_{k=1}^K p_{mk}^2$$

The tree selects the optimal feature $j^*$ and threshold $\\theta^*$ that maximizes the **Information Gain** (reduction in impurity):
$$\\Delta I(S_m, j, \\theta) = I_G(S_m) - \\left( \\frac{|S_L|}{|S_m|} I_G(S_L) + \\frac{|S_R|}{|S_m|} I_G(S_R) \\right)$$
where $S_L = \\{\\mathbf{x} \\in S_m \\mid x_j \\le \\theta\\}$ and $S_R = \\{\\mathbf{x} \\in S_m \\mid x_j > \\theta\\}$.

### Strengths & Weaknesses for Sonobuoy Acoustics:
- **Pros:** White-box transparency; rules can be directly translated into operator decision cards; handles non-linear feature combinations; invariant to monotonic feature scaling.
- **Cons:** High variance; prone to overfitting on high-dimensional acoustic feature vectors without strict depth pruning.
"""),
    code("""import os
import sys
import time
import json
from pathlib import Path

# Configure paths
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.inspection import permutation_importance

# Ensure output directories exist
Path("plots").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(parents=True, exist_ok=True)

# Set global styles
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("Environment loaded successfully for Person 3: Decision Tree Classifier.")
"""),
    md("""## 2. Load Shared Acoustic Features and Selected Descriptors
We load the shared tabular feature dataset extracted from ShipsEar, DeepShip, and Synthetic underwater recordings.
"""),
    code("""parq_path = Path("../data/features/features.parquet")
csv_path  = Path("../data/features/features.csv")
sel_path  = Path("../data/features/selected_features.json")

if parq_path.exists():
    df = pd.read_parquet(parq_path)
    print(f"Loaded features from Parquet: {df.shape}")
elif csv_path.exists():
    df = pd.read_csv(csv_path)
    print(f"Loaded features from CSV: {df.shape}")
else:
    raise FileNotFoundError("Feature dataset not found! Please run 00_data_prep_and_features first.")

if sel_path.exists():
    with open(sel_path) as f:
        raw_sel = json.load(f)
        feature_cols = raw_sel["selected_features"] if isinstance(raw_sel, dict) and "selected_features" in raw_sel else raw_sel
    print(f"Loaded {len(feature_cols)} selected acoustic features from {sel_path.name}")
else:
    NON_FEATURE_COLS = {"clip_id", "source_clip_id", "label", "split", "start_sec", "end_sec", "is_poor_quality"}
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    print(f"Using all {len(feature_cols)} numeric features.")

print(f"\\nClass Breakdown:")
print(df['label'].value_counts())
"""),
    md("""## 3. Stratified Splitting (80% Train / 10% Val / 10% Test) & Standardization
We maintain stratified class proportions across splits.
"""),
    code("""le = LabelEncoder()
df['target'] = le.fit_transform(df['label'])
classes = list(le.classes_)
num_classes = len(classes)
print(f"Classes ({num_classes}): {dict(zip(range(num_classes), classes))}")

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

print(f"Dataset Split Sizes:")
print(f"  Training Set   : {len(X_train)} samples")
print(f"  Validation Set : {len(X_val)} samples")
print(f"  Held-Out Test  : {len(X_test)} samples")
"""),
    md("""## 4. Hyperparameter Sweeps: Analyzing the Overfitting Curve
Decision Trees are notorious for overfitting when allowed to grow unconstrained.
We sweep `max_depth` from 2 to 14 (and unconstrained `None`) to analyze the **Bias-Variance Tradeoff**.
"""),
    code("""depth_candidates = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, None]
train_scores = []
val_scores = []
cv_scores = []

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

print("Sweeping Decision Tree max_depth...")
for d in depth_candidates:
    dt = DecisionTreeClassifier(max_depth=d, class_weight='balanced', random_state=RANDOM_SEED)
    
    # 5-fold cross validation
    cv_res = cross_validate(dt, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
    cv_f1 = cv_res['test_score'].mean()
    cv_scores.append(cv_f1)
    
    # Train vs Val fitting
    dt.fit(X_train, y_train)
    tr_f1 = f1_score(y_train, dt.predict(X_train), average='macro')
    va_f1 = f1_score(y_val, dt.predict(X_val), average='macro')
    
    train_scores.append(tr_f1)
    val_scores.append(va_f1)
    
    depth_label = str(d) if d is not None else "None (Full)"
    print(f"  max_depth={depth_label:<12} -> Train F1: {tr_f1:.3f} | Val F1: {va_f1:.3f} | CV F1: {cv_f1:.3f}")

# Find optimal depth
best_idx = np.argmax(cv_scores)
best_depth = depth_candidates[best_idx]
print(f"\\nOptimal Pruning Parameter: max_depth={best_depth} (CV Macro F1 = {cv_scores[best_idx]:.3f})")

# Plot Overfitting / Bias-Variance Curve
plt.figure(figsize=(9, 5))
x_axis = [str(d) if d is not None else "None" for d in depth_candidates]
plt.plot(x_axis, train_scores, 'o--', color='#e76f51', lw=2, label='Training Macro F1 (Overfitting)')
plt.plot(x_axis, val_scores, 's-', color='#2a9d8f', lw=2.2, label='Validation Macro F1')
plt.plot(x_axis, cv_scores, '^-.', color='#2b5c8f', lw=2, label='5-Fold CV Macro F1')
plt.axvline(x=x_axis[best_idx], color='red', linestyle=':', label=f'Optimal depth={best_depth}')

plt.title("Decision Tree: Bias-Variance Trade-off across Tree Depth", fontsize=12, fontweight='bold')
plt.xlabel("Tree Maximum Depth (max_depth)")
plt.ylabel("Macro F1-Score")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/decision_tree_overfitting_curve.png", dpi=300, bbox_inches="tight")
plt.show()
"""),
    md("""## 5. Train Optimal Pruned Decision Tree and Evaluate on Held-Out Test Set"""),
    code("""best_model = DecisionTreeClassifier(
    max_depth=best_depth,
    criterion='gini',
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=RANDOM_SEED
)

t0 = time.time()
best_model.fit(X_train, y_train)
train_time = time.time() - t0

# Latency benchmark (100 single-segment predictions)
t_start = time.perf_counter()
n_iters = 100
for _ in range(n_iters):
    _ = best_model.predict(X_test[:1])
latency_ms = ((time.perf_counter() - t_start) / n_iters) * 1000.0

# Predict test set
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
auc_val = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

print("="*60)
print("  OFFICIAL TEST SET BENCHMARK RESULTS — DECISION TREE")
print("="*60)
print(f"  Test Accuracy           : {acc:.2%}")
print(f"  Macro Precision         : {prec:.2%}")
print(f"  Macro Recall            : {rec:.2%}")
print(f"  Macro F1-Score          : {f1:.2%}")
print(f"  Multi-Class ROC-AUC     : {auc_val:.3f}")
print(f"  Inference Latency       : {latency_ms:.3f} ms / 4s segment")
print(f"  Model Training Time     : {train_time:.3f} seconds")
print("="*60)

print("\\nPer-Class Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=classes, digits=3))
"""),
    md("""## 6. Confusion Matrix & Multi-Class ROC Curves"""),
    code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Normalized Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=range(num_classes))
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_norm = np.nan_to_num(cm_norm, nan=0.0)

sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues", ax=axes[0],
            xticklabels=classes, yticklabels=classes, cbar=False)
axes[0].set_title(f"Confusion Matrix (Normalized) — Decision Tree (depth={best_depth})", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Predicted Acoustic Class")
axes[0].set_ylabel("True Ground-Truth Class")

# 2. One-vs-Rest ROC Curves
for i, cls in enumerate(classes):
    y_bin = (y_test == i).astype(int)
    fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
    roc_auc_cls = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, lw=2, label=f"{cls} (AUC = {roc_auc_cls:.3f})")

axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6)
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title(f"Multi-Class ROC Curves (OvR) — Decision Tree", fontsize=11, fontweight='bold')
axes[1].legend(loc="lower right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/confusion_and_roc_decision_tree.png", dpi=300, bbox_inches="tight")
plt.show()
"""),
    md("""## 7. White-Box Tree Architecture Visualization
One of the primary academic advantages of a Decision Tree is that an operator can trace the exact logical threshold rules.
We visualize the top levels of the learned tree:
"""),
    code("""plt.figure(figsize=(16, 8))
plot_tree(
    best_model,
    max_depth=3,
    feature_names=feature_cols,
    class_names=classes,
    filled=True,
    rounded=True,
    fontsize=9,
    precision=2
)
plt.title(f"Decision Tree Architecture (Pruned to Top 3 Levels for Clarity)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("plots/decision_tree_architecture.png", dpi=300, bbox_inches="tight")
plt.show()

# Print textual tree rules for first 2 levels
print("First 2 Levels of Decision Rules:")
print(export_text(best_model, feature_names=feature_cols, max_depth=2))
"""),
    md("""## 8. Feature Importance Analysis (MDI Gini)"""),
    code("""importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 5))
importances.plot(kind='barh', color='#2a9d8f')
plt.title("Decision Tree: Top 15 Acoustic Descriptors by Gini Importance", fontsize=11, fontweight='bold')
plt.xlabel("Mean Decrease in Impurity (Gini Reduction)")
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/feature_importance_decision_tree.png", dpi=300, bbox_inches="tight")
plt.show()
"""),
    md("""## 9. Export Model Artifacts & Telemetry for Final Showdown"""),
    code("""joblib.dump(best_model, "models/model_decision_tree.joblib")
joblib.dump(scaler,     "models/scaler_decision_tree.joblib")
joblib.dump(le,         "models/label_encoder_decision_tree.joblib")

metrics = {
    "algorithm": "Decision Tree",
    "slug": "decision_tree",
    "best_hyperparameters": {
        "max_depth": best_depth,
        "criterion": "gini",
        "min_samples_split": 4,
        "min_samples_leaf": 2
    },
    "test_accuracy": float(acc),
    "macro_precision": float(prec),
    "macro_recall": float(rec),
    "macro_f1": float(f1),
    "roc_auc_macro": float(auc_val),
    "latency_ms": float(latency_ms),
    "training_time_s": float(train_time),
    "classes": classes,
    "feature_count": len(feature_cols)
}

with open("models/metrics_decision_tree.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved model artifacts to models/model_decision_tree.joblib")
print("Saved metrics to models/metrics_decision_tree.json:")
print(json.dumps(metrics, indent=2))
""")
]

with open("03_person3_decision_tree_and_random_forest/decision_tree.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_nb(dt_cells), f, indent=2)

print("Created 03_person3_decision_tree_and_random_forest/decision_tree.ipynb")
