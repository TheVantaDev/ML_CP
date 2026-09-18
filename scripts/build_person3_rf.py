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
# MASTERPIECE: RANDOM FOREST (PERSON 3 - MODEL 2)
# ==============================================================================
rf_cells = [
    md("""# Random Forest Classifier — Underwater Acoustic Event Classification
### Sonobuoy Sound Detective: Offline Ground-Station Acoustic System
**Assigned Team Member:** Person 3 (Tree & Ensemble Specialist)  
**Algorithm:** Bagged Ensemble of Random Trees (Breiman, 2001)  
**Folder:** `03_person3_decision_tree_and_random_forest/`

---

## 1. Theoretical Foundation & Mathematical Formulation

The **Random Forest** is an ensemble meta-estimator that combines $B$ diverse decision trees $\\{T_1, T_2, \\dots, T_B\\}$ to reduce the variance of individual tree predictors without increasing bias.

### 1. Bootstrap Aggregation (Bagging)
Each tree $T_b$ is trained on an independent bootstrap sample $D_b$ drawn with replacement from training dataset $D$ ($|D_b| = |D|$).
Each sample has an expected $\\approx 63.2\\%$ unique training examples, leaving $\\approx 36.8\\%$ as **Out-of-Bag (OOB)** samples for validation:
$$\\lim_{N \\to \\infty} \\left(1 - \\frac{1}{N}\\right)^N = \\frac{1}{e} \\approx 0.368$$

### 2. Random Subspace Feature De-correlation
At every split inside each tree, only a random subset of $m = \\lfloor\\sqrt{D}\\rfloor$ features is considered.
This prevents a single dominant acoustic feature (such as dominant tonal peak frequency) from dictating all trees, dramatically lowering pairwise tree correlation $\\rho$.

### 3. Variance Reduction Law
The variance of an average of $B$ identically distributed (but correlated with coefficient $\\rho$) tree estimators is:
$$\\text{Var}\\left(\\frac{1}{B} \\sum_{b=1}^B T_b(\\mathbf{x})\\right) = \\rho \\sigma^2 + \\frac{1 - \\rho}{B} \\sigma^2$$
As $B \\to \\infty$, the second term approaches zero, leaving $\\rho \\sigma^2$. By forcing $\\rho$ lower through random feature selection, Random Forest achieves far lower variance than standard Bagging!
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.inspection import permutation_importance

Path("plots").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("Environment loaded for Person 3: Random Forest Classifier.")
"""),
    md("""## 2. Load Shared Acoustic Features and Selected Descriptors"""),
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

print(f"\\nClass Distribution:\\n{df['label'].value_counts()}")
"""),
    md("""## 3. Stratified Splitting & Scaling"""),
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

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
"""),
    md("""## 4. Hyperparameter Sweeps: Out-of-Bag (OOB) Error vs Number of Trees
We analyze how the ensemble stabilizes as the number of estimators increases from 10 to 250 using **Out-of-Bag (OOB) Error**.
"""),
    code("""n_trees_candidates = [10, 25, 50, 75, 100, 150, 200, 250]
oob_scores = []
cv_scores = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

print("Sweeping n_estimators with OOB error tracking...")
for n in n_trees_candidates:
    rf = RandomForestClassifier(
        n_estimators=n,
        max_depth=12,
        oob_score=True,
        bootstrap=True,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    oob_scores.append(rf.oob_score_)
    
    cv_res = cross_validate(rf, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
    cv_scores.append(cv_res['test_score'].mean())
    
    print(f"  n_estimators={n:<4} -> OOB Accuracy: {rf.oob_score_:.3f} | 5-Fold CV F1: {cv_res['test_score'].mean():.3f}")

best_n_idx = np.argmax(cv_scores)
best_n_est = n_trees_candidates[best_n_idx]
print(f"\\nOptimal Ensemble Size: n_estimators={best_n_est}")

# Plot OOB Accuracy & CV F1 vs Number of Estimators
plt.figure(figsize=(9, 4.5))
plt.plot(n_trees_candidates, oob_scores, 'o-', color='#2a9d8f', lw=2, label='OOB Accuracy')
plt.plot(n_trees_candidates, cv_scores, 's--', color='#2b5c8f', lw=2, label='5-Fold CV Macro F1')
plt.axvline(x=best_n_est, color='red', linestyle=':', label=f'Optimal n={best_n_est}')
plt.title("Random Forest: Ensemble Stabilization Curve (OOB Accuracy vs n_estimators)", fontsize=11, fontweight='bold')
plt.xlabel("Number of Estimator Trees (n_estimators)")
plt.ylabel("Score")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/rf_oob_stabilization_curve.png", dpi=300, bbox_inches="tight")
plt.show()
"""),
    md("""## 5. 2D Hyperparameter Grid Search (n_estimators vs max_depth)"""),
    code("""depth_grid = [6, 10, 14, None]
n_grid = [50, 100, 150, 200]
heatmap_data = np.zeros((len(depth_grid), len(n_grid)))

print("Running 2D Grid Evaluation...")
for i, d in enumerate(depth_grid):
    for j, n in enumerate(n_grid):
        rf_grid = RandomForestClassifier(n_estimators=n, max_depth=d, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)
        res = cross_validate(rf_grid, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
        heatmap_data[i, j] = res['test_score'].mean()

df_heat = pd.DataFrame(
    heatmap_data,
    index=[str(d) for d in depth_grid],
    columns=[str(n) for n in n_grid]
)

# Plot Heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(df_heat, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': '5-Fold CV Macro F1'})
plt.title("Random Forest: 2D Parameter Grid Search (max_depth vs n_estimators)", fontsize=11, fontweight='bold')
plt.ylabel("Max Depth")
plt.xlabel("Number of Trees")
plt.tight_layout()
plt.savefig("plots/rf_grid_search_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

best_depth = 12
"""),
    md("""## 6. Train Optimal Random Forest & Evaluate on Held-Out Test Set"""),
    code("""best_model = RandomForestClassifier(
    n_estimators=best_n_est,
    max_depth=best_depth,
    min_samples_split=4,
    class_weight='balanced',
    random_state=RANDOM_SEED,
    n_jobs=-1
)

t0 = time.time()
best_model.fit(X_train, y_train)
train_time = time.time() - t0

# Latency Benchmark
t_start = time.perf_counter()
n_iters = 100
for _ in range(n_iters):
    _ = best_model.predict(X_test[:1])
latency_ms = ((time.perf_counter() - t_start) / n_iters) * 1000.0

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
auc_val = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

print("="*60)
print("  OFFICIAL TEST SET BENCHMARK RESULTS — RANDOM FOREST")
print("="*60)
print(f"  Test Accuracy           : {acc:.2%}")
print(f"  Macro Precision         : {prec:.2%}")
print(f"  Macro Recall            : {rec:.2%}")
print(f"  Macro F1-Score          : {f1:.2%}")
print(f"  Multi-Class ROC-AUC     : {auc_val:.3f}")
print(f"  Inference Latency       : {latency_ms:.3f} ms / 4s segment")
print(f"  Training Time           : {train_time:.3f} seconds")
print("="*60)

print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=classes, digits=3))
"""),
    md("""## 7. Confusion Matrix & Multi-Class ROC Curves"""),
    code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=range(num_classes))
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_norm = np.nan_to_num(cm_norm, nan=0.0)

sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues", ax=axes[0],
            xticklabels=classes, yticklabels=classes, cbar=False)
axes[0].set_title(f"Confusion Matrix (Normalized) — Random Forest (n={best_n_est})", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Predicted Acoustic Class")
axes[0].set_ylabel("True Ground-Truth Class")

# 2. ROC Curves
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
axes[1].set_title(f"Multi-Class ROC Curves (OvR) — Random Forest", fontsize=11, fontweight='bold')
axes[1].legend(loc="lower right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/confusion_and_roc_random_forest.png", dpi=300, bbox_inches="tight")
plt.show()
"""),
    md("""## 8. Feature Importance Analysis: Gini (MDI) vs Permutation Importance
We compare **Mean Decrease in Impurity (Gini)** against **Permutation Importance** on the held-out test set to ensure high-cardinality features are not artificially favored.
"""),
    code("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. Gini Importance (MDI)
gini_imp = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)
gini_imp.plot(kind='barh', ax=axes[0], color='#2b5c8f')
axes[0].set_title("Top 15 Features by Gini Importance (MDI)", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Mean Decrease in Impurity")
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3)

# 2. Permutation Importance
print("Computing Permutation Importance on Test Set...")
perm = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=RANDOM_SEED, n_jobs=-1)
perm_imp = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False).head(15)
perm_imp.plot(kind='barh', ax=axes[1], color='#2a9d8f')
axes[1].set_title("Top 15 Features by Permutation Importance (Test Set)", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Mean Decrease in Accuracy")
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/feature_importance_comparison_rf.png", dpi=300, bbox_inches="tight")
plt.show()
"""),
    md("""## 9. Export Model Artifacts & Telemetry for Final Showdown"""),
    code("""joblib.dump(best_model, "models/model_random_forest.joblib")
joblib.dump(scaler,     "models/scaler_random_forest.joblib")
joblib.dump(le,         "models/label_encoder_random_forest.joblib")

metrics = {
    "algorithm": "Random Forest",
    "slug": "random_forest",
    "best_hyperparameters": {
        "n_estimators": best_n_est,
        "max_depth": best_depth,
        "min_samples_split": 4,
        "class_weight": "balanced"
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

with open("models/metrics_random_forest.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved model artifacts to models/model_random_forest.joblib")
print("Saved metrics to models/metrics_random_forest.json:")
print(json.dumps(metrics, indent=2))
""")
]

with open("03_person3_decision_tree_and_random_forest/random_forest.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_nb(rf_cells), f, indent=2)

print("Created 03_person3_decision_tree_and_random_forest/random_forest.ipynb")
