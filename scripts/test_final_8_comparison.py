import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc

parq_path = Path("data/features/features.parquet")
sel_path  = Path("data/features/selected_features.json")

df = pd.read_parquet(parq_path)
with open(sel_path) as f:
    raw_sel = json.load(f)
    feature_cols = raw_sel["selected_features"] if isinstance(raw_sel, dict) and "selected_features" in raw_sel else raw_sel

test_mask = df["split"] == "test"
df_test = df[test_mask].copy()

classes = sorted(df["label"].unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
y_test = df_test["label"].map(class_to_idx).values
X_test_raw = np.nan_to_num(df_test[feature_cols].values, nan=0.0)

model_registry_list = [
    ("Logistic Regression", Path("01_person1_lr_and_knn/models"), "logistic_regression"),
    ("K-Nearest Neighbors", Path("01_person1_lr_and_knn/models"), "knn"),
    ("Support Vector Machine", Path("02_person2_svm_and_naive_bayes/models"), "svm_rbf"),
    ("Gaussian Naive Bayes", Path("02_person2_svm_and_naive_bayes/models"), "naive_bayes"),
    ("Decision Tree", Path("03_person3_decision_tree_and_random_forest/models"), "decision_tree"),
    ("Random Forest", Path("03_person3_decision_tree_and_random_forest/models"), "random_forest"),
    ("AdaBoost", Path("04_person4_adaboost_and_xgboost/models"), "adaboost"),
    ("XGBoost", Path("04_person4_adaboost_and_xgboost/models"), "xgboost"),
]

records = []
predictions = {}
probabilities = {}

for name, m_dir, slug in model_registry_list:
    m = joblib.load(m_dir / f"model_{slug}.joblib")
    s = joblib.load(m_dir / f"scaler_{slug}.joblib") if (m_dir / f"scaler_{slug}.joblib").exists() else None
    
    X_scaled = s.transform(X_test_raw) if s else X_test_raw
    y_pred = m.predict(X_scaled)
    y_prob = m.predict_proba(X_scaled) if hasattr(m, "predict_proba") else None
    
    predictions[name] = y_pred
    probabilities[name] = y_prob
    
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    auc_val = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro") if y_prob is not None else 0.0
    
    records.append({
        "Model": name,
        "Accuracy": f"{acc:.2%}",
        "Macro F1": f"{f1:.2%}",
        "ROC-AUC": f"{auc_val:.3f}"
    })

# Save master CSV
df_sum = pd.DataFrame(records).sort_values("Macro F1", ascending=False)
df_sum.to_csv("05_final_comparison_and_system/reports/master_8_models_comparison.csv", index=False)

# Save 2x4 Confusion Matrix Grid
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
for ax, (name, y_pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, y_pred, labels=range(len(classes)))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm, nan=0.0)
    sns.heatmap(cm_norm, annot=True, fmt=".0%", cmap="Blues", cbar=False, ax=ax,
                xticklabels=classes, yticklabels=classes)
    ax.set_title(name, fontsize=9, fontweight='bold')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
plt.tight_layout()
plt.savefig("05_final_comparison_and_system/plots/all_8_confusion_matrices.png", dpi=200, bbox_inches="tight")
plt.close()

# Save unified ROC Curves
plt.figure(figsize=(8, 5.5))
palette = sns.color_palette("tab10", len(probabilities))
for (name, y_prob), color in zip(probabilities.items(), palette):
    if y_prob is not None:
        all_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(classes)):
            y_bin = (y_test == i).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr /= len(classes)
        macro_auc = auc(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr, lw=1.8, color=color, label=f"{name} (AUC = {macro_auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.6)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Master ROC Curves (All 8 Classical ML Algorithms)", fontsize=11, fontweight='bold')
plt.legend(loc="lower right", fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("05_final_comparison_and_system/plots/all_8_models_roc_curves.png", dpi=200, bbox_inches="tight")
plt.close()

print("="*65)
print("FINAL BENCHMARK COMPARISON TABLE (ALL 8 CLASSICAL ALGORITHMS)")
print("="*65)
print(df_sum.to_string(index=False))
print("\nGenerated and saved master plots into 05_final_comparison_and_system/plots/!")
