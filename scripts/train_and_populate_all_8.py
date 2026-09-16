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

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

from src.utils.config import FEATURES_PARQ, FEATURES_CSV, SEL_FEATURES, RANDOM_SEED

# Load dataset
path = FEATURES_PARQ if FEATURES_PARQ.exists() else FEATURES_CSV
df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)

with open(SEL_FEATURES) as f:
    raw_sel = json.load(f)
    feature_cols = raw_sel["selected_features"] if isinstance(raw_sel, dict) and "selected_features" in raw_sel else raw_sel

le = LabelEncoder()
df["target"] = le.fit_transform(df["label"])
classes = list(le.classes_)
num_classes = len(classes)

train_mask = df["split"] == "train" if "split" in df.columns else df.index.isin(df.sample(frac=0.8, random_state=RANDOM_SEED).index)
test_mask  = df["split"] == "test" if "split" in df.columns else ~train_mask

X_train_raw = np.nan_to_num(df.loc[train_mask, feature_cols].values, nan=0.0)
y_train     = df.loc[train_mask, "target"].values

X_test_raw  = np.nan_to_num(df.loc[test_mask, feature_cols].values, nan=0.0)
y_test      = df.loc[test_mask, "target"].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

# Definition of all 8 algorithms across 4 teammates
algo_configs = [
    # Person 1
    ("01_person1_lr_and_knn", "logistic_regression", "Logistic Regression",
     LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED)),
    ("01_person1_lr_and_knn", "knn", "K-Nearest Neighbors",
     KNeighborsClassifier(n_neighbors=5, weights='distance')),
    
    # Person 2
    ("02_person2_svm_and_naive_bayes", "svm_rbf", "Support Vector Machine (RBF)",
     SVC(C=10.0, kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_SEED)),
    ("02_person2_svm_and_naive_bayes", "naive_bayes", "Gaussian Naive Bayes",
     GaussianNB(var_smoothing=1e-8)),
    
    # Person 3
    ("03_person3_decision_tree_and_random_forest", "decision_tree", "Decision Tree",
     DecisionTreeClassifier(max_depth=7, class_weight='balanced', random_state=RANDOM_SEED)),
    ("03_person3_decision_tree_and_random_forest", "random_forest", "Random Forest",
     RandomForestClassifier(n_estimators=150, max_depth=12, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)),
    
    # Person 4
    ("04_person4_adaboost_and_xgboost", "adaboost", "AdaBoost",
     AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=100, learning_rate=0.1, random_state=RANDOM_SEED)),
    ("04_person4_adaboost_and_xgboost", "xgboost", "XGBoost",
     XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss", random_state=RANDOM_SEED, n_jobs=-1))
]

print("="*80)
print("TRAINING & POPULATING ALL 8 CLASSICAL ML ALGORITHMS")
print("="*80)

summary_rows = []

for folder, slug, name, model in algo_configs:
    f_dir = Path(folder)
    m_dir = f_dir / "models"
    p_dir = f_dir / "plots"
    m_dir.mkdir(parents=True, exist_ok=True)
    p_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure .gitkeep
    (m_dir / ".gitkeep").touch()
    (p_dir / ".gitkeep").touch()
    
    # Train
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    
    # Latency benchmark
    t_start = time.perf_counter()
    for _ in range(50):
        _ = model.predict(X_test[:1])
    latency_ms = ((time.perf_counter() - t_start) / 50) * 1000.0
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
    auc_val = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro") if y_prob is not None else 0.0
    
    # Save artifacts
    joblib.dump(model,  m_dir / f"model_{slug}.joblib")
    joblib.dump(scaler, m_dir / f"scaler_{slug}.joblib")
    joblib.dump(le,     m_dir / f"label_encoder_{slug}.joblib")
    
    # Also save generic model.joblib if first in folder
    if not (m_dir / "model.joblib").exists():
        joblib.dump(model,  m_dir / "model.joblib")
        joblib.dump(scaler, m_dir / "scaler.joblib")
    
    metrics = {
        "algorithm": name,
        "slug": slug,
        "test_accuracy": float(acc),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "roc_auc_macro": float(auc_val),
        "latency_ms": float(latency_ms),
        "training_time_s": float(fit_time),
        "classes": classes
    }
    with open(m_dir / f"metrics_{slug}.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Generate and save Confusion Matrix Plot
    plt.figure(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred, labels=range(num_classes))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm, nan=0.0)
    sns.heatmap(cm_norm, annot=True, fmt=".0%", cmap="Blues", cbar=False,
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix — {name}", fontsize=10, fontweight='bold')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(p_dir / f"confusion_matrix_{slug}.png", dpi=200, bbox_inches="tight")
    plt.close()
    
    # Generate and save ROC Curve Plot
    if y_prob is not None:
        plt.figure(figsize=(6, 4.5))
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
        plt.title(f"ROC Curves — {name}", fontsize=10, fontweight='bold')
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(p_dir / f"roc_curves_{slug}.png", dpi=200, bbox_inches="tight")
        plt.close()
        
    summary_rows.append({
        "Folder": folder,
        "Algorithm": name,
        "Accuracy": f"{acc:.2%}",
        "Macro F1": f"{f1:.2%}",
        "ROC-AUC": f"{auc_val:.3f}",
        "Latency": f"{latency_ms:.2f} ms"
    })
    print(f"  [DONE] {name:<30} -> Acc: {acc:.2%}, F1: {f1:.2%}, Latency: {latency_ms:.2f} ms")

print("\n" + "="*80)
print("ALL 8 MODELS TRAINED, EVALUATED, AND PLOTS SAVED!")
print("="*80)
df_sum = pd.DataFrame(summary_rows)
print(df_sum.to_string(index=False))
