import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)

df = pd.read_parquet("data/features/features.parquet")
with open("data/features/selected_features.json") as f:
    feature_cols = json.load(f)["selected_features"]

train_df = df[df["split"] == "train"].copy()
val_df   = df[df["split"] == "val"].copy()
test_df  = df[df["split"] == "test"].copy()

le = LabelEncoder().fit(sorted(df["label"].unique()))
classes = list(le.classes_)

X_train = np.nan_to_num(train_df[feature_cols].values, nan=0.0)
y_train = le.transform(train_df["label"])

X_test = np.nan_to_num(test_df[feature_cols].values, nan=0.0)
y_test = le.transform(test_df["label"])

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

p4_models_dir = Path("04_person4_adaboost_and_xgboost/models")
p4_plots_dir  = Path("04_person4_adaboost_and_xgboost/plots")
p4_models_dir.mkdir(parents=True, exist_ok=True)
p4_plots_dir.mkdir(parents=True, exist_ok=True)

# 1. AdaBoost
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)
t0 = time.perf_counter()
ada.fit(X_train_s, y_train)
train_time_ada = time.perf_counter() - t0

t0 = time.perf_counter()
y_pred_ada = ada.predict(X_test_s)
lat_ada = (time.perf_counter() - t0) * 1000 / len(y_test)
y_prob_ada = ada.predict_proba(X_test_s)

acc_ada  = float(accuracy_score(y_test, y_pred_ada))
prec_ada = float(precision_score(y_test, y_pred_ada, average="macro", zero_division=0))
rec_ada  = float(recall_score(y_test, y_pred_ada, average="macro", zero_division=0))
f1_ada   = float(f1_score(y_test, y_pred_ada, average="macro", zero_division=0))
auc_ada  = float(roc_auc_score(y_test, y_prob_ada, multi_class="ovr", average="macro"))

joblib.dump(ada, p4_models_dir / "model_adaboost.joblib")
joblib.dump(scaler, p4_models_dir / "scaler_adaboost.joblib")
joblib.dump(le, p4_models_dir / "label_encoder_adaboost.joblib")

metrics_ada = {
    "algorithm": "AdaBoost",
    "slug": "adaboost",
    "best_hyperparameters": {"n_estimators": 200, "learning_rate": 0.1, "base_estimator": "DecisionTreeClassifier(max_depth=1)"},
    "test_accuracy": acc_ada,
    "macro_precision": prec_ada,
    "macro_recall": rec_ada,
    "macro_f1": f1_ada,
    "roc_auc_macro": auc_ada,
    "latency_ms": lat_ada,
    "training_time_s": train_time_ada,
    "classes": classes,
    "feature_count": len(feature_cols)
}
with open(p4_models_dir / "metrics_adaboost.json", "w") as f:
    json.dump(metrics_ada, f, indent=2)

# Ada plot - Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_ada)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.title("AdaBoost Confusion Matrix (Held-out Test)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(p4_plots_dir / "confusion_matrix_adaboost.png", dpi=200)
plt.close()

# 2. XGBoost
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.2,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss"
)
t0 = time.perf_counter()
xgb.fit(X_train_s, y_train)
train_time_xgb = time.perf_counter() - t0

t0 = time.perf_counter()
y_pred_xgb = xgb.predict(X_test_s)
lat_xgb = (time.perf_counter() - t0) * 1000 / len(y_test)
y_prob_xgb = xgb.predict_proba(X_test_s)

acc_xgb  = float(accuracy_score(y_test, y_pred_xgb))
prec_xgb = float(precision_score(y_test, y_pred_xgb, average="macro", zero_division=0))
rec_xgb  = float(recall_score(y_test, y_pred_xgb, average="macro", zero_division=0))
f1_xgb   = float(f1_score(y_test, y_pred_xgb, average="macro", zero_division=0))
auc_xgb  = float(roc_auc_score(y_test, y_prob_xgb, multi_class="ovr", average="macro"))

joblib.dump(xgb, p4_models_dir / "model_xgboost.joblib")
joblib.dump(scaler, p4_models_dir / "scaler_xgboost.joblib")
joblib.dump(le, p4_models_dir / "label_encoder_xgboost.joblib")

metrics_xgb = {
    "algorithm": "XGBoost",
    "slug": "xgboost",
    "best_hyperparameters": {"n_estimators": 200, "learning_rate": 0.2, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8},
    "test_accuracy": acc_xgb,
    "macro_precision": prec_xgb,
    "macro_recall": rec_xgb,
    "macro_f1": f1_xgb,
    "roc_auc_macro": auc_xgb,
    "latency_ms": lat_xgb,
    "training_time_s": train_time_xgb,
    "classes": classes,
    "feature_count": len(feature_cols)
}
with open(p4_models_dir / "metrics_xgboost.json", "w") as f:
    json.dump(metrics_xgb, f, indent=2)

# XGB plot - Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_xgb)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.title("XGBoost Confusion Matrix (Held-out Test)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(p4_plots_dir / "confusion_matrix_xgboost.png", dpi=200)
plt.close()

print("Person 4 models, scalers, metrics, and plots exported successfully!")
