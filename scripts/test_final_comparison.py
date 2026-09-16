import os
import sys
import time
import json
from pathlib import Path

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

model_configs = [
    ("Logistic Regression", Path("01_logistic_regression/models")),
    ("K-Nearest Neighbors", Path("02_k_nearest_neighbors/models")),
    ("Support Vector Machine", Path("03_support_vector_machine/models")),
    ("Random Forest", Path("04_random_forest/models")),
    ("XGBoost", Path("05_xgboost/models"))
]

records = []
for name, m_dir in model_configs:
    m = joblib.load(m_dir / "model.joblib")
    s = joblib.load(m_dir / "scaler.joblib") if (m_dir / "scaler.joblib").exists() else None
    
    X_scaled = s.transform(X_test_raw) if s else X_test_raw
    y_pred = m.predict(X_scaled)
    y_prob = m.predict_proba(X_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    auc_val = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    
    records.append({
        "Model": name,
        "Accuracy": f"{acc:.2%}",
        "Macro F1": f"{f1:.2%}",
        "ROC-AUC": f"{auc_val:.3f}"
    })

df_summary = pd.DataFrame(records)
print("="*60)
print("FINAL BENCHMARK COMPARISON TABLE (TEST SET)")
print("="*60)
print(df_summary.to_string(index=False))
