import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.utils.config import FEATURES_PARQ, FEATURES_CSV, SEL_FEATURES, RANDOM_SEED

# Load data
path = FEATURES_PARQ if FEATURES_PARQ.exists() else FEATURES_CSV
df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)

with open(SEL_FEATURES) as f:
    raw_sel = json.load(f)
    feature_cols = raw_sel["selected_features"] if isinstance(raw_sel, dict) and "selected_features" in raw_sel else raw_sel

le = LabelEncoder()
df["target"] = le.fit_transform(df["label"])
classes = list(le.classes_)

train_mask = df["split"] == "train" if "split" in df.columns else df.index.isin(df.sample(frac=0.8, random_state=RANDOM_SEED).index)
test_mask  = df["split"] == "test" if "split" in df.columns else ~train_mask

X_train_raw = np.nan_to_num(df.loc[train_mask, feature_cols].values, nan=0.0)
y_train     = df.loc[train_mask, "target"].values

X_test_raw  = np.nan_to_num(df.loc[test_mask, feature_cols].values, nan=0.0)
y_test      = df.loc[test_mask, "target"].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

algo_defs = [
    ("01_logistic_regression", "Logistic Regression", LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED)),
    ("02_k_nearest_neighbors", "K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5, weights='distance')),
    ("03_support_vector_machine", "Support Vector Machine", SVC(C=10.0, kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_SEED)),
    ("04_random_forest", "Random Forest", RandomForestClassifier(n_estimators=150, max_depth=12, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)),
    ("05_xgboost", "XGBoost", XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss", random_state=RANDOM_SEED, n_jobs=-1))
]

print("Training baseline artifacts for all 5 algorithm directories...")
for folder, name, model in algo_defs:
    m_dir = Path(folder) / "models"
    m_dir.mkdir(parents=True, exist_ok=True)
    
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    
    # Latency
    t_start = time.perf_counter()
    for _ in range(50):
        _ = model.predict(X_test[:1])
    latency_ms = ((time.perf_counter() - t_start) / 50) * 1000.0
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    auc_val = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    
    joblib.dump(model,  m_dir / "model.joblib")
    joblib.dump(scaler, m_dir / "scaler.joblib")
    joblib.dump(le,     m_dir / "label_encoder.joblib")
    
    metrics = {
        "algorithm": name,
        "test_accuracy": float(acc),
        "macro_f1": float(f1),
        "roc_auc_macro": float(auc_val),
        "latency_ms": float(latency_ms),
        "training_time_s": float(fit_time),
        "classes": classes
    }
    with open(m_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"  [DONE] {name:<25} -> Test Acc: {acc:.2%}, Macro F1: {f1:.2%}, Latency: {latency_ms:.2f} ms")

print("\nAll algorithm baseline models populated successfully!")
