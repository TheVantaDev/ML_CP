import json
from pathlib import Path
import joblib

with open("data/features/selected_features.json") as f:
    sf_66 = json.load(f)["selected_features"]
print("selected_features.json count:", len(sf_66))

models_to_check = [
    ("Person 1 LR", "01_person1_lr_and_knn/models/model_logistic_regression.joblib", "01_person1_lr_and_knn/models/scaler_logistic_regression.joblib"),
    ("Person 1 KNN", "01_person1_lr_and_knn/models/model_knn.joblib", "01_person1_lr_and_knn/models/scaler_knn.joblib"),
    ("Person 2 SVM", "02_person2_svm_and_naive_bayes/models/model_svm_rbf.joblib", "02_person2_svm_and_naive_bayes/models/scaler_svm_rbf.joblib"),
    ("Person 2 NB", "02_person2_svm_and_naive_bayes/models/model_naive_bayes.joblib", "02_person2_svm_and_naive_bayes/models/scaler_naive_bayes.joblib"),
    ("Person 3 DT", "03_person3_decision_tree_and_random_forest/models/model_decision_tree.joblib", "03_person3_decision_tree_and_random_forest/models/scaler_decision_tree.joblib"),
    ("Person 3 RF", "03_person3_decision_tree_and_random_forest/models/model_random_forest.joblib", "03_person3_decision_tree_and_random_forest/models/scaler_random_forest.joblib"),
    ("Person 4 Ada", "04_person4_adaboost_and_xgboost/models/model_adaboost.joblib", "04_person4_adaboost_and_xgboost/models/scaler_adaboost.joblib"),
    ("Person 4 XGB", "04_person4_adaboost_and_xgboost/models/model_xgboost.joblib", "04_person4_adaboost_and_xgboost/models/scaler_xgboost.joblib"),
]

for label, m_path, s_path in models_to_check:
    m = joblib.load(m_path)
    s = joblib.load(s_path) if s_path and Path(s_path).exists() else None
    s_feat = getattr(s, "feature_names_in_", None)
    m_feat = getattr(m, "feature_names_in_", None)
    print(f"\n[{label}]")
    print(f"  Model features: {getattr(m, 'n_features_in_', None)} (names: {len(m_feat) if m_feat is not None else 'None'})")
    print(f"  Scaler features: {getattr(s, 'n_features_in_', None)} (names: {len(s_feat) if s_feat is not None else 'None'})")
    if s_feat is not None:
        diff_missing = set(sf_66) - set(s_feat)
        diff_extra = set(s_feat) - set(sf_66)
        print(f"  Diff from 66 - Missing: {diff_missing}, Extra: {diff_extra}")
