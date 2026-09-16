from pathlib import Path

# Add .gitkeep to every subfolder
for p in Path(".").glob("0*/**/"):
    (p / ".gitkeep").touch()

# README for 00_data_prep_and_features
Path("00_data_prep_and_features/README.md").write_text("""# 00: Data Ingestion, Signal Preprocessing & Handcrafted Feature Extraction
**Role:** Shared Data Preparation for All Team Members  
**Primary Notebook:** `data_prep_and_feature_extraction.ipynb`

### Overview:
- Audio ingestion (ShipsEar, DeepShip, Synthetic).
- DSP filtering (20–5,000 Hz Butterworth bandpass).
- Sliding-window 4.0-second segmentation.
- Extracts ~100 handcrafted acoustic features (MFCCs, spectral descriptors, tonal persistence) into `data/features/features.parquet`.
- Runs ANOVA F-test, Mutual Information, and Random Forest feature selection to save `data/features/selected_features.json`.
""", encoding="utf-8")

# README for Person 1
Path("01_person1_lr_and_knn/README.md").write_text("""# Person 1: Linear & Distance-Based Classifiers
**Assigned Team Member:** Person 1  
**Notebooks in this folder:**
1. `logistic_regression.ipynb` — Multinomial Logistic Regression with Softmax & L2 Regularization
2. `knn.ipynb` — Distance-Weighted K-Nearest Neighbors (k=1..25 tuning)

### Contents:
- `models/`: Saved `.joblib` model binaries, scalers, and metric reports.
- `plots/`: Confusion matrices, multi-class ROC curves, and coefficient plots.
""", encoding="utf-8")

# README for Person 2
Path("02_person2_svm_and_naive_bayes/README.md").write_text("""# Person 2: Kernel & Probabilistic Classifiers
**Assigned Team Member:** Person 2  
**Notebooks in this folder:**
1. `svm_rbf.ipynb` — Support Vector Machine with Radial Basis Function (RBF) Kernel & Grid Search
2. `naive_bayes.ipynb` — Gaussian Naive Bayes with variance smoothing tuning

### Contents:
- `models/`: Saved `.joblib` model binaries, scalers, and metric reports.
- `plots/`: Confusion matrices, ROC curves, and permutation importance charts.
""", encoding="utf-8")

# README for Person 3
Path("03_person3_decision_tree_and_random_forest/README.md").write_text("""# Person 3: Tree & Bagging Classifiers
**Assigned Team Member:** Person 3  
**Notebooks in this folder:**
1. `decision_tree.ipynb` — Decision Tree (CART) with max-depth pruning
2. `random_forest.ipynb` — Random Forest Bagged Ensemble (n_estimators and depth tuning)

### Contents:
- `models/`: Saved `.joblib` model binaries, scalers, and metric reports.
- `plots/`: Confusion matrices, ROC curves, and Gini feature importance (MDI) plots.
""", encoding="utf-8")

# README for Person 4
Path("04_person4_adaboost_and_xgboost/README.md").write_text("""# Person 4: Boosting Ensemble Classifiers
**Assigned Team Member:** Person 4  
**Notebooks in this folder:**
1. `adaboost.ipynb` — AdaBoost (Adaptive Boosting) with decision stumps
2. `xgboost.ipynb` — Extreme Gradient Boosting (XGBoost) with shrinkage & second-order Taylor loss

### Contents:
- `models/`: Saved `.joblib` model binaries, scalers, and metric reports.
- `plots/`: Confusion matrices, ROC curves, and feature gain importance plots.
""", encoding="utf-8")

# README for Final Comparison
Path("05_final_comparison_and_system/README.md").write_text("""# 05: Master 8-Model Comparison & Ground-Station Decision Support
**Role:** Final Team Showdown & Operational Prototype (All 4 Team Members)  
**Notebooks in this folder:**
1. `final_model_comparison.ipynb` — Side-by-side benchmark of all 8 algorithms across Accuracy, Macro-F1, ROC-AUC, and Latency.
2. `decision_support_system.ipynb` — Simulates offline sonobuoy stream ingestion, applies deterministic acoustic rules, and renders operator decision cards.

### Contents:
- `models/`: Active production model pipeline.
- `plots/`: Side-by-side confusion matrices (2x4 grid) and unified 8-model ROC curves.
- `reports/`: `master_8_models_comparison.csv`
""", encoding="utf-8")

print("Created all READMEs and ensured .gitkeep in every subfolder!")
