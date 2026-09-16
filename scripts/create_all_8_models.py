import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from scripts.builder_single_model import build_single_model_nb

# --- 1. LOGISTIC REGRESSION ---
lr_theory = """**Multinomial Logistic Regression** with L2 regularization is our primary linear reference.
It estimates class posterior distributions via the softmax function:
$$P(y = c \\mid \\mathbf{x}) = \\frac{e^{\\mathbf{w}_c^T \\mathbf{x} + b_c}}{\\sum_{j=1}^K e^{\\mathbf{w}_j^T \\mathbf{x} + b_j}}$$
Regularization parameter $C$ controls the trade-off between margin width and empirical cross-entropy loss."""

lr_tuning = """from sklearn.linear_model import LogisticRegression

c_values = [0.01, 0.1, 1.0, 10.0, 50.0]
cv_scores = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

print("Tuning Regularization (C)...")
for c in c_values:
    m = LogisticRegression(C=c, max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED)
    s = cross_validate(m, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
    cv_scores.append((c, s['test_score'].mean(), s['test_score'].std()))

df_c = pd.DataFrame(cv_scores, columns=['C', 'Mean_F1', 'Std_F1'])
best_c = float(df_c.loc[df_c['Mean_F1'].idxmax(), 'C'])
print(f"Optimal C: {best_c}")

best_model = LogisticRegression(C=best_c, max_iter=1500, class_weight='balanced', random_state=RANDOM_SEED)"""

lr_interp = """plt.figure(figsize=(10, 5))
v_idx = classes.index("vessel_like") if "vessel_like" in classes else 0
weights = pd.Series(best_model.coef_[v_idx], index=feature_cols).sort_values()
top_w = pd.concat([weights.head(8), weights.tail(8)])
top_w.plot(kind='barh', color=(top_w > 0).map({True: '#2b5c8f', False: '#e76f51'}))
plt.title(f"Logistic Regression: Feature Weights for {classes[v_idx]}", fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/feature_coefficients_logistic_regression.png", dpi=300, bbox_inches="tight")
plt.show()"""

with open("01_person1_lr_and_knn/logistic_regression.ipynb", "w", encoding="utf-8") as f:
    json.dump(build_single_model_nb("Logistic Regression", "logistic_regression", "01_person1_lr_and_knn", lr_theory, lr_tuning, lr_interp), f, indent=2)

# --- 2. K-NEAREST NEIGHBORS (KNN) ---
knn_theory = """**K-Nearest Neighbors (KNN)** is an instance-based metric classifier that predicts class membership from the $k$ closest neighbors:
$$d(\\mathbf{x}, \\mathbf{z}) = \\sqrt{\\sum_{j=1}^D (x_j - z_j)^2}$$
We use distance-weighted voting ($w_i = 1 / d_i$) to reduce the influence of distant outliers."""

knn_tuning = """from sklearn.neighbors import KNeighborsClassifier

k_values = [1, 3, 5, 7, 9, 11, 15, 21]
cv_scores = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

print("Tuning k neighbors...")
for k in k_values:
    m = KNeighborsClassifier(n_neighbors=k, weights='distance')
    s = cross_validate(m, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
    cv_scores.append((k, s['test_score'].mean()))

df_k = pd.DataFrame(cv_scores, columns=['k', 'Mean_F1'])
best_k = int(df_k.loc[df_k['Mean_F1'].idxmax(), 'k'])
print(f"Optimal k: {best_k}")

best_model = KNeighborsClassifier(n_neighbors=best_k, weights='distance')"""

knn_interp = """from sklearn.inspection import permutation_importance
perm = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=RANDOM_SEED, n_jobs=-1)
p_series = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 5))
p_series.plot(kind='barh', color='#2a9d8f')
plt.title("KNN: Permutation Feature Importance", fontsize=11, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/permutation_importance_knn.png", dpi=300, bbox_inches="tight")
plt.show()"""

with open("01_person1_lr_and_knn/knn.ipynb", "w", encoding="utf-8") as f:
    json.dump(build_single_model_nb("K-Nearest Neighbors", "knn", "01_person1_lr_and_knn", knn_theory, knn_tuning, knn_interp), f, indent=2)

# --- 3. SUPPORT VECTOR MACHINE (SVM) ---
svm_theory = """**Support Vector Machine (SVM) with Radial Basis Function (RBF) Kernel** separates classes by projecting feature vectors into high-dimensional Hilbert space:
$$K(\\mathbf{x}, \\mathbf{z}) = \\exp\\left(-\\gamma \\|\\mathbf{x} - \\mathbf{z}\\|^2\\right)$$
Slack variables $\\xi_i$ and penalty $C$ govern soft-margin boundaries for non-linearly separable acoustic clusters."""

svm_tuning = """from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.5, 1.0, 10.0], 'gamma': ['scale', 0.01, 0.05]}
grid = GridSearchCV(SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_SEED),
                    param_grid, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED),
                    scoring='f1_macro', n_jobs=-1)
grid.fit(X_train, y_train)
best_c = grid.best_params_['C']
best_g = grid.best_params_['gamma']
print(f"Optimal SVM: C={best_c}, gamma={best_g}")

best_model = SVC(kernel='rbf', C=best_c, gamma=best_g, probability=True, class_weight='balanced', random_state=RANDOM_SEED)"""

svm_interp = """from sklearn.inspection import permutation_importance
perm = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=RANDOM_SEED, n_jobs=-1)
p_series = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 5))
p_series.plot(kind='barh', color='#e76f51')
plt.title("SVM: Top 15 Permutation Feature Importances", fontsize=11, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/permutation_importance_svm.png", dpi=300, bbox_inches="tight")
plt.show()"""

with open("02_person2_svm_and_naive_bayes/svm_rbf.ipynb", "w", encoding="utf-8") as f:
    json.dump(build_single_model_nb("Support Vector Machine (RBF)", "svm_rbf", "02_person2_svm_and_naive_bayes", svm_theory, svm_tuning, svm_interp), f, indent=2)

# --- 4. GAUSSIAN NAIVE BAYES ---
gnb_theory = """**Gaussian Naive Bayes (GNB)** is a probabilistic generative classifier that assumes features are conditionally independent given the class label:
$$P(\\mathbf{x} \\mid y = c) = \\prod_{j=1}^D \\frac{1}{\\sqrt{2\\pi \\sigma_{cj}^2}} \\exp\\left( -\\frac{(x_j - \\mu_{cj})^2}{2\\sigma_{cj}^2} \\right)$$
Despite the independence assumption, Gaussian Naive Bayes is extremely fast, highly robust to noise, and provides a classic probabilistic baseline."""

gnb_tuning = """from sklearn.naive_bayes import GaussianNB

var_smoothing_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
cv_scores = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

print("Tuning GaussianNB var_smoothing...")
for vs in var_smoothing_values:
    m = GaussianNB(var_smoothing=vs)
    s = cross_validate(m, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
    cv_scores.append((vs, s['test_score'].mean()))

df_vs = pd.DataFrame(cv_scores, columns=['var_smoothing', 'Mean_F1'])
best_vs = float(df_vs.loc[df_vs['Mean_F1'].idxmax(), 'var_smoothing'])
print(f"Optimal var_smoothing: {best_vs}")

best_model = GaussianNB(var_smoothing=best_vs)"""

gnb_interp = """from sklearn.inspection import permutation_importance
perm = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=RANDOM_SEED, n_jobs=-1)
p_series = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 5))
p_series.plot(kind='barh', color='#f4a261')
plt.title("Gaussian Naive Bayes: Permutation Feature Importance", fontsize=11, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/permutation_importance_naive_bayes.png", dpi=300, bbox_inches="tight")
plt.show()"""

with open("02_person2_svm_and_naive_bayes/naive_bayes.ipynb", "w", encoding="utf-8") as f:
    json.dump(build_single_model_nb("Gaussian Naive Bayes", "naive_bayes", "02_person2_svm_and_naive_bayes", gnb_theory, gnb_tuning, gnb_interp), f, indent=2)

# --- 5. DECISION TREE (CART) ---
dt_theory = """**Decision Trees (CART)** partition the acoustic feature space into axis-aligned rectangular regions using recursive binary splitting.
At each node, the split that maximizes the reduction in **Gini Impurity** is chosen:
$$I_G(p) = 1 - \\sum_{c=1}^K p_c^2$$
Single decision trees offer total rule interpretability and serve as the foundation for Random Forests and Gradient Boosting."""

dt_tuning = """from sklearn.tree import DecisionTreeClassifier

depth_list = [3, 5, 7, 9, 12, None]
cv_scores = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

print("Tuning Decision Tree max_depth...")
for d in depth_list:
    m = DecisionTreeClassifier(max_depth=d, class_weight='balanced', random_state=RANDOM_SEED)
    s = cross_validate(m, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
    cv_scores.append((str(d), s['test_score'].mean()))

df_dt = pd.DataFrame(cv_scores, columns=['depth', 'Mean_F1'])
best_d_str = df_dt.loc[df_dt['Mean_F1'].idxmax(), 'depth']
best_depth = None if best_d_str == 'None' else int(best_d_str)
print(f"Optimal max_depth: {best_depth}")

best_model = DecisionTreeClassifier(max_depth=best_depth, class_weight='balanced', random_state=RANDOM_SEED)"""

dt_interp = """from sklearn.tree import plot_tree
importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 5))
importances.plot(kind='barh', color='#2a9d8f')
plt.title("Decision Tree: Gini Feature Importance", fontsize=11, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/feature_importance_decision_tree.png", dpi=300, bbox_inches="tight")
plt.show()"""

with open("03_person3_decision_tree_and_random_forest/decision_tree.ipynb", "w", encoding="utf-8") as f:
    json.dump(build_single_model_nb("Decision Tree", "decision_tree", "03_person3_decision_tree_and_random_forest", dt_theory, dt_tuning, dt_interp), f, indent=2)

# --- 6. RANDOM FOREST ---
rf_theory = """**Random Forest** builds an ensemble of de-correlated Decision Trees using **Bootstrap Aggregation (Bagging)** and random feature subspacing:
$$\\hat{P}(y = c \\mid \\mathbf{x}) = \\frac{1}{B} \\sum_{b=1}^B P_b(y = c \\mid \\mathbf{x})$$
Random feature subspacing ensures that individual trees are not dominated by the same single acoustic descriptor (such as dominant tonal frequency)."""

rf_tuning = """from sklearn.ensemble import RandomForestClassifier

n_est_list = [50, 100, 150]
depth_list = [8, 12, None]
cv_scores = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

print("Tuning Random Forest...")
for n in n_est_list:
    for d in depth_list:
        m = RandomForestClassifier(n_estimators=n, max_depth=d, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)
        s = cross_validate(m, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
        cv_scores.append((n, d, s['test_score'].mean()))

df_rf = pd.DataFrame(cv_scores, columns=['n_estimators', 'max_depth', 'Mean_F1'])
best_row = df_rf.loc[df_rf['Mean_F1'].idxmax()]
best_n = int(best_row['n_estimators'])
best_d = None if pd.isna(best_row['max_depth']) else int(best_row['max_depth'])
print(f"Optimal RF: n_estimators={best_n}, max_depth={best_d}")

best_model = RandomForestClassifier(n_estimators=best_n, max_depth=best_d, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)"""

rf_interp = """importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 5))
importances.plot(kind='barh', color='#2b5c8f')
plt.title("Random Forest: Top 15 Gini Feature Importances", fontsize=11, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/feature_importance_random_forest.png", dpi=300, bbox_inches="tight")
plt.show()"""

with open("03_person3_decision_tree_and_random_forest/random_forest.ipynb", "w", encoding="utf-8") as f:
    json.dump(build_single_model_nb("Random Forest", "random_forest", "03_person3_decision_tree_and_random_forest", rf_theory, rf_tuning, rf_interp), f, indent=2)

# --- 7. ADABOOST ---
ada_theory = """**AdaBoost (Adaptive Boosting)** sequentially trains weak base learners (typically decision stumps) where misclassified instances receive higher sample weights in subsequent rounds:
$$w_{i}^{(m+1)} = w_i^{(m)} \\exp\\left( \\alpha_m \\cdot \\mathbb{I}(y_i \\neq h_m(\\mathbf{x}_i)) \\right)$$
This forces subsequent estimators to concentrate specifically on difficult acoustic boundary cases."""

ada_tuning = """from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

n_est_list = [50, 100, 150]
lr_list = [0.05, 0.1, 0.5, 1.0]
cv_scores = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

print("Tuning AdaBoost...")
for n in n_est_list:
    for lr in lr_list:
        m = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=n, learning_rate=lr, random_state=RANDOM_SEED)
        s = cross_validate(m, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
        cv_scores.append((n, lr, s['test_score'].mean()))

df_ada = pd.DataFrame(cv_scores, columns=['n_estimators', 'learning_rate', 'Mean_F1'])
best_row = df_ada.loc[df_ada['Mean_F1'].idxmax()]
best_n = int(best_row['n_estimators'])
best_lr = float(best_row['learning_rate'])
print(f"Optimal AdaBoost: n_estimators={best_n}, learning_rate={best_lr}")

best_model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=best_n, learning_rate=best_lr, random_state=RANDOM_SEED)"""

ada_interp = """importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 5))
importances.plot(kind='barh', color='#e76f51')
plt.title("AdaBoost: Feature Importance Ranking", fontsize=11, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/feature_importance_adaboost.png", dpi=300, bbox_inches="tight")
plt.show()"""

with open("04_person4_adaboost_and_xgboost/adaboost.ipynb", "w", encoding="utf-8") as f:
    json.dump(build_single_model_nb("AdaBoost", "adaboost", "04_person4_adaboost_and_xgboost", ada_theory, ada_tuning, ada_interp), f, indent=2)

# --- 8. XGBOOST ---
xgb_theory = """**XGBoost (Extreme Gradient Boosting)** minimizes regularized objective functions through second-order Taylor expansion approximations:
$$\\mathcal{L}^{(t)} \\approx \\sum_{i=1}^N \\left[ g_i f_t(\\mathbf{x}_i) + \\frac{1}{2} h_i f_t^2(\\mathbf{x}_i) \\right] + \\Omega(f_t)$$
Subsampling, column subsampling, and shrinkage parameters ensure state-of-the-art generalization on tabular acoustic vectors."""

xgb_tuning = """from xgboost import XGBClassifier

lr_list = [0.03, 0.08, 0.15]
depth_list = [4, 6]
cv_scores = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

print("Tuning XGBoost...")
for lr in lr_list:
    for d in depth_list:
        m = XGBClassifier(n_estimators=120, max_depth=d, learning_rate=lr, subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss", random_state=RANDOM_SEED, n_jobs=-1)
        s = cross_validate(m, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
        cv_scores.append((lr, d, s['test_score'].mean()))

df_xgb = pd.DataFrame(cv_scores, columns=['learning_rate', 'max_depth', 'Mean_F1'])
best_row = df_xgb.loc[df_xgb['Mean_F1'].idxmax()]
best_lr = float(best_row['learning_rate'])
best_d = int(best_row['max_depth'])
print(f"Optimal XGBoost: lr={best_lr}, depth={best_d}")

best_model = XGBClassifier(n_estimators=150, max_depth=best_d, learning_rate=best_lr, subsample=0.85, colsample_bytree=0.85, eval_metric="mlogloss", random_state=RANDOM_SEED, n_jobs=-1)"""

xgb_interp = """gain_imp = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 5))
gain_imp.plot(kind='barh', color='#9b5de5')
plt.title("XGBoost: Top 15 Features by Information Gain", fontsize=11, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/xgboost_gain_importance.png", dpi=300, bbox_inches="tight")
plt.show()"""

with open("04_person4_adaboost_and_xgboost/xgboost.ipynb", "w", encoding="utf-8") as f:
    json.dump(build_single_model_nb("XGBoost", "xgboost", "04_person4_adaboost_and_xgboost", xgb_theory, xgb_tuning, xgb_interp), f, indent=2)

print("Generated all 8 individual algorithm notebooks!")
