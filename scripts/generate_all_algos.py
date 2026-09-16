import json
import sys
from pathlib import Path
sys.path.insert(0, ".")
from scripts.build_algo_template import build_algo_notebook

# ==============================================================================
# 01. LOGISTIC REGRESSION
# ==============================================================================
lr_theory = """**Multinomial Logistic Regression** serves as our primary linear baseline.
It models the posterior probability of each acoustic class via the softmax transformation:
$$P(y = c \\mid \\mathbf{x}) = \\frac{e^{\\mathbf{w}_c^T \\mathbf{x} + b_c}}{\\sum_{j=1}^K e^{\\mathbf{w}_j^T \\mathbf{x} + b_j}}$$

We apply **L2 Regularization (Ridge)** to prevent overfitting on collinear acoustic descriptors:
$$\\mathcal{L}(\\mathbf{w}) = -\\sum_{i=1}^N \\log P(y_i \\mid \\mathbf{x}_i) + \\frac{1}{2C} \\sum_{c=1}^K \\|\\mathbf{w}_c\\|_2^2$$
where $C$ controls the inverse regularization strength."""

lr_tuning = """from sklearn.linear_model import LogisticRegression

# Regularization strength sweep
c_values = [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
cv_scores = []

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

print("Tuning Inverse Regularization Strength (C)...")
for c in c_values:
    model = LogisticRegression(C=c, max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED)
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
    cv_scores.append((c, scores['test_score'].mean(), scores['test_score'].std()))
    print(f"  C={c:<6} -> CV Macro F1: {scores['test_score'].mean():.3f} (±{scores['test_score'].std():.3f})")

# Plot C-tuning curve
df_c = pd.DataFrame(cv_scores, columns=['C', 'Mean_F1', 'Std_F1'])
best_c = df_c.loc[df_c['Mean_F1'].idxmax(), 'C']
print(f"\\nOptimal C selected: {best_c}")

plt.figure(figsize=(7, 4))
plt.errorbar(df_c['C'], df_c['Mean_F1'], yerr=df_c['Std_F1'], fmt='-o', color='#2b5c8f', capsize=5)
plt.xscale('log')
plt.title("Logistic Regression: Cross-Validation F1 vs Regularization (C)", fontsize=11, fontweight='bold')
plt.xlabel("Regularization Strength C (log scale)")
plt.ylabel("5-Fold CV Macro F1")
plt.grid(True, alpha=0.3)
plt.savefig("plots/hyperparameter_tuning.png", dpi=300, bbox_inches="tight")
plt.show()"""

lr_init = """best_name = f"Logistic Regression (C={best_c})"
best_model = LogisticRegression(C=best_c, max_iter=1500, class_weight='balanced', random_state=RANDOM_SEED)

t0 = time.time()
best_model.fit(X_train, y_train)
train_time = time.time() - t0

# Latency benchmark
t_start = time.perf_counter()
for _ in range(100):
    _ = best_model.predict(X_test[:1])
latency_ms = ((time.perf_counter() - t_start) / 100) * 1000.0

# Evaluate
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
auc_val = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

print(f"=== TEST SET BENCHMARK ===")
print(f"Accuracy        : {acc:.2%}")
print(f"Macro Precision : {prec:.2%}")
print(f"Macro Recall    : {rec:.2%}")
print(f"Macro F1-Score  : {f1:.2%}")
print(f"Macro ROC-AUC   : {auc_val:.3f}")
print(f"Inference Time  : {latency_ms:.3f} ms / segment")"""

lr_interp = """# Inspect Top Learned Feature Weights for Vessel vs Ambient
plt.figure(figsize=(10, 5))
vessel_idx = classes.index("vessel_like") if "vessel_like" in classes else 0
weights = pd.Series(best_model.coef_[vessel_idx], index=feature_cols).sort_values()
top_weights = pd.concat([weights.head(8), weights.tail(8)])

top_weights.plot(kind='barh', color=(top_weights > 0).map({True: '#2b5c8f', False: '#e76f51'}))
plt.title(f"Logistic Regression Feature Weights for Class: {classes[vessel_idx]}", fontsize=11, fontweight='bold')
plt.xlabel("Learned Coefficient Value")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/feature_coefficients.png", dpi=300, bbox_inches="tight")
plt.show()"""

with open("01_logistic_regression/logistic_regression.ipynb", "w", encoding="utf-8") as f:
    json.dump(build_algo_notebook("Logistic Regression", "01_logistic_regression", lr_theory, lr_init, lr_tuning, lr_interp), f, indent=2)
print("Created 01_logistic_regression/logistic_regression.ipynb")

# ==============================================================================
# 02. K-NEAREST NEIGHBORS (KNN)
# ==============================================================================
knn_theory = """**K-Nearest Neighbors (KNN)** is an instance-based non-parametric baseline classifier.
It assigns class membership based on majority vote among the $k$ closest neighbors in Euclidean feature space:
$$d(\\mathbf{x}, \\mathbf{z}) = \\sqrt{\\sum_{j=1}^D (x_j - z_j)^2}$$

To handle class imbalance and continuous acoustic boundaries, we utilize **distance-weighted voting** ($w_i = 1 / d_i$)."""

knn_tuning = """from sklearn.neighbors import KNeighborsClassifier

k_values = [1, 3, 5, 7, 9, 11, 15, 21, 25]
cv_scores = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

print("Tuning Number of Neighbors (k)...")
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='minkowski', p=2)
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
    cv_scores.append((k, scores['test_score'].mean(), scores['test_score'].std()))
    print(f"  k={k:<3} -> CV Macro F1: {scores['test_score'].mean():.3f} (±{scores['test_score'].std():.3f})")

df_k = pd.DataFrame(cv_scores, columns=['k', 'Mean_F1', 'Std_F1'])
best_k = int(df_k.loc[df_k['Mean_F1'].idxmax(), 'k'])
print(f"\\nOptimal k selected: {best_k}")

plt.figure(figsize=(7, 4))
plt.plot(df_k['k'], df_k['Mean_F1'], '-o', color='#2a9d8f', lw=2)
plt.fill_between(df_k['k'], df_k['Mean_F1'] - df_k['Std_F1'], df_k['Mean_F1'] + df_k['Std_F1'], alpha=0.15, color='#2a9d8f')
plt.title("KNN: 5-Fold CV Macro F1 vs Number of Neighbors (k)", fontsize=11, fontweight='bold')
plt.xlabel("Neighbors (k)")
plt.ylabel("5-Fold CV Macro F1")
plt.grid(True, alpha=0.3)
plt.savefig("plots/k_tuning_curve.png", dpi=300, bbox_inches="tight")
plt.show()"""

knn_init = """best_name = f"KNN (k={best_k})"
best_model = KNeighborsClassifier(n_neighbors=best_k, weights='distance', metric='minkowski', p=2)

t0 = time.time()
best_model.fit(X_train, y_train)
train_time = time.time() - t0

# Latency benchmark
t_start = time.perf_counter()
for _ in range(100):
    _ = best_model.predict(X_test[:1])
latency_ms = ((time.perf_counter() - t_start) / 100) * 1000.0

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
auc_val = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

print(f"=== TEST SET BENCHMARK ===")
print(f"Accuracy        : {acc:.2%}")
print(f"Macro Precision : {prec:.2%}")
print(f"Macro Recall    : {rec:.2%}")
print(f"Macro F1-Score  : {f1:.2%}")
print(f"Macro ROC-AUC   : {auc_val:.3f}")
print(f"Inference Time  : {latency_ms:.3f} ms / segment")"""

knn_interp = """# Permutation Feature Importance for KNN
from sklearn.inspection import permutation_importance
print("Computing Permutation Importance for KNN on Test Set...")
perm = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=RANDOM_SEED, n_jobs=-1)
perm_series = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 5))
perm_series.plot(kind='barh', color='#2a9d8f')
plt.title("KNN: Top 15 Features by Permutation Importance", fontsize=11, fontweight='bold')
plt.xlabel("Drop in Accuracy when Feature Shuffled")
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/permutation_importance.png", dpi=300, bbox_inches="tight")
plt.show()"""

with open("02_k_nearest_neighbors/knn.ipynb", "w", encoding="utf-8") as f:
    json.dump(build_algo_notebook("K-Nearest Neighbors", "02_k_nearest_neighbors", knn_theory, knn_init, knn_tuning, knn_interp), f, indent=2)
print("Created 02_k_nearest_neighbors/knn.ipynb")

# ==============================================================================
# 03. SUPPORT VECTOR MACHINE (SVM)
# ==============================================================================
svm_theory = """**Support Vector Machines (SVM)** maximize the margin between separating hyperplanes.
Using the **Radial Basis Function (RBF) Kernel**, SVM projects acoustic feature vectors into infinite-dimensional Hilbert space:
$$K(\\mathbf{x}, \\mathbf{z}) = \\exp\\left(-\\gamma \\|\\mathbf{x} - \\mathbf{z}\\|^2\\right)$$

The optimization problem minimizes margin violations controlled by penalty parameter $C$:
$$\\min_{\\mathbf{w}, b, \\xi} \\frac{1}{2} \\|\\mathbf{w}\\|^2 + C \\sum_{i=1}^N \\xi_i$$"""

svm_tuning = """from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.5, 1.0, 5.0, 10.0, 25.0],
    'gamma': ['scale', 'auto', 0.01, 0.05]
}

print("Running Grid Search on C and Gamma for RBF-SVM...")
grid = GridSearchCV(
    SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_SEED),
    param_grid, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED),
    scoring='f1_macro', n_jobs=-1
)
grid.fit(X_train, y_train)

best_c = grid.best_params_['C']
best_gamma = grid.best_params_['gamma']
print(f"Optimal parameters found: C={best_c}, gamma={best_gamma}")
print(f"Best CV Macro F1: {grid.best_score_:.3f}")"""

svm_init = """best_name = f"SVM (RBF, C={best_c}, gamma={best_gamma})"
best_model = SVC(kernel='rbf', C=best_c, gamma=best_gamma, probability=True, class_weight='balanced', random_state=RANDOM_SEED)

t0 = time.time()
best_model.fit(X_train, y_train)
train_time = time.time() - t0

# Latency benchmark
t_start = time.perf_counter()
for _ in range(100):
    _ = best_model.predict(X_test[:1])
latency_ms = ((time.perf_counter() - t_start) / 100) * 1000.0

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
auc_val = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

print(f"=== TEST SET BENCHMARK ===")
print(f"Accuracy        : {acc:.2%}")
print(f"Macro Precision : {prec:.2%}")
print(f"Macro Recall    : {rec:.2%}")
print(f"Macro F1-Score  : {f1:.2%}")
print(f"Macro ROC-AUC   : {auc_val:.3f}")
print(f"Inference Time  : {latency_ms:.3f} ms / segment")"""

svm_interp = """# Permutation Feature Importance for SVM
from sklearn.inspection import permutation_importance
print("Computing Permutation Importance for SVM on Test Set...")
perm = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=RANDOM_SEED, n_jobs=-1)
perm_series = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 5))
perm_series.plot(kind='barh', color='#e76f51')
plt.title("SVM: Top 15 Most Influential Features (Permutation Importance)", fontsize=11, fontweight='bold')
plt.xlabel("Decrease in Macro Accuracy")
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/svm_feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()"""

with open("03_support_vector_machine/svm.ipynb", "w", encoding="utf-8") as f:
    json.dump(build_algo_notebook("Support Vector Machine", "03_support_vector_machine", svm_theory, svm_init, svm_tuning, svm_interp), f, indent=2)
print("Created 03_support_vector_machine/svm.ipynb")

# ==============================================================================
# 04. RANDOM FOREST
# ==============================================================================
rf_theory = """**Random Forest** is an ensemble of de-correlated Decision Trees trained using **Bagging** and random feature sub-spacing.
For each split, a random subset of $m = \\sqrt{D}$ acoustic descriptors is evaluated.
Predictions are aggregated through soft voting across $B$ trees:
$$\\hat{P}(y = c \\mid \\mathbf{x}) = \\frac{1}{B} \\sum_{b=1}^B P_b(y = c \\mid \\mathbf{x})$$

Feature importance is directly computed via **Mean Decrease in Impurity (MDI / Gini Importance)**."""

rf_tuning = """from sklearn.ensemble import RandomForestClassifier

n_trees_list = [50, 100, 150, 200]
depth_list   = [6, 10, 14, None]

cv_results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

print("Tuning Random Forest Estimators & Tree Depth...")
for n_est in n_trees_list:
    for depth in depth_list:
        rf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, min_samples_split=4,
                                    class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)
        scores = cross_validate(rf, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
        cv_results.append({
            'n_estimators': n_est,
            'max_depth': str(depth),
            'f1_mean': scores['test_score'].mean()
        })

df_rf_tune = pd.DataFrame(cv_results)
best_row = df_rf_tune.loc[df_rf_tune['f1_mean'].idxmax()]
best_n_est = int(best_row['n_estimators'])
best_depth = None if best_row['max_depth'] == 'None' else int(best_row['max_depth'])
print(f"\\nOptimal RF parameters: n_estimators={best_n_est}, max_depth={best_depth}")
print(f"Top CV Macro F1: {best_row['f1_mean']:.3f}")"""

rf_init = """best_name = f"Random Forest (n={best_n_est}, depth={best_depth})"
best_model = RandomForestClassifier(
    n_estimators=best_n_est, max_depth=best_depth, min_samples_split=4,
    class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1
)

t0 = time.time()
best_model.fit(X_train, y_train)
train_time = time.time() - t0

# Latency benchmark
t_start = time.perf_counter()
for _ in range(100):
    _ = best_model.predict(X_test[:1])
latency_ms = ((time.perf_counter() - t_start) / 100) * 1000.0

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
auc_val = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

print(f"=== TEST SET BENCHMARK ===")
print(f"Accuracy        : {acc:.2%}")
print(f"Macro Precision : {prec:.2%}")
print(f"Macro Recall    : {rec:.2%}")
print(f"Macro F1-Score  : {f1:.2%}")
print(f"Macro ROC-AUC   : {auc_val:.3f}")
print(f"Inference Time  : {latency_ms:.3f} ms / segment")"""

rf_interp = """# Top 15 Gini Feature Importance (MDI)
importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 5))
importances.plot(kind='barh', color='#2b5c8f')
plt.title(f"Random Forest: Top 15 Most Informative Acoustic Descriptors (Gini Importance)", fontsize=11, fontweight='bold')
plt.xlabel("Mean Decrease in Impurity (Gini)")
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/rf_feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()"""

with open("04_random_forest/random_forest.ipynb", "w", encoding="utf-8") as f:
    json.dump(build_algo_notebook("Random Forest", "04_random_forest", rf_theory, rf_init, rf_tuning, rf_interp), f, indent=2)
print("Created 04_random_forest/random_forest.ipynb")

# ==============================================================================
# 05. XGBOOST
# ==============================================================================
xgb_theory = """**XGBoost (Extreme Gradient Boosting)** builds trees sequentially, fitting each new tree to the negative gradient of the multi-class log-loss:
$$\\mathcal{L}^{(t)} = \\sum_{i=1}^N \\left[ g_i f_t(\\mathbf{x}_i) + \\frac{1}{2} h_i f_t^2(\\mathbf{x}_i) \\right] + \\gamma T + \\frac{1}{2} \\lambda \\sum_{j=1}^T w_j^2$$
where $g_i$ and $h_i$ are first- and second-order Taylor expansion gradients.
Shrinkage (learning rate $\\eta$) scales each tree's contribution to prevent overfitting."""

xgb_tuning = """from xgboost import XGBClassifier

lr_list = [0.03, 0.08, 0.15]
depth_list = [4, 6, 8]

cv_results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

print("Tuning XGBoost Learning Rate & Tree Depth...")
for lr in lr_list:
    for d in depth_list:
        xgb = XGBClassifier(
            n_estimators=120, max_depth=d, learning_rate=lr,
            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
            random_state=RANDOM_SEED, n_jobs=-1
        )
        scores = cross_validate(xgb, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
        cv_results.append({
            'learning_rate': lr,
            'max_depth': d,
            'f1_mean': scores['test_score'].mean()
        })

df_xgb_tune = pd.DataFrame(cv_results)
best_row = df_xgb_tune.loc[df_xgb_tune['f1_mean'].idxmax()]
best_lr = float(best_row['learning_rate'])
best_depth = int(best_row['max_depth'])
print(f"\\nOptimal XGBoost parameters: learning_rate={best_lr}, max_depth={best_depth}")
print(f"Top CV Macro F1: {best_row['f1_mean']:.3f}")"""

xgb_init = """best_name = f"XGBoost (lr={best_lr}, depth={best_depth})"
best_model = XGBClassifier(
    n_estimators=150, max_depth=best_depth, learning_rate=best_lr,
    subsample=0.85, colsample_bytree=0.85, eval_metric="mlogloss",
    random_state=RANDOM_SEED, n_jobs=-1
)

t0 = time.time()
best_model.fit(X_train, y_train)
train_time = time.time() - t0

# Latency benchmark
t_start = time.perf_counter()
for _ in range(100):
    _ = best_model.predict(X_test[:1])
latency_ms = ((time.perf_counter() - t_start) / 100) * 1000.0

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
auc_val = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

print(f"=== TEST SET BENCHMARK ===")
print(f"Accuracy        : {acc:.2%}")
print(f"Macro Precision : {prec:.2%}")
print(f"Macro Recall    : {rec:.2%}")
print(f"Macro F1-Score  : {f1:.2%}")
print(f"Macro ROC-AUC   : {auc_val:.3f}")
print(f"Inference Time  : {latency_ms:.3f} ms / segment")"""

xgb_interp = """# Top 15 XGBoost Gain Feature Importance
gain_imp = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 5))
gain_imp.plot(kind='barh', color='#e76f51')
plt.title("XGBoost: Top 15 Features by Information Gain", fontsize=11, fontweight='bold')
plt.xlabel("Relative Gain Importance")
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/xgboost_gain_importance.png", dpi=300, bbox_inches="tight")
plt.show()"""

with open("05_xgboost/xgboost.ipynb", "w", encoding="utf-8") as f:
    json.dump(build_algo_notebook("XGBoost", "05_xgboost", xgb_theory, xgb_init, xgb_tuning, xgb_interp), f, indent=2)
print("Created 05_xgboost/xgboost.ipynb")
