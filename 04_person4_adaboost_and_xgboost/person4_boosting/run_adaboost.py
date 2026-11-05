"""
Run AdaBoost training directly — optimized for speed.
Uses a compact 3x3 grid (9 combos x 5 folds = 45 fits).
"""
import os, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='deep', font_scale=1.1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')
MODEL_DIR  = os.path.join(SCRIPT_DIR, 'models')
PLOT_DIR   = os.path.join(SCRIPT_DIR, 'plots')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

RANDOM_STATE = 42
CLASS_NAMES  = ['vessel_like', 'marine_life', 'ambient', 'unknown']

# ── 1. Load data ─────────────────────────────────────────────────────────────
print("[1/9] Loading data splits...")
df_train   = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
df_test    = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
df_holdout = pd.read_csv(os.path.join(DATA_DIR, 'unseen_holdout.csv'))

with open(os.path.join(DATA_DIR, 'selected_features.json')) as f:
    FEATURE_NAMES = json.load(f)['selected_features']

print(f"  Train: {len(df_train)} | Test: {len(df_test)} | Holdout: {len(df_holdout)}")
print(f"  Features: {len(FEATURE_NAMES)}")

# ── 2. Feature matrices ───────────────────────────────────────────────────────
print("[2/9] Preparing feature matrices...")
X_train   = df_train[FEATURE_NAMES].values
X_test    = df_test[FEATURE_NAMES].values
X_holdout = df_holdout[FEATURE_NAMES].values

le = LabelEncoder()
le.fit(CLASS_NAMES)
y_train   = le.transform(df_train['label'])
y_test    = le.transform(df_test['label'])
y_holdout = le.transform(df_holdout['label'])

scaler = StandardScaler()
X_train_scaled   = scaler.fit_transform(X_train)
X_test_scaled    = scaler.transform(X_test)
X_holdout_scaled = scaler.transform(X_holdout)
print(f"  Shapes — train {X_train_scaled.shape}, test {X_test_scaled.shape}, holdout {X_holdout_scaled.shape}")

# ── 3. Baseline AdaBoost ──────────────────────────────────────────────────────
print("[3/9] Training baseline AdaBoost (50 stumps, default params)...")
# Note: 'algorithm' param removed (deprecated in sklearn 1.6+, uses best available)
ada_baseline = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE),
    n_estimators=50, learning_rate=1.0, random_state=RANDOM_STATE
)
t0 = time.time()
ada_baseline.fit(X_train_scaled, y_train)
t_base = time.time() - t0
y_pred_base = ada_baseline.predict(X_test_scaled)
print(f"  Baseline test acc: {accuracy_score(y_test, y_pred_base):.4f}  [{t_base:.1f}s]")

# ── 4. Compact GridSearchCV ────────────────────────────────────────────────────
print("[4/9] GridSearchCV (9 combos x 5 folds = 45 fits)...")
# Focused grid around best-practice ranges
param_grid = {
    'n_estimators':  [50, 100, 200],
    'learning_rate': [0.1, 0.5, 1.0],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid_search = GridSearchCV(
    AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE
    ),
    param_grid=param_grid, cv=cv, scoring='f1_macro',
    n_jobs=-1, verbose=1, return_train_score=True
)
t0 = time.time()
grid_search.fit(X_train_scaled, y_train)
t_gs = time.time() - t0
best_params = grid_search.best_params_
print(f"  Best params: {best_params}")
print(f"  Best CV F1:  {grid_search.best_score_:.4f}  [{t_gs:.1f}s]")

# Heatmap plot
print("[5/9] Plotting hyperparameter heatmap...")
results_df = pd.DataFrame(grid_search.cv_results_)
pivot = results_df.pivot_table(
    index='param_learning_rate', columns='param_n_estimators', values='mean_test_score'
)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', linewidths=0.5, ax=axes[0],
            cbar_kws={'label': 'CV F1 Macro'})
axes[0].set_title('AdaBoost: F1 Macro vs Hyperparameters', fontweight='bold')
axes[0].set_xlabel('n_estimators'); axes[0].set_ylabel('learning_rate')
for lr in param_grid['learning_rate']:
    mask = results_df['param_learning_rate'] == lr
    subset = results_df[mask].sort_values('param_n_estimators')
    axes[1].plot(subset['param_n_estimators'], subset['mean_test_score'],
                 marker='o', linewidth=2, label=f'lr={lr}')
axes[1].set_xlabel('n_estimators'); axes[1].set_ylabel('CV F1 Macro')
axes[1].set_title('Boosting Rounds vs Performance', fontweight='bold')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'adaboost_hyperparam_search.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: adaboost_hyperparam_search.png")

# ── 6. Final model ─────────────────────────────────────────────────────────────
print("[6/9] Training final model...")
ada_final = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE),
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    random_state=RANDOM_STATE
)
t0 = time.time()
ada_final.fit(X_train_scaled, y_train)
train_time_final = time.time() - t0

y_pred_train = ada_final.predict(X_train_scaled)
y_pred_test  = ada_final.predict(X_test_scaled)

x_single = X_test_scaled[0:1]
latencies = []
for _ in range(500):
    t0 = time.perf_counter()
    ada_final.predict(x_single)
    latencies.append((time.perf_counter() - t0) * 1000)
avg_latency = float(np.mean(latencies))

test_accuracy  = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test, average='macro')
test_recall    = recall_score(y_test, y_pred_test, average='macro')
test_f1        = f1_score(y_test, y_pred_test, average='macro')

print(f"  Train acc: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"  Test  acc: {test_accuracy:.4f} | F1: {test_f1:.4f}")
print(f"  Latency:   {avg_latency:.3f} ms  [{train_time_final:.1f}s train]")
print(f"\n{classification_report(y_test, y_pred_test, target_names=CLASS_NAMES, digits=4)}")

# ── 7. Evaluation plots ────────────────────────────────────────────────────────
print("[7/9] Generating evaluation plots...")
# Confusion matrix — test
cm = confusion_matrix(y_test, y_pred_test)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[0].set_title('AdaBoost Confusion Matrix (Counts)', fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='YlOrRd', ax=axes[1],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[1].set_title('AdaBoost Confusion Matrix (Normalized)', fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'adaboost_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()

# ROC curves
y_test_bin = label_binarize(y_test, classes=list(range(len(CLASS_NAMES))))
y_score    = ada_final.decision_function(X_test_scaled)
fig, ax = plt.subplots(figsize=(8, 6))
for i, (cls, col) in enumerate(zip(CLASS_NAMES, ['#e74c3c','#2ecc71','#3498db','#f39c12'])):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    ax.plot(fpr, tpr, color=col, linewidth=2.5, label=f'{cls} (AUC={auc(fpr,tpr):.3f})')
ax.plot([0,1],[0,1],'k--',linewidth=1,alpha=0.5); ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('AdaBoost: ROC Curves', fontweight='bold')
ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'adaboost_roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

# Feature importance
importances = ada_final.feature_importances_
feat_df = pd.DataFrame({'Feature': FEATURE_NAMES, 'Importance': importances}).sort_values('Importance')
top20 = feat_df.tail(20)
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(top20)), top20['Importance'],
        color=plt.cm.viridis(np.linspace(0.2, 0.9, len(top20))))
ax.set_yticks(range(len(top20))); ax.set_yticklabels(top20['Feature'], fontsize=10)
ax.set_title('AdaBoost: Top 20 Feature Importances', fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'adaboost_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()

# Convergence curve
train_errors, test_errors = [], []
for y_st, y_ss in zip(ada_final.staged_predict(X_train_scaled), ada_final.staged_predict(X_test_scaled)):
    train_errors.append(1 - accuracy_score(y_train, y_st))
    test_errors.append(1 - accuracy_score(y_test, y_ss))
rounds = range(1, len(train_errors) + 1)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(rounds, train_errors, label='Train Error', color='#2ecc71', linewidth=2)
ax.plot(rounds, test_errors,  label='Test Error',  color='#3498db', linewidth=2, linestyle='--')
ax.set_xlabel('Boosting Round'); ax.set_ylabel('Error Rate')
ax.set_title('AdaBoost: Error Convergence', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'adaboost_convergence.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 4 evaluation plots.")

# ── 8. Unseen holdout ─────────────────────────────────────────────────────────
print("[8/9] Evaluating unseen holdout set...")
y_pred_holdout = ada_final.predict(X_holdout_scaled)
holdout_accuracy  = accuracy_score(y_holdout, y_pred_holdout)
holdout_precision = precision_score(y_holdout, y_pred_holdout, average='macro')
holdout_recall    = recall_score(y_holdout, y_pred_holdout, average='macro')
holdout_f1        = f1_score(y_holdout, y_pred_holdout, average='macro')
gap = abs(test_accuracy - holdout_accuracy)
print(f"  Holdout acc: {holdout_accuracy:.4f} | F1: {holdout_f1:.4f}")
print(f"  Generalization gap: {gap:.4f}  ({'LOW' if gap<0.05 else 'MODERATE' if gap<0.10 else 'HIGH'})")
print(f"\n{classification_report(y_holdout, y_pred_holdout, target_names=CLASS_NAMES, digits=4)}")

cm_h = confusion_matrix(y_holdout, y_pred_holdout)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm_h, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[0].set_title('AdaBoost Holdout Confusion Matrix', fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
cm_h_norm = cm_h.astype('float') / cm_h.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_h_norm, annot=True, fmt='.2%', cmap='YlOrRd', ax=axes[1],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[1].set_title('AdaBoost Holdout Confusion Matrix (Normalized)', fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'adaboost_holdout_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── 9. Save model & metrics ───────────────────────────────────────────────────
print("[9/9] Saving model and metrics...")
model_path = os.path.join(MODEL_DIR, 'adaboost_model.joblib')
joblib.dump(ada_final, model_path)
model_size_kb = os.path.getsize(model_path) / 1024
joblib.dump(scaler, os.path.join(MODEL_DIR, 'adaboost_scaler.joblib'))
joblib.dump(le,     os.path.join(MODEL_DIR, 'adaboost_label_encoder.joblib'))

metrics_report = {
    'model': 'AdaBoost',
    'base_estimator': 'DecisionTreeClassifier(max_depth=1)',
    'best_hyperparameters': best_params,
    'cv_best_f1_macro': float(grid_search.best_score_),
    'test_metrics': {
        'accuracy':        float(test_accuracy),
        'precision_macro': float(test_precision),
        'recall_macro':    float(test_recall),
        'f1_macro':        float(test_f1),
    },
    'holdout_metrics': {
        'accuracy':        float(holdout_accuracy),
        'precision_macro': float(holdout_precision),
        'recall_macro':    float(holdout_recall),
        'f1_macro':        float(holdout_f1),
    },
    'generalization_gap': float(gap),
    'training_time_seconds': float(train_time_final),
    'inference_latency_ms':  float(avg_latency),
    'model_size_kb':         float(model_size_kb),
    'n_features':            len(FEATURE_NAMES),
    'n_train_samples':       int(len(y_train)),
    'n_test_samples':        int(len(y_test)),
    'n_holdout_samples':     int(len(y_holdout)),
    'class_names':           CLASS_NAMES,
}
with open(os.path.join(MODEL_DIR, 'adaboost_metrics.json'), 'w') as f:
    json.dump(metrics_report, f, indent=2)

print(f"\n{'='*60}")
print("ADABOOST COMPLETE")
print(f"{'='*60}")
print(f"  Best n_estimators:  {best_params['n_estimators']}")
print(f"  Best learning_rate: {best_params['learning_rate']}")
print(f"  CV F1 (macro):      {grid_search.best_score_:.4f}")
print(f"  Test Accuracy:      {test_accuracy:.4f}")
print(f"  Test F1 (macro):    {test_f1:.4f}")
print(f"  Holdout Accuracy:   {holdout_accuracy:.4f}")
print(f"  Holdout F1 (macro): {holdout_f1:.4f}")
print(f"  Generalization Gap: {gap:.4f}")
print(f"  Inference Latency:  {avg_latency:.3f} ms")
print(f"  Model Size:         {model_size_kb:.1f} KB")
print(f"  Edge Deployable:    {'YES' if model_size_kb < 5000 and avg_latency < 100 else 'REVIEW'}")
print(f"{'='*60}")