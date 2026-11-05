"""
Run XGBoost training directly (faster than nbconvert).
Saves model, metrics and plots to person4_boosting directories.
"""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb

from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, log_loss
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

print(f"XGBoost version: {xgb.__version__}")

# ── 1. Load data ────────────────────────────────────────────────────────────
print("[1/10] Loading data splits...")
df_train   = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
df_test    = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
df_holdout = pd.read_csv(os.path.join(DATA_DIR, 'unseen_holdout.csv'))

with open(os.path.join(DATA_DIR, 'selected_features.json')) as f:
    FEATURE_NAMES = json.load(f)['selected_features']

print(f"  Train: {len(df_train)} | Test: {len(df_test)} | Holdout: {len(df_holdout)}")
print(f"  Features: {len(FEATURE_NAMES)}")

# ── 2. Prepare matrices ─────────────────────────────────────────────────────
print("[2/10] Preparing feature matrices...")
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

# DMatrices
dtrain   = xgb.DMatrix(X_train_scaled,   label=y_train,   feature_names=FEATURE_NAMES)
dtest    = xgb.DMatrix(X_test_scaled,    label=y_test,    feature_names=FEATURE_NAMES)
dholdout = xgb.DMatrix(X_holdout_scaled, label=y_holdout, feature_names=FEATURE_NAMES)
print(f"  DMatrix: train {dtrain.num_row()}x{dtrain.num_col()}, test {dtest.num_row()}x{dtest.num_col()}")

# ── 3. Baseline model ───────────────────────────────────────────────────────
print("[3/10] Training baseline XGBoost...")
params_base = {
    'objective': 'multi:softprob', 'num_class': len(CLASS_NAMES),
    'eval_metric': 'mlogloss', 'eta': 0.3, 'max_depth': 6,
    'subsample': 1.0, 'colsample_bytree': 1.0,
    'lambda': 1.0, 'alpha': 0.0, 'seed': RANDOM_STATE, 'verbosity': 0,
}
evals_base = [(dtrain, 'train'), (dtest, 'test')]
t0 = time.time()
bst_base = xgb.train(params_base, dtrain, num_boost_round=300,
                     evals=evals_base, early_stopping_rounds=30, verbose_eval=50)
t_base = time.time() - t0
y_pred_base = np.argmax(bst_base.predict(dtest), axis=1)
print(f"  Baseline test acc: {accuracy_score(y_test, y_pred_base):.4f}  [{t_base:.1f}s]")

# Loss convergence plot
print("[4/10] Plotting loss convergence...")
evals_result_track = {}
bst_track = xgb.train(params_base, dtrain, num_boost_round=300,
                       evals=evals_base, evals_result=evals_result_track,
                       early_stopping_rounds=30, verbose_eval=False)
fig, ax = plt.subplots(figsize=(10, 5))
epochs = range(1, len(evals_result_track['train']['mlogloss']) + 1)
ax.plot(epochs, evals_result_track['train']['mlogloss'], label='Train Loss', color='#2ecc71', linewidth=2)
ax.plot(epochs, evals_result_track['test']['mlogloss'],  label='Test Loss',  color='#3498db', linewidth=2)
ax.axvline(x=bst_track.best_iteration, color='#e74c3c', linestyle='--', linewidth=1.5,
           label=f'Best iter ({bst_track.best_iteration})')
ax.set_xlabel('Boosting Round'); ax.set_ylabel('Multi-Class Log Loss')
ax.set_title('XGBoost: Loss Convergence', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'xgboost_loss_convergence.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── 5. GridSearchCV ─────────────────────────────────────────────────────────
print("[5/10] GridSearchCV (108 combos x 5 folds = 540 fits)...")
param_grid = {
    'n_estimators':     [100, 200, 300],
    'max_depth':        [3, 5, 7],
    'learning_rate':    [0.01, 0.1, 0.3],
    'subsample':        [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}
xgb_grid = xgb.XGBClassifier(
    objective='multi:softprob', num_class=len(CLASS_NAMES), eval_metric='mlogloss',
    reg_lambda=1.0, reg_alpha=0.0, random_state=RANDOM_STATE,
    use_label_encoder=False, verbosity=0,
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid_search = GridSearchCV(xgb_grid, param_grid=param_grid, cv=cv,
                           scoring='f1_macro', n_jobs=-1, verbose=1, return_train_score=True)
t0 = time.time()
grid_search.fit(X_train_scaled, y_train)
t_gs = time.time() - t0
best_params = grid_search.best_params_
print(f"  Best params: {best_params}")
print(f"  Best CV F1:  {grid_search.best_score_:.4f}  [{t_gs:.1f}s]")

# Grid search viz
print("[6/10] Plotting grid search results...")
results_df = pd.DataFrame(grid_search.cv_results_)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
pivot1 = results_df.groupby(['param_max_depth','param_learning_rate'])['mean_test_score'].max().unstack()
sns.heatmap(pivot1, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0],
            linewidths=0.5, cbar_kws={'label': 'Best F1 Macro'})
axes[0].set_title('XGBoost: Max Depth vs Learning Rate', fontweight='bold')
for lr in [0.01, 0.1, 0.3]:
    subset = results_df[results_df['param_learning_rate']==lr].groupby('param_n_estimators')['mean_test_score'].max()
    axes[1].plot(subset.index, subset.values, marker='o', label=f'lr={lr}', linewidth=2)
axes[1].set_xlabel('n_estimators'); axes[1].set_ylabel('Best F1'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].set_title('CV Performance vs n_estimators', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'xgboost_gridsearch.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── 7. Final model ─────────────────────────────────────────────────────────
print("[7/10] Training final optimized XGBoost...")
final_params = {
    'objective': 'multi:softprob', 'num_class': len(CLASS_NAMES), 'eval_metric': 'mlogloss',
    'eta': best_params['learning_rate'], 'max_depth': best_params['max_depth'],
    'subsample': best_params['subsample'], 'colsample_bytree': best_params['colsample_bytree'],
    'lambda': 1.0, 'alpha': 0.0, 'seed': RANDOM_STATE, 'verbosity': 0,
}
evals_result_final = {}
t0 = time.time()
xgb_final = xgb.train(
    final_params, dtrain,
    num_boost_round=best_params['n_estimators'],
    evals=[(dtrain, 'train'), (dtest, 'test')],
    evals_result=evals_result_final,
    early_stopping_rounds=30, verbose_eval=50
)
train_time_final = time.time() - t0

y_prob_train = xgb_final.predict(dtrain); y_pred_train = np.argmax(y_prob_train, axis=1)
y_prob_test  = xgb_final.predict(dtest);  y_pred_test  = np.argmax(y_prob_test,  axis=1)

# Latency
x_single_dm = xgb.DMatrix(X_test_scaled[0:1], feature_names=FEATURE_NAMES)
latencies = []
for _ in range(1000):
    t0 = time.perf_counter()
    xgb_final.predict(x_single_dm)
    latencies.append((time.perf_counter() - t0) * 1000)
avg_latency = float(np.mean(latencies))

test_accuracy  = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test, average='macro')
test_recall    = recall_score(y_test, y_pred_test, average='macro')
test_f1        = f1_score(y_test, y_pred_test, average='macro')
test_logloss   = log_loss(y_test, y_prob_test)

print(f"  Train acc: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"  Test  acc: {test_accuracy:.4f} | F1: {test_f1:.4f} | LogLoss: {test_logloss:.4f}")
print(f"  Latency:   {avg_latency:.3f} ms  [{train_time_final:.1f}s train]")
print(f"\n  Test Classification Report:\n{classification_report(y_test, y_pred_test, target_names=CLASS_NAMES, digits=4)}")

# ── 8. Evaluation plots ─────────────────────────────────────────────────────
print("[8/10] Generating evaluation plots...")
# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[0].set_title('XGBoost Confusion Matrix (Counts)', fontweight='bold'); axes[0].tick_params(axis='x', rotation=45)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='YlOrRd', ax=axes[1], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[1].set_title('XGBoost Confusion Matrix (Normalized)', fontweight='bold'); axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'xgboost_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()

# ROC
y_test_bin = label_binarize(y_test, classes=list(range(len(CLASS_NAMES))))
fig, ax = plt.subplots(figsize=(8, 6))
for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, ['#e74c3c','#2ecc71','#3498db','#f39c12'])):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob_test[:, i])
    ax.plot(fpr, tpr, color=color, linewidth=2.5, label=f'{cls_name} (AUC={auc(fpr,tpr):.3f})')
ax.plot([0,1],[0,1],'k--',linewidth=1,alpha=0.5,label='Random')
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('XGBoost: ROC Curves', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'xgboost_roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

# Feature importance
imp_gain   = xgb_final.get_score(importance_type='gain')
imp_weight = xgb_final.get_score(importance_type='weight')
imp_df = pd.DataFrame({'Feature': FEATURE_NAMES,
                        'Gain':   [imp_gain.get(f, 0)   for f in FEATURE_NAMES],
                        'Weight': [imp_weight.get(f, 0) for f in FEATURE_NAMES]}).sort_values('Gain')
top20 = imp_df.tail(20)
top20w = imp_df.sort_values('Weight').tail(20)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
bars = axes[0].barh(range(len(top20)), top20['Gain'], color=plt.cm.plasma(np.linspace(0.2, 0.9, len(top20))))
axes[0].set_yticks(range(len(top20))); axes[0].set_yticklabels(top20['Feature'], fontsize=10)
axes[0].set_title('XGBoost: Top 20 by Gain', fontweight='bold'); axes[0].grid(True, axis='x', alpha=0.3)
axes[1].barh(range(len(top20w)), top20w['Weight'], color=plt.cm.viridis(np.linspace(0.2, 0.9, len(top20w))))
axes[1].set_yticks(range(len(top20w))); axes[1].set_yticklabels(top20w['Feature'], fontsize=10)
axes[1].set_title('XGBoost: Top 20 by Frequency', fontweight='bold'); axes[1].grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'xgboost_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()

# Shrinkage analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ep_fin = range(1, len(evals_result_final['train']['mlogloss']) + 1)
axes[0].plot(ep_fin, evals_result_final['train']['mlogloss'], label='Train', color='#2ecc71', linewidth=2)
axes[0].plot(ep_fin, evals_result_final['test']['mlogloss'],  label='Test',  color='#3498db', linewidth=2)
axes[0].axvline(x=xgb_final.best_iteration, color='#e74c3c', linestyle='--',
                linewidth=1.5, label=f'Best ({xgb_final.best_iteration})')
axes[0].set_title('Final Model Loss Convergence', fontweight='bold'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
for eta in [0.01, 0.05, 0.1, 0.3]:
    ep = final_params.copy(); ep['eta'] = eta
    er = {}
    xgb.train(ep, dtrain, 200, [(dtest, 'test')], evals_result=er, verbose_eval=False)
    axes[1].plot(range(1, 201), er['test']['mlogloss'], label=f'eta={eta}', linewidth=2)
axes[1].set_xlabel('Boosting Round'); axes[1].set_ylabel('Test Log Loss')
axes[1].set_title('Shrinkage Effect', fontweight='bold'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'xgboost_shrinkage_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 5 evaluation plots.")

# ── 9. Holdout evaluation ─────────────────────────────────────────────────
print("[9/10] Evaluating on unseen holdout...")
y_prob_holdout = xgb_final.predict(dholdout)
y_pred_holdout = np.argmax(y_prob_holdout, axis=1)
holdout_accuracy  = accuracy_score(y_holdout, y_pred_holdout)
holdout_precision = precision_score(y_holdout, y_pred_holdout, average='macro')
holdout_recall    = recall_score(y_holdout, y_pred_holdout, average='macro')
holdout_f1        = f1_score(y_holdout, y_pred_holdout, average='macro')
holdout_logloss   = log_loss(y_holdout, y_prob_holdout)
gap = abs(test_accuracy - holdout_accuracy)
print(f"  Holdout acc: {holdout_accuracy:.4f} | F1: {holdout_f1:.4f} | gap: {gap:.4f}")
print(f"\n  Holdout Classification Report:\n{classification_report(y_holdout, y_pred_holdout, target_names=CLASS_NAMES, digits=4)}")

cm_h = confusion_matrix(y_holdout, y_pred_holdout)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm_h, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[0].set_title('XGBoost Holdout Confusion Matrix', fontweight='bold'); axes[0].tick_params(axis='x', rotation=45)
cm_h_norm = cm_h.astype('float') / cm_h.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_h_norm, annot=True, fmt='.2%', cmap='YlOrRd', ax=axes[1], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[1].set_title('XGBoost Holdout Confusion Matrix (Normalized)', fontweight='bold'); axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'xgboost_holdout_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()

# Comparison with AdaBoost
ada_path = os.path.join(MODEL_DIR, 'adaboost_metrics.json')
if os.path.exists(ada_path):
    with open(ada_path) as f:
        ada_m = json.load(f)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    test_met = ['Accuracy','F1 Macro','Precision','Recall']
    ada_tv = [ada_m['test_metrics']['accuracy'], ada_m['test_metrics']['f1_macro'],
              ada_m['test_metrics']['precision_macro'], ada_m['test_metrics']['recall_macro']]
    xgb_tv = [test_accuracy, test_f1, test_precision, test_recall]
    x = np.arange(len(test_met)); w = 0.35
    axes[0].bar(x-w/2, ada_tv, w, label='AdaBoost', color='#e74c3c', alpha=0.85)
    axes[0].bar(x+w/2, xgb_tv, w, label='XGBoost',  color='#3498db', alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(test_met, rotation=15)
    axes[0].set_title('Test Set Performance', fontweight='bold'); axes[0].legend(); axes[0].set_ylim(0, 1.1)
    hm = ['Accuracy','F1 Macro']
    ada_hv = [ada_m['holdout_metrics']['accuracy'], ada_m['holdout_metrics']['f1_macro']]
    xgb_hv = [holdout_accuracy, holdout_f1]
    x2 = np.arange(len(hm))
    axes[1].bar(x2-w/2, ada_hv, w, label='AdaBoost', color='#c0392b', alpha=0.85)
    axes[1].bar(x2+w/2, xgb_hv, w, label='XGBoost',  color='#2980b9', alpha=0.85)
    axes[1].set_xticks(x2); axes[1].set_xticklabels(hm)
    axes[1].set_title('Holdout Performance', fontweight='bold'); axes[1].legend(); axes[1].set_ylim(0, 1.1)
    axes[2].scatter([ada_m['inference_latency_ms']], [ada_m['test_metrics']['accuracy']],
                    s=300, color='#e74c3c', zorder=5, label='AdaBoost')
    axes[2].scatter([avg_latency], [test_accuracy], s=300, color='#3498db', zorder=5, label='XGBoost')
    axes[2].set_xlabel('Latency (ms)'); axes[2].set_ylabel('Test Accuracy')
    axes[2].set_title('Accuracy vs Latency', fontweight='bold'); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'boosting_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved comparison plot.")

# ── 10. Save model & metrics ─────────────────────────────────────────────────
print("[10/10] Saving model, scaler, encoder, and metrics...")
native_path = os.path.join(MODEL_DIR, 'xgboost_model.json')
xgb_final.save_model(native_path)
model_native_size_kb = os.path.getsize(native_path) / 1024

xgb_sk = xgb.XGBClassifier(**best_params, objective='multi:softprob', num_class=len(CLASS_NAMES),
                             eval_metric='mlogloss', random_state=RANDOM_STATE,
                             use_label_encoder=False, verbosity=0)
xgb_sk.fit(X_train_scaled, y_train)
joblib_path = os.path.join(MODEL_DIR, 'xgboost_model.joblib')
joblib.dump(xgb_sk, joblib_path)
model_joblib_size_kb = os.path.getsize(joblib_path) / 1024

joblib.dump(scaler, os.path.join(MODEL_DIR, 'xgboost_scaler.joblib'))
joblib.dump(le,     os.path.join(MODEL_DIR, 'xgboost_label_encoder.joblib'))

metrics_report = {
    'model': 'XGBoost (Extreme Gradient Boosting)',
    'loss_function': 'multi:softprob (Softmax with 2nd-order Taylor expansion)',
    'best_hyperparameters': best_params,
    'cv_best_f1_macro': float(grid_search.best_score_),
    'test_metrics': {
        'accuracy':        float(test_accuracy),
        'precision_macro': float(test_precision),
        'recall_macro':    float(test_recall),
        'f1_macro':        float(test_f1),
        'log_loss':        float(test_logloss),
    },
    'holdout_metrics': {
        'accuracy':        float(holdout_accuracy),
        'precision_macro': float(holdout_precision),
        'recall_macro':    float(holdout_recall),
        'f1_macro':        float(holdout_f1),
        'log_loss':        float(holdout_logloss),
    },
    'generalization_gap': float(gap),
    'training_time_seconds': float(train_time_final),
    'inference_latency_ms':  float(avg_latency),
    'model_size_kb': {
        'native_json': float(model_native_size_kb),
        'joblib':      float(model_joblib_size_kb),
    },
    'best_iteration':   int(xgb_final.best_iteration),
    'n_features':       len(FEATURE_NAMES),
    'n_train_samples':  int(len(y_train)),
    'n_test_samples':   int(len(y_test)),
    'n_holdout_samples': int(len(y_holdout)),
    'class_names':      CLASS_NAMES,
}
metrics_path = os.path.join(MODEL_DIR, 'xgboost_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics_report, f, indent=2)

print(f"\n{'='*60}")
print("XGBOOST COMPLETE")
print(f"{'='*60}")
print(f"  Best learning_rate:    {best_params['learning_rate']}")
print(f"  Best max_depth:        {best_params['max_depth']}")
print(f"  Best n_estimators:     {best_params['n_estimators']}")
print(f"  Early Stopped at:      {xgb_final.best_iteration}")
print(f"  Test Accuracy:         {test_accuracy:.4f}")
print(f"  Test F1 (macro):       {test_f1:.4f}")
print(f"  Holdout Accuracy:      {holdout_accuracy:.4f}")
print(f"  Holdout F1 (macro):    {holdout_f1:.4f}")
print(f"  Generalization Gap:    {gap:.4f}")
print(f"  Inference Latency:     {avg_latency:.3f} ms")
print(f"  Model Size (JSON):     {model_native_size_kb:.1f} KB")
print(f"  Edge Deployable:       {'YES' if model_native_size_kb < 5000 and avg_latency < 100 else 'REVIEW'}")
print(f"  Metrics saved:         {metrics_path}")
print(f"{'='*60}")

