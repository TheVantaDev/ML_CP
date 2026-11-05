"""
Targeted fix for random_forest.ipynb:
  - Cell 9 got wrongly replaced with final_model code (should be 2D grid search)
  - Restore cell 9 to 2D grid search content
  - Replace cell 11 with the upgraded final_model (with test_metrics dict)
  - Fix cell 7 to use StratifiedGroupKFold (was left as StratifiedKFold)
"""
import json
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

RF_PATH = Path("03_person3_decision_tree_and_random_forest/random_forest.ipynb")

GRID_SEARCH_CELL = """depth_grid = [6, 10, 14, None]
n_grid = [50, 100, 150, 200]
heatmap_data = np.zeros((len(depth_grid), len(n_grid)))

print("Running 2D Grid Evaluation...")
for i, d in enumerate(depth_grid):
    for j, n in enumerate(n_grid):
        rf_grid = RandomForestClassifier(n_estimators=n, max_depth=d, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)
        res = cross_validate(rf_grid, X_train, y_train, groups=groups_train, cv=cv, scoring='f1_macro', n_jobs=-1)
        heatmap_data[i, j] = res['test_score'].mean()

df_heat = pd.DataFrame(
    heatmap_data,
    index=[str(d) for d in depth_grid],
    columns=[str(n) for n in n_grid]
)

plt.figure(figsize=(7, 5))
sns.heatmap(df_heat, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': '5-Fold CV Macro F1'})
plt.title("Random Forest: 2D Grid (n_estimators x max_depth) - Grouped CV Macro F1", fontsize=11, fontweight='bold')
plt.xlabel("n_estimators")
plt.ylabel("max_depth")
plt.tight_layout()
plt.savefig("plots/rf_2d_grid_search.png", dpi=200, bbox_inches="tight")
plt.show()

best_flat_idx = heatmap_data.argmax()
best_i, best_j = divmod(best_flat_idx, len(n_grid))
best_depth = depth_grid[best_i]
best_n_est = n_grid[best_j]
print(f"Best from 2D grid: n_estimators={best_n_est}, max_depth={best_depth} "
      f"(CV Macro F1 = {heatmap_data[best_i, best_j]:.3f})")
"""

FINAL_MODEL_CELL = """best_model = RandomForestClassifier(
    n_estimators=best_n_est,
    max_depth=best_depth,
    min_samples_split=4,
    class_weight='balanced',
    random_state=RANDOM_SEED,
    n_jobs=-1
)

t0 = time.time()
best_model.fit(X_train, y_train)
train_time = time.time() - t0

# Latency Benchmark
t_start = time.perf_counter()
n_iters = 100
for _ in range(n_iters):
    _ = best_model.predict(X_test[:1])
latency_ms = ((time.perf_counter() - t_start) / n_iters) * 1000.0

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
auc_val = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

# Collect into a dict (used by bootstrap CI and save cells below)
test_metrics = {
    "test_accuracy"  : float(acc),
    "macro_precision": float(prec),
    "macro_recall"   : float(rec),
    "macro_f1"       : float(f1),
    "roc_auc_macro"  : float(auc_val),
    "latency_ms"     : float(latency_ms),
    "training_time_s": float(train_time),
}

print("=" * 60)
print("  OFFICIAL TEST SET BENCHMARK - RANDOM FOREST")
print("=" * 60)
print(f"  Test Accuracy        : {acc:.2%}")
print(f"  Macro Precision      : {prec:.2%}")
print(f"  Macro Recall         : {rec:.2%}")
print(f"  Macro F1-Score       : {f1:.2%}")
print(f"  Multi-Class ROC-AUC  : {auc_val:.3f}")
print(f"  Inference Latency    : {latency_ms:.3f} ms / 4 s segment")
print(f"  Training Time        : {train_time:.3f} s")
print("=" * 60)
print()
print(classification_report(y_test, y_pred, target_names=classes, digits=3))
"""

# Fix cell 7 (n_trees sweep) — update it to use groups_train with the grouped CV
# (cell 7 still uses old StratifiedKFold without groups — fix it)
N_TREES_SWEEP_CELL = """n_trees_candidates = [10, 25, 50, 75, 100, 150, 200, 250]
oob_scores = []
cv_scores = []

print("Sweeping n_estimators with OOB error tracking...")
for n in n_trees_candidates:
    rf = RandomForestClassifier(
        n_estimators=n,
        max_depth=12,
        oob_score=True,
        bootstrap=True,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    oob_scores.append(rf.oob_score_)

    cv_res = cross_validate(rf, X_train_raw, y_train, groups=groups_train,
                            cv=cv, scoring='f1_macro', n_jobs=-1)
    cv_scores.append(cv_res['test_score'].mean())
    print(f"  n_estimators={n:<4} -> OOB Accuracy: {rf.oob_score_:.3f} | "
          f"5-Fold Grouped CV F1: {cv_res['test_score'].mean():.3f}")

best_n_from_oob = n_trees_candidates[int(np.argmax(cv_scores))]
print(f"\\nBest n_estimators by CV F1: {best_n_from_oob}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(n_trees_candidates, oob_scores, 'o-', color='#2a78d6', lw=2)
axes[0].set_title("OOB Accuracy vs n_estimators", fontweight="bold")
axes[0].set_xlabel("n_estimators"); axes[0].set_ylabel("OOB Accuracy")
axes[0].grid(True, alpha=0.3)

axes[1].plot(n_trees_candidates, cv_scores, 's-', color='#eb6834', lw=2)
axes[1].set_title("Grouped CV Macro F1 vs n_estimators", fontweight="bold")
axes[1].set_xlabel("n_estimators"); axes[1].set_ylabel("CV Macro F1")
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("plots/rf_oob_vs_ntrees.png", dpi=200, bbox_inches="tight")
plt.show()
"""

with open(RF_PATH, encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Fix cell 7: n_trees sweep → use groups_train
cells[7]["source"] = N_TREES_SWEEP_CELL

# Restore cell 9: 2D grid search (was overwritten by mistake)
cells[9]["source"] = GRID_SEARCH_CELL

# Replace cell 11: final model (with test_metrics dict)
cells[11]["source"] = FINAL_MODEL_CELL

# Clear all outputs so it runs fresh
for cell in cells:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

nb["cells"] = cells

with open(RF_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"[OK] Fixed random_forest.ipynb ({len(cells)} cells)")
print("     - Cell 7: n_trees sweep now uses groups_train (grouped CV)")
print("     - Cell 9: restored 2D grid search")
print("     - Cell 11: replaced with final_model + test_metrics dict")
