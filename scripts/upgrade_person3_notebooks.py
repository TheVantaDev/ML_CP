"""
Upgrades Person 3's decision_tree.ipynb and random_forest.ipynb
to match Person 2's rigour:

  1. Imports → add StratifiedGroupKFold, cross_val_predict, DummyClassifier, clone
  2. Split cell → extract groups_train, assert split column exists, assert no leakage
  3. CV cell → switch to StratifiedGroupKFold, add DummyClassifier baseline
  4. After final-model cell → insert:
        a) Bootstrap CI on test set (by clip)
        b) Confidence-threshold / decision-support analysis
        c) Per-source-dataset breakdown
  5. Save cell → add frozen result assertion + saved-model smoke-test

Run from ML_CP root:
    python scripts/upgrade_person3_notebooks.py
"""
import json
import copy
import sys
from pathlib import Path

# Fix Windows console encoding so print() works with any characters
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DT_PATH = Path("03_person3_decision_tree_and_random_forest/decision_tree.ipynb")
RF_PATH = Path("03_person3_decision_tree_and_random_forest/random_forest.ipynb")


def code_cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src
    }


def md_cell(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src
    }


# ──────────────────────────────────────────────────────────────────────────────
# Shared new-cell content (parameterised by slug / algo name)
# ──────────────────────────────────────────────────────────────────────────────

def imports_cell(algo: str, extra_imports: str) -> str:
    return f"""import os
import sys
import time
import json
from pathlib import Path

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

{extra_imports}

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (
    StratifiedGroupKFold, StratifiedKFold, cross_validate,
    cross_val_predict, train_test_split
)
from sklearn.dummy import DummyClassifier
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.inspection import permutation_importance

Path("plots").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Dataset info
TOTAL_ROWS  = 3579   # shape of features.parquet
TRAIN_ROWS  = 2756
VAL_ROWS    = 341
TEST_ROWS   = 482

print(f"Environment loaded for Person 3: {algo}.")
print(f"Dataset: {{TOTAL_ROWS}} total segments | Train={{TRAIN_ROWS}} Val={{VAL_ROWS}} Test={{TEST_ROWS}}")
"""


def split_cell(algo: str) -> str:
    return """le = LabelEncoder()
df['target'] = le.fit_transform(df['label'])
classes = list(le.classes_)
num_classes = len(classes)
print(f"Classes ({num_classes}): {dict(zip(range(num_classes), classes))}")

# Require the shared pre-assigned split column (same as Person 2)
assert "split" in df.columns and set(df["split"].unique()) >= {"train", "val", "test"}, \\
    "Split column missing or incomplete — run 00_data_prep_and_features first."

train_mask = df["split"] == "train"
val_mask   = df["split"] == "val"
test_mask  = df["split"] == "test"

X_train_raw = np.nan_to_num(df.loc[train_mask, feature_cols].values, nan=0.0)
y_train     = df.loc[train_mask, "target"].values
groups_train = df.loc[train_mask, "source_clip_id"].values   # for grouped CV (no clip leakage)

X_val_raw   = np.nan_to_num(df.loc[val_mask, feature_cols].values, nan=0.0)
y_val       = df.loc[val_mask, "target"].values

X_test_raw  = np.nan_to_num(df.loc[test_mask, feature_cols].values, nan=0.0)
y_test      = df.loc[test_mask, "target"].values
test_clips  = df.loc[test_mask, "source_clip_id"].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val   = scaler.transform(X_val_raw)
X_test  = scaler.transform(X_test_raw)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"Unique clips in train: {len(set(groups_train))}")

# Grouped CV — no audio clip crosses the fold boundary
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
print("\\nVerifying no clip leaks across CV folds:")
for fold, (tr, va) in enumerate(cv.split(X_train_raw, y_train, groups_train), 1):
    leaked = set(groups_train[tr]) & set(groups_train[va])
    assert not leaked, f"Fold {fold}: clip leaked! {leaked}"
    print(f"  Fold {fold}: train={len(tr)} val={len(va)} "
          f"class dist={np.bincount(y_train[va], minlength=num_classes)}")
print("No clip leaked across any fold. ✓")

# DummyClassifier baseline (most-frequent)
dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
dummy_cv = cross_validate(dummy, X_train_raw, y_train, groups=groups_train,
                          cv=cv, scoring="f1_macro")
print(f"\\nDummy (most-frequent) grouped CV macro F1: {dummy_cv['test_score'].mean():.3f} "
      f"(beats this = model is learning)")
dummy.fit(X_train, y_train)
dummy_val_f1 = f1_score(y_val, dummy.predict(X_val), average="macro")
print(f"Dummy validation macro F1: {dummy_val_f1:.3f}")
"""


def bootstrap_cell(slug: str, algo_name: str) -> str:
    return f"""# ── Bootstrap Confidence Intervals on Test Set (by clip) ──────────────────
LOW_CONFIDENCE_THRESHOLD = 0.60   # same value as src/utils/config.py
CONF_BINS = [0, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]


def bootstrap_clip_ci(y, pred, clips, n_boot=2000):
    \"\"\"95% bootstrap CI for macro F1 and accuracy, resampling whole clips.\"\"\"
    rng = np.random.default_rng(RANDOM_SEED)
    positions = list(pd.DataFrame({{"clip": clips}}).groupby("clip").indices.values())
    labels = list(range(num_classes))
    f1s, accs = [], []
    for _ in range(n_boot):
        picks = rng.integers(0, len(positions), len(positions))
        idx = np.concatenate([positions[p] for p in picks])
        f1s.append(f1_score(y[idx], pred[idx], average="macro",
                            labels=labels, zero_division=0))
        accs.append((y[idx] == pred[idx]).mean())
    return np.percentile(f1s, [2.5, 97.5]), np.percentile(accs, [2.5, 97.5])


f1_ci, acc_ci = bootstrap_clip_ci(y_test, y_pred, test_clips)
print(f"Test clips: {{len(set(test_clips))}} clips, {{len(y_test)}} segments")
print(f"Macro F1 : {{test_metrics['macro_f1']:.3f}}  "
      f"(95% CI {{f1_ci[0]:.3f}} – {{f1_ci[1]:.3f}})")
print(f"Accuracy : {{test_metrics['test_accuracy']:.3f}}  "
      f"(95% CI {{acc_ci[0]:.3f}} – {{acc_ci[1]:.3f}})")


# ── Confidence / Decision-Support Analysis ─────────────────────────────────
y_prob_test = best_model.predict_proba(X_test)
conf = y_prob_test.max(axis=1)
correct = (y_pred == y_test)
flagged = conf < LOW_CONFIDENCE_THRESHOLD

print(f"\\nConfidence < {{LOW_CONFIDENCE_THRESHOLD:.0%}} → flagged for human review:")
print(f"  Flagged  : {{flagged.sum()}} of {{len(y_test)}} ({{flagged.mean():.1%}})")
print(f"  Accuracy of kept segments   : {{correct[~flagged].mean():.3f}} "
      f"({{(~flagged).sum()}} segments)")
print(f"  Error rate of kept segments : {{1 - correct[~flagged].mean():.3f}}")

bands = pd.cut(conf, CONF_BINS, include_lowest=True)
conf_table = (
    pd.DataFrame({{"conf_band": bands, "correct": correct}})
    .groupby("conf_band", observed=True)
    .agg(segments=("correct", "size"), accuracy=("correct", "mean"))
)
print("\\nAccuracy by confidence band:")
print(conf_table.round(3).to_string())

# Plot confidence histogram
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(conf[correct], bins=20, alpha=0.7, color="#2a78d6", label="Correct")
axes[0].hist(conf[~correct], bins=20, alpha=0.7, color="#eb6834", label="Wrong")
axes[0].axvline(LOW_CONFIDENCE_THRESHOLD, color="red", linestyle="--",
                label=f"Threshold {{LOW_CONFIDENCE_THRESHOLD}}")
axes[0].set_xlabel("Max Class Probability (Confidence)")
axes[0].set_ylabel("Segment Count")
axes[0].set_title(f"{{'{algo_name}'}} — Confidence Distribution", fontweight="bold")
axes[0].legend()
conf_table["accuracy"].plot(kind="bar", ax=axes[1], color="#2b5c8f", edgecolor="white")
axes[1].set_title("Accuracy by Confidence Band", fontweight="bold")
axes[1].set_xlabel("Confidence Band")
axes[1].set_ylabel("Accuracy")
axes[1].set_xticklabels([str(b) for b in conf_table.index], rotation=30, ha="right")
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("plots/confidence_analysis_{slug}.png", dpi=200, bbox_inches="tight")
plt.show()
print("Saved → plots/confidence_analysis_{slug}.png")
"""


def source_breakdown_cell(slug: str) -> str:
    return f"""# ── Per-Source-Dataset Breakdown ──────────────────────────────────────────
meta_path = Path("../data/metadata/master_metadata.csv")
if meta_path.exists():
    meta = pd.read_csv(meta_path)
    clip_source = dict(zip(meta["clip_id"], meta["source_dataset"]))
    source_col = [clip_source.get(c, "unknown_source") for c in test_clips]
else:
    # Fall back to source_clip_id prefix if metadata CSV is missing
    source_col = [c.split("_")[0] if isinstance(c, str) else "unknown" for c in test_clips]

results = pd.DataFrame({{
    "clip"      : test_clips,
    "source"    : source_col,
    "true"      : [classes[i] for i in y_test],
    "pred"      : [classes[i] for i in y_pred],
    "confidence": best_model.predict_proba(X_test).max(axis=1),
}})
results["correct"] = results["true"] == results["pred"]

print("Recall by true class and data source:")
print(results.groupby(["true", "source"]).agg(
    segments=("correct", "size"), recall=("correct", "mean")).round(3))
print("\\nAccuracy by data source:")
print(results.groupby("source").agg(
    segments=("correct", "size"), accuracy=("correct", "mean")).round(3))

errors = results[~results["correct"]]
print(f"\\nErrors: {{len(errors)}} of {{len(results)}}, "
      f"from {{errors['clip'].nunique()}} clips")
print("\\nMost confident errors:")
print(errors.sort_values("confidence", ascending=False)
      .head(10)[["source","true","pred","confidence"]].round(3).to_string(index=False))
"""


def save_cell_dt(slug: str = "decision_tree") -> str:
    return f"""joblib.dump(best_model, "models/model_{slug}.joblib")
joblib.dump(scaler,     "models/scaler_{slug}.joblib")
joblib.dump(le,         "models/label_encoder_{slug}.joblib")

metrics = {{
    "algorithm": "Decision Tree",
    "slug": "{slug}",
    "best_hyperparameters": {{
        "max_depth": best_depth,
        "criterion": "gini",
        "min_samples_split": 4,
        "min_samples_leaf": 2
    }},
    "test_accuracy"   : float(test_metrics["test_accuracy"]),
    "macro_precision" : float(test_metrics["macro_precision"]),
    "macro_recall"    : float(test_metrics["macro_recall"]),
    "macro_f1"        : float(test_metrics["macro_f1"]),
    "roc_auc_macro"   : float(test_metrics["roc_auc_macro"]),
    "latency_ms"      : float(test_metrics["latency_ms"]),
    "training_time_s" : float(test_metrics["training_time_s"]),
    "f1_ci_low"       : float(f1_ci[0]),
    "f1_ci_high"      : float(f1_ci[1]),
    "classes"         : classes,
    "feature_count"   : len(feature_cols)
}}

with open("models/metrics_{slug}.json", "w") as fh:
    json.dump(metrics, fh, indent=2)

print("Saved model artifacts.")
print(json.dumps(metrics, indent=2))

# ── Frozen result assertion ────────────────────────────────────────────────
# After the FIRST successful run, copy the printed macro_f1 here:
RECORDED_TEST_F1 = None   # e.g.  = 0.8120  (fill in after first run)
if RECORDED_TEST_F1 is not None:
    assert abs(metrics["macro_f1"] - RECORDED_TEST_F1) < 0.001, (
        f"Test macro F1 {{metrics['macro_f1']:.4f}} differs from "
        f"recorded {{RECORDED_TEST_F1}}. Check for data or code changes."
    )
    print(f"Frozen result confirmed: macro F1 = {{RECORDED_TEST_F1}}")

# ── Smoke-test: reload and verify saved artifacts ─────────────────────────
saved_model   = joblib.load(f"models/model_{slug}.joblib")
saved_scaler  = joblib.load(f"models/scaler_{slug}.joblib")
saved_encoder = joblib.load(f"models/label_encoder_{slug}.joblib")

fresh    = pd.read_parquet("../data/features/features.parquet") if Path("../data/features/features.parquet").exists() \\
           else pd.read_csv("../data/features/features.csv")
fresh_test = fresh[fresh["split"] == "test"]
raw_ft   = np.nan_to_num(fresh_test[feature_cols].values, nan=0.0)
ready_ft = saved_scaler.transform(raw_ft)
saved_pred = saved_model.predict(ready_ft)
saved_f1   = f1_score(fresh_test["label"].map({{c: i for i, c in enumerate(saved_encoder.classes_)}}).values,
                      saved_pred, average="macro")

assert raw_ft.shape[1] == len(feature_cols)
assert list(saved_encoder.classes_) == classes
assert (saved_pred == y_pred).all(), "Saved model predictions differ from notebook run."
print(f"Smoke-test passed — reloaded macro F1 = {{saved_f1:.4f}} ✓")
print("Files in models/:", sorted(p.name for p in Path("models").iterdir() if p.name != ".gitkeep"))
"""


def save_cell_rf(slug: str = "random_forest") -> str:
    return f"""joblib.dump(best_model, "models/model_{slug}.joblib")
joblib.dump(scaler,     "models/scaler_{slug}.joblib")
joblib.dump(le,         "models/label_encoder_{slug}.joblib")

metrics = {{
    "algorithm": "Random Forest",
    "slug": "{slug}",
    "best_hyperparameters": {{
        "n_estimators"    : best_n_est,
        "max_depth"       : best_depth,
        "min_samples_split": 4,
        "class_weight"    : "balanced"
    }},
    "test_accuracy"   : float(test_metrics["test_accuracy"]),
    "macro_precision" : float(test_metrics["macro_precision"]),
    "macro_recall"    : float(test_metrics["macro_recall"]),
    "macro_f1"        : float(test_metrics["macro_f1"]),
    "roc_auc_macro"   : float(test_metrics["roc_auc_macro"]),
    "latency_ms"      : float(test_metrics["latency_ms"]),
    "training_time_s" : float(test_metrics["training_time_s"]),
    "f1_ci_low"       : float(f1_ci[0]),
    "f1_ci_high"      : float(f1_ci[1]),
    "classes"         : classes,
    "feature_count"   : len(feature_cols)
}}

with open("models/metrics_{slug}.json", "w") as fh:
    json.dump(metrics, fh, indent=2)

print("Saved model artifacts.")
print(json.dumps(metrics, indent=2))

# ── Frozen result assertion ────────────────────────────────────────────────
RECORDED_TEST_F1 = None   # fill in after first successful run
if RECORDED_TEST_F1 is not None:
    assert abs(metrics["macro_f1"] - RECORDED_TEST_F1) < 0.001, (
        f"Test macro F1 {{metrics['macro_f1']:.4f}} differs from "
        f"recorded {{RECORDED_TEST_F1}}. Check for data or code changes."
    )
    print(f"Frozen result confirmed: macro F1 = {{RECORDED_TEST_F1}}")

# ── Smoke-test: reload and verify saved artifacts ─────────────────────────
saved_model   = joblib.load(f"models/model_{slug}.joblib")
saved_scaler  = joblib.load(f"models/scaler_{slug}.joblib")
saved_encoder = joblib.load(f"models/label_encoder_{slug}.joblib")

fresh    = pd.read_parquet("../data/features/features.parquet") if Path("../data/features/features.parquet").exists() \\
           else pd.read_csv("../data/features/features.csv")
fresh_test = fresh[fresh["split"] == "test"]
raw_ft   = np.nan_to_num(fresh_test[feature_cols].values, nan=0.0)
ready_ft = saved_scaler.transform(raw_ft)
saved_pred = saved_model.predict(ready_ft)
saved_f1   = f1_score(fresh_test["label"].map({{c: i for i, c in enumerate(saved_encoder.classes_)}}).values,
                      saved_pred, average="macro")

assert raw_ft.shape[1] == len(feature_cols)
assert list(saved_encoder.classes_) == classes
assert (saved_pred == y_pred).all(), "Saved model predictions differ from notebook run."
print(f"Smoke-test passed — reloaded macro F1 = {{saved_f1:.4f}} ✓")
print("Files in models/:", sorted(p.name for p in Path("models").iterdir() if p.name != ".gitkeep"))
"""


# ── Cell after the final-model evaluation that bundles test_metrics dict ───
# DT cell 9 currently prints raw vars — we upgrade it to also build test_metrics
def final_model_cell_dt() -> str:
    return """best_model = DecisionTreeClassifier(
    max_depth=best_depth,
    criterion='gini',
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=RANDOM_SEED
)

t0 = time.time()
best_model.fit(X_train, y_train)
train_time = time.time() - t0

# Latency benchmark (100 single-segment predictions)
t_start = time.perf_counter()
n_iters = 100
for _ in range(n_iters):
    _ = best_model.predict(X_test[:1])
latency_ms = ((time.perf_counter() - t_start) / n_iters) * 1000

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
print("  OFFICIAL TEST SET BENCHMARK — DECISION TREE")
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


def final_model_cell_rf() -> str:
    return """best_model = RandomForestClassifier(
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
print("  OFFICIAL TEST SET BENCHMARK — RANDOM FOREST")
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


# ──────────────────────────────────────────────────────────────────────────────
# Notebook patcher
# ──────────────────────────────────────────────────────────────────────────────

def patch_notebook(nb_path: Path, is_rf: bool):
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]
    algo  = "Random Forest" if is_rf else "Decision Tree"
    slug  = "random_forest" if is_rf else "decision_tree"

    extra_imports = (
        "from sklearn.ensemble import RandomForestClassifier"
        if is_rf else
        "from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text"
    )

    # ── 1. Replace imports (cell 1) ──────────────────────────────────────────
    cells[1]["source"] = imports_cell(algo, extra_imports)

    # ── 2. Replace split cell (cell 5) ───────────────────────────────────────
    cells[5]["source"] = split_cell(algo)

    # ── 3. Replace final-model cell (cell 9) ─────────────────────────────────
    cells[9]["source"] = final_model_cell_rf() if is_rf else final_model_cell_dt()

    # ── 4. Replace save cell (cell 17) with upgraded version ─────────────────
    cells[17]["source"] = save_cell_rf(slug) if is_rf else save_cell_dt(slug)

    # ── 5. Insert new cells AFTER cell 11 (confusion/ROC) ────────────────────
    #      We insert: md, bootstrap+confidence code, md, source breakdown code
    #      Insert at index 12 (before existing cell 12 markdown)
    new_cells = [
        md_cell("## 7b. Bootstrap Confidence Intervals & Decision-Support Confidence Analysis\n"
                 "We resample whole clips (not segments) to get honest uncertainty estimates,\n"
                 "and analyse the confidence threshold for flagging uncertain predictions."),
        code_cell(bootstrap_cell(slug, algo)),
        md_cell("## 7c. Per-Source-Dataset Breakdown\n"
                 "We break down accuracy and recall by the originating dataset "
                 "(ShipsEar / DeepShip / Synthetic / NOAA / Watkins) to surface any domain gaps."),
        code_cell(source_breakdown_cell(slug)),
    ]

    # Find the index of the markdown cell for section 7 (cell 12 original)
    # We want to insert AFTER cell 11 (the confusion/ROC code cell)
    insert_idx = 12
    cells[insert_idx:insert_idx] = new_cells

    nb["cells"] = cells

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"[OK] Patched {nb_path} ({len(nb['cells'])} cells total)")


if __name__ == "__main__":
    patch_notebook(DT_PATH, is_rf=False)
    patch_notebook(RF_PATH, is_rf=True)
    print("\nDone — both notebooks upgraded with Person-2-level rigour.")
