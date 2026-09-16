import json
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def make_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.13"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

# ==============================================================================
# NOTEBOOK 3: Feature Analysis and Selection
# ==============================================================================
nb3_cells = [
    md("""# 03. Feature Analysis, Collinearity Filtering & Feature Selection
### Sonobuoy Sound Detective: Offline Underwater Acoustic Event Detection
**ML Philosophy:** Classical ML & Interpretability — Selecting the most discriminative acoustic features.

In high-dimensional classical feature sets (~100 features), many descriptors are collinear (e.g. MFCC means and medians, spectral rolloff and centroid) or noisy.
This notebook implements a rigorous **3-pronged feature selection strategy**:
1. Inspecting feature matrix and handling inf/NaN values.
2. Filtering out severe multicollinearity (correlation > 0.90).
3. **ANOVA F-Test (`f_classif`)**: Ranking features by linear class separability.
4. **Mutual Information (`mutual_info_classif`)**: Ranking features by non-linear dependency.
5. **Random Forest Feature Importance (MDI)**: Ranking features based on impurity reduction in decision trees.
6. Combining rankings into a **consensus feature subset (~30-40 features)** and saving `data/features/selected_features.json`.
"""),
    code("""import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path("..").resolve() if Path(".").resolve().name == "notebooks" else Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

from src.utils.config import (
    FEATURES_PARQ, FEATURES_CSV, SEL_FEATURES, RANDOM_SEED
)

print("Libraries imported successfully.")
"""),
    md("""## 1. Load Feature Matrix and Separate Targets
"""),
    code("""path = FEATURES_PARQ if FEATURES_PARQ.exists() else FEATURES_CSV
df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Non-feature metadata columns
NON_FEATURE_COLS = {
    "clip_id", "source_clip_id", "label", "split",
    "start_sec", "end_sec", "is_poor_quality"
}

feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
print(f"Available numeric candidate features: {len(feature_cols)}")

# Clean any inf or nan values
X_raw = df[feature_cols].copy()
X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(X_raw.median())

le = LabelEncoder()
y_enc = le.fit_transform(df['label'])
class_names = list(le.classes_)
print(f"Classes encoded: {dict(zip(range(len(class_names)), class_names))}")
"""),
    md("""## 2. Correlation Analysis & Collinearity Filtering
When two features have a correlation > 0.90 (e.g. mean vs median of the same descriptor), keeping both introduces redundancy without adding information.
"""),
    code("""corr_matrix = X_raw.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.90
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
print(f"Dropping {len(to_drop)} highly collinear features (r > 0.90):")
print(to_drop[:10], "... (truncated)")

# Retain non-collinear features
X_filtered = X_raw.drop(columns=to_drop)
filtered_features = list(X_filtered.columns)
print(f"Retained non-collinear features: {len(filtered_features)}")
"""),
    md("""## 3. Multi-Criteria Feature Selection
We evaluate each candidate feature using three distinct statistical tests:
1. **ANOVA F-score:** Measures the ratio of between-class variance to within-class variance (linear).
2. **Mutual Information:** Measures non-linear mutual dependence between feature and class.
3. **Random Forest MDI:** Measures importance inside decision trees.
"""),
    code("""# 1. ANOVA F-Score
f_scores, _ = f_classif(X_filtered, y_enc)
f_scores = np.nan_to_num(f_scores, 0)
rank_anova = pd.Series(f_scores, index=filtered_features).sort_values(ascending=False)

# 2. Mutual Information
mi_scores = mutual_info_classif(X_filtered, y_enc, random_state=RANDOM_SEED)
rank_mi = pd.Series(mi_scores, index=filtered_features).sort_values(ascending=False)

# 3. Random Forest Feature Importance
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, max_depth=8, n_jobs=-1)
rf.fit(X_filtered, y_enc)
rank_rf = pd.Series(rf.feature_importances_, index=filtered_features).sort_values(ascending=False)

# Visualize top 10 for each
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

rank_anova.head(10).plot(kind='barh', ax=axes[0], color='#2b5c8f')
axes[0].set_title("Top 10: ANOVA F-Score", fontsize=11, fontweight='bold')
axes[0].invert_yaxis()

rank_mi.head(10).plot(kind='barh', ax=axes[1], color='#2a9d8f')
axes[1].set_title("Top 10: Mutual Information", fontsize=11, fontweight='bold')
axes[1].invert_yaxis()

rank_rf.head(10).plot(kind='barh', ax=axes[2], color='#e76f51')
axes[2].set_title("Top 10: Random Forest Importance", fontsize=11, fontweight='bold')
axes[2].invert_yaxis()

plt.tight_layout()
plt.show()
"""),
    md("""## 4. Consensus Ranking & Final Feature Subset Selection
We combine the normalized ranks of the three methods to select the **top 35 consensus features**.
"""),
    code("""# Compute normalized percentile rank (higher is better)
r_anova = rank_anova.rank(pct=True)
r_mi    = rank_mi.rank(pct=True)
r_rf    = rank_rf.rank(pct=True)

consensus_score = (r_anova + r_mi + r_rf) / 3.0
top_selected_features = consensus_score.sort_values(ascending=False).head(35).index.tolist()

print(f"Selected {len(top_selected_features)} consensus features:")
for i, f in enumerate(top_selected_features[:15], 1):
    print(f"  {i:>2}. {f:<30} (Score: {consensus_score[f]:.3f})")

# Save to selected_features.json
with open(SEL_FEATURES, "w") as f:
    json.dump(top_selected_features, f, indent=2)

print(f"\nSaved selected feature set to: {SEL_FEATURES}")
"""),
    md("""## 5. Correlation Heatmap of Selected Features
Let's verify that the selected feature subset is well-diversified and free of redundancy.
"""),
    code("""plt.figure(figsize=(10, 8))
top_corr = X_filtered[top_selected_features[:15]].corr()
sns.heatmap(top_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar_kws={'label': 'Pearson Correlation'})
plt.title("Correlation Matrix of Top 15 Selected Features", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
""")
]

with open(NOTEBOOKS_DIR / "03_feature_analysis_and_selection.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_nb(nb3_cells), f, indent=2)

print("Created 03_feature_analysis_and_selection.ipynb")
