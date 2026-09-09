"""
selection.py
Feature engineering and selection pipeline.

Steps:
  1. Drop constant or near-constant features
  2. Remove highly correlated features (|r| > threshold)
  3. ANOVA F-score ranking (for categorical target)
  4. Mutual information ranking
  5. Tree-based feature importance (quick RF pass)
  6. Final selection: union of top-N from each method
  7. Save selected_features.json

Usage:
  python -m src.features.selection
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif
)
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    FEATURES_PARQ, FEATURES_CSV, SEL_FEATURES, RANDOM_SEED
)


NON_FEATURE_COLS = {
    "clip_id", "source_clip_id", "label", "split",
    "start_sec", "end_sec", "is_poor_quality",
}

TOP_N_PER_METHOD = 40   # Take top 40 from each ranking method
CORR_THRESHOLD   = 0.95


def load_features(path: Path = None) -> pd.DataFrame:
    if path is None:
        path = FEATURES_PARQ if FEATURES_PARQ.exists() else FEATURES_CSV
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)


def select_features(df: pd.DataFrame = None) -> list[str]:
    """
    Run the full feature selection pipeline.
    Returns a list of selected feature names.
    """
    if df is None:
        df = load_features()

    # Use only training split for selection
    train_df = df[df["split"] == "train"].copy() if "split" in df.columns else df.copy()

    # ── Identify feature columns ─────────────────────────────────────────────────
    feature_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
    X = train_df[feature_cols].copy()
    y_raw = train_df["label"]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # ── 1. Drop constant features ────────────────────────────────────────────────
    constant_mask = X.std(axis=0) < 1e-9
    constant_cols = X.columns[constant_mask].tolist()
    X.drop(columns=constant_cols, inplace=True)
    print(f"Dropped {len(constant_cols)} constant features.")

    # ── 2. Fill NaN / inf ───────────────────────────────────────────────────────
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

    # ── 3. Remove highly correlated features ─────────────────────────────────────
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > CORR_THRESHOLD)]
    X.drop(columns=to_drop, inplace=True)
    print(f"Dropped {len(to_drop)} highly correlated features.")

    remaining = X.columns.tolist()
    print(f"{len(remaining)} features remaining after correlation pruning.")

    # ── 4. ANOVA F-score ─────────────────────────────────────────────────────────
    k_anova = min(TOP_N_PER_METHOD, len(remaining))
    selector_f = SelectKBest(f_classif, k=k_anova).fit(X.values, y)
    anova_scores = pd.Series(selector_f.scores_, index=remaining)
    top_anova = anova_scores.nlargest(k_anova).index.tolist()
    print(f"Top ANOVA features: {top_anova[:5]} ...")

    # ── 5. Mutual information ────────────────────────────────────────────────────
    mi_scores = mutual_info_classif(X.values, y, random_state=RANDOM_SEED)
    mi_series = pd.Series(mi_scores, index=remaining)
    top_mi = mi_series.nlargest(TOP_N_PER_METHOD).index.tolist()
    print(f"Top MI features: {top_mi[:5]} ...")

    # ── 6. Tree-based importance ─────────────────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X.values, y)
    imp_series = pd.Series(rf.feature_importances_, index=remaining)
    top_rf = imp_series.nlargest(TOP_N_PER_METHOD).index.tolist()
    print(f"Top RF importance features: {top_rf[:5]} ...")

    # ── 7. Union of top features ─────────────────────────────────────────────────
    selected = list(dict.fromkeys(top_anova + top_mi + top_rf))  # preserves order, deduplicates

    print(f"\nSelected {len(selected)} features (union of top-{TOP_N_PER_METHOD} from each method).")

    # ── Save ─────────────────────────────────────────────────────────────────────
    SEL_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "selected_features": selected,
        "n_features":        len(selected),
        "method_counts": {
            "anova_top_n":   len(top_anova),
            "mi_top_n":      len(top_mi),
            "rf_top_n":      len(top_rf),
        },
        "dropped_constant":    len(constant_cols),
        "dropped_correlated":  len(to_drop),
        "corr_threshold":      CORR_THRESHOLD,
        "top_anova":           top_anova[:10],
        "top_mi":              top_mi[:10],
        "top_rf_importance":   top_rf[:10],
    }
    with open(SEL_FEATURES, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {SEL_FEATURES}")
    return selected


if __name__ == "__main__":
    select_features()
