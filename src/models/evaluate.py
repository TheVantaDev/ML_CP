"""
evaluate.py
Model evaluation: metrics, confusion matrix, ROC-AUC, and plots.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import PLOTS_DIR


def evaluate_model(
    model,
    X_test:       np.ndarray,
    y_te:         np.ndarray,
    X_val:        np.ndarray = None,
    y_val:        np.ndarray = None,
    class_names:  list = None,
    model_name:   str  = "model",
) -> dict:
    """
    Evaluate a trained model on test (and optionally val) sets.
    Saves confusion matrix plot.
    Returns a dict of metrics.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(y_te)))]

    def _metrics_set(model, X, y, split_name):
        y_pred = model.predict(X)
        acc    = accuracy_score(y, y_pred)
        f1     = f1_score(y, y_pred, average="macro", zero_division=0)
        report = classification_report(y, y_pred, target_names=class_names,
                                       output_dict=True, zero_division=0)
        # ROC-AUC (one-vs-rest, requires predict_proba)
        auc = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X)
                n_cls = y_prob.shape[1]
                if n_cls == 2:
                    auc = float(roc_auc_score(y, y_prob[:, 1]))
                else:
                    auc = float(roc_auc_score(y, y_prob, multi_class="ovr", average="macro"))
            except Exception:
                auc = None

        return {
            "accuracy":     float(acc),
            "macro_f1":     float(f1),
            "roc_auc_macro":auc,
            "per_class":    {k: v for k, v in report.items() if k in class_names},
        }

    metrics = {}
    metrics["test"] = _metrics_set(model, X_test, y_te, "test")
    if X_val is not None and y_val is not None:
        metrics["val"] = _metrics_set(model, X_val, y_val, "val")

    # ── Confusion matrix plot ────────────────────────────────────────────────────
    y_pred_test = model.predict(X_test)
    cm = confusion_matrix(y_te, y_pred_test)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name} (test set)")
    plt.tight_layout()
    cm_path = PLOTS_DIR / f"confusion_matrix_{model_name}.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved -> {cm_path}")

    # ── Feature importance plot (RF / XGBoost) ────────────────────────────────────
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        top_k = min(20, len(imp))
        idx   = np.argsort(imp)[-top_k:][::-1]
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.bar(range(top_k), imp[idx])
        ax2.set_title(f"Feature Importance (top {top_k}) — {model_name}")
        ax2.set_xticks(range(top_k))
        ax2.set_xticklabels([str(i) for i in idx], rotation=45, ha="right", fontsize=7)
        plt.tight_layout()
        fi_path = PLOTS_DIR / f"feature_importance_{model_name}.png"
        fig2.savefig(fi_path, dpi=150)
        plt.close(fig2)
        print(f"  Feature importance saved -> {fi_path}")

    return metrics
