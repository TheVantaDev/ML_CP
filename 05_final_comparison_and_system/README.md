# 05: Master 8-Model Comparison & Ground-Station Decision Support
**Role:** Final Team Showdown & Operational Prototype (All 4 Team Members)  

## Final Benchmark Results (Identical Held-Out Test Split)

Evaluated on 482 segments across the 3 target acoustic classes (`vessel_like`, `ambient`, `unknown`) using the shared **66 handcrafted acoustic features**:

| Rank | Team Member | Algorithm | Test Accuracy | Balanced Acc | Macro-F1 | ROC-AUC | Inference Latency |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Person 3** | **Random Forest** | **99.38%** | **96.59%** | **96.38%** | **0.9947** | 0.116 ms |
| 🥈 | **Person 1** | **K-Nearest Neighbors** | 99.17% | 89.26% | 93.93% | 0.9852 | 0.373 ms |
| 🥉 | **Person 4** | **XGBoost** | 99.17% | 89.26% | 93.93% | **0.9968** | **0.007 ms** |
| 4 | Person 3 | Decision Tree | 98.13% | 92.97% | 92.35% | 0.9417 | 0.001 ms |
| 5 | Person 2 | Gaussian Naive Bayes | 97.72% | 89.19% | 88.57% | 0.9639 | 0.018 ms |
| 6 | Person 2 | SVM (RBF) | 98.13% | 85.71% | 88.52% | 0.9787 | 0.033 ms |
| 7 | Person 1 | Logistic Regression | 97.30% | 83.37% | 84.21% | 0.9924 | 0.003 ms |
| 8 | Person 4 | AdaBoost | 96.89% | 56.67% | 60.24% | 0.9780 | 0.070 ms |

---

## Winning Candidate & Operational Selection
- **Overall Champion:** **Random Forest (Person 3)** wins on classification performance with a **96.38% Macro-F1** and **0.9947 ROC-AUC**, successfully capturing the complex non-linear acoustic boundary between ambient marine conditions and vessel signatures.
- **Top Edge-Streaming Candidate:** **XGBoost (Person 4)** achieves the highest overall **ROC-AUC (0.9968)** and identical top-tier Macro-F1 (93.93%) with blistering **0.007 ms** inference per segment.

---

## Contents
- `final_model_comparison.ipynb`: Fully executed side-by-side benchmark notebook containing confusion matrices, ROC curves, latency analysis, and per-class breakdowns.
- `decision_support_system.ipynb`: Production simulation with deterministic acoustic cross-check rules.
- `plots/`:
  - `all_8_confusion_matrices.png`: 2×4 normalized confusion matrix grid.
  - `all_8_models_roc_curves.png`: Macro One-vs-Rest ROC curve comparison.
  - `latency_vs_f1_tradeoff.png`: Quality vs computational complexity scatter plot.
  - `per_class_f1_breakdown.png`: Performance breakdown per acoustic event class.
- `reports/`:
  - `master_8_models_comparison.csv`: Official benchmark rankings and metrics.
