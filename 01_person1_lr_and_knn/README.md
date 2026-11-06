# Person 1: Linear & Distance-Based Classifiers
**Assigned Team Member:** Person 1  
**Notebooks in this folder:**
1. `logistic_regression.ipynb` — Multinomial Logistic Regression with Softmax & L2 Regularization
2. `knn.ipynb` — Distance-Weighted K-Nearest Neighbors (k=1..25 tuning)

### Contents:
- `models/`: Saved `.joblib` model binaries, scalers, and metric reports.
- `plots/`: Confusion matrices, multi-class ROC curves, tuning, permutation-importance, and coefficient plots.

## KNN implementation

`knn.ipynb` uses the shared `data/features/features.csv` table and the 66
features listed in `data/features/selected_features.json`. Raw-audio
preprocessing is performed upstream by the shared project pipeline; this
notebook performs model-level median imputation and standardization inside the
KNN pipeline.

The notebook keeps the official train/validation/test split, verifies that
source clips do not cross those splits, and groups five-fold cross-validation
by `source_clip_id`. It documents the class distribution and compares:

- `k` values `[3, 5, 7, 9, 11, 15, 21]`
- Euclidean and Manhattan distances
- Uniform and distance-weighted voting
- A final compact grid search using macro F1

It also produces learning-curve, probability-reliability, confusion-matrix,
ROC, precision-recall, source-wise error-analysis, and validation permutation
importance outputs. The current executed run selected:

| Setting | Value |
|---|---:|
| `n_neighbors` | 7 |
| `weights` | `distance` |
| Minkowski `p` | 1 (Manhattan distance) |
| Grouped CV macro F1 | 0.937 |
| Held-out test accuracy | 0.992 |
| Held-out test balanced accuracy | 0.893 |
| Held-out test macro F1 | 0.939 |
| Held-out test ROC-AUC | 0.985 |
| Latency | About 2.04 ms per segment |

The notebook saves `models/knn_tuning_results.csv` and
`models/knn_metadata.json`. Permutation importance is displayed and plotted
in the notebook without a separate CSV export. It reloads the exported
model/scaler/label encoder and verifies that the reloaded model reproduces
the notebook test predictions and probabilities.

## Logistic Regression implementation

`logistic_regression.ipynb` follows the same shared-data and evaluation
protocol. It uses the official train/validation/test assignments, checks
source-clip leakage, fits median imputation and standardization inside a
multinomial Logistic Regression pipeline, and selects `C` and
`class_weight` using five-fold `StratifiedGroupKFold` with macro F1.

The notebook generates C-selection, class-weight, learning-curve,
probability-reliability, confusion-matrix, ROC/precision-recall,
coefficient-importance, and permutation-importance plots. It also reports
validation and locked-test metrics, training and prediction timing, and
source-wise error analysis, printed in the notebook without exporting a separate error-analysis CSV.

The latest executed run selected `C=0.1` with no class weighting:

| Setting | Value |
|---|---:|
| `C` | 0.1 |
| `class_weight` | `None` |
| Grouped CV macro F1 | 0.921 |
| Held-out test accuracy | 0.973 |
| Held-out test balanced accuracy | 0.834 |
| Held-out test macro F1 | 0.842 |
| Held-out test ROC-AUC | 0.992 |
| Held-out test log loss | 0.132 |

The notebook saves the Logistic Regression model, scaler, label encoder,
metrics, metadata, tuning results, coefficient importance, permutation
importance, and test error-analysis files under `models/`. It reloads the
exported model and verifies that predictions and probabilities match the
final test evaluation.
