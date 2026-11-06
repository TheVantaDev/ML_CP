# Person 2: Kernel & Probabilistic Classifiers
**Assigned Team Member:** Person 2

Two classical models for underwater acoustic event classification, trained on the shared features in `data/features/`.

## Notebooks

| Notebook | What it does |
|---|---|
| `eda_person2.ipynb` | Checks the data and whether it suits each model |
| `svm_rbf.ipynb` | Tunes, evaluates and exports the SVM with an RBF kernel |
| `naive_bayes.ipynb` | Tunes, calibrates, evaluates and exports Gaussian Naive Bayes |

## Final results (held-out test set)

Data: 482 test segments from 91 clips, in 3 classes.

| Measure | SVM (RBF) | Naive Bayes + `isotonic` |
|---|---:|---:|
| **Macro F1** | **0.885** | **0.886** |
| 95% interval (resampling clips) | 0.780 to 0.955 | 0.767 to 0.979 |
| Accuracy | 98.1% | 97.7% |
| ROC-AUC | 0.979 | 0.964 |
| Log loss | 0.063 | 0.075 |
| Recall: `ambient` | 0.800 | 0.800 |
| Recall: `unknown` | 0.778 | 0.889 |
| Recall: `vessel_like` | 0.993 | 0.987 |
| Latency per segment | About 0.11 ms | About 4 ms |

The two models can't be separated on macro F1. The SVM is stronger on grouped CV (0.961 against 0.884), validation (0.920 against 0.843), accuracy, ROC-AUC, log loss and latency.

**Baseline:** always predicting the most common class gives a macro F1 of about 0.32.

## Final settings

| Model | Settings |
|---|---|
| SVM (RBF) | `StandardScaler`, `C=10`, `gamma=0.01515`, `class_weight=None`, `probability=True` |
| Gaussian Naive Bayes | No scaling, learned priors, `var_smoothing=1e-07`, `isotonic` calibration |

## Method

- **Data:** the official train, validation and test split, with the shared 66 selected features.
- **Main metric:** macro F1, because 92% of segments are `vessel_like`.
- **Tuning:** 5-fold CV on training data only, grouped by `source_clip_id`, because one clip can produce up to 101 segments.
- **Model choice:** the simplest setting within one standard error of the top score.
- **Test set:** kept locked until the settings were frozen, and scored only with those settings.

## Limitations

- **Real ambient audio is the main concern:** recall on real ShipsEar `ambient` test segments is 1 of 4 for the SVM and 2 of 4 for Naive Bayes, against 15 of 16 and 14 of 16 on synthetic audio. The sample is small, so this is a warning and not a measurement.
- **Synthetic-heavy classes:** all `unknown` clips and most `ambient` clips are synthetic, so results may not carry over to real recordings.
- **Few minority test segments:** only 20 `ambient` and 9 `unknown`, so scores carry wide intervals.
- **`marine_life` has no data:** the models cover 3 of the 4 planned classes.
- **Calibration trade-off:** `isotonic` removes most false `unknown` alarms in Naive Bayes, but it misses more true `unknown` segments.
- **The 60% review rule catches only some errors:** some mistakes are made with high confidence.

This is an academic prototype, and these results shouldn't be read as operational performance.

## Files

- **`models/`:** `model_*`, `scaler_*`, `label_encoder_*` and `metrics_*` for `svm_rbf` and `naive_bayes`, plus the tuning results as CSV files. The Naive Bayes scaler file leaves features unchanged, so the same loading code works for both models.
- **`plots/`:** data exploration (`eda_*`), tuning (`svm_*`, `nb_*`), `calibration_*`, `confusion_matrix_*`, `roc_curves_*`, `pr_curves_*` and `permutation_importance_*`.

## How to re-run

1. Use Python 3.10 with `pip install -r requirements.txt` from the project root. The saved models were created with **scikit-learn 1.7.2**, so use the same version to load them.
2. Open a notebook from this folder, because its file paths are relative.
3. Run all cells. Each notebook stops before saving if the test score differs from the recorded result. If you change the data or settings, update `RECORDED_TEST_F1` in the export cell only after reviewing why the score changed.
