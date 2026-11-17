# 06 — GMM Generative Model (Branch B)

## What This Is

This folder contains the **Gaussian Mixture Model (GMM)** — a *generative probabilistic* classifier for underwater acoustic sound classification.

This is **Branch B** of the dual-engine system. It works alongside the existing Random Forest (Branch A) to provide a second opinion on every prediction.

---

## Why GMM Is Different From All Other Models

| Property | Branch A (Random Forest / SVM / etc.) | Branch B (GMM — This folder) |
|---|---|---|
| **Paradigm** | Discriminative — learns **boundaries** between classes | Generative — learns **statistical distribution** of each class |
| **What it learns** | P(Y\|X) — the best boundary line | P(X\|Y) — what each sound class "looks like" |
| **Unknown sounds** | Forces a class label even if sound is completely new | Flags it as **ANOMALY** if it doesn't fit any known distribution |
| **Output** | Class label + probability | Class label + log-likelihood score + anomaly flag |

---

## Folder Structure

```
06_gmm_model/
├── train_gmm.py          ← Training script (run this once)
├── models/
│   ├── gmm_vessel_like.joblib
│   ├── gmm_ambient.joblib
│   ├── gmm_unknown.joblib
│   ├── gmm_scaler.joblib
│   └── gmm_metadata.json
├── plots/
│   ├── gmm_confusion_matrix.png
│   └── gmm_likelihood_distributions.png
└── README.md
```

---

## How to Train

```bash
python 06_gmm_model/train_gmm.py
```

Uses the same `data/features/features.csv` and `data/features/selected_features.json` as all other models. No new data needed.

---

## How It Makes Decisions

```
New Audio → Extract Features (same pipeline as Branch A)
                │
                ▼
    Ask each class GMM: "How likely is this sound?"
    ┌──────────────────────────────────────────┐
    │  vessel_like GMM → score = -12.4         │
    │  ambient GMM     → score = -85.1         │
    │  unknown GMM     → score = -78.3         │
    └──────────────────────────────────────────┘
                │
                ▼
    Winner: vessel_like (highest score)
                │
                ▼
    Sanity check: is -12.4 above anomaly threshold?
    ├── YES → Predict: vessel_like ✅
    └── NO  → Predict: ANOMALY ⚠️ (flag for operator review)
```
