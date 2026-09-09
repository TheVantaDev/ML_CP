# Sonobuoy Sound Detective

> Offline Underwater Acoustic Event Detection, Classical ML Classification,
> and Operator Decision-Support System

**ML approach:** Classical ML only — no deep learning.  
**Deployment:** Offline ground-station prototype.  
**Data:** Public / synthetic only. No classified or restricted data.

---

## Project Structure

```
ML_CP/
├── data/
│   ├── raw/          shipsear/ deepship/ noaa/ synthetic/
│   ├── processed/    vessel_like/ marine_life/ ambient/ unknown/
│   ├── metadata/     master_metadata.csv  labels.csv
│   └── features/     features.parquet  selected_features.json
├── src/
│   ├── data/         download.py  synthetic.py  organizer.py
│   ├── preprocessing/ audio_utils.py  segmenter.py  quality_check.py
│   ├── features/     extractor.py  event_detection.py  selection.py  build_dataset.py
│   ├── models/       train.py  evaluate.py  registry.py
│   └── utils/        config.py
├── models/           saved model artifacts (versioned)
├── reports/plots/    confusion matrices, feature importance plots
├── tests/            test_pipeline.py
├── run_pipeline.py   master orchestration script
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline (synthetic data only — fastest)
```bash
python run_pipeline.py --synthetic-only
```

### 3. Run individual steps
```bash
# Step 1: Generate synthetic audio
python -m src.data.synthetic --vessel 200 --ambient 200 --unknown 100

# Step 2: Download public datasets (DeepShip + NOAA)
python -m src.data.download --source all

# Step 3: Organize raw data into processed/ + metadata
python -m src.data.organizer

# Step 4: Extract features from all processed files
python -m src.features.build_dataset

# Step 5: Feature selection
python -m src.features.selection

# Step 6: Train + evaluate all models
python -m src.models.train

# Train only one model
python -m src.models.train --model random_forest
```

### 4. Run tests
```bash
pytest tests/ -v
```

---

## Classes

| Class         | Description |
|---------------|-------------|
| `vessel_like` | Ship, boat, tug, fishing-vessel, machinery-like radiated noise |
| `marine_life` | Whale/dolphin/fish bioacoustic calls |
| `ambient`     | Waves, rain, wind, ocean background noise |
| `unknown`     | Low-quality, mixed, or unlabelled sounds |

---

## Datasets

| Dataset   | Status   | Content |
|-----------|----------|---------|
| Synthetic | Auto-generated | Controlled vessel tones + ambient noise |
| DeepShip  | Auto-download attempted | 47h ship recordings, 4 classes |
| NOAA SanctSound | Auto-download attempted | Passive ocean ambient recordings |
| ShipsEar  | Manual — request from `atlantis.tic.uvigo.es/shipsear/` | Vessel noise + natural background |

---

## Technology Stack

| Layer       | Tool |
|-------------|------|
| Audio I/O   | `librosa`, `soundfile` |
| DSP         | `scipy`, `numpy` |
| Classical ML| `scikit-learn`, `xgboost` |
| Data        | `pandas` |
| Plots       | `matplotlib`, `seaborn` |
| Models      | `joblib` (versioned saving) |

---

## Pipeline Architecture

```
Synthetic / Raw Audio
        |
        v
[Audio Preprocessing]  -- mono, resample 22050Hz, bandpass, noise gate, RMS norm
        |
        v
[Segmentation]         -- 4s windows, 50% overlap, silence removal
        |
        v
[Event Detection]      -- energy threshold, FFT peaks, tonal persistence
        |
        v
[Feature Extraction]   -- MFCCs, spectral, energy, temporal, tonal (~100 features)
        |
        v
[Feature Selection]    -- ANOVA + MI + RF importance -> selected_features.json
        |
        v
[Model Training]       -- LR, KNN, SVM, RandomForest, XGBoost
        |
        v
[Evaluation]           -- confusion matrix, macro-F1, ROC-AUC, feature importance
        |
        v
[Model Registry]       -- models/<name>_v1/ with metadata.json
```
