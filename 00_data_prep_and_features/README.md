# 00: Data Ingestion, Signal Preprocessing & Handcrafted Feature Extraction
**Role:** Shared Data Preparation for All Team Members  
**Primary Notebook:** `data_prep_and_feature_extraction.ipynb`

### Overview:
- Audio ingestion (ShipsEar, DeepShip, Synthetic).
- DSP filtering (20–5,000 Hz Butterworth bandpass).
- Sliding-window 4.0-second segmentation.
- Extracts ~100 handcrafted acoustic features (MFCCs, spectral descriptors, tonal persistence) into `data/features/features.parquet`.
- Runs ANOVA F-test, Mutual Information, and Random Forest feature selection to save `data/features/selected_features.json`.
