"""
Sonobuoy Sound Detective — Synthetic Feature Dataset Generator
================================================================
Generates realistic underwater acoustic feature data matching the
6-stage pipeline described in the review preparation guide.

Output:
  - sonobuoy_features.csv           (full feature matrix + labels)
  - selected_features.json          (top-35 feature names)
  - train.csv / val.csv / test.csv  (stratified, recording-isolated splits)
"""

import numpy as np
import pandas as pd
import json
import os

np.random.seed(42)

# ── Class definitions (4-class task) ─────────────────────────────────────────
CLASSES = ["vessel_like", "marine_life", "ambient", "unknown"]
CLASS_COUNTS = {
    "vessel_like": 280,
    "marine_life": 220,
    "ambient":     210,
    "unknown":     153,
}  # Total ≈ 863 segments (matching guide)

# ── 35 Selected Features (physics-based, matching guide Section 3 & 4) ───────
FEATURE_NAMES = [
    # Perceptual / MFCC features (coefficients 1-13 stats → top selected)
    "mfcc_1_mean", "mfcc_2_mean", "mfcc_3_mean", "mfcc_4_mean",
    "mfcc_1_std", "mfcc_2_std", "mfcc_3_std",
    "mfcc_1_median",
    "delta_mfcc_1_mean", "delta_mfcc_2_mean", "delta_mfcc_3_mean",

    # Spectral descriptors
    "spectral_centroid_mean", "spectral_centroid_std",
    "spectral_rolloff_mean",
    "spectral_bandwidth_mean",
    "spectral_flatness_mean", "spectral_flatness_std",
    "spectral_contrast_band1", "spectral_contrast_band2",
    "spectral_contrast_band3", "spectral_contrast_band4",

    # Underwater domain / tonal features
    "dominant_frequency", "peak_freq_2", "peak_freq_3",
    "tonal_persistence",
    "tonal_to_broadband_ratio",

    # Temporal & energy features
    "zcr_mean", "zcr_std",
    "rms_energy_mean", "rms_energy_std",
    "dynamic_range",
    "attack_decay_ratio",

    # Quality telemetry
    "snr_estimate",

    # Additional selected features from collinearity pruning
    "spectral_entropy",
    "harmonic_ratio",
]

assert len(FEATURE_NAMES) == 35, f"Expected 35 features, got {len(FEATURE_NAMES)}"


def _class_distribution(class_name: str, n: int) -> np.ndarray:
    """
    Generate feature vectors with class-specific statistical profiles
    that mimic real underwater acoustic characteristics.
    """
    rng = np.random.default_rng(hash(class_name) % (2**31))
    X = np.zeros((n, 35))

    if class_name == "vessel_like":
        # Vessels: low spectral centroid, high tonal persistence, high energy,
        # strong harmonic structure, low spectral flatness
        X[:, 0] = rng.normal(-12.0, 3.0, n)   # mfcc_1_mean (low)
        X[:, 1] = rng.normal(8.0, 2.5, n)     # mfcc_2_mean
        X[:, 2] = rng.normal(-3.0, 1.8, n)    # mfcc_3_mean
        X[:, 3] = rng.normal(2.5, 1.5, n)     # mfcc_4_mean
        X[:, 4] = rng.normal(4.0, 1.0, n)     # mfcc_1_std
        X[:, 5] = rng.normal(3.5, 0.8, n)     # mfcc_2_std
        X[:, 6] = rng.normal(2.8, 0.7, n)     # mfcc_3_std
        X[:, 7] = rng.normal(-13.0, 3.5, n)   # mfcc_1_median
        X[:, 8] = rng.normal(0.3, 0.15, n)    # delta_mfcc_1_mean
        X[:, 9] = rng.normal(-0.1, 0.12, n)   # delta_mfcc_2_mean
        X[:, 10] = rng.normal(0.05, 0.1, n)   # delta_mfcc_3_mean
        X[:, 11] = rng.normal(450, 120, n)     # spectral_centroid_mean (LOW for vessels)
        X[:, 12] = rng.normal(80, 25, n)      # spectral_centroid_std
        X[:, 13] = rng.normal(1200, 300, n)   # spectral_rolloff_mean
        X[:, 14] = rng.normal(800, 200, n)    # spectral_bandwidth_mean
        X[:, 15] = rng.normal(0.05, 0.02, n)  # spectral_flatness_mean (LOW — tonal)
        X[:, 16] = rng.normal(0.03, 0.01, n)  # spectral_flatness_std
        X[:, 17] = rng.normal(25, 5, n)       # spectral_contrast_band1 (HIGH)
        X[:, 18] = rng.normal(22, 4, n)       # spectral_contrast_band2
        X[:, 19] = rng.normal(18, 4, n)       # spectral_contrast_band3
        X[:, 20] = rng.normal(12, 3, n)       # spectral_contrast_band4
        X[:, 21] = rng.normal(120, 40, n)     # dominant_frequency (shaft-rate range)
        X[:, 22] = rng.normal(240, 60, n)     # peak_freq_2 (blade-rate harmonic)
        X[:, 23] = rng.normal(360, 80, n)     # peak_freq_3 (2nd harmonic)
        X[:, 24] = rng.normal(0.85, 0.08, n)  # tonal_persistence (HIGH)
        X[:, 25] = rng.normal(3.5, 0.8, n)    # tonal_to_broadband_ratio (HIGH)
        X[:, 26] = rng.normal(0.03, 0.01, n)  # zcr_mean (LOW)
        X[:, 27] = rng.normal(0.008, 0.003, n)# zcr_std
        X[:, 28] = rng.normal(0.15, 0.04, n)  # rms_energy_mean (HIGH)
        X[:, 29] = rng.normal(0.03, 0.01, n)  # rms_energy_std
        X[:, 30] = rng.normal(35, 8, n)       # dynamic_range (dB)
        X[:, 31] = rng.normal(0.4, 0.15, n)   # attack_decay_ratio
        X[:, 32] = rng.normal(18, 4, n)       # snr_estimate (HIGH)
        X[:, 33] = rng.normal(3.2, 0.5, n)    # spectral_entropy (LOW — structured)
        X[:, 34] = rng.normal(0.75, 0.1, n)   # harmonic_ratio (HIGH)

    elif class_name == "marine_life":
        # Marine life: mid-range centroid, moderate tonal persistence,
        # frequency sweeps (whistles), variable energy
        X[:, 0] = rng.normal(-6.0, 4.0, n)
        X[:, 1] = rng.normal(5.0, 3.0, n)
        X[:, 2] = rng.normal(1.0, 2.5, n)
        X[:, 3] = rng.normal(0.5, 2.0, n)
        X[:, 4] = rng.normal(5.5, 1.5, n)
        X[:, 5] = rng.normal(4.0, 1.2, n)
        X[:, 6] = rng.normal(3.5, 1.0, n)
        X[:, 7] = rng.normal(-7.0, 4.5, n)
        X[:, 8] = rng.normal(0.5, 0.25, n)     # Higher delta (frequency sweeps)
        X[:, 9] = rng.normal(0.3, 0.2, n)
        X[:, 10] = rng.normal(0.2, 0.15, n)
        X[:, 11] = rng.normal(1800, 500, n)     # Higher centroid (whistles)
        X[:, 12] = rng.normal(300, 80, n)       # High variance (sweeps)
        X[:, 13] = rng.normal(3200, 600, n)
        X[:, 14] = rng.normal(1500, 400, n)
        X[:, 15] = rng.normal(0.15, 0.06, n)
        X[:, 16] = rng.normal(0.08, 0.03, n)
        X[:, 17] = rng.normal(15, 5, n)
        X[:, 18] = rng.normal(18, 5, n)
        X[:, 19] = rng.normal(14, 4, n)
        X[:, 20] = rng.normal(10, 3, n)
        X[:, 21] = rng.normal(800, 300, n)
        X[:, 22] = rng.normal(1500, 500, n)
        X[:, 23] = rng.normal(2500, 700, n)
        X[:, 24] = rng.normal(0.55, 0.15, n)    # Moderate persistence
        X[:, 25] = rng.normal(1.5, 0.6, n)
        X[:, 26] = rng.normal(0.08, 0.03, n)
        X[:, 27] = rng.normal(0.025, 0.01, n)
        X[:, 28] = rng.normal(0.08, 0.03, n)
        X[:, 29] = rng.normal(0.04, 0.02, n)
        X[:, 30] = rng.normal(25, 6, n)
        X[:, 31] = rng.normal(0.7, 0.2, n)      # Quick attack (clicks)
        X[:, 32] = rng.normal(12, 5, n)
        X[:, 33] = rng.normal(5.0, 0.8, n)
        X[:, 34] = rng.normal(0.45, 0.15, n)

    elif class_name == "ambient":
        # Ambient: high spectral flatness (white-noise-like), no tonal persistence,
        # low energy, high ZCR, high spectral entropy
        X[:, 0] = rng.normal(-2.0, 2.5, n)
        X[:, 1] = rng.normal(1.0, 2.0, n)
        X[:, 2] = rng.normal(0.5, 1.5, n)
        X[:, 3] = rng.normal(-0.5, 1.2, n)
        X[:, 4] = rng.normal(2.5, 0.8, n)
        X[:, 5] = rng.normal(2.0, 0.6, n)
        X[:, 6] = rng.normal(1.5, 0.5, n)
        X[:, 7] = rng.normal(-2.5, 2.8, n)
        X[:, 8] = rng.normal(0.02, 0.05, n)
        X[:, 9] = rng.normal(0.01, 0.04, n)
        X[:, 10] = rng.normal(0.0, 0.03, n)
        X[:, 11] = rng.normal(2800, 400, n)      # High centroid (broadband)
        X[:, 12] = rng.normal(150, 50, n)
        X[:, 13] = rng.normal(4200, 500, n)
        X[:, 14] = rng.normal(2200, 350, n)
        X[:, 15] = rng.normal(0.65, 0.12, n)     # HIGH flatness (noise-like)
        X[:, 16] = rng.normal(0.10, 0.04, n)
        X[:, 17] = rng.normal(5, 2, n)           # LOW contrast
        X[:, 18] = rng.normal(6, 2, n)
        X[:, 19] = rng.normal(5, 2, n)
        X[:, 20] = rng.normal(4, 1.5, n)
        X[:, 21] = rng.normal(2000, 800, n)      # Random dominant frequency
        X[:, 22] = rng.normal(3000, 900, n)
        X[:, 23] = rng.normal(4000, 800, n)
        X[:, 24] = rng.normal(0.10, 0.07, n)     # Very LOW tonal persistence
        X[:, 25] = rng.normal(0.3, 0.15, n)      # LOW tonal-to-broadband
        X[:, 26] = rng.normal(0.15, 0.04, n)     # HIGH ZCR (noise)
        X[:, 27] = rng.normal(0.03, 0.01, n)
        X[:, 28] = rng.normal(0.02, 0.008, n)    # LOW energy
        X[:, 29] = rng.normal(0.005, 0.002, n)
        X[:, 30] = rng.normal(12, 4, n)
        X[:, 31] = rng.normal(1.0, 0.3, n)
        X[:, 32] = rng.normal(5, 3, n)           # LOW SNR
        X[:, 33] = rng.normal(7.5, 0.6, n)       # HIGH entropy (unstructured)
        X[:, 34] = rng.normal(0.10, 0.06, n)     # LOW harmonic ratio

    elif class_name == "unknown":
        # Unknown: mixture / anomalous — broad distributions, overlapping
        X[:, 0] = rng.normal(-5.0, 6.0, n)
        X[:, 1] = rng.normal(3.0, 4.0, n)
        X[:, 2] = rng.normal(0.0, 3.0, n)
        X[:, 3] = rng.normal(0.0, 2.5, n)
        X[:, 4] = rng.normal(4.0, 2.0, n)
        X[:, 5] = rng.normal(3.0, 1.5, n)
        X[:, 6] = rng.normal(2.5, 1.2, n)
        X[:, 7] = rng.normal(-6.0, 6.5, n)
        X[:, 8] = rng.normal(0.2, 0.3, n)
        X[:, 9] = rng.normal(0.1, 0.2, n)
        X[:, 10] = rng.normal(0.05, 0.15, n)
        X[:, 11] = rng.normal(1500, 800, n)
        X[:, 12] = rng.normal(200, 100, n)
        X[:, 13] = rng.normal(2800, 800, n)
        X[:, 14] = rng.normal(1400, 500, n)
        X[:, 15] = rng.normal(0.35, 0.2, n)
        X[:, 16] = rng.normal(0.08, 0.05, n)
        X[:, 17] = rng.normal(12, 6, n)
        X[:, 18] = rng.normal(13, 6, n)
        X[:, 19] = rng.normal(10, 5, n)
        X[:, 20] = rng.normal(7, 4, n)
        X[:, 21] = rng.normal(900, 600, n)
        X[:, 22] = rng.normal(1800, 700, n)
        X[:, 23] = rng.normal(2800, 900, n)
        X[:, 24] = rng.normal(0.35, 0.2, n)
        X[:, 25] = rng.normal(1.0, 0.8, n)
        X[:, 26] = rng.normal(0.09, 0.04, n)
        X[:, 27] = rng.normal(0.02, 0.01, n)
        X[:, 28] = rng.normal(0.06, 0.04, n)
        X[:, 29] = rng.normal(0.02, 0.015, n)
        X[:, 30] = rng.normal(20, 8, n)
        X[:, 31] = rng.normal(0.6, 0.25, n)
        X[:, 32] = rng.normal(8, 5, n)
        X[:, 33] = rng.normal(5.5, 1.5, n)
        X[:, 34] = rng.normal(0.35, 0.2, n)

    return X


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # ── Generate class-specific feature matrices ─────────────────────────────
    all_X, all_y, all_rec = [], [], []
    rec_id = 0
    for cls, count in CLASS_COUNTS.items():
        X_cls = _class_distribution(cls, count)
        all_X.append(X_cls)
        all_y.extend([cls] * count)

        # Assign recording IDs (simulate ~5-12 segments per recording)
        segs_left = count
        while segs_left > 0:
            batch = min(np.random.randint(5, 13), segs_left)
            all_rec.extend([f"rec_{rec_id:04d}"] * batch)
            rec_id += 1
            segs_left -= batch

    X = np.vstack(all_X)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = all_y
    df["recording_id"] = all_rec

    # ── Clip physically impossible values ────────────────────────────────────
    for col in ["tonal_persistence", "harmonic_ratio"]:
        df[col] = df[col].clip(0, 1)
    for col in ["spectral_flatness_mean", "spectral_flatness_std"]:
        df[col] = df[col].clip(0, 1)
    for col in ["rms_energy_mean", "rms_energy_std", "zcr_mean", "zcr_std"]:
        df[col] = df[col].clip(0, None)
    df["snr_estimate"] = df["snr_estimate"].clip(0, None)
    df["dynamic_range"] = df["dynamic_range"].clip(0, None)

    # ── Save full dataset ────────────────────────────────────────────────────
    df.to_csv(os.path.join(data_dir, "sonobuoy_features.csv"), index=False)
    print(f"[OK] Saved sonobuoy_features.csv  ({len(df)} samples x {len(FEATURE_NAMES)} features)")

    # ── Save selected_features.json ──────────────────────────────────────────
    with open(os.path.join(data_dir, "selected_features.json"), "w") as f:
        json.dump({"selected_features": FEATURE_NAMES, "count": len(FEATURE_NAMES)}, f, indent=2)
    print(f"[OK] Saved selected_features.json  ({len(FEATURE_NAMES)} features)")

    # ── Two-Stage Recording-Level Isolated Split (20% Unseen + 80% Dev -> 80/20 Train/Test) ──
    # Stage 1: Hold out 20% for Unseen Testing (Phase 11)
    from sklearn.model_selection import GroupShuffleSplit

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    dev_idx, unseen_idx = next(gss1.split(df, df["label"], groups=df["recording_id"]))

    df_dev = df.iloc[dev_idx].reset_index(drop=True)
    df_unseen = df.iloc[unseen_idx].reset_index(drop=True)

    # Stage 2: Split 80% Development set into 80% Train and 20% Test
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss2.split(df_dev, df_dev["label"], groups=df_dev["recording_id"]))

    df_train = df_dev.iloc[train_idx].reset_index(drop=True)
    df_test = df_dev.iloc[test_idx].reset_index(drop=True)

    # Verify recording-level isolation (zero data leakage)
    train_recs = set(df_train["recording_id"])
    test_recs = set(df_test["recording_id"])
    unseen_recs = set(df_unseen["recording_id"])

    assert len(train_recs & test_recs) == 0, "LEAKAGE: Train and Test share recordings!"
    assert len(train_recs & unseen_recs) == 0, "LEAKAGE: Train and Unseen share recordings!"
    assert len(test_recs & unseen_recs) == 0, "LEAKAGE: Test and Unseen share recordings!"

    # Save all splits
    df_train.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    df_test.to_csv(os.path.join(data_dir, "test.csv"), index=False)
    df_unseen.to_csv(os.path.join(data_dir, "unseen_holdout.csv"), index=False)
    df_dev.to_csv(os.path.join(data_dir, "dev.csv"), index=False)

    print(f"[OK] Full Dataset: {len(df)} samples")
    print(f"     |-- Development Set (80%): {len(df_dev)} samples")
    print(f"     |    |-- Train Set (80% of Dev = 64% total): {len(df_train)} samples")
    print(f"     |    +-- Test Set  (20% of Dev = 16% total): {len(df_test)} samples")
    print(f"     +-- Unseen Holdout (20% total):              {len(df_unseen)} samples")
    print(f"\n[OK] Zero recording leakage verified across all splits.")
    print(f"\nTrain class distribution:\n{df_train['label'].value_counts().to_string()}")
    print(f"\nTest class distribution:\n{df_test['label'].value_counts().to_string()}")
    print(f"\nUnseen Holdout class distribution:\n{df_unseen['label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
