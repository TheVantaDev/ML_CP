"""
extractor.py
Handcrafted acoustic feature extraction for the Sonobuoy Sound Detective.

Produces a structured numeric feature vector (~100 features) per audio segment:
  - MFCC features (13 coeffs x {mean, std, min, max} + delta)
  - Energy features
  - Spectral features (centroid, bandwidth, rolloff, flatness, contrast)
  - Frequency features (dominant freq, top peaks)
  - Temporal features (ZCR, attack/decay)
  - Tonal features (persistent peaks, tonal-to-broadband ratio)
  - Quality features (SNR, silence %, clipping %)

All frame-level features are summarized into statistics before being
returned as a flat dict suitable for a Pandas DataFrame row.
"""

import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    SAMPLE_RATE, N_MFCC, N_FFT, HOP_LENGTH, N_MELS, TOP_K_PEAKS
)
from src.features.event_detection import detect_events
from src.preprocessing.quality_check import check_quality


def _stats(arr: np.ndarray, prefix: str) -> dict:
    """Return {prefix_mean, prefix_std, prefix_min, prefix_max, prefix_median}."""
    return {
        f"{prefix}_mean":   float(np.mean(arr)),
        f"{prefix}_std":    float(np.std(arr)),
        f"{prefix}_min":    float(np.min(arr)),
        f"{prefix}_max":    float(np.max(arr)),
        f"{prefix}_median": float(np.median(arr)),
    }


def extract_features(
    signal:   np.ndarray,
    sr:       int = SAMPLE_RATE,
    clip_id:  str = "",
    label:    str = "",
    split:    str = "",
) -> dict:
    """
    Extract the full handcrafted feature vector from one audio segment.

    Args:
        signal:   1-D float32, already preprocessed
        sr:       sample rate
        clip_id:  optional identifier (propagated to output)
        label:    ground-truth label (propagated to output)
        split:    train/val/test (propagated to output)

    Returns:
        Flat dict of feature_name -> float value.
    """
    features: dict = {}

    # ── Metadata passthrough ────────────────────────────────────────────────────
    features["clip_id"] = clip_id
    features["label"]   = label
    features["split"]   = split

    # ── MFCC (13 coefficients) ──────────────────────────────────────────────────
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    for i in range(N_MFCC):
        features.update(_stats(mfccs[i], f"mfcc_{i+1}"))

    # Delta MFCC (first-order derivative)
    delta_mfccs = librosa.feature.delta(mfccs)
    for i in range(N_MFCC):
        features.update(_stats(delta_mfccs[i], f"delta_mfcc_{i+1}"))

    # ── Energy features ─────────────────────────────────────────────────────────
    rms_frames = librosa.feature.rms(y=signal, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    features.update(_stats(rms_frames, "rms"))
    features["dynamic_range"] = float(rms_frames.max() - rms_frames.min())

    # ── Spectral features ───────────────────────────────────────────────────────
    centroid  = librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    rolloff   = librosa.feature.spectral_rolloff(y=signal, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    flatness  = librosa.feature.spectral_flatness(y=signal, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]

    features.update(_stats(centroid,  "spectral_centroid"))
    features.update(_stats(bandwidth, "spectral_bandwidth"))
    features.update(_stats(rolloff,   "spectral_rolloff"))
    features.update(_stats(flatness,  "spectral_flatness"))

    # Spectral contrast (6 bands + 1 = 7 sub-arrays)
    contrast = librosa.feature.spectral_contrast(y=signal, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    for band_i in range(contrast.shape[0]):
        features.update(_stats(contrast[band_i], f"spectral_contrast_band{band_i+1}"))

    # ── Temporal features ───────────────────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(signal, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    features.update(_stats(zcr, "zcr"))

    # Attack: how fast signal reaches its peak (first 25% of RMS frames)
    n_quarter = max(1, len(rms_frames) // 4)
    features["attack_rms_mean"]  = float(rms_frames[:n_quarter].mean())
    features["decay_rms_mean"]   = float(rms_frames[-n_quarter:].mean())
    features["attack_decay_ratio"] = float(
        features["attack_rms_mean"] / (features["decay_rms_mean"] + 1e-9)
    )

    # ── Tonal / frequency features ──────────────────────────────────────────────
    event = detect_events(signal, sr=sr)
    features["dominant_frequency"]  = event.dominant_frequency
    features["tonal_peak_count"]    = float(event.tonal_peak_count)
    features["tonal_persistence"]   = event.tonal_persistence
    features["freq_spread"]         = event.freq_spread
    features["has_event"]           = float(event.has_event)

    for k, fq in enumerate(event.peak_frequencies[:TOP_K_PEAKS]):
        features[f"peak_freq_{k+1}"] = fq
    for k, amp in enumerate(event.peak_amplitudes[:TOP_K_PEAKS]):
        features[f"peak_amp_{k+1}"] = amp
    # Pad if fewer than TOP_K peaks
    for k in range(len(event.peak_frequencies), TOP_K_PEAKS):
        features[f"peak_freq_{k+1}"] = 0.0
        features[f"peak_amp_{k+1}"]  = -120.0

    # Tonal-to-broadband ratio: ratio of energy near peaks vs. total energy
    S_power = np.abs(librosa.stft(signal, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
    total_energy = S_power.sum()
    if total_energy > 0 and event.peak_frequencies:
        freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
        tonal_energy = 0.0
        for pf in event.peak_frequencies:
            bin_idx = np.argmin(np.abs(freqs - pf))
            lo = max(0, bin_idx - 3)
            hi = min(S_power.shape[0], bin_idx + 4)
            tonal_energy += S_power[lo:hi, :].sum()
        features["tonal_broadband_ratio"] = float(tonal_energy / total_energy)
    else:
        features["tonal_broadband_ratio"] = 0.0

    # ── Quality features ────────────────────────────────────────────────────────
    quality = check_quality(signal, sr=sr)
    features["snr_estimate"]  = quality.snr_db
    features["silence_pct"]   = quality.silence_pct
    features["clipping_pct"]  = quality.clipping_pct
    features["is_poor_quality"] = float(quality.is_poor_quality)

    return features
