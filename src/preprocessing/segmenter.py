"""
segmenter.py
Sliding-window segmentation of audio clips.

Segments a long recording into fixed-length windows, discards
silent/low-energy segments, and returns a list of (start_sec, end_sec, array).
"""

import sys
from pathlib import Path
from typing import Generator

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    SAMPLE_RATE, SEGMENT_DURATION, OVERLAP_FRACTION,
    MIN_SEGMENT_ENERGY, MAX_SILENCE_PCT
)


def segment_signal(
    signal: np.ndarray,
    sr: int = SAMPLE_RATE,
    duration: float = SEGMENT_DURATION,
    overlap: float = OVERLAP_FRACTION,
    min_energy: float = MIN_SEGMENT_ENERGY,
) -> list[tuple[float, float, np.ndarray]]:
    """
    Split signal into overlapping windows.

    Args:
        signal:     1-D float32 audio array
        sr:         sample rate
        duration:   window length in seconds
        overlap:    fraction of window to overlap (0.0–0.99)
        min_energy: discard windows with RMS below this threshold

    Returns:
        List of (start_sec, end_sec, window_array) tuples for non-silent windows.
    """
    win_samples  = int(duration * sr)
    hop_samples  = int(win_samples * (1 - overlap))

    if len(signal) < win_samples:
        # Short clip: pad with zeros and return as single segment
        padded = np.zeros(win_samples, dtype=np.float32)
        padded[:len(signal)] = signal
        rms = np.sqrt(np.mean(padded ** 2))
        if rms >= min_energy:
            return [(0.0, duration, padded)]
        return []

    segments = []
    for start in range(0, len(signal) - win_samples + 1, hop_samples):
        window = signal[start: start + win_samples].copy()
        rms    = np.sqrt(np.mean(window ** 2))
        if rms < min_energy:
            continue  # silence — skip
        start_sec = start / sr
        end_sec   = (start + win_samples) / sr
        segments.append((start_sec, end_sec, window))

    return segments


def segment_file(
    path: str | Path,
    sr: int = SAMPLE_RATE,
    duration: float = SEGMENT_DURATION,
    overlap: float = OVERLAP_FRACTION,
    apply_preprocess: bool = True,
) -> list[tuple[float, float, np.ndarray]]:
    """
    Load, optionally preprocess, and segment an audio file.
    Returns list of (start_sec, end_sec, window_array).
    """
    if apply_preprocess:
        from src.preprocessing.audio_utils import preprocess
        signal, sr = preprocess(path, sr=sr)
    else:
        from src.preprocessing.audio_utils import load_audio
        signal, sr = load_audio(path, sr=sr)

    return segment_signal(signal, sr=sr, duration=duration, overlap=overlap)
