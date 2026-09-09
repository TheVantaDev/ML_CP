"""
quality_check.py
Audio quality assessment for each segment or file.

Returns a QualityReport dataclass with:
  - snr_db          estimated signal-to-noise ratio
  - silence_pct     fraction of frames below energy threshold
  - clipping_pct    fraction of samples near clipping
  - is_poor_quality boolean flag
  - notes           list of warning strings
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    MIN_SNR_DB, MAX_SILENCE_PCT, MAX_CLIPPING_PCT, SAMPLE_RATE
)


@dataclass
class QualityReport:
    snr_db:           float
    silence_pct:      float
    clipping_pct:     float
    is_poor_quality:  bool
    notes:            list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "snr_db":          round(self.snr_db, 2),
            "silence_pct":     round(self.silence_pct, 4),
            "clipping_pct":    round(self.clipping_pct, 4),
            "is_poor_quality": self.is_poor_quality,
            "notes":           "; ".join(self.notes),
        }


def estimate_snr(signal: np.ndarray, sr: int = SAMPLE_RATE, percentile: float = 10.0) -> float:
    """
    Estimate SNR by comparing high-energy frames to low-energy (noise) frames.
    SNR_dB = 10 * log10(signal_power / noise_power)
    Returns 0.0 if signal is silent.
    """
    frame_size = int(0.05 * sr)   # 50 ms frames
    hop        = frame_size // 2
    n_frames   = (len(signal) - frame_size) // hop + 1
    if n_frames < 2:
        return 0.0

    powers = np.array([
        np.mean(signal[i * hop: i * hop + frame_size] ** 2)
        for i in range(n_frames)
    ])
    powers = np.where(powers < 1e-12, 1e-12, powers)  # guard log(0)

    noise_power  = np.percentile(powers, percentile)
    signal_power = np.percentile(powers, 100 - percentile)

    if noise_power < 1e-12:
        return 60.0  # very clean
    return float(10 * np.log10(signal_power / noise_power))


def estimate_silence_pct(signal: np.ndarray, sr: int = SAMPLE_RATE, threshold: float = 1e-4) -> float:
    """Fraction of 50-ms frames with RMS below threshold."""
    frame_size = int(0.05 * sr)
    hop        = frame_size // 2
    n_frames   = (len(signal) - frame_size) // hop + 1
    if n_frames == 0:
        return 1.0
    silent = sum(
        1 for i in range(n_frames)
        if np.sqrt(np.mean(signal[i * hop: i * hop + frame_size] ** 2)) < threshold
    )
    return silent / n_frames


def estimate_clipping_pct(signal: np.ndarray, clip_threshold: float = 0.99) -> float:
    """Fraction of samples at or near full-scale (clipped)."""
    return float(np.mean(np.abs(signal) >= clip_threshold))


def check_quality(
    signal: np.ndarray,
    sr: int = SAMPLE_RATE,
    min_snr: float = MIN_SNR_DB,
    max_silence: float = MAX_SILENCE_PCT,
    max_clipping: float = MAX_CLIPPING_PCT,
) -> QualityReport:
    """
    Run all quality checks and return a QualityReport.
    """
    snr      = estimate_snr(signal, sr)
    sil_pct  = estimate_silence_pct(signal, sr)
    clip_pct = estimate_clipping_pct(signal)

    notes = []
    poor  = False

    if snr < min_snr:
        notes.append(f"Low SNR: {snr:.1f} dB < {min_snr} dB")
        poor = True
    if sil_pct > max_silence:
        notes.append(f"High silence: {sil_pct * 100:.1f}% > {max_silence * 100:.0f}%")
        poor = True
    if clip_pct > max_clipping:
        notes.append(f"Clipping: {clip_pct * 100:.2f}% > {max_clipping * 100:.0f}%")
        poor = True

    return QualityReport(
        snr_db=snr,
        silence_pct=sil_pct,
        clipping_pct=clip_pct,
        is_poor_quality=poor,
        notes=notes,
    )
