"""
audio_utils.py
Core audio preprocessing utilities for the Sonobuoy Sound Detective project.

Pipeline per clip:
  1. Load audio (any format)
  2. Convert to mono
  3. Resample to target SR
  4. Remove DC offset
  5. Apply band-pass Butterworth filter
  6. Spectral noise gating (optional, light)
  7. RMS normalisation

All functions are pure (return new array, do not modify input).
"""

import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    SAMPLE_RATE, BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ, BANDPASS_ORDER
)


# ─── Load ──────────────────────────────────────────────────────────────────────

def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """
    Load an audio file and return (signal_float32, sample_rate).
    Handles WAV, FLAC, MP3, OGG via librosa / soundfile.
    Always returns mono, float32.
    """
    signal, native_sr = librosa.load(str(path), sr=None, mono=True)
    if native_sr != sr:
        signal = librosa.resample(signal, orig_sr=native_sr, target_sr=sr)
    return signal.astype(np.float32), sr


# ─── Individual operations ─────────────────────────────────────────────────────

def to_mono(signal: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono by averaging channels."""
    if signal.ndim == 2:
        return signal.mean(axis=0).astype(np.float32)
    return signal.astype(np.float32)


def resample(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return signal
    return librosa.resample(signal, orig_sr=orig_sr, target_sr=target_sr)


def remove_dc_offset(signal: np.ndarray) -> np.ndarray:
    """Subtract mean to centre the waveform around zero."""
    return (signal - signal.mean()).astype(np.float32)


def bandpass_filter(
    signal: np.ndarray,
    sr: int = SAMPLE_RATE,
    low_hz: float = BANDPASS_LOW_HZ,
    high_hz: float = BANDPASS_HIGH_HZ,
    order: int = BANDPASS_ORDER,
) -> np.ndarray:
    """Apply a Butterworth band-pass filter."""
    nyq = sr / 2.0
    low  = max(low_hz,  1.0) / nyq   # guard against 0
    high = min(high_hz, nyq - 1.0) / nyq
    if low >= high:
        return signal
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfilt(sos, signal).astype(np.float32)


def spectral_noise_gate(
    signal: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = 2048,
    hop: int = 512,
    percentile: float = 15.0,
    reduction_db: float = 12.0,
) -> np.ndarray:
    """
    Light spectral gating: estimate noise floor from the quietest frames,
    then attenuate bins below the noise floor.
    Does NOT remove the noise entirely — just reduces it gently.
    """
    S = librosa.stft(signal, n_fft=n_fft, hop_length=hop)
    mag, phase = np.abs(S), np.angle(S)

    # Estimate noise profile from the lowest-energy percentile of frames
    frame_energy = mag.mean(axis=0)
    noise_thresh_idx = frame_energy < np.percentile(frame_energy, percentile)
    if noise_thresh_idx.any():
        noise_profile = mag[:, noise_thresh_idx].mean(axis=1, keepdims=True)
    else:
        noise_profile = mag.min(axis=1, keepdims=True)

    reduction = librosa.db_to_amplitude(-reduction_db)
    mask = (mag > noise_profile * (1 / reduction)).astype(np.float32)
    # Smooth mask slightly across time
    from scipy.ndimage import uniform_filter1d
    mask = uniform_filter1d(mask, size=3, axis=1)

    mag_gated = mag * mask
    S_gated = mag_gated * np.exp(1j * phase)
    return librosa.istft(S_gated, hop_length=hop, length=len(signal)).astype(np.float32)


def rms_normalize(signal: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Scale signal so its RMS equals target_rms."""
    current_rms = np.sqrt(np.mean(signal ** 2))
    if current_rms < 1e-9:
        return signal
    return (signal * (target_rms / current_rms)).astype(np.float32)


# ─── Full pipeline ─────────────────────────────────────────────────────────────

def preprocess(
    path: str | Path,
    sr: int = SAMPLE_RATE,
    apply_bandpass: bool = True,
    apply_noise_gate: bool = True,
    apply_normalize: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Full preprocessing pipeline: load -> mono -> resample -> DC remove
    -> bandpass -> noise gate -> RMS normalize.

    Returns:
        (signal, sr)
    """
    signal, sr = load_audio(path, sr=sr)
    signal = remove_dc_offset(signal)
    if apply_bandpass:
        signal = bandpass_filter(signal, sr=sr)
    if apply_noise_gate:
        try:
            signal = spectral_noise_gate(signal, sr=sr)
        except Exception:
            pass  # Non-fatal; continue without gating
    if apply_normalize:
        signal = rms_normalize(signal)
    return signal, sr
