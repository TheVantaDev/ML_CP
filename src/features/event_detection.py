"""
event_detection.py
Classical signal-processing algorithms to detect candidate acoustic events
within a 4-second audio segment.

No ML is used here — only energy thresholding, FFT peak finding,
and tonal persistence checks.

Output: EventReport dataclass with event flag and metadata.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import (
    SAMPLE_RATE, N_FFT, HOP_LENGTH, TOP_K_PEAKS, TONAL_PERSIST_FRAMES
)


@dataclass
class EventReport:
    has_event:           bool
    peak_frequencies:    list[float]   # top-K dominant frequencies (Hz)
    peak_amplitudes:     list[float]   # corresponding amplitudes (dB)
    tonal_peak_count:    int           # number of persistently tonal peaks
    tonal_persistence:   float         # mean persistence (fraction of frames)
    dominant_frequency:  float         # single strongest frequency (Hz)
    snr_estimate:        float         # rough SNR from spectral analysis
    freq_spread:         float         # spread of energy across spectrum (Hz)

    def to_dict(self) -> dict:
        return {
            "has_event":          self.has_event,
            "dominant_frequency": round(self.dominant_frequency, 2),
            "tonal_peak_count":   self.tonal_peak_count,
            "tonal_persistence":  round(self.tonal_persistence, 4),
            "snr_estimate":       round(self.snr_estimate, 2),
            "freq_spread":        round(self.freq_spread, 2),
            "peak_frequencies":   [round(f, 1) for f in self.peak_frequencies],
            "peak_amplitudes":    [round(a, 2) for a in self.peak_amplitudes],
        }


def detect_events(
    signal: np.ndarray,
    sr:     int   = SAMPLE_RATE,
    n_fft:  int   = N_FFT,
    hop:    int   = HOP_LENGTH,
    top_k:  int   = TOP_K_PEAKS,
    persist_frames: int = TONAL_PERSIST_FRAMES,
) -> EventReport:
    """
    Run classical event detection on a single audio segment.

    Args:
        signal:         1-D float32 audio array (pre-processed)
        sr:             sample rate
        n_fft:          FFT window size
        hop:            hop length for STFT
        top_k:          number of top frequency peaks to track
        persist_frames: minimum frames a peak must persist to be "tonal"

    Returns:
        EventReport
    """
    # ── Short-time Fourier transform ────────────────────────────────────────────
    S = np.abs(np.fft.rfft(signal * np.hanning(len(signal)), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    # ── Averaged power spectrum (STFT over frames) ──────────────────────────────
    n_samples  = len(signal)
    n_frames   = max(1, (n_samples - n_fft) // hop + 1)
    spec_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        frame = signal[i * hop: i * hop + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        window = np.hanning(n_fft)
        spec_matrix[:, i] = np.abs(np.fft.rfft(frame * window))

    mean_spectrum = spec_matrix.mean(axis=1)   # shape: (n_fft//2+1,)

    # ── Top-K peak frequencies ──────────────────────────────────────────────────
    # Avoid DC bin
    search_spectrum = mean_spectrum.copy()
    search_spectrum[0] = 0.0

    peaks, props = find_peaks(search_spectrum, distance=max(1, n_fft // 200), height=0)
    if len(peaks) == 0:
        peaks = np.array([np.argmax(search_spectrum)])
        props = {"peak_heights": np.array([search_spectrum[peaks[0]]])}

    # Sort by amplitude descending
    order = np.argsort(props["peak_heights"])[::-1]
    peaks_sorted = peaks[order]

    top_peaks = peaks_sorted[:top_k]
    peak_freqs  = [float(freqs[p]) for p in top_peaks]
    peak_amps_db = [float(20 * np.log10(mean_spectrum[p] + 1e-9)) for p in top_peaks]

    dominant_freq = peak_freqs[0] if peak_freqs else 0.0

    # ── Tonal persistence ───────────────────────────────────────────────────────
    # For each top peak, count how many frames have energy within ±2 bins
    tonal_counts = []
    for p in top_peaks:
        lo = max(0, p - 2)
        hi = min(spec_matrix.shape[0], p + 3)
        # frame is "active" for this peak if its local energy exceeds median
        band_energy = spec_matrix[lo:hi, :].sum(axis=0)
        median_e = np.median(band_energy)
        active_frames = int(np.sum(band_energy > median_e * 1.5))
        tonal_counts.append(active_frames / n_frames)

    tonal_persistence   = float(np.mean(tonal_counts)) if tonal_counts else 0.0
    tonal_peak_count    = int(sum(1 for c in tonal_counts if c >= persist_frames / n_frames))

    # ── SNR estimate from spectral peaks vs floor ───────────────────────────────
    S_db = 20 * np.log10(mean_spectrum + 1e-9)
    noise_floor_db = float(np.percentile(S_db, 15))
    signal_peak_db = float(np.percentile(S_db, 90))
    snr_est = max(0.0, signal_peak_db - noise_floor_db)

    # ── Frequency spread (spectral bandwidth above median) ───────────────────────
    threshold = np.median(mean_spectrum) * 2
    active_bins = freqs[mean_spectrum > threshold]
    freq_spread = float(active_bins.max() - active_bins.min()) if len(active_bins) > 1 else 0.0

    # ── Event flag ──────────────────────────────────────────────────────────────
    # Flag as "event" if there are persistent tonal peaks or high SNR
    has_event = (tonal_peak_count >= 1 and snr_est >= 5.0) or snr_est >= 10.0

    return EventReport(
        has_event=has_event,
        peak_frequencies=peak_freqs,
        peak_amplitudes=peak_amps_db,
        tonal_peak_count=tonal_peak_count,
        tonal_persistence=tonal_persistence,
        dominant_frequency=dominant_freq,
        snr_estimate=snr_est,
        freq_spread=freq_spread,
    )
