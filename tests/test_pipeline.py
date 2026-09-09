"""
tests/test_pipeline.py
Basic smoke tests for the Sonobuoy Sound Detective pipeline.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import SAMPLE_RATE, SEGMENT_DURATION


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _make_sine(freq=200, duration=4.0, sr=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _make_noise(duration=4.0, sr=SAMPLE_RATE):
    rng = np.random.default_rng(42)
    return rng.standard_normal(int(sr * duration)).astype(np.float32) * 0.1


# ── Synthetic generator tests ────────────────────────────────────────────────────

def test_synthetic_vessel():
    from src.data.synthetic import generate_vessel_like
    signal, meta = generate_vessel_like(duration=4.0, sr=SAMPLE_RATE)
    assert signal.dtype == np.float32
    assert len(signal) == int(SAMPLE_RATE * 4.0)
    assert "fundamental_hz" in meta
    assert np.max(np.abs(signal)) <= 1.01


def test_synthetic_ambient():
    from src.data.synthetic import generate_ambient
    signal, meta = generate_ambient(duration=4.0, sr=SAMPLE_RATE)
    assert signal.dtype == np.float32
    assert len(signal) == int(SAMPLE_RATE * 4.0)


def test_synthetic_unknown():
    from src.data.synthetic import generate_unknown
    signal, meta = generate_unknown(duration=4.0, sr=SAMPLE_RATE)
    assert signal.dtype == np.float32


# ── Preprocessing tests ──────────────────────────────────────────────────────────

def test_mono_conversion():
    from src.preprocessing.audio_utils import to_mono
    stereo = np.random.randn(2, 22050).astype(np.float32)
    mono = to_mono(stereo)
    assert mono.ndim == 1
    assert len(mono) == 22050


def test_bandpass_filter():
    from src.preprocessing.audio_utils import bandpass_filter
    noise = _make_noise()
    filtered = bandpass_filter(noise, sr=SAMPLE_RATE)
    assert filtered.shape == noise.shape
    assert filtered.dtype == np.float32


def test_rms_normalize():
    from src.preprocessing.audio_utils import rms_normalize
    signal = _make_sine()
    normed = rms_normalize(signal, target_rms=0.1)
    rms = float(np.sqrt(np.mean(normed ** 2)))
    assert abs(rms - 0.1) < 0.01


def test_dc_removal():
    from src.preprocessing.audio_utils import remove_dc_offset
    signal = _make_sine() + 0.5   # add DC
    out = remove_dc_offset(signal)
    assert abs(out.mean()) < 0.01


# ── Segmenter tests ──────────────────────────────────────────────────────────────

def test_segment_normal_signal():
    from src.preprocessing.segmenter import segment_signal
    signal = _make_sine(duration=12.0)
    segs = segment_signal(signal, sr=SAMPLE_RATE, duration=4.0, overlap=0.5)
    assert len(segs) > 0
    for start, end, window in segs:
        assert end > start
        assert len(window) == int(SAMPLE_RATE * 4.0)


def test_segment_short_clip():
    from src.preprocessing.segmenter import segment_signal
    signal = _make_sine(duration=2.0)   # shorter than segment duration
    segs = segment_signal(signal, sr=SAMPLE_RATE, duration=4.0)
    assert len(segs) <= 1   # at most 1 padded segment


def test_segment_silent_discarded():
    from src.preprocessing.segmenter import segment_signal
    silent = np.zeros(int(SAMPLE_RATE * 8.0), dtype=np.float32)
    segs = segment_signal(silent, sr=SAMPLE_RATE)
    assert len(segs) == 0


# ── Quality check tests ──────────────────────────────────────────────────────────

def test_quality_check_clean():
    from src.preprocessing.quality_check import check_quality
    signal = _make_sine(freq=200)
    report = check_quality(signal, sr=SAMPLE_RATE)
    assert report.snr_db >= 0
    assert 0 <= report.silence_pct <= 1.0
    assert 0 <= report.clipping_pct <= 1.0


# ── Feature extractor tests ──────────────────────────────────────────────────────

def test_feature_extractor_output_shape():
    from src.features.extractor import extract_features
    signal = _make_sine(freq=150)
    features = extract_features(signal, sr=SAMPLE_RATE, clip_id="test001", label="vessel_like")
    # Must have numeric values for all feature keys (except metadata strings)
    numeric_keys = [k for k, v in features.items()
                    if k not in ("clip_id", "label", "split")]
    assert len(numeric_keys) > 50, "Expected at least 50 features"
    for k in numeric_keys:
        assert isinstance(features[k], (int, float)), f"Feature {k} is not numeric"


def test_feature_extractor_ambient():
    from src.features.extractor import extract_features
    signal = _make_noise()
    features = extract_features(signal, sr=SAMPLE_RATE, label="ambient")
    assert "spectral_centroid_mean" in features
    assert "mfcc_1_mean" in features


# ── Event detection tests ────────────────────────────────────────────────────────

def test_event_detection_tonal():
    from src.features.event_detection import detect_events
    signal = _make_sine(freq=200)  # clear tonal
    report = detect_events(signal, sr=SAMPLE_RATE)
    assert report.dominant_frequency > 0
    assert report.snr_estimate >= 0


def test_event_detection_noise():
    from src.features.event_detection import detect_events
    signal = _make_noise()
    report = detect_events(signal, sr=SAMPLE_RATE)
    assert isinstance(report.has_event, bool)
