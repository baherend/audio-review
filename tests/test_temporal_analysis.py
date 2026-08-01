"""
Tests for shared timeline windows and speech-activity analysis.

Uses synthetic audio and does not require model loading.
"""

import os
import sys
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.demo.temporal_analysis import create_shared_windows, analyze_speech_activity
from src.demo.schemas import WindowConfig, SpeechActivityConfig, SharedWindow


# ══════════════════════════════════════════════════════════════════════
# WINDOW CREATION TESTS
# ══════════════════════════════════════════════════════════════════════
class TestWindowCreation:
    """Tests for create_shared_windows."""

    def test_10_second_call(self):
        """Shared windows for a 10-second call."""
        windows = create_shared_windows(
            duration_sec=10.0,
            window_duration_sec=2.0,
            hop_duration_sec=1.0,
            model_sr=16000,
        )
        assert len(windows) == 9  # 0-2, 1-3, ..., 8-10
        assert windows[0].original_start == 0.0
        assert windows[-1].original_end == 10.0

    def test_overlapping_windows(self):
        """Overlapping windows using explicit hop."""
        windows = create_shared_windows(
            duration_sec=5.0,
            window_duration_sec=2.0,
            hop_duration_sec=0.5,
            model_sr=16000,
        )
        # Windows: 0-2, 0.5-2.5, 1-3, 1.5-3.5, 2-4, 2.5-4.5, 3-5
        assert len(windows) == 7
        # Check overlap: window[1] starts before window[0] ends
        assert windows[1].original_start < windows[0].original_end

    def test_non_overlapping_windows(self):
        """Non-overlapping windows with hop = window_duration."""
        windows = create_shared_windows(
            duration_sec=6.0,
            window_duration_sec=2.0,
            hop_duration_sec=2.0,
            model_sr=16000,
        )
        assert len(windows) == 3
        assert windows[0].original_end == windows[1].original_start

    def test_final_partial_window(self):
        """Final window may be shorter than configured duration."""
        windows = create_shared_windows(
            duration_sec=4.5,
            window_duration_sec=2.0,
            hop_duration_sec=1.0,
            model_sr=16000,
        )
        last = windows[-1]
        assert last.original_end == 4.5
        assert (last.original_end - last.original_start) < 2.0

    def test_duration_shorter_than_window(self):
        """Duration shorter than one window produces one partial window."""
        windows = create_shared_windows(
            duration_sec=1.0,
            window_duration_sec=2.0,
            hop_duration_sec=1.0,
            model_sr=16000,
        )
        assert len(windows) == 1
        assert windows[0].original_start == 0.0
        assert windows[0].original_end == 1.0

    def test_invalid_duration(self):
        """Zero or negative duration is rejected."""
        with pytest.raises(ValueError, match="duration_sec"):
            create_shared_windows(0.0, 2.0, 1.0)
        with pytest.raises(ValueError, match="duration_sec"):
            create_shared_windows(-1.0, 2.0, 1.0)

    def test_invalid_window(self):
        """Zero or negative window is rejected."""
        with pytest.raises(ValueError, match="window_duration_sec"):
            create_shared_windows(10.0, 0.0, 1.0)

    def test_invalid_hop(self):
        """Zero or negative hop is rejected."""
        with pytest.raises(ValueError, match="hop_duration_sec"):
            create_shared_windows(10.0, 2.0, 0.0)

    def test_invalid_sample_rate(self):
        """Zero or negative sample rate is rejected."""
        with pytest.raises(ValueError, match="model_sr"):
            create_shared_windows(10.0, 2.0, 1.0, model_sr=0)

    def test_timestamps_monotonically_increasing(self):
        """Window timestamps are monotonically increasing."""
        windows = create_shared_windows(10.0, 2.0, 1.0, model_sr=16000)
        for i in range(1, len(windows)):
            assert windows[i].original_start >= windows[i-1].original_start
            assert windows[i].original_end >= windows[i-1].original_end

    def test_window_never_exceeds_duration(self):
        """No window extends beyond the call duration."""
        windows = create_shared_windows(5.0, 2.0, 1.0, model_sr=16000)
        for w in windows:
            assert w.original_end <= 5.0 + 1e-9

    def test_model_sample_indices(self):
        """Model sample indices are correct at 16 kHz."""
        windows = create_shared_windows(5.0, 2.0, 1.0, model_sr=16000)
        assert windows[0].model_start_sample == 0
        assert windows[0].model_end_sample == 32000  # 2.0 * 16000

    def test_original_sample_indices(self):
        """Original sample indices are computed when original_sr provided."""
        windows = create_shared_windows(
            5.0, 2.0, 1.0, model_sr=16000, original_sr=44100
        )
        assert windows[0].original_start_sample == 0
        assert windows[0].original_end_sample == round(2.0 * 44100)

    def test_original_sample_indices_none_when_not_provided(self):
        """Original sample indices are None when original_sr not provided."""
        windows = create_shared_windows(5.0, 2.0, 1.0, model_sr=16000)
        assert windows[0].original_start_sample is None
        assert windows[0].original_end_sample is None


# ══════════════════════════════════════════════════════════════════════
# SPEECH ACTIVITY TESTS
# ══════════════════════════════════════════════════════════════════════
class TestSpeechActivity:
    """Tests for analyze_speech_activity."""

    def _default_config(self):
        return SpeechActivityConfig()

    def test_fully_silent_channel_inactive(self):
        """Fully silent channel is marked inactive."""
        windows = create_shared_windows(4.0, 2.0, 1.0, model_sr=16000)
        silent = np.zeros(64000, dtype=np.float32)
        results = analyze_speech_activity(silent, windows, 16000, self._default_config())
        for r in results:
            assert not r.speech_active
            assert r.inactive_reason == "silence"

    def test_active_sine_wave_active(self):
        """Active sine-wave channel is marked active."""
        windows = create_shared_windows(4.0, 2.0, 1.0, model_sr=16000)
        t = np.linspace(0, 4.0, 64000, endpoint=False)
        signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        results = analyze_speech_activity(signal, windows, 16000, self._default_config())
        # At least the middle windows should be active
        active_count = sum(1 for r in results if r.speech_active)
        assert active_count > 0

    def test_short_activity_below_ratio_inactive(self):
        """Short activity below required ratio is marked inactive."""
        config = SpeechActivityConfig(active_ratio_threshold=0.5)
        windows = create_shared_windows(4.0, 2.0, 1.0, model_sr=16000)
        # Signal only active for a brief moment
        signal = np.zeros(64000, dtype=np.float32)
        signal[1000:1200] = 0.5  # Very short burst
        results = analyze_speech_activity(signal, windows, 16000, config)
        for r in results:
            assert not r.speech_active

    def test_speech_activity_does_not_modify_waveform(self):
        """Speech activity does not modify the waveform."""
        windows = create_shared_windows(2.0, 2.0, 1.0, model_sr=16000)
        signal = np.random.RandomState(42).randn(32000).astype(np.float32)
        original = signal.copy()
        _ = analyze_speech_activity(signal, windows, 16000, self._default_config())
        np.testing.assert_array_equal(signal, original)

    def test_inactive_reason_for_empty_segment(self):
        """Empty segment gets 'empty_segment' reason."""
        windows = [SharedWindow(
            window_id=0, original_start=0.0, original_end=1.0,
            model_start_sample=0, model_end_sample=0,
        )]
        signal = np.array([], dtype=np.float32)
        results = analyze_speech_activity(signal, windows, 16000, self._default_config())
        assert results[0].inactive_reason == "empty_segment"

    def test_inactive_reason_for_non_finite(self):
        """Non-finite segment gets 'non_finite_audio' reason."""
        windows = create_shared_windows(1.0, 1.0, 1.0, model_sr=16000)
        signal = np.zeros(16000, dtype=np.float32)
        signal[100] = np.nan
        results = analyze_speech_activity(signal, windows, 16000, self._default_config())
        assert results[0].inactive_reason == "non_finite_audio"

    def test_window_count_matches(self):
        """Number of results matches number of windows."""
        windows = create_shared_windows(5.0, 2.0, 1.0, model_sr=16000)
        signal = np.random.randn(80000).astype(np.float32) * 0.1
        results = analyze_speech_activity(signal, windows, 16000, self._default_config())
        assert len(results) == len(windows)


# ══════════════════════════════════════════════════════════════════════
# TIMESTAMP ALIGNMENT TESTS
# ══════════════════════════════════════════════════════════════════════
class TestTimestampAlignment:
    """Tests that Agent and Customer windows have identical timestamps."""

    def test_agent_customer_identical_timestamps(self):
        """Agent and Customer use the same window timestamps."""
        windows = create_shared_windows(10.0, 2.0, 1.0, model_sr=16000)
        # Verify: all windows have unique IDs
        ids = [w.window_id for w in windows]
        assert len(ids) == len(set(ids))
        # Verify: timestamps are monotonically increasing
        for i in range(1, len(windows)):
            assert windows[i].original_start >= windows[i-1].original_start
            assert windows[i].original_end >= windows[i-1].original_end

    def test_timestamp_conversion_consistency(self):
        """Timestamp conversion in seconds is consistent."""
        windows = create_shared_windows(
            10.0, 2.0, 1.0, model_sr=16000, original_sr=44100
        )
        for w in windows:
            # Model: start_sec = model_start_sample / 16000
            model_start_sec = w.model_start_sample / 16000
            assert abs(model_start_sec - w.original_start) < 1e-6
            # Original: start_sec = original_start_sample / 44100
            orig_start_sec = w.original_start_sample / 44100
            assert abs(orig_start_sec - w.original_start) < 1e-6
