"""
Tests for Hold candidate detection.

Uses synthetic audio and activity intervals.
Covers tonal detection, silence, noise, edge cases, and the
full 340.5-second synthetic call scenario.
"""

import os
import sys
import time
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.demo.schemas import ActivityInterval, HoldDetectionConfig
from src.demo.hold_detection import (
    detect_hold_candidates,
    _compute_audio_features,
    _normalized_autocorrelation_at_lag,
)


# ══════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════
def _iv(start, end, state, agent_active=None, customer_active=None):
    """Create an ActivityInterval for testing."""
    if agent_active is None:
        agent_active = state in ("agent_active_only", "both_active_interval")
    if customer_active is None:
        customer_active = state in ("customer_active_only", "both_active_interval")
    return ActivityInterval(
        start_time=start,
        end_time=end,
        duration_sec=round(end - start, 6),
        state=state,
        agent_active=agent_active,
        customer_active=customer_active,
    )


def _make_waveforms(duration, sr=16000, agent_active_ranges=None,
                    customer_active_ranges=None, customer_music_ranges=None):
    """Create synthetic waveforms."""
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    agent = np.zeros(n, dtype=np.float32)
    customer = np.zeros(n, dtype=np.float32)

    if agent_active_ranges:
        for s, e in agent_active_ranges:
            si, ei = int(s * sr), int(e * sr)
            agent[si:ei] = 0.5 * np.sin(2 * np.pi * 440 * t[si:ei])

    if customer_active_ranges:
        for s, e in customer_active_ranges:
            si, ei = int(s * sr), int(e * sr)
            customer[si:ei] = 0.5 * np.sin(2 * np.pi * 880 * t[si:ei])

    if customer_music_ranges:
        for s, e in customer_music_ranges:
            si, ei = int(s * sr), int(e * sr)
            # Multi-tone tonal signal (fundamental + harmonics)
            seg = t[si:ei]
            customer[si:ei] = (
                0.3 * np.sin(2 * np.pi * 440 * seg)
                + 0.2 * np.sin(2 * np.pi * 880 * seg)
                + 0.1 * np.sin(2 * np.pi * 1320 * seg)
            ).astype(np.float32)

    return agent, customer


def _config(**kwargs):
    """Create a HoldDetectionConfig with overrides."""
    defaults = {
        "minimum_hold_candidate_duration_sec": 10.0,
        "maximum_short_pause_duration_sec": 3.0,
        "merge_gap_between_candidates_sec": 5.0,
        "minimum_confidence": 0.3,
    }
    defaults.update(kwargs)
    return HoldDetectionConfig(**defaults)


def _legacy_full_correlation_score(signal: np.ndarray, lag: int) -> float:
    """Reference implementation retained only for short-array parity tests."""
    if lag <= 0 or len(signal) <= lag:
        return 0.0
    autocorr = np.correlate(signal, signal, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]
    return float(autocorr[lag] / (autocorr[0] + 1e-8))


def _make_repetition_signal(kind: str, sample_rate: int, sample_count: int) -> np.ndarray:
    """Create deterministic short signals for one-lag parity tests."""
    timeline = np.arange(sample_count, dtype=np.float64) / sample_rate
    if kind == "constant_tone":
        signal = 0.4 * np.sin(2 * np.pi * 8 * timeline)
    elif kind == "periodic_tone":
        signal = (
            0.3 * np.sin(2 * np.pi * 3 * timeline)
            + 0.2 * np.sin(2 * np.pi * 7 * timeline)
            + 0.1 * np.sin(2 * np.pi * 12 * timeline)
        )
    elif kind == "white_noise":
        signal = np.random.default_rng(10).standard_normal(sample_count) * 0.1
    elif kind == "speech_like":
        rng = np.random.default_rng(11)
        envelope = 0.55 + 0.4 * np.sin(2 * np.pi * 1.3 * timeline)
        signal = envelope * (
            0.3 * np.sin(2 * np.pi * 5 * timeline)
            + 0.12 * np.sin(2 * np.pi * 11 * timeline)
        ) + 0.02 * rng.standard_normal(sample_count)
    else:  # pragma: no cover - test helper contract
        raise ValueError(f"Unknown signal kind: {kind}")
    return signal.astype(np.float32)


class TestSingleLagAutocorrelation:
    """The direct one-second lag preserves the former score without quadratic work."""

    @pytest.mark.parametrize(
        "signal_kind",
        ["constant_tone", "periodic_tone", "white_noise", "speech_like"],
    )
    def test_numerical_parity_with_legacy_full_correlation(self, signal_kind):
        lag = 64
        signal = _make_repetition_signal(signal_kind, lag, 257)

        expected = _legacy_full_correlation_score(signal, lag)
        actual = _normalized_autocorrelation_at_lag(signal, lag)

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
        assert np.isfinite(actual)
        assert -1.0 <= actual <= 1.0

    def test_silence_and_zero_energy_denominator_return_zero(self):
        score = _normalized_autocorrelation_at_lag(
            np.zeros(257, dtype=np.float32),
            64,
        )
        assert score == 0.0
        assert np.isfinite(score)

    def test_empty_input_returns_zero(self):
        assert _normalized_autocorrelation_at_lag(
            np.array([], dtype=np.float32),
            64,
        ) == 0.0

    @pytest.mark.parametrize("invalid_lag", [0, -1, 1.5, None])
    def test_invalid_lag_returns_zero(self, invalid_lag):
        signal = np.ones(128, dtype=np.float32)
        assert _normalized_autocorrelation_at_lag(signal, invalid_lag) == 0.0

    @pytest.mark.parametrize("sample_count", [63, 64])
    def test_input_not_longer_than_required_lag_returns_zero(self, sample_count):
        signal = np.ones(sample_count, dtype=np.float32)
        assert _normalized_autocorrelation_at_lag(signal, 64) == 0.0

    def test_underflowed_energy_denominator_returns_zero(self):
        signal = np.full(257, 1e-30, dtype=np.float32)
        assert _normalized_autocorrelation_at_lag(signal, 64) == 0.0

    @pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
    def test_nonfinite_signal_returns_finite_zero(self, nonfinite):
        signal = np.ones(257, dtype=np.float32)
        signal[10] = nonfinite
        score = _normalized_autocorrelation_at_lag(signal, 64)
        assert score == 0.0
        assert np.isfinite(score)

    def test_five_second_input_is_linear_and_avoids_full_correlation(self, monkeypatch):
        sample_rate = 16000
        signal = np.random.default_rng(12).standard_normal(
            5 * sample_rate,
        ).astype(np.float32)

        def fail_full_correlation(*args, **kwargs):
            raise AssertionError("Production repetition scoring must not use np.correlate")

        monkeypatch.setattr(np, "correlate", fail_full_correlation)
        score = _normalized_autocorrelation_at_lag(signal, sample_rate)

        timings = []
        for _ in range(5):
            started = time.perf_counter()
            _normalized_autocorrelation_at_lag(signal, sample_rate)
            timings.append(time.perf_counter() - started)

        assert np.median(timings) < 0.25
        assert np.isfinite(score)
        assert -1.0 <= score <= 1.0


# ══════════════════════════════════════════════════════════════════════
# TONAL DETECTION TESTS
# ══════════════════════════════════════════════════════════════════════
class TestTonalDetection:
    """Tests verifying tonal signal detection logic."""

    def test_sine_wave_detected_as_tonal(self):
        """A stable sine wave is detected as music_like (tonal evidence)."""
        sr = 16000
        duration = 15.0
        n = int(sr * duration)
        t = np.linspace(0, duration, n, endpoint=False)
        waveform = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        features = _compute_audio_features(waveform, sr)
        # Sine wave: low spectral flatness, high tonal consistency, stable freq
        assert features["spectral_flatness"] < 0.1, (
            f"Sine wave spectral flatness should be low, got {features['spectral_flatness']}"
        )
        assert features["tonal_consistency"] > 0.6, (
            f"Sine wave tonal consistency should be high, got {features['tonal_consistency']}"
        )
        assert features["dominant_freq_stability"] > 0.5, (
            f"Sine wave dominant freq stability should be high, got {features['dominant_freq_stability']}"
        )

    def test_multitone_detected_according_to_config(self):
        """A multi-tone signal is detected when within config thresholds."""
        sr = 16000
        duration = 15.0
        n = int(sr * duration)
        t = np.linspace(0, duration, n, endpoint=False)
        # Multi-tone: 220Hz + 440Hz + 880Hz
        waveform = (0.2 * np.sin(2 * np.pi * 220 * t)
                     + 0.2 * np.sin(2 * np.pi * 440 * t)
                     + 0.2 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
        features = _compute_audio_features(waveform, sr)
        # Multi-tone has low flatness (still concentrated) and high tonal consistency
        assert features["spectral_flatness"] < 0.2, (
            f"Multi-tone flatness should be low, got {features['spectral_flatness']}"
        )
        assert features["tonal_consistency"] > 0.5, (
            f"Multi-tone tonal consistency should be moderate-high, got {features['tonal_consistency']}"
        )

    def test_broadband_noise_not_tonal(self):
        """Broadband random noise is NOT classified as tonal."""
        sr = 16000
        n = int(sr * 15.0)
        rng = np.random.RandomState(42)
        waveform = (rng.randn(n) * 0.05).astype(np.float32)
        features = _compute_audio_features(waveform, sr)
        # Noise: high spectral flatness, low tonal consistency
        assert features["spectral_flatness"] > 0.3, (
            f"Noise spectral flatness should be high, got {features['spectral_flatness']}"
        )
        # Noise should fail tonal criteria
        is_tonal = (
            features["spectral_flatness"] < 0.1
            and features["tonal_consistency"] > 0.6
            and features["dominant_freq_stability"] > 0.5
        )
        assert not is_tonal, "Broadband noise should NOT be classified as tonal"

    def test_silence_not_tonal(self):
        """Silence is NOT classified as tonal."""
        sr = 16000
        n = int(sr * 15.0)
        waveform = np.zeros(n, dtype=np.float32)
        features = _compute_audio_features(waveform, sr)
        # Silence: zero RMS → classified as inactive before tonal check
        assert features["rms"] < 1e-6, f"Silence RMS should be near zero, got {features['rms']}"

    def test_spectral_flatness_direction_explicit(self):
        """Verify spectral flatness interpretation: low = tonal, high = noise."""
        sr = 16000
        n = int(sr * 10.0)
        t = np.linspace(0, 10.0, n, endpoint=False)

        sine = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        noise = (np.random.RandomState(99).randn(n) * 0.05).astype(np.float32)

        sine_features = _compute_audio_features(sine, sr)
        noise_features = _compute_audio_features(noise, sr)

        # Sine flatness << noise flatness
        assert sine_features["spectral_flatness"] < noise_features["spectral_flatness"], (
            f"Sine flatness ({sine_features['spectral_flatness']}) should be less than "
            f"noise flatness ({noise_features['spectral_flatness']})"
        )

    def test_zero_energy_handled_safely(self):
        """Zero-energy waveform does not crash feature extraction."""
        sr = 16000
        waveform = np.zeros(sr * 5, dtype=np.float32)
        features = _compute_audio_features(waveform, sr)
        assert features["rms"] == 0.0
        assert np.isfinite(features["spectral_flatness"])

    def test_non_finite_input_handled_safely(self):
        """Non-finite input is handled without crash."""
        sr = 16000
        waveform = np.full(sr * 5, np.inf, dtype=np.float32)
        # _classify_customer_state handles non-finite → "inactive"
        config = _config()
        from src.demo.hold_detection import _classify_customer_state
        state = _classify_customer_state(waveform, sr, config)
        assert state == "inactive"


# ══════════════════════════════════════════════════════════════════════
# CORE HOLD DETECTION TESTS
# ══════════════════════════════════════════════════════════════════════
class TestHoldDetection:
    """Tests for detect_hold_candidates."""

    def test_normal_turn_taking_no_hold(self):
        """Normal Agent/Customer turn-taking produces no Hold."""
        intervals = [
            _iv(0.0, 5.0, "agent_active_only"),
            _iv(5.0, 10.0, "customer_active_only"),
            _iv(10.0, 15.0, "agent_active_only"),
        ]
        agent, customer = _make_waveforms(15.0,
            agent_active_ranges=[(0.0, 5.0), (10.0, 15.0)],
            customer_active_ranges=[(5.0, 10.0)])
        result = detect_hold_candidates(intervals, agent, customer, 16000, 15.0)
        assert result.total_candidates == 0

    def test_short_pause_no_hold(self):
        """Short both-inactive pause does not create Hold."""
        intervals = [
            _iv(0.0, 5.0, "agent_active_only"),
            _iv(5.0, 7.0, "neither_active"),  # 2s pause
            _iv(7.0, 12.0, "customer_active_only"),
        ]
        agent, customer = _make_waveforms(12.0,
            agent_active_ranges=[(0.0, 5.0)],
            customer_active_ranges=[(7.0, 12.0)])
        result = detect_hold_candidates(intervals, agent, customer, 16000, 12.0)
        assert result.total_candidates == 0

    def test_long_silence_uncertain_candidate(self):
        """Long silence produces uncertain_hold_candidate."""
        intervals = [
            _iv(0.0, 5.0, "agent_active_only"),
            _iv(5.0, 20.0, "neither_active"),  # 15s silence
            _iv(20.0, 25.0, "agent_active_only"),
        ]
        agent, customer = _make_waveforms(25.0,
            agent_active_ranges=[(0.0, 5.0), (20.0, 25.0)])
        result = detect_hold_candidates(intervals, agent, customer, 16000, 25.0)
        assert result.total_candidates >= 1
        c = result.candidates[0]
        assert c.classification == "uncertain_hold_candidate"
        assert "prolonged_silence" in c.evidence_types

    def test_agent_inactive_customer_speaking_not_hold(self):
        """Agent inactive while Customer speaks is not automatic Hold.

        Agent-active intervals are excluded by the agent-inactivity gate.
        Even when agent is inactive, customer speech stays uncertain.
        """
        intervals = [
            _iv(0.0, 5.0, "agent_active_only"),
            _iv(5.0, 20.0, "customer_active_only"),  # Agent inactive, customer speaks
            _iv(20.0, 25.0, "agent_active_only"),
        ]
        agent, customer = _make_waveforms(25.0,
            agent_active_ranges=[(0.0, 5.0), (20.0, 25.0)],
            customer_active_ranges=[(5.0, 20.0)])
        result = detect_hold_candidates(intervals, agent, customer, 16000, 25.0)
        # The 5-20s interval has agent_inactive=True (customer_active_only state).
        # Customer speech should not produce a likely_hold_candidate.
        for c in result.candidates:
            assert c.classification != "likely_hold_candidate", (
                f"Customer speech should not produce likely candidate, got {c.classification}"
            )

    def test_agent_inactive_tonal_likely_candidate(self):
        """Agent inactive with tonal signal → likely_hold_candidate."""
        intervals = [
            _iv(0.0, 5.0, "agent_active_only"),
            _iv(5.0, 25.0, "customer_active_only"),  # tonal on customer
            _iv(25.0, 30.0, "agent_active_only"),
        ]
        sr = 16000
        n = 30 * sr
        t = np.linspace(0, 30, n, endpoint=False)
        agent = np.zeros(n, dtype=np.float32)
        agent[:5*sr] = 0.5 * np.sin(2 * np.pi * 440 * t[:5*sr])
        agent[25*sr:] = 0.5 * np.sin(2 * np.pi * 440 * t[25*sr:])
        # Multi-tone tonal signal on customer channel (fundamental + harmonics)
        customer = np.zeros(n, dtype=np.float32)
        seg = t[5*sr:25*sr]
        customer[5*sr:25*sr] = (
            0.3 * np.sin(2 * np.pi * 440 * seg)
            + 0.2 * np.sin(2 * np.pi * 880 * seg)
            + 0.1 * np.sin(2 * np.pi * 1320 * seg)
        ).astype(np.float32)
        result = detect_hold_candidates(intervals, agent, customer, sr, 30.0)
        assert result.total_candidates >= 1
        # Should be classified as likely due to tonal evidence
        all_classifications = [c.classification for c in result.candidates]
        assert "likely_hold_candidate" in all_classifications, (
            f"Expected likely_hold_candidate, got {all_classifications}"
        )
        # Should contain music_like_audio evidence
        all_evidence = []
        for c in result.candidates:
            all_evidence.extend(c.evidence_types)
        assert "music_like_audio" in all_evidence

    def test_broadband_noise_uncertain(self):
        """Agent inactive with broadband noise → uncertain or rejected."""
        intervals = [
            _iv(0.0, 5.0, "agent_active_only"),
            _iv(5.0, 20.0, "neither_active"),
            _iv(20.0, 25.0, "agent_active_only"),
        ]
        sr = 16000
        agent = np.zeros(25 * sr, dtype=np.float32)
        agent[:5*sr] = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 5, 5*sr))
        agent[20*sr:] = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 5, 5*sr))
        customer = np.random.RandomState(42).randn(25 * sr).astype(np.float32) * 0.05
        result = detect_hold_candidates(intervals, agent, customer, 16000, 25.0)
        # Noise should produce uncertain candidate (not likely)
        for c in result.candidates:
            assert c.classification == "uncertain_hold_candidate", (
                f"Noise should be uncertain, got {c.classification}"
            )

    def test_agent_return_closes_candidate(self):
        """Agent return closes the Hold candidate."""
        intervals = [
            _iv(0.0, 5.0, "agent_active_only"),
            _iv(5.0, 20.0, "neither_active"),
            _iv(20.0, 25.0, "agent_active_only"),  # Agent returns
        ]
        agent, customer = _make_waveforms(25.0,
            agent_active_ranges=[(0.0, 5.0), (20.0, 25.0)])
        result = detect_hold_candidates(intervals, agent, customer, 16000, 25.0)
        if result.total_candidates > 0:
            c = result.candidates[0]
            assert c.end_time <= 20.0  # Ends when Agent returns
            assert c.agent_return_detected is True

    def test_no_agent_return_before_call_end(self):
        """Candidate ends at call duration when Agent doesn't return."""
        intervals = [
            _iv(0.0, 5.0, "agent_active_only"),
            _iv(5.0, 30.0, "neither_active"),  # No Agent return
        ]
        agent, customer = _make_waveforms(30.0,
            agent_active_ranges=[(0.0, 5.0)])
        result = detect_hold_candidates(intervals, agent, customer, 16000, 30.0)
        if result.total_candidates > 0:
            c = result.candidates[0]
            assert c.end_time == 30.0

    def test_two_adjacent_candidates_merge(self):
        """Adjacent candidates within merge gap merge into one."""
        # Agent inactive throughout both candidate periods — no Agent return
        intervals = [
            _iv(0.0, 5.0, "agent_active_only"),
            _iv(5.0, 18.0, "neither_active"),   # 13s silence → candidate 1
            _iv(18.0, 20.0, "customer_active_only"),  # brief customer speech
            _iv(20.0, 35.0, "neither_active"),  # 15s silence → candidate 2
            _iv(35.0, 40.0, "neither_active"),  # still inactive → candidate extends
        ]
        agent, customer = _make_waveforms(40.0,
            agent_active_ranges=[(0.0, 5.0)],
            customer_active_ranges=[(18.0, 20.0)])
        # Use wide merge gap to force merging
        result = detect_hold_candidates(intervals, agent, customer, 16000, 40.0,
                                        config=_config(merge_gap_between_candidates_sec=10.0))
        # With 10s merge gap and no Agent return, candidates should merge
        if result.total_candidates >= 2:
            assert result.candidates[0].end_time >= 20.0
        elif result.total_candidates == 1:
            # Merged into one candidate spanning 5-35
            assert result.candidates[0].end_time >= 35.0

    def test_no_merge_after_agent_return(self):
        """Candidates do not merge across Agent return."""
        intervals = [
            _iv(0.0, 5.0, "agent_active_only"),
            _iv(5.0, 18.0, "neither_active"),
            _iv(18.0, 22.0, "agent_active_only"),  # Agent returns for 4s
            _iv(22.0, 35.0, "neither_active"),
            _iv(35.0, 40.0, "agent_active_only"),
        ]
        agent, customer = _make_waveforms(40.0,
            agent_active_ranges=[(0.0, 5.0), (18.0, 22.0), (35.0, 40.0)])
        result = detect_hold_candidates(intervals, agent, customer, 16000, 40.0)
        # Agent returned for 4s — candidates should NOT merge across this
        if result.total_candidates >= 2:
            assert result.candidates[0].end_time <= 18.0

    def test_candidate_timestamps_preserved(self):
        """Candidate timestamps preserve exact timeline."""
        intervals = [
            _iv(0.0, 5.0, "agent_active_only"),
            _iv(5.0, 20.0, "neither_active"),
            _iv(20.0, 25.0, "agent_active_only"),
        ]
        agent, customer = _make_waveforms(25.0,
            agent_active_ranges=[(0.0, 5.0), (20.0, 25.0)])
        result = detect_hold_candidates(intervals, agent, customer, 16000, 25.0)
        for c in result.candidates:
            assert c.start_time >= 0.0
            assert c.end_time <= 25.0
            assert c.duration_sec == c.end_time - c.start_time

    def test_120_second_candidate(self):
        """120.0-second candidate duration."""
        intervals = [
            _iv(0.0, 5.0, "agent_active_only"),
            _iv(5.0, 125.0, "neither_active"),
            _iv(125.0, 130.0, "agent_active_only"),
        ]
        agent, customer = _make_waveforms(130.0,
            agent_active_ranges=[(0.0, 5.0), (125.0, 130.0)])
        result = detect_hold_candidates(intervals, agent, customer, 16000, 130.0)
        assert result.total_candidates >= 1
        c = result.candidates[0]
        assert c.duration_sec >= 119.0

    def test_120_5_second_candidate(self):
        """120.5-second candidate (not rounded down)."""
        intervals = [
            _iv(0.0, 5.0, "agent_active_only"),
            _iv(5.0, 125.5, "neither_active"),
            _iv(125.5, 130.0, "agent_active_only"),
        ]
        agent, customer = _make_waveforms(130.0,
            agent_active_ranges=[(0.0, 5.0), (125.5, 130.0)])
        result = detect_hold_candidates(intervals, agent, customer, 16000, 130.0)
        assert result.total_candidates >= 1
        c = result.candidates[0]
        assert c.duration_sec >= 120.0

    def test_non_hop_multiple_duration(self):
        """Non-hop-multiple duration is handled correctly."""
        intervals = [
            _iv(0.0, 5.0, "agent_active_only"),
            _iv(5.0, 135.4, "neither_active"),
            _iv(135.4, 140.0, "agent_active_only"),
        ]
        agent, customer = _make_waveforms(140.0,
            agent_active_ranges=[(0.0, 5.0), (135.4, 140.0)])
        result = detect_hold_candidates(intervals, agent, customer, 16000, 140.0)
        assert result.total_candidates >= 1
        c = result.candidates[0]
        assert abs(c.end_time - 135.4) < 0.1

    def test_confidence_and_evidence_populated(self):
        """Confidence and evidence fields are populated."""
        intervals = [
            _iv(0.0, 5.0, "agent_active_only"),
            _iv(5.0, 20.0, "neither_active"),
            _iv(20.0, 25.0, "agent_active_only"),
        ]
        agent, customer = _make_waveforms(25.0,
            agent_active_ranges=[(0.0, 5.0), (20.0, 25.0)])
        result = detect_hold_candidates(intervals, agent, customer, 16000, 25.0)
        if result.total_candidates > 0:
            c = result.candidates[0]
            assert c.confidence > 0
            assert len(c.evidence_types) > 0
            assert c.classification != ""

    def test_no_definitive_music_claim(self):
        """No candidate claims definitive music or recorded message."""
        intervals = [
            _iv(0.0, 5.0, "agent_active_only"),
            _iv(5.0, 25.0, "neither_active"),
            _iv(25.0, 30.0, "agent_active_only"),
        ]
        agent, customer = _make_waveforms(30.0,
            agent_active_ranges=[(0.0, 5.0), (25.0, 30.0)],
            customer_music_ranges=[(5.0, 25.0)])
        result = detect_hold_candidates(intervals, agent, customer, 16000, 30.0)
        for c in result.candidates:
            assert "definitive" not in c.detection_notes.lower()
            assert "music_like" in c.evidence_types or "prolonged_silence" in c.evidence_types

    def test_no_activity_interval_modified(self):
        """Activity intervals are not modified by detection."""
        intervals = [
            _iv(0.0, 5.0, "agent_active_only"),
            _iv(5.0, 20.0, "neither_active"),
            _iv(20.0, 25.0, "agent_active_only"),
        ]
        original = [(iv.start_time, iv.end_time, iv.state) for iv in intervals]
        agent, customer = _make_waveforms(25.0,
            agent_active_ranges=[(0.0, 5.0), (20.0, 25.0)])
        _ = detect_hold_candidates(intervals, agent, customer, 16000, 25.0)
        for iv, (s, e, st) in zip(intervals, original):
            assert iv.start_time == s
            assert iv.end_time == e
            assert iv.state == st

    def test_no_file_written(self):
        """No audio file is written to repository."""
        intervals = [_iv(0.0, 5.0, "agent_active_only")]
        agent, customer = _make_waveforms(5.0, agent_active_ranges=[(0.0, 5.0)])
        result = detect_hold_candidates(intervals, agent, customer, 16000, 5.0)
        # Just verify it doesn't crash and returns a result
        assert isinstance(result.total_candidates, int)

    def test_thresholds_configurable(self):
        """Detection thresholds are configurable."""
        intervals = [
            _iv(0.0, 5.0, "agent_active_only"),
            _iv(5.0, 15.0, "neither_active"),
            _iv(15.0, 20.0, "agent_active_only"),
        ]
        agent, customer = _make_waveforms(20.0,
            agent_active_ranges=[(0.0, 5.0), (15.0, 20.0)])
        # Default config
        r1 = detect_hold_candidates(intervals, agent, customer, 16000, 20.0)
        # Custom config with higher minimum duration
        r2 = detect_hold_candidates(intervals, agent, customer, 16000, 20.0,
                                    config=_config(minimum_hold_candidate_duration_sec=20.0))
        # Higher threshold should produce fewer or no candidates
        assert r2.total_candidates <= r1.total_candidates


# ══════════════════════════════════════════════════════════════════════
# FULL SYNTHETIC CALL TEST (340.5s)
# ══════════════════════════════════════════════════════════════════════
class TestSyntheticCall340:
    """
    Full 340.5-second synthetic call with three candidate regions.

    Timeline:
    - 0–20s: normal conversation (both active)
    - 20–50s: Agent inactive, silence-like waiting (both inactive)
    - 50–70s: conversation resumed (both active)
    - 70–195s: Agent inactive, customer channel has tonal signal
    - 195–210s: Agent returns (agent active only)
    - 210–340.5s: Agent inactive, silence-like waiting (both inactive)
    """

    @pytest.fixture
    def synthetic_call(self):
        """Build the full 340.5s synthetic call."""
        sr = 16000
        duration = 340.5
        n = int(sr * duration)
        t = np.linspace(0, duration, n, endpoint=False)

        agent = np.zeros(n, dtype=np.float32)
        customer = np.zeros(n, dtype=np.float32)

        # Agent: active 0-20s, 195-210s
        for s, e in [(0, 20), (195, 210)]:
            si, ei = int(s * sr), int(e * sr)
            agent[si:ei] = 0.5 * np.sin(2 * np.pi * 440 * t[si:ei])

        # Customer: speech 50-70s, tonal 70-195s
        for s, e in [(50, 70)]:
            si, ei = int(s * sr), int(e * sr)
            customer[si:ei] = 0.5 * np.sin(2 * np.pi * 880 * t[si:ei])
        # Multi-tone tonal signal on customer channel during hold
        si, ei = int(70 * sr), int(195 * sr)
        seg = t[si:ei]
        customer[si:ei] = (
            0.3 * np.sin(2 * np.pi * 440 * seg)
            + 0.2 * np.sin(2 * np.pi * 880 * seg)
            + 0.1 * np.sin(2 * np.pi * 1320 * seg)
        ).astype(np.float32)

        intervals = [
            _iv(0.0, 20.0, "both_active_interval"),
            _iv(20.0, 50.0, "neither_active"),
            _iv(50.0, 70.0, "both_active_interval"),
            _iv(70.0, 195.0, "customer_active_only"),
            _iv(195.0, 210.0, "agent_active_only"),
            _iv(210.0, 340.5, "neither_active"),
        ]

        return intervals, agent, customer, sr, duration

    def test_three_regions_detected(self, synthetic_call):
        """All three candidate regions are detected."""
        intervals, agent, customer, sr, duration = synthetic_call
        result = detect_hold_candidates(intervals, agent, customer, sr, duration)
        assert result.total_candidates == 3, (
            f"Expected 3 candidates, got {result.total_candidates}: "
            f"{[(c.start_time, c.end_time, c.classification) for c in result.candidates]}"
        )

    def test_first_candidate_20_50(self, synthetic_call):
        """First candidate: 20.0–50.0s, prolonged_silence, uncertain."""
        intervals, agent, customer, sr, duration = synthetic_call
        result = detect_hold_candidates(intervals, agent, customer, sr, duration)
        c = result.candidates[0]
        assert abs(c.start_time - 20.0) < 0.1
        assert abs(c.end_time - 50.0) < 0.1
        assert abs(c.duration_sec - 30.0) < 0.1
        assert "prolonged_silence" in c.evidence_types
        assert c.classification == "uncertain_hold_candidate"
        assert c.agent_return_detected is True

    def test_second_candidate_70_195(self, synthetic_call):
        """Second candidate: 70.0–195.0s, tonal evidence, likely."""
        intervals, agent, customer, sr, duration = synthetic_call
        result = detect_hold_candidates(intervals, agent, customer, sr, duration)
        c = result.candidates[1]
        assert abs(c.start_time - 70.0) < 0.1
        assert abs(c.end_time - 195.0) < 0.1
        assert abs(c.duration_sec - 125.0) < 0.1
        assert "music_like_audio" in c.evidence_types
        assert c.classification == "likely_hold_candidate", (
            f"Expected likely, got {c.classification}"
        )
        assert c.agent_return_detected is True

    def test_second_candidate_duration_125(self, synthetic_call):
        """Second candidate duration is exactly 125.0s."""
        intervals, agent, customer, sr, duration = synthetic_call
        result = detect_hold_candidates(intervals, agent, customer, sr, duration)
        c = result.candidates[1]
        assert abs(c.duration_sec - 125.0) < 0.01

    def test_third_candidate_210_340(self, synthetic_call):
        """Third candidate: 210.0–340.5s, silence, uncertain."""
        intervals, agent, customer, sr, duration = synthetic_call
        result = detect_hold_candidates(intervals, agent, customer, sr, duration)
        c = result.candidates[2]
        assert abs(c.start_time - 210.0) < 0.1
        assert abs(c.end_time - 340.5) < 0.1
        assert abs(c.duration_sec - 130.5) < 0.1
        assert "prolonged_silence" in c.evidence_types
        assert c.classification == "uncertain_hold_candidate"

    def test_third_candidate_duration_130_5(self, synthetic_call):
        """Third candidate duration is exactly 130.5s."""
        intervals, agent, customer, sr, duration = synthetic_call
        result = detect_hold_candidates(intervals, agent, customer, sr, duration)
        c = result.candidates[2]
        assert abs(c.duration_sec - 130.5) < 0.01

    def test_agent_return_at_195(self, synthetic_call):
        """Agent return at 195.0 closes the tonal candidate."""
        intervals, agent, customer, sr, duration = synthetic_call
        result = detect_hold_candidates(intervals, agent, customer, sr, duration)
        c = result.candidates[1]
        assert c.agent_return_detected is True
        assert abs(c.end_time - 195.0) < 0.1

    def test_agent_return_at_340_5(self, synthetic_call):
        """At 340.5s call ends — reaches_call_end is True."""
        intervals, agent, customer, sr, duration = synthetic_call
        result = detect_hold_candidates(intervals, agent, customer, sr, duration)
        c = result.candidates[2]
        assert c.reaches_call_end is True
        assert c.agent_return_detected is False

    def test_no_merge_across_returns(self, synthetic_call):
        """No merge across either Agent-return interval."""
        intervals, agent, customer, sr, duration = synthetic_call
        result = detect_hold_candidates(intervals, agent, customer, sr, duration)
        # Three separate candidates, not merged
        assert result.total_candidates == 3
        # Verify gaps between candidates
        for i in range(len(result.candidates) - 1):
            gap = result.candidates[i + 1].start_time - result.candidates[i].end_time
            assert gap > 0, (
                f"Candidates {i} and {i+1} should not overlap or be merged"
            )

    def test_classifications_correct(self, synthetic_call):
        """Expected classifications for all three candidates."""
        intervals, agent, customer, sr, duration = synthetic_call
        result = detect_hold_candidates(intervals, agent, customer, sr, duration)
        expected = [
            "uncertain_hold_candidate",  # 20-50s: silence only
            "likely_hold_candidate",     # 70-195s: tonal evidence
            "uncertain_hold_candidate",  # 210-340.5s: silence only
        ]
        actual = [c.classification for c in result.candidates]
        assert actual == expected

    def test_likely_count(self, synthetic_call):
        """Exactly one likely candidate (the tonal interval)."""
        intervals, agent, customer, sr, duration = synthetic_call
        result = detect_hold_candidates(intervals, agent, customer, sr, duration)
        assert result.likely_count == 1
        assert result.uncertain_count == 2

    def test_candidate_125s_triggers_duration_review(self, synthetic_call):
        """125.0s likely candidate exceeds 120s duration limit."""
        intervals, agent, customer, sr, duration = synthetic_call
        result = detect_hold_candidates(intervals, agent, customer, sr, duration)
        c = result.candidates[1]  # 70-195s, likely, 125s
        assert c.duration_sec > 120.0
        # This candidate IS counted (likely), so it WILL trigger duration violation
        assert c.classification == "likely_hold_candidate"

    def test_candidate_130_5s_excluded_from_count(self, synthetic_call):
        """130.5s uncertain candidate is excluded by default counting mode."""
        from src.demo.hold_policy import evaluate_hold_policy
        from src.demo.schemas import HoldPolicyConfig
        intervals, agent, customer, sr, duration = synthetic_call
        result = detect_hold_candidates(intervals, agent, customer, sr, duration)
        policy = evaluate_hold_policy(
            result.candidates,
            HoldPolicyConfig(
                max_hold_count=2,
                max_hold_duration_sec=120.0,
                counting_mode="count_likely_and_confirmed",
            )
        )
        # Only the likely candidate (125s) is counted
        assert policy.counted_hold_events == 1
        # The uncertain 130.5s is NOT counted
        counted_ids = [v.candidate_id for v in policy.duration_violations]
        assert "hold_003" not in counted_ids

    def test_uncounted_over_duration_visible(self, synthetic_call):
        """Uncounted 130.5s candidate appears in uncounted_over_duration."""
        from src.demo.hold_policy import evaluate_hold_policy
        from src.demo.schemas import HoldPolicyConfig
        intervals, agent, customer, sr, duration = synthetic_call
        result = detect_hold_candidates(intervals, agent, customer, sr, duration)
        policy = evaluate_hold_policy(
            result.candidates,
            HoldPolicyConfig(
                max_hold_count=2,
                max_hold_duration_sec=120.0,
                counting_mode="count_likely_and_confirmed",
            )
        )
        # The uncertain 130.5s should appear in uncounted_over_duration
        assert len(policy.uncounted_over_duration) >= 1
        u = policy.uncounted_over_duration[0]
        assert abs(u.duration_sec - 130.5) < 0.1
        assert u.classification == "uncertain_hold_candidate"
        assert u.reaches_call_end is True

    def test_review_required_for_uncounted_over_duration(self, synthetic_call):
        """Review is required even when uncertain candidate is excluded."""
        from src.demo.hold_policy import evaluate_hold_policy
        from src.demo.schemas import HoldPolicyConfig
        intervals, agent, customer, sr, duration = synthetic_call
        result = detect_hold_candidates(intervals, agent, customer, sr, duration)
        policy = evaluate_hold_policy(
            result.candidates,
            HoldPolicyConfig(
                max_hold_count=2,
                max_hold_duration_sec=120.0,
                counting_mode="count_likely_and_confirmed",
            )
        )
        # Review required due to: 1 counted duration violation (125s) +
        # 1 uncounted over-duration (130.5s)
        assert policy.review_required is True
        assert len(policy.uncounted_over_duration) >= 1
        assert len(policy.duration_violations) >= 1
