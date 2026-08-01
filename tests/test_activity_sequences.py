"""
Tests for activity-state sequences, merged intervals, and inactivity metadata.

Uses synthetic SharedWindowResult sequences with waveform-based activity.
"""

import os
import sys
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.engine import EmotionEngine
from src.demo.schemas import (
    SharedWindow, SharedWindowResult, SpeakerWindowResult,
    SpeakerProcessingResult, WindowConfig, SpeechActivityConfig,
    StereoLoadResult, OriginalStereoAudio, ModelStereoAudio,
    ValidationResult, ActivityInterval,
)
from src.demo.speaker_summary import (
    build_activity_states, merge_activity_states, compute_inactivity_metadata,
    compute_call_summary, compute_speaker_summary,
)
from src.demo.speaker_processor import process_stereo_call


# ══════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════
def _make_wr(window_id, start, end, agent_active, customer_active,
             agent_label=None, customer_label=None):
    """Create a synthetic SharedWindowResult for testing."""
    window = SharedWindow(
        window_id=window_id, original_start=start, original_end=end,
        model_start_sample=int(start * 16000), model_end_sample=int(end * 16000),
    )

    def _sp(active, label, channel, role):
        pred = None
        if label:
            from src.inference.schemas import InferenceOutput
            pred = InferenceOutput(
                label=label, label_id=0,
                probabilities={"Low": 0.1, "Neutral": 0.7, "High": 0.2},
                confidence=0.9, model_version="1.0.0", device="cpu",
                segment_duration_sec=end - start,
            )
        return SpeakerWindowResult(
            channel=channel, role=role, speech_active=active,
            active_ratio=1.0 if active else 0.0, rms=0.3 if active else 0.0,
            prediction=pred, inactive_reason=None if active else "silence",
        )

    return SharedWindowResult(
        window=window,
        agent=_sp(agent_active, agent_label, "left", "Agent"),
        customer=_sp(customer_active, customer_label, "right", "Customer"),
    )


def _make_waveforms(duration, sr=16000, agent_active_ranges=None, customer_active_ranges=None):
    """
    Create synthetic waveforms with specified active ranges.

    Args:
        duration: Total duration in seconds.
        sr: Sample rate.
        agent_active_ranges: List of (start_sec, end_sec) tuples where agent is active.
        customer_active_ranges: List of (start_sec, end_sec) tuples where customer is active.
    """
    n = int(sr * duration)
    agent = np.zeros(n, dtype=np.float32)
    customer = np.zeros(n, dtype=np.float32)
    t = np.linspace(0, duration, n, endpoint=False)

    if agent_active_ranges:
        for s, e in agent_active_ranges:
            si, ei = int(s * sr), int(e * sr)
            agent[si:ei] = 0.5 * np.sin(2 * np.pi * 440 * t[si:ei])

    if customer_active_ranges:
        for s, e in customer_active_ranges:
            si, ei = int(s * sr), int(e * sr)
            customer[si:ei] = 0.5 * np.sin(2 * np.pi * 880 * t[si:ei])

    return agent, customer


def _makeAlternating_wrSet():
    """Create the standard 8-second alternating-activity window results."""
    return [
        _make_wr(0, 0.0, 2.0, True, False, "Low Emotion", None),
        _make_wr(1, 1.0, 3.0, False, True, None, "High Emotion"),
        _make_wr(2, 2.0, 4.0, True, True, "Neutral Emotion", "High Emotion"),
        _make_wr(3, 3.0, 5.0, True, True, "Low Emotion", "High Emotion"),
        _make_wr(4, 4.0, 6.0, True, True, "Low Emotion", "High Emotion"),
        _make_wr(5, 5.0, 7.0, True, True, "Low Emotion", "High Emotion"),
        _make_wr(6, 6.0, 8.0, False, False, None, None),
    ]


# ══════════════════════════════════════════════════════════════════════
# ACTIVITY STATE TESTS (with waveforms)
# ══════════════════════════════════════════════════════════════════════
class TestActivityStates:
    """Tests for build_activity_states with waveform-based activity."""

    def test_agent_active_only(self):
        """Agent-active-only interval."""
        wrs = [_make_wr(0, 0.0, 2.0, True, False)]
        agent, customer = _make_waveforms(2.0, agent_active_ranges=[(0.0, 2.0)])
        states = build_activity_states(wrs, agent, customer, call_duration_sec=2.0)
        assert len(states) >= 1
        assert states[0].state == "agent_active_only"

    def test_customer_active_only(self):
        """Customer-active-only interval."""
        wrs = [_make_wr(0, 0.0, 2.0, False, True)]
        agent, customer = _make_waveforms(2.0, customer_active_ranges=[(0.0, 2.0)])
        states = build_activity_states(wrs, agent, customer, call_duration_sec=2.0)
        assert states[0].state == "customer_active_only"

    def test_both_active(self):
        """Both-active interval (true simultaneous activity)."""
        wrs = [_make_wr(0, 0.0, 2.0, True, True)]
        agent, customer = _make_waveforms(2.0,
            agent_active_ranges=[(0.0, 2.0)],
            customer_active_ranges=[(0.0, 2.0)])
        states = build_activity_states(wrs, agent, customer, call_duration_sec=2.0)
        assert states[0].state == "both_active_interval"

    def test_neither_active(self):
        """Neither-active interval."""
        wrs = [_make_wr(0, 0.0, 2.0, False, False)]
        agent, customer = _make_waveforms(2.0)
        states = build_activity_states(wrs, agent, customer, call_duration_sec=2.0)
        assert states[0].state == "neither_active"

    def test_alternating_activity(self):
        """Alternating activity with waveforms produces accurate base intervals."""
        wrs = _makeAlternating_wrSet()
        agent, customer = _make_waveforms(8.0,
            agent_active_ranges=[(0.0, 2.0), (4.0, 6.0)],
            customer_active_ranges=[(2.0, 4.0), (4.0, 6.0)])
        states = build_activity_states(wrs, agent, customer, call_duration_sec=8.0)
        # 8 base intervals (hop=1s, duration=8s): 0-1,1-2,...,7-8
        assert len(states) == 8
        assert states[0].original_start == 0.0
        assert states[-1].original_end == 8.0

    def test_no_waveforms_raises_error(self):
        """Missing waveforms raises ValueError."""
        wrs = [_make_wr(0, 0.0, 2.0, True, False)]
        with pytest.raises(ValueError, match="waveforms"):
            build_activity_states(wrs)

    def test_missing_customer_waveform_raises_error(self):
        """Missing customer waveform raises ValueError."""
        wrs = [_make_wr(0, 0.0, 2.0, True, False)]
        agent = np.zeros(32000, dtype=np.float32)
        with pytest.raises(ValueError, match="waveforms"):
            build_activity_states(wrs, agent_waveform=agent)


# ══════════════════════════════════════════════════════════════════════
# MERGED INTERVAL TESTS
# ══════════════════════════════════════════════════════════════════════
class TestMergedIntervals:
    """Tests for merge_activity_states."""

    def test_continuous_inactivity_merged(self):
        """Continuous inactivity is merged into one interval."""
        wrs = [
            _make_wr(0, 0.0, 2.0, False, False),
            _make_wr(1, 1.0, 3.0, False, False),
            _make_wr(2, 2.0, 4.0, False, False),
        ]
        agent, customer = _make_waveforms(4.0)
        states = build_activity_states(wrs, agent, customer, call_duration_sec=4.0)
        intervals = merge_activity_states(states)
        assert len(intervals) == 1
        assert intervals[0].state == "neither_active"

    def test_merged_intervals_no_overlaps(self):
        """Merged intervals have no time overlaps."""
        wrs = [
            _make_wr(0, 0.0, 2.0, True, False),
            _make_wr(1, 1.0, 3.0, True, False),
            _make_wr(2, 2.0, 4.0, False, True),
            _make_wr(3, 3.0, 5.0, False, True),
        ]
        agent, customer = _make_waveforms(5.0,
            agent_active_ranges=[(0.0, 2.0)],
            customer_active_ranges=[(2.0, 4.0)])
        states = build_activity_states(wrs, agent, customer, call_duration_sec=5.0)
        intervals = merge_activity_states(states)
        for i in range(1, len(intervals)):
            assert intervals[i].start_time >= intervals[i-1].end_time - 1e-6

    def test_intervals_cover_full_timeline(self):
        """Merged intervals cover the full call timeline."""
        wrs = [
            _make_wr(0, 0.0, 2.0, True, False),
            _make_wr(1, 1.0, 3.0, False, True),
            _make_wr(2, 2.0, 4.0, True, True),
        ]
        agent, customer = _make_waveforms(4.0,
            agent_active_ranges=[(0.0, 1.0), (3.0, 4.0)],
            customer_active_ranges=[(1.0, 2.0), (3.0, 4.0)])
        states = build_activity_states(wrs, agent, customer, call_duration_sec=4.0)
        intervals = merge_activity_states(states)
        assert len(intervals) > 0
        assert intervals[0].start_time == 0.0
        assert intervals[-1].end_time == 4.0

    def test_interval_durations_sum_to_call_duration(self):
        """Interval durations sum to call duration."""
        wrs = [
            _make_wr(0, 0.0, 2.0, True, False),
            _make_wr(1, 1.0, 3.0, False, True),
            _make_wr(2, 2.0, 4.0, True, True),
        ]
        agent, customer = _make_waveforms(4.0,
            agent_active_ranges=[(0.0, 1.0), (3.0, 4.0)],
            customer_active_ranges=[(1.0, 2.0), (3.0, 4.0)])
        states = build_activity_states(wrs, agent, customer, call_duration_sec=4.0)
        intervals = merge_activity_states(states)
        total = sum(iv.duration_sec for iv in intervals)
        assert abs(total - 4.0) < 0.1

    def test_no_interval_labeled_hold(self):
        """No interval is labeled Hold."""
        wrs = [_make_wr(0, 0.0, 2.0, False, False), _make_wr(1, 1.0, 3.0, False, False)]
        agent, customer = _make_waveforms(3.0)
        states = build_activity_states(wrs, agent, customer, call_duration_sec=3.0)
        intervals = merge_activity_states(states)
        for iv in intervals:
            assert "hold" not in iv.state.lower()
            assert "waiting" not in iv.state.lower()

    def test_no_violation_generated(self):
        """No policy violation is generated."""
        wrs = [_make_wr(0, 0.0, 2.0, False, False), _make_wr(1, 1.0, 3.0, False, False)]
        agent, customer = _make_waveforms(3.0)
        states = build_activity_states(wrs, agent, customer, call_duration_sec=3.0)
        intervals = merge_activity_states(states)
        meta = compute_inactivity_metadata(intervals)
        assert "longest_neither_active_interval" in meta
        assert meta["longest_neither_active_interval"] > 0


# ══════════════════════════════════════════════════════════════════════
# INACTIVITY METADATA TESTS
# ══════════════════════════════════════════════════════════════════════
class TestInactivityMetadata:
    """Tests for compute_inactivity_metadata."""

    def test_longest_agent_inactivity(self):
        """Longest Agent inactivity is calculated correctly."""
        wrs = [
            _make_wr(0, 0.0, 2.0, True, False),
            _make_wr(1, 1.0, 3.0, False, True),
            _make_wr(2, 2.0, 4.0, False, True),
            _make_wr(3, 3.0, 5.0, False, True),
            _make_wr(4, 4.0, 6.0, True, False),
        ]
        agent, customer = _make_waveforms(6.0,
            agent_active_ranges=[(0.0, 1.0), (5.0, 6.0)],
            customer_active_ranges=[(1.0, 5.0)])
        states = build_activity_states(wrs, agent, customer, call_duration_sec=6.0)
        intervals = merge_activity_states(states)
        meta = compute_inactivity_metadata(intervals)
        assert meta["longest_agent_inactive_interval"] >= 3.9

    def test_longest_neither_active(self):
        """Longest neither-active interval is calculated correctly."""
        wrs = [
            _make_wr(0, 0.0, 2.0, True, False),
            _make_wr(1, 1.0, 3.0, False, False),
            _make_wr(2, 2.0, 4.0, False, False),
            _make_wr(3, 3.0, 5.0, False, False),
            _make_wr(4, 4.0, 6.0, True, False),
        ]
        agent, customer = _make_waveforms(6.0,
            agent_active_ranges=[(0.0, 1.0), (5.0, 6.0)])
        states = build_activity_states(wrs, agent, customer, call_duration_sec=6.0)
        intervals = merge_activity_states(states)
        meta = compute_inactivity_metadata(intervals)
        assert meta["longest_neither_active_interval"] >= 3.9


# ══════════════════════════════════════════════════════════════════════
# HOLD-READY DURATION ACCURACY TESTS
# ══════════════════════════════════════════════════════════════════════
class TestDurationAccuracy:
    """Tests for exact duration accuracy in activity intervals."""

    def _run_test(self, duration, agent_ranges, customer_ranges, expected_states=None):
        """Helper: create waveforms, build states, merge, verify."""
        wrs = []
        for i in range(int(duration)):  # one window per second
            a_active = any(s <= i < e for s, e in (agent_ranges or []))
            c_active = any(s <= i < e for s, e in (customer_ranges or []))
            wrs.append(_make_wr(i, float(i), float(i + 1), a_active, c_active))
        # Add final partial if needed
        if duration % 1.0 > 0.01:
            i = int(duration)
            a_active = any(s <= i < e for s, e in (agent_ranges or []))
            c_active = any(s <= i < e for s, e in (customer_ranges or []))
            wrs.append(_make_wr(i, float(i), duration, a_active, c_active))

        agent, customer = _make_waveforms(duration,
            agent_active_ranges=agent_ranges, customer_active_ranges=customer_ranges)
        states = build_activity_states(wrs, agent, customer, call_duration_sec=duration)
        intervals = merge_activity_states(states)
        return states, intervals

    def test_120_second_agent_inactive(self):
        """120.0-second call with Agent active 0-5s, inactive 5-120s."""
        states, intervals = self._run_test(120.0, [(0.0, 5.0)], [])
        total = sum(iv.duration_sec for iv in intervals)
        assert abs(total - 120.0) < 0.1
        agent_inactive = [iv for iv in intervals if not iv.agent_active]
        longest = max(iv.duration_sec for iv in agent_inactive) if agent_inactive else 0
        # Agent inactive from 5-120s = 115 seconds
        assert longest >= 114.0

    def test_120_5_second_agent_inactive(self):
        """120.5-second call with Agent active 0-5s, inactive 5-120.5s."""
        states, intervals = self._run_test(120.5, [(0.0, 5.0)], [])
        total = sum(iv.duration_sec for iv in intervals)
        assert abs(total - 120.5) < 0.1
        agent_inactive = [iv for iv in intervals if not iv.agent_active]
        longest = max(iv.duration_sec for iv in agent_inactive) if agent_inactive else 0
        # Agent inactive from 5-120.5s = 115.5 seconds
        assert longest >= 114.0

    def test_120_second_neither_active(self):
        """120.0-second neither-active interval."""
        states, intervals = self._run_test(120.0, [], [])
        total = sum(iv.duration_sec for iv in intervals)
        assert abs(total - 120.0) < 0.1
        assert all(iv.state == "neither_active" for iv in intervals)

    def test_120_5_second_neither_active(self):
        """120.5-second neither-active interval."""
        states, intervals = self._run_test(120.5, [], [])
        total = sum(iv.duration_sec for iv in intervals)
        assert abs(total - 120.5) < 0.1

    def test_inactivity_at_end_of_call(self):
        """Inactivity continuing until exact end of call."""
        states, intervals = self._run_test(10.0, [(0.0, 3.0)], [])
        last_interval = intervals[-1]
        assert last_interval.end_time == 10.0
        assert last_interval.state == "neither_active"

    def test_inactivity_beginning_inside_final_window(self):
        """Inactivity beginning inside the final analysis window."""
        states, intervals = self._run_test(5.0, [(0.0, 2.0)], [(4.0, 5.0)])
        # Check that the interval at 4-5s is customer_active_only
        at_4 = [iv for iv in intervals if abs(iv.start_time - 4.0) < 0.1]
        assert len(at_4) == 1
        assert at_4[0].state == "customer_active_only"

    def test_non_hop_multiple_8_4_seconds(self):
        """Non-hop-multiple call duration of 8.4 seconds is fully covered."""
        states, intervals = self._run_test(8.4, [(0.0, 2.0)], [(2.0, 4.0)])
        total = sum(iv.duration_sec for iv in intervals)
        assert abs(total - 8.4) < 0.1
        assert intervals[-1].end_time == 8.4
        # No gaps
        for i in range(1, len(intervals)):
            assert abs(intervals[i].start_time - intervals[i-1].end_time) < 1e-6

    def test_shorter_than_one_window(self):
        """Call shorter than one analysis window."""
        states, intervals = self._run_test(0.5, [(0.0, 0.5)], [])
        total = sum(iv.duration_sec for iv in intervals)
        assert abs(total - 0.5) < 0.1
        assert intervals[0].state == "agent_active_only"

    def test_shorter_than_one_hop(self):
        """Call duration shorter than one hop."""
        # With hop=1s, a 0.3s call should produce one interval
        wrs = [_make_wr(0, 0.0, 0.3, True, False)]
        agent, customer = _make_waveforms(0.3, agent_active_ranges=[(0.0, 0.3)])
        states = build_activity_states(wrs, agent, customer, call_duration_sec=0.3)
        intervals = merge_activity_states(states)
        total = sum(iv.duration_sec for iv in intervals)
        assert abs(total - 0.3) < 0.1

    def test_final_partial_interval(self):
        """Final partial interval is included."""
        states, intervals = self._run_test(3.7, [(0.0, 2.0)], [])
        total = sum(iv.duration_sec for iv in intervals)
        assert abs(total - 3.7) < 0.1
        assert intervals[-1].end_time == 3.7

    def test_no_gaps(self):
        """No gaps between intervals."""
        states, intervals = self._run_test(6.0, [(0.0, 2.0), (4.0, 6.0)], [(2.0, 4.0)])
        for i in range(1, len(intervals)):
            assert abs(intervals[i].start_time - intervals[i-1].end_time) < 1e-6

    def test_no_overlaps(self):
        """No overlaps between intervals."""
        states, intervals = self._run_test(6.0, [(0.0, 2.0), (4.0, 6.0)], [(2.0, 4.0)])
        for i in range(1, len(intervals)):
            assert intervals[i].start_time >= intervals[i-1].end_time - 1e-6


# ══════════════════════════════════════════════════════════════════════
# CALL SUMMARY TEST
# ══════════════════════════════════════════════════════════════════════
class TestCallSummary:
    """Tests for compute_call_summary."""

    def test_call_summary_basic(self):
        """Basic call summary is populated."""
        wrs = [
            _make_wr(0, 0.0, 2.0, True, True, "Low Emotion", "High Emotion"),
            _make_wr(1, 1.0, 3.0, True, False, "Neutral Emotion", None),
        ]
        agent, customer = _make_waveforms(3.0,
            agent_active_ranges=[(0.0, 2.0)],
            customer_active_ranges=[(0.0, 1.0)])

        agent_result = SpeakerProcessingResult(channel="left", role="Agent", windows=wrs)
        cust_result = SpeakerProcessingResult(channel="right", role="Customer", windows=wrs)
        agent_summary = compute_speaker_summary(agent_result, 3.0)
        cust_summary = compute_speaker_summary(cust_result, 3.0)

        states = build_activity_states(wrs, agent, customer, call_duration_sec=3.0)
        intervals = merge_activity_states(states)
        inact_meta = compute_inactivity_metadata(intervals)

        from src.demo.schemas import StereoProcessingResult
        proc_result = StereoProcessingResult(
            duration_sec=3.0, shared_windows=[w.window for w in wrs],
            window_results=wrs, model_version="1.0.0",
        )

        call_summary = compute_call_summary(
            proc_result, agent_summary, cust_summary, intervals, inact_meta,
        )
        assert call_summary.call_duration_sec == 3.0
        assert call_summary.agent_summary is not None
        assert call_summary.customer_summary is not None
        assert len(call_summary.activity_state_intervals) > 0
        assert call_summary.model_version == "1.0.0"
        assert call_summary.processing_timestamp != ""
        assert "number_of_customer_inactive_intervals" in {
            "number_of_agent_inactive_intervals": call_summary.number_of_agent_inactive_intervals,
            "number_of_customer_inactive_intervals": call_summary.number_of_customer_inactive_intervals,
        }


# ══════════════════════════════════════════════════════════════════════
# PRACTICAL RUNTIME VERIFICATION
# ══════════════════════════════════════════════════════════════════════
class TestPracticalActivity:
    """Practical test using real EmotionEngine with alternating activity."""

    @pytest.fixture(scope="class")
    def engine(self):
        eng = EmotionEngine()
        eng.load()
        return eng

    def test_alternating_activity_practical(self, engine):
        """8-second alternating-activity call produces correct intervals."""
        sr = 16000
        duration = 8.0
        agent, customer = _make_waveforms(duration,
            agent_active_ranges=[(0.0, 2.0), (4.0, 6.0)],
            customer_active_ranges=[(2.0, 4.0), (4.0, 6.0)])

        load_result = StereoLoadResult(
            original_audio=OriginalStereoAudio(
                waveform=np.stack([agent, customer]),
                original_sr=sr, duration_sec=duration,
                filename="alternating.wav", format="wav",
            ),
            model_audio=ModelStereoAudio(
                agent_waveform=agent, customer_waveform=customer,
                model_sr=sr, duration_sec=duration,
            ),
            validation=ValidationResult(is_valid=True),
        )

        proc_result = process_stereo_call(
            load_result, engine,
            WindowConfig(window_duration_sec=2.0, hop_duration_sec=1.0),
            SpeechActivityConfig(),
        )

        agent_summary = compute_speaker_summary(proc_result.agent_result, duration)
        cust_summary = compute_speaker_summary(proc_result.customer_result, duration)

        states = build_activity_states(
            proc_result.window_results,
            agent_waveform=agent, customer_waveform=customer,
            sample_rate=sr, call_duration_sec=duration,
        )
        intervals = merge_activity_states(states)
        inact_meta = compute_inactivity_metadata(intervals)

        call_summary = compute_call_summary(
            proc_result, agent_summary, cust_summary, intervals, inact_meta,
        )

        # Verification
        assert call_summary.call_duration_sec == 8.0
        assert agent_summary.detected_active_duration_sec <= 8.0
        assert cust_summary.detected_active_duration_sec <= 8.0

        # 6-8s is neither_active
        for iv in intervals:
            if iv.start_time >= 5.5:
                assert iv.state == "neither_active"

        # No Hold event
        for iv in intervals:
            assert "hold" not in iv.state.lower()

        # Full coverage
        total = sum(iv.duration_sec for iv in intervals)
        assert abs(total - 8.0) < 0.1
