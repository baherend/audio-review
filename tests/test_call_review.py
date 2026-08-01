"""
Tests for Phase VII — Final Call Review.

Covers aggregation, deduplication, status determination, ordering,
terminology, limitations, and full synthetic integration.
"""

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.demo.schemas import (
    ActivityInterval,
    CallDurationResult,
    CallSummary,
    ChannelSerSummary,
    FinalCallReview,
    FinalCallReviewConfig,
    HoldCandidate,
    HoldDetectionResult,
    HoldDurationViolation,
    HoldPolicyResult,
    HoldReviewCandidate,
    HoldReviewSection,
    OperationalAlert,
    OperationalAlertResult,
    SpeakerActivitySection,
    SpeakerSummary,
    TimelineItem,
    UncountedOverDuration,
)
from src.demo.call_review import build_call_review


# ══════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════

def _empty_call_summary():
    """Empty CallSummary."""
    return CallSummary()


def _call_summary_with_activity():
    """CallSummary with activity data."""
    return CallSummary(
        call_duration_sec=100.0,
        agent_active_only_duration_sec=30.0,
        customer_active_only_duration_sec=20.0,
        both_active_duration_sec=10.0,
        neither_active_duration_sec=40.0,
        activity_state_intervals=[
            ActivityInterval(
                start_time=0.0, end_time=30.0, duration_sec=30.0,
                state="agent_active_only", agent_active=True, customer_active=False,
            ),
            ActivityInterval(
                start_time=30.0, end_time=50.0, duration_sec=20.0,
                state="customer_active_only", agent_active=False, customer_active=True,
            ),
            ActivityInterval(
                start_time=50.0, end_time=60.0, duration_sec=10.0,
                state="both_active_interval", agent_active=True, customer_active=True,
            ),
            ActivityInterval(
                start_time=60.0, end_time=100.0, duration_sec=40.0,
                state="neither_active", agent_active=False, customer_active=False,
            ),
        ],
    )


def _call_summary_with_ser():
    """CallSummary with SER data."""
    cs = _call_summary_with_activity()
    cs.agent_summary = SpeakerSummary(
        role="Agent", channel="left",
        analyzed_windows=10, uncertain_windows=1,
        dominant_dataset_label="Moderate Vocal Activation",
        dataset_label_distribution={"Low Vocal Activation": 2, "Moderate Vocal Activation": 7, "High Vocal Activation": 1},
        dataset_label_percentages={"Low Vocal Activation": 0.2, "Moderate Vocal Activation": 0.7, "High Vocal Activation": 0.1},
        average_confidence=0.75,
    )
    cs.customer_summary = SpeakerSummary(
        role="Customer", channel="right",
        analyzed_windows=8, uncertain_windows=2,
        dominant_dataset_label="High Vocal Activation",
        dataset_label_distribution={"Low Vocal Activation": 1, "Moderate Vocal Activation": 3, "High Vocal Activation": 4},
        dataset_label_percentages={"Low Vocal Activation": 0.125, "Moderate Vocal Activation": 0.375, "High Vocal Activation": 0.5},
        average_confidence=0.65,
    )
    return cs


def _empty_detection():
    """Empty HoldDetectionResult."""
    return HoldDetectionResult()


def _empty_policy():
    """Empty HoldPolicyResult."""
    return HoldPolicyResult()


def _empty_alerts():
    """Empty OperationalAlertResult."""
    return OperationalAlertResult()


def _alert(alert_id, alert_type, severity, start=None, end=None, dur=None,
           source="silence", title="", metadata=None, requires_review=False):
    """Create an OperationalAlert."""
    return OperationalAlert(
        alert_id=alert_id,
        alert_type=alert_type,
        severity=severity,
        start_time=start,
        end_time=end,
        duration_sec=dur,
        title=title,
        source_category=source,
        metadata=metadata or {},
        requires_manual_review=requires_review,
    )


def _candidate(cid, start, end, classification="likely_hold_candidate",
               evidence=None, reaches_end=False):
    """Create a HoldCandidate."""
    return HoldCandidate(
        candidate_id=cid,
        start_time=start,
        end_time=end,
        duration_sec=round(end - start, 6),
        classification=classification,
        evidence_types=evidence or ["prolonged_silence"],
        agent_inactive=True,
        customer_state="inactive",
        confidence=0.5,
        detection_notes="test",
        agent_return_detected=True,
        reaches_call_end=reaches_end,
    )


# ══════════════════════════════════════════════════════════════════════
# EMPTY AND BASIC TESTS
# ══════════════════════════════════════════════════════════════════════
class TestEmptyAndBasic:
    """Tests for empty inputs and basic construction."""

    def test_empty_call_review(self):
        """Empty inputs produce a valid FinalCallReview."""
        result = build_call_review(
            call_duration_sec=0.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
        )
        assert isinstance(result, FinalCallReview)
        assert result.call_duration_sec == 0.0
        assert result.call_type == "unknown"
        assert result.review_status == "no_immediate_review"
        assert len(result.review_reasons) == 0
        assert len(result.evidence_limitations) > 0

    def test_no_alerts_no_immediate_review(self):
        """No alerts → no_immediate_review."""
        result = build_call_review(
            call_duration_sec=100.0, call_type="technical",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
        )
        assert result.review_status == "no_immediate_review"

    def test_missing_call_id(self):
        """Missing call_id is handled as None."""
        result = build_call_review(
            call_duration_sec=60.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
        )
        assert result.call_id is None

    def test_invalid_negative_duration(self):
        """Negative call duration is normalized to 0."""
        result = build_call_review(
            call_duration_sec=-10.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
        )
        assert result.call_duration_sec == 0.0

    def test_invalid_call_type_normalized(self):
        """Invalid call type is normalized to 'unknown'."""
        result = build_call_review(
            call_duration_sec=60.0, call_type="invalid_type",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
        )
        assert result.call_type == "unknown"

    def test_empty_hold_result(self):
        """Empty Hold result is handled."""
        result = build_call_review(
            call_duration_sec=60.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
        )
        assert result.hold_review.total_candidates == 0

    def test_empty_operational_alert_result(self):
        """Empty alert result is handled."""
        result = build_call_review(
            call_duration_sec=60.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
        )
        assert result.operational_alerts_summary["total_alerts"] == 0

    def test_missing_speaker_summaries(self):
        """Missing speaker summaries are handled as None."""
        result = build_call_review(
            call_duration_sec=60.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
        )
        assert result.ser_summary["agent"] is None
        assert result.ser_summary["customer"] is None

    def test_limitations_always_included(self):
        """Evidence limitations are always present."""
        result = build_call_review(
            call_duration_sec=60.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
        )
        assert len(result.evidence_limitations) >= 5
        codes = [l["code"] for l in result.evidence_limitations]
        assert "hold_audio_derived" in codes
        assert "no_numeric_score" in codes


# ══════════════════════════════════════════════════════════════════════
# STATUS DETERMINATION TESTS
# ══════════════════════════════════════════════════════════════════════
class TestReviewStatus:
    """Tests for review status determination."""

    def test_high_alert_priority_review(self):
        """High-severity alert → priority_review."""
        alerts = OperationalAlertResult(
            alerts=[_alert("a1", "dead_air", "high", 10.0, 70.0, 60.0)],
            alert_counts_by_severity={"high": 1},
        )
        result = build_call_review(
            call_duration_sec=100.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=alerts,
        )
        assert result.review_status == "priority_review"

    def test_hold_count_violation_priority_review(self):
        """Hold count violation → priority_review."""
        policy = HoldPolicyResult(counted_hold_events=3, count_limit=2, count_exceeded=True)
        result = build_call_review(
            call_duration_sec=100.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=policy,
            operational_alert_result=_empty_alerts(),
        )
        assert result.review_status == "priority_review"

    def test_hold_duration_violation_priority_review(self):
        """Hold duration violation → priority_review."""
        policy = HoldPolicyResult(
            duration_violations=[
                HoldDurationViolation("h1", 10.0, 140.0, 130.0, 120.0, 10.0)
            ]
        )
        result = build_call_review(
            call_duration_sec=150.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=policy,
            operational_alert_result=_empty_alerts(),
        )
        assert result.review_status == "priority_review"

    def test_warning_alert_review_suggested(self):
        """Warning alert → review_suggested."""
        alerts = OperationalAlertResult(
            alerts=[_alert("a1", "dead_air", "warning", 10.0, 60.0, 50.0)],
            alert_counts_by_severity={"warning": 1},
        )
        result = build_call_review(
            call_duration_sec=100.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=alerts,
        )
        assert result.review_status == "review_suggested"

    def test_manual_review_suggested_propagation(self):
        """manual_review_suggested=True propagates."""
        alerts = OperationalAlertResult(
            manual_review_suggested=True,
        )
        result = build_call_review(
            call_duration_sec=100.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=alerts,
        )
        assert result.manual_review_suggested is True
        assert result.review_status == "review_suggested"

    def test_uncounted_over_duration_review_suggested(self):
        """Uncounted over-duration → review_suggested."""
        policy = HoldPolicyResult(
            uncounted_over_duration=[
                UncountedOverDuration("h2", 100.0, 250.0, 150.0,
                                     "uncertain_hold_candidate", 30.0, True)
            ]
        )
        result = build_call_review(
            call_duration_sec=260.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=policy,
            operational_alert_result=_empty_alerts(),
        )
        assert result.review_status == "review_suggested"

    def test_above_expected_duration_review_suggested(self):
        """Above-expected call duration → review_suggested."""
        alerts = OperationalAlertResult(
            call_duration_result=CallDurationResult(
                "technical", 700.0, 300.0, 600.0, "above_expected_range"
            ),
        )
        result = build_call_review(
            call_duration_sec=700.0, call_type="technical",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=alerts,
        )
        assert result.review_status == "review_suggested"

    def test_info_only_no_immediate_review(self):
        """Info-only alerts → no_immediate_review."""
        alerts = OperationalAlertResult(
            alerts=[_alert("a1", "extended_overlap", "info", 5.0, 15.0, 10.0)],
            alert_counts_by_severity={"info": 1},
        )
        result = build_call_review(
            call_duration_sec=100.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=alerts,
        )
        assert result.review_status == "no_immediate_review"


# ══════════════════════════════════════════════════════════════════════
# DEDUPLICATION TESTS
# ══════════════════════════════════════════════════════════════════════
class TestDeduplication:
    """Tests for review reason deduplication."""

    def test_hold_in_all_three_deduplicated(self):
        """Same Hold in detection, policy, and alerts → one primary reason."""
        candidate = _candidate("h1", 10.0, 140.0)
        detection = HoldDetectionResult(
            candidates=[candidate], total_candidates=1, likely_count=1
        )
        policy = HoldPolicyResult(
            counted_hold_events=1,
            duration_violations=[
                HoldDurationViolation("h1", 10.0, 140.0, 130.0, 120.0, 10.0)
            ],
            counted_hold_ranges=[(10.0, 140.0)],
        )
        alerts = OperationalAlertResult(
            alerts=[_alert(
                "a1", "hold_duration_violation", "high",
                10.0, 140.0, 130.0, "hold", "Hold Exceeded",
                {"candidate_id": "h1"},
            )],
        )
        result = build_call_review(
            call_duration_sec=150.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=detection,
            hold_policy_result=policy,
            operational_alert_result=alerts,
        )
        # Should have exactly one Hold duration reason, not three
        hold_reasons = [r for r in result.review_reasons
                       if r.reason_code == "hold_duration_exceeded"]
        assert len(hold_reasons) == 1
        assert "h1" in hold_reasons[0].related_candidate_ids

    def test_unrelated_overlapping_events_separate(self):
        """Unrelated events with overlapping timestamps remain separate."""
        detection = HoldDetectionResult(
            candidates=[_candidate("h1", 10.0, 50.0)],
            total_candidates=1, likely_count=1,
        )
        policy = HoldPolicyResult()
        alerts = OperationalAlertResult(
            alerts=[
                _alert("a1", "dead_air", "warning", 10.0, 50.0, 40.0, "silence",
                       "Dead Air", {"candidate_id": ""}),
                _alert("a2", "extended_overlap", "info", 10.0, 50.0, 40.0, "overlap",
                       "Overlap"),
            ],
            alert_counts_by_severity={"warning": 1, "info": 1},
        )
        result = build_call_review(
            call_duration_sec=60.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=detection,
            hold_policy_result=policy,
            operational_alert_result=alerts,
        )
        # Dead Air, Overlap, and Hold candidate should be separate reasons
        reason_codes = [r.reason_code for r in result.review_reasons]
        assert "dead_air" in reason_codes
        assert "extended_overlap" in reason_codes
        assert "likely_hold_candidate" in reason_codes

    def test_uncertain_hold_not_duplicated(self):
        """Uncertain Hold in detection and uncounted over-duration → one reason."""
        candidate = _candidate("h2", 50.0, 200.0,
                              classification="uncertain_hold_candidate")
        detection = HoldDetectionResult(
            candidates=[candidate], total_candidates=1, uncertain_count=1
        )
        policy = HoldPolicyResult(
            uncounted_over_duration=[
                UncountedOverDuration("h2", 50.0, 200.0, 150.0,
                                     "uncertain_hold_candidate", 30.0, False)
            ],
        )
        alerts = OperationalAlertResult()
        result = build_call_review(
            call_duration_sec=210.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=detection,
            hold_policy_result=policy,
            operational_alert_result=alerts,
        )
        h2_reasons = [r for r in result.review_reasons if "h2" in str(r.related_candidate_ids)]
        assert len(h2_reasons) == 1


# ══════════════════════════════════════════════════════════════════════
# HOLD REVIEW SECTION TESTS
# ══════════════════════════════════════════════════════════════════════
class TestHoldReview:
    """Tests for Hold review section."""

    def test_likely_hold_candidate_wording(self):
        """Likely Hold candidate uses correct wording."""
        candidate = _candidate("h1", 10.0, 100.0, "likely_hold_candidate")
        detection = HoldDetectionResult(
            candidates=[candidate], total_candidates=1, likely_count=1
        )
        # Provide counted_hold_ranges so counted_by_policy=True
        policy = HoldPolicyResult(
            counted_hold_events=1,
            counted_hold_ranges=[(10.0, 100.0)],
        )
        result = build_call_review(
            call_duration_sec=110.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=detection,
            hold_policy_result=policy,
            operational_alert_result=_empty_alerts(),
        )
        assert result.hold_review.likely_candidates == 1
        c = result.hold_review.candidates[0]
        assert c.classification == "likely_hold_candidate"
        assert c.counted_by_policy is True

    def test_uncertain_hold_candidate_wording(self):
        """Uncertain Hold candidate uses correct wording."""
        candidate = _candidate("h2", 20.0, 80.0, "uncertain_hold_candidate")
        detection = HoldDetectionResult(
            candidates=[candidate], total_candidates=1, uncertain_count=1
        )
        result = build_call_review(
            call_duration_sec=90.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=detection,
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
        )
        assert result.hold_review.uncertain_candidates == 1
        c = result.hold_review.candidates[0]
        assert c.classification == "uncertain_hold_candidate"
        assert c.counted_by_policy is False

    def test_counted_hold_range_included(self):
        """Counted Hold ranges are included in the review."""
        policy = HoldPolicyResult(
            counted_hold_events=1,
            counted_hold_ranges=[(10.0, 90.0)],
        )
        result = build_call_review(
            call_duration_sec=100.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=policy,
            operational_alert_result=_empty_alerts(),
        )
        assert len(result.hold_review.counted_hold_ranges) == 1
        assert result.hold_review.counted_hold_ranges[0] == (10.0, 90.0)

    def test_hold_count_violation(self):
        """Hold count violation is reflected."""
        policy = HoldPolicyResult(
            counted_hold_events=3, count_limit=2, count_exceeded=True,
        )
        result = build_call_review(
            call_duration_sec=100.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=policy,
            operational_alert_result=_empty_alerts(),
        )
        assert result.hold_review.count_exceeded is True

    def test_uncounted_over_duration_item(self):
        """Uncounted over-duration item is included."""
        policy = HoldPolicyResult(
            uncounted_over_duration=[
                UncountedOverDuration("h3", 50.0, 200.0, 150.0,
                                     "uncertain_hold_candidate", 30.0, True)
            ],
        )
        result = build_call_review(
            call_duration_sec=210.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=policy,
            operational_alert_result=_empty_alerts(),
        )
        assert len(result.hold_review.uncounted_over_duration) == 1

    def test_hold_described_as_audio_derived(self):
        """Hold is described as audio-derived, not confirmed."""
        candidate = _candidate("h1", 10.0, 50.0)
        detection = HoldDetectionResult(
            candidates=[candidate], total_candidates=1, likely_count=1
        )
        result = build_call_review(
            call_duration_sec=60.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=detection,
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
        )
        # Check that limitations mention audio-derived hold
        hold_lim = [l for l in result.evidence_limitations if l["code"] == "hold_audio_derived"]
        assert len(hold_lim) == 1
        assert "audio" in hold_lim[0]["text"].lower()


# ══════════════════════════════════════════════════════════════════════
# OPERATIONAL ALERTS ORDERING TESTS
# ══════════════════════════════════════════════════════════════════════
class TestAlertOrdering:
    """Tests for alert ordering in the summary."""

    def test_severity_ordering(self):
        """Alerts ordered by severity: high → warning → info."""
        alerts = OperationalAlertResult(
            alerts=[
                _alert("a3", "extended_overlap", "info", 5.0, 15.0, 10.0),
                _alert("a1", "dead_air", "high", 10.0, 70.0, 60.0),
                _alert("a2", "dead_air", "warning", 20.0, 60.0, 40.0),
            ],
            alert_counts_by_severity={"high": 1, "warning": 1, "info": 1},
        )
        result = build_call_review(
            call_duration_sec=80.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=alerts,
        )
        sorted_a = result.operational_alerts_summary["sorted_alerts"]
        severities = [a["severity"] for a in sorted_a]
        assert severities == ["high", "warning", "info"]

    def test_same_severity_chronological(self):
        """Same-severity alerts ordered by start_time."""
        alerts = OperationalAlertResult(
            alerts=[
                _alert("a2", "dead_air", "warning", 30.0, 70.0, 40.0),
                _alert("a1", "dead_air", "warning", 10.0, 50.0, 40.0),
            ],
            alert_counts_by_severity={"warning": 2},
        )
        result = build_call_review(
            call_duration_sec=80.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=alerts,
        )
        sorted_a = result.operational_alerts_summary["sorted_alerts"]
        starts = [a["start_time"] for a in sorted_a]
        assert starts == [10.0, 30.0]

    def test_alerts_without_timestamps_last(self):
        """Alerts without timestamps go last."""
        alerts = OperationalAlertResult(
            alerts=[
                _alert("a1", "hold_count_violation", "high"),
                _alert("a2", "dead_air", "high", 10.0, 70.0, 60.0),
            ],
            alert_counts_by_severity={"high": 2},
        )
        result = build_call_review(
            call_duration_sec=80.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=alerts,
        )
        sorted_a = result.operational_alerts_summary["sorted_alerts"]
        # a2 has start_time=10.0, a1 has start_time=None
        assert sorted_a[0]["alert_id"] == "a2"
        assert sorted_a[1]["alert_id"] == "a1"


# ══════════════════════════════════════════════════════════════════════
# SPEAKER ACTIVITY TESTS
# ══════════════════════════════════════════════════════════════════════
class TestSpeakerActivity:
    """Tests for speaker activity section."""

    def test_activity_aggregation(self):
        """Activity durations are correctly aggregated."""
        result = build_call_review(
            call_duration_sec=100.0, call_type="unknown",
            call_summary=_call_summary_with_activity(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
        )
        sa = result.speaker_activity
        assert sa.agent_active_duration_sec == 40.0  # 30 + 10
        assert sa.customer_active_duration_sec == 30.0  # 20 + 10
        assert sa.overlap_duration_sec == 10.0
        assert sa.both_inactive_duration_sec == 40.0

    def test_binary_activity_terminology(self):
        """Uses channel activity terms, not 'talk time'."""
        result = build_call_review(
            call_duration_sec=100.0, call_type="unknown",
            call_summary=_call_summary_with_activity(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
        )
        sa = result.speaker_activity
        assert hasattr(sa, "agent_active_duration_sec")
        assert hasattr(sa, "customer_active_duration_sec")
        assert hasattr(sa, "overlap_duration_sec")
        assert hasattr(sa, "both_inactive_duration_sec")

    def test_zero_duration_call(self):
        """Zero-duration call produces zero ratios."""
        result = build_call_review(
            call_duration_sec=0.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
        )
        sa = result.speaker_activity
        assert sa.agent_activity_ratio == 0.0
        assert sa.call_duration_sec == 0.0


# ══════════════════════════════════════════════════════════════════════
# SER SUMMARY TESTS
# ══════════════════════════════════════════════════════════════════════
class TestSerSummary:
    """Tests for SER summary section."""

    def test_agent_ser_summary(self):
        """Agent SER summary contains correct fields."""
        result = build_call_review(
            call_duration_sec=100.0, call_type="unknown",
            call_summary=_call_summary_with_ser(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
        )
        agent = result.ser_summary["agent"]
        assert agent.role == "Agent"
        assert agent.dominant_label == "Moderate Vocal Activation"
        assert agent.low_emotion_ratio == 0.2
        assert agent.neutral_emotion_ratio == 0.7
        assert agent.high_emotion_ratio == 0.1
        assert agent.analyzed_windows == 10

    def test_customer_ser_summary(self):
        """Customer SER summary contains correct fields."""
        result = build_call_review(
            call_duration_sec=100.0, call_type="unknown",
            call_summary=_call_summary_with_ser(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
        )
        customer = result.ser_summary["customer"]
        assert customer.role == "Customer"
        assert customer.dominant_label == "High Vocal Activation"
        assert customer.high_emotion_ratio == 0.5

    def test_no_emotional_interpretation(self):
        """SER summary includes safeguard note."""
        result = build_call_review(
            call_duration_sec=100.0, call_type="unknown",
            call_summary=_call_summary_with_ser(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
        )
        assert "Do not interpret" in result.ser_summary["note"]

    def test_no_emotion_in_limitations(self):
        """Limitations explicitly state SER labels are coarse only."""
        result = build_call_review(
            call_duration_sec=60.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
        )
        ser_lim = [l for l in result.evidence_limitations if l["code"] == "ser_coarse_labels"]
        assert len(ser_lim) == 1
        assert "anger" in ser_lim[0]["text"].lower() or "satisfaction" in ser_lim[0]["text"].lower()


# ══════════════════════════════════════════════════════════════════════
# TIMELINE TESTS
# ══════════════════════════════════════════════════════════════════════
class TestTimeline:
    """Tests for timeline construction."""

    def test_dead_air_timeline_item(self):
        """Dead Air appears in timeline."""
        alerts = OperationalAlertResult(
            alerts=[_alert("a1", "dead_air", "warning", 10.0, 60.0, 50.0, "silence",
                          "Dead Air")],
        )
        result = build_call_review(
            call_duration_sec=70.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=alerts,
        )
        types = [t.item_type for t in result.timeline]
        assert "dead_air" in types

    def test_extended_overlap_timeline_item(self):
        """Extended Overlap appears in timeline."""
        alerts = OperationalAlertResult(
            alerts=[_alert("a1", "extended_overlap", "info", 5.0, 20.0, 15.0, "overlap",
                          "Overlap")],
        )
        result = build_call_review(
            call_duration_sec=25.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=alerts,
        )
        types = [t.item_type for t in result.timeline]
        assert "extended_overlap" in types

    def test_event_reaching_call_end(self):
        """Event reaching call end is included."""
        candidate = _candidate("h1", 50.0, 100.0, reaches_end=True)
        detection = HoldDetectionResult(
            candidates=[candidate], total_candidates=1, likely_count=1
        )
        result = build_call_review(
            call_duration_sec=100.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=detection,
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
        )
        reaching = [t for t in result.timeline if t.item_type == "hold_candidate"]
        assert len(reaching) == 1
        assert reaching[0].end_time == 100.0

    def test_zero_duration_upstream_item(self):
        """Zero-duration upstream items are handled."""
        candidate = _candidate("h1", 50.0, 50.0)
        candidate.duration_sec = 0.0
        detection = HoldDetectionResult(
            candidates=[candidate], total_candidates=1
        )
        result = build_call_review(
            call_duration_sec=60.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=detection,
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
        )
        # Should not crash
        assert isinstance(result.timeline, list)

    def test_maximum_timeline_items(self):
        """Timeline respects max_timeline_items."""
        candidates = [
            _candidate(f"h{i}", i * 10.0, i * 10.0 + 5.0)
            for i in range(20)
        ]
        detection = HoldDetectionResult(
            candidates=candidates, total_candidates=20
        )
        config = FinalCallReviewConfig(max_timeline_items=5)
        result = build_call_review(
            call_duration_sec=250.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=detection,
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
            config=config,
        )
        assert len(result.timeline) <= 5

    def test_deterministic_timeline_ordering(self):
        """Timeline is deterministic across calls."""
        candidates = [
            _candidate("h2", 30.0, 60.0),
            _candidate("h1", 10.0, 25.0),
        ]
        detection = HoldDetectionResult(
            candidates=candidates, total_candidates=2
        )
        r1 = build_call_review(
            call_duration_sec=70.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=detection,
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
        )
        r2 = build_call_review(
            call_duration_sec=70.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=detection,
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
        )
        t1 = [(t.start_time, t.end_time) for t in r1.timeline]
        t2 = [(t.start_time, t.end_time) for t in r2.timeline]
        assert t1 == t2


# ══════════════════════════════════════════════════════════════════════
# REVIEW REASONS WORDING TESTS
# ══════════════════════════════════════════════════════════════════════
class TestReviewReasons:
    """Tests for review reason content and wording."""

    def test_hold_duration_violation_creates_one_reason(self):
        """Hold duration violation creates exactly one primary reason."""
        policy = HoldPolicyResult(
            counted_hold_events=1,
            duration_violations=[
                HoldDurationViolation("h1", 10.0, 140.0, 130.0, 120.0, 10.0)
            ],
        )
        detection = HoldDetectionResult(
            candidates=[_candidate("h1", 10.0, 140.0)],
            total_candidates=1, likely_count=1,
        )
        alerts = OperationalAlertResult(
            alerts=[_alert(
                "a1", "hold_duration_violation", "high",
                10.0, 140.0, 130.0, "hold", "Hold Exceeded",
                {"candidate_id": "h1"},
            )],
        )
        result = build_call_review(
            call_duration_sec=150.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=detection,
            hold_policy_result=policy,
            operational_alert_result=alerts,
        )
        reasons = [r for r in result.review_reasons if "h1" in str(r.related_candidate_ids)]
        assert len(reasons) == 1
        assert reasons[0].severity == "high"

    def test_no_numeric_score(self):
        """No numeric QA score is produced."""
        result = build_call_review(
            call_duration_sec=100.0, call_type="unknown",
            call_summary=_call_summary_with_ser(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
        )
        assert not hasattr(result, "score")
        assert not hasattr(result, "qa_score")
        assert not hasattr(result, "pass_fail")

    def test_call_duration_below_expected(self):
        """Call duration below expected creates a review reason."""
        alerts = OperationalAlertResult(
            call_duration_result=CallDurationResult(
                "technical", 200.0, 300.0, 600.0, "below_expected_range"
            ),
        )
        result = build_call_review(
            call_duration_sec=200.0, call_type="technical",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=alerts,
        )
        codes = [r.reason_code for r in result.review_reasons]
        assert "call_duration_below_expected" in codes

    def test_call_duration_above_expected(self):
        """Call duration above expected creates a review reason."""
        alerts = OperationalAlertResult(
            call_duration_result=CallDurationResult(
                "technical", 700.0, 300.0, 600.0, "above_expected_range"
            ),
        )
        result = build_call_review(
            call_duration_sec=700.0, call_type="technical",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=alerts,
        )
        codes = [r.reason_code for r in result.review_reasons]
        assert "call_duration_above_expected" in codes

    def test_call_duration_within_range_no_reason(self):
        """Call duration within range creates no duration reason."""
        alerts = OperationalAlertResult(
            call_duration_result=CallDurationResult(
                "technical", 450.0, 300.0, 600.0, "within_expected_range"
            ),
        )
        result = build_call_review(
            call_duration_sec=450.0, call_type="technical",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=alerts,
        )
        codes = [r.reason_code for r in result.review_reasons]
        assert "call_duration_above_expected" not in codes
        assert "call_duration_below_expected" not in codes


# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION TESTS
# ══════════════════════════════════════════════════════════════════════
class TestConfiguration:
    """Tests for FinalCallReviewConfig."""

    def test_configuration_snapshot_included(self):
        """Configuration snapshot is included by default."""
        result = build_call_review(
            call_duration_sec=60.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
        )
        assert "priority_severities" in result.configuration_snapshot

    def test_configuration_snapshot_excluded(self):
        """Configuration snapshot excluded when configured."""
        config = FinalCallReviewConfig(include_configuration_snapshot=False)
        result = build_call_review(
            call_duration_sec=60.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
            config=config,
        )
        assert len(result.configuration_snapshot) == 0

    def test_exclude_uncertain_candidates(self):
        """Uncertain Hold candidates excluded when configured."""
        candidates = [
            _candidate("h1", 10.0, 50.0, "likely_hold_candidate"),
            _candidate("h2", 60.0, 100.0, "uncertain_hold_candidate"),
        ]
        detection = HoldDetectionResult(
            candidates=candidates, total_candidates=2,
            likely_count=1, uncertain_count=1,
        )
        config = FinalCallReviewConfig(include_uncertain_hold_candidates=False)
        result = build_call_review(
            call_duration_sec=110.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=detection,
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
            config=config,
        )
        assert result.hold_review.total_candidates == 1
        assert result.hold_review.candidates[0].candidate_id == "h1"

    def test_max_review_reasons(self):
        """Review reasons capped at max_review_reasons."""
        candidates = [
            _candidate(f"h{i}", i * 10.0, i * 10.0 + 5.0)
            for i in range(20)
        ]
        detection = HoldDetectionResult(
            candidates=candidates, total_candidates=20
        )
        config = FinalCallReviewConfig(max_review_reasons=3)
        result = build_call_review(
            call_duration_sec=250.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=detection,
            hold_policy_result=_empty_policy(),
            operational_alert_result=_empty_alerts(),
            config=config,
        )
        assert len(result.review_reasons) <= 3


# ══════════════════════════════════════════════════════════════════════
# DETERMINISTIC OUTPUT TEST
# ══════════════════════════════════════════════════════════════════════
class TestDeterminism:
    """Tests for deterministic output."""

    def test_deterministic_repeated_output(self):
        """Same inputs produce identical outputs."""
        detection = HoldDetectionResult(
            candidates=[_candidate("h1", 10.0, 100.0)],
            total_candidates=1, likely_count=1,
        )
        policy = HoldPolicyResult(
            counted_hold_events=1,
            duration_violations=[
                HoldDurationViolation("h1", 10.0, 100.0, 90.0, 80.0, 10.0)
            ],
            counted_hold_ranges=[(10.0, 100.0)],
        )
        alerts = OperationalAlertResult(
            alerts=[
                _alert("a1", "dead_air", "warning", 50.0, 90.0, 40.0, "silence", "Dead Air"),
                _alert("a2", "extended_overlap", "info", 5.0, 15.0, 10.0, "overlap", "Overlap"),
            ],
            alert_counts_by_severity={"warning": 1, "info": 1},
        )
        inputs = dict(
            call_duration_sec=110.0, call_type="technical",
            call_summary=_call_summary_with_activity(),
            hold_detection_result=detection,
            hold_policy_result=policy,
            operational_alert_result=alerts,
        )
        r1 = build_call_review(**inputs)
        r2 = build_call_review(**inputs)
        assert r1.review_status == r2.review_status
        assert len(r1.review_reasons) == len(r2.review_reasons)
        assert len(r1.timeline) == len(r2.timeline)
        r1_codes = [r.reason_code for r in r1.review_reasons]
        r2_codes = [r.reason_code for r in r2.review_reasons]
        assert r1_codes == r2_codes


# ══════════════════════════════════════════════════════════════════════
# FULL SYNTHETIC INTEGRATION
# ══════════════════════════════════════════════════════════════════════
class TestIntegration:
    """Full synthetic call integration test."""

    def test_complete_synthetic_call(self):
        """Full synthetic call with all required elements.

        - technical call within expected range
        - two Extended Overlap events
        - one Likely Hold candidate exceeding duration
        - one Dead Air event reaching call end
        - Hold policy violation
        - operational alerts
        - Agent and Customer SER summaries
        - speaker activity summaries
        """
        # Call summary with activity
        cs = CallSummary(
            call_duration_sec=400.0,
            agent_active_only_duration_sec=100.0,
            customer_active_only_duration_sec=50.0,
            both_active_duration_sec=80.0,
            neither_active_duration_sec=170.0,
            activity_state_intervals=[
                ActivityInterval(0.0, 80.0, 80.0, "both_active_interval", True, True),
                ActivityInterval(80.0, 120.0, 40.0, "customer_active_only", False, True),
                ActivityInterval(120.0, 200.0, 80.0, "both_active_interval", True, True),
                ActivityInterval(200.0, 340.0, 140.0, "neither_active", False, False),
                ActivityInterval(340.0, 400.0, 60.0, "agent_active_only", True, False),
            ],
        )
        cs.agent_summary = SpeakerSummary(
            role="Agent", channel="left",
            analyzed_windows=15, uncertain_windows=1,
            dominant_dataset_label="Moderate Vocal Activation",
            dataset_label_distribution={"Low Vocal Activation": 3, "Moderate Vocal Activation": 10, "High Vocal Activation": 2},
            dataset_label_percentages={"Low Vocal Activation": 0.2, "Moderate Vocal Activation": 0.667, "High Vocal Activation": 0.133},
            average_confidence=0.72,
        )
        cs.customer_summary = SpeakerSummary(
            role="Customer", channel="right",
            analyzed_windows=12, uncertain_windows=2,
            dominant_dataset_label="High Vocal Activation",
            dataset_label_distribution={"Low Vocal Activation": 2, "Moderate Vocal Activation": 4, "High Vocal Activation": 6},
            dataset_label_percentages={"Low Vocal Activation": 0.167, "Moderate Vocal Activation": 0.333, "High Vocal Activation": 0.5},
            average_confidence=0.68,
        )

        # Hold detection
        candidate = _candidate("h1", 200.0, 340.0, "likely_hold_candidate",
                              evidence=["music_like_audio"], reaches_end=True)
        detection = HoldDetectionResult(
            candidates=[candidate], total_candidates=1, likely_count=1
        )

        # Hold policy with violation
        policy = HoldPolicyResult(
            counted_hold_events=1, count_limit=2,
            duration_violations=[
                HoldDurationViolation("h1", 200.0, 340.0, 140.0, 120.0, 20.0)
            ],
            counted_hold_ranges=[(200.0, 340.0)],
        )

        # Operational alerts
        alerts = OperationalAlertResult(
            alerts=[
                _alert("a1", "extended_overlap", "info", 0.0, 80.0, 80.0, "overlap",
                       "Extended Simultaneous Speech"),
                _alert("a2", "extended_overlap", "info", 120.0, 200.0, 80.0, "overlap",
                       "Extended Simultaneous Speech"),
                _alert("a3", "dead_air", "high", 200.0, 340.0, 140.0, "silence",
                       "Dead Air Detected", {"candidate_id": ""}),
                _alert("a4", "hold_duration_violation", "high", 200.0, 340.0, 140.0, "hold",
                       "Hold Duration Exceeded", {"candidate_id": "h1"}),
            ],
            alert_counts_by_severity={"high": 2, "info": 2},
            manual_review_suggested=True,
            call_duration_result=CallDurationResult(
                "technical", 400.0, 300.0, 600.0, "within_expected_range"
            ),
        )

        result = build_call_review(
            call_duration_sec=400.0, call_type="technical",
            call_summary=cs,
            hold_detection_result=detection,
            hold_policy_result=policy,
            operational_alert_result=alerts,
            call_id="CALL-001",
        )

        # ── Assertions ──
        # Status
        assert result.review_status == "priority_review"
        assert result.call_id == "CALL-001"
        assert result.call_type == "technical"

        # Hold review
        hr = result.hold_review
        assert hr.total_candidates == 1
        assert hr.likely_candidates == 1
        assert hr.count_exceeded is False
        assert len(hr.duration_violations) == 1

        # Deduplication: Hold violation in detection + policy + alerts → one reason
        hold_reasons = [r for r in result.review_reasons
                       if r.reason_code == "hold_duration_exceeded"]
        assert len(hold_reasons) == 1
        assert hold_reasons[0].severity == "high"

        # Dead Air reason exists
        dead_reasons = [r for r in result.review_reasons if r.reason_code == "dead_air"]
        assert len(dead_reasons) >= 1

        # Extended Overlap reasons
        overlap_reasons = [r for r in result.review_reasons
                          if r.reason_code == "extended_overlap"]
        assert len(overlap_reasons) == 2

        # No duplicate Hold problem
        all_hold_reasons = [r for r in result.review_reasons if r.source_category == "hold"]
        assert len(all_hold_reasons) == 1

        # No numeric score
        assert not hasattr(result, "score")
        assert not hasattr(result, "qa_score")

        # Limitations present
        assert len(result.evidence_limitations) >= 5

        # Hold described as audio-derived
        hold_lim = [l for l in result.evidence_limitations if l["code"] == "hold_audio_derived"]
        assert len(hold_lim) == 1

        # Speaker activity
        sa = result.speaker_activity
        assert sa.agent_active_duration_sec == 180.0  # 100 + 80
        assert sa.overlap_duration_sec == 80.0

        # SER
        assert result.ser_summary["agent"].dominant_label == "Moderate Vocal Activation"
        assert result.ser_summary["customer"].dominant_label == "High Vocal Activation"

        # Timeline has items
        assert len(result.timeline) > 0

        # All alert IDs unique
        ids = [a["alert_id"] for a in result.operational_alerts_summary["sorted_alerts"]]
        assert len(ids) == len(set(ids))


# ══════════════════════════════════════════════════════════════════════
# PHASE VII-A: DEAD AIR SEVERITY PRESERVATION
# ══════════════════════════════════════════════════════════════════════
class TestDeadAirSeverityPreservation:
    """Phase VII-A: Final Call Review never changes upstream alert severity."""

    def test_59_9s_dead_air_preserves_warning(self):
        """59.9s Dead Air preserves warning severity (< 60s threshold)."""
        alerts = OperationalAlertResult(
            alerts=[_alert("a1", "dead_air", "warning", 10.0, 69.9, 59.9,
                          "silence", "Dead Air")],
            alert_counts_by_severity={"warning": 1},
        )
        result = build_call_review(
            call_duration_sec=70.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=alerts,
        )
        sorted_a = result.operational_alerts_summary["sorted_alerts"]
        assert sorted_a[0]["severity"] == "warning"

    def test_60s_dead_air_preserves_high(self):
        """Exactly 60s Dead Air preserves high severity."""
        alerts = OperationalAlertResult(
            alerts=[_alert("a1", "dead_air", "high", 10.0, 70.0, 60.0,
                          "silence", "Dead Air")],
            alert_counts_by_severity={"high": 1},
        )
        result = build_call_review(
            call_duration_sec=75.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=alerts,
        )
        sorted_a = result.operational_alerts_summary["sorted_alerts"]
        assert sorted_a[0]["severity"] == "high"

    def test_140s_dead_air_preserves_high(self):
        """140s Dead Air preserves high severity."""
        alerts = OperationalAlertResult(
            alerts=[_alert("a1", "dead_air", "high", 200.0, 340.0, 140.0,
                          "silence", "Dead Air")],
            alert_counts_by_severity={"high": 1},
        )
        result = build_call_review(
            call_duration_sec=350.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=alerts,
        )
        sorted_a = result.operational_alerts_summary["sorted_alerts"]
        assert sorted_a[0]["severity"] == "high"
        assert sorted_a[0]["duration_sec"] == 140.0

    def test_review_never_changes_upstream_severity(self):
        """Final Call Review never changes any upstream alert severity."""
        alerts = OperationalAlertResult(
            alerts=[
                _alert("a1", "dead_air", "info", 5.0, 15.0, 10.0, "silence", "Info"),
                _alert("a2", "extended_overlap", "warning", 20.0, 35.0, 15.0, "overlap", "Warn"),
                _alert("a3", "dead_air", "high", 40.0, 100.0, 60.0, "silence", "High"),
            ],
            alert_counts_by_severity={"info": 1, "warning": 1, "high": 1},
        )
        result = build_call_review(
            call_duration_sec=110.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=_empty_detection(),
            hold_policy_result=_empty_policy(),
            operational_alert_result=alerts,
        )
        sorted_a = result.operational_alerts_summary["sorted_alerts"]
        severities = {a["alert_id"]: a["severity"] for a in sorted_a}
        assert severities["a1"] == "info"
        assert severities["a2"] == "warning"
        assert severities["a3"] == "high"


# ══════════════════════════════════════════════════════════════════════
# PHASE VII-A: COUNTED_BY_POLICY DERIVATION
# ══════════════════════════════════════════════════════════════════════
class TestCountedByPolicy:
    """Phase VII-A: counted_by_policy derived from structured HoldPolicyResult."""

    def test_likely_candidate_not_in_ranges_not_counted(self):
        """Likely candidate NOT in counted_hold_ranges → counted_by_policy=False."""
        candidate = _candidate("h1", 10.0, 50.0, "likely_hold_candidate")
        detection = HoldDetectionResult(
            candidates=[candidate], total_candidates=1, likely_count=1
        )
        # Policy has no counted_hold_ranges (empty)
        policy = HoldPolicyResult(counted_hold_events=0)
        result = build_call_review(
            call_duration_sec=60.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=detection,
            hold_policy_result=policy,
            operational_alert_result=_empty_alerts(),
        )
        c = result.hold_review.candidates[0]
        assert c.counted_by_policy is False

    def test_candidate_in_ranges_counted(self):
        """Candidate present in counted_hold_ranges → counted_by_policy=True."""
        candidate = _candidate("h1", 10.0, 50.0, "likely_hold_candidate")
        detection = HoldDetectionResult(
            candidates=[candidate], total_candidates=1, likely_count=1
        )
        policy = HoldPolicyResult(
            counted_hold_events=1,
            counted_hold_ranges=[(10.0, 50.0)],
        )
        result = build_call_review(
            call_duration_sec=60.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=detection,
            hold_policy_result=policy,
            operational_alert_result=_empty_alerts(),
        )
        c = result.hold_review.candidates[0]
        assert c.counted_by_policy is True

    def test_uncertain_candidate_in_ranges_counted(self):
        """Uncertain candidate in counted_hold_ranges → counted_by_policy=True."""
        candidate = _candidate("h1", 10.0, 50.0, "uncertain_hold_candidate")
        detection = HoldDetectionResult(
            candidates=[candidate], total_candidates=1, uncertain_count=1
        )
        policy = HoldPolicyResult(
            counted_hold_events=1,
            counted_hold_ranges=[(10.0, 50.0)],
        )
        result = build_call_review(
            call_duration_sec=60.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=detection,
            hold_policy_result=policy,
            operational_alert_result=_empty_alerts(),
        )
        c = result.hold_review.candidates[0]
        assert c.counted_by_policy is True

    def test_timestamp_tolerance_boundary(self):
        """Candidate within 0.5s tolerance of counted range → counted."""
        candidate = _candidate("h1", 10.3, 49.8, "likely_hold_candidate")
        detection = HoldDetectionResult(
            candidates=[candidate], total_candidates=1, likely_count=1
        )
        # Range is 10.0–50.0, candidate is 10.3–49.8
        # |10.3 - 10.0| = 0.3 <= 0.5, |49.8 - 50.0| = 0.2 <= 0.5 → counted
        policy = HoldPolicyResult(
            counted_hold_events=1,
            counted_hold_ranges=[(10.0, 50.0)],
        )
        result = build_call_review(
            call_duration_sec=60.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=detection,
            hold_policy_result=policy,
            operational_alert_result=_empty_alerts(),
        )
        c = result.hold_review.candidates[0]
        assert c.counted_by_policy is True

    def test_timestamp_outside_tolerance_not_counted(self):
        """Candidate outside 0.5s tolerance → not counted."""
        candidate = _candidate("h1", 11.0, 49.0, "likely_hold_candidate")
        detection = HoldDetectionResult(
            candidates=[candidate], total_candidates=1, likely_count=1
        )
        # Range is 10.0–50.0, candidate is 11.0–49.0
        # |11.0 - 10.0| = 1.0 > 0.5 → not counted
        policy = HoldPolicyResult(
            counted_hold_events=1,
            counted_hold_ranges=[(10.0, 50.0)],
        )
        result = build_call_review(
            call_duration_sec=60.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=detection,
            hold_policy_result=policy,
            operational_alert_result=_empty_alerts(),
        )
        c = result.hold_review.candidates[0]
        assert c.counted_by_policy is False

    def test_unrelated_overlapping_range_not_counted(self):
        """Unrelated Hold range overlapping in time does not mark candidate."""
        candidate = _candidate("h1", 30.0, 70.0, "likely_hold_candidate")
        detection = HoldDetectionResult(
            candidates=[candidate], total_candidates=1, likely_count=1
        )
        # Different range (0.0–40.0) overlaps but has different boundaries
        policy = HoldPolicyResult(
            counted_hold_events=1,
            counted_hold_ranges=[(0.0, 40.0)],
        )
        result = build_call_review(
            call_duration_sec=80.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=detection,
            hold_policy_result=policy,
            operational_alert_result=_empty_alerts(),
        )
        c = result.hold_review.candidates[0]
        assert c.counted_by_policy is False

    def test_identical_ranges_multiple_counted(self):
        """Multiple candidates with identical ranges both counted."""
        c1 = _candidate("h1", 10.0, 50.0, "likely_hold_candidate")
        c2 = _candidate("h2", 10.0, 50.0, "likely_hold_candidate")
        detection = HoldDetectionResult(
            candidates=[c1, c2], total_candidates=2, likely_count=2
        )
        policy = HoldPolicyResult(
            counted_hold_events=2,
            counted_hold_ranges=[(10.0, 50.0), (10.0, 50.0)],
        )
        result = build_call_review(
            call_duration_sec=60.0, call_type="unknown",
            call_summary=_empty_call_summary(),
            hold_detection_result=detection,
            hold_policy_result=policy,
            operational_alert_result=_empty_alerts(),
        )
        for c in result.hold_review.candidates:
            assert c.counted_by_policy is True
