"""
Tests for Phase VI-A — Corrected Audio-Based Operational Alerts.

Phase VI-A semantic correction:
- Single-speaker intervals (customer_active_only, agent_active_only)
  do NOT generate silence alerts. One speaker being active while the
  other is silent does not prove problematic silence.
- Dead Air severity escalates at two thresholds: warning (≥45s) and
  high (≥60s).
- Dead Air overlapping ≥90% with Hold events is suppressed.
- manual_review_suggested is computed from structured requires_manual_review
  fields.

Tests encoding incorrect Phase VI silence semantics have been replaced
with corrected tests. No approved prior-phase tests were removed,
skipped, weakened, or renamed.
"""

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.demo.schemas import (
    ActivityInterval,
    HoldCandidate,
    HoldDurationViolation,
    HoldPolicyResult,
    OperationalAlertConfig,
    UncountedOverDuration,
)
from src.demo.operational_alerts import (
    generate_operational_alerts,
    _compute_overlap_ratio,
)


# ══════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════

def _iv(start, end, state):
    """Create an ActivityInterval."""
    agent = state in ("agent_active_only", "both_active_interval")
    customer = state in ("customer_active_only", "both_active_interval")
    return ActivityInterval(
        start_time=start,
        end_time=end,
        duration_sec=round(end - start, 6),
        state=state,
        agent_active=agent,
        customer_active=customer,
    )


def _empty_policy():
    """Empty HoldPolicyResult."""
    return HoldPolicyResult()


def _policy(counted=0, limit=2, violations=None, uncounted=None,
            count_exceeded=False):
    """Create a HoldPolicyResult for testing."""
    return HoldPolicyResult(
        counted_hold_events=counted,
        count_limit=limit,
        count_exceeded=count_exceeded,
        duration_violations=violations or [],
        uncounted_over_duration=uncounted or [],
        compliant=not count_exceeded and not violations,
        review_required=count_exceeded or bool(violations) or bool(uncounted),
    )


def _config(**kwargs):
    """Create OperationalAlertConfig with overrides."""
    defaults = {}
    defaults.update(kwargs)
    return OperationalAlertConfig(**defaults)


# ══════════════════════════════════════════════════════════════════════
# SINGLE-SPEAKER NO-SILENCE-ALERT TESTS (Phase VI-A correction)
# ══════════════════════════════════════════════════════════════════════
class TestSingleSpeakerNoSilenceAlert:
    """Phase VI-A: single-speaker intervals must NOT generate silence alerts.

    customer_active_only means the Customer channel is active (speaking or
    producing audio) while the Agent is inactive. This does NOT prove
    problematic Agent silence — the Agent may be listening.

    agent_active_only means the Agent channel is active while the Customer
    is inactive. This does NOT prove problematic Customer silence.
    """

    def test_customer_active_only_no_agent_silence_alert(self):
        """125s customer_active_only → NO long_agent_silence alert."""
        intervals = [
            _iv(0.0, 10.0, "agent_active_only"),
            _iv(10.0, 135.0, "customer_active_only"),  # 125s
            _iv(135.0, 145.0, "agent_active_only"),
        ]
        result = generate_operational_alerts(
            intervals, _empty_policy(), 145.0, "unknown"
        )
        agent_silence = [a for a in result.alerts
                        if a.alert_type == "long_agent_silence"]
        assert len(agent_silence) == 0, (
            "customer_active_only must not generate long_agent_silence"
        )

    def test_agent_active_only_no_customer_silence_alert(self):
        """90s agent_active_only → NO long_customer_silence alert."""
        intervals = [
            _iv(0.0, 10.0, "customer_active_only"),
            _iv(10.0, 100.0, "agent_active_only"),  # 90s
            _iv(100.0, 110.0, "customer_active_only"),
        ]
        result = generate_operational_alerts(
            intervals, _empty_policy(), 110.0, "unknown"
        )
        cust_silence = [a for a in result.alerts
                       if a.alert_type == "long_customer_silence"]
        assert len(cust_silence) == 0, (
            "agent_active_only must not generate long_customer_silence"
        )

    def test_long_customer_active_only_no_silence_alerts(self):
        """Very long customer_active_only → no silence alerts of any kind."""
        intervals = [
            _iv(0.0, 5.0, "agent_active_only"),
            _iv(5.0, 200.0, "customer_active_only"),  # 195s
            _iv(200.0, 210.0, "agent_active_only"),
        ]
        result = generate_operational_alerts(
            intervals, _empty_policy(), 210.0, "unknown"
        )
        silence_alerts = [a for a in result.alerts
                         if "silence" in a.alert_type]
        assert len(silence_alerts) == 0

    def test_long_agent_active_only_no_silence_alerts(self):
        """Very long agent_active_only → no silence alerts of any kind."""
        intervals = [
            _iv(0.0, 5.0, "customer_active_only"),
            _iv(5.0, 200.0, "agent_active_only"),  # 195s
            _iv(200.0, 210.0, "customer_active_only"),
        ]
        result = generate_operational_alerts(
            intervals, _empty_policy(), 210.0, "unknown"
        )
        silence_alerts = [a for a in result.alerts
                         if "silence" in a.alert_type]
        assert len(silence_alerts) == 0

    def test_no_long_agent_silence_alert_type_exists(self):
        """The alert_type 'long_agent_silence' is never generated."""
        intervals = [
            _iv(0.0, 100.0, "customer_active_only"),
        ]
        result = generate_operational_alerts(
            intervals, _empty_policy(), 100.0, "unknown"
        )
        types = {a.alert_type for a in result.alerts}
        assert "long_agent_silence" not in types

    def test_no_long_customer_silence_alert_type_exists(self):
        """The alert_type 'long_customer_silence' is never generated."""
        intervals = [
            _iv(0.0, 100.0, "agent_active_only"),
        ]
        result = generate_operational_alerts(
            intervals, _empty_policy(), 100.0, "unknown"
        )
        types = {a.alert_type for a in result.alerts}
        assert "long_customer_silence" not in types


# ══════════════════════════════════════════════════════════════════════
# DEAD AIR TESTS (Phase VI-A: two-tier severity)
# ══════════════════════════════════════════════════════════════════════
class TestDeadAir:
    """Tests for Dead Air detection with warning/high severity tiers."""

    def test_below_warning_threshold(self):
        """30s both inactive → no alert."""
        intervals = [
            _iv(0.0, 10.0, "agent_active_only"),
            _iv(10.0, 40.0, "neither_active"),  # 30s < 45s
            _iv(40.0, 50.0, "agent_active_only"),
        ]
        result = generate_operational_alerts(
            intervals, _empty_policy(), 50.0, "unknown"
        )
        dead = [a for a in result.alerts if a.alert_type == "dead_air"]
        assert len(dead) == 0

    def test_exactly_warning_threshold(self):
        """45s both inactive → warning."""
        intervals = [
            _iv(0.0, 10.0, "agent_active_only"),
            _iv(10.0, 55.0, "neither_active"),  # 45s
            _iv(55.0, 65.0, "agent_active_only"),
        ]
        result = generate_operational_alerts(
            intervals, _empty_policy(), 65.0, "unknown"
        )
        dead = [a for a in result.alerts if a.alert_type == "dead_air"]
        assert len(dead) == 1
        assert dead[0].duration_sec == 45.0
        assert dead[0].severity == "warning"

    def test_between_warning_and_high(self):
        """50s both inactive → warning (below 60s high)."""
        intervals = [
            _iv(0.0, 10.0, "agent_active_only"),
            _iv(10.0, 60.0, "neither_active"),  # 50s
            _iv(60.0, 70.0, "agent_active_only"),
        ]
        result = generate_operational_alerts(
            intervals, _empty_policy(), 70.0, "unknown"
        )
        dead = [a for a in result.alerts if a.alert_type == "dead_air"]
        assert len(dead) == 1
        assert dead[0].severity == "warning"

    def test_exactly_high_threshold(self):
        """60s both inactive → high."""
        intervals = [
            _iv(0.0, 10.0, "agent_active_only"),
            _iv(10.0, 70.0, "neither_active"),  # 60s
            _iv(70.0, 80.0, "agent_active_only"),
        ]
        result = generate_operational_alerts(
            intervals, _empty_policy(), 80.0, "unknown"
        )
        dead = [a for a in result.alerts if a.alert_type == "dead_air"]
        assert len(dead) == 1
        assert dead[0].severity == "high"
        assert dead[0].duration_sec == 60.0

    def test_above_high_threshold(self):
        """90s both inactive → high."""
        intervals = [
            _iv(0.0, 10.0, "agent_active_only"),
            _iv(10.0, 100.0, "neither_active"),  # 90s
            _iv(100.0, 110.0, "agent_active_only"),
        ]
        result = generate_operational_alerts(
            intervals, _empty_policy(), 110.0, "unknown"
        )
        dead = [a for a in result.alerts if a.alert_type == "dead_air"]
        assert len(dead) == 1
        assert dead[0].severity == "high"

    def test_dead_air_contains_reassurance_recommendation(self):
        """Dead Air alert includes reassurance review recommendation."""
        intervals = [
            _iv(0.0, 10.0, "agent_active_only"),
            _iv(10.0, 60.0, "neither_active"),  # 50s
            _iv(60.0, 70.0, "agent_active_only"),
        ]
        result = generate_operational_alerts(
            intervals, _empty_policy(), 70.0, "unknown"
        )
        dead = [a for a in result.alerts if a.alert_type == "dead_air"]
        assert len(dead) == 1
        assert "reassurance" in dead[0].review_recommendation.lower()

    def test_dead_air_does_not_duplicate_silence_alert(self):
        """Dead Air does not generate a separate silence alert."""
        intervals = [
            _iv(0.0, 10.0, "agent_active_only"),
            _iv(10.0, 70.0, "neither_active"),  # 60s
            _iv(70.0, 80.0, "agent_active_only"),
        ]
        result = generate_operational_alerts(
            intervals, _empty_policy(), 80.0, "unknown"
        )
        dead = [a for a in result.alerts if a.alert_type == "dead_air"]
        silence = [a for a in result.alerts if "silence" in a.alert_type]
        assert len(dead) == 1
        assert len(silence) == 0  # No separate silence alert

    def test_merges_contiguous_dead_air(self):
        """Contiguous neither_active intervals merge into one alert."""
        intervals = [
            _iv(0.0, 5.0, "agent_active_only"),
            _iv(5.0, 25.0, "neither_active"),
            _iv(25.0, 55.0, "neither_active"),  # continuous from 5-55
            _iv(55.0, 65.0, "agent_active_only"),
        ]
        result = generate_operational_alerts(
            intervals, _empty_policy(), 65.0, "unknown"
        )
        dead = [a for a in result.alerts if a.alert_type == "dead_air"]
        assert len(dead) == 1
        assert dead[0].start_time == 5.0
        assert dead[0].end_time == 55.0
        assert dead[0].duration_sec == 50.0

    def test_custom_dead_air_thresholds(self):
        """Custom warning and high thresholds are applied."""
        intervals = [
            _iv(0.0, 5.0, "agent_active_only"),
            _iv(5.0, 20.0, "neither_active"),  # 15s
            _iv(20.0, 30.0, "agent_active_only"),
        ]
        # Default: 15s < 45s → no alert
        r1 = generate_operational_alerts(intervals, _empty_policy(), 30.0, "unknown")
        dead1 = [a for a in r1.alerts if a.alert_type == "dead_air"]
        assert len(dead1) == 0

        # Custom warning=10s, high=20s: 15s >= 10s → warning
        r2 = generate_operational_alerts(
            intervals, _empty_policy(), 30.0, "unknown",
            config=_config(dead_air_warning_sec=10.0, dead_air_high_sec=20.0)
        )
        dead2 = [a for a in r2.alerts if a.alert_type == "dead_air"]
        assert len(dead2) == 1
        assert dead2[0].severity == "warning"

        # Custom warning=5s, high=10s: 15s >= 10s → high
        r3 = generate_operational_alerts(
            intervals, _empty_policy(), 30.0, "unknown",
            config=_config(dead_air_warning_sec=5.0, dead_air_high_sec=10.0)
        )
        dead3 = [a for a in r3.alerts if a.alert_type == "dead_air"]
        assert len(dead3) == 1
        assert dead3[0].severity == "high"


# ══════════════════════════════════════════════════════════════════════
# EXTENDED OVERLAP TESTS
# ══════════════════════════════════════════════════════════════════════
class TestExtendedOverlap:
    """Tests for Extended Overlap detection."""

    def test_below_threshold(self):
        """5s both active → no alert."""
        intervals = [
            _iv(0.0, 5.0, "neither_active"),
            _iv(5.0, 10.0, "both_active_interval"),  # 5s < 10s
            _iv(10.0, 15.0, "neither_active"),
        ]
        result = generate_operational_alerts(
            intervals, _empty_policy(), 15.0, "unknown"
        )
        overlap = [a for a in result.alerts if a.alert_type == "extended_overlap"]
        assert len(overlap) == 0

    def test_exactly_threshold(self):
        """10s both active → alert."""
        intervals = [
            _iv(0.0, 5.0, "neither_active"),
            _iv(5.0, 15.0, "both_active_interval"),  # 10s
            _iv(15.0, 20.0, "neither_active"),
        ]
        result = generate_operational_alerts(
            intervals, _empty_policy(), 20.0, "unknown"
        )
        overlap = [a for a in result.alerts if a.alert_type == "extended_overlap"]
        assert len(overlap) == 1
        assert overlap[0].duration_sec == 10.0
        assert overlap[0].severity == "info"

    def test_above_threshold(self):
        """30s both active → alert."""
        intervals = [
            _iv(0.0, 5.0, "neither_active"),
            _iv(5.0, 35.0, "both_active_interval"),  # 30s
            _iv(35.0, 40.0, "neither_active"),
        ]
        result = generate_operational_alerts(
            intervals, _empty_policy(), 40.0, "unknown"
        )
        overlap = [a for a in result.alerts if a.alert_type == "extended_overlap"]
        assert len(overlap) == 1
        assert overlap[0].duration_sec == 30.0


# ══════════════════════════════════════════════════════════════════════
# HOLD PRECEDENCE AND SUPPRESSION TESTS
# ══════════════════════════════════════════════════════════════════════
class TestHoldPrecedence:
    """Tests for Hold precedence and Dead Air suppression."""

    def test_hold_violation_suppresses_overlapping_dead_air(self):
        """Dead Air overlapping ≥90% with Hold violation is suppressed."""
        # Dead Air 70-195s (125s) overlaps with Hold violation 70-195s (125s)
        # Overlap ratio = 125/125 = 1.0 ≥ 0.9 → suppressed
        intervals = [
            _iv(0.0, 20.0, "both_active_interval"),
            _iv(20.0, 50.0, "neither_active"),
            _iv(50.0, 70.0, "both_active_interval"),
            _iv(70.0, 195.0, "neither_active"),
            _iv(195.0, 210.0, "agent_active_only"),
            _iv(210.0, 340.5, "neither_active"),
        ]
        violation = HoldDurationViolation(
            candidate_id="hold_002", start_time=70.0, end_time=195.0,
            actual_duration_sec=125.0, allowed_duration_sec=120.0,
            excess_duration_sec=5.0,
        )
        policy = _policy(counted=1, limit=2, violations=[violation])
        result = generate_operational_alerts(intervals, policy, 340.5, "unknown")

        dead = [a for a in result.alerts if a.alert_type == "dead_air"]
        # 20-50s (30s) is below threshold → no alert
        # 70-195s (125s) overlaps 100% with Hold → suppressed
        # 210-340.5s (130.5s) has no Hold overlap → retained
        assert len(dead) == 1
        assert abs(dead[0].start_time - 210.0) < 0.1

    def test_partial_overlap_below_suppression_threshold(self):
        """Dead Air with <90% Hold overlap is retained with metadata."""
        # Dead Air 50-150s (100s), Hold 80-120s (40s)
        # Overlap = 40/100 = 40% < 90% → retained
        intervals = [
            _iv(0.0, 10.0, "agent_active_only"),
            _iv(10.0, 160.0, "neither_active"),  # 150s dead air
            _iv(160.0, 170.0, "agent_active_only"),
        ]
        violation = HoldDurationViolation(
            candidate_id="hold_001", start_time=80.0, end_time=120.0,
            actual_duration_sec=40.0, allowed_duration_sec=30.0,
            excess_duration_sec=10.0,
        )
        policy = _policy(violations=[violation])
        result = generate_operational_alerts(intervals, policy, 170.0, "unknown")

        dead = [a for a in result.alerts if a.alert_type == "dead_air"]
        assert len(dead) == 1
        assert dead[0].metadata.get("hold_overlap_ratio", 0) > 0

    def test_exact_overlap_ratio_boundary(self):
        """At exactly 90% overlap, Dead Air is suppressed."""
        # Dead Air 0-100s (100s), Hold 10-100s (90s)
        # Overlap = 90/100 = 90% → suppressed at boundary
        intervals = [
            _iv(0.0, 100.0, "neither_active"),
            _iv(100.0, 110.0, "agent_active_only"),
        ]
        violation = HoldDurationViolation(
            candidate_id="hold_001", start_time=10.0, end_time=100.0,
            actual_duration_sec=90.0, allowed_duration_sec=80.0,
            excess_duration_sec=10.0,
        )
        policy = _policy(violations=[violation])
        result = generate_operational_alerts(intervals, policy, 110.0, "unknown")

        dead = [a for a in result.alerts if a.alert_type == "dead_air"]
        assert len(dead) == 0  # Suppressed at 90%

    def test_custom_suppression_ratio(self):
        """Custom suppression ratio is applied."""
        intervals = [
            _iv(0.0, 100.0, "neither_active"),
            _iv(100.0, 110.0, "agent_active_only"),
        ]
        violation = HoldDurationViolation(
            candidate_id="hold_001", start_time=50.0, end_time=100.0,
            actual_duration_sec=50.0, allowed_duration_sec=40.0,
            excess_duration_sec=10.0,
        )
        policy = _policy(violations=[violation])

        # Default ratio 0.90: overlap = 50/100 = 50% < 90% → retained
        r1 = generate_operational_alerts(intervals, policy, 110.0, "unknown")
        dead1 = [a for a in r1.alerts if a.alert_type == "dead_air"]
        assert len(dead1) == 1

        # Custom ratio 0.40: overlap = 50/100 = 50% ≥ 40% → suppressed
        r2 = generate_operational_alerts(
            intervals, policy, 110.0, "unknown",
            config=_config(hold_suppression_overlap_ratio=0.40)
        )
        dead2 = [a for a in r2.alerts if a.alert_type == "dead_air"]
        assert len(dead2) == 0

    def test_hold_alerts_preserved_when_dead_air_suppressed(self):
        """Hold alerts remain when overlapping Dead Air is suppressed."""
        intervals = [
            _iv(0.0, 10.0, "agent_active_only"),
            _iv(10.0, 140.0, "neither_active"),  # 130s
            _iv(140.0, 150.0, "agent_active_only"),
        ]
        violation = HoldDurationViolation(
            candidate_id="hold_001", start_time=10.0, end_time=140.0,
            actual_duration_sec=130.0, allowed_duration_sec=120.0,
            excess_duration_sec=10.0,
        )
        policy = _policy(violations=[violation])
        result = generate_operational_alerts(intervals, policy, 150.0, "unknown")

        hold = [a for a in result.alerts if a.source_category == "hold"]
        dead = [a for a in result.alerts if a.alert_type == "dead_air"]
        assert len(hold) == 1  # Hold preserved
        assert len(dead) == 0  # Dead Air suppressed

    def test_uncounted_hold_suppresses_dead_air(self):
        """Uncounted over-duration Hold also suppresses Dead Air."""
        intervals = [
            _iv(0.0, 10.0, "agent_active_only"),
            _iv(10.0, 200.0, "neither_active"),  # 190s
            _iv(200.0, 210.0, "agent_active_only"),
        ]
        uncounted = UncountedOverDuration(
            candidate_id="hold_003", start_time=10.0, end_time=200.0,
            duration_sec=190.0, classification="uncertain_hold_candidate",
            exceeds_duration_by_sec=70.0, reaches_call_end=False,
        )
        policy = _policy(uncounted=[uncounted])
        result = generate_operational_alerts(intervals, policy, 210.0, "unknown")

        dead = [a for a in result.alerts if a.alert_type == "dead_air"]
        hold = [a for a in result.alerts if a.source_category == "hold"]
        assert len(hold) == 1
        assert len(dead) == 0  # Suppressed

    def test_no_hold_events_no_suppression(self):
        """Without Hold events, no Dead Air suppression occurs."""
        intervals = [
            _iv(0.0, 10.0, "agent_active_only"),
            _iv(10.0, 70.0, "neither_active"),  # 60s
            _iv(70.0, 80.0, "agent_active_only"),
        ]
        result = generate_operational_alerts(
            intervals, _empty_policy(), 80.0, "unknown"
        )
        dead = [a for a in result.alerts if a.alert_type == "dead_air"]
        assert len(dead) == 1
        assert dead[0].severity == "high"

    def test_compliant_hold_suppresses_dead_air(self):
        """A counted Hold event within duration limit suppresses Dead Air.

        Hold candidate: 10–90s (80s), within 120s limit → counted, compliant.
        neither_active interval: 10–90s → would be Dead Air but is a Hold.
        Expected: no Dead Air alert for this interval because the
        counted_hold_ranges from HoldPolicyResult provides the time range.
        """
        from src.demo.schemas import HoldCandidate

        # Build a HoldCandidate that will be counted (likely, within limit)
        candidate = HoldCandidate(
            candidate_id="hold_001",
            start_time=10.0,
            end_time=90.0,
            duration_sec=80.0,
            classification="likely_hold_candidate",
            evidence_types=["music_like_audio"],
            agent_inactive=True,
            customer_state="music_like",
            confidence=0.55,
            detection_notes="test",
            agent_return_detected=True,
        )
        # Policy: 1 counted event, within limit → compliant
        from src.demo.hold_policy import evaluate_hold_policy
        from src.demo.schemas import HoldPolicyConfig
        policy = evaluate_hold_policy(
            [candidate],
            HoldPolicyConfig(max_hold_count=2, max_hold_duration_sec=120.0)
        )
        assert policy.counted_hold_events == 1
        assert len(policy.counted_hold_ranges) == 1
        assert policy.counted_hold_ranges[0] == (10.0, 90.0)

        # Intervals: neither_active from 10–90s (same as Hold)
        intervals = [
            _iv(0.0, 10.0, "agent_active_only"),
            _iv(10.0, 90.0, "neither_active"),  # 80s, this is the Hold
            _iv(90.0, 100.0, "agent_active_only"),
        ]
        result = generate_operational_alerts(
            intervals, policy, 100.0, "unknown"
        )
        dead = [a for a in result.alerts if a.alert_type == "dead_air"]
        # Dead Air should be suppressed because counted_hold_ranges
        # provides (10.0, 90.0) which overlaps 100% of the 10–90s interval
        assert len(dead) == 0, (
            f"Dead Air should be suppressed for compliant Hold interval. "
            f"Got {len(dead)} dead air alerts: "
            f"{[(a.start_time, a.end_time) for a in dead]}"
        )

    def test_compliant_hold_partial_overlap(self):
        """Compliant Hold partially overlapping Dead Air adds metadata."""
        from src.demo.schemas import HoldCandidate
        from src.demo.hold_policy import evaluate_hold_policy
        from src.demo.schemas import HoldPolicyConfig

        candidate = HoldCandidate(
            candidate_id="hold_001",
            start_time=30.0,
            end_time=70.0,
            duration_sec=40.0,
            classification="likely_hold_candidate",
            evidence_types=["music_like_audio"],
            agent_inactive=True,
            customer_state="music_like",
            confidence=0.55,
            detection_notes="test",
            agent_return_detected=True,
        )
        policy = evaluate_hold_policy(
            [candidate],
            HoldPolicyConfig(max_hold_count=2, max_hold_duration_sec=120.0)
        )

        # Dead Air from 0–100s (100s), Hold from 30–70s (40s overlap = 40%)
        intervals = [
            _iv(0.0, 100.0, "neither_active"),
            _iv(100.0, 110.0, "agent_active_only"),
        ]
        result = generate_operational_alerts(
            intervals, policy, 110.0, "unknown"
        )
        dead = [a for a in result.alerts if a.alert_type == "dead_air"]
        assert len(dead) == 1
        assert dead[0].metadata.get("hold_overlap_ratio", 0) > 0


# ══════════════════════════════════════════════════════════════════════
# MERGING AND BOUNDARY TESTS
# ══════════════════════════════════════════════════════════════════════
class TestMergingAndBoundaries:
    """Tests for merging, boundaries, and edge cases."""

    def test_multiple_separate_events(self):
        """Multiple separate Dead Air events produce separate alerts."""
        intervals = [
            _iv(0.0, 5.0, "agent_active_only"),
            _iv(5.0, 55.0, "neither_active"),   # 50s dead air 1
            _iv(55.0, 65.0, "agent_active_only"),
            _iv(65.0, 120.0, "neither_active"),  # 55s dead air 2
            _iv(120.0, 130.0, "agent_active_only"),
        ]
        result = generate_operational_alerts(
            intervals, _empty_policy(), 130.0, "unknown"
        )
        dead = [a for a in result.alerts if a.alert_type == "dead_air"]
        assert len(dead) == 2
        assert dead[0].start_time == 5.0
        assert dead[0].end_time == 55.0
        assert dead[1].start_time == 65.0
        assert dead[1].end_time == 120.0

    def test_no_merging_across_speech_returns(self):
        """Dead Air intervals separated by speech do not merge."""
        intervals = [
            _iv(0.0, 5.0, "neither_active"),
            _iv(5.0, 10.0, "agent_active_only"),
            _iv(10.0, 60.0, "neither_active"),
            _iv(60.0, 65.0, "agent_active_only"),
            _iv(65.0, 115.0, "neither_active"),
        ]
        result = generate_operational_alerts(
            intervals, _empty_policy(), 115.0, "unknown"
        )
        dead = [a for a in result.alerts if a.alert_type == "dead_air"]
        assert len(dead) == 2

    def test_event_reaching_call_end(self):
        """Dead Air reaching call end is flagged in metadata."""
        intervals = [
            _iv(0.0, 10.0, "agent_active_only"),
            _iv(10.0, 60.0, "neither_active"),
        ]
        result = generate_operational_alerts(
            intervals, _empty_policy(), 60.0, "unknown"
        )
        dead = [a for a in result.alerts if a.alert_type == "dead_air"]
        assert len(dead) == 1
        assert dead[0].metadata["reaches_call_end"] is True

    def test_empty_intervals(self):
        """Empty interval list produces no alerts."""
        result = generate_operational_alerts(
            [], _empty_policy(), 60.0, "unknown"
        )
        assert len(result.alerts) == 0
        assert result.manual_review_suggested is False

    def test_zero_duration_interval_skipped(self):
        """Zero-duration intervals do not produce alerts."""
        intervals = [
            ActivityInterval(
                start_time=10.0, end_time=10.0, duration_sec=0.0,
                state="neither_active", agent_active=False, customer_active=False,
            ),
        ]
        result = generate_operational_alerts(
            intervals, _empty_policy(), 60.0, "unknown"
        )
        dead = [a for a in result.alerts if a.alert_type == "dead_air"]
        assert len(dead) == 0

    def test_invalid_call_duration_handled(self):
        """Zero or negative call duration does not crash."""
        intervals = [_iv(0.0, 5.0, "agent_active_only")]
        result = generate_operational_alerts(
            intervals, _empty_policy(), 0.0, "unknown"
        )
        assert isinstance(result.alerts, list)


# ══════════════════════════════════════════════════════════════════════
# OVERLAP RATIO COMPUTATION TESTS
# ══════════════════════════════════════════════════════════════════════
class TestOverlapRatio:
    """Tests for _compute_overlap_ratio helper."""

    def test_full_overlap(self):
        """Identical ranges → ratio 1.0."""
        assert _compute_overlap_ratio((10, 50), (10, 50)) == 1.0

    def test_partial_overlap(self):
        """50% overlap → ratio 0.5."""
        assert _compute_overlap_ratio((10, 30), (0, 40)) == 0.5

    def test_no_overlap(self):
        """Non-overlapping ranges → ratio 0.0."""
        assert _compute_overlap_ratio((0, 10), (20, 30)) == 0.0

    def test_contained_overlap(self):
        """Range A fully inside range B → ratio depends on B's duration."""
        # A=20-30 (10s), B=0-100 (100s) → overlap 10/100 = 0.1
        assert abs(_compute_overlap_ratio((20, 30), (0, 100)) - 0.1) < 1e-6

    def test_zero_duration_range_b(self):
        """Zero-duration range B → ratio 0.0."""
        assert _compute_overlap_ratio((0, 10), (5, 5)) == 0.0


# ══════════════════════════════════════════════════════════════════════
# CALL DURATION TESTS
# ══════════════════════════════════════════════════════════════════════
class TestCallDuration:
    """Tests for Call Duration Review."""

    def test_technical_below_range(self):
        """Technical call under 300s → below_expected_range."""
        result = generate_operational_alerts(
            [], _empty_policy(), 200.0, "technical"
        )
        cdr = result.call_duration_result
        assert cdr.comparison == "below_expected_range"
        duration_alerts = [a for a in result.alerts if a.alert_type == "call_duration"]
        assert len(duration_alerts) == 1
        assert duration_alerts[0].severity == "info"

    def test_technical_within_range(self):
        """Technical call 450s → within_expected_range."""
        result = generate_operational_alerts(
            [], _empty_policy(), 450.0, "technical"
        )
        assert result.call_duration_result.comparison == "within_expected_range"
        duration_alerts = [a for a in result.alerts if a.alert_type == "call_duration"]
        assert len(duration_alerts) == 0

    def test_technical_above_range(self):
        """Technical call over 600s → above_expected_range."""
        result = generate_operational_alerts(
            [], _empty_policy(), 700.0, "technical"
        )
        assert result.call_duration_result.comparison == "above_expected_range"
        duration_alerts = [a for a in result.alerts if a.alert_type == "call_duration"]
        assert len(duration_alerts) == 1
        assert duration_alerts[0].severity == "warning"

    def test_non_technical_below_range(self):
        """Non-technical call under 60s → below_expected_range."""
        result = generate_operational_alerts(
            [], _empty_policy(), 30.0, "non_technical"
        )
        assert result.call_duration_result.comparison == "below_expected_range"

    def test_non_technical_within_range(self):
        """Non-technical call 120s → within_expected_range."""
        result = generate_operational_alerts(
            [], _empty_policy(), 120.0, "non_technical"
        )
        assert result.call_duration_result.comparison == "within_expected_range"

    def test_non_technical_above_range(self):
        """Non-technical call over 180s → above_expected_range."""
        result = generate_operational_alerts(
            [], _empty_policy(), 250.0, "non_technical"
        )
        assert result.call_duration_result.comparison == "above_expected_range"

    def test_unknown_call_type(self):
        """Unknown call type → not_evaluated, no alert."""
        result = generate_operational_alerts(
            [], _empty_policy(), 100.0, "unknown"
        )
        assert result.call_duration_result.comparison == "not_evaluated"
        duration_alerts = [a for a in result.alerts if a.alert_type == "call_duration"]
        assert len(duration_alerts) == 0


# ══════════════════════════════════════════════════════════════════════
# HOLD POLICY ALERT TESTS
# ══════════════════════════════════════════════════════════════════════
class TestHoldPolicyAlerts:
    """Tests for Hold Policy → Alert conversion."""

    def test_duration_violation_converted(self):
        """Hold duration violation becomes an alert."""
        violation = HoldDurationViolation(
            candidate_id="hold_001", start_time=10.0, end_time=140.0,
            actual_duration_sec=130.0, allowed_duration_sec=120.0,
            excess_duration_sec=10.0,
        )
        policy = _policy(violations=[violation])
        result = generate_operational_alerts([], policy, 200.0, "unknown")
        hold_alerts = [a for a in result.alerts if a.source_category == "hold"]
        assert len(hold_alerts) == 1
        assert hold_alerts[0].alert_type == "hold_duration_violation"
        assert hold_alerts[0].severity == "high"
        assert hold_alerts[0].requires_manual_review is True

    def test_count_violation_converted(self):
        """Hold count violation becomes an alert."""
        policy = _policy(counted=3, limit=2, count_exceeded=True)
        result = generate_operational_alerts([], policy, 200.0, "unknown")
        hold_alerts = [a for a in result.alerts if a.source_category == "hold"]
        count_alerts = [a for a in hold_alerts if a.alert_type == "hold_count_violation"]
        assert len(count_alerts) == 1
        assert count_alerts[0].severity == "high"
        assert count_alerts[0].requires_manual_review is True

    def test_uncounted_over_duration_converted(self):
        """Uncounted over-duration candidate becomes a review alert."""
        uncounted = UncountedOverDuration(
            candidate_id="hold_003", start_time=200.0, end_time=340.0,
            duration_sec=140.0, classification="uncertain_hold_candidate",
            exceeds_duration_by_sec=20.0, reaches_call_end=True,
        )
        policy = _policy(uncounted=[uncounted])
        result = generate_operational_alerts([], policy, 340.0, "unknown")
        hold_alerts = [a for a in result.alerts if a.source_category == "hold"]
        uncounted_alerts = [a for a in hold_alerts
                           if a.alert_type == "hold_uncounted_over_duration"]
        assert len(uncounted_alerts) == 1
        assert uncounted_alerts[0].severity == "warning"
        assert uncounted_alerts[0].requires_manual_review is True

    def test_no_duplicate_hold_alerts(self):
        """Multiple hold issues produce distinct alerts, no duplicates."""
        violation = HoldDurationViolation(
            candidate_id="hold_001", start_time=10.0, end_time=140.0,
            actual_duration_sec=130.0, allowed_duration_sec=120.0,
            excess_duration_sec=10.0,
        )
        uncounted = UncountedOverDuration(
            candidate_id="hold_002", start_time=200.0, end_time=340.0,
            duration_sec=140.0, classification="uncertain_hold_candidate",
            exceeds_duration_by_sec=20.0, reaches_call_end=True,
        )
        policy = _policy(counted=2, limit=2, violations=[violation],
                         uncounted=[uncounted], count_exceeded=False)
        result = generate_operational_alerts([], policy, 340.0, "unknown")
        hold_alerts = [a for a in result.alerts if a.source_category == "hold"]
        assert len(hold_alerts) == 2
        ids = [a.alert_id for a in hold_alerts]
        assert len(ids) == len(set(ids))

    def test_empty_policy_no_hold_alerts(self):
        """Empty policy produces no hold alerts."""
        result = generate_operational_alerts([], _empty_policy(), 100.0, "unknown")
        hold_alerts = [a for a in result.alerts if a.source_category == "hold"]
        assert len(hold_alerts) == 0

    def test_uncertain_hold_not_converted_to_confirmed(self):
        """Uncertain Hold candidates are not converted into confirmed Holds."""
        uncounted = UncountedOverDuration(
            candidate_id="hold_003", start_time=200.0, end_time=340.0,
            duration_sec=140.0, classification="uncertain_hold_candidate",
            exceeds_duration_by_sec=20.0, reaches_call_end=True,
        )
        policy = _policy(uncounted=[uncounted])
        result = generate_operational_alerts([], policy, 340.0, "unknown")
        hold_alerts = [a for a in result.alerts if a.source_category == "hold"]
        assert hold_alerts[0].metadata["classification"] == "uncertain_hold_candidate"


# ══════════════════════════════════════════════════════════════════════
# MANUAL REVIEW AND SEVERITY TESTS
# ══════════════════════════════════════════════════════════════════════
class TestReviewAndSeverity:
    """Tests for manual_review_suggested and severity counts."""

    def test_info_only_no_manual_review(self):
        """Info-only alerts do NOT trigger manual review."""
        intervals = [
            _iv(0.0, 5.0, "neither_active"),
            _iv(5.0, 20.0, "both_active_interval"),  # 15s overlap → info
            _iv(20.0, 25.0, "neither_active"),
        ]
        result = generate_operational_alerts(
            intervals, _empty_policy(), 25.0, "technical"
        )
        # Only info alerts (overlap), within range
        assert result.manual_review_suggested is False

    def test_high_severity_triggers_review(self):
        """High-severity alert → manual_review_suggested = True."""
        intervals = [
            _iv(0.0, 10.0, "agent_active_only"),
            _iv(10.0, 70.0, "neither_active"),  # 60s → high
            _iv(70.0, 80.0, "agent_active_only"),
        ]
        result = generate_operational_alerts(
            intervals, _empty_policy(), 80.0, "unknown"
        )
        assert result.manual_review_suggested is True

    def test_requires_manual_review_flag_triggers(self):
        """Warning with requires_manual_review=True → manual review."""
        uncounted = UncountedOverDuration(
            candidate_id="hold_003", start_time=200.0, end_time=340.0,
            duration_sec=140.0, classification="uncertain_hold_candidate",
            exceeds_duration_by_sec=20.0, reaches_call_end=True,
        )
        policy = _policy(uncounted=[uncounted])
        result = generate_operational_alerts([], policy, 340.0, "unknown")
        # uncounted alert is warning + requires_manual_review=True
        assert result.manual_review_suggested is True

    def test_above_expected_range_triggers_review(self):
        """Call above expected range → manual review."""
        result = generate_operational_alerts(
            [], _empty_policy(), 700.0, "technical"
        )
        assert result.manual_review_suggested is True

    def test_within_range_no_review_by_itself(self):
        """Within-range duration alone does not trigger manual review."""
        result = generate_operational_alerts(
            [], _empty_policy(), 450.0, "technical"
        )
        assert result.manual_review_suggested is False

    def test_severity_counts(self):
        """Severity counts are computed correctly."""
        intervals = [
            _iv(0.0, 5.0, "neither_active"),
            _iv(5.0, 20.0, "both_active_interval"),  # 15s overlap → info
            _iv(20.0, 30.0, "neither_active"),
            _iv(30.0, 100.0, "neither_active"),  # 70s dead air → high
            _iv(100.0, 110.0, "agent_active_only"),
        ]
        result = generate_operational_alerts(
            intervals, _empty_policy(), 110.0, "unknown"
        )
        counts = result.alert_counts_by_severity
        assert counts.get("info", 0) >= 1  # overlap
        assert counts.get("high", 0) >= 1  # dead air

    def test_alerts_grouped_by_category(self):
        """Alerts are grouped by source_category."""
        intervals = [
            _iv(0.0, 5.0, "neither_active"),
            _iv(5.0, 20.0, "both_active_interval"),  # 15s overlap
            _iv(20.0, 30.0, "neither_active"),
            _iv(30.0, 100.0, "neither_active"),  # 70s dead air
            _iv(100.0, 110.0, "agent_active_only"),
        ]
        result = generate_operational_alerts(
            intervals, _empty_policy(), 110.0, "unknown"
        )
        assert "overlap" in result.alerts_by_category
        assert "silence" in result.alerts_by_category


# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION SNAPSHOT TEST
# ══════════════════════════════════════════════════════════════════════
class TestConfiguration:
    """Tests for configuration snapshot and custom thresholds."""

    def test_configuration_snapshot_populated(self):
        """Configuration snapshot contains all thresholds."""
        result = generate_operational_alerts(
            [], _empty_policy(), 60.0, "unknown",
            config=_config(dead_air_warning_sec=30.0)
        )
        snap = result.configuration_snapshot
        assert "dead_air_warning_sec" in snap
        assert snap["dead_air_warning_sec"] == 30.0
        assert "hold_suppression_overlap_ratio" in snap

    def test_custom_thresholds_applied(self):
        """Custom thresholds are used in detection."""
        intervals = [
            _iv(0.0, 5.0, "agent_active_only"),
            _iv(5.0, 20.0, "neither_active"),  # 15s
            _iv(20.0, 30.0, "agent_active_only"),
        ]
        # Default: 15s < 45s → no alert
        r1 = generate_operational_alerts(intervals, _empty_policy(), 30.0, "unknown")
        dead1 = [a for a in r1.alerts if a.alert_type == "dead_air"]
        assert len(dead1) == 0

        # Custom: 15s >= 10s → alert
        r2 = generate_operational_alerts(
            intervals, _empty_policy(), 30.0, "unknown",
            config=_config(dead_air_warning_sec=10.0, dead_air_high_sec=15.0)
        )
        dead2 = [a for a in r2.alerts if a.alert_type == "dead_air"]
        assert len(dead2) == 1


# ══════════════════════════════════════════════════════════════════════
# INTEGRATION TEST
# ══════════════════════════════════════════════════════════════════════
class TestIntegration:
    """Full synthetic-call integration test — Phase VI-A corrected."""

    def test_340s_technical_call(self):
        """340.5s technical call — corrected Phase VI-A alert set.

        Timeline:
        - 0–20s: both_active_interval (20s overlap → info alert)
        - 20–50s: neither_active (30s dead air → below 45s → no alert)
        - 50–70s: both_active_interval (20s overlap → info alert)
        - 70–195s: neither_active (125s dead air → BUT suppressed by
          Hold duration violation at same interval)
        - 195–210s: agent_active_only (15s → NO silence alert)
        - 210–340.5s: neither_active (130.5s dead air → high alert)
        """
        intervals = [
            _iv(0.0, 20.0, "both_active_interval"),
            _iv(20.0, 50.0, "neither_active"),
            _iv(50.0, 70.0, "both_active_interval"),
            _iv(70.0, 195.0, "neither_active"),
            _iv(195.0, 210.0, "agent_active_only"),
            _iv(210.0, 340.5, "neither_active"),
        ]

        violation = HoldDurationViolation(
            candidate_id="hold_002", start_time=70.0, end_time=195.0,
            actual_duration_sec=125.0, allowed_duration_sec=120.0,
            excess_duration_sec=5.0,
        )
        policy = _policy(counted=1, limit=2, violations=[violation])

        result = generate_operational_alerts(
            intervals, policy, 340.5, "technical"
        )

        by_type = {}
        for a in result.alerts:
            by_type.setdefault(a.alert_type, []).append(a)

        # Overlap: 0-20s and 50-70s → 2 info alerts
        overlap = by_type.get("extended_overlap", [])
        assert len(overlap) == 2

        # Dead Air: 210-340.5s (130.5s) → high (only this one survives)
        # 20-50s (30s) → below threshold, no alert
        # 70-195s (125s) → suppressed by Hold violation (100% overlap)
        dead = by_type.get("dead_air", [])
        assert len(dead) == 1
        assert abs(dead[0].start_time - 210.0) < 0.1
        assert dead[0].severity == "high"

        # Hold: 1 duration violation → 1 high alert
        hold = [a for a in result.alerts if a.source_category == "hold"]
        assert len(hold) == 1
        assert hold[0].alert_type == "hold_duration_violation"

        # NO silence alerts
        silence = [a for a in result.alerts if "silence" in a.alert_type]
        assert len(silence) == 0

        # Call duration: 340.5s technical → within 300-600 → no alert
        assert result.call_duration_result.comparison == "within_expected_range"

        # Manual review: high-severity dead air + hold violation
        assert result.manual_review_suggested is True

        # All alert IDs unique
        ids = [a.alert_id for a in result.alerts]
        assert len(ids) == len(set(ids))
