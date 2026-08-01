"""
Tests for WE-inspired Hold policy evaluation.
"""

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.demo.schemas import HoldCandidate, HoldPolicyConfig
from src.demo.hold_policy import evaluate_hold_policy


# ══════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════
def _candidate(cid, start, end, classification="likely_hold_candidate",
               manual_status="unreviewed", reaches_call_end=False):
    """Create a HoldCandidate for testing."""
    return HoldCandidate(
        candidate_id=cid,
        start_time=start,
        end_time=end,
        duration_sec=round(end - start, 6),
        classification=classification,
        evidence_types=["prolonged_silence"],
        agent_inactive=True,
        customer_state="inactive",
        confidence=0.5,
        detection_notes="test",
        agent_return_detected=True,
        reaches_call_end=reaches_call_end,
        source_interval_ids=[],
        manual_status=manual_status,
    )


def _config(**kwargs):
    """Create HoldPolicyConfig with overrides."""
    defaults = {
        "max_hold_count": 2,
        "max_hold_duration_sec": 120.0,
        "counting_mode": "count_likely_and_confirmed",
        "policy_source": "test policy",
        "policy_version": "test_v1",
    }
    defaults.update(kwargs)
    return HoldPolicyConfig(**defaults)


# ══════════════════════════════════════════════════════════════════════
# HOLD POLICY TESTS
# ══════════════════════════════════════════════════════════════════════
class TestHoldPolicy:
    """Tests for evaluate_hold_policy."""

    def test_zero_holds_compliant(self):
        """Zero confirmed/countable Holds → compliant."""
        result = evaluate_hold_policy([], _config())
        assert result.compliant is True
        assert result.count_exceeded is False
        assert result.review_required is False

    def test_one_hold_under_120_compliant(self):
        """One Hold under 120 seconds → compliant."""
        events = [_candidate("h1", 0.0, 60.0)]
        result = evaluate_hold_policy(events, _config())
        assert result.compliant is True
        assert result.counted_hold_events == 1

    def test_two_holds_under_120_compliant(self):
        """Two Holds under 120 seconds → compliant."""
        events = [
            _candidate("h1", 0.0, 60.0),
            _candidate("h2", 100.0, 160.0),
        ]
        result = evaluate_hold_policy(events, _config())
        assert result.compliant is True
        assert result.counted_hold_events == 2

    def test_three_holds_count_exceeded(self):
        """Three Holds → count limit exceeded."""
        events = [
            _candidate("h1", 0.0, 60.0),
            _candidate("h2", 100.0, 160.0),
            _candidate("h3", 200.0, 260.0),
        ]
        result = evaluate_hold_policy(events, _config())
        assert result.compliant is False
        assert result.count_exceeded is True
        assert result.counted_hold_events == 3

    def test_one_hold_exactly_120_compliant(self):
        """One Hold exactly 120.0 seconds → compliant."""
        events = [_candidate("h1", 0.0, 120.0)]
        result = evaluate_hold_policy(events, _config())
        assert result.compliant is True
        assert len(result.duration_violations) == 0

    def test_one_hold_120_5_exceeds_duration(self):
        """One Hold 120.5 seconds → duration limit exceeded."""
        events = [_candidate("h1", 0.0, 120.5)]
        result = evaluate_hold_policy(events, _config())
        assert result.compliant is False
        assert len(result.duration_violations) == 1
        assert result.duration_violations[0].excess_duration_sec == 0.5

    def test_two_holds_one_exceeds_duration(self):
        """Two Holds where one exceeds duration."""
        events = [
            _candidate("h1", 0.0, 60.0),
            _candidate("h2", 100.0, 230.0),  # 130s > 120s
        ]
        result = evaluate_hold_policy(events, _config())
        assert result.compliant is False
        assert len(result.duration_violations) == 1
        assert result.duration_violations[0].candidate_id == "h2"

    def test_three_holds_one_exceeds_duration(self):
        """Three Holds where one exceeds duration."""
        events = [
            _candidate("h1", 0.0, 60.0),
            _candidate("h2", 100.0, 160.0),
            _candidate("h3", 200.0, 330.0),  # 130s
        ]
        result = evaluate_hold_policy(events, _config())
        assert result.compliant is False
        assert result.count_exceeded is True
        assert len(result.duration_violations) == 1

    def test_uncertain_excluded_when_counting_mode(self):
        """Uncertain candidates excluded when counting mode requires confirmation."""
        events = [
            _candidate("h1", 0.0, 60.0, classification="uncertain_hold_candidate"),
            _candidate("h2", 100.0, 160.0, classification="likely_hold_candidate"),
        ]
        result = evaluate_hold_policy(events, _config(counting_mode="count_likely_and_confirmed"))
        assert result.counted_hold_events == 1  # Only likely counted

    def test_likely_included_when_configured(self):
        """Likely candidates included when configured."""
        events = [
            _candidate("h1", 0.0, 60.0, classification="likely_hold_candidate"),
        ]
        result = evaluate_hold_policy(events, _config(counting_mode="count_likely_and_confirmed"))
        assert result.counted_hold_events == 1

    def test_manual_confirmed_counted(self):
        """Manual confirmed candidate counted."""
        events = [
            _candidate("h1", 0.0, 60.0, classification="likely_hold_candidate",
                       manual_status="confirmed_hold"),
        ]
        result = evaluate_hold_policy(events, _config(counting_mode="count_likely_and_confirmed"))
        assert result.counted_hold_events == 1

    def test_manual_rejected_excluded(self):
        """Manual rejected candidate excluded."""
        events = [
            _candidate("h1", 0.0, 60.0, classification="likely_hold_candidate",
                       manual_status="rejected_hold"),
        ]
        result = evaluate_hold_policy(events, _config(counting_mode="count_likely_and_confirmed"))
        assert result.counted_hold_events == 0

    def test_exact_excess_duration(self):
        """Excess duration is calculated exactly."""
        events = [_candidate("h1", 0.0, 125.3)]
        result = evaluate_hold_policy(events, _config())
        assert len(result.duration_violations) == 1
        v = result.duration_violations[0]
        assert v.actual_duration_sec == 125.3
        assert v.allowed_duration_sec == 120.0
        assert v.excess_duration_sec == 5.3

    def test_policy_source_included(self):
        """Policy source is included in result."""
        result = evaluate_hold_policy([], _config(policy_source="WE-inspired test"))
        assert result.policy_source == "WE-inspired test"

    def test_policy_values_overridable(self):
        """Policy values can be overridden."""
        events = [_candidate("h1", 0.0, 50.0), _candidate("h2", 60.0, 110.0)]
        # Override: max count = 1
        result = evaluate_hold_policy(events, _config(max_hold_count=1))
        assert result.count_exceeded is True
        assert result.count_limit == 1

    def test_no_general_emotion_alerts(self):
        """No general emotion alerts generated — only Hold policy."""
        result = evaluate_hold_policy([], _config())
        # Result should only contain Hold-related fields
        assert hasattr(result, "counted_hold_events")
        assert hasattr(result, "compliant")
        assert not hasattr(result, "emotion_alerts")
        assert not hasattr(result, "satisfaction_score")

    def test_no_employee_disciplinary_conclusion(self):
        """No employee disciplinary conclusion generated."""
        events = [_candidate("h1", 0.0, 200.0)]
        result = evaluate_hold_policy(events, _config())
        # Result recommends review, not punishment
        assert result.review_required is True
        assert "review" in result.review_reason.lower()
        assert "fault" not in result.review_reason.lower()
        assert "punishment" not in result.review_reason.lower()

    def test_counting_mode_count_all(self):
        """count_all_candidates counts everything."""
        events = [
            _candidate("h1", 0.0, 60.0, classification="uncertain_hold_candidate"),
            _candidate("h2", 100.0, 160.0, classification="likely_hold_candidate"),
        ]
        result = evaluate_hold_policy(events, _config(counting_mode="count_all_candidates"))
        assert result.counted_hold_events == 2

    def test_counting_mode_confirmed_only(self):
        """count_confirmed_only counts only confirmed."""
        events = [
            _candidate("h1", 0.0, 60.0, classification="likely_hold_candidate"),
            _candidate("h2", 100.0, 160.0, classification="confirmed_by_metadata_or_manual_input"),
        ]
        result = evaluate_hold_policy(events, _config(counting_mode="count_confirmed_only"))
        assert result.counted_hold_events == 1


# ══════════════════════════════════════════════════════════════════════
# UNCOUNTED OVER-DURATION TESTS
# ══════════════════════════════════════════════════════════════════════
class TestUncountedOverDuration:
    """Tests for uncounted over-duration candidate reporting."""

    def test_uncounted_candidate_surfaced(self):
        """Uncertain candidate exceeding duration is surfaced for review."""
        events = [
            _candidate("h1", 0.0, 60.0, classification="likely_hold_candidate"),
            _candidate("h2", 100.0, 240.0,  # 140s, uncertain
                       classification="uncertain_hold_candidate",
                       reaches_call_end=True),
        ]
        result = evaluate_hold_policy(events, _config())
        assert len(result.uncounted_over_duration) == 1
        u = result.uncounted_over_duration[0]
        assert u.candidate_id == "h2"
        assert u.duration_sec == 140.0
        assert u.exceeds_duration_by_sec == 20.0
        assert u.classification == "uncertain_hold_candidate"
        assert u.reaches_call_end is True

    def test_review_required_for_uncounted(self):
        """Review is required when uncounted over-duration exists."""
        events = [
            _candidate("h1", 0.0, 60.0, classification="likely_hold_candidate"),
            _candidate("h2", 100.0, 240.0,
                       classification="uncertain_hold_candidate"),
        ]
        result = evaluate_hold_policy(events, _config())
        assert result.review_required is True
        assert "uncounted" in result.review_reason.lower()

    def test_counted_violation_also_surfaced(self):
        """Counted duration violations still appear in duration_violations."""
        events = [
            _candidate("h1", 0.0, 130.0, classification="likely_hold_candidate"),
        ]
        result = evaluate_hold_policy(events, _config())
        assert len(result.duration_violations) == 1
        assert result.duration_violations[0].candidate_id == "h1"
        assert len(result.uncounted_over_duration) == 0

    def test_both_violations_and_uncounted(self):
        """Both counted violations and uncounted over-duration are surfaced."""
        events = [
            _candidate("h1", 0.0, 130.0, classification="likely_hold_candidate"),
            _candidate("h2", 200.0, 350.0,  # 150s, uncertain
                       classification="uncertain_hold_candidate"),
        ]
        result = evaluate_hold_policy(events, _config())
        assert len(result.duration_violations) == 1
        assert len(result.uncounted_over_duration) == 1
        assert result.review_required is True

    def test_compliant_no_uncounted(self):
        """Compliant result has no uncounted over-duration."""
        events = [
            _candidate("h1", 0.0, 60.0, classification="likely_hold_candidate"),
        ]
        result = evaluate_hold_policy(events, _config())
        assert result.compliant is True
        assert len(result.uncounted_over_duration) == 0

    def test_uncounted_not_in_violations(self):
        """Uncounted candidates do NOT appear in duration_violations."""
        events = [
            _candidate("h1", 0.0, 60.0, classification="likely_hold_candidate"),
            _candidate("h2", 100.0, 250.0,  # 150s, uncertain
                       classification="uncertain_hold_candidate"),
        ]
        result = evaluate_hold_policy(events, _config())
        violation_ids = [v.candidate_id for v in result.duration_violations]
        assert "h2" not in violation_ids

    def test_reaches_call_end_flag_preserved(self):
        """reaches_call_end flag is preserved in uncounted_over_duration."""
        events = [
            _candidate("h1", 0.0, 60.0, classification="likely_hold_candidate"),
            _candidate("h2", 200.0, 340.5,
                       classification="uncertain_hold_candidate",
                       reaches_call_end=True),
        ]
        result = evaluate_hold_policy(events, _config())
        u = result.uncounted_over_duration[0]
        assert u.reaches_call_end is True

    def test_review_reason_includes_uncounted_details(self):
        """Review reason includes details about uncounted candidates."""
        events = [
            _candidate("h1", 0.0, 60.0, classification="likely_hold_candidate"),
            _candidate("h2", 200.0, 350.0,
                       classification="uncertain_hold_candidate",
                       reaches_call_end=True),
        ]
        result = evaluate_hold_policy(events, _config())
        assert "uncounted" in result.review_reason.lower()
        assert "h2" in result.review_reason
