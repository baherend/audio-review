"""
Audio-based operational review alerts — Phase VI-A corrected.

Generates explainable alerts from audio-derived metadata and approved
Hold Policy output. Uses only ActivityInterval timeline data —
no text analysis, no emotion model changes.

Alert categories:
- Dead Air (both channels inactive)
- Extended Overlap (both channels active)
- Call Duration Review
- Hold Policy Alerts (from HoldPolicyResult)

Phase VI-A semantic correction:
- Single-speaker activity (customer_active_only, agent_active_only) does
  NOT generate silence alerts. One speaker being active while the other
  is silent does not prove problematic silence. The active speaker may
  be talking, the other listening. These intervals are not labeled as
  silence violations.
- Dead Air severity escalates at two thresholds: warning (≥45s) and
  high (≥60s).
- Dead Air events that materially overlap Hold-related events are
  suppressed because the Hold alert already explains the interval.
- manual_review_suggested is computed from structured requires_manual_review
  fields, not from undocumented severity heuristics.

Precedence rule:
- Dead Air handles both-inactive intervals.
- Extended Overlap handles both-active intervals.
- Hold alerts take precedence: Dead Air overlapping ≥90% with a Hold
  event is suppressed.
- Single-speaker intervals (customer_active_only, agent_active_only)
  generate NO silence alerts. Single-speaker duration analysis is
  deferred to a future conversation-flow phase.

Schema limitation:
- ActivityInterval uses binary active/inactive states. It cannot
  distinguish real speech from music, tones, or environmental sound.
- Do not interpret customer_active=True as confirmed Customer speech.
- Do not interpret agent_active=True as confirmed Agent speech.
- Use neutral wording where appropriate.
"""

import logging
from typing import Dict, List, Optional, Tuple

from .schemas import (
    ActivityInterval,
    HoldPolicyResult,
    OperationalAlert,
    OperationalAlertConfig,
    OperationalAlertResult,
    CallDurationResult,
)

logger = logging.getLogger(__name__)


def generate_operational_alerts(
    intervals: List[ActivityInterval],
    hold_policy_result: HoldPolicyResult,
    call_duration_sec: float,
    call_type: str = "unknown",
    config: Optional[OperationalAlertConfig] = None,
) -> OperationalAlertResult:
    """
    Generate operational review alerts from audio-derived metadata.

    Args:
        intervals: Non-overlapping activity intervals from the pipeline.
        hold_policy_result: Approved Hold Policy evaluation result.
        call_duration_sec: Total call duration in seconds.
        call_type: "technical" | "non_technical" | "unknown".
        config: Alert configuration with thresholds.

    Returns:
        OperationalAlertResult with all alerts and metadata.
    """
    if config is None:
        config = OperationalAlertConfig()

    alerts: List[OperationalAlert] = []
    alert_counter = 0

    # ── 1. Hold Policy Alerts (generated first for precedence) ────────
    hold_alerts, alert_counter = _convert_hold_policy_to_alerts(
        hold_policy_result, alert_counter
    )
    alerts.extend(hold_alerts)

    # Collect Hold-related time ranges for overlap suppression
    # Includes both alert-based ranges (violations, uncounted) and
    # counted hold ranges from the policy result (compliant Holds)
    hold_ranges = _collect_hold_ranges(hold_alerts, hold_policy_result)

    # ── 2. Dead Air (both inactive) ────────────────────────────────────
    dead_air_alerts, alert_counter = _detect_dead_air(
        intervals, call_duration_sec, alert_counter, config
    )
    alerts.extend(dead_air_alerts)

    # ── 3. Hold precedence: suppress Dead Air overlapping Hold events ──
    alerts = _suppress_dead_air_by_hold(alerts, hold_ranges, config)

    # ── 4. Extended Overlap (both active) ──────────────────────────────
    overlap_alerts, alert_counter = _detect_extended_overlap(
        intervals, call_duration_sec, alert_counter, config
    )
    alerts.extend(overlap_alerts)

    # ── 5. Call Duration Review ────────────────────────────────────────
    call_duration_result = _evaluate_call_duration(
        call_duration_sec, call_type, config
    )
    if call_duration_result.comparison in ("below_expected_range", "above_expected_range"):
        alert_counter += 1
        alerts.append(_build_call_duration_alert(
            alert_counter, call_duration_result
        ))

    # ── Build result ───────────────────────────────────────────────────
    return _build_result(alerts, call_duration_result, config)


# ══════════════════════════════════════════════════════════════════════
# HOLD RANGE COLLECTION
# ══════════════════════════════════════════════════════════════════════

def _collect_hold_ranges(
    hold_alerts: List[OperationalAlert],
    policy: HoldPolicyResult,
) -> List[Tuple[float, float]]:
    """Extract time ranges from Hold-related sources for overlap suppression.

    Combines ranges from:
    1. Hold-related alerts (violations, uncounted over-duration)
    2. Counted Hold ranges from HoldPolicyResult (compliant Holds within
       the duration limit)

    This ensures that compliant Hold intervals (e.g., a 80s Hold within
    the 120s limit) also suppress overlapping Dead Air alerts.
    """
    ranges = []
    seen = set()

    # From alerts (violations, uncounted)
    for a in hold_alerts:
        if a.start_time is not None and a.end_time is not None:
            if a.end_time > a.start_time:
                key = (round(a.start_time, 6), round(a.end_time, 6))
                if key not in seen:
                    ranges.append((a.start_time, a.end_time))
                    seen.add(key)

    # From counted hold ranges (compliant Holds)
    for start, end in policy.counted_hold_ranges:
        if end > start:
            key = (round(start, 6), round(end, 6))
            if key not in seen:
                ranges.append((start, end))
                seen.add(key)

    return ranges


def _compute_overlap_ratio(
    range_a: Tuple[float, float],
    range_b: Tuple[float, float],
) -> float:
    """Compute overlap ratio of range_a relative to range_b's duration.

    Returns the fraction of range_b that overlaps with range_a.
    """
    overlap_start = max(range_a[0], range_b[0])
    overlap_end = min(range_a[1], range_b[1])
    overlap_duration = max(0.0, overlap_end - overlap_start)
    range_b_duration = range_b[1] - range_b[0]
    if range_b_duration <= 0:
        return 0.0
    return overlap_duration / range_b_duration


# ══════════════════════════════════════════════════════════════════════
# DEAD AIR
# ══════════════════════════════════════════════════════════════════════

def _detect_dead_air(
    intervals: List[ActivityInterval],
    call_duration_sec: float,
    counter: int,
    config: OperationalAlertConfig,
) -> tuple:
    """Detect Dead Air: both Agent and Customer inactive.

    Severity:
    - duration >= dead_air_warning_sec AND < dead_air_high_sec → warning
    - duration >= dead_air_high_sec → high
    - duration < dead_air_warning_sec → no alert

    Includes reassurance review recommendation for the reviewer to
    verify whether the customer received a verbal reassurance or
    progress update.
    """
    alerts = []
    merged_start = None
    merged_end = None

    for iv in intervals:
        if not iv.agent_active and not iv.customer_active:
            if merged_start is None:
                merged_start = iv.start_time
            merged_end = iv.end_time
        else:
            if merged_start is not None:
                _flush_dead_air(
                    merged_start, merged_end, call_duration_sec,
                    counter, alerts, config
                )
                counter += len(alerts)
                merged_start = None
                merged_end = None

    if merged_start is not None:
        _flush_dead_air(
            merged_start, merged_end, call_duration_sec,
            counter, alerts, config
        )
        counter += len(alerts)

    return alerts, counter


def _flush_dead_air(
    start: float, end: float, call_duration: float,
    base_counter: int, alerts: list, config: OperationalAlertConfig,
) -> None:
    """Create a Dead Air alert if threshold is met."""
    duration = round(end - start, 6)
    if duration <= 0:
        return
    if duration < config.dead_air_warning_sec:
        return

    reaches_end = abs(end - call_duration) < 0.01

    if duration >= config.dead_air_high_sec:
        severity = "high"
        title = "Dead Air Detected — Review Required"
        explanation = (
            f"Both Agent and Customer channels were inactive for "
            f"{duration:.1f}s ({start:.1f}s–{end:.1f}s), exceeding "
            f"the review threshold of {config.dead_air_high_sec:.0f}s. "
            f"This is separate from Hold Detection and may indicate "
            f"a connection issue, unrecorded activity, or system gap."
        )
    else:
        severity = "warning"
        title = "Dead Air Detected"
        explanation = (
            f"Both Agent and Customer channels were inactive for "
            f"{duration:.1f}s ({start:.1f}s–{end:.1f}s), exceeding "
            f"the warning threshold of {config.dead_air_warning_sec:.0f}s. "
            f"This is separate from Hold Detection and may indicate "
            f"a connection issue, unrecorded activity, or system gap."
        )

    alert_id = f"alert_{base_counter + len(alerts) + 1:03d}"
    alerts.append(OperationalAlert(
        alert_id=alert_id,
        alert_type="dead_air",
        severity=severity,
        start_time=start,
        end_time=end,
        duration_sec=duration,
        title=title,
        explanation=explanation,
        review_recommendation=(
            "Review whether the customer received an appropriate verbal "
            "reassurance or progress update during this interval. "
            "If Hold music was playing, the Hold Detection system should "
            "have flagged this separately."
        ),
        source_category="silence",
        metadata={
            "warning_threshold_sec": config.dead_air_warning_sec,
            "high_threshold_sec": config.dead_air_high_sec,
            "reaches_call_end": reaches_end,
            "note": (
                "Schema limitation: ActivityInterval uses binary "
                "active/inactive. Cannot distinguish non-speech audio "
                "(music, tones) from silence."
            ),
        },
    ))


# ══════════════════════════════════════════════════════════════════════
# HOLD PRECEDENCE AND SUPPRESSION
# ══════════════════════════════════════════════════════════════════════

def _suppress_dead_air_by_hold(
    alerts: List[OperationalAlert],
    hold_ranges: List[Tuple[float, float]],
    config: OperationalAlertConfig,
) -> List[OperationalAlert]:
    """Suppress Dead Air alerts that materially overlap Hold events.

    When a Dead Air event overlaps ≥ hold_suppression_overlap_ratio
    with a Hold-related event, the Dead Air is suppressed because the
    Hold alert already explains the interval.

    For partial overlaps below the suppression threshold, the Dead Air
    is retained with overlap metadata.
    """
    if not hold_ranges:
        return alerts

    result = []
    for a in alerts:
        if a.alert_type != "dead_air":
            result.append(a)
            continue

        dead_air_range = (a.start_time, a.end_time)
        max_overlap = 0.0

        for hr in hold_ranges:
            overlap = _compute_overlap_ratio(hr, dead_air_range)
            max_overlap = max(max_overlap, overlap)

        if max_overlap >= config.hold_suppression_overlap_ratio:
            # Suppress: Hold event explains this interval
            logger.debug(
                "Dead Air %s suppressed: %.1f%% overlap with Hold event",
                a.alert_id, max_overlap * 100
            )
            continue

        # Retain with overlap metadata
        if max_overlap > 0:
            a.metadata["hold_overlap_ratio"] = round(max_overlap, 4)
            a.metadata["hold_overlap_note"] = (
                f"Partial overlap ({max_overlap*100:.1f}%) with a Hold event. "
                f"Below suppression threshold of "
                f"{config.hold_suppression_overlap_ratio*100:.0f}%."
            )
        result.append(a)

    return result


# ══════════════════════════════════════════════════════════════════════
# EXTENDED OVERLAP
# ══════════════════════════════════════════════════════════════════════

def _detect_extended_overlap(
    intervals: List[ActivityInterval],
    call_duration_sec: float,
    counter: int,
    config: OperationalAlertConfig,
) -> tuple:
    """Detect Extended Overlap: both Agent and Customer active."""
    alerts = []
    merged_start = None
    merged_end = None

    for iv in intervals:
        if iv.agent_active and iv.customer_active:
            if merged_start is None:
                merged_start = iv.start_time
            merged_end = iv.end_time
        else:
            if merged_start is not None:
                _flush_extended_overlap(
                    merged_start, merged_end, call_duration_sec,
                    counter, alerts, config
                )
                counter += len(alerts)
                merged_start = None
                merged_end = None

    if merged_start is not None:
        _flush_extended_overlap(
            merged_start, merged_end, call_duration_sec,
            counter, alerts, config
        )
        counter += len(alerts)

    return alerts, counter


def _flush_extended_overlap(
    start: float, end: float, call_duration: float,
    base_counter: int, alerts: list, config: OperationalAlertConfig,
) -> None:
    """Create an Extended Overlap alert if threshold is met."""
    duration = round(end - start, 6)
    if duration <= 0:
        return
    if duration < config.extended_overlap_review_sec:
        return

    reaches_end = abs(end - call_duration) < 0.01
    alert_id = f"alert_{base_counter + len(alerts) + 1:03d}"
    alerts.append(OperationalAlert(
        alert_id=alert_id,
        alert_type="extended_overlap",
        severity="info",
        start_time=start,
        end_time=end,
        duration_sec=duration,
        title="Extended Simultaneous Speech",
        explanation=(
            f"Both Agent and Customer channels were active simultaneously "
            f"for {duration:.1f}s ({start:.1f}s–{end:.1f}s). "
            f"Extended simultaneous speech detected; conversation flow "
            f"review suggested."
        ),
        review_recommendation=(
            "Review conversation flow. This alert does not assign blame "
            "to either speaker."
        ),
        source_category="overlap",
        metadata={
            "threshold_sec": config.extended_overlap_review_sec,
            "reaches_call_end": reaches_end,
            "note": (
                "Schema limitation: cannot distinguish real speech from "
                "music, tones, or environmental sound."
            ),
        },
    ))


# ══════════════════════════════════════════════════════════════════════
# CALL DURATION
# ══════════════════════════════════════════════════════════════════════

def _evaluate_call_duration(
    call_duration_sec: float,
    call_type: str,
    config: OperationalAlertConfig,
) -> CallDurationResult:
    """Evaluate call duration against expected range for call type."""
    if call_type == "technical":
        expected_min = config.technical_call_min_sec
        expected_max = config.technical_call_max_sec
    elif call_type == "non_technical":
        expected_min = config.non_technical_call_min_sec
        expected_max = config.non_technical_call_max_sec
    else:
        return CallDurationResult(
            call_type=call_type,
            call_duration_sec=call_duration_sec,
            comparison="not_evaluated",
        )

    if call_duration_sec < expected_min:
        comparison = "below_expected_range"
    elif call_duration_sec > expected_max:
        comparison = "above_expected_range"
    else:
        comparison = "within_expected_range"

    return CallDurationResult(
        call_type=call_type,
        call_duration_sec=call_duration_sec,
        expected_min_sec=expected_min,
        expected_max_sec=expected_max,
        comparison=comparison,
    )


def _build_call_duration_alert(
    alert_counter: int,
    result: CallDurationResult,
) -> OperationalAlert:
    """Build a call-duration review alert."""
    if result.comparison == "below_expected_range":
        severity = "info"
        title = "Call Duration Below Expected Range"
        explanation = (
            f"Call duration {result.call_duration_sec:.1f}s is below the "
            f"expected range ({result.expected_min_sec:.0f}s–"
            f"{result.expected_max_sec:.0f}s) for {result.call_type} calls."
        )
    elif result.comparison == "above_expected_range":
        severity = "warning"
        title = "Call Duration Above Expected Range"
        explanation = (
            f"Call duration {result.call_duration_sec:.1f}s exceeds the "
            f"expected range ({result.expected_min_sec:.0f}s–"
            f"{result.expected_max_sec:.0f}s) for {result.call_type} calls."
        )
    else:
        severity = "info"
        title = "Call Duration Not Evaluated"
        explanation = (
            f"Call duration {result.call_duration_sec:.1f}s. "
            f"No duration-range judgment for {result.call_type} calls."
        )

    return OperationalAlert(
        alert_id=f"alert_{alert_counter:03d}",
        alert_type="call_duration",
        severity=severity,
        start_time=0.0,
        end_time=result.call_duration_sec,
        duration_sec=result.call_duration_sec,
        title=title,
        explanation=explanation,
        review_recommendation=(
            "These are configurable operational guidelines based on "
            "project-owner call-center experience. Not official company "
            "policies. Not Pass/Fail rules."
        ),
        source_category="duration",
        metadata={
            "call_type": result.call_type,
            "comparison": result.comparison,
            "expected_min_sec": result.expected_min_sec,
            "expected_max_sec": result.expected_max_sec,
        },
    )


# ══════════════════════════════════════════════════════════════════════
# HOLD POLICY ALERT CONVERSION
# ══════════════════════════════════════════════════════════════════════

def _convert_hold_policy_to_alerts(
    policy: HoldPolicyResult,
    counter: int,
) -> tuple:
    """Convert HoldPolicyResult into operational alerts.

    Generates alerts for:
    - Counted Hold duration violations
    - Hold count limit violations
    - Uncounted over-duration review items

    Does NOT:
    - Count uncertain Hold candidates when policy excludes them
    - Convert uncertain candidates into confirmed Holds
    """
    alerts = []

    # ── Duration violations (counted events) ───────────────────────────
    for v in policy.duration_violations:
        counter += 1
        alerts.append(OperationalAlert(
            alert_id=f"alert_{counter:03d}",
            alert_type="hold_duration_violation",
            severity="high",
            requires_manual_review=True,
            start_time=v.start_time,
            end_time=v.end_time,
            duration_sec=v.actual_duration_sec,
            title=f"Hold Duration Exceeded — {v.candidate_id}",
            explanation=(
                f"Hold event {v.candidate_id} lasted {v.actual_duration_sec:.1f}s, "
                f"exceeding the limit of {v.allowed_duration_sec:.1f}s "
                f"(excess: {v.excess_duration_sec:.1f}s)."
            ),
            review_recommendation=(
                "Review the Hold event context. This is a policy-threshold "
                "alert, not an automatic conclusion of employee fault."
            ),
            source_category="hold",
            metadata={
                "candidate_id": v.candidate_id,
                "actual_duration_sec": v.actual_duration_sec,
                "allowed_duration_sec": v.allowed_duration_sec,
                "excess_duration_sec": v.excess_duration_sec,
            },
        ))

    # ── Count limit violation ───────────────────────────────────────────
    if policy.count_exceeded:
        counter += 1
        alerts.append(OperationalAlert(
            alert_id=f"alert_{counter:03d}",
            alert_type="hold_count_violation",
            severity="high",
            requires_manual_review=True,
            start_time=0.0,
            end_time=None,
            duration_sec=None,
            title="Hold Count Limit Exceeded",
            explanation=(
                f"{policy.counted_hold_events} Hold events were counted, "
                f"exceeding the limit of {policy.count_limit}."
            ),
            review_recommendation=(
                "Review the Hold events. This is a policy-threshold "
                "alert, not an automatic conclusion of employee fault."
            ),
            source_category="hold",
            metadata={
                "counted_hold_events": policy.counted_hold_events,
                "count_limit": policy.count_limit,
            },
        ))

    # ── Uncounted over-duration review items ────────────────────────────
    for u in policy.uncounted_over_duration:
        counter += 1
        alerts.append(OperationalAlert(
            alert_id=f"alert_{counter:03d}",
            alert_type="hold_uncounted_over_duration",
            severity="warning",
            requires_manual_review=True,
            start_time=u.start_time,
            end_time=u.end_time,
            duration_sec=u.duration_sec,
            title=f"Uncounted Hold Candidate Exceeds Duration — {u.candidate_id}",
            explanation=(
                f"Hold candidate {u.candidate_id} ({u.classification}) "
                f"lasted {u.duration_sec:.1f}s, exceeding the duration limit "
                f"(excess: {u.exceeds_duration_by_sec:.1f}s). "
                f"This candidate was excluded from counted policy events "
                f"by the configured counting mode."
            ),
            review_recommendation=(
                "Review this candidate. It was not counted toward policy "
                "limits but exceeds the duration threshold. "
                f"Reaches call end: {u.reaches_call_end}."
            ),
            source_category="hold",
            metadata={
                "candidate_id": u.candidate_id,
                "classification": u.classification,
                "duration_sec": u.duration_sec,
                "exceeds_duration_by_sec": u.exceeds_duration_by_sec,
                "reaches_call_end": u.reaches_call_end,
            },
        ))

    return alerts, counter


# ══════════════════════════════════════════════════════════════════════
# RESULT CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════

def _build_result(
    alerts: List[OperationalAlert],
    call_duration_result: CallDurationResult,
    config: OperationalAlertConfig,
) -> OperationalAlertResult:
    """Build the final OperationalAlertResult.

    manual_review_suggested is True when ANY of:
    - any high-severity alert exists
    - any alert has requires_manual_review=True
    - a Hold count violation exists
    - a Hold duration violation exists
    - an uncounted over-duration Hold review item exists
    - call duration is above expected range

    Information-only alerts (severity=info, requires_manual_review=False)
    must NOT automatically trigger manual review.
    """
    # Group by category
    by_category: Dict[str, List[OperationalAlert]] = {}
    for a in alerts:
        by_category.setdefault(a.source_category, []).append(a)

    # Count by severity
    severity_counts: Dict[str, int] = {}
    for a in alerts:
        severity_counts[a.severity] = severity_counts.get(a.severity, 0) + 1

    # Manual review: explicit rule from structured fields
    manual_review = (
        any(a.severity == "high" for a in alerts)
        or any(a.requires_manual_review for a in alerts)
        or call_duration_result.comparison == "above_expected_range"
    )

    return OperationalAlertResult(
        alerts=alerts,
        alerts_by_category=by_category,
        alert_counts_by_severity=severity_counts,
        manual_review_suggested=manual_review,
        call_duration_result=call_duration_result,
        configuration_snapshot={
            "dead_air_warning_sec": config.dead_air_warning_sec,
            "dead_air_high_sec": config.dead_air_high_sec,
            "hold_suppression_overlap_ratio": config.hold_suppression_overlap_ratio,
            "extended_overlap_review_sec": config.extended_overlap_review_sec,
            "technical_call_min_sec": config.technical_call_min_sec,
            "technical_call_max_sec": config.technical_call_max_sec,
            "non_technical_call_min_sec": config.non_technical_call_min_sec,
            "non_technical_call_max_sec": config.non_technical_call_max_sec,
        },
    )
