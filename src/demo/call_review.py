"""
Phase VII — Final Call Review.

Aggregates all approved pipeline outputs into a single structured
Team Leader review package. No new detection logic is created.
This module only aggregates, normalizes, prioritizes, summarizes,
and exposes existing evidence.

Do NOT:
- Create new detection logic
- Produce a numeric QA score
- Produce Pass/Fail
- Claim employee fault
- Infer anger, satisfaction, professionalism, or intent
- Describe Hold as a confirmed system event
"""

import logging
from typing import Dict, List, Optional, Tuple

from .schemas import (
    ActivityInterval,
    CallDurationResult,
    CallSummary,
    ChannelSerSummary,
    FinalCallReview,
    FinalCallReviewConfig,
    HoldDetectionResult,
    HoldDurationViolation,
    HoldPolicyResult,
    HoldReviewCandidate,
    HoldReviewSection,
    OperationalAlert,
    OperationalAlertResult,
    ReviewReason,
    SpeakerActivitySection,
    SpeakerSummary,
    TimelineItem,
    UncountedOverDuration,
)

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# EVIDENCE LIMITATIONS
# ══════════════════════════════════════════════════════════════════════

_EVIDENCE_LIMITATIONS = [
    {
        "code": "hold_audio_derived",
        "text": (
            "Hold detection is inferred from audio features, not from "
            "a system Hold-button event log. Hold candidates represent "
            "audio-derived evidence, not confirmed Hold events."
        ),
    },
    {
        "code": "activity_binary",
        "text": (
            "Activity states use binary active/inactive classification. "
            "They do not prove semantic speech and cannot distinguish "
            "speech from tones, music, or environmental audio."
        ),
    },
    {
        "code": "ser_coarse_labels",
        "text": (
            "SER labels are coarse model-predicted classes "
            "(Low Vocal Activation, Moderate Vocal Activation, High Vocal Activation) only. "
            "They do not infer anger, satisfaction, or any specific emotion."
        ),
    },
    {
        "code": "audio_only_scope",
        "text": (
            "This system analyzes audio-derived evidence only. "
            "ASR, transcription, NLP, and linguistic analysis are "
            "outside the project scope."
        ),
    },
    {
        "code": "alerts_review_suggestions",
        "text": (
            "Operational alerts are review suggestions for a Team Leader, "
            "not proof of employee fault or automatic policy violations."
        ),
    },
    {
        "code": "thresholds_prototype",
        "text": (
            "All thresholds are configurable prototype operational "
            "guidelines based on project-owner call-center experience. "
            "They are not official company policies."
        ),
    },
    {
        "code": "no_numeric_score",
        "text": (
            "No numeric QA score is produced. The review status is a "
            "structured recommendation, not a quantitative evaluation."
        ),
    },
]


# ══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════

def build_call_review(
    call_duration_sec: float,
    call_type: str,
    call_summary: CallSummary,
    hold_detection_result: HoldDetectionResult,
    hold_policy_result: HoldPolicyResult,
    operational_alert_result: OperationalAlertResult,
    call_id: Optional[str] = None,
    config: Optional[FinalCallReviewConfig] = None,
) -> FinalCallReview:
    """
    Build a Final Call Review from all approved pipeline outputs.

    This is an aggregation layer only. No new detection logic is created.

    Args:
        call_duration_sec: Total call duration in seconds.
        call_type: "technical" | "non_technical" | "unknown".
        call_summary: Speaker summaries, activity intervals, and
            inactivity metadata from the pipeline.
        hold_detection_result: Hold candidate detection output.
        hold_policy_result: Hold policy evaluation output.
        operational_alert_result: Operational alert output.
        call_id: Optional call identifier.
        config: Review construction configuration.

    Returns:
        FinalCallReview with all sections populated.
    """
    if config is None:
        config = FinalCallReviewConfig()

    # ── Validate inputs ────────────────────────────────────────────────
    call_duration_sec = max(0.0, call_duration_sec)
    call_type = _normalize_call_type(call_type)

    # ── Build Hold Review Section ──────────────────────────────────────
    hold_section = _build_hold_review_section(
        hold_detection_result, hold_policy_result, config
    )

    # ── Build Speaker Activity Section ─────────────────────────────────
    speaker_activity = _build_speaker_activity(
        call_summary, call_duration_sec
    )

    # ── Build SER Summary ──────────────────────────────────────────────
    ser_summary = _build_ser_summary(call_summary)

    # ── Build Operational Alerts Summary ───────────────────────────────
    alerts_summary = _build_alerts_summary(operational_alert_result)

    # ── Build Review Reasons (deduplicated) ────────────────────────────
    review_reasons = _build_review_reasons(
        hold_detection_result, hold_policy_result,
        operational_alert_result, config
    )

    # ── Build Timeline ─────────────────────────────────────────────────
    timeline = _build_timeline(
        hold_detection_result, hold_policy_result,
        operational_alert_result, call_duration_sec, config
    )

    # ── Determine Review Status ────────────────────────────────────────
    review_status = _determine_review_status(
        operational_alert_result, hold_policy_result, review_reasons
    )

    # ── Configuration snapshot ─────────────────────────────────────────
    config_snapshot = {}
    if config.include_configuration_snapshot:
        config_snapshot = {
            "priority_severities": config.priority_severities,
            "include_info_alerts": config.include_info_alerts,
            "include_uncertain_hold_candidates": config.include_uncertain_hold_candidates,
            "max_timeline_items": config.max_timeline_items,
            "max_review_reasons": config.max_review_reasons,
        }

    return FinalCallReview(
        call_id=call_id,
        call_duration_sec=call_duration_sec,
        call_type=call_type,
        review_status=review_status,
        manual_review_suggested=operational_alert_result.manual_review_suggested,
        review_reasons=review_reasons,
        hold_review=hold_section,
        operational_alerts_summary=alerts_summary,
        speaker_activity=speaker_activity,
        ser_summary=ser_summary,
        timeline=timeline,
        evidence_limitations=list(_EVIDENCE_LIMITATIONS),
        configuration_snapshot=config_snapshot,
    )


# ══════════════════════════════════════════════════════════════════════
# REVIEW STATUS
# ══════════════════════════════════════════════════════════════════════

def _determine_review_status(
    alert_result: OperationalAlertResult,
    policy_result: HoldPolicyResult,
    review_reasons: List[ReviewReason],
) -> str:
    """Determine the final review status from structured fields only.

    Rules:
    - priority_review: any high-severity alert, Hold count violation,
      Hold duration violation, or other explicitly high-priority item.
    - review_suggested: manual_review_suggested=True, warning items,
      uncounted over-duration Hold, above-expected call duration,
      significant Dead Air or overlap.
    - no_immediate_review: no structured item requiring review.
    """
    # Check for priority items
    has_high_severity = any(a.severity == "high" for a in alert_result.alerts)
    has_count_violation = policy_result.count_exceeded
    has_duration_violation = len(policy_result.duration_violations) > 0

    if has_high_severity or has_count_violation or has_duration_violation:
        return "priority_review"

    # Check for review-suggested items
    if alert_result.manual_review_suggested:
        return "review_suggested"
    if len(policy_result.uncounted_over_duration) > 0:
        return "review_suggested"
    if (alert_result.call_duration_result is not None
            and alert_result.call_duration_result.comparison == "above_expected_range"):
        return "review_suggested"
    has_warning = any(a.severity == "warning" for a in alert_result.alerts)
    if has_warning:
        return "review_suggested"

    return "no_immediate_review"


# ══════════════════════════════════════════════════════════════════════
# HOLD REVIEW SECTION
# ══════════════════════════════════════════════════════════════════════

def _build_hold_review_section(
    detection: HoldDetectionResult,
    policy: HoldPolicyResult,
    config: FinalCallReviewConfig,
) -> HoldReviewSection:
    """Build the Hold review section with candidate details.

    counted_by_policy is derived from HoldPolicyResult.counted_hold_ranges
    using timestamp matching, NOT from the candidate's classification label.

    Timestamp matching rules:
    - Tolerance: 0.5 seconds (half a second) to handle floating-point
      boundary differences between detection and policy ranges.
    - A candidate is "counted" if its (start_time, end_time) falls
      within any counted_hold_range entry, within tolerance.
    - If multiple ranges match, the candidate is counted.
    - If no ranges match, counted_by_policy=False regardless of
      classification.
    """
    # Build candidate details
    candidates = []

    for c in detection.candidates:
        if not config.include_uncertain_hold_candidates:
            if c.classification == "uncertain_hold_candidate":
                continue

        counted = _is_counted_by_policy(c, policy.counted_hold_ranges)
        candidates.append(HoldReviewCandidate(
            candidate_id=c.candidate_id,
            start_time=c.start_time,
            end_time=c.end_time,
            duration_sec=c.duration_sec,
            classification=c.classification,
            evidence_types=list(c.evidence_types),
            reaches_call_end=c.reaches_call_end,
            counted_by_policy=counted,
        ))

    return HoldReviewSection(
        total_candidates=len(candidates),
        likely_candidates=sum(1 for c in candidates if c.classification == "likely_hold_candidate"),
        uncertain_candidates=sum(1 for c in candidates if c.classification == "uncertain_hold_candidate"),
        counted_hold_events=policy.counted_hold_events,
        counted_hold_ranges=list(policy.counted_hold_ranges),
        count_limit=policy.count_limit,
        count_exceeded=policy.count_exceeded,
        duration_violations=list(policy.duration_violations),
        uncounted_over_duration=list(policy.uncounted_over_duration),
        counting_mode=policy.policy_version,
        candidates=candidates,
        policy_source=policy.policy_source,
        policy_version=policy.policy_version,
        configuration_snapshot=policy.__dict__ if config.include_configuration_snapshot else {},
    )


def _is_counted_by_policy(
    candidate,
    counted_hold_ranges: List[Tuple[float, float]],
    tolerance: float = 0.5,
) -> bool:
    """Determine if a Hold candidate is counted by the configured policy.

    Matches candidate time range against counted_hold_ranges from
    HoldPolicyResult using timestamp comparison with tolerance.

    Rules:
    - Tolerance: 0.5 seconds. A candidate is matched if its start_time
      and end_time each fall within tolerance of a counted range boundary.
    - Duplicate ranges: if multiple ranges match, the candidate is counted.
    - Overlapping ranges: if any range overlaps the candidate within
      tolerance, the candidate is counted.
    - No classification-based inference: counted_by_policy is derived
      solely from structured HoldPolicyResult evidence.
    """
    for range_start, range_end in counted_hold_ranges:
        if (abs(candidate.start_time - range_start) <= tolerance
                and abs(candidate.end_time - range_end) <= tolerance):
            return True
    return False


# ══════════════════════════════════════════════════════════════════════
# SPEAKER ACTIVITY
# ══════════════════════════════════════════════════════════════════════

def _build_speaker_activity(
    call_summary: CallSummary,
    call_duration: float,
) -> SpeakerActivitySection:
    """Build speaker activity section from CallSummary."""
    if call_duration <= 0:
        return SpeakerActivitySection(call_duration_sec=call_duration)

    agent_active = call_summary.agent_active_only_duration_sec + call_summary.both_active_duration_sec
    customer_active = call_summary.customer_active_only_duration_sec + call_summary.both_active_duration_sec
    overlap = call_summary.both_active_duration_sec
    both_inactive = call_summary.neither_active_duration_sec

    return SpeakerActivitySection(
        agent_active_duration_sec=round(agent_active, 6),
        customer_active_duration_sec=round(customer_active, 6),
        overlap_duration_sec=round(overlap, 6),
        both_inactive_duration_sec=round(both_inactive, 6),
        agent_activity_ratio=round(agent_active / call_duration, 6) if call_duration > 0 else 0.0,
        customer_activity_ratio=round(customer_active / call_duration, 6) if call_duration > 0 else 0.0,
        overlap_ratio=round(overlap / call_duration, 6) if call_duration > 0 else 0.0,
        inactivity_ratio=round(both_inactive / call_duration, 6) if call_duration > 0 else 0.0,
        total_intervals=len(call_summary.activity_state_intervals),
        call_duration_sec=call_duration,
    )


# ══════════════════════════════════════════════════════════════════════
# SER SUMMARY
# ══════════════════════════════════════════════════════════════════════

def _build_ser_summary(call_summary: CallSummary) -> Dict:
    """Build SER summary section from CallSummary."""
    channels = {}
    for summary in [call_summary.agent_summary, call_summary.customer_summary]:
        if summary is None:
            continue
        analyzed = summary.analyzed_windows
        dist = summary.dataset_label_distribution
        pcts = summary.dataset_label_percentages

        channels[summary.role] = ChannelSerSummary(
            role=summary.role,
            dominant_label=summary.dominant_dataset_label,
            low_emotion_ratio=round(pcts.get("Low Vocal Activation", 0.0), 4),
            neutral_emotion_ratio=round(pcts.get("Moderate Vocal Activation", 0.0), 4),
            high_emotion_ratio=round(pcts.get("High Vocal Activation", 0.0), 4),
            analyzed_windows=analyzed,
            uncertain_windows=summary.uncertain_windows,
            average_confidence=summary.average_confidence,
        )

    return {
        "agent": channels.get("Agent"),
        "customer": channels.get("Customer"),
        "note": (
            "Model-predicted SER level distribution. "
            "Do not interpret as emotion, satisfaction, or professionalism."
        ),
    }


# ══════════════════════════════════════════════════════════════════════
# ALERTS SUMMARY
# ══════════════════════════════════════════════════════════════════════

def _build_alerts_summary(alert_result: OperationalAlertResult) -> Dict:
    """Build operational alerts summary with sorted alerts."""
    sorted_alerts = _sort_alerts(alert_result.alerts)

    return {
        "total_alerts": len(alert_result.alerts),
        "alerts_by_severity": dict(alert_result.alert_counts_by_severity),
        "alerts_by_category": {
            k: len(v) for k, v in alert_result.alerts_by_category.items()
        },
        "sorted_alerts": [
            {
                "alert_id": a.alert_id,
                "alert_type": a.alert_type,
                "severity": a.severity,
                "start_time": a.start_time,
                "end_time": a.end_time,
                "duration_sec": a.duration_sec,
                "title": a.title,
                "requires_manual_review": a.requires_manual_review,
            }
            for a in sorted_alerts
        ],
        "manual_review_suggested": alert_result.manual_review_suggested,
        "call_duration_result": (
            {
                "call_type": alert_result.call_duration_result.call_type,
                "comparison": alert_result.call_duration_result.comparison,
                "expected_min_sec": alert_result.call_duration_result.expected_min_sec,
                "expected_max_sec": alert_result.call_duration_result.expected_max_sec,
            }
            if alert_result.call_duration_result else None
        ),
    }


def _sort_alerts(alerts: List[OperationalAlert]) -> List[OperationalAlert]:
    """Sort alerts: high → warning → info, then by start_time.

    Alerts without timestamps go last within their severity group.
    """
    severity_order = {"high": 0, "warning": 1, "info": 2}

    def sort_key(a: OperationalAlert):
        sev = severity_order.get(a.severity, 3)
        # Alerts without start_time go last
        start = a.start_time if a.start_time is not None else float("inf")
        return (sev, start)

    return sorted(alerts, key=sort_key)


# ══════════════════════════════════════════════════════════════════════
# REVIEW REASONS (deduplicated)
# ══════════════════════════════════════════════════════════════════════

def _build_review_reasons(
    detection: HoldDetectionResult,
    policy: HoldPolicyResult,
    alert_result: OperationalAlertResult,
    config: FinalCallReviewConfig,
) -> List[ReviewReason]:
    """Build deduplicated review reasons from all sources.

    Precedence for the same Hold candidate:
    1. Hold policy violation (from duration_violations)
    2. Hold operational alert (from alert_result)
    3. Hold candidate evidence (from detection)

    Unrelated events remain separate even if timestamps overlap.
    """
    reasons = []
    seen_candidate_ids = set()
    seen_alert_ids = set()

    # ── 1. Hold policy violations (highest precedence) ─────────────────
    for v in policy.duration_violations:
        seen_candidate_ids.add(v.candidate_id)
        # Find related alert
        related_alert = _find_alert_for_hold(alert_result, v.candidate_id)
        alert_id = related_alert.alert_id if related_alert else ""
        if alert_id:
            seen_alert_ids.add(alert_id)

        reasons.append(ReviewReason(
            reason_code="hold_duration_exceeded",
            title=f"Hold Duration Exceeded — {v.candidate_id}",
            explanation=(
                f"Audio-derived Hold candidate {v.candidate_id} lasted "
                f"{v.actual_duration_sec:.1f}s, exceeding the configured "
                f"limit of {v.allowed_duration_sec:.1f}s "
                f"(excess: {v.excess_duration_sec:.1f}s). "
                f"Counted under the configured prototype policy."
            ),
            severity="high",
            source_category="hold",
            related_alert_ids=[alert_id] if alert_id else [],
            related_candidate_ids=[v.candidate_id],
            start_time=v.start_time,
            end_time=v.end_time,
            duration_sec=v.actual_duration_sec,
            evidence_level="policy_threshold",
            metadata={
                "allowed_duration_sec": v.allowed_duration_sec,
                "excess_duration_sec": v.excess_duration_sec,
            },
        ))

    # ── 2. Hold count violation ────────────────────────────────────────
    if policy.count_exceeded:
        reasons.append(ReviewReason(
            reason_code="hold_count_exceeded",
            title="Hold Count Limit Exceeded",
            explanation=(
                f"{policy.counted_hold_events} Hold events were counted, "
                f"exceeding the limit of {policy.count_limit}."
            ),
            severity="high",
            source_category="hold",
            evidence_level="policy_threshold",
        ))

    # ── 3. Uncounted over-duration Hold review items ───────────────────
    for u in policy.uncounted_over_duration:
        seen_candidate_ids.add(u.candidate_id)
        related_alert = _find_alert_for_hold(alert_result, u.candidate_id)
        alert_id = related_alert.alert_id if related_alert else ""
        if alert_id:
            seen_alert_ids.add(alert_id)

        reasons.append(ReviewReason(
            reason_code="uncounted_hold_duration_review",
            title=f"Uncounted Hold Review — {u.candidate_id}",
            explanation=(
                f"Hold candidate {u.candidate_id} ({u.classification}) "
                f"lasted {u.duration_sec:.1f}s, exceeding the duration "
                f"limit (excess: {u.exceeds_duration_by_sec:.1f}s). "
                f"Excluded from counted policy events by the configured "
                f"counting mode."
            ),
            severity="warning",
            source_category="hold",
            related_alert_ids=[alert_id] if alert_id else [],
            related_candidate_ids=[u.candidate_id],
            start_time=u.start_time,
            end_time=u.end_time,
            duration_sec=u.duration_sec,
            evidence_level="policy_threshold",
            metadata={"classification": u.classification},
        ))

    # ── 4. Hold candidate evidence (lowest precedence for Hold) ────────
    for c in detection.candidates:
        if c.candidate_id in seen_candidate_ids:
            continue
        if not config.include_uncertain_hold_candidates:
            if c.classification == "uncertain_hold_candidate":
                continue

        code = "likely_hold_candidate" if c.classification == "likely_hold_candidate" else "uncertain_hold_candidate"
        reasons.append(ReviewReason(
            reason_code=code,
            title=f"Hold Candidate — {c.candidate_id}",
            explanation=(
                f"Audio-derived {c.classification.replace('_', ' ')} "
                f"({c.candidate_id}) from {c.start_time:.1f}s to "
                f"{c.end_time:.1f}s ({c.duration_sec:.1f}s). "
                f"Evidence: {', '.join(c.evidence_types)}."
            ),
            severity="warning" if c.classification == "likely_hold_candidate" else "info",
            source_category="hold",
            related_candidate_ids=[c.candidate_id],
            start_time=c.start_time,
            end_time=c.end_time,
            duration_sec=c.duration_sec,
            evidence_level="audio_derived",
            metadata={"classification": c.classification},
        ))

    # ── 5. Operational alerts (non-Hold) ───────────────────────────────
    for a in alert_result.alerts:
        if a.alert_id in seen_alert_ids:
            continue
        if a.source_category == "hold":
            # Hold alerts already covered above
            continue

        if a.severity == "info" and not config.include_info_alerts:
            continue

        code = _alert_to_reason_code(a)
        reasons.append(ReviewReason(
            reason_code=code,
            title=a.title,
            explanation=a.explanation,
            severity=a.severity,
            source_category=a.source_category,
            related_alert_ids=[a.alert_id],
            start_time=a.start_time,
            end_time=a.end_time,
            duration_sec=a.duration_sec,
            evidence_level="audio_derived",
            metadata=a.metadata,
        ))

    # ── 6. Call duration review ────────────────────────────────────────
    cdr = alert_result.call_duration_result
    if cdr and cdr.comparison in ("above_expected_range", "below_expected_range"):
        code = "call_duration_above_expected" if cdr.comparison == "above_expected_range" else "call_duration_below_expected"
        sev = "warning" if cdr.comparison == "above_expected_range" else "info"
        reasons.append(ReviewReason(
            reason_code=code,
            title=f"Call Duration {cdr.comparison.replace('_', ' ').title()}",
            explanation=(
                f"Call duration {cdr.call_duration_sec:.1f}s is "
                f"{cdr.comparison.replace('_', ' ')} for "
                f"{cdr.call_type} calls "
                f"({cdr.expected_min_sec:.0f}s–{cdr.expected_max_sec:.0f}s)."
            ),
            severity=sev,
            source_category="duration",
            start_time=0.0,
            end_time=cdr.call_duration_sec,
            duration_sec=cdr.call_duration_sec,
            evidence_level="policy_threshold",
        ))

    # Cap at max_review_reasons
    return reasons[:config.max_review_reasons]


def _find_alert_for_hold(
    alert_result: OperationalAlertResult,
    candidate_id: str,
) -> Optional[OperationalAlert]:
    """Find an operational alert related to a Hold candidate_id."""
    for a in alert_result.alerts:
        if a.source_category == "hold":
            if a.metadata.get("candidate_id") == candidate_id:
                return a
    return None


def _alert_to_reason_code(alert: OperationalAlert) -> str:
    """Map an operational alert type to a review reason code."""
    mapping = {
        "dead_air": "dead_air",
        "extended_overlap": "extended_overlap",
        "call_duration": "other_operational_alert",
        "hold_duration_violation": "hold_duration_exceeded",
        "hold_count_violation": "hold_count_exceeded",
        "hold_uncounted_over_duration": "uncounted_hold_duration_review",
    }
    return mapping.get(alert.alert_type, "other_operational_alert")


# ══════════════════════════════════════════════════════════════════════
# TIMELINE
# ══════════════════════════════════════════════════════════════════════

def _build_timeline(
    detection: HoldDetectionResult,
    policy: HoldPolicyResult,
    alert_result: OperationalAlertResult,
    call_duration: float,
    config: FinalCallReviewConfig,
) -> List[TimelineItem]:
    """Build a chronological timeline of key review events.

    Includes only review-relevant events. Deduplicates using the
    same rules as review reasons.
    """
    items = []
    seen_candidate_ids = set()

    # Hold policy violations
    for v in policy.duration_violations:
        seen_candidate_ids.add(v.candidate_id)
        items.append(TimelineItem(
            item_type="hold_violation",
            start_time=v.start_time,
            end_time=v.end_time,
            duration_sec=v.actual_duration_sec,
            severity="high",
            title=f"Hold Duration Exceeded — {v.candidate_id}",
            source="hold_policy",
            related_ids=[v.candidate_id],
            explanation=(
                f"Audio-derived Hold candidate exceeded duration limit "
                f"by {v.excess_duration_sec:.1f}s."
            ),
        ))

    # Uncounted over-duration
    for u in policy.uncounted_over_duration:
        seen_candidate_ids.add(u.candidate_id)
        items.append(TimelineItem(
            item_type="hold_candidate",
            start_time=u.start_time,
            end_time=u.end_time,
            duration_sec=u.duration_sec,
            severity="warning",
            title=f"Uncounted Hold — {u.candidate_id}",
            source="hold_policy",
            related_ids=[u.candidate_id],
            explanation=(
                f"Hold candidate ({u.classification}) excluded from "
                f"counted events but exceeds duration limit."
            ),
        ))

    # Hold candidates not yet seen
    for c in detection.candidates:
        if c.candidate_id in seen_candidate_ids:
            continue
        if not config.include_uncertain_hold_candidates:
            if c.classification == "uncertain_hold_candidate":
                continue
        sev = "warning" if c.classification == "likely_hold_candidate" else "info"
        items.append(TimelineItem(
            item_type="hold_candidate",
            start_time=c.start_time,
            end_time=c.end_time,
            duration_sec=c.duration_sec,
            severity=sev,
            title=f"Hold Candidate — {c.candidate_id}",
            source="hold_detection",
            related_ids=[c.candidate_id],
            explanation=(
                f"Audio-derived {c.classification.replace('_', ' ')}."
            ),
        ))

    # Operational alerts (non-Hold, already on timeline as reasons)
    for a in alert_result.alerts:
        if a.source_category == "hold":
            continue
        if a.start_time is None:
            continue
        items.append(TimelineItem(
            item_type=a.alert_type,
            start_time=a.start_time,
            end_time=a.end_time or a.start_time,
            duration_sec=a.duration_sec or 0.0,
            severity=a.severity,
            title=a.title,
            source="operational_alerts",
            related_ids=[a.alert_id],
            explanation=a.title,
        ))

    # Sort chronologically
    items.sort(key=lambda x: (x.start_time, x.end_time))

    return items[:config.max_timeline_items]


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def _normalize_call_type(call_type: str) -> str:
    """Normalize call type to supported values."""
    normalized = call_type.strip().lower().replace(" ", "_")
    if normalized in ("technical", "non_technical", "unknown"):
        return normalized
    return "unknown"
