"""
WE-inspired Hold policy evaluation.

Evaluates Hold candidates against configurable operational policy thresholds.
This is a prototype policy based on project-owner operational experience.

Do NOT describe this as:
- An official WE policy
- A universal call-center standard
- Legally or scientifically validated thresholds
- Employee fault or punishment
"""

from typing import List

from .schemas import (
    HoldCandidate,
    HoldPolicyConfig,
    HoldDurationViolation,
    HoldPolicyResult,
    UncountedOverDuration,
)


def evaluate_hold_policy(
    hold_events: List[HoldCandidate],
    config: HoldPolicyConfig,
) -> HoldPolicyResult:
    """
    Evaluate Hold candidates against the WE-inspired policy.

    Only events matching the configured counting_mode are counted
    toward the policy limits. Uncounted candidates that exceed the
    duration threshold are surfaced for review but do not trigger
    policy violations.

    Args:
        hold_events: List of HoldCandidate objects from detection.
        config: Policy configuration with limits and counting behavior.

    Returns:
        HoldPolicyResult with compliance status and any violations.
    """
    # ── Filter events by counting mode ─────────────────────────────────
    counted = _filter_by_counting_mode(hold_events, config.counting_mode)
    counted_ids = {e.candidate_id for e in counted}

    # ── Check count limit ──────────────────────────────────────────────
    count_exceeded = len(counted) > config.max_hold_count

    # ── Check duration limits on counted events ────────────────────────
    duration_violations = []
    counted_hold_ranges = []
    for event in counted:
        counted_hold_ranges.append((event.start_time, event.end_time))
        if event.duration_sec > config.max_hold_duration_sec:
            duration_violations.append(HoldDurationViolation(
                candidate_id=event.candidate_id,
                start_time=event.start_time,
                end_time=event.end_time,
                actual_duration_sec=event.duration_sec,
                allowed_duration_sec=config.max_hold_duration_sec,
                excess_duration_sec=round(event.duration_sec - config.max_hold_duration_sec, 6),
            ))

    # ── Surface uncounted candidates exceeding duration ────────────────
    uncounted_over_duration = []
    for event in hold_events:
        if event.candidate_id in counted_ids:
            continue
        if event.duration_sec > config.max_hold_duration_sec:
            uncounted_over_duration.append(UncountedOverDuration(
                candidate_id=event.candidate_id,
                start_time=event.start_time,
                end_time=event.end_time,
                duration_sec=event.duration_sec,
                classification=event.classification,
                exceeds_duration_by_sec=round(event.duration_sec - config.max_hold_duration_sec, 6),
                reaches_call_end=event.reaches_call_end,
            ))

    # ── Overall compliance ─────────────────────────────────────────────
    compliant = not count_exceeded and len(duration_violations) == 0
    review_required = (
        not compliant
        or len(uncounted_over_duration) > 0
    )

    # ── Build review reason ────────────────────────────────────────────
    review_reason = ""
    if count_exceeded:
        review_reason += (
            f"Hold count limit exceeded: {len(counted)} counted events "
            f"(limit: {config.max_hold_count}). "
        )
    if duration_violations:
        for v in duration_violations:
            review_reason += (
                f"Hold {v.candidate_id} duration {v.actual_duration_sec:.1f}s "
                f"exceeds limit of {v.allowed_duration_sec:.1f}s "
                f"(excess: {v.excess_duration_sec:.1f}s). "
            )
    if uncounted_over_duration:
        for u in uncounted_over_duration:
            review_reason += (
                f"Uncounted candidate {u.candidate_id} duration {u.duration_sec:.1f}s "
                f"exceeds limit (excess: {u.exceeds_duration_by_sec:.1f}s, "
                f"classification: {u.classification}, "
                f"reaches_call_end: {u.reaches_call_end}). "
            )
    if not review_reason:
        review_reason = "Policy compliant. No review required."
    else:
        review_reason = "Review suggested: " + review_reason

    return HoldPolicyResult(
        counted_hold_events=len(counted),
        total_candidate_events=len(hold_events),
        count_limit=config.max_hold_count,
        count_exceeded=count_exceeded,
        duration_limit_sec=config.max_hold_duration_sec,
        duration_violations=duration_violations,
        uncounted_over_duration=uncounted_over_duration,
        counted_hold_ranges=counted_hold_ranges,
        compliant=compliant,
        review_required=review_required,
        policy_source=config.policy_source,
        policy_version=config.policy_version,
        review_reason=review_reason,
    )


def _filter_by_counting_mode(
    events: List[HoldCandidate],
    counting_mode: str,
) -> List[HoldCandidate]:
    """Filter Hold events based on the configured counting mode."""
    if counting_mode == "count_all_candidates":
        return events
    elif counting_mode == "count_confirmed_only":
        return [
            e for e in events
            if e.classification == "confirmed_by_metadata_or_manual_input"
            or e.manual_status == "confirmed_hold"
        ]
    elif counting_mode == "count_likely_and_confirmed":
        return [
            e for e in events
            if e.manual_status != "rejected_hold"
            and (e.classification in ("likely_hold_candidate", "confirmed_by_metadata_or_manual_input")
                 or e.manual_status == "confirmed_hold")
        ]
    else:
        # Unknown mode — count nothing
        return []
