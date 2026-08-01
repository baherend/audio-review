"""
Per-speaker summary computation, timeline generation, and activity analysis.

Computes speaker summaries from processing results.
Generates non-overlapping activity-state intervals for Hold-ready analysis.
"""

import json
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from .schemas import (
    SpeakerProcessingResult,
    SpeakerSummary,
    SharedWindowResult,
    EmotionTimelineEntry,
    ActivityStateEntry,
    ActivityInterval,
    CallSummary,
)


# ══════════════════════════════════════════════════════════════════════
# SPEAKER SUMMARY
# ══════════════════════════════════════════════════════════════════════
def compute_speaker_summary(
    speaker_result: SpeakerProcessingResult,
    call_duration_sec: float,
    minimum_confidence: Optional[float] = None,
) -> SpeakerSummary:
    """
    Compute per-speaker summary from processing results.

    Duration metrics use hop-based non-overlapping calculation to avoid
    double-counting from overlapping windows.
    """
    windows = speaker_result.windows
    if not windows:
        return SpeakerSummary(
            role=speaker_result.role,
            channel=speaker_result.channel,
            role_assignment_method=speaker_result.role_assignment_method,
            total_call_duration_sec=call_duration_sec,
        )

    role = speaker_result.role
    total_windows = len(windows)
    active_windows = sum(1 for w in windows if _speaker(w, role).speech_active)
    inactive_windows = total_windows - active_windows

    # Analyzed = has a prediction (active and inference succeeded)
    analyzed_windows = sum(1 for w in windows if _speaker(w, role).prediction is not None)

    # Uncertain = has prediction but below minimum confidence
    uncertain_windows = 0
    if minimum_confidence is not None:
        uncertain_windows = sum(
            1 for w in windows
            if _speaker(w, role).prediction is not None
            and _speaker(w, role).prediction.confidence < minimum_confidence
        )

    # ── Duration metrics (non-overlapping via hop-based calculation) ───
    # The hop defines the non-overlapping base interval.
    # Each window covers [start, start + window_duration].
    # With hop=h, each base interval of length h belongs to exactly one window.
    # We assign activity to a base interval based on the window that starts at it.
    active_duration, inactive_duration = _compute_hop_durations(windows, speaker_result.role)

    # ── Label distribution (analyzed active windows only) ──────────────
    labels = []
    confidences = []
    for w in windows:
        sp = _speaker(w, role)
        if sp.prediction is not None and sp.speech_active:
            labels.append(sp.prediction.label)
            confidences.append(sp.prediction.confidence)

    label_counts = Counter(labels)
    total_analyzed = len(labels)

    if total_analyzed > 0:
        dominant_label = label_counts.most_common(1)[0][0]
        # Tie-breaking: most_common returns first encountered for ties
        label_percentages = {
            label: round(count / total_analyzed * 100, 2)
            for label, count in label_counts.items()
        }
    else:
        dominant_label = None
        label_percentages = {}

    # ── Confidence ─────────────────────────────────────────────────────
    avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else None
    min_conf = round(min(confidences), 4) if confidences else None
    max_conf = round(max(confidences), 4) if confidences else None

    # ── Label transitions (active windows only, in order) ──────────────
    active_labels = []
    for w in windows:
        sp = _speaker(w, role)
        if sp.prediction is not None and sp.speech_active:
            active_labels.append(sp.prediction.label)

    transitions = 0
    for i in range(1, len(active_labels)):
        if active_labels[i] != active_labels[i - 1]:
            transitions += 1

    # ── First/last active ──────────────────────────────────────────────
    first_active_time = None
    last_active_time = None
    first_active_label = None
    last_active_label = None
    for w in windows:
        sp = _speaker(w, role)
        if sp.speech_active and sp.prediction is not None:
            if first_active_time is None:
                first_active_time = w.window.original_start
                first_active_label = sp.prediction.label
            last_active_time = w.window.original_end
            last_active_label = sp.prediction.label

    return SpeakerSummary(
        role=speaker_result.role,
        channel=speaker_result.channel,
        role_assignment_method=speaker_result.role_assignment_method,
        total_windows=total_windows,
        active_windows=active_windows,
        inactive_windows=inactive_windows,
        analyzed_windows=analyzed_windows,
        uncertain_windows=uncertain_windows,
        total_call_duration_sec=call_duration_sec,
        detected_active_duration_sec=round(active_duration, 4),
        detected_inactive_duration_sec=round(inactive_duration, 4),
        dominant_dataset_label=dominant_label,
        dataset_label_distribution=dict(label_counts),
        dataset_label_percentages=label_percentages,
        average_confidence=avg_conf,
        minimum_confidence=min_conf,
        maximum_confidence=max_conf,
        label_transition_count=transitions,
        first_active_time=first_active_time,
        last_active_time=last_active_time,
        first_active_label=first_active_label,
        last_active_label=last_active_label,
    )


def _speaker(wr: SharedWindowResult, role: str = None):
    """Get the speaker result for the given role."""
    if role == "Agent" or (role is None and hasattr(wr, 'agent')):
        return wr.agent
    return wr.customer


def _compute_hop_durations(
    windows: List[SharedWindowResult], role: str
) -> Tuple[float, float]:
    """
    Compute active/inactive durations using hop-based non-overlapping intervals.

    Method: Build base intervals [0, hop, 2*hop, ..., call_duration].
    Each base interval is assigned the state of the overlapping window
    that starts at that base interval's start time. The final interval
    extends to call_duration_sec.

    Returns: (active_duration, inactive_duration) in seconds.
    """
    if not windows:
        return 0.0, 0.0

    # Infer hop
    if len(windows) >= 2:
        hop = windows[1].window.original_start - windows[0].window.original_start
    else:
        hop = windows[0].window.original_end - windows[0].window.original_start

    if hop <= 0:
        hop = windows[0].window.original_end - windows[0].window.original_start

    # Call duration from last window
    call_duration = windows[-1].window.original_end

    # Build base interval boundaries covering full call
    boundaries = []
    t = 0.0
    while t < call_duration:
        boundaries.append(round(t, 6))
        t += hop
    boundaries.append(round(call_duration, 6))

    # Map window starts to window results for lookup
    window_by_start = {w.window.original_start: w for w in windows}

    active_duration = 0.0
    inactive_duration = 0.0

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        interval_duration = end - start

        # Find the overlapping window that starts at this base interval's start
        wr = window_by_start.get(start)
        if wr is None:
            # No window starts exactly here — find the closest preceding window
            for ws in sorted(window_by_start.keys()):
                if ws <= start:
                    wr = window_by_start[ws]
                else:
                    break

        if wr is not None:
            sp = _speaker(wr, role)
            if sp.speech_active:
                active_duration += interval_duration
            else:
                inactive_duration += interval_duration
        else:
            inactive_duration += interval_duration

    return active_duration, inactive_duration


# ══════════════════════════════════════════════════════════════════════
# EMOTION TIMELINE
# ══════════════════════════════════════════════════════════════════════
def build_emotion_timeline(
    speaker_result: SpeakerProcessingResult,
) -> List[EmotionTimelineEntry]:
    """
    Build a per-speaker emotion timeline from processing results.

    Includes all windows (active and inactive). Never deletes silence.
    """
    entries = []
    role = speaker_result.role
    for wr in speaker_result.windows:
        sp = _speaker(wr, role)
        prediction = sp.prediction

        entries.append(EmotionTimelineEntry(
            window_id=wr.window.window_id,
            original_start=wr.window.original_start,
            original_end=wr.window.original_end,
            speech_active=sp.speech_active,
            dataset_label=prediction.label if prediction else None,
            probabilities=dict(prediction.probabilities) if prediction else None,
            confidence=prediction.confidence if prediction else None,
            uncertainty_reason=sp.uncertainty_reason,
            inactive_reason=sp.inactive_reason,
        ))

    return entries


# ══════════════════════════════════════════════════════════════════════
# ACTIVITY STATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════
def build_activity_states(
    window_results: List[SharedWindowResult],
    agent_waveform: Optional[np.ndarray] = None,
    customer_waveform: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
    call_duration_sec: Optional[float] = None,
    rms_threshold_db: float = -40.0,
    active_ratio_threshold: float = 0.3,
) -> List[ActivityStateEntry]:
    """
    Build per-base-interval activity states from shared window results.

    Computes activity directly from waveforms for each non-overlapping base interval.
    Does NOT inherit overlapping-window states. Covers the full call duration.

    Args:
        window_results: Shared window results (for window structure).
        agent_waveform: Agent channel waveform (for re-computing activity).
        customer_waveform: Customer channel waveform (for re-computing activity).
        sample_rate: Sample rate of the waveforms.
        call_duration_sec: Total call duration (for final interval).
        rms_threshold_db: RMS threshold for speech-active detection.
        active_ratio_threshold: Minimum active ratio for speech-active.
    """
    if not window_results:
        return []

    # Infer hop
    if len(window_results) >= 2:
        hop = window_results[1].window.original_start - window_results[0].window.original_start
    else:
        hop = window_results[0].window.original_end - window_results[0].window.original_start

    if hop <= 0:
        hop = 1.0

    # Infer call duration from last window if not provided
    if call_duration_sec is None:
        call_duration_sec = window_results[-1].window.original_end

    # Build base interval boundaries: [0, hop, 2*hop, ..., call_duration_sec]
    boundaries = []
    t = 0.0
    while t < call_duration_sec:
        boundaries.append(round(t, 6))
        t += hop
    boundaries.append(round(call_duration_sec, 6))

    # Compute activity for each base interval
    base_intervals = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]

        # Find which overlapping window(s) cover this base interval
        covering_window_ids = []
        for wr in window_results:
            w_start = wr.window.original_start
            w_end = wr.window.original_end
            if w_start < end and w_end > start:
                covering_window_ids.append(wr.window.window_id)

        # Recompute activity from waveforms for this exact interval
        if agent_waveform is None or customer_waveform is None:
            raise ValueError(
                "Accurate activity-duration computation requires channel waveforms. "
                "Pass agent_waveform and customer_waveform to build_activity_states()."
            )
        a_active, a_ratio = _compute_base_interval_activity(
            agent_waveform, start, end, sample_rate, rms_threshold_db, active_ratio_threshold
        )
        c_active, c_ratio = _compute_base_interval_activity(
            customer_waveform, start, end, sample_rate, rms_threshold_db, active_ratio_threshold
        )

        # Determine state
        if a_active and c_active:
            state = "both_active_interval"
        elif a_active and not c_active:
            state = "agent_active_only"
        elif not a_active and c_active:
            state = "customer_active_only"
        elif not a_active and not c_active:
            state = "neither_active"
        else:
            state = "uncertain_activity"

        base_intervals.append(ActivityStateEntry(
            original_start=start,
            original_end=end,
            state=state,
            agent_active=a_active,
            customer_active=c_active,
            agent_active_ratio=round(a_ratio, 4),
            customer_active_ratio=round(c_ratio, 4),
            source_window_ids=covering_window_ids,
        ))

    return base_intervals


def _compute_base_interval_activity(
    waveform: np.ndarray,
    start_sec: float,
    end_sec: float,
    sample_rate: int,
    rms_threshold_db: float,
    active_ratio_threshold: float,
) -> tuple:
    """
    Compute speech activity for a single non-overlapping base interval.

    Returns: (speech_active: bool, active_ratio: float)
    """
    import librosa

    start_sample = int(round(start_sec * sample_rate))
    end_sample = int(round(end_sec * sample_rate))
    segment = waveform[start_sample:end_sample]

    if len(segment) == 0:
        return False, 0.0

    if not np.all(np.isfinite(segment)):
        return False, 0.0

    rms = librosa.feature.rms(
        y=segment, frame_length=2048, hop_length=512
    )[0]

    if len(rms) == 0:
        return False, 0.0

    segment_rms = float(np.sqrt(np.mean(segment ** 2)))
    if segment_rms < 1e-8:
        return False, 0.0

    threshold_linear = 10 ** (rms_threshold_db / 20) * np.max(rms)
    active_frames = int(np.sum(rms > threshold_linear))
    active_ratio = active_frames / len(rms)

    return active_ratio >= active_ratio_threshold, active_ratio


def merge_activity_states(
    base_intervals: List[ActivityStateEntry],
) -> List[ActivityInterval]:
    """
    Merge consecutive base intervals with the same state into non-overlapping intervals.

    Returns a list of ActivityInterval covering the full call duration.
    """
    if not base_intervals:
        return []

    intervals = []
    current = base_intervals[0]
    current_start = current.original_start
    current_end = current.original_end
    current_source_ids = list(current.source_window_ids)

    for i in range(1, len(base_intervals)):
        bi = base_intervals[i]
        if bi.state == current.state:
            # Extend current interval
            current_end = bi.original_end
            current_source_ids.extend(bi.source_window_ids)
        else:
            # Save current interval
            intervals.append(ActivityInterval(
                start_time=current_start,
                end_time=current_end,
                duration_sec=round(current_end - current_start, 6),
                state=current.state,
                agent_active=current.agent_active,
                customer_active=current.customer_active,
                source_window_ids=current_source_ids,
            ))
            # Start new interval
            current = bi
            current_start = bi.original_start
            current_end = bi.original_end
            current_source_ids = list(bi.source_window_ids)

    # Save last interval
    intervals.append(ActivityInterval(
        start_time=current_start,
        end_time=current_end,
        duration_sec=round(current_end - current_start, 6),
        state=current.state,
        agent_active=current.agent_active,
        customer_active=current.customer_active,
        source_window_ids=current_source_ids,
    ))

    return intervals


def compute_inactivity_metadata(
    intervals: List[ActivityInterval],
) -> Dict:
    """
    Compute inactivity statistics from activity intervals.

    Does NOT detect Hold events, waiting states, or generate violations.
    Only computes raw inactivity duration statistics.

    Note: Hold detection, waiting-state detection, and policy violation
    logic are NOT implemented. These are future-phase features.
    """
    if not intervals:
        return {
            "longest_agent_inactive_interval": 0.0,
            "longest_customer_inactive_interval": 0.0,
            "longest_neither_active_interval": 0.0,
            "number_of_agent_inactive_intervals": 0,
            "number_of_customer_inactive_intervals": 0,
            "number_of_neither_active_intervals": 0,
        }

    agent_inactive_durations = []
    customer_inactive_durations = []
    neither_durations = []

    for iv in intervals:
        if not iv.agent_active:
            agent_inactive_durations.append(iv.duration_sec)
        if not iv.customer_active:
            customer_inactive_durations.append(iv.duration_sec)
        if iv.state == "neither_active":
            neither_durations.append(iv.duration_sec)

    return {
        "longest_agent_inactive_interval": max(agent_inactive_durations) if agent_inactive_durations else 0.0,
        "longest_customer_inactive_interval": max(customer_inactive_durations) if customer_inactive_durations else 0.0,
        "longest_neither_active_interval": max(neither_durations) if neither_durations else 0.0,
        "number_of_agent_inactive_intervals": len(agent_inactive_durations),
        "number_of_customer_inactive_intervals": len(customer_inactive_durations),
        "number_of_neither_active_intervals": len(neither_durations),
    }


# ══════════════════════════════════════════════════════════════════════
# CALL SUMMARY
# ══════════════════════════════════════════════════════════════════════
def compute_call_summary(
    processing_result,  # StereoProcessingResult
    agent_summary: SpeakerSummary,
    customer_summary: SpeakerSummary,
    activity_intervals: List[ActivityInterval],
    inactivity_metadata: Dict,
) -> CallSummary:
    """Compute call-level summary combining both speakers and activity analysis."""
    # Activity state counts from intervals
    state_counts = Counter(iv.state for iv in activity_intervals)
    state_durations = {}
    for iv in activity_intervals:
        state_durations[iv.state] = state_durations.get(iv.state, 0.0) + iv.duration_sec

    return CallSummary(
        call_duration_sec=processing_result.duration_sec,
        total_shared_windows=len(processing_result.shared_windows),
        agent_summary=agent_summary,
        customer_summary=customer_summary,
        agent_active_only_count=state_counts.get("agent_active_only", 0),
        customer_active_only_count=state_counts.get("customer_active_only", 0),
        both_active_count=state_counts.get("both_active_interval", 0),
        neither_active_count=state_counts.get("neither_active", 0),
        uncertain_count=state_counts.get("uncertain_activity", 0),
        agent_active_only_duration_sec=round(state_durations.get("agent_active_only", 0.0), 4),
        customer_active_only_duration_sec=round(state_durations.get("customer_active_only", 0.0), 4),
        both_active_duration_sec=round(state_durations.get("both_active_interval", 0.0), 4),
        neither_active_duration_sec=round(state_durations.get("neither_active", 0.0), 4),
        longest_agent_inactive_interval=inactivity_metadata["longest_agent_inactive_interval"],
        longest_customer_inactive_interval=inactivity_metadata["longest_customer_inactive_interval"],
        longest_neither_active_interval=inactivity_metadata["longest_neither_active_interval"],
        number_of_agent_inactive_intervals=inactivity_metadata["number_of_agent_inactive_intervals"],
        number_of_customer_inactive_intervals=inactivity_metadata["number_of_customer_inactive_intervals"],
        number_of_neither_active_intervals=inactivity_metadata["number_of_neither_active_intervals"],
        activity_state_intervals=activity_intervals,
        model_version=processing_result.model_version,
        role_assignment_method=processing_result.role_assignment_method,
        processing_timestamp=datetime.now(timezone.utc).isoformat(),
    )
