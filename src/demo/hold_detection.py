"""
Hold candidate detection from activity intervals and audio features.

Detects potential Hold events using:
- Prolonged silence evidence
- Tonal / music-like audio evidence (synthetic prototype features)
- Agent return detection

This module does NOT:
- Detect or classify real hold music (no validated classifier)
- Detect recorded messages (no validated classifier)
- Generate general emotion alerts
- Make disciplinary conclusions

It produces HoldCandidate objects with evidence and confidence levels.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from .schemas import (
    ActivityInterval,
    HoldDetectionConfig,
    HoldCandidate,
    HoldDetectionResult,
)

logger = logging.getLogger(__name__)


def detect_hold_candidates(
    intervals: List[ActivityInterval],
    agent_waveform: np.ndarray,
    customer_waveform: np.ndarray,
    sample_rate: int,
    call_duration_sec: float,
    config: Optional[HoldDetectionConfig] = None,
) -> HoldDetectionResult:
    """
    Detect Hold candidates from non-overlapping activity intervals.

    Uses exact interval boundaries (not overlapping SER windows).
    Produces evidence-based candidates with confidence levels.

    Note: Hold music detection and recorded-message detection are NOT
    implemented as validated classifiers. The features extracted are
    prototype evidence indicators only.
    """
    if config is None:
        config = HoldDetectionConfig()

    candidates = []
    candidate_id_counter = 0

    # ── Pass 1: Identify candidate intervals ───────────────────────────
    raw_candidates = []

    for iv in intervals:
        # Skip short pauses
        if iv.duration_sec <= config.maximum_short_pause_duration_sec:
            continue

        # Skip intervals shorter than minimum candidate duration
        if iv.duration_sec < config.minimum_hold_candidate_duration_sec:
            continue

        # Gate: Hold candidates require Agent inactivity.
        # An active Agent means conversation is in progress — not a Hold.
        if iv.agent_active:
            continue

        # Analyze evidence for this interval
        evidence, confidence, customer_state, notes = _analyze_interval_evidence(
            iv, agent_waveform, customer_waveform, sample_rate, config
        )

        if confidence < config.minimum_confidence:
            continue

        # Determine classification based on evidence strength
        classification = _classify_candidate(evidence, confidence, config)

        raw_candidates.append({
            "interval": iv,
            "evidence": evidence,
            "confidence": confidence,
            "customer_state": customer_state,
            "classification": classification,
            "notes": notes,
        })

    # ── Pass 2: Check for Agent return to close candidates ─────────────
    for rc in raw_candidates:
        iv = rc["interval"]

        # Check if Agent returns after this interval
        agent_returned = _check_agent_return(
            intervals, iv.end_time, call_duration_sec,
            config.agent_return_confirmation_duration_sec
        )

        # Check if candidate reaches the call boundary
        reaches_end = abs(iv.end_time - call_duration_sec) < 0.01

        # Determine if candidate should be closed
        should_close = True
        if config.require_agent_return_to_close_hold and not agent_returned:
            should_close = False

        if not should_close:
            continue

        candidate_id_counter += 1
        candidates.append(HoldCandidate(
            candidate_id=f"hold_{candidate_id_counter:03d}",
            start_time=iv.start_time,
            end_time=iv.end_time,
            duration_sec=iv.duration_sec,
            classification=rc["classification"],
            evidence_types=rc["evidence"],
            agent_inactive=not iv.agent_active,
            customer_state=rc["customer_state"],
            confidence=rc["confidence"],
            detection_notes=rc["notes"],
            agent_return_detected=agent_returned,
            reaches_call_end=reaches_end and not agent_returned,
            source_interval_ids=iv.source_window_ids,
        ))

    # ── Pass 3: Merge adjacent candidates ──────────────────────────────
    merged = _merge_candidates(candidates, config.merge_gap_between_candidates_sec)

    # ── Build result ───────────────────────────────────────────────────
    uncertain_count = sum(1 for c in merged if c.classification == "uncertain_hold_candidate")
    likely_count = sum(1 for c in merged if c.classification == "likely_hold_candidate")

    return HoldDetectionResult(
        candidates=merged,
        call_duration_sec=call_duration_sec,
        detection_config=config,
        total_candidates=len(merged),
        uncertain_count=uncertain_count,
        likely_count=likely_count,
    )


def _analyze_interval_evidence(
    iv: ActivityInterval,
    agent_waveform: np.ndarray,
    customer_waveform: np.ndarray,
    sample_rate: int,
    config: HoldDetectionConfig,
) -> Tuple[list, float, str, str]:
    """
    Analyze audio evidence within an activity interval.

    Evidence aggregation rules:
    - prolonged_silence: both inactive → +0.3
    - music_like_audio: tonal non-speech customer → +0.4
    - recorded_message_candidate: repetitive tonal → +0.35
    - agent_inactive_customer_speaking: agent inactive + customer speech → +0.1
    - agent_inactive_uncertain_audio: agent inactive + uncertain → +0.15

    Classification thresholds (applied AFTER aggregation):
    - music_like or recorded evidence + confidence >= 0.5 → likely
    - otherwise → uncertain

    Silence-only NEVER reaches likely (max possible = 0.3).
    Tonal + agent_inactive_uncertain reaches 0.55 → likely.
    """
    start_sample = int(round(iv.start_time * sample_rate))
    end_sample = int(round(iv.end_time * sample_rate))

    agent_segment = agent_waveform[start_sample:end_sample]
    customer_segment = customer_waveform[start_sample:end_sample]

    evidence = []
    notes_parts = []
    confidence = 0.0

    # ── Agent inactivity ───────────────────────────────────────────────
    agent_inactive = not iv.agent_active

    # ── Customer state analysis ────────────────────────────────────────
    customer_state = _classify_customer_state(
        customer_segment, sample_rate, config
    )

    # ── Silence evidence ───────────────────────────────────────────────
    both_inactive = (not iv.agent_active) and (not iv.customer_active)
    if both_inactive and iv.duration_sec >= config.minimum_hold_candidate_duration_sec:
        evidence.append("prolonged_silence")
        confidence += 0.3
        notes_parts.append("Both channels inactive for {:.1f}s".format(iv.duration_sec))

    # ── Tonal / music-like evidence ────────────────────────────────────
    if customer_state == "music_like":
        evidence.append("music_like_audio")
        confidence += 0.4
        notes_parts.append("Customer channel shows tonal/music-like characteristics")
    elif customer_state == "recorded_message_candidate":
        evidence.append("recorded_message_candidate")
        confidence += 0.35
        notes_parts.append("Customer channel shows repetitive tonal patterns")

    # ── Agent inactive + Customer active (speech) ──────────────────────
    if agent_inactive and customer_state == "speech_like":
        evidence.append("agent_inactive_customer_speaking")
        confidence += 0.1
        notes_parts.append("Agent inactive while Customer channel has speech-like activity")

    # ── Agent inactive + uncertain customer ─────────────────────────────
    if agent_inactive and customer_state == "uncertain":
        evidence.append("agent_inactive_uncertain_audio")
        confidence += 0.15
        notes_parts.append("Agent inactive with uncertain Customer channel activity")

    # ── Agent-inactive bonus for tonal evidence ────────────────────────
    # When Agent is inactive AND customer shows tonal non-speech evidence,
    # the combination is stronger than either alone. This bonus ensures
    # tonal+agent_inactive reaches the likely threshold (0.5).
    if agent_inactive and ("music_like_audio" in evidence or "recorded_message_candidate" in evidence):
        confidence += 0.15
        notes_parts.append("Agent inactivity combined with tonal evidence strengthens Hold signal")

    # ── Confidence capping ─────────────────────────────────────────────
    confidence = min(confidence, 1.0)

    return evidence, confidence, customer_state, " | ".join(notes_parts) if notes_parts else "No significant Hold evidence"


def _classify_candidate(evidence: list, confidence: float, config: HoldDetectionConfig) -> str:
    """Classify a Hold candidate based on evidence strength.

    Rules:
    - music_like or recorded evidence with confidence >= 0.5 → likely
    - silence-only (no tonal/recording evidence) → uncertain (max 0.3)
    - other combinations → uncertain
    """
    has_music = "music_like_audio" in evidence
    has_recorded = "recorded_message_candidate" in evidence

    if has_music or has_recorded:
        if confidence >= 0.5:
            return "likely_hold_candidate"
        return "uncertain_hold_candidate"

    # Silence-only or agent_inactive evidence — never likely
    return "uncertain_hold_candidate"


def _classify_customer_state(
    waveform: np.ndarray,
    sample_rate: int,
    config: HoldDetectionConfig,
) -> str:
    """
    Classify Customer channel state within a segment.

    Returns one of: "inactive", "speech_like", "music_like",
    "recorded_message_candidate", "uncertain"

    Tonal detection uses LOW spectral flatness (concentrated energy)
    combined with HIGH tonal consistency and frequency stability.
    """
    if len(waveform) == 0 or not np.all(np.isfinite(waveform)):
        return "inactive"

    # Compute features
    features = _compute_audio_features(waveform, sample_rate)

    if features["rms"] < 1e-6:
        return "inactive"

    # ── Tonal / music-like detection ───────────────────────────────────
    # Tonal signals have LOW spectral flatness (energy concentrated in
    # narrow frequency bands), HIGH tonal consistency (stable harmonic
    # structure), and sufficient spectral bandwidth (pure single-frequency
    # sine waves are excluded — they have near-zero bandwidth).
    is_tonal = (
        features["spectral_flatness"] < config.maximum_tonal_spectral_flatness
        and features["tonal_consistency"] > config.minimum_tonal_consistency
        and features["dominant_freq_stability"] > config.minimum_dominant_freq_stability
        and features["spectral_bandwidth"] > config.minimum_tonal_spectral_bandwidth
    )

    if is_tonal:
        return "music_like"

    # ── Recorded message candidate ─────────────────────────────────────
    # Repetitive patterns with moderate flatness AND sufficient bandwidth.
    # Pure single-frequency sine waves are excluded (near-zero bandwidth).
    if (features["repetition_score"] > config.minimum_recorded_message_candidate_ratio
            and features["spectral_flatness"] < 0.3
            and features["spectral_bandwidth"] > config.minimum_tonal_spectral_bandwidth):
        return "recorded_message_candidate"

    # ── Speech-like ────────────────────────────────────────────────────
    # High ZCR variation and moderate energy → likely speech
    if features["rms"] > 0.01 and features["zero_crossing_rate"] > 0.05:
        return "speech_like"

    return "uncertain"


def _normalized_autocorrelation_at_lag(
    signal: np.ndarray,
    lag: int,
) -> float:
    """Return normalized autocorrelation for one positive lag in linear time."""
    values = np.asarray(signal)
    if (
        values.ndim != 1
        or values.size == 0
        or not isinstance(lag, (int, np.integer))
        or lag <= 0
        or values.size <= lag
    ):
        return 0.0

    with np.errstate(invalid="ignore", over="ignore"):
        energy = float(np.dot(values, values))
        if not np.isfinite(energy) or energy <= 0.0:
            return 0.0

        lag_product = float(np.dot(values[:-lag], values[lag:]))
        score = lag_product / (energy + 1e-8)

    if not np.isfinite(score):
        return 0.0
    return float(np.clip(score, -1.0, 1.0))


def _compute_audio_features(waveform: np.ndarray, sample_rate: int) -> dict:
    """
    Compute lightweight audio features for Hold evidence.

    Uses only existing dependencies (numpy, librosa).
    These are prototype features — not a validated music classifier.

    Analysis is limited to a 5-second subsample for performance.
    """
    import librosa

    # Subsample for feature extraction (max 5 seconds)
    max_samples = min(len(waveform), 5 * sample_rate)
    analysis = waveform[:max_samples]

    # RMS energy (computed on full segment for accuracy)
    rms_val = float(np.sqrt(np.mean(waveform ** 2)))

    # Zero-crossing rate (on subsample)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(analysis)[0]))

    # Spectral features (on subsample)
    spec = np.abs(librosa.stft(analysis))
    spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(S=spec)))
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(S=spec, sr=sample_rate)))
    spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(S=spec, sr=sample_rate)))

    # Tonal consistency (harmonic ratio on subsample)
    harmonic = librosa.effects.harmonic(analysis)
    harmonic_ratio = float(np.mean(np.abs(harmonic)) / (np.mean(np.abs(analysis)) + 1e-8))

    # Dominant frequency stability across frames
    dominant_freq_stability = _compute_dominant_freq_stability(analysis, sample_rate)

    # Repetition score at the existing one-second lag.
    repetition_score = _normalized_autocorrelation_at_lag(
        analysis,
        sample_rate,
    )

    return {
        "rms": rms_val,
        "zero_crossing_rate": zcr,
        "spectral_flatness": spectral_flatness,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "tonal_consistency": harmonic_ratio,
        "dominant_freq_stability": dominant_freq_stability,
        "repetition_score": repetition_score,
    }


def _compute_dominant_freq_stability(waveform: np.ndarray, sample_rate: int) -> float:
    """Compute stability of dominant frequency across short frames.

    A tonal signal has a stable dominant frequency (coefficient of
    variation < 0.1). Broadband noise has no stable dominant frequency.

    Returns value in [0, 1] where 1 = perfectly stable.
    """
    import librosa

    frame_length = min(2048, len(waveform))
    hop_length = min(512, frame_length // 2)
    if hop_length < 1:
        hop_length = 1

    # Compute STFT magnitude
    S = np.abs(librosa.stft(waveform, n_fft=frame_length, hop_length=hop_length))

    if S.shape[1] < 2:
        return 0.0

    # Find dominant frequency index per frame
    dominant_indices = np.argmax(S, axis=0)

    # Convert to frequency
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=frame_length)
    dominant_freqs = freqs[dominant_indices]

    # Coefficient of variation — low means stable
    mean_freq = np.mean(dominant_freqs)
    std_freq = np.std(dominant_freqs)

    if mean_freq < 1.0:
        return 0.0

    cv = std_freq / mean_freq

    # Map CV to stability: cv=0 → stability=1, cv>=0.3 → stability=0
    stability = max(0.0, 1.0 - cv / 0.3)
    return float(stability)


def _check_agent_return(
    intervals: List[ActivityInterval],
    after_time: float,
    call_duration: float,
    confirmation_duration: float,
) -> bool:
    """Check if Agent returns and remains active for confirmation_duration."""
    for iv in intervals:
        if iv.start_time < after_time:
            continue
        if iv.agent_active and iv.duration_sec >= confirmation_duration:
            return True
        if iv.start_time >= call_duration:
            break
    return False


def _check_normal_resumption(
    intervals: List[ActivityInterval],
    after_time: float,
    call_duration: float,
) -> bool:
    """Check if normal conversation resumes after the interval."""
    for iv in intervals:
        if iv.start_time < after_time:
            continue
        if iv.state in ("agent_active_only", "both_active_interval"):
            return True
        if iv.start_time >= call_duration:
            break
    return False


def _merge_candidates(
    candidates: List[HoldCandidate],
    merge_gap: float,
) -> List[HoldCandidate]:
    """Merge adjacent Hold candidates when gap is below threshold.

    Does NOT merge across intervals where agent_return_detected is True.
    """
    if not candidates:
        return []

    merged = [candidates[0]]

    for c in candidates[1:]:
        prev = merged[-1]
        gap = c.start_time - prev.end_time

        if gap <= merge_gap and not prev.agent_return_detected:
            # Merge: extend previous candidate
            merged[-1] = HoldCandidate(
                candidate_id=prev.candidate_id,
                start_time=prev.start_time,
                end_time=c.end_time,
                duration_sec=round(c.end_time - prev.start_time, 6),
                classification=_merge_classification(prev.classification, c.classification),
                evidence_types=list(set(prev.evidence_types + c.evidence_types)),
                agent_inactive=prev.agent_inactive and c.agent_inactive,
                customer_state=c.customer_state,
                confidence=max(prev.confidence, c.confidence),
                detection_notes=prev.detection_notes + " | Merged with " + c.candidate_id,
                agent_return_detected=c.agent_return_detected,
                reaches_call_end=c.reaches_call_end,
                source_interval_ids=prev.source_interval_ids + c.source_interval_ids,
            )
        else:
            merged.append(c)

    return merged


def _merge_classification(c1: str, c2: str) -> str:
    """Merge two classification levels (take the higher confidence level)."""
    priority = {
        "uncertain_hold_candidate": 0,
        "likely_hold_candidate": 1,
        "confirmed_by_metadata_or_manual_input": 2,
    }
    if priority.get(c2, 0) > priority.get(c1, 0):
        return c2
    return c1
