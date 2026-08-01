"""
Shared timeline windowing and speech-activity analysis.

Creates aligned analysis windows for both Agent and Customer channels.
Measures speech activity without modifying the waveform.
"""

import numpy as np
from typing import List, Optional

from .schemas import (
    SharedWindow,
    WindowConfig,
    SpeechActivityConfig,
    SpeechActivityResult,
)


def create_shared_windows(
    duration_sec: float,
    window_duration_sec: float,
    hop_duration_sec: float,
    model_sr: int = 16000,
    original_sr: Optional[int] = None,
) -> List[SharedWindow]:
    """
    Create shared analysis windows aligned to the original timeline.

    Windows are defined in seconds. Both Agent and Customer use the
    exact same window boundaries. No samples are deleted or shifted.

    Args:
        duration_sec: Total call duration in seconds.
        window_duration_sec: Duration of each window in seconds.
        hop_duration_sec: Hop (stride) between windows in seconds.
        model_sr: Model sample rate (default 16000).
        original_sr: Original sample rate (for original sample indices).

    Returns:
        List of SharedWindow, monotonically increasing.

    Raises:
        ValueError: If any parameter is invalid.
    """
    # ── Validate inputs ────────────────────────────────────────────────
    if duration_sec <= 0:
        raise ValueError(f"duration_sec must be positive, got {duration_sec}")
    if window_duration_sec <= 0:
        raise ValueError(f"window_duration_sec must be positive, got {window_duration_sec}")
    if hop_duration_sec <= 0:
        raise ValueError(f"hop_duration_sec must be positive, got {hop_duration_sec}")
    if model_sr <= 0:
        raise ValueError(f"model_sr must be positive, got {model_sr}")
    if original_sr is not None and original_sr <= 0:
        raise ValueError(f"original_sr must be positive, got {original_sr}")

    windows = []
    window_id = 0
    start_sec = 0.0

    while start_sec < duration_sec:
        end_sec = min(start_sec + window_duration_sec, duration_sec)

        # If this window's start is beyond the last window's start AND the end
        # doesn't extend coverage, this is a duplicate — stop
        if windows and start_sec > windows[-1].original_start:
            if end_sec <= windows[-1].original_end:
                break

        # Model sample indices (16 kHz)
        model_start = round(start_sec * model_sr)
        model_end = round(end_sec * model_sr)

        # Original sample indices (if original_sr provided)
        orig_start = round(start_sec * original_sr) if original_sr else None
        orig_end = round(end_sec * original_sr) if original_sr else None

        windows.append(SharedWindow(
            window_id=window_id,
            original_start=round(start_sec, 6),
            original_end=round(end_sec, 6),
            model_start_sample=model_start,
            model_end_sample=model_end,
            original_start_sample=orig_start,
            original_end_sample=orig_end,
        ))

        window_id += 1
        start_sec += hop_duration_sec

    return windows


def analyze_speech_activity(
    waveform: np.ndarray,
    windows: List[SharedWindow],
    sample_rate: int,
    config: SpeechActivityConfig,
) -> List[SpeechActivityResult]:
    """
    Measure speech activity for one channel across shared windows.

    Does not modify the waveform. Does not shift timestamps.
    This is RMS-based prototype activity detection, not speaker diarization.

    Args:
        waveform: 1D float32 waveform at the given sample_rate.
        windows: Shared window list (same windows for both channels).
        sample_rate: Sample rate of the waveform.
        config: Speech-activity configuration.

    Returns:
        List of SpeechActivityResult, one per window, in window order.
    """
    import librosa

    results = []

    for window in windows:
        start_sample = window.model_start_sample
        end_sample = window.model_end_sample

        # Extract segment without modifying original
        segment = waveform[start_sample:end_sample]

        # ── Empty segment ──────────────────────────────────────────────
        if len(segment) == 0:
            results.append(SpeechActivityResult(
                window_id=window.window_id,
                speech_active=False,
                active_ratio=0.0,
                rms=0.0,
                inactive_reason="empty_segment",
            ))
            continue

        # ── Non-finite audio ───────────────────────────────────────────
        if not np.all(np.isfinite(segment)):
            results.append(SpeechActivityResult(
                window_id=window.window_id,
                speech_active=False,
                active_ratio=0.0,
                rms=0.0,
                inactive_reason="non_finite_audio",
            ))
            continue

        # ── Compute RMS per frame ──────────────────────────────────────
        rms = librosa.feature.rms(
            y=segment,
            frame_length=config.frame_length,
            hop_length=config.hop_length,
        )[0]

        if len(rms) == 0:
            results.append(SpeechActivityResult(
                window_id=window.window_id,
                speech_active=False,
                active_ratio=0.0,
                rms=0.0,
                inactive_reason="insufficient_duration",
            ))
            continue

        segment_rms = float(np.sqrt(np.mean(segment ** 2)))

        # ── Silence check ──────────────────────────────────────────────
        if segment_rms < 1e-8:
            results.append(SpeechActivityResult(
                window_id=window.window_id,
                speech_active=False,
                active_ratio=0.0,
                rms=segment_rms,
                inactive_reason="silence",
            ))
            continue

        # ── Active frame detection ─────────────────────────────────────
        threshold_linear = 10 ** (config.rms_threshold_db / 20) * np.max(rms)
        active_frames = np.sum(rms > threshold_linear)
        active_ratio = float(active_frames / len(rms))

        speech_active = active_ratio >= config.active_ratio_threshold
        inactive_reason = None if speech_active else "insufficient_active_ratio"

        results.append(SpeechActivityResult(
            window_id=window.window_id,
            speech_active=speech_active,
            active_ratio=round(active_ratio, 4),
            rms=round(segment_rms, 6),
            inactive_reason=inactive_reason,
        ))

    return results
