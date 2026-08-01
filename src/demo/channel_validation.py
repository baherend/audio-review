"""
Stereo channel validation.

Validates stereo audio input quality and returns structured results.
All thresholds come from StereoValidationConfig — nothing is hardcoded.
"""

import numpy as np
from typing import Tuple

from .schemas import (
    StereoValidationConfig,
    ValidationResult,
    ValidationError,
    ValidationWarning,
    ValidationMetrics,
)


def validate_stereo(
    agent_waveform: np.ndarray,
    customer_waveform: np.ndarray,
    sample_rate: int,
    config: StereoValidationConfig,
) -> ValidationResult:
    """
    Validate stereo audio quality for the demo.

    Args:
        agent_waveform: Left channel (1D float32).
        customer_waveform: Right channel (1D float32).
        sample_rate: Sample rate of the audio.
        config: Validation thresholds.

    Returns:
        ValidationResult with is_valid, errors, warnings, metrics.
    """
    errors = []
    warnings = []
    metrics = ValidationMetrics(
        channel_count=2,
        samples_per_channel=len(agent_waveform),
        sample_rate=sample_rate,
        duration_sec=round(len(agent_waveform) / sample_rate, 4),
    )

    # ── Reject: completely empty audio ─────────────────────────────────
    if len(agent_waveform) == 0 or len(customer_waveform) == 0:
        errors.append(ValidationError(
            code="EMPTY_AUDIO",
            message="Audio contains no samples.",
        ))
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings, metrics=metrics)

    # ── Reject: non-finite samples ─────────────────────────────────────
    agent_finite = np.all(np.isfinite(agent_waveform))
    customer_finite = np.all(np.isfinite(customer_waveform))
    if not agent_finite or not customer_finite:
        errors.append(ValidationError(
            code="NON_FINITE_SAMPLES",
            message="Audio contains non-finite values (NaN or Inf).",
        ))
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings, metrics=metrics)

    # ── Internal assertion: equal channel lengths ──────────────────────
    assert len(agent_waveform) == len(customer_waveform), (
        f"Channel length mismatch: agent={len(agent_waveform)}, "
        f"customer={len(customer_waveform)}"
    )

    # ── Compute metrics ────────────────────────────────────────────────
    metrics.agent_rms = float(np.sqrt(np.mean(agent_waveform ** 2)))
    metrics.customer_rms = float(np.sqrt(np.mean(customer_waveform ** 2)))

    # Correlation (Pearson)
    if metrics.agent_rms > 0 and metrics.customer_rms > 0:
        correlation = np.corrcoef(agent_waveform, customer_waveform)[0, 1]
        metrics.channel_correlation = float(correlation)
    else:
        metrics.channel_correlation = 0.0

    # Clipping ratio
    metrics.agent_clipping_ratio = float(
        np.mean(np.abs(agent_waveform) >= config.clipping_amplitude)
    )
    metrics.customer_clipping_ratio = float(
        np.mean(np.abs(customer_waveform) >= config.clipping_amplitude)
    )

    # ── Reject: completely silent two-channel file ─────────────────────
    if metrics.agent_rms < config.silent_rms_threshold and \
       metrics.customer_rms < config.silent_rms_threshold:
        errors.append(ValidationError(
            code="COMPLETELY_SILENT",
            message="Both channels are silent (RMS below threshold).",
        ))
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings, metrics=metrics)

    # ── Warn: identical channels (sample-level comparison) ─────────────
    mean_abs_diff = float(np.mean(np.abs(agent_waveform - customer_waveform)))
    if mean_abs_diff <= config.identical_sample_tolerance:
        warnings.append(ValidationWarning(
            code="IDENTICAL_CHANNELS",
            message=(
                f"Channels appear identical (mean abs sample diff={mean_abs_diff:.8f}). "
                "Speaker separation may not be meaningful."
            ),
        ))

    # ── Warn: highly correlated channels (correlation-based) ──────────
    elif metrics.channel_correlation >= config.high_correlation_threshold:
        warnings.append(ValidationWarning(
            code="HIGH_CORRELATION",
            message=(
                f"Channels are highly correlated (correlation={metrics.channel_correlation:.4f}). "
                "May contain mixed audio."
            ),
        ))

    # ── Warn: silent Agent channel ─────────────────────────────────────
    if metrics.agent_rms < config.silent_rms_threshold:
        warnings.append(ValidationWarning(
            code="SILENT_AGENT_CHANNEL",
            message=(
                f"Agent channel (left) appears silent (RMS={metrics.agent_rms:.6f}). "
                "No speech will be detected for the Agent."
            ),
        ))

    # ── Warn: silent Customer channel ──────────────────────────────────
    if metrics.customer_rms < config.silent_rms_threshold:
        warnings.append(ValidationWarning(
            code="SILENT_CUSTOMER_CHANNEL",
            message=(
                f"Customer channel (right) appears silent (RMS={metrics.customer_rms:.6f}). "
                "No speech will be detected for the Customer."
            ),
        ))

    # ── Warn: low overall energy ───────────────────────────────────────
    overall_rms = (metrics.agent_rms + metrics.customer_rms) / 2
    if overall_rms < config.low_energy_threshold:
        warnings.append(ValidationWarning(
            code="LOW_ENERGY",
            message=(
                f"Audio energy is very low (overall RMS={overall_rms:.6f}). "
                "Results may be unreliable."
            ),
        ))

    # ── Warn: clipping ─────────────────────────────────────────────────
    if metrics.agent_clipping_ratio > config.clipping_ratio_threshold:
        warnings.append(ValidationWarning(
            code="AGENT_CLIPPING",
            message=(
                f"Agent channel has {metrics.agent_clipping_ratio:.1%} clipped samples. "
                "Audio may be distorted."
            ),
        ))

    if metrics.customer_clipping_ratio > config.clipping_ratio_threshold:
        warnings.append(ValidationWarning(
            code="CUSTOMER_CLIPPING",
            message=(
                f"Customer channel has {metrics.customer_clipping_ratio:.1%} clipped samples. "
                "Audio may be distorted."
            ),
        ))

    # ── Warn: very short duration ──────────────────────────────────────
    if metrics.duration_sec < config.minimum_duration_sec:
        warnings.append(ValidationWarning(
            code="VERY_SHORT_DURATION",
            message=(
                f"Audio is very short ({metrics.duration_sec:.2f}s). "
                f"Minimum recommended: {config.minimum_duration_sec:.1f}s."
            ),
        ))

    return ValidationResult(
        is_valid=True,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
    )
