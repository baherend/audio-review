"""
Stereo audio loading and channel separation.

Loads stereo audio, preserves the original waveform, creates 16 kHz model copies,
and runs validation. Never trims, never deletes samples, never compresses time.
"""

import io
from pathlib import Path
from typing import BinaryIO, Optional, Union

import numpy as np

from .schemas import (
    OriginalStereoAudio,
    ModelStereoAudio,
    StereoLoadResult,
    StereoValidationConfig,
)
from .channel_validation import validate_stereo


def load_stereo_audio(
    source: Union[str, Path, BinaryIO],
    filename: Optional[str] = None,
    config: Optional[StereoValidationConfig] = None,
) -> StereoLoadResult:
    """
    Load a stereo audio file and prepare it for the demo.

    Steps:
        1. Decode audio without forcing mono
        2. Preserve original sample rate and waveform
        3. Validate stereo input quality
        4. Resample each channel to 16 kHz for model inference
        5. Return typed schemas

    Args:
        source: File path (str/Path) or file-like object (BinaryIO).
        filename: Original filename (for metadata). Inferred from path if None.
        config: Validation thresholds. Uses defaults if None.

    Returns:
        StereoLoadResult with original audio, model audio, validation, and role info.

    Raises:
        ValueError: If input is mono, has >2 channels, or cannot be decoded.
        FileNotFoundError: If file path does not exist.
    """
    import librosa

    if config is None:
        config = StereoValidationConfig()

    # ── Resolve source ─────────────────────────────────────────────────
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        if filename is None:
            filename = path.name
        file_format = path.suffix.lstrip(".").lower() or "unknown"
    elif hasattr(source, "read"):
        # BinaryIO — read all bytes into a buffer
        source.seek(0)
        audio_bytes = source.read()
        buffer = io.BytesIO(audio_bytes)
        if filename is None:
            filename = "uploaded_audio"
        file_format = "unknown"
        path = None
    else:
        raise TypeError(f"source must be str, Path, or BinaryIO, got {type(source).__name__}")

    # ── Decode audio ───────────────────────────────────────────────────
    try:
        if path is not None:
            waveform, original_sr = librosa.load(
                str(path), sr=None, mono=False
            )
        else:
            # For BinaryIO, librosa needs a file path or file-like with proper handling
            # Write to a temporary in-memory buffer for librosa
            buffer.seek(0)
            waveform, original_sr = librosa.load(
                buffer, sr=None, mono=False
            )
    except Exception as e:
        raise ValueError(f"Could not decode audio: {e}") from e

    # ── Normalize orientation to (channels, samples) ───────────────────
    if waveform.ndim == 1:
        # Mono — rejected
        raise ValueError(
            "Stereo audio required for two-speaker demo. "
            "Received mono audio (1 channel)."
        )
    elif waveform.ndim == 2:
        # Check orientation: librosa returns (channels, samples) when mono=False
        # Verify which axis is channels
        if waveform.shape[0] > waveform.shape[1]:
            # Transpose: likely (samples, channels) from some loaders
            waveform = waveform.T
    else:
        raise ValueError(
            f"Unexpected audio shape: {waveform.shape}. "
            "Expected 1D (mono) or 2D (stereo)."
        )

    # ── Validate channel count ─────────────────────────────────────────
    n_channels = waveform.shape[0]
    if n_channels == 1:
        raise ValueError(
            "Stereo audio required for two-speaker demo. "
            "Received mono audio (1 channel)."
        )
    if n_channels > 2:
        raise ValueError(
            f"Audio must have exactly 2 channels, got {n_channels}. "
            "Multi-channel files are not supported."
        )

    # ── Create OriginalStereoAudio ─────────────────────────────────────
    duration_sec = round(waveform.shape[1] / original_sr, 4)

    original_audio = OriginalStereoAudio(
        waveform=waveform.copy(),  # preserve original
        original_sr=original_sr,
        duration_sec=duration_sec,
        filename=filename,
        format=file_format,
    )

    # ── Run validation ─────────────────────────────────────────────────
    agent_channel = waveform[0]
    customer_channel = waveform[1]

    validation = validate_stereo(
        agent_waveform=agent_channel,
        customer_waveform=customer_channel,
        sample_rate=original_sr,
        config=config,
    )

    if not validation.is_valid:
        # Return partial result with validation errors
        # Caller should check validation.is_valid before using model_audio
        model_audio = ModelStereoAudio(
            agent_waveform=np.array([], dtype=np.float32),
            customer_waveform=np.array([], dtype=np.float32),
            model_sr=16000,
            duration_sec=0.0,
        )
        return StereoLoadResult(
            original_audio=original_audio,
            model_audio=model_audio,
            validation=validation,
            role_assignment_method="channel_convention",
        )

    # ── Resample to 16 kHz for model ──────────────────────────────────
    if original_sr != 16000:
        agent_model = librosa.resample(
            agent_channel, orig_sr=original_sr, target_sr=16000
        ).astype(np.float32)
        customer_model = librosa.resample(
            customer_channel, orig_sr=original_sr, target_sr=16000
        ).astype(np.float32)
    else:
        agent_model = agent_channel.astype(np.float32)
        customer_model = customer_channel.astype(np.float32)

    model_duration = round(len(agent_model) / 16000, 4)

    model_audio = ModelStereoAudio(
        agent_waveform=agent_model,
        customer_waveform=customer_model,
        model_sr=16000,
        duration_sec=model_duration,
    )

    return StereoLoadResult(
        original_audio=original_audio,
        model_audio=model_audio,
        validation=validation,
        role_assignment_method="channel_convention",
    )
