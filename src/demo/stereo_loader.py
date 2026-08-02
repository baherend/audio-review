"""
Stereo audio loading and channel separation.

Loads stereo audio, preserves the original waveform, creates 16 kHz model copies,
and runs validation. Never trims, never deletes samples, never compresses time.
"""

import io
from pathlib import Path
from typing import BinaryIO, Optional, Union

import numpy as np
import soxr

from .schemas import (
    OriginalStereoAudio,
    ModelStereoAudio,
    StereoLoadResult,
    StereoValidationConfig,
)
from .channel_validation import validate_stereo


class AudioInputError(ValueError):
    """Controlled validation error for audio supplied by a user."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _reject_truncated_wav(header: bytes, actual_size: int) -> None:
    """Reject RIFF WAV files whose declared container length is unavailable."""
    if len(header) < 12 or header[8:12] != b"WAVE":
        return
    if header[:4] == b"RIFF":
        byte_order = "little"
    elif header[:4] == b"RIFX":
        byte_order = "big"
    else:
        return

    declared_size = int.from_bytes(header[4:8], byteorder=byte_order)
    if declared_size not in (0, 0xFFFFFFFF) and declared_size + 8 > actual_size:
        raise AudioInputError(
            "INVALID_AUDIO",
            "Could not decode audio. The file may be corrupt, incomplete, "
            "or use an unsupported encoding.",
        )


def _resample_channel_for_model(
    waveform: np.ndarray,
    original_sr: int,
    target_sr: int = 16000,
) -> np.ndarray:
    """Resample one channel with SoXR HQ and preserve the prior output contract."""
    expected_samples = int(
        np.ceil(waveform.shape[0] * (float(target_sr) / original_sr))
    )
    resampled = soxr.resample(
        waveform,
        in_rate=original_sr,
        out_rate=target_sr,
        quality="HQ",
    )
    if resampled.shape[0] < expected_samples:
        resampled = np.pad(
            resampled,
            (0, expected_samples - resampled.shape[0]),
        )
    elif resampled.shape[0] > expected_samples:
        resampled = resampled[:expected_samples]
    return np.asarray(resampled, dtype=np.float32)


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
        AudioInputError: If input is empty, corrupt, mono, or not stereo.
        FileNotFoundError: If file path does not exist.
    """
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
        file_size = path.stat().st_size
        if file_size == 0:
            raise AudioInputError(
                "EMPTY_FILE",
                "The uploaded audio file is empty. Choose a non-empty stereo WAV file.",
            )
        with path.open("rb") as audio_file:
            _reject_truncated_wav(audio_file.read(12), file_size)
    elif hasattr(source, "read"):
        # BinaryIO — read all bytes into a buffer
        source.seek(0)
        audio_bytes = source.read()
        if not audio_bytes:
            raise AudioInputError(
                "EMPTY_FILE",
                "The uploaded audio file is empty. Choose a non-empty stereo WAV file.",
            )
        _reject_truncated_wav(audio_bytes[:12], len(audio_bytes))
        buffer = io.BytesIO(audio_bytes)
        if filename is None:
            filename = "uploaded_audio"
        file_format = "unknown"
        path = None
    else:
        raise TypeError(f"source must be str, Path, or BinaryIO, got {type(source).__name__}")

    # ── Decode audio ───────────────────────────────────────────────────
    try:
        import soundfile as sf

        decode_source = str(path) if path is not None else buffer
        audio_info = sf.info(decode_source)
        if audio_info.frames <= 0:
            raise AudioInputError(
                "EMPTY_FILE",
                "The uploaded audio file is empty. Choose a non-empty stereo WAV file.",
            )
        if audio_info.channels == 1:
            raise AudioInputError(
                "MONO_INPUT",
                "Stereo audio is required. The file contains one channel. "
                "Left must be Agent and Right must be Customer.",
            )
        if audio_info.channels != 2:
            raise AudioInputError(
                "UNSUPPORTED_CHANNEL_COUNT",
                "Audio must have exactly 2 channels. Multi-channel files are not supported. "
                "Left must be Agent and Right must be Customer.",
            )

        if path is None:
            buffer.seek(0)
        waveform, original_sr = sf.read(
            decode_source,
            dtype="float32",
            always_2d=True,
        )
        waveform = waveform.T
    except AudioInputError:
        raise
    except Exception as e:
        raise AudioInputError(
            "INVALID_AUDIO",
            "Could not decode audio. The file may be corrupt, incomplete, "
            "or use an unsupported encoding.",
        ) from e

    # ── Normalize orientation to (channels, samples) ───────────────────
    if waveform.ndim == 1:
        # Mono — rejected
        raise AudioInputError(
            "MONO_INPUT",
            "Stereo audio is required. The file contains one channel. "
            "Left must be Agent and Right must be Customer.",
        )
    elif waveform.ndim == 2:
        # SoundFile is read with always_2d=True and transposed above.
        # Retain a defensive check for an unexpected (samples, channels) layout.
        if waveform.shape[0] > waveform.shape[1]:
            # Transpose: likely (samples, channels) from some loaders
            waveform = waveform.T
    else:
        raise AudioInputError(
            "INVALID_AUDIO",
            "Could not decode audio. The file may be corrupt, incomplete, "
            "or use an unsupported encoding.",
        )

    # ── Validate channel count ─────────────────────────────────────────
    n_channels = waveform.shape[0]
    if n_channels == 1:
        raise AudioInputError(
            "MONO_INPUT",
            "Stereo audio is required. The file contains one channel. "
            "Left must be Agent and Right must be Customer.",
        )
    if n_channels > 2:
        raise AudioInputError(
            "UNSUPPORTED_CHANNEL_COUNT",
            "Audio must have exactly 2 channels. Multi-channel files are not supported. "
            "Left must be Agent and Right must be Customer.",
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
        agent_model = _resample_channel_for_model(agent_channel, original_sr)
        customer_model = _resample_channel_for_model(customer_channel, original_sr)
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
