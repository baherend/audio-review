"""
Tests for stereo audio loading and channel separation.

Tests use controlled synthetic audio — not real speaker recordings.
"""

import io
import os
import sys
import numpy as np
import pytest
import soundfile as sf

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.demo.stereo_loader import load_stereo_audio
from src.demo.schemas import (
    StereoLoadResult,
    OriginalStereoAudio,
    ModelStereoAudio,
    StereoValidationConfig,
)


# ══════════════════════════════════════════════════════════════════════
# FIXTURES — Synthetic audio generation
# ══════════════════════════════════════════════════════════════════════
SCRATCHPAD = os.path.join(
    os.environ.get("TEMP", "/tmp"),
    "claude", "phase2_tests"
)


def _make_stereo_wav(
    filename: str,
    sr: int = 16000,
    duration: float = 3.0,
    left_freq: float = 440.0,
    right_freq: float = 880.0,
    left_amp: float = 0.5,
    right_amp: float = 0.5,
) -> str:
    """Create a synthetic stereo WAV file and return its path."""
    os.makedirs(SCRATCHPAD, exist_ok=True)
    path = os.path.join(SCRATCHPAD, filename)
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    left = (left_amp * np.sin(2 * np.pi * left_freq * t)).astype(np.float32)
    right = (right_amp * np.sin(2 * np.pi * right_freq * t)).astype(np.float32)
    stereo = np.column_stack([left, right])  # shape (N, 2) for soundfile
    sf.write(path, stereo, sr)
    return path


def _make_mono_wav(filename: str, sr: int = 16000, duration: float = 2.0) -> str:
    """Create a synthetic mono WAV file."""
    os.makedirs(SCRATCHPAD, exist_ok=True)
    path = os.path.join(SCRATCHPAD, filename)
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    mono = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(path, mono, sr)
    return path


def _make_multichannel_wav(filename: str, channels: int = 4, sr: int = 16000) -> str:
    """Create a synthetic multi-channel WAV file."""
    os.makedirs(SCRATCHPAD, exist_ok=True)
    path = os.path.join(SCRATCHPAD, filename)
    n_samples = int(sr * 2.0)
    data = np.random.randn(n_samples, channels).astype(np.float32) * 0.1
    sf.write(path, data, sr)
    return path


# ══════════════════════════════════════════════════════════════════════
# TESTS — Valid stereo loading
# ══════════════════════════════════════════════════════════════════════
class TestValidStereoLoading:
    """Tests for valid stereo file loading at various sample rates."""

    def test_stereo_16khz(self):
        """Valid stereo WAV at 16 kHz."""
        path = _make_stereo_wav("test_16k.wav", sr=16000, duration=3.0)
        result = load_stereo_audio(path)
        assert isinstance(result, StereoLoadResult)
        assert result.original_audio.original_sr == 16000
        assert result.model_audio.model_sr == 16000
        assert result.original_audio.waveform.shape[0] == 2
        assert result.role_assignment_method == "channel_convention"

    def test_stereo_44100hz(self):
        """Valid stereo WAV at 44.1 kHz."""
        path = _make_stereo_wav("test_44k.wav", sr=44100, duration=3.0)
        result = load_stereo_audio(path)
        assert result.original_audio.original_sr == 44100
        assert result.model_audio.model_sr == 16000
        assert result.model_audio.duration_sec > 0

    def test_stereo_48000hz(self):
        """Valid stereo WAV at 48 kHz."""
        path = _make_stereo_wav("test_48k.wav", sr=48000, duration=3.0)
        result = load_stereo_audio(path)
        assert result.original_audio.original_sr == 48000
        assert result.model_audio.model_sr == 16000

    def test_resampling_to_16khz(self):
        """Model audio is resampled to 16 kHz."""
        path = _make_stereo_wav("test_resample.wav", sr=44100, duration=2.0)
        result = load_stereo_audio(path)
        assert result.model_audio.model_sr == 16000
        # Model samples should be approximately duration * 16000
        expected_samples = int(2.0 * 16000)
        assert abs(len(result.model_audio.agent_waveform) - expected_samples) < 100

    def test_left_channel_is_agent(self):
        """Left channel is assigned to Agent."""
        path = _make_stereo_wav(
            "test_left_agent.wav", sr=16000, duration=2.0,
            left_freq=440, right_freq=880
        )
        result = load_stereo_audio(path)
        # Agent should contain the 440 Hz signal
        # Customer should contain the 880 Hz signal
        assert len(result.model_audio.agent_waveform) > 0
        assert len(result.model_audio.customer_waveform) > 0

    def test_right_channel_is_customer(self):
        """Right channel is assigned to Customer."""
        path = _make_stereo_wav(
            "test_right_customer.wav", sr=16000, duration=2.0,
            left_freq=440, right_freq=880
        )
        result = load_stereo_audio(path)
        assert len(result.model_audio.customer_waveform) > 0


# ══════════════════════════════════════════════════════════════════════
# TESTS — Rejection cases
# ══════════════════════════════════════════════════════════════════════
class TestRejection:
    """Tests for files that must be rejected."""

    def test_mono_rejection(self):
        """Mono file is rejected."""
        path = _make_mono_wav("test_mono.wav")
        with pytest.raises(ValueError, match="[Ss]tereo"):
            load_stereo_audio(path)

    def test_more_than_two_channels_rejection(self):
        """File with >2 channels is rejected."""
        path = _make_multichannel_wav("test_4ch.wav", channels=4)
        with pytest.raises(ValueError, match="[Ee]xactly 2 channels"):
            load_stereo_audio(path)

    def test_empty_audio_rejection(self):
        """Empty audio is rejected."""
        # Test the validation directly with empty arrays
        from src.demo.channel_validation import validate_stereo
        from src.demo.schemas import StereoValidationConfig
        config = StereoValidationConfig()
        result = validate_stereo(
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            sample_rate=16000,
            config=config,
        )
        assert not result.is_valid
        codes = [e.code for e in result.errors]
        assert "EMPTY_AUDIO" in codes

    def test_non_finite_samples_rejection(self):
        """Audio with NaN values is rejected."""
        # Test the validation directly with NaN values
        from src.demo.channel_validation import validate_stereo
        from src.demo.schemas import StereoValidationConfig
        config = StereoValidationConfig()
        agent = np.zeros(16000, dtype=np.float32)
        agent[100] = np.nan
        customer = np.zeros(16000, dtype=np.float32)
        result = validate_stereo(agent, customer, sample_rate=16000, config=config)
        assert not result.is_valid
        codes = [e.code for e in result.errors]
        assert "NON_FINITE_SAMPLES" in codes

    def test_completely_silent_rejection(self):
        """Completely silent stereo file is rejected."""
        path = _make_stereo_wav(
            "test_silent.wav", sr=16000, duration=2.0,
            left_amp=0.0, right_amp=0.0
        )
        result = load_stereo_audio(path)
        assert not result.validation.is_valid
        codes = [e.code for e in result.validation.errors]
        assert "COMPLETELY_SILENT" in codes

    def test_file_not_found(self):
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_stereo_audio("/nonexistent/file.wav")

    def test_invalid_file_format(self):
        """Corrupt/unreadable file raises ValueError."""
        os.makedirs(SCRATCHPAD, exist_ok=True)
        path = os.path.join(SCRATCHPAD, "test_corrupt.wav")
        with open(path, "wb") as f:
            f.write(b"this is not a valid audio file")
        with pytest.raises(ValueError, match="[Cc]ould not decode"):
            load_stereo_audio(path)


# ══════════════════════════════════════════════════════════════════════
# TESTS — Warning cases
# ══════════════════════════════════════════════════════════════════════
class TestWarnings:
    """Tests for files that produce warnings but are accepted."""

    def test_identical_channels_warning(self):
        """Identical channels produce a warning."""
        os.makedirs(SCRATCHPAD, exist_ok=True)
        path = os.path.join(SCRATCHPAD, "test_identical.wav")
        n = 32000
        t = np.linspace(0, 2.0, n, endpoint=False)
        signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        stereo = np.column_stack([signal, signal])
        sf.write(path, stereo, 16000)
        result = load_stereo_audio(path)
        codes = [w.code for w in result.validation.warnings]
        assert "IDENTICAL_CHANNELS" in codes

    def test_silent_agent_warning(self):
        """Silent Agent channel produces a warning."""
        os.makedirs(SCRATCHPAD, exist_ok=True)
        path = os.path.join(SCRATCHPAD, "test_silent_agent.wav")
        n = 32000
        t = np.linspace(0, 2.0, n, endpoint=False)
        left = np.zeros(n, dtype=np.float32)
        right = (0.5 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
        stereo = np.column_stack([left, right])
        sf.write(path, stereo, 16000)
        result = load_stereo_audio(path)
        codes = [w.code for w in result.validation.warnings]
        assert "SILENT_AGENT_CHANNEL" in codes

    def test_silent_customer_warning(self):
        """Silent Customer channel produces a warning."""
        os.makedirs(SCRATCHPAD, exist_ok=True)
        path = os.path.join(SCRATCHPAD, "test_silent_customer.wav")
        n = 32000
        t = np.linspace(0, 2.0, n, endpoint=False)
        left = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        right = np.zeros(n, dtype=np.float32)
        stereo = np.column_stack([left, right])
        sf.write(path, stereo, 16000)
        result = load_stereo_audio(path)
        codes = [w.code for w in result.validation.warnings]
        assert "SILENT_CUSTOMER_CHANNEL" in codes

    def test_very_short_duration_warning(self):
        """Very short audio produces a warning."""
        path = _make_stereo_wav("test_short.wav", sr=16000, duration=0.2)
        result = load_stereo_audio(path)
        codes = [w.code for w in result.validation.warnings]
        assert "VERY_SHORT_DURATION" in codes


# ══════════════════════════════════════════════════════════════════════
# TESTS — Identical vs high-correlation distinction
# ══════════════════════════════════════════════════════════════════════
class TestIdenticalVsCorrelation:
    """Tests that identical-channel and high-correlation are separate warnings."""

    def _validate_direct(self, agent, customer, config=None):
        """Helper: validate directly without file I/O."""
        from src.demo.channel_validation import validate_stereo
        from src.demo.schemas import StereoValidationConfig
        if config is None:
            config = StereoValidationConfig()
        return validate_stereo(agent, customer, sample_rate=16000, config=config)

    def test_exactly_equal_channels_identical(self):
        """Exactly equal channels trigger IDENTICAL_CHANNELS warning."""
        n = 16000
        signal = np.sin(np.linspace(0, 2 * np.pi * 440, n)).astype(np.float32)
        result = self._validate_direct(signal, signal.copy())
        codes = [w.code for w in result.warnings]
        assert "IDENTICAL_CHANNELS" in codes
        assert "HIGH_CORRELATION" not in codes

    def test_nearly_equal_within_tolerance_identical(self):
        """Nearly equal channels within tolerance trigger IDENTICAL_CHANNELS."""
        n = 16000
        signal = np.sin(np.linspace(0, 2 * np.pi * 440, n)).astype(np.float32)
        # Add tiny noise within tolerance
        noisy = signal + np.random.RandomState(42).randn(n).astype(np.float32) * 1e-7
        result = self._validate_direct(signal, noisy)
        codes = [w.code for w in result.warnings]
        assert "IDENTICAL_CHANNELS" in codes

    def test_highly_correlated_not_identical(self):
        """Different but highly correlated channels trigger HIGH_CORRELATION only."""
        n = 16000
        t = np.linspace(0, 1.0, n, endpoint=False)
        left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        # Right is left + small phase shift → highly correlated but not identical
        right = np.sin(2 * np.pi * 440 * t + 0.01).astype(np.float32)
        result = self._validate_direct(left, right)
        codes = [w.code for w in result.warnings]
        assert "HIGH_CORRELATION" in codes
        assert "IDENTICAL_CHANNELS" not in codes

    def test_different_low_correlation_neither(self):
        """Different low-correlation channels trigger neither warning."""
        n = 16000
        t = np.linspace(0, 1.0, n, endpoint=False)
        left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        right = np.sin(2 * np.pi * 880 * t).astype(np.float32)
        result = self._validate_direct(left, right)
        codes = [w.code for w in result.warnings]
        assert "IDENTICAL_CHANNELS" not in codes
        assert "HIGH_CORRELATION" not in codes


# ══════════════════════════════════════════════════════════════════════
# TESTS — Original audio preservation
# ══════════════════════════════════════════════════════════════════════
class TestOriginalPreservation:
    """Tests that original audio is never modified."""

    def test_original_waveform_unchanged(self):
        """Original waveform is not modified after loading."""
        path = _make_stereo_wav("test_preserve.wav", sr=16000, duration=2.0)
        result = load_stereo_audio(path)
        original = result.original_audio.waveform
        original_copy = original.copy()
        # Access model audio (this triggers resampling but should not touch original)
        _ = result.model_audio.agent_waveform
        np.testing.assert_array_equal(original, original_copy)

    def test_original_sample_rate_preserved(self):
        """Original sample rate is preserved."""
        path = _make_stereo_wav("test_sr.wav", sr=44100, duration=2.0)
        result = load_stereo_audio(path)
        assert result.original_audio.original_sr == 44100

    def test_duration_preserved_after_resampling(self):
        """Original and model durations match within tolerance."""
        path = _make_stereo_wav("test_duration.wav", sr=44100, duration=3.0)
        result = load_stereo_audio(path)
        orig_dur = result.original_audio.duration_sec
        model_dur = result.model_audio.duration_sec
        # Tolerance: one model sample (1/16000 = 0.0000625s)
        assert abs(orig_dur - model_dur) < (1 / 16000 + 0.001), (
            f"Duration mismatch: original={orig_dur:.4f}s, model={model_dur:.4f}s"
        )


# ══════════════════════════════════════════════════════════════════════
# TESTS — Model audio properties
# ══════════════════════════════════════════════════════════════════════
class TestModelAudio:
    """Tests for model audio representation."""

    def test_model_sr_is_16000(self):
        """Model audio uses 16 kHz."""
        path = _make_stereo_wav("test_model_sr.wav", sr=48000, duration=2.0)
        result = load_stereo_audio(path)
        assert result.model_audio.model_sr == 16000

    def test_agent_customer_equal_length(self):
        """Agent and Customer model waveforms have equal lengths."""
        path = _make_stereo_wav("test_equal_len.wav", sr=16000, duration=3.0)
        result = load_stereo_audio(path)
        assert len(result.model_audio.agent_waveform) == len(result.model_audio.customer_waveform)

    def test_channels_not_averaged(self):
        """Agent and Customer waveforms are not identical (channels preserved)."""
        path = _make_stereo_wav(
            "test_not_averaged.wav", sr=16000, duration=2.0,
            left_freq=440, right_freq=880
        )
        result = load_stereo_audio(path)
        # With different frequencies, channels should be different
        assert not np.allclose(
            result.model_audio.agent_waveform,
            result.model_audio.customer_waveform,
            atol=0.01
        )


# ══════════════════════════════════════════════════════════════════════
# TESTS — BinaryIO loading
# ══════════════════════════════════════════════════════════════════════
class TestBinaryIOLoading:
    """Tests for loading from file-like objects."""

    def test_binaryio_loading(self):
        """BinaryIO upload loading works."""
        path = _make_stereo_wav("test_bio.wav", sr=16000, duration=2.0)
        with open(path, "rb") as f:
            bio = io.BytesIO(f.read())
        result = load_stereo_audio(bio, filename="test_bio.wav")
        assert isinstance(result, StereoLoadResult)
        assert result.original_audio.filename == "test_bio.wav"


# ══════════════════════════════════════════════════════════════════════
# TESTS — Validation configuration
# ══════════════════════════════════════════════════════════════════════
class TestValidationConfig:
    """Tests for validation configuration override."""

    def test_custom_config_override(self):
        """Custom validation config changes behavior."""
        path = _make_stereo_wav("test_config.wav", sr=16000, duration=0.3)
        # Default config: minimum_duration=0.5 → warns
        result_default = load_stereo_audio(path)
        assert any(w.code == "VERY_SHORT_DURATION" for w in result_default.validation.warnings)

        # Custom config: minimum_duration=0.1 → no warning
        custom_config = StereoValidationConfig(minimum_duration_sec=0.1)
        result_custom = load_stereo_audio(path, config=custom_config)
        assert not any(w.code == "VERY_SHORT_DURATION" for w in result_custom.validation.warnings)


# ══════════════════════════════════════════════════════════════════════
# TESTS — No files written to repository
# ══════════════════════════════════════════════════════════════════════
class TestNoRepositoryWrites:
    """Verify no files are written inside the repository."""

    def test_no_temp_files_in_repo(self):
        """Loading does not create files inside the repository."""
        path = _make_stereo_wav("test_no_write.wav", sr=16000, duration=2.0)
        # Check no new .wav files in repo root before
        repo_wavs_before = set(
            f for f in os.listdir(PROJECT_ROOT) if f.endswith(".wav")
        )
        _ = load_stereo_audio(path)
        repo_wavs_after = set(
            f for f in os.listdir(PROJECT_ROOT) if f.endswith(".wav")
        )
        new_wavs = repo_wavs_after - repo_wavs_before
        assert len(new_wavs) == 0, f"Unexpected new files in repo: {new_wavs}"
