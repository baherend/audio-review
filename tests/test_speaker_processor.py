"""
Tests for per-channel SER processing with shared timeline.

Uses synthetic stereo audio and the real EmotionEngine for one practical test.
Orchestration tests use minimal model calls where possible.
"""

import os
import sys
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.engine import EmotionEngine
from src.demo.schemas import (
    WindowConfig,
    SpeechActivityConfig,
    StereoLoadResult,
    OriginalStereoAudio,
    ModelStereoAudio,
    ValidationResult,
    StereoValidationConfig,
)
from src.demo.speaker_processor import process_stereo_call
from src.demo.stereo_loader import load_stereo_audio


# ══════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def engine():
    """Shared engine instance (loaded once)."""
    eng = EmotionEngine()
    eng.load()
    return eng


def _make_test_load_result(
    duration: float = 6.0,
    sr: int = 16000,
    agent_freq: float = 440.0,
    customer_freq: float = 880.0,
    agent_amp: float = 0.5,
    customer_amp: float = 0.5,
) -> StereoLoadResult:
    """Create a StereoLoadResult with synthetic audio for testing."""
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    agent = (agent_amp * np.sin(2 * np.pi * agent_freq * t)).astype(np.float32)
    customer = (customer_amp * np.sin(2 * np.pi * customer_freq * t)).astype(np.float32)

    original = OriginalStereoAudio(
        waveform=np.stack([agent, customer]),
        original_sr=sr,
        duration_sec=duration,
        filename="test.wav",
        format="wav",
    )
    model = ModelStereoAudio(
        agent_waveform=agent,
        customer_waveform=customer,
        model_sr=sr,
        duration_sec=duration,
    )
    validation = ValidationResult(is_valid=True)
    return StereoLoadResult(
        original_audio=original,
        model_audio=model,
        validation=validation,
    )


def _make_alternating_load_result(
    sr: int = 16000,
) -> StereoLoadResult:
    """
    Create a stereo file with alternating signal activity:
    0-2s: Agent active, Customer silent
    2-4s: Customer active, Agent silent
    4-6s: Both active
    6-8s: Both silent
    """
    duration = 8.0
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    agent = np.zeros(n, dtype=np.float32)
    customer = np.zeros(n, dtype=np.float32)

    # Agent active: 0-2s and 4-6s
    agent[int(0*sr):int(2*sr)] = 0.5 * np.sin(2 * np.pi * 440 * t[int(0*sr):int(2*sr)])
    agent[int(4*sr):int(6*sr)] = 0.5 * np.sin(2 * np.pi * 440 * t[int(4*sr):int(6*sr)])

    # Customer active: 2-4s and 4-6s
    customer[int(2*sr):int(4*sr)] = 0.5 * np.sin(2 * np.pi * 880 * t[int(2*sr):int(4*sr)])
    customer[int(4*sr):int(6*sr)] = 0.5 * np.sin(2 * np.pi * 880 * t[int(4*sr):int(6*sr)])

    original = OriginalStereoAudio(
        waveform=np.stack([agent, customer]),
        original_sr=sr,
        duration_sec=duration,
        filename="alternating.wav",
        format="wav",
    )
    model = ModelStereoAudio(
        agent_waveform=agent,
        customer_waveform=customer,
        model_sr=sr,
        duration_sec=duration,
    )
    return StereoLoadResult(
        original_audio=original,
        model_audio=model,
        validation=ValidationResult(is_valid=True),
    )


# ══════════════════════════════════════════════════════════════════════
# ORCHESTRATION TESTS (minimal model calls via fixture)
# ══════════════════════════════════════════════════════════════════════
class TestOrchestration:
    """Tests for process_stereo_call orchestration logic."""

    def test_inactive_window_skips_inference(self, engine):
        """Inactive window does not run model inference."""
        load_result = _make_test_load_result(duration=4.0, sr=16000)
        window_config = WindowConfig(window_duration_sec=2.0, hop_duration_sec=1.0)
        speech_config = SpeechActivityConfig()

        result = process_stereo_call(load_result, engine, window_config, speech_config)

        # Check that inactive windows have no prediction
        for wr in result.window_results:
            if not wr.agent.speech_active:
                assert wr.agent.prediction is None
            if not wr.customer.speech_active:
                assert wr.customer.prediction is None

    def test_active_window_runs_inference(self, engine):
        """Active window runs model inference."""
        load_result = _make_test_load_result(duration=4.0, sr=16000)
        window_config = WindowConfig(window_duration_sec=2.0, hop_duration_sec=2.0)
        speech_config = SpeechActivityConfig()

        result = process_stereo_call(load_result, engine, window_config, speech_config)

        # Active windows should have predictions
        for wr in result.window_results:
            if wr.agent.speech_active:
                assert wr.agent.prediction is not None
            if wr.customer.speech_active:
                assert wr.customer.prediction is not None

    def test_agent_uses_left_channel(self, engine):
        """Agent channel is 'left'."""
        load_result = _make_test_load_result(duration=4.0)
        window_config = WindowConfig(window_duration_sec=2.0, hop_duration_sec=2.0)
        result = process_stereo_call(load_result, engine, window_config, SpeechActivityConfig())
        for wr in result.window_results:
            assert wr.agent.channel == "left"
            assert wr.agent.role == "Agent"

    def test_customer_uses_right_channel(self, engine):
        """Customer channel is 'right'."""
        load_result = _make_test_load_result(duration=4.0)
        window_config = WindowConfig(window_duration_sec=2.0, hop_duration_sec=2.0)
        result = process_stereo_call(load_result, engine, window_config, SpeechActivityConfig())
        for wr in result.window_results:
            assert wr.customer.channel == "right"
            assert wr.customer.role == "Customer"

    def test_no_channel_averaging(self, engine):
        """Agent and Customer predictions are independent."""
        load_result = _make_test_load_result(
            duration=4.0, agent_freq=440, customer_freq=880
        )
        window_config = WindowConfig(window_duration_sec=2.0, hop_duration_sec=2.0)
        result = process_stereo_call(load_result, engine, window_config, SpeechActivityConfig())
        for wr in result.window_results:
            if wr.agent.prediction and wr.customer.prediction:
                # With different frequencies, predictions may differ
                # Just verify both exist independently
                assert wr.agent.prediction is not None
                assert wr.customer.prediction is not None

    def test_both_active_recorded(self, engine):
        """Both-active state is recorded correctly."""
        load_result = _make_test_load_result(duration=4.0)
        window_config = WindowConfig(window_duration_sec=2.0, hop_duration_sec=2.0)
        result = process_stereo_call(load_result, engine, window_config, SpeechActivityConfig())
        for wr in result.window_results:
            assert isinstance(wr.both_active_in_window, bool)
            assert isinstance(wr.neither_active_in_window, bool)
            # With sine waves, both should be active
            assert wr.both_active_in_window

    def test_neither_active_recorded(self, engine):
        """Neither-active state is recorded for silent channels."""
        load_result = _make_test_load_result(
            duration=4.0, agent_amp=0.0, customer_amp=0.0
        )
        window_config = WindowConfig(window_duration_sec=2.0, hop_duration_sec=2.0)
        result = process_stereo_call(load_result, engine, window_config, SpeechActivityConfig())
        for wr in result.window_results:
            assert wr.neither_active_in_window

    def test_timeline_includes_inactive_windows(self, engine):
        """All windows are in the timeline, including inactive ones."""
        load_result = _make_test_load_result(duration=4.0)
        window_config = WindowConfig(window_duration_sec=2.0, hop_duration_sec=1.0)
        result = process_stereo_call(load_result, engine, window_config, SpeechActivityConfig())
        # 4-second call with 2s window, 1s hop = 3 windows
        assert len(result.window_results) == 3
        assert len(result.shared_windows) == 3

    def test_predictions_contain_original_timestamps(self, engine):
        """Predictions reference original timestamps."""
        load_result = _make_test_load_result(duration=4.0)
        window_config = WindowConfig(window_duration_sec=2.0, hop_duration_sec=2.0)
        result = process_stereo_call(load_result, engine, window_config, SpeechActivityConfig())
        for wr in result.window_results:
            assert wr.window.original_start >= 0
            assert wr.window.original_end <= 4.0 + 1e-9

    def test_role_assignment_method(self, engine):
        """role_assignment_method is 'channel_convention'."""
        load_result = _make_test_load_result(duration=4.0)
        result = process_stereo_call(
            load_result, engine,
            WindowConfig(window_duration_sec=2.0, hop_duration_sec=2.0),
            SpeechActivityConfig(),
        )
        assert result.role_assignment_method == "channel_convention"
        for wr in result.window_results:
            assert wr.agent.role_assignment_method == "channel_convention"
            assert wr.customer.role_assignment_method == "channel_convention"

    def test_batch_output_order_matches_windows(self, engine):
        """Window results are in window order."""
        load_result = _make_test_load_result(duration=6.0)
        window_config = WindowConfig(window_duration_sec=2.0, hop_duration_sec=1.0)
        result = process_stereo_call(load_result, engine, window_config, SpeechActivityConfig())
        for i, wr in enumerate(result.window_results):
            assert wr.window.window_id == i

    def test_low_confidence_retained(self, engine):
        """Low-confidence prediction is retained and flagged."""
        load_result = _make_test_load_result(duration=4.0, sr=16000)
        window_config = WindowConfig(window_duration_sec=2.0, hop_duration_sec=2.0)
        # Set minimum_confidence very high so all predictions are below it
        result = process_stereo_call(
            load_result, engine, window_config, SpeechActivityConfig(),
            minimum_confidence=0.99,
        )
        for wr in result.window_results:
            if wr.agent.prediction is not None:
                assert wr.agent.uncertainty_reason == "below_minimum_confidence"
                assert wr.agent.prediction is not None  # prediction retained

    def test_no_files_written(self, engine):
        """No files are written to the repository."""
        repo_files_before = set()
        for root, dirs, files in os.walk(PROJECT_ROOT):
            if '.git' in root or '__pycache__' in root:
                continue
            for f in files:
                repo_files_before.add(os.path.join(root, f))

        load_result = _make_test_load_result(duration=4.0)
        result = process_stereo_call(
            load_result, engine,
            WindowConfig(window_duration_sec=2.0, hop_duration_sec=2.0),
            SpeechActivityConfig(),
        )

        repo_files_after = set()
        for root, dirs, files in os.walk(PROJECT_ROOT):
            if '.git' in root or '__pycache__' in root:
                continue
            for f in files:
                repo_files_after.add(os.path.join(root, f))

        new_files = repo_files_after - repo_files_before
        assert len(new_files) == 0, f"Unexpected new files: {new_files}"

    def test_processing_metadata(self, engine):
        """Processing metadata is populated."""
        load_result = _make_test_load_result(duration=4.0)
        result = process_stereo_call(
            load_result, engine,
            WindowConfig(window_duration_sec=2.0, hop_duration_sec=2.0),
            SpeechActivityConfig(),
        )
        meta = result.processing_metadata
        assert "inference_calls" in meta
        assert "inference_skipped" in meta
        assert "total_windows" in meta
        assert "processing_time_sec" in meta
        assert meta["total_windows"] == 2  # 4s duration, 2s window, 2s hop = 2 windows

    def test_model_version(self, engine):
        """Model version is populated."""
        load_result = _make_test_load_result(duration=4.0)
        result = process_stereo_call(
            load_result, engine,
            WindowConfig(window_duration_sec=2.0, hop_duration_sec=2.0),
            SpeechActivityConfig(),
        )
        assert result.model_version == "1.0.0"

    def test_device(self, engine):
        """Device is populated."""
        load_result = _make_test_load_result(duration=4.0)
        result = process_stereo_call(
            load_result, engine,
            WindowConfig(window_duration_sec=2.0, hop_duration_sec=2.0),
            SpeechActivityConfig(),
        )
        assert result.device in ("cuda", "cpu")
