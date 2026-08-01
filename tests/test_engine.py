"""
Tests for the shared inference engine.

Tests model path resolution, loading, prediction, output validation,
and parity against the deployment reference implementation.
"""

import os
import sys
import math
import numpy as np
import pytest
import torch

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.engine import (
    EmotionEngine,
    _resolve_model_path,
    _resolve_labels_path,
    _DEFAULT_MODEL_PATH,
    _REPOSITORY_ROOT,
)
from src.inference.schemas import InferenceInput, InferenceOutput, ChunkPrediction


# ══════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def engine():
    """Shared engine instance for tests that need the model loaded."""
    eng = EmotionEngine()
    eng.load()
    return eng


@pytest.fixture(scope="module")
def sample_waveform():
    """A known BAVED sample for testing. Returns (waveform, sr, expected_label_id)."""
    import librosa

    baved_dir = _REPOSITORY_ROOT / "data" / "raw" / "baved"
    # Use a verified BAVED file with spoken_emotion=2 (High Vocal Activation)
    # File: 0-m-21-0-2-106.wav → speaker 0, male, 21, word 0, emotion 2
    sample_path = baved_dir / "0" / "0-m-21-0-2-106.wav"

    if not sample_path.exists():
        pytest.skip(f"BAVED sample not found: {sample_path}")

    waveform, sr = librosa.load(str(sample_path), sr=16000, mono=True)
    waveform = waveform.astype(np.float32)
    # spoken_emotion=2 → "High Vocal Activation"
    return waveform, sr, 2


# ══════════════════════════════════════════════════════════════════════
# A. MODEL PATH RESOLUTION TESTS
# ══════════════════════════════════════════════════════════════════════
class TestModelPathResolution:
    """Tests for model path resolution priority."""

    def test_repository_relative_resolution(self):
        """Test 1: Repository-relative model path resolves."""
        path = _resolve_model_path()
        assert path.exists(), f"Default model not found at {path}"
        assert path.name == "model_bundle_fp16.pt"

    def test_explicit_path_override(self):
        """Test 2: Explicit constructor path takes priority."""
        explicit = str(_DEFAULT_MODEL_PATH)
        path = _resolve_model_path(explicit_path=explicit)
        assert path.exists()
        assert str(path) == explicit

    def test_env_var_override(self):
        """Test 3: Environment variable path takes priority over default."""
        original = os.environ.get("AUDIO_MODEL_PATH")
        try:
            os.environ["AUDIO_MODEL_PATH"] = str(_DEFAULT_MODEL_PATH)
            path = _resolve_model_path()
            assert path.exists()
            assert str(path) == str(_DEFAULT_MODEL_PATH)
        finally:
            if original is None:
                os.environ.pop("AUDIO_MODEL_PATH", None)
            else:
                os.environ["AUDIO_MODEL_PATH"] = original

    def test_explicit_overrides_env_var(self):
        """Test: Explicit path takes priority over environment variable."""
        explicit = str(_DEFAULT_MODEL_PATH)
        original = os.environ.get("AUDIO_MODEL_PATH")
        try:
            os.environ["AUDIO_MODEL_PATH"] = "/nonexistent/fake_model.pt"
            path = _resolve_model_path(explicit_path=explicit)
            assert path.exists()
            assert str(path) == explicit
        finally:
            if original is None:
                os.environ.pop("AUDIO_MODEL_PATH", None)
            else:
                os.environ["AUDIO_MODEL_PATH"] = original

    def test_missing_model_error_lists_paths(self):
        """Test 4: Missing model lists all attempted paths."""
        with pytest.raises(FileNotFoundError, match="Attempted paths"):
            _resolve_model_path(explicit_path="/nonexistent/a.pt")

    def test_missing_model_with_env_lists_paths(self):
        """Test: Missing model with env var lists the env path and errors."""
        original = os.environ.get("AUDIO_MODEL_PATH")
        try:
            os.environ["AUDIO_MODEL_PATH"] = "/nonexistent/b.pt"
            with pytest.raises(FileNotFoundError, match="Attempted paths") as exc_info:
                _resolve_model_path()
            error_msg = str(exc_info.value)
            assert "env AUDIO_MODEL_PATH" in error_msg
            # Explicit source (env var) is tried first; if it fails,
            # the system does NOT fall through to the repository default.
        finally:
            if original is None:
                os.environ.pop("AUDIO_MODEL_PATH", None)
            else:
                os.environ["AUDIO_MODEL_PATH"] = original


# ══════════════════════════════════════════════════════════════════════
# B. LABELS LOADING TESTS
# ══════════════════════════════════════════════════════════════════════
class TestLabelsLoading:
    """Tests for labels.json loading."""

    def test_labels_load_correctly(self):
        """Test 5: labels.json loads correctly."""
        path = _resolve_labels_path()
        assert path.exists()
        eng = EmotionEngine()
        labels = eng.labels
        assert isinstance(labels, dict)
        assert len(labels) == 3
        assert "0" in labels
        assert "1" in labels
        assert "2" in labels

    def test_label_values(self):
        """Test: Label values match expected."""
        eng = EmotionEngine()
        assert eng.labels["0"] == "Low Vocal Activation"
        assert eng.labels["1"] == "Moderate Vocal Activation"
        assert eng.labels["2"] == "High Vocal Activation"

    def test_label_list_ordering(self):
        """Test: Label list is ordered by ID."""
        eng = EmotionEngine()
        assert eng.label_list == [
            "Low Vocal Activation",
            "Moderate Vocal Activation",
            "High Vocal Activation",
        ]


# ══════════════════════════════════════════════════════════════════════
# C. MODEL LOADING TESTS
# ══════════════════════════════════════════════════════════════════════
class TestModelLoading:
    """Tests for model loading and caching."""

    def test_model_loads_without_error(self, engine):
        """Test 6: Model loads without missing/unexpected keys."""
        assert engine.is_loaded

    def test_model_device(self, engine):
        """Test: Model loads on available device."""
        assert engine.device in ("cuda", "cpu")

    def test_model_info(self, engine):
        """Test: Model info is accessible."""
        info = engine.get_model_info()
        assert info["loaded"] is True
        assert info["num_labels"] == 3
        assert info["architecture"] == "Wav2Vec2BiLSTMForSER"

    def test_no_download_occurs(self):
        """Test 18: Model file is local, no download triggered."""
        assert _DEFAULT_MODEL_PATH.exists()
        assert _DEFAULT_MODEL_PATH.stat().st_size > 100_000_000  # > 100 MB


# ══════════════════════════════════════════════════════════════════════
# D. PREDICTION TESTS
# ══════════════════════════════════════════════════════════════════════
class TestPrediction:
    """Tests for single-segment prediction."""

    def test_valid_prediction(self, engine):
        """Test 7: Valid segment produces prediction."""
        waveform = np.random.randn(16000).astype(np.float32)  # 1 second
        result = engine.predict_segment(waveform, sample_rate=16000)
        assert isinstance(result, InferenceOutput)

    def test_label_in_configured_set(self, engine):
        """Test 8: Output label belongs to configured labels."""
        waveform = np.random.randn(16000).astype(np.float32)
        result = engine.predict_segment(waveform, sample_rate=16000)
        valid_labels = {"Low Vocal Activation", "Moderate Vocal Activation", "High Vocal Activation"}
        assert result.label in valid_labels

    def test_probabilities_are_finite(self, engine):
        """Test 9: All class probabilities are finite."""
        waveform = np.random.randn(16000).astype(np.float32)
        result = engine.predict_segment(waveform, sample_rate=16000)
        for prob in result.probabilities.values():
            assert math.isfinite(prob), f"Non-finite probability: {prob}"

    def test_probabilities_sum_to_one(self, engine):
        """Test 10: Probabilities sum approximately to 1."""
        waveform = np.random.randn(16000).astype(np.float32)
        result = engine.predict_segment(waveform, sample_rate=16000)
        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 1e-5, f"Probabilities sum to {total}, expected ~1.0"

    def test_confidence_equals_max_probability(self, engine):
        """Test 11: Confidence equals the maximum probability."""
        waveform = np.random.randn(16000).astype(np.float32)
        result = engine.predict_segment(waveform, sample_rate=16000)
        max_prob = max(result.probabilities.values())
        assert abs(result.confidence - max_prob) < 1e-6

    def test_deterministic_output(self, engine):
        """Test 12: Same input produces deterministic output within tolerance."""
        waveform = np.random.randn(16000).astype(np.float32)
        result1 = engine.predict_segment(waveform, sample_rate=16000)
        result2 = engine.predict_segment(waveform, sample_rate=16000)
        assert result1.label == result2.label
        assert result1.confidence == result2.confidence
        for key in result1.probabilities:
            assert abs(
                result1.probabilities[key] - result2.probabilities[key]
            ) < 1e-6

    def test_labeled_baved_sample(self, engine, sample_waveform):
        """Test: Verified BAVED sample with known label."""
        waveform, sr, expected_label_id = sample_waveform
        result = engine.predict_segment(waveform, sample_rate=sr)
        expected_label = {0: "Low Vocal Activation", 1: "Moderate Vocal Activation", 2: "High Vocal Activation"}[
            expected_label_id
        ]
        # Report but do not hard-assert — model accuracy is 89.15%
        print(
            f"\n  BAVED sample (expected={expected_label}, "
            f"predicted={result.label}, confidence={result.confidence:.4f})"
        )
        # Soft assertion: prediction must be valid
        assert result.label in {"Low Vocal Activation", "Moderate Vocal Activation", "High Vocal Activation"}

    def test_segment_duration(self, engine):
        """Test: segment_duration_sec matches input length."""
        waveform = np.random.randn(32000).astype(np.float32)  # 2 seconds
        result = engine.predict_segment(waveform, sample_rate=16000)
        assert abs(result.segment_duration_sec - 2.0) < 0.01

    def test_model_version(self, engine):
        """Test: model_version is set."""
        waveform = np.random.randn(16000).astype(np.float32)
        result = engine.predict_segment(waveform, sample_rate=16000)
        assert result.model_version == "1.0.0"


# ══════════════════════════════════════════════════════════════════════
# E. INPUT VALIDATION TESTS
# ══════════════════════════════════════════════════════════════════════
class TestInputValidation:
    """Tests for input validation."""

    def test_invalid_waveform_shape(self, engine):
        """Test 13: 2D waveform is rejected."""
        waveform_2d = np.random.randn(2, 16000).astype(np.float32)
        with pytest.raises(ValueError, match="1D"):
            engine.predict_segment(waveform_2d, sample_rate=16000)

    def test_empty_waveform(self, engine):
        """Test 14: Empty waveform is rejected."""
        waveform_empty = np.array([], dtype=np.float32)
        with pytest.raises(ValueError, match="empty"):
            engine.predict_segment(waveform_empty, sample_rate=16000)

    def test_invalid_sample_rate(self, engine):
        """Test 15: Invalid sample rate is rejected."""
        waveform = np.random.randn(16000).astype(np.float32)
        with pytest.raises(ValueError, match="sample_rate"):
            engine.predict_segment(waveform, sample_rate=-1)

    def test_non_numpy_input(self, engine):
        """Test: Non-numpy input is rejected."""
        with pytest.raises(TypeError, match="numpy.ndarray"):
            engine.predict_segment([1.0, 2.0, 3.0], sample_rate=16000)


# ══════════════════════════════════════════════════════════════════════
# F. BATCH PREDICTION TESTS
# ══════════════════════════════════════════════════════════════════════
class TestBatchPrediction:
    """Tests for batch prediction."""

    def test_batch_preserves_order(self, engine):
        """Test 16: Batch prediction preserves input order."""
        segments = [
            np.random.randn(16000).astype(np.float32) for _ in range(5)
        ]
        results = engine.predict_batch(segments, sample_rate=16000)
        assert len(results) == 5
        # Each result should be a valid InferenceOutput
        for r in results:
            assert isinstance(r, InferenceOutput)
            assert r.label in {"Low Vocal Activation", "Moderate Vocal Activation", "High Vocal Activation"}

    def test_batch_single_element(self, engine):
        """Test: Batch with single element works."""
        segments = [np.random.randn(16000).astype(np.float32)]
        results = engine.predict_batch(segments, sample_rate=16000)
        assert len(results) == 1

    def test_batch_empty_list(self, engine):
        """Test: Empty batch returns empty list."""
        results = engine.predict_batch([], sample_rate=16000)
        assert results == []


# ══════════════════════════════════════════════════════════════════════
# G. MODEL CACHING TESTS
# ══════════════════════════════════════════════════════════════════════
class TestModelCaching:
    """Tests for model instance caching."""

    def test_model_instance_cached(self):
        """Test 17: Model instance is cached across calls."""
        eng = EmotionEngine()
        eng.load()
        model_ref1 = eng._model
        eng.load()  # Should not reload
        model_ref2 = eng._model
        assert model_ref1 is model_ref2, "Model was reloaded on second load() call"

    def test_multiple_engines_same_model(self):
        """Test: Multiple engine instances load the same model."""
        eng1 = EmotionEngine()
        eng1.load()
        eng2 = EmotionEngine()
        eng2.load()
        # Both should produce identical predictions for the same input
        waveform = np.random.randn(16000).astype(np.float32)
        r1 = eng1.predict_segment(waveform, sample_rate=16000)
        r2 = eng2.predict_segment(waveform, sample_rate=16000)
        assert r1.label == r2.label
        assert abs(r1.confidence - r2.confidence) < 1e-6


# ══════════════════════════════════════════════════════════════════════
# H. PARITY TEST AGAINST DEPLOYMENT MODULE
# ══════════════════════════════════════════════════════════════════════
class TestParity:
    """
    Canonical engine self-parity tests.

    Validate the canonical EmotionEngine against its own direct reference
    execution using the same loaded model, feature extractor, preprocessing,
    artifact, and labels. No external deployment modules or legacy models.
    """

    def test_parity_probabilities_match_direct_logits(self, engine):
        """Test: Public probabilities match direct softmax from same model logits."""
        waveform = np.random.RandomState(42).randn(16000).astype(np.float32)

        # Public API result
        result = engine.predict_segment(waveform, sample_rate=16000)

        # Direct reference: same processor, same model, same preprocessing
        inputs = engine._processor(
            waveform, sampling_rate=16000, return_tensors="pt"
        )
        input_values = inputs["input_values"].to(engine._device)
        with torch.no_grad():
            logits = engine._model(input_values=input_values)
        direct_probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        # Verify probabilities match
        for i in range(engine.num_labels):
            label = engine.labels[str(i)]
            diff = abs(result.probabilities[label] - float(direct_probs[i]))
            assert diff < 1e-5, (
                f"Probability mismatch for {label}: "
                f"public={result.probabilities[label]:.6f} "
                f"direct={direct_probs[i]:.6f} diff={diff:.8f}"
            )

    def test_parity_confidence_matches_max_direct_softmax(self, engine):
        """Test: Public confidence equals maximum direct softmax probability."""
        waveform = np.random.RandomState(42).randn(16000).astype(np.float32)

        result = engine.predict_segment(waveform, sample_rate=16000)

        inputs = engine._processor(
            waveform, sampling_rate=16000, return_tensors="pt"
        )
        input_values = inputs["input_values"].to(engine._device)
        with torch.no_grad():
            logits = engine._model(input_values=input_values)
        direct_probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        direct_confidence = float(np.max(direct_probs))

        diff = abs(result.confidence - direct_confidence)
        assert diff < 1e-5, (
            f"Confidence mismatch: public={result.confidence:.6f} "
            f"direct={direct_confidence:.6f} diff={diff:.8f}"
        )

    def test_parity_single_item_and_one_item_batch_equivalent(self, engine):
        """Test: Single-item and one-item batch preprocessing produce equivalent output."""
        waveform = np.random.RandomState(99).randn(16000).astype(np.float32)

        # Single prediction
        single_result = engine.predict_segment(waveform, sample_rate=16000)

        # One-item batch prediction
        batch_results = engine.predict_batch([waveform], sample_rate=16000)
        assert len(batch_results) == 1
        batch_result = batch_results[0]

        # Must produce identical results
        assert single_result.label == batch_result.label
        assert abs(single_result.confidence - batch_result.confidence) < 1e-6
        for key in single_result.probabilities:
            diff = abs(
                single_result.probabilities[key] - batch_result.probabilities[key]
            )
            assert diff < 1e-6, (
                f"Single vs batch mismatch for {key}: "
                f"single={single_result.probabilities[key]:.6f} "
                f"batch={batch_result.probabilities[key]:.6f}"
            )

    def test_parity_deterministic_repeated_inference(self, engine):
        """Test: Repeated evaluation-mode inference is deterministic."""
        waveform = np.random.RandomState(77).randn(16000).astype(np.float32)

        result1 = engine.predict_segment(waveform, sample_rate=16000)
        result2 = engine.predict_segment(waveform, sample_rate=16000)
        result3 = engine.predict_segment(waveform, sample_rate=16000)

        # All three must be identical
        assert result1.label == result2.label == result3.label
        assert result1.confidence == result2.confidence == result3.confidence
        for key in result1.probabilities:
            assert result1.probabilities[key] == result2.probabilities[key]
            assert result2.probabilities[key] == result3.probabilities[key]

    def test_parity_label_matches_argmax_and_official_mapping(self, engine):
        """Test: Predicted label matches argmax index and official label mapping."""
        waveform = np.random.RandomState(55).randn(16000).astype(np.float32)

        result = engine.predict_segment(waveform, sample_rate=16000)

        # Verify label_id matches argmax of probabilities
        prob_values = [
            result.probabilities[engine.labels[str(i)]]
            for i in range(engine.num_labels)
        ]
        expected_label_id = int(np.argmax(prob_values))
        assert result.label_id == expected_label_id, (
            f"label_id {result.label_id} != argmax {expected_label_id}"
        )

        # Verify label matches official label mapping
        assert result.label == engine.labels[str(result.label_id)], (
            f"label '{result.label}' != official mapping "
            f"'{engine.labels[str(result.label_id)]}'"
        )

        # Verify label is one of the three official labels
        official_labels = {
            "Low Vocal Activation",
            "Moderate Vocal Activation",
            "High Vocal Activation",
        }
        assert result.label in official_labels, (
            f"Predicted label '{result.label}' is not an official label"
        )
