"""
Shared Emotion Inference Engine.

Provides model loading, caching, and prediction for Arabic SER.
Used by both the Streamlit demo and (Stage 2) the FastAPI deployment.

Architecture: Wav2Vec2-Large-XLSR-53 + BiLSTM (50 hidden) + Linear(100 → 3)
Dataset: BAVED (3 classes: Low Vocal Activation, Moderate Vocal Activation, High Vocal Activation)
"""

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .schemas import InferenceInput, InferenceOutput

logger = logging.getLogger(__name__)

# ── Suppress harmless HF warnings ─────────────────────────────────────
warnings.filterwarnings("ignore", message=".*Failed to load image.*")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# ── Constants ──────────────────────────────────────────────────────────
ENGINE_VERSION = "1.0.0"
DEFAULT_MODEL_SR = 16000

# ── Repository-relative paths ──────────────────────────────────────────
_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_MODEL_PATH = (
    _REPOSITORY_ROOT
    / "artifacts"
    / "models"
    / "vocal_activation"
    / "elgeish_baved_v1"
    / "model_bundle_fp16.pt"
)
_DEFAULT_LABELS_PATH = (
    _REPOSITORY_ROOT
    / "artifacts"
    / "models"
    / "vocal_activation"
    / "elgeish_baved_v1"
    / "labels.json"
)


# ══════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE — Wav2Vec2-Large-XLSR-53 + BiLSTM
# ══════════════════════════════════════════════════════════════════════
class Wav2Vec2BiLSTMForSER(nn.Module):
    """
    Wav2Vec2-Large-XLSR-53 + Bi-LSTM (50 hidden) + Linear(100 → 3).
    Replicates Mohamed & Aly (2021) architecture for Arabic SER.
    """

    def __init__(
        self,
        backbone_name: str,
        num_labels: int,
        lstm_hidden_size: int = 50,
        mask_time_prob: float = 0.03,
        dropout: float = 0.2,
    ):
        super().__init__()
        from transformers import Wav2Vec2Config, Wav2Vec2Model

        config = Wav2Vec2Config.from_pretrained(backbone_name)
        config.mask_time_prob = mask_time_prob
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            backbone_name, config=config, ignore_mismatched_sizes=True
        )
        self.wav2vec2.freeze_feature_encoder()

        hidden_size = config.hidden_size  # 1024 for XLSR-53-Large
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_labels)
        self.num_labels = num_labels

    def forward(self, input_values, attention_mask=None, **kwargs):
        outputs = self.wav2vec2(
            input_values=input_values, attention_mask=attention_mask
        )
        hidden = outputs.last_hidden_state       # [B, T, 1024]
        lstm_out, _ = self.bilstm(hidden)         # [B, T, 100]

        # ── Masked mean pooling ────────────────────────────────────────
        if attention_mask is not None:
            # Derive feature-vector mask from the input-sample mask
            feat_len = hidden.shape[1]
            feat_mask = self.wav2vec2._get_feature_vector_attention_mask(
                feat_len, attention_mask
            )  # (B, T_feat), dtype=torch.long

            # Convert to same device and compatible dtype as lstm_out
            feat_mask = feat_mask.to(
                device=lstm_out.device, dtype=lstm_out.dtype
            )

            # Expand mask over BiLSTM hidden dimension
            valid_mask = feat_mask.unsqueeze(-1)  # (B, T_feat, 1)

            # Sum valid positions only
            summed = (lstm_out * valid_mask).sum(dim=1)  # (B, 100)

            # Count valid positions, clamped to avoid division by zero
            num_valid = valid_mask.sum(dim=1).clamp(min=1.0)  # (B, 1)

            # Mean of valid positions only
            pooled = summed / num_valid  # (B, 100)
        else:
            # Fallback: simple mean pooling when no mask provided
            pooled = lstm_out.mean(dim=1)  # [B, 100]

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)          # [B, 3]
        return logits


# ══════════════════════════════════════════════════════════════════════
# MODEL PATH RESOLUTION
# ══════════════════════════════════════════════════════════════════════
def _resolve_model_path(explicit_path: Optional[str] = None) -> Path:
    """
    Resolve the model file path using priority order:
      1. Explicit constructor/config path
      2. AUDIO_MODEL_PATH environment variable
      3. Repository-relative audio_emotion_package/audio_model_fp16.pt
      4. Raise FileNotFoundError listing all attempted paths

    Precedence: explicit_path overrides env var, which overrides default.
    """
    attempted: List[str] = []

    # Priority 1: Explicit path — if provided, it MUST exist
    if explicit_path is not None:
        p = Path(explicit_path)
        attempted.append(f"explicit path: {p}")
        if p.exists():
            return p
        # Explicit path provided but not found — error immediately
        paths_str = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(attempted))
        raise FileNotFoundError(
            f"Model file not found. Attempted paths:\n{paths_str}"
        )

    # Priority 2: Environment variable
    env_path = os.environ.get("AUDIO_MODEL_PATH")
    if env_path is not None:
        p = Path(env_path)
        attempted.append(f"env AUDIO_MODEL_PATH: {p}")
        if p.exists():
            return p
        # Env var provided but not found — error immediately
        paths_str = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(attempted))
        raise FileNotFoundError(
            f"Model file not found. Attempted paths:\n{paths_str}"
        )

    # Priority 3: Repository-relative default
    attempted.append(f"repository default: {_DEFAULT_MODEL_PATH}")
    if _DEFAULT_MODEL_PATH.exists():
        return _DEFAULT_MODEL_PATH

    # Priority 4: Error
    paths_str = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(attempted))
    raise FileNotFoundError(
        f"Model file not found. Attempted paths:\n{paths_str}"
    )


def _resolve_labels_path() -> Path:
    """Resolve the labels.json path."""
    if _DEFAULT_LABELS_PATH.exists():
        return _DEFAULT_LABELS_PATH
    raise FileNotFoundError(f"Labels file not found: {_DEFAULT_LABELS_PATH}")


# ══════════════════════════════════════════════════════════════════════
# EMOTION ENGINE
# ══════════════════════════════════════════════════════════════════════
class EmotionEngine:
    """
    Shared inference engine for Arabic Speech Emotion Recognition.

    Responsibilities:
      - Model path resolution
      - Model loading and caching
      - Label mapping from labels.json
      - Single-segment prediction
      - Batch prediction
      - Device selection (CUDA / CPU)

    Prohibitions:
      - No Streamlit UI logic
      - No stereo role assignment
      - No alert rules
      - No report generation
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the engine.

        Args:
            model_path: Optional explicit path to the model file.
                       If None, resolves via env var then repository default.
        """
        self._model_path = _resolve_model_path(model_path)
        self._labels = self._load_labels()
        self._label_to_id = {v: k for k, v in self._labels.items()}
        self._device: Optional[torch.device] = None
        self._model: Optional[Wav2Vec2BiLSTMForSER] = None
        self._processor = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether the model is loaded and ready for inference."""
        return self._loaded

    @property
    def device(self) -> str:
        """Current device ('cuda' or 'cpu')."""
        if self._device is None:
            return "uninitialized"
        return str(self._device)

    @property
    def model_path(self) -> Path:
        """Resolved model file path."""
        return self._model_path

    @property
    def labels(self) -> Dict[str, str]:
        """Label mapping: {"0": "Low Vocal Activation", ...}."""
        return dict(self._labels)

    @property
    def label_list(self) -> List[str]:
        """Ordered list of labels: ["Low Vocal Activation", "Moderate Vocal Activation", "High Vocal Activation"]."""
        return [self._labels[str(i)] for i in range(len(self._labels))]

    @property
    def num_labels(self) -> int:
        """Number of emotion classes."""
        return len(self._labels)

    def _load_labels(self) -> Dict[str, str]:
        """Load labels from labels.json."""
        path = _resolve_labels_path()
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load(self) -> bool:
        """
        Load the model and feature extractor.

        Returns True if successful.
        Raises RuntimeError on architecture or weight mismatch.
        """
        if self._loaded:
            return True

        try:
            from transformers import Wav2Vec2FeatureExtractor

            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            logger.info(
                f"Loading model from {self._model_path} on {self._device}..."
            )

            # Load checkpoint
            bundle = torch.load(
                self._model_path,
                map_location=self._device,
                weights_only=False,
            )

            # Validate checkpoint structure
            if "config" not in bundle:
                raise RuntimeError(
                    f"Checkpoint missing 'config' key. "
                    f"Available keys: {list(bundle.keys())}"
                )
            if "state_dict" not in bundle:
                raise RuntimeError(
                    f"Checkpoint missing 'state_dict' key. "
                    f"Available keys: {list(bundle.keys())}"
                )

            config = bundle["config"]

            # Build model with saved config
            self._model = Wav2Vec2BiLSTMForSER(
                backbone_name=config["backbone_name"],
                num_labels=config["num_labels"],
                lstm_hidden_size=config["lstm_hidden_size"],
                mask_time_prob=config["mask_time_prob"],
                dropout=config["dropout"],
            )

            # Load weights with strict checking
            self._model.load_state_dict(bundle["state_dict"], strict=True)

            self._model.to(self._device)
            self._model.eval()

            # Load feature extractor
            self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
                config["backbone_name"]
            )

            self._loaded = True
            logger.info(f"Model loaded successfully on {self._device}")
            return True

        except Exception as e:
            self._loaded = False
            self._model = None
            self._processor = None
            raise RuntimeError(f"Failed to load model: {e}") from e

    def predict_segment(
        self,
        waveform: np.ndarray,
        sample_rate: int = DEFAULT_MODEL_SR,
    ) -> InferenceOutput:
        """
        Predict emotion for a single audio segment.

        Args:
            waveform: 1D float32 numpy array of audio samples.
            sample_rate: Sample rate of the input (must be 16000).

        Returns:
            InferenceOutput with label, probabilities, confidence, etc.

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If input is invalid.
        """
        if not self._loaded:
            raise RuntimeError(
                "Model not loaded. Call engine.load() first."
            )

        # Validate input
        inp = InferenceInput(waveform=waveform, sample_rate=sample_rate)

        # Run inference
        probs = self._predict_probs(inp.waveform, inp.sample_rate)

        # Build output
        label_id = int(np.argmax(probs))
        label = self._labels[str(label_id)]
        confidence = float(probs[label_id])

        prob_dict = {
            self._labels[str(i)]: round(float(probs[i]), 6)
            for i in range(len(self._labels))
        }

        duration_sec = round(len(inp.waveform) / inp.sample_rate, 4)

        return InferenceOutput(
            label=label,
            label_id=label_id,
            probabilities=prob_dict,
            confidence=round(confidence, 6),
            model_version=ENGINE_VERSION,
            device=self.device,
            segment_duration_sec=duration_sec,
        )

    def predict_batch(
        self,
        segments: List[np.ndarray],
        sample_rate: int = DEFAULT_MODEL_SR,
    ) -> List[InferenceOutput]:
        """
        Predict emotion for multiple audio segments.

        Args:
            segments: List of 1D float32 numpy arrays.
            sample_rate: Sample rate (must be 16000).

        Returns:
            List of InferenceOutput, one per segment, in input order.

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If any input is invalid.
        """
        if not self._loaded:
            raise RuntimeError(
                "Model not loaded. Call engine.load() first."
            )

        results = []
        for i, seg in enumerate(segments):
            try:
                result = self.predict_segment(seg, sample_rate)
                results.append(result)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Invalid segment at index {i}: {e}"
                ) from e

        return results

    def _predict_probs(
        self, waveform: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """
        Run raw inference on a waveform segment.
        Returns probabilities array of shape (num_labels,).
        """
        inputs = self._processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self._device)

        with torch.no_grad():
            logits = self._model(input_values=input_values)

        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        return probs

    def get_model_info(self) -> Dict:
        """Return model metadata for inspection."""
        if not self._loaded:
            return {"loaded": False}

        config = None
        try:
            bundle = torch.load(
                self._model_path,
                map_location="cpu",
                weights_only=False,
            )
            config = bundle.get("config", {})
        except Exception:
            pass

        return {
            "loaded": True,
            "model_path": str(self._model_path),
            "device": self.device,
            "engine_version": ENGINE_VERSION,
            "num_labels": self.num_labels,
            "labels": self.labels,
            "backbone": config.get("backbone_name", "unknown") if config else "unknown",
            "architecture": "Wav2Vec2BiLSTMForSER",
            "model_config": config,
        }
