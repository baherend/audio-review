"""
Data schemas for the shared inference engine.

Defines typed input/output structures for emotion prediction.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class InferenceInput:
    """Input to the inference engine."""

    waveform: "np.ndarray"  # 1D float32 waveform
    sample_rate: int = 16000

    def __post_init__(self):
        import numpy as np

        if not isinstance(self.waveform, np.ndarray):
            raise TypeError(
                f"waveform must be numpy.ndarray, got {type(self.waveform).__name__}"
            )
        if self.waveform.ndim != 1:
            raise ValueError(
                f"waveform must be 1D, got shape {self.waveform.shape}"
            )
        if self.waveform.size == 0:
            raise ValueError("waveform must not be empty")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")


@dataclass
class ChunkPrediction:
    """Prediction for a single audio chunk."""

    label: str  # e.g., "Low Vocal Activation"
    label_id: int  # e.g., 0
    probabilities: Dict[str, float]  # {"Low Vocal Activation": 0.12, ...}
    confidence: float  # max probability


@dataclass
class InferenceOutput:
    """Complete output from the inference engine for one segment."""

    label: str  # dominant emotion label
    label_id: int  # dominant label index
    probabilities: Dict[str, float]  # all class probabilities
    confidence: float  # max probability
    model_version: str  # engine version string
    device: str  # "cuda" or "cpu"
    segment_duration_sec: float  # duration of input waveform in seconds
