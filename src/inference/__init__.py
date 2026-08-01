"""
Shared inference engine for Arabic Speech Emotion Recognition.

Provides model loading, prediction, and label mapping used by
both the Streamlit demo and (Stage 2) the FastAPI deployment.
"""

from .engine import EmotionEngine
from .schemas import InferenceInput, InferenceOutput, ChunkPrediction

__all__ = ["EmotionEngine", "InferenceInput", "InferenceOutput", "ChunkPrediction"]
