"""
Per-channel SER processing with shared timeline alignment.

Processes Agent and Customer channels through the emotion model
using shared windows. Skips inference for inactive windows.
"""

import logging
import time
from typing import Optional

import numpy as np

from src.inference.engine import EmotionEngine

from .schemas import (
    StereoLoadResult,
    WindowConfig,
    SpeechActivityConfig,
    SharedWindow,
    SpeechActivityResult,
    SpeakerWindowResult,
    SharedWindowResult,
    SpeakerProcessingResult,
    StereoProcessingResult,
)
from .temporal_analysis import create_shared_windows, analyze_speech_activity

logger = logging.getLogger(__name__)


def process_stereo_call(
    load_result: StereoLoadResult,
    engine: EmotionEngine,
    window_config: WindowConfig,
    speech_config: SpeechActivityConfig,
    minimum_confidence: Optional[float] = None,
) -> StereoProcessingResult:
    """
    Process a stereo call through the full Phase III pipeline.

    Flow:
        1. Create shared windows
        2. Measure speech activity per channel per window
        3. Run SER on active channel-windows only
        4. Preserve all windows including inactive ones

    Does NOT:
        - Compute alerts
        - Compute satisfaction or skill scores
        - Create UI
        - Write files to disk
    """
    if not engine.is_loaded:
        raise RuntimeError("EmotionEngine not loaded. Call engine.load() first.")

    start_time = time.time()

    model_audio = load_result.model_audio
    agent_waveform = model_audio.agent_waveform
    customer_waveform = model_audio.customer_waveform
    duration_sec = model_audio.duration_sec
    model_sr = model_audio.model_sr
    original_sr = load_result.original_audio.original_sr

    # ── Step 1: Create shared windows ──────────────────────────────────
    windows = create_shared_windows(
        duration_sec=duration_sec,
        window_duration_sec=window_config.window_duration_sec,
        hop_duration_sec=window_config.hop_duration_sec,
        model_sr=model_sr,
        original_sr=original_sr,
    )

    # ── Step 2: Measure speech activity per channel ────────────────────
    agent_activity = analyze_speech_activity(
        waveform=agent_waveform,
        windows=windows,
        sample_rate=model_sr,
        config=speech_config,
    )
    customer_activity = analyze_speech_activity(
        waveform=customer_waveform,
        windows=windows,
        sample_rate=model_sr,
        config=speech_config,
    )

    # ── Step 3: Process each window ────────────────────────────────────
    window_results = []
    inference_calls = 0
    inference_skipped = 0

    for i, window in enumerate(windows):
        agent_act = agent_activity[i]
        customer_act = customer_activity[i]

        # Agent processing
        agent_result = _process_speaker_window(
            engine=engine,
            waveform=agent_waveform,
            window=window,
            activity=agent_act,
            channel="left",
            role="Agent",
            model_sr=model_sr,
            minimum_confidence=minimum_confidence,
        )

        # Customer processing
        customer_result = _process_speaker_window(
            engine=engine,
            waveform=customer_waveform,
            window=window,
            activity=customer_act,
            channel="right",
            role="Customer",
            model_sr=model_sr,
            minimum_confidence=minimum_confidence,
        )

        # Count inference calls
        if agent_result.prediction is not None:
            inference_calls += 1
        else:
            inference_skipped += 1
        if customer_result.prediction is not None:
            inference_calls += 1
        else:
            inference_skipped += 1

        window_results.append(SharedWindowResult(
            window=window,
            agent=agent_result,
            customer=customer_result,
        ))

    # ── Step 4: Build results ──────────────────────────────────────────
    processing_time = time.time() - start_time

    agent_processing = SpeakerProcessingResult(
        channel="left",
        role="Agent",
        role_assignment_method="channel_convention",
        windows=window_results,
    )
    customer_processing = SpeakerProcessingResult(
        channel="right",
        role="Customer",
        role_assignment_method="channel_convention",
        windows=window_results,
    )

    return StereoProcessingResult(
        role_assignment_method="channel_convention",
        duration_sec=duration_sec,
        window_config=window_config,
        speech_config=speech_config,
        shared_windows=windows,
        agent_result=agent_processing,
        customer_result=customer_processing,
        window_results=window_results,
        processing_metadata={
            "inference_calls": inference_calls,
            "inference_skipped": inference_skipped,
            "total_windows": len(windows),
            "processing_time_sec": round(processing_time, 4),
        },
        model_version=engine.get_model_info().get("engine_version", "unknown"),
        device=engine.device,
    )


def _process_speaker_window(
    engine: EmotionEngine,
    waveform: np.ndarray,
    window: SharedWindow,
    activity: SpeechActivityResult,
    channel: str,
    role: str,
    model_sr: int,
    minimum_confidence: Optional[float],
) -> SpeakerWindowResult:
    """Process one speaker in one window."""
    prediction = None
    uncertainty_reason = None

    if activity.speech_active:
        # Extract segment for inference
        segment = waveform[window.model_start_sample:window.model_end_sample]

        if len(segment) > 0 and np.all(np.isfinite(segment)):
            try:
                result = engine.predict_segment(segment, sample_rate=model_sr)
                prediction = result

                # Low-confidence handling
                if (minimum_confidence is not None and
                        result.confidence < minimum_confidence):
                    uncertainty_reason = "below_minimum_confidence"
            except Exception as e:
                logger.warning(
                    f"Inference failed for {role} window {window.window_id}: {e}"
                )

    return SpeakerWindowResult(
        channel=channel,
        role=role,
        role_assignment_method="channel_convention",
        speech_active=activity.speech_active,
        active_ratio=activity.active_ratio,
        rms=activity.rms,
        prediction=prediction,
        inactive_reason=activity.inactive_reason,
        uncertainty_reason=uncertainty_reason,
    )
