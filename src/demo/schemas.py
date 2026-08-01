"""
Data schemas for the stereo demo.

Defines typed structures for audio loading, validation, and channel separation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════════
# VALIDATION CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
@dataclass
class StereoValidationConfig:
    """
    Thresholds for stereo input quality validation.

    All values are technical input-quality thresholds, NOT alert thresholds.
    They control whether the input audio is acceptable for processing.
    """

    # Channel identity check (sample-level comparison)
    identical_sample_tolerance: float = 1e-6
    """Maximum mean absolute sample difference for channels to be identical.
    Default 1e-6: channels are sample-for-sample equal within floating-point tolerance."""

    # Correlation check (separate from identity)
    high_correlation_threshold: float = 0.95
    """Pearson correlation above which channels are warned as highly correlated.
    Default 0.95: channels are very similar but not identical."""

    # Energy checks
    silent_rms_threshold: float = 1e-4
    """RMS below which a channel is considered silent.
    Default 1e-4: very low amplitude, effectively silence."""

    low_energy_threshold: float = 0.005
    """RMS below which overall audio is considered low energy.
    Default 0.005: quiet but potentially valid."""

    # Clipping checks
    clipping_amplitude: float = 0.99
    """Amplitude at or above which samples are considered clipped.
    Default 0.99: near the digital maximum of [-1, 1]."""

    clipping_ratio_threshold: float = 0.01
    """Fraction of clipped samples above which clipping is warned.
    Default 0.01: more than 1% of samples clipped."""

    # Duration checks
    minimum_duration_sec: float = 0.5
    """Minimum audio duration in seconds.
    Default 0.5: very short calls may produce unreliable results."""

    # Speech activity detection
    speech_rms_threshold_db: float = -40.0
    """RMS threshold in dB below peak for speech-active frame detection.
    Default -40 dB: frames quieter than this are considered non-speech."""

    speech_active_ratio: float = 0.3
    """Minimum fraction of frames that must be active for a window to be speech-active.
    Default 0.3: at least 30% of frames must contain speech."""


# ══════════════════════════════════════════════════════════════════════
# VALIDATION RESULT
# ══════════════════════════════════════════════════════════════════════
@dataclass
class ValidationError:
    """A single validation error (rejects the input)."""

    code: str  # e.g., "MONO_INPUT", "MORE_THAN_TWO_CHANNELS"
    message: str


@dataclass
class ValidationWarning:
    """A single validation warning (input accepted with caveat)."""

    code: str  # e.g., "IDENTICAL_CHANNELS", "HIGH_CORRELATION"
    message: str


@dataclass
class ValidationMetrics:
    """Quantitative metrics computed during validation."""

    channel_count: int = 0
    samples_per_channel: int = 0
    sample_rate: int = 0
    duration_sec: float = 0.0
    agent_rms: float = 0.0
    customer_rms: float = 0.0
    channel_correlation: float = 0.0
    agent_clipping_ratio: float = 0.0
    customer_clipping_ratio: float = 0.0


@dataclass
class ValidationResult:
    """Complete validation result for stereo input."""

    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationWarning] = field(default_factory=list)
    metrics: ValidationMetrics = field(default_factory=ValidationMetrics)


# ══════════════════════════════════════════════════════════════════════
# AUDIO REPRESENTATIONS
# ══════════════════════════════════════════════════════════════════════
@dataclass
class OriginalStereoAudio:
    """
    Original decoded stereo audio.

    Used for full-call playback and risky-segment extraction.
    Never trimmed, never resampled, never compressed in time.
    """

    waveform: np.ndarray  # shape (2, N_original)
    original_sr: int
    duration_sec: float
    filename: str
    format: str  # e.g., "wav", "mp3", "flac"

    def __post_init__(self):
        if self.waveform.ndim != 2:
            raise ValueError(
                f"waveform must be 2D (channels, samples), got shape {self.waveform.shape}"
            )
        if self.waveform.shape[0] != 2:
            raise ValueError(
                f"waveform must have exactly 2 channels, got {self.waveform.shape[0]}"
            )


@dataclass
class ModelStereoAudio:
    """
    Stereo audio resampled to 16 kHz for SER inference.

    Left channel = Agent, Right channel = Customer.
    Duration in seconds matches OriginalStereoAudio within tolerance.
    """

    agent_waveform: np.ndarray   # shape (N_model,) — left channel
    customer_waveform: np.ndarray  # shape (N_model,) — right channel
    model_sr: int = 16000
    duration_sec: float = 0.0

    def __post_init__(self):
        if self.agent_waveform.ndim != 1:
            raise ValueError(
                f"agent_waveform must be 1D, got shape {self.agent_waveform.shape}"
            )
        if self.customer_waveform.ndim != 1:
            raise ValueError(
                f"customer_waveform must be 1D, got shape {self.customer_waveform.shape}"
            )
        if len(self.agent_waveform) != len(self.customer_waveform):
            raise ValueError(
                f"Channel lengths must match: agent={len(self.agent_waveform)}, "
                f"customer={len(self.customer_waveform)}"
            )


@dataclass
class StereoLoadResult:
    """
    Complete result from stereo loading and validation.

    Contains original audio, model audio, validation, and role assignment.
    """

    original_audio: OriginalStereoAudio
    model_audio: ModelStereoAudio
    validation: ValidationResult
    role_assignment_method: str = "channel_convention"


# ══════════════════════════════════════════════════════════════════════
# WINDOW AND SPEECH-ACTIVITY CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
@dataclass
class WindowConfig:
    """Configuration for shared analysis windows."""

    window_duration_sec: float = 2.0
    """Duration of each analysis window in seconds.
    Default 2.0: matches the model's trained chunk duration."""

    hop_duration_sec: float = 1.0
    """Hop (stride) between windows in seconds.
    Default 1.0: 50% overlap between consecutive windows."""


@dataclass
class SpeechActivityConfig:
    """Configuration for RMS-based speech-activity detection."""

    rms_threshold_db: float = -40.0
    """RMS threshold in dB below peak for active frame detection.
    Default -40 dB: frames quieter than this are non-speech."""

    active_ratio_threshold: float = 0.3
    """Minimum fraction of active frames for a window to be speech-active.
    Default 0.3: at least 30% of frames must contain speech."""

    frame_length: int = 2048
    """RMS frame length in samples.
    Default 2048: standard for 16 kHz audio."""

    hop_length: int = 512
    """RMS hop length in samples.
    Default 512: standard for 16 kHz audio."""


# ══════════════════════════════════════════════════════════════════════
# SHARED WINDOW
# ══════════════════════════════════════════════════════════════════════
@dataclass
class SharedWindow:
    """A single shared analysis window aligned to the original timeline."""

    window_id: int
    original_start: float  # seconds from call start
    original_end: float    # seconds from call start
    model_start_sample: int  # sample index in 16 kHz model waveform
    model_end_sample: int    # sample index in 16 kHz model waveform
    original_start_sample: Optional[int] = None  # sample index in original waveform
    original_end_sample: Optional[int] = None    # sample index in original waveform


# ══════════════════════════════════════════════════════════════════════
# SPEECH-ACTIVITY RESULT
# ══════════════════════════════════════════════════════════════════════
@dataclass
class SpeechActivityResult:
    """Speech-activity measurement for one channel in one window."""

    window_id: int
    speech_active: bool
    active_ratio: float  # fraction of active frames
    rms: float  # RMS energy of the segment
    inactive_reason: Optional[str] = None
    # "silence" | "insufficient_active_ratio" | "insufficient_duration" |
    # "non_finite_audio" | "empty_segment" | None


# ══════════════════════════════════════════════════════════════════════
# SPEAKER WINDOW RESULT
# ══════════════════════════════════════════════════════════════════════
@dataclass
class SpeakerWindowResult:
    """Result for one speaker (Agent or Customer) in one shared window."""

    channel: str  # "left" | "right"
    role: str  # "Agent" | "Customer"
    role_assignment_method: str = "channel_convention"
    speech_active: bool = False
    active_ratio: float = 0.0
    rms: float = 0.0
    prediction: Optional["InferenceOutput"] = None
    inactive_reason: Optional[str] = None
    uncertainty_reason: Optional[str] = None
    # uncertainty_reason: "below_minimum_confidence" | None


# ══════════════════════════════════════════════════════════════════════
# SHARED WINDOW RESULT
# ══════════════════════════════════════════════════════════════════════
@dataclass
class SharedWindowResult:
    """Combined result for one shared window across both speakers."""

    window: SharedWindow
    agent: SpeakerWindowResult
    customer: SpeakerWindowResult

    @property
    def both_active_in_window(self) -> bool:
        """Both channels contain sufficient speech in this window."""
        return self.agent.speech_active and self.customer.speech_active

    @property
    def neither_active_in_window(self) -> bool:
        """Neither channel contains sufficient speech in this window."""
        return not self.agent.speech_active and not self.customer.speech_active


# ══════════════════════════════════════════════════════════════════════
# SPEAKER PROCESSING RESULT
# ══════════════════════════════════════════════════════════════════════
@dataclass
class SpeakerProcessingResult:
    """Complete result for one speaker across all windows."""

    channel: str  # "left" | "right"
    role: str  # "Agent" | "Customer"
    role_assignment_method: str = "channel_convention"
    windows: List[SharedWindowResult] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════
# STEREO PROCESSING RESULT
# ══════════════════════════════════════════════════════════════════════
@dataclass
class StereoProcessingResult:
    """Complete result from processing a stereo call through the pipeline."""

    role_assignment_method: str = "channel_convention"
    duration_sec: float = 0.0
    window_config: Optional[WindowConfig] = None
    speech_config: Optional[SpeechActivityConfig] = None
    shared_windows: List[SharedWindow] = field(default_factory=list)
    agent_result: Optional[SpeakerProcessingResult] = None
    customer_result: Optional[SpeakerProcessingResult] = None
    window_results: List[SharedWindowResult] = field(default_factory=list)
    processing_metadata: Dict = field(default_factory=dict)
    model_version: str = ""
    device: str = ""


# ══════════════════════════════════════════════════════════════════════
# SPEAKER SUMMARY
# ══════════════════════════════════════════════════════════════════════
@dataclass
class SpeakerSummary:
    """Per-speaker summary derived from processing results."""

    role: str  # "Agent" | "Customer"
    channel: str  # "left" | "right"
    role_assignment_method: str = "channel_convention"

    # Window counts
    total_windows: int = 0
    active_windows: int = 0
    inactive_windows: int = 0
    analyzed_windows: int = 0  # windows with a prediction
    uncertain_windows: int = 0  # windows with prediction below minimum_confidence

    # Duration metrics (non-overlapping, hop-based)
    total_call_duration_sec: float = 0.0
    detected_active_duration_sec: float = 0.0
    detected_inactive_duration_sec: float = 0.0

    # Label distribution (analyzed active windows only)
    dominant_dataset_label: Optional[str] = None
    dataset_label_distribution: Dict[str, int] = field(default_factory=dict)
    dataset_label_percentages: Dict[str, float] = field(default_factory=dict)

    # Confidence
    average_confidence: Optional[float] = None
    minimum_confidence: Optional[float] = None
    maximum_confidence: Optional[float] = None

    # Timeline
    label_transition_count: int = 0
    first_active_time: Optional[float] = None
    last_active_time: Optional[float] = None
    first_active_label: Optional[str] = None
    last_active_label: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════
# EMOTION TIMELINE ENTRY
# ══════════════════════════════════════════════════════════════════════
@dataclass
class EmotionTimelineEntry:
    """Single entry in a per-speaker emotion timeline."""

    window_id: int
    original_start: float
    original_end: float
    speech_active: bool
    dataset_label: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None
    uncertainty_reason: Optional[str] = None
    inactive_reason: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════
# ACTIVITY STATE
# ══════════════════════════════════════════════════════════════════════
@dataclass
class ActivityStateEntry:
    """Activity state for one base interval in the shared timeline."""

    original_start: float
    original_end: float
    state: str  # "agent_active_only" | "customer_active_only" |
                # "both_active_in_window" | "neither_active" | "uncertain_activity"
    agent_active: bool
    customer_active: bool
    agent_active_ratio: float
    customer_active_ratio: float
    source_window_ids: List[int] = field(default_factory=list)


@dataclass
class ActivityInterval:
    """Merged non-overlapping activity interval."""

    start_time: float
    end_time: float
    duration_sec: float
    state: str
    agent_active: bool
    customer_active: bool
    source_window_ids: List[int] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════
# CALL SUMMARY
# ══════════════════════════════════════════════════════════════════════
@dataclass
class CallSummary:
    """Call-level summary combining both speakers and activity analysis."""

    call_duration_sec: float = 0.0
    total_shared_windows: int = 0

    agent_summary: Optional[SpeakerSummary] = None
    customer_summary: Optional[SpeakerSummary] = None

    # Activity state counts
    agent_active_only_count: int = 0
    customer_active_only_count: int = 0
    both_active_count: int = 0
    neither_active_count: int = 0
    uncertain_count: int = 0

    # Non-overlapping activity durations
    agent_active_only_duration_sec: float = 0.0
    customer_active_only_duration_sec: float = 0.0
    both_active_duration_sec: float = 0.0
    neither_active_duration_sec: float = 0.0

    # Inactivity statistics (NOT Hold detection — Hold is not implemented)
    longest_agent_inactive_interval: float = 0.0
    longest_customer_inactive_interval: float = 0.0
    longest_neither_active_interval: float = 0.0
    number_of_agent_inactive_intervals: int = 0
    number_of_customer_inactive_intervals: int = 0
    number_of_neither_active_intervals: int = 0
    activity_state_intervals: List[ActivityInterval] = field(default_factory=list)

    # Metadata
    model_version: str = ""
    role_assignment_method: str = "channel_convention"
    processing_timestamp: str = ""


# ══════════════════════════════════════════════════════════════════════
# HOLD DETECTION CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
@dataclass
class HoldDetectionConfig:
    """Configuration for Hold candidate detection.

    Technical detection thresholds. All values are prototype defaults
    requiring validation on real call-center data.
    """

    # Duration thresholds
    minimum_hold_candidate_duration_sec: float = 10.0
    """Minimum duration for an interval to be a Hold candidate.
    Default 10.0: very short pauses are not Hold."""

    maximum_short_pause_duration_sec: float = 3.0
    """Intervals shorter than this are classified as short_pause.
    Default 3.0: normal turn-taking pauses."""

    agent_return_confirmation_duration_sec: float = 2.0
    """Agent must be active for this duration to confirm return.
    Default 2.0: avoids closing Hold on a brief utterance."""

    merge_gap_between_candidates_sec: float = 5.0
    """Gap below which adjacent candidates merge into one Hold event.
    Default 5.0: short interruptions within a Hold."""

    # Evidence thresholds
    minimum_non_speech_evidence_ratio: float = 0.3
    """Minimum ratio of non-speech frames for non-speech evidence.
    Default 0.3."""

    # Tonal detection thresholds
    # Tonal signals have LOW spectral flatness (energy concentrated in
    # narrow frequency bands), HIGH tonal consistency, and stable
    # dominant frequency across frames.
    maximum_tonal_spectral_flatness: float = 0.1
    """Maximum spectral flatness for tonal detection.
    Default 0.1: tonal signals have concentrated energy (low flatness).
    Broadband noise has flatness near 1.0 and is NOT tonal."""

    minimum_tonal_consistency: float = 0.6
    """Minimum harmonic ratio for tonal consistency.
    Default 0.6: tonal signals maintain harmonic structure."""

    minimum_dominant_freq_stability: float = 0.5
    """Minimum dominant-frequency stability across frames.
    Default 0.5: tonal signals have a stable fundamental frequency.
    Coefficient of variation of dominant frequency must be < 0.15."""

    minimum_tonal_spectral_bandwidth: float = 100.0
    """Minimum spectral bandwidth for tonal detection.
    Default 100.0 Hz: pure sine waves have near-zero bandwidth (< 50 Hz)
    and are NOT classified as tonal/hold-music. Real tonal signals
    (multi-tone, harmonics, hold music) have wider bandwidth > 100 Hz.
    This prevents single-frequency test signals from being misclassified."""

    minimum_recorded_message_candidate_ratio: float = 0.5
    """Minimum repetition consistency for recorded-message candidate.
    Default 0.5."""

    # Confidence
    minimum_confidence: float = 0.3
    """Minimum confidence to generate a Hold candidate.
    Default 0.3."""

    # Behavior
    allow_silence_only_candidate: bool = True
    """Whether prolonged silence alone can generate an uncertain candidate.
    Default True."""

    require_agent_return_to_close_hold: bool = False
    """If True, Hold can only end when Agent returns.
    If False, Hold can also end when non-conversational evidence ends.
    Default False."""


# ══════════════════════════════════════════════════════════════════════
# HOLD POLICY CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
@dataclass
class HoldPolicyConfig:
    """WE-inspired configurable prototype policy.

    Based on project-owner operational experience at WE.
    Not an official WE policy, not legally validated, not a universal standard.
    """

    max_hold_count: int = 2
    """Maximum allowed Hold events per call.
    Approved default: 2."""

    max_hold_duration_sec: float = 120.0
    """Maximum allowed duration of each Hold event in seconds.
    Approved default: 120.0."""

    counting_mode: str = "count_likely_and_confirmed"
    """How Hold events are counted toward policy limits.
    Options: "count_likely_and_confirmed", "count_confirmed_only", "count_all_candidates".
    Default: "count_likely_and_confirmed"."""

    policy_source: str = "WE-inspired configurable prototype policy based on project-owner operational experience"
    """Description of the policy origin."""

    policy_version: str = "prototype_v1.0"
    """Version identifier for the policy configuration."""


# ══════════════════════════════════════════════════════════════════════
# HOLD CANDIDATE
# ══════════════════════════════════════════════════════════════════════
@dataclass
class HoldCandidate:
    """A detected Hold candidate with evidence and classification."""

    candidate_id: str
    start_time: float
    end_time: float
    duration_sec: float
    classification: str  # "uncertain_hold_candidate" | "likely_hold_candidate" |
                         # "confirmed_by_metadata_or_manual_input"
    evidence_types: List[str]  # e.g., ["prolonged_silence", "music_like_audio"]
    agent_inactive: bool
    customer_state: str  # "inactive" | "speech_like" | "music_like" | "recorded_message_candidate" | "uncertain"
    confidence: float
    detection_notes: str
    agent_return_detected: bool
    reaches_call_end: bool = False
    source_interval_ids: List[int] = field(default_factory=list)
    manual_status: str = "unreviewed"  # "unreviewed" | "confirmed_hold" | "rejected_hold"


# ══════════════════════════════════════════════════════════════════════
# HOLD DETECTION RESULT
# ══════════════════════════════════════════════════════════════════════
@dataclass
class HoldDetectionResult:
    """Complete result from Hold candidate detection."""

    candidates: List[HoldCandidate] = field(default_factory=list)
    call_duration_sec: float = 0.0
    detection_config: Optional[HoldDetectionConfig] = None
    total_candidates: int = 0
    uncertain_count: int = 0
    likely_count: int = 0


# ══════════════════════════════════════════════════════════════════════
# HOLD POLICY
# ══════════════════════════════════════════════════════════════════════
@dataclass
class HoldDurationViolation:
    """A single duration limit violation."""

    candidate_id: str
    start_time: float
    end_time: float
    actual_duration_sec: float
    allowed_duration_sec: float
    excess_duration_sec: float


@dataclass
class UncountedOverDuration:
    """A candidate excluded from counting but exceeding the duration threshold."""

    candidate_id: str
    start_time: float
    end_time: float
    duration_sec: float
    classification: str
    exceeds_duration_by_sec: float
    reaches_call_end: bool


@dataclass
class HoldPolicyResult:
    """Result from WE-inspired Hold policy evaluation."""

    counted_hold_events: int = 0
    total_candidate_events: int = 0
    count_limit: int = 2
    count_exceeded: bool = False
    duration_limit_sec: float = 120.0
    duration_violations: List[HoldDurationViolation] = field(default_factory=list)
    uncounted_over_duration: List[UncountedOverDuration] = field(default_factory=list)
    counted_hold_ranges: List[Tuple[float, float]] = field(default_factory=list)
    """Time ranges (start, end) of all counted Hold events, including
    those within the duration limit. Used by the operational alert engine
    to suppress Dead Air alerts that are actually Hold intervals."""
    compliant: bool = True
    review_required: bool = False
    policy_source: str = ""
    policy_version: str = ""
    review_reason: str = ""


# ══════════════════════════════════════════════════════════════════════
# PHASE VI — OPERATIONAL ALERTS
# ══════════════════════════════════════════════════════════════════════

@dataclass
class OperationalAlertConfig:
    """Configuration for audio-based operational review alerts.

    All thresholds are configurable prototype operational guidelines
    based on project-owner call-center experience.
    NOT official company policies. NOT Pass/Fail rules.
    """

    # Dead Air thresholds (both channels inactive)
    dead_air_warning_sec: float = 45.0
    """Both inactive for this duration → warning severity.
    Default 45.0 seconds. Separate from Hold Detection."""

    dead_air_high_sec: float = 60.0
    """Both inactive for this duration → high severity.
    Default 60.0 seconds. Configurable escalation threshold."""

    # Hold overlap suppression
    hold_suppression_overlap_ratio: float = 0.90
    """Minimum overlap ratio between a Dead Air event and a Hold-related
    event for the Dead Air to be suppressed. Default 0.90 (90%).
    When a Dead Air event overlaps ≥90% with a Hold event, the Dead Air
    is suppressed because the Hold alert already explains the interval."""

    # Extended Overlap threshold
    extended_overlap_review_sec: float = 10.0
    """Both active for this duration → review.
    Default 10.0 seconds."""

    # Call duration expected ranges (seconds)
    technical_call_min_sec: float = 300.0
    technical_call_max_sec: float = 600.0
    non_technical_call_min_sec: float = 60.0
    non_technical_call_max_sec: float = 180.0


@dataclass
class OperationalAlert:
    """A single operational review alert."""

    alert_id: str
    alert_type: str  # "dead_air" | "extended_overlap" | "call_duration" |
                     # "hold_duration_violation" | "hold_count_violation" |
                     # "hold_uncounted_over_duration"
    severity: str    # "info" | "warning" | "high"
    requires_manual_review: bool = False
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_sec: Optional[float] = None
    title: str = ""
    explanation: str = ""
    review_recommendation: str = ""
    confidence: Optional[float] = None
    source_category: str = ""  # "silence" | "overlap" | "duration" | "hold"
    metadata: Dict = field(default_factory=dict)


@dataclass
class CallDurationResult:
    """Result of call-duration range evaluation."""

    call_type: str  # "technical" | "non_technical" | "unknown"
    call_duration_sec: float = 0.0
    expected_min_sec: Optional[float] = None
    expected_max_sec: Optional[float] = None
    comparison: str = "not_evaluated"  # "below_expected_range" |
                                       # "within_expected_range" |
                                       # "above_expected_range" |
                                       # "not_evaluated"


@dataclass
class OperationalAlertResult:
    """Complete result from operational alert generation."""

    alerts: List[OperationalAlert] = field(default_factory=list)
    alerts_by_category: Dict[str, List[OperationalAlert]] = field(default_factory=dict)
    alert_counts_by_severity: Dict[str, int] = field(default_factory=dict)
    manual_review_suggested: bool = False
    call_duration_result: Optional[CallDurationResult] = None
    configuration_snapshot: Dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════
# PHASE VII — FINAL CALL REVIEW
# ══════════════════════════════════════════════════════════════════════

@dataclass
class FinalCallReviewConfig:
    """Configuration for Final Call Review construction."""

    priority_severities: List[str] = field(default_factory=lambda: ["high"])
    """Severities that trigger priority_review status."""

    include_info_alerts: bool = True
    """Whether to include info-severity alerts in the review."""

    include_uncertain_hold_candidates: bool = True
    """Whether to include uncertain Hold candidates."""

    max_timeline_items: int = 50
    """Maximum number of timeline items."""

    max_review_reasons: int = 50
    """Maximum number of review reasons."""

    include_configuration_snapshot: bool = True
    """Whether to include configuration snapshots in sections."""


@dataclass
class ReviewReason:
    """A single deduplicated review reason."""

    reason_code: str
    title: str
    explanation: str
    severity: str  # "info" | "warning" | "high"
    source_category: str  # "hold" | "silence" | "overlap" | "duration"
    related_alert_ids: List[str] = field(default_factory=list)
    related_candidate_ids: List[str] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_sec: Optional[float] = None
    evidence_level: str = ""  # "audio_derived" | "policy_threshold" | "model_prediction"
    metadata: Dict = field(default_factory=dict)


@dataclass
class HoldReviewCandidate:
    """Hold candidate details for the review."""

    candidate_id: str
    start_time: float
    end_time: float
    duration_sec: float
    classification: str
    evidence_types: List[str] = field(default_factory=list)
    reaches_call_end: bool = False
    counted_by_policy: bool = False
    related_violation_id: Optional[str] = None
    related_alert_id: Optional[str] = None


@dataclass
class HoldReviewSection:
    """Hold review section of the Final Call Review."""

    total_candidates: int = 0
    likely_candidates: int = 0
    uncertain_candidates: int = 0
    counted_hold_events: int = 0
    counted_hold_ranges: List[Tuple[float, float]] = field(default_factory=list)
    count_limit: int = 2
    count_exceeded: bool = False
    duration_violations: List[HoldDurationViolation] = field(default_factory=list)
    uncounted_over_duration: List[UncountedOverDuration] = field(default_factory=list)
    counting_mode: str = ""
    candidates: List[HoldReviewCandidate] = field(default_factory=list)
    policy_source: str = ""
    policy_version: str = ""
    configuration_snapshot: Dict = field(default_factory=dict)


@dataclass
class SpeakerActivitySection:
    """Speaker activity section of the Final Call Review."""

    agent_active_duration_sec: float = 0.0
    customer_active_duration_sec: float = 0.0
    overlap_duration_sec: float = 0.0
    both_inactive_duration_sec: float = 0.0
    agent_activity_ratio: float = 0.0
    customer_activity_ratio: float = 0.0
    overlap_ratio: float = 0.0
    inactivity_ratio: float = 0.0
    total_intervals: int = 0
    call_duration_sec: float = 0.0


@dataclass
class ChannelSerSummary:
    """SER summary for one channel."""

    role: str = ""  # "Agent" | "Customer"
    dominant_label: Optional[str] = None
    low_emotion_ratio: float = 0.0
    neutral_emotion_ratio: float = 0.0
    high_emotion_ratio: float = 0.0
    analyzed_windows: int = 0
    uncertain_windows: int = 0
    average_confidence: Optional[float] = None


@dataclass
class TimelineItem:
    """A single item in the call timeline."""

    item_type: str  # "hold_candidate" | "hold_violation" | "dead_air" |
                    # "extended_overlap" | "call_duration" | "call_start" | "call_end"
    start_time: float = 0.0
    end_time: float = 0.0
    duration_sec: float = 0.0
    severity: str = "info"
    title: str = ""
    source: str = ""
    related_ids: List[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class FinalCallReview:
    """Complete Final Call Review — structured Team Leader review package."""

    # Call identity
    call_id: Optional[str] = None
    call_duration_sec: float = 0.0
    call_type: str = "unknown"

    # Overall status
    review_status: str = "no_immediate_review"
    manual_review_suggested: bool = False

    # Review reasons
    review_reasons: List[ReviewReason] = field(default_factory=list)

    # Sections
    hold_review: Optional[HoldReviewSection] = None
    operational_alerts_summary: Optional[Dict] = None
    speaker_activity: Optional[SpeakerActivitySection] = None
    ser_summary: Optional[Dict] = None
    timeline: List[TimelineItem] = field(default_factory=list)

    # Limitations
    evidence_limitations: List[Dict] = field(default_factory=list)

    # Configuration
    configuration_snapshot: Dict = field(default_factory=dict)
