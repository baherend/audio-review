"""
Audio Intelligence — Stereo Customer-Service Call Analysis

Analyzes recorded stereo customer-service calls through the approved
Audio Intelligence pipeline. No ASR, NLP, transcription, or scoring.

Channel convention:
  Left channel = Agent
  Right channel = Customer

Usage:
    streamlit run app.py
"""

import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st

# ── Project root on path ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.engine import EmotionEngine
from src.demo.stereo_loader import AudioInputError, load_stereo_audio
from src.demo.speaker_processor import process_stereo_call
from src.demo.speaker_summary import (
    compute_speaker_summary,
    compute_inactivity_metadata,
    compute_call_summary,
    build_activity_states,
    merge_activity_states,
)
from src.demo.temporal_analysis import create_shared_windows, analyze_speech_activity
from src.demo.hold_detection import detect_hold_candidates
from src.demo.hold_policy import evaluate_hold_policy
from src.demo.operational_alerts import generate_operational_alerts
from src.demo.call_review import build_call_review
from src.demo.pdf_report import build_review_pdf, generate_pdf_filename
from src.demo.schemas import (
    WindowConfig,
    SpeechActivityConfig,
    HoldDetectionConfig,
    HoldPolicyConfig,
    OperationalAlertConfig,
    FinalCallReviewConfig,
    StereoLoadResult,
)

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Audio Intelligence — Call Analysis",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Professional presentation style
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* ── Hero header ─────────────────────────────────────────────── */
    .hero-panel {
        background: linear-gradient(135deg, #0a1628 0%, #112240 100%);
        border-radius: 12px;
        padding: 28px 32px;
        margin-bottom: 20px;
        border: 1px solid #1e3a5f;
    }
    .hero-panel h1 {
        color: #e6f1ff;
        font-size: 1.8em;
        margin: 0 0 4px 0;
    }
    .hero-panel .hero-subtitle {
        color: #8892b0;
        font-size: 0.95em;
        margin: 0 0 12px 0;
    }
    .hero-panel .hero-version {
        color: #00897B;
        font-size: 0.8em;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    /* ── Evidence banner ─────────────────────────────────────────── */
    .evidence-banner {
        background-color: #0a1628;
        color: #e6f1ff;
        border-left: 4px solid #00897B;
        padding: 12px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-size: 0.95em;
        line-height: 1.6;
    }
    .evidence-banner strong {
        color: #00897B;
    }
    /* ── Channel badges (main page) ──────────────────────────────── */
    .channel-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 16px;
        font-size: 0.82em;
        font-weight: 600;
        margin-right: 8px;
        margin-top: 6px;
    }
    .badge-stereo {
        background-color: rgba(0,137,123,0.15);
        color: #4DB6AC;
        border: 1px solid rgba(0,137,123,0.3);
    }
    .agent-badge {
        background-color: rgba(21,101,192,0.15);
        color: #64B5F6;
        border: 1px solid rgba(21,101,192,0.3);
    }
    .customer-badge {
        background-color: rgba(123,31,162,0.15);
        color: #CE93D8;
        border: 1px solid rgba(123,31,162,0.3);
    }
    /* ── Primary buttons ─────────────────────────────────────────── */
    .stButton>button[kind="primary"] {
        background-color: #00897B;
        border-color: #00897B;
        color: #ffffff;
    }
    .stButton>button[kind="primary"]:hover {
        background-color: #00695C;
        border-color: #00695C;
        color: #ffffff;
    }
    /* ── Section cards ───────────────────────────────────────────── */
    .section-card {
        background-color: #112240;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }
    .section-card h3 {
        color: #e6f1ff;
        margin: 0 0 12px 0;
        font-size: 1.15em;
    }
    .section-card-accent-agent {
        border-left: 4px solid #1565C0;
    }
    .section-card-accent-customer {
        border-left: 4px solid #7B1FA2;
    }
    .section-card-accent-teal {
        border-left: 4px solid #00897B;
    }
    .section-card-accent-amber {
        border-left: 4px solid #FF8F00;
    }
    /* ── Agent / Customer headers ────────────────────────────────── */
    .agent-header { color: #64B5F6; }
    .customer-header { color: #CE93D8; }
    /* ── Sidebar cards ───────────────────────────────────────────── */
    .sidebar-card {
        background-color: #112240;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 12px;
    }
    .sidebar-card h4 {
        color: #00897B;
        font-size: 0.8em;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0 0 8px 0;
    }
    .sidebar-label {
        display: block;
        padding: 4px 0;
        font-size: 0.9em;
        color: #ccd6f6;
    }
    .sidebar-scope-item {
        display: block;
        padding: 2px 0;
        font-size: 0.85em;
        color: #8892b0;
    }
    .sidebar-channel {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 4px 0;
        font-size: 0.9em;
        color: #ccd6f6;
    }
    .sidebar-channel-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }
    .sidebar-channel-dot.agent { background-color: #1565C0; }
    .sidebar-channel-dot.customer { background-color: #7B1FA2; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# MODEL LOADING (cached)
# ══════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_engine() -> Optional[EmotionEngine]:
    """Load the EmotionEngine once. Cached across reruns."""
    try:
        engine = EmotionEngine()
        engine.load()
        return engine
    except Exception as e:
        st.error(f"Failed to load SER model: {e}")
        st.caption(f"Attempted path: {PROJECT_ROOT / 'audio_emotion_package' / 'audio_model_fp16.pt'}")
        return None


# ══════════════════════════════════════════════════════════════════════
# SERIALIZATION HELPER
# ══════════════════════════════════════════════════════════════════════

def _safe_serialize(obj: Any) -> Any:
    """Convert dataclasses, numpy, and Path objects to JSON-safe types."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _safe_serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    return obj


# ══════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════

def _pct(value: float, total: float) -> str:
    """Format a ratio as percentage string."""
    if total <= 0:
        return "0.0%"
    return f"{(value / total) * 100:.1f}%"


def _fmt_sec(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.1f}s"


def _display_validation(validation):
    """Display stereo validation results."""
    if validation.is_valid:
        st.success("✅ Stereo validation passed")
    else:
        st.error("❌ Stereo validation failed")

    for err in validation.errors:
        st.error(f"• {err.message}")
    for warn in validation.warnings:
        st.warning(f"• {warn.message}")

    m = validation.metrics
    st.caption(
        f"Channels: {m.channel_count} | "
        f"Samples: {m.samples_per_channel:,} | "
        f"SR: {m.sample_rate} Hz | "
        f"Duration: {m.duration_sec:.1f}s | "
        f"Agent RMS: {m.agent_rms:.4f} | "
        f"Customer RMS: {m.customer_rms:.4f}"
    )


# ══════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════

_GENERIC_AUDIO_VALIDATION_MESSAGE = (
    "The uploaded file could not be validated as stereo audio. "
    "Choose a valid WAV file with Left=Agent and Right=Customer."
)


def _validation_failure(
    error_code: str,
    error_message: str,
    load_result: Optional[StereoLoadResult] = None,
) -> Dict[str, Any]:
    """Return the single failure contract used for rejected audio."""
    return {
        "error": "validation_failed",
        "error_code": error_code,
        "error_message": error_message,
        "load_result": load_result,
    }


def _validation_failure_from_load_result(
    load_result: StereoLoadResult,
) -> Dict[str, Any]:
    """Convert a typed validation rejection to the shared failure contract."""
    validation_errors = load_result.validation.errors
    if validation_errors:
        first_error = validation_errors[0]
        return _validation_failure(
            first_error.code,
            first_error.message,
            load_result=load_result,
        )
    return _validation_failure(
        "INVALID_AUDIO",
        _GENERIC_AUDIO_VALIDATION_MESSAGE,
        load_result=load_result,
    )


def validate_uploaded_audio(
    source: Any,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    """Decode and validate user audio without initializing the model."""
    try:
        load_result = load_stereo_audio(source, filename=filename)
    except AudioInputError as exc:
        return _validation_failure(exc.code, str(exc))
    except (FileNotFoundError, OSError, ValueError):
        return _validation_failure(
            "INVALID_AUDIO",
            _GENERIC_AUDIO_VALIDATION_MESSAGE,
        )

    if not load_result.validation.is_valid:
        return _validation_failure_from_load_result(load_result)
    return {"load_result": load_result}


def _display_audio_validation_failure(result: Dict[str, Any]) -> None:
    """Display a validation failure without assuming decoded audio exists."""
    st.header("🔍 Validation Results")
    load_result = result.get("load_result")
    if load_result is not None:
        _display_validation(load_result.validation)
        return

    st.error("❌ Stereo validation failed")
    st.error(result.get("error_message", _GENERIC_AUDIO_VALIDATION_MESSAGE))


def run_pipeline(
    audio_path: str,
    engine: EmotionEngine,
    call_type: str = "unknown",
    call_id: Optional[str] = None,
    prevalidated_load_result: Optional[StereoLoadResult] = None,
) -> Dict[str, Any]:
    """Run the full Audio Intelligence pipeline. Returns all results."""
    results = {}

    # 1. Load and validate stereo audio
    if prevalidated_load_result is None:
        validation_result = validate_uploaded_audio(audio_path)
        if validation_result.get("error") == "validation_failed":
            return validation_result
        load_result = validation_result.get("load_result")
    else:
        load_result = prevalidated_load_result

    if load_result is None:
        return _validation_failure(
            "INVALID_AUDIO",
            _GENERIC_AUDIO_VALIDATION_MESSAGE,
        )
    if not load_result.validation.is_valid:
        return _validation_failure_from_load_result(load_result)

    results["load_result"] = load_result

    # 2. Process stereo call (windows + speech activity + SER)
    window_config = WindowConfig()
    speech_config = SpeechActivityConfig()
    processing_result = process_stereo_call(
        load_result=load_result,
        engine=engine,
        window_config=window_config,
        speech_config=speech_config,
    )
    results["processing_result"] = processing_result

    call_duration = processing_result.duration_sec

    # 3. Speaker summaries
    agent_summary = compute_speaker_summary(
        processing_result.agent_result, call_duration
    )
    customer_summary = compute_speaker_summary(
        processing_result.customer_result, call_duration
    )
    results["agent_summary"] = agent_summary
    results["customer_summary"] = customer_summary

    # 4. Activity intervals
    activity_intervals = build_activity_states(
        window_results=processing_result.window_results,
        agent_waveform=load_result.model_audio.agent_waveform,
        customer_waveform=load_result.model_audio.customer_waveform,
        sample_rate=load_result.model_audio.model_sr,
        call_duration_sec=call_duration,
    )
    activity_intervals = merge_activity_states(activity_intervals)
    results["activity_intervals"] = activity_intervals

    # 5. Inactivity metadata
    inactivity_metadata = compute_inactivity_metadata(activity_intervals)
    results["inactivity_metadata"] = inactivity_metadata

    # 6. Call summary
    call_summary = compute_call_summary(
        processing_result, agent_summary, customer_summary,
        activity_intervals, inactivity_metadata,
    )
    results["call_summary"] = call_summary

    # 7. Hold detection
    hold_result = detect_hold_candidates(
        intervals=activity_intervals,
        agent_waveform=load_result.model_audio.agent_waveform,
        customer_waveform=load_result.model_audio.customer_waveform,
        sample_rate=load_result.model_audio.model_sr,
        call_duration_sec=call_duration,
    )
    results["hold_detection"] = hold_result

    # 8. Hold policy
    hold_policy_result = evaluate_hold_policy(
        hold_result.candidates, HoldPolicyConfig()
    )
    results["hold_policy"] = hold_policy_result

    # 9. Operational alerts
    alert_result = generate_operational_alerts(
        intervals=activity_intervals,
        hold_policy_result=hold_policy_result,
        call_duration_sec=call_duration,
        call_type=call_type,
    )
    results["operational_alerts"] = alert_result

    # 10. Final Call Review
    call_review = build_call_review(
        call_duration_sec=call_duration,
        call_type=call_type,
        call_summary=call_summary,
        hold_detection_result=hold_result,
        hold_policy_result=hold_policy_result,
        operational_alert_result=alert_result,
        call_id=call_id,
    )
    results["call_review"] = call_review

    return results


# ══════════════════════════════════════════════════════════════════════
# RESULT DISPLAY SECTIONS
# ══════════════════════════════════════════════════════════════════════

def _show_call_overview(results: Dict):
    """Section B: Call Overview."""
    st.subheader("📋 Call Overview")
    cr = results["call_review"]
    proc = results["processing_result"]
    agent_sum = results["agent_summary"]
    cust_sum = results["customer_summary"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Duration", _fmt_sec(cr.call_duration_sec))
        st.metric("Analysis Windows", len(proc.shared_windows))
    with c2:
        st.metric("Agent Active", _fmt_sec(agent_sum.detected_active_duration_sec))
        st.metric("Customer Active", _fmt_sec(cust_sum.detected_active_duration_sec))
    with c3:
        st.metric("Agent Inactive", _fmt_sec(agent_sum.detected_inactive_duration_sec))
        st.metric("Customer Inactive", _fmt_sec(cust_sum.detected_inactive_duration_sec))

    # Activity ratios
    ci = results["call_summary"]
    st.caption(
        f"Both active: {_fmt_sec(ci.both_active_duration_sec)} "
        f"({_pct(ci.both_active_duration_sec, cr.call_duration_sec)}) | "
        f"Neither active: {_fmt_sec(ci.neither_active_duration_sec)} "
        f"({_pct(ci.neither_active_duration_sec, cr.call_duration_sec)})"
    )


def _show_ser_summary(role: str, summary):
    """Section C/D: Vocal Activation Summary for one channel."""
    if role == "Agent":
        st.markdown(f'<h3 class="agent-header">🧑‍💼 {role} Vocal Activation Summary</h3>', unsafe_allow_html=True)
    else:
        st.markdown(f'<h3 class="customer-header">👤 {role} Vocal Activation Summary</h3>', unsafe_allow_html=True)

    if summary is None:
        st.info(f"No {role} summary available.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Analyzed Windows", summary.analyzed_windows)
        st.metric("Dominant Label", summary.dominant_dataset_label or "—")
    with c2:
        dist = summary.dataset_label_percentages
        st.metric("Low Vocal Activation", f"{dist.get('Low Vocal Activation', 0):.1f}%")
        st.metric("Moderate Vocal Activation", f"{dist.get('Moderate Vocal Activation', 0):.1f}%")
    with c3:
        st.metric("High Vocal Activation", f"{dist.get('High Vocal Activation', 0):.1f}%")
        st.metric("Uncertain Windows", summary.uncertain_windows)

    if summary.average_confidence is not None:
        st.caption(
            f"Confidence: avg {summary.average_confidence:.3f} | "
            f"min {summary.minimum_confidence or 0:.3f} | "
            f"max {summary.maximum_confidence or 0:.3f}"
        )

    if summary.label_transition_count:
        st.caption(f"Label transitions: {summary.label_transition_count}")


def _show_activity_timeline(results: Dict):
    """Section E: Activity Timeline (collapsed by default)."""
    intervals = results["activity_intervals"]
    with st.expander("⏱️ Activity Timeline", expanded=False):
        if not intervals:
            st.info("No activity intervals.")
            return

        rows = []
        for iv in intervals:
            rows.append({
                "Start": f"{iv.start_time:.1f}s",
                "End": f"{iv.end_time:.1f}s",
                "Duration": f"{iv.duration_sec:.1f}s",
                "State": iv.state,
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _show_hold_review(results: Dict):
    """Section F: Hold Review."""
    st.subheader("📞 Hold Review")
    hd = results["hold_detection"]
    hp = results["hold_policy"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Candidates", hd.total_candidates)
        st.metric("Likely Candidates", hd.likely_count)
    with c2:
        st.metric("Uncertain Candidates", hd.uncertain_count)
        st.metric("Counted by Policy", hp.counted_hold_events)
    with c3:
        st.metric("Duration Violations", len(hp.duration_violations))
        st.metric("Count Exceeded", "Yes" if hp.count_exceeded else "No")

    if hp.uncounted_over_duration:
        st.warning(
            f"{len(hp.uncounted_over_duration)} candidate(s) exceed duration "
            f"but were excluded by counting mode — review suggested."
        )

    if hd.candidates:
        rows = []
        for c in hd.candidates:
            rows.append({
                "ID": c.candidate_id,
                "Classification": c.classification,
                "Start": f"{c.start_time:.1f}s",
                "End": f"{c.end_time:.1f}s",
                "Duration": f"{c.duration_sec:.1f}s",
                "Confidence": f"{c.confidence:.3f}",
                "Evidence": ", ".join(c.evidence_types),
                "Agent Return": "Yes" if c.agent_return_detected else "No",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No Hold candidates detected.")


def _show_operational_alerts(results: Dict):
    """Section G: Operational Alerts."""
    st.subheader("⚠️ Operational Alerts")
    ar = results["operational_alerts"]

    if not ar.alerts:
        st.info("No operational alerts.")
        return

    # Severity counts
    counts = ar.alert_counts_by_severity
    st.caption(
        f"High: {counts.get('high', 0)} | "
        f"Warning: {counts.get('warning', 0)} | "
        f"Info: {counts.get('info', 0)} | "
        f"Manual review: {'Yes' if ar.manual_review_suggested else 'No'}"
    )

    rows = []
    for a in ar.alerts:
        rows.append({
            "Type": a.alert_type,
            "Severity": a.severity,
            "Start": f"{a.start_time:.1f}s" if a.start_time is not None else "—",
            "End": f"{a.end_time:.1f}s" if a.end_time is not None else "—",
            "Duration": f"{a.duration_sec:.1f}s" if a.duration_sec else "—",
            "Title": a.title,
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # Call duration result
    cdr = ar.call_duration_result
    if cdr and cdr.comparison != "not_evaluated":
        st.caption(
            f"Call duration: {cdr.call_duration_sec:.0f}s | "
            f"Expected range ({cdr.call_type}): "
            f"{cdr.expected_min_sec:.0f}s–{cdr.expected_max_sec:.0f}s | "
            f"Result: {cdr.comparison}"
        )


def _show_call_review(results: Dict):
    """Section H: Final Call Review."""
    st.subheader("📝 Final Call Review")
    cr = results["call_review"]

    # Status — amber for review evidence, green for technical success
    status_config = {
        "priority_review": ("🟠", "orange"),
        "review_suggested": ("🟡", "orange"),
        "no_immediate_review": ("🟢", "green"),
    }
    icon, color = status_config.get(cr.review_status, ("⚪", "gray"))
    st.markdown(
        f"**{icon} Review Status:** "
        f":{color}[{cr.review_status.replace('_', ' ').title()}]"
    )
    st.caption(f"Manual review suggested: {'Yes' if cr.manual_review_suggested else 'No'}")

    # Review reasons — amber for evidence
    if cr.review_reasons:
        st.markdown("**Review Reasons:**")
        for r in cr.review_reasons:
            sev_icon = {"high": "🟠", "warning": "🟡", "info": "🔵"}.get(r.severity, "⚪")
            st.markdown(f"{sev_icon} **{r.title}**")
            st.caption(r.explanation)

    # Evidence limitations
    if cr.evidence_limitations:
        with st.expander("Evidence Limitations"):
            for lim in cr.evidence_limitations:
                st.markdown(f"• **{lim['code']}**: {lim['text']}")


# ══════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════

def main():
    # ── Hero Header ─────────────────────────────────────────────────
    st.markdown(
        '<div class="hero-panel">'
        '<h1>🎧 Audio Intelligence — Call Analysis</h1>'
        '<p class="hero-subtitle">'
        'Analyzes recorded stereo customer-service calls through the '
        'approved Audio Intelligence pipeline. No ASR, NLP, transcription, or scoring.'
        '</p>'
        '<span class="hero-version">Elgeish BAVED v1.0.0</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Evidence Banner ─────────────────────────────────────────────
    st.markdown(
        '<div class="evidence-banner">'
        '<strong>📋 Evidence Only</strong> — Audio-derived review evidence. '
        'Not a pass/fail judgment.<br>'
        '<span class="channel-badge badge-stereo">🎵 Stereo Audio</span>'
        '<span class="channel-badge agent-badge">🔵 Left Channel: Agent</span>'
        '<span class="channel-badge customer-badge">🟣 Right Channel: Customer</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        # ── Call Metadata ─────────────────────────────────────────────
        st.markdown(
            '<div class="sidebar-card">'
            '<h4>📋 Call Metadata</h4>'
            '</div>',
            unsafe_allow_html=True,
        )
        call_id = st.text_input(
            "Call ID (Optional)",
            placeholder="e.g. CALL-001",
            help="Used only to identify the exported review.",
        )
        # call_type kept in backend, hidden from sidebar
        call_type = "unknown"

        # ── Audio Configuration ───────────────────────────────────────
        st.markdown(
            '<div class="sidebar-card">'
            '<h4>🎵 Audio Configuration</h4>'
            '<div class="sidebar-channel">'
            '<span class="sidebar-channel-dot agent"></span>'
            '<span><strong>Stereo Audio</strong></span>'
            '</div>'
            '<div class="sidebar-channel">'
            '<span class="sidebar-channel-dot agent"></span>'
            '<span>Left Channel: <strong>Agent</strong></span>'
            '</div>'
            '<div class="sidebar-channel">'
            '<span class="sidebar-channel-dot customer"></span>'
            '<span>Right Channel: <strong>Customer</strong></span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # ── Vocal Activation Labels ──────────────────────────────────
        st.markdown(
            '<div class="sidebar-card">'
            '<h4>🏷️ Vocal Activation Labels</h4>'
            '<span class="sidebar-label">• Low Vocal Activation</span>'
            '<span class="sidebar-label">• Moderate Vocal Activation</span>'
            '<span class="sidebar-label">• High Vocal Activation</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        # ── System Scope ─────────────────────────────────────────────
        st.markdown(
            '<div class="sidebar-card">'
            '<h4>🔍 System Scope</h4>'
            '<span class="sidebar-scope-item">✓ Audio-Derived Evidence</span>'
            '<span class="sidebar-scope-item">✓ Human Review Required</span>'
            '<span class="sidebar-scope-item">✓ No ASR / No NLP</span>'
            '<span class="sidebar-scope-item">✓ Not a Pass/Fail Judgment</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── File Upload ────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-card section-card-accent-teal">',
        unsafe_allow_html=True,
    )
    st.header("📁 Upload Stereo Call")

    uploaded_file = st.file_uploader(
        "Choose a stereo WAV file",
        type=["wav"],
        help="Stereo audio required. Left=Agent, Right=Customer.",
    )

    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")

        if st.button("🚀 Run Audio Intelligence Analysis", type="primary",
                     use_container_width=True):

            # Save to temp file
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False, dir=str(PROJECT_ROOT)
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Validate user-controlled audio before any model initialization.
                validation_result = validate_uploaded_audio(
                    tmp_path,
                    filename=uploaded_file.name,
                )
                if validation_result.get("error") == "validation_failed":
                    _display_audio_validation_failure(validation_result)
                    st.stop()

                load_result = validation_result.get("load_result")
                if load_result is None:
                    _display_audio_validation_failure(
                        _validation_failure(
                            "INVALID_AUDIO",
                            _GENERIC_AUDIO_VALIDATION_MESSAGE,
                        )
                    )
                    st.stop()

                # Load engine
                engine = load_engine()
                if engine is None:
                    st.stop()

                # Run pipeline with progress
                st.markdown("---")
                progress = st.progress(0, text="Loading and validating stereo audio…")

                try:
                    t0 = time.time()
                    results = run_pipeline(
                        audio_path=tmp_path,
                        engine=engine,
                        call_type=call_type,
                        call_id=call_id or None,
                        prevalidated_load_result=load_result,
                    )
                    elapsed = time.time() - t0
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")
                    with st.expander("Technical diagnostic"):
                        st.code(f"{type(e).__name__}: {e}")
                    st.stop()
                finally:
                    progress.empty()

                if results.get("error") == "validation_failed":
                    _display_audio_validation_failure(results)
                    st.stop()

                # Store results in session
                st.session_state.results = results
                st.session_state.filename = uploaded_file.name
                st.session_state.elapsed = elapsed

                st.success(f"✅ Analysis complete in {elapsed:.1f}s")
                st.rerun()

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    st.markdown('</div>', unsafe_allow_html=True)  # close upload card

    # ── Results Display ────────────────────────────────────────────────
    if "results" not in st.session_state:
        st.info("👆 Upload a stereo WAV file and click **Run Audio Intelligence Analysis**.")
        return

    results = st.session_state.results
    load_result = results.get("load_result")
    if load_result is None:
        st.error(_GENERIC_AUDIO_VALIDATION_MESSAGE)
        return

    # ── Input Information Card ──────────────────────────────────────
    st.markdown('<div class="section-card section-card-accent-teal">', unsafe_allow_html=True)
    with st.expander("📎 Input Information", expanded=True):
        st.markdown(f"**File:** {st.session_state.get('filename', '—')}")
        st.markdown(f"**Processing time:** {st.session_state.get('elapsed', 0):.1f}s")
        _display_validation(load_result.validation)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Call Overview Card ──────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    _show_call_overview(results)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Agent / Customer Summary Cards ──────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-card section-card-accent-agent">', unsafe_allow_html=True)
        _show_ser_summary("Agent", results["agent_summary"])
        st.markdown('</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="section-card section-card-accent-customer">', unsafe_allow_html=True)
        _show_ser_summary("Customer", results["customer_summary"])
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Activity Timeline Card ──────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    _show_activity_timeline(results)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Hold Review Card ────────────────────────────────────────────
    st.markdown('<div class="section-card section-card-accent-amber">', unsafe_allow_html=True)
    _show_hold_review(results)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Operational Alerts Card ─────────────────────────────────────
    st.markdown('<div class="section-card section-card-accent-amber">', unsafe_allow_html=True)
    _show_operational_alerts(results)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Final Call Review Card ──────────────────────────────────────
    st.markdown('<div class="section-card section-card-accent-amber">', unsafe_allow_html=True)
    _show_call_review(results)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Export Card ─────────────────────────────────────────────────
    st.markdown('<div class="section-card section-card-accent-teal">', unsafe_allow_html=True)
    st.subheader("📥 Export")
    try:
        export_data = _safe_serialize(results["call_review"])
        export_json = json.dumps(export_data, indent=2, ensure_ascii=False)
        filename = st.session_state.get("filename", "call")
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in filename)
        export_filename = f"call_review_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # ── Export Summary (compact, presentation-friendly) ─────────
        hr = export_data.get("hold_review", {}) or {}
        reasons = export_data.get("review_reasons", []) or []
        first_reason = reasons[0].get("title", "—") if reasons else "—"

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Call Duration", _fmt_sec(export_data.get("call_duration_sec", 0)))
            st.metric("Review Status", export_data.get("review_status", "—").replace("_", " ").title())
            st.metric("Manual Review Suggested", "Yes" if export_data.get("manual_review_suggested") else "No")
        with c2:
            st.metric("Hold Candidates", hr.get("total_candidates", 0))
            st.metric("Uncertain Hold Candidates", hr.get("uncertain_candidates", 0))
            st.metric("First Review Reason", first_reason)
        st.caption(f"**Export Filename:** `{export_filename}`")

        # ── Full JSON Preview (collapsed) ───────────────────────────
        with st.expander("📄 JSON Preview", expanded=False):
            st.code(export_json, language="json")

        # ── Download buttons (PDF + JSON) ────────────────────────────
        pdf_filename = generate_pdf_filename(export_data)
        pdf_bytes = build_review_pdf(export_data)

        col_pdf, col_json = st.columns(2)
        with col_pdf:
            st.download_button(
                label="📄 Download Review Report (PDF)",
                data=pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
        with col_json:
            st.download_button(
                label="📥 Download Technical Evidence (JSON)",
                data=export_json,
                file_name=export_filename,
                mime="application/json",
                use_container_width=True,
            )
    except Exception as e:
        st.error(f"Could not serialize results for export: {e}")
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
