"""
Tests for the Streamlit app integration helpers.

Verifies:
- Old AudioEmotionPredictor is no longer imported
- EmotionEngine is used
- Stereo input requirement is enforced
- Result serialization works
- No banned output labels or scoring terms
"""

import os
import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ══════════════════════════════════════════════════════════════════════
# APP CONTENT CHECKS
# ══════════════════════════════════════════════════════════════════════

class TestAppContent:
    """Verify app.py uses approved components only."""

    def test_no_audio_emotion_predictor_import(self):
        """app.py must not import AudioEmotionPredictor."""
        app_path = os.path.join(PROJECT_ROOT, "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "AudioEmotionPredictor" not in content, (
            "app.py still references AudioEmotionPredictor"
        )

    def test_no_audio_module_import(self):
        """app.py must not import from audio_emotion_package.audio_module."""
        app_path = os.path.join(PROJECT_ROOT, "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "audio_emotion_package.audio_module" not in content, (
            "app.py still imports from audio_emotion_package.audio_module"
        )

    def test_uses_emotion_engine(self):
        """app.py must use EmotionEngine."""
        app_path = os.path.join(PROJECT_ROOT, "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "EmotionEngine" in content
        assert "from src.inference.engine import EmotionEngine" in content

    def test_no_satisfaction_score(self):
        """app.py must not compute satisfaction scores."""
        app_path = os.path.join(PROJECT_ROOT, "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "satisfaction_score" not in content.lower()
        assert "calculate_satisfaction" not in content.lower()

    def test_no_agent_skill_score(self):
        """app.py must not compute agent skill scores."""
        app_path = os.path.join(PROJECT_ROOT, "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "agent_skill" not in content.lower()
        assert "calculate_agent_skill" not in content.lower()

    def test_no_hard_call_score(self):
        """app.py must not compute hard call scores."""
        app_path = os.path.join(PROJECT_ROOT, "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "hard_call" not in content.lower()
        assert "detect_hard_call" not in content.lower()

    def test_no_banned_scoring_terms(self):
        """app.py must not contain banned scoring/grading terms."""
        app_path = os.path.join(PROJECT_ROOT, "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read().lower()
        banned = [
            "satisfaction score",
            "agent skill score",
            "anger",
            "happiness",
            "stress",
            "professionalism",
            "confirmed hold",
        ]
        for term in banned:
            # Allow in comments/documentation contexts but not in computation
            # Simple check: the term should not appear as a variable or metric
            assert term not in content, f"Banned term found: {term}"
        # "pass/fail" is allowed in disclaimer context ("Not a pass/fail judgment")
        # but not as a standalone metric or score
        lines = content.split("\n")
        for line in lines:
            stripped = line.strip()
            if "pass/fail" in stripped:
                # Allow if it's in a disclaimer/negative context
                assert "not" in stripped or "no" in stripped or "never" in stripped, (
                    f"Banned term 'pass/fail' found in non-disclaimer context: {line}"
                )

    def test_no_asr_transcript_references(self):
        """app.py must not import or use ASR or transcript functionality."""
        app_path = os.path.join(PROJECT_ROOT, "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Should not import ASR modules
        assert "from src.demo.asr" not in content
        assert "import asr" not in content
        # Should not use transcript-related classes
        assert "ChannelTranscript" not in content
        assert "CallTranscript" not in content
        assert "TranscriptSegment" not in content
        assert "TranscriptWord" not in content


# ══════════════════════════════════════════════════════════════════════
# SERIALIZATION TEST
# ══════════════════════════════════════════════════════════════════════

class TestSerialization:
    """Verify JSON serialization handles all pipeline output types."""

    def test_safe_serialize_dataclass(self):
        """Dataclasses serialize to dicts."""
        from src.demo.schemas import SpeakerSummary
        summary = SpeakerSummary(role="Agent", channel="left", analyzed_windows=5)
        # Import the helper from app
        sys.path.insert(0, PROJECT_ROOT)
        from app import _safe_serialize
        result = _safe_serialize(summary)
        assert isinstance(result, dict)
        assert result["role"] == "Agent"
        assert result["analyzed_windows"] == 5

    def test_safe_serialize_numpy(self):
        """Numpy types serialize to Python primitives."""
        from app import _safe_serialize
        assert _safe_serialize(np.int64(42)) == 42
        assert _safe_serialize(np.float64(3.14)) == pytest.approx(3.14)
        arr = _safe_serialize(np.array([1.0, 2.0]))
        assert arr == [1.0, 2.0]

    def test_safe_serialize_nested(self):
        """Nested dicts with numpy values serialize correctly."""
        from app import _safe_serialize
        data = {
            "a": np.float32(0.5),
            "b": {"c": np.int64(10), "d": [np.float64(1.0)]},
        }
        result = _safe_serialize(data)
        assert result["a"] == pytest.approx(0.5)
        assert result["b"]["c"] == 10
        assert result["b"]["d"] == [1.0]

    def test_json_roundtrip(self):
        """Serialized output is valid JSON."""
        from app import _safe_serialize
        from src.demo.schemas import FinalCallReview
        review = FinalCallReview(call_duration_sec=60.0, call_type="technical")
        export = _safe_serialize(review)
        json_str = json.dumps(export, indent=2, ensure_ascii=False)
        parsed = json.loads(json_str)
        assert parsed["call_duration_sec"] == 60.0
        assert parsed["call_type"] == "technical"


# ══════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATION TEST
# ══════════════════════════════════════════════════════════════════════

class TestPipelineOrchestration:
    """Verify the pipeline orchestration in app.py works end-to-end."""

    def test_run_pipeline_with_synthetic_stereo(self):
        """Full pipeline runs on a synthetic stereo WAV."""
        from src.inference.engine import EmotionEngine
        engine = EmotionEngine()
        engine.load()

        # Create synthetic stereo
        sr = 16000
        dur = 5.0
        n = int(sr * dur)
        t = np.linspace(0, dur, n, endpoint=False)
        agent = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        customer = (0.2 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
        stereo = np.stack([agent, customer], axis=0).T

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, stereo, sr)
            tmp_path = tmp.name

        try:
            from app import run_pipeline
            results = run_pipeline(tmp_path, engine, call_type="unknown")

            assert "load_result" in results
            assert "processing_result" in results
            assert "agent_summary" in results
            assert "customer_summary" in results
            assert "activity_intervals" in results
            assert "hold_detection" in results
            assert "hold_policy" in results
            assert "operational_alerts" in results
            assert "call_review" in results

            assert results["load_result"].validation.is_valid
            assert results["processing_result"].duration_sec > 0
            assert results["call_review"].call_duration_sec > 0
            assert results["call_review"].review_status in (
                "no_immediate_review", "review_suggested", "priority_review"
            )
        finally:
            os.unlink(tmp_path)

    def test_stereo_required(self):
        """Mono input fails validation."""
        from src.inference.engine import EmotionEngine
        engine = EmotionEngine()
        engine.load()

        # Create mono audio (1D → soundfile writes as mono)
        sr = 16000
        dur = 3.0
        n = int(sr * dur)
        mono = (0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, dur, n))).astype(np.float32)

        tmp_path = os.path.join(PROJECT_ROOT, "_test_mono.wav")
        import soundfile as sf
        sf.write(tmp_path, mono, sr)

        try:
            from app import run_pipeline
            results = run_pipeline(tmp_path, engine)
            assert results.get("error") == "validation_failed"
            assert "error_message" in results
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_review_status_valid(self):
        """Review status is one of the three allowed values."""
        from src.inference.engine import EmotionEngine
        engine = EmotionEngine()
        engine.load()

        sr = 16000
        dur = 5.0
        n = int(sr * dur)
        t = np.linspace(0, dur, n, endpoint=False)
        agent = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        customer = (0.2 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
        stereo = np.stack([agent, customer], axis=0).T

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, stereo, sr)
            tmp_path = tmp.name

        try:
            from app import run_pipeline
            results = run_pipeline(tmp_path, engine)
            status = results["call_review"].review_status
            assert status in ("no_immediate_review", "review_suggested", "priority_review")
        finally:
            os.unlink(tmp_path)

    def test_no_numeric_qa_score(self):
        """Pipeline results contain no numeric QA score."""
        from src.inference.engine import EmotionEngine
        engine = EmotionEngine()
        engine.load()

        sr = 16000
        dur = 5.0
        n = int(sr * dur)
        t = np.linspace(0, dur, n, endpoint=False)
        agent = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        customer = (0.2 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
        stereo = np.stack([agent, customer], axis=0).T

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, stereo, sr)
            tmp_path = tmp.name

        try:
            from app import run_pipeline
            results = run_pipeline(tmp_path, engine)
            review = results["call_review"]
            assert not hasattr(review, "qa_score")
            assert not hasattr(review, "satisfaction_score")
            assert not hasattr(review, "agent_skill_score")
        finally:
            os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════════════════
# UI POLISH TESTS — Phase 6 Step 3B
# ══════════════════════════════════════════════════════════════════════

class TestAppUIPolish:
    """Verify Streamlit app UI polish requirements."""

    def _read_app(self):
        """Read app.py content."""
        app_path = os.path.join(PROJECT_ROOT, "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            return f.read()

    def test_vocal_activation_summary_headings_present(self):
        """app.py must use 'Vocal Activation Summary' headings."""
        content = self._read_app()
        assert "Vocal Activation Summary" in content, (
            "Missing 'Vocal Activation Summary' heading"
        )

    def test_ser_summary_headings_absent(self):
        """app.py must not use old 'SER Summary' headings."""
        content = self._read_app()
        # Allow in docstrings/comments but not in display strings
        assert "' SER Summary'" not in content, (
            "Old 'SER Summary' heading still present in display code"
        )
        assert '" SER Summary"' not in content, (
            "Old 'SER Summary' heading still present in display code"
        )

    def test_percentage_formatter_no_double_multiply(self):
        """Percentage display must use .1f% not .1% to avoid double multiplication."""
        content = self._read_app()
        # The .1% format would multiply by 100 again on already-0-to-100 values
        assert ":.1f}%" in content, (
            "Missing .1f% format for percentages"
        )
        # Ensure no .1% format is used for label percentages
        lines = content.split("\n")
        for line in lines:
            if "dataset_label_percentages" in line or "dist.get" in line:
                assert ":.1%}" not in line, (
                    f"Double-multiplication format found: {line.strip()}"
                )

    def test_channel_mapping_text_exists(self):
        """app.py must display channel mapping prominently."""
        content = self._read_app()
        assert "Left Channel: Agent" in content or "Left = Agent" in content, (
            "Channel mapping text not found"
        )
        assert "Right Channel: Customer" in content or "Right = Customer" in content, (
            "Channel mapping text not found"
        )

    def test_evidence_only_banner(self):
        """app.py must display 'Evidence Only' prominently."""
        content = self._read_app()
        assert "Evidence Only" in content or "evidence only" in content.lower(), (
            "Evidence Only banner not found"
        )

    def test_json_preview_uses_existing_payload(self):
        """JSON preview must use the existing Final Call Review serialized payload."""
        content = self._read_app()
        assert "JSON Preview" in content, (
            "JSON Preview expander not found"
        )
        assert "export_json" in content, (
            "JSON preview must use existing export_json variable"
        )
        assert "st.code" in content, (
            "JSON preview must use st.code to display JSON"
        )

    def test_activity_timeline_in_expander(self):
        """Activity Timeline must be wrapped in a Streamlit expander."""
        content = self._read_app()
        assert "Activity Timeline" in content, (
            "Activity Timeline heading not found"
        )
        # Check that the timeline uses an expander
        timeline_idx = content.find("Activity Timeline")
        surrounding = content[max(0, timeline_idx-200):timeline_idx+200]
        assert "st.expander" in surrounding, (
            "Activity Timeline not wrapped in expander"
        )

    def test_official_labels_complete(self):
        """All three official labels must appear in full (not truncated)."""
        content = self._read_app()
        assert "Low Vocal Activation" in content, (
            "Missing 'Low Vocal Activation' label"
        )
        assert "Moderate Vocal Activation" in content, (
            "Missing 'Moderate Vocal Activation' label"
        )
        assert "High Vocal Activation" in content, (
            "Missing 'High Vocal Activation' label"
        )

    def test_no_truncated_labels(self):
        """Labels must not be truncated (e.g., 'High Voc...')."""
        content = self._read_app()
        assert "High Voc..." not in content, (
            "Truncated label 'High Voc...' found"
        )
        assert "Moderate Voc..." not in content, (
            "Truncated label 'Moderate Voc...' found"
        )
        assert "Low Voc..." not in content, (
            "Truncated label 'Low Voc...' found"
        )

    def test_professional_color_scheme(self):
        """app.py must use professional color scheme."""
        content = self._read_app()
        # Agent accent: blue
        assert "1565C0" in content or "#1565C0" in content, (
            "Agent blue accent color not found"
        )
        # Customer accent: purple
        assert "7B1FA2" in content or "#7B1FA2" in content, (
            "Customer purple accent color not found"
        )


class TestSidebarPolish:
    """Tests for sidebar presentation polish (Step 3D)."""

    @staticmethod
    def _read_app():
        return Path("app.py").read_text(encoding="utf-8")

    def test_call_type_absent_from_visible_ui(self):
        """Call Type selectbox must be hidden from sidebar."""
        content = self._read_app()
        # call_type should not appear as a selectbox in sidebar
        assert 'st.selectbox(\n            "Call Type"' not in content, (
            "Call Type selectbox should be hidden from sidebar"
        )

    def test_call_id_optional_present(self):
        """Call ID (Optional) input remains in sidebar."""
        content = self._read_app()
        assert "Call ID (Optional)" in content, (
            "Call ID (Optional) label should be present in sidebar"
        )

    def test_channel_mapping_in_sidebar(self):
        """Agent and Customer channel mapping in sidebar."""
        content = self._read_app()
        sidebar_section = content[content.find("with st.sidebar:"):content.find("# ── File Upload")]
        assert "Left Channel" in sidebar_section, "Left Channel mapping missing from sidebar"
        assert "Right Channel" in sidebar_section, "Right Channel mapping missing from sidebar"
        assert "Agent" in sidebar_section, "Agent label missing from sidebar channel mapping"
        assert "Customer" in sidebar_section, "Customer label missing from sidebar channel mapping"

    def test_labels_separate_lines(self):
        """All three vocal activation labels appear on separate lines."""
        content = self._read_app()
        sidebar_section = content[content.find("with st.sidebar:"):content.find("# ── File Upload")]
        assert "Low Vocal Activation" in sidebar_section, "Low Vocal Activation missing from sidebar"
        assert "Moderate Vocal Activation" in sidebar_section, "Moderate Vocal Activation missing from sidebar"
        assert "High Vocal Activation" in sidebar_section, "High Vocal Activation missing from sidebar"
        # Each label should be in its own element
        assert sidebar_section.count("sidebar-label") >= 3, (
            "Each label should be in its own sidebar-label element"
        )

    def test_evidence_scope_present(self):
        """System scope items present in sidebar."""
        content = self._read_app()
        sidebar_section = content[content.find("with st.sidebar:"):content.find("# ── File Upload")]
        assert "Audio-Derived Evidence" in sidebar_section, "Audio-Derived Evidence scope item missing"
        assert "Human Review Required" in sidebar_section, "Human Review Required scope item missing"
        assert "No ASR / No NLP" in sidebar_section, "No ASR / No NLP scope item missing"
        assert "Not a Pass/Fail Judgment" in sidebar_section, "Not a Pass/Fail Judgment scope item missing"

    def test_no_emotion_labels_in_sidebar(self):
        """No old Emotion labels appear in sidebar."""
        content = self._read_app()
        sidebar_section = content[content.find("with st.sidebar:"):content.find("# ── File Upload")]
        for label in ["Angry", "Happy", "Sad", "Neutral", "Fear", "Disgust", "Surprise"]:
            assert label not in sidebar_section, (
                f"Old emotion label '{label}' found in sidebar section"
            )

    def test_no_pipeline_logic_changed(self):
        """Pipeline function signatures unchanged."""
        content = self._read_app()
        assert "def run_pipeline(" in content, "run_pipeline function missing"
        assert "call_type: str" in content, "call_type parameter should still exist in run_pipeline"
        assert "def main():" in content, "main function missing"


# ══════════════════════════════════════════════════════════════════════
# MAIN PAGE VISUAL POLISH TESTS — Phase 6 Step 3E
# ══════════════════════════════════════════════════════════════════════

class TestMainPageVisualPolish:
    """Verify main page visual polish requirements (Step 3E)."""

    @staticmethod
    def _read_app():
        return Path("app.py").read_text(encoding="utf-8")

    def test_dark_hero_section_exists(self):
        """Dark hero panel with title and version must exist."""
        content = self._read_app()
        assert "hero-panel" in content, "hero-panel CSS class not found"
        assert "hero-subtitle" in content, "hero-subtitle CSS class not found"
        assert "hero-version" in content, "hero-version CSS class not found"
        assert "Elgeish BAVED v1.0.0" in content, "Version string not found"

    def test_evidence_banner_dark_contrast(self):
        """Evidence banner must use dark high-contrast class."""
        content = self._read_app()
        # Check that evidence-banner has dark background (not pale #E0F2F1)
        css_start = content.find(".evidence-banner {")
        assert css_start != -1, "evidence-banner CSS not found"
        css_block = content[css_start:css_start + 200]
        assert "#0a1628" in css_block, (
            "Evidence banner should use dark navy background #0a1628"
        )
        assert "#E0F2F1" not in css_block, (
            "Evidence banner should not use pale background #E0F2F1"
        )

    def test_evidence_only_full_sentence(self):
        """Full Evidence Only sentence must be present."""
        content = self._read_app()
        assert "Evidence Only" in content, "'Evidence Only' text missing"
        assert "Audio-derived review evidence" in content, (
            "'Audio-derived review evidence' text missing"
        )
        assert "Not a pass/fail judgment" in content, (
            "'Not a pass/fail judgment' text missing"
        )

    def test_badges_present(self):
        """Stereo, Agent, and Customer badges must be present."""
        content = self._read_app()
        assert "badge-stereo" in content, "Stereo badge class missing"
        assert "agent-badge" in content, "Agent badge class missing"
        assert "customer-badge" in content, "Customer badge class missing"
        assert "Stereo Audio" in content, "Stereo Audio badge text missing"

    def test_section_card_classes_rendered(self):
        """Main section card classes must be rendered in the page."""
        content = self._read_app()
        assert "section-card" in content, "section-card CSS class not found"
        assert "section-card-accent-teal" in content, "teal accent card class missing"
        assert "section-card-accent-agent" in content, "agent accent card class missing"
        assert "section-card-accent-customer" in content, "customer accent card class missing"
        assert "section-card-accent-amber" in content, "amber accent card class missing"

    def test_agent_summary_blue_accent(self):
        """Agent summary must have blue accent."""
        content = self._read_app()
        # Agent card uses section-card-accent-agent
        agent_card_idx = content.find('section-card-accent-agent')
        assert agent_card_idx != -1, "Agent accent card not found"
        # Agent header uses agent-header class
        assert "agent-header" in content, "agent-header CSS class missing"
        assert "#64B5F6" in content or "#1565C0" in content, (
            "Agent blue accent color not found"
        )

    def test_customer_summary_purple_accent(self):
        """Customer summary must have purple accent."""
        content = self._read_app()
        customer_card_idx = content.find('section-card-accent-customer')
        assert customer_card_idx != -1, "Customer accent card not found"
        assert "customer-header" in content, "customer-header CSS class missing"
        assert "#CE93D8" in content or "#7B1FA2" in content, (
            "Customer purple accent color not found"
        )

    def test_teal_primary_styling(self):
        """Teal primary button styling must exist."""
        content = self._read_app()
        assert "#00897B" in content, "Teal primary color #00897B not found"
        # Check button styling
        btn_idx = content.find('.stButton>button[kind="primary"]')
        assert btn_idx != -1, "Primary button CSS not found"
        btn_css = content[btn_idx:btn_idx + 200]
        assert "#00897B" in btn_css, "Primary button not using teal"
        assert "#ffffff" in btn_css, "Primary button should have white text"

    def test_no_red_normal_action_buttons(self):
        """Normal action buttons must not use red styling."""
        content = self._read_app()
        # Check that primary button doesn't use red
        btn_idx = content.find('.stButton>button[kind="primary"]')
        assert btn_idx != -1, "Primary button CSS not found"
        btn_css = content[btn_idx:btn_idx + 300]
        # Red colors that should NOT be on normal buttons
        red_colors = ["#f44336", "#d32f2f", "#c62828", "#b71c1c"]
        for red in red_colors:
            assert red not in btn_css, (
                f"Red color {red} found in primary button styling"
            )

    def test_amber_review_styling(self):
        """Amber review evidence styling must exist."""
        content = self._read_app()
        assert "#FF8F00" in content, "Amber color #FF8F00 not found"
        assert "section-card-accent-amber" in content, "Amber accent card class missing"
        # Review status uses orange
        assert "orange" in content, "Orange review status color missing"

    def test_json_preview_present(self):
        """JSON Preview expander must remain present."""
        content = self._read_app()
        assert "JSON Preview" in content, "JSON Preview expander missing"
        assert "export_json" in content, "export_json variable missing"
        assert "st.code" in content, "st.code for JSON display missing"

    def test_sidebar_step3d_unchanged(self):
        """Sidebar Step 3D content must remain unchanged."""
        content = self._read_app()
        sidebar_section = content[content.find("with st.sidebar:"):content.find("# ── File Upload")]
        # Step 3D sidebar elements
        assert "sidebar-card" in sidebar_section, "sidebar-card class missing from sidebar"
        assert "Call Metadata" in sidebar_section, "Call Metadata section missing"
        assert "Audio Configuration" in sidebar_section, "Audio Configuration section missing"
        assert "Vocal Activation Labels" in sidebar_section, "Vocal Activation Labels section missing"
        assert "System Scope" in sidebar_section, "System Scope section missing"
        assert "Call ID (Optional)" in sidebar_section, "Call ID input missing"

    def test_percentage_formatting_preserved(self):
        """Percentage formatting must remain .1f% (not .1%)."""
        content = self._read_app()
        assert ":.1f}%" in content, ".1f% format missing"
        # Ensure no .1% on label percentage lines
        lines = content.split("\n")
        for line in lines:
            if "dist.get" in line:
                assert ":.1%}" not in line, (
                    f"Double-multiply .1% format found: {line.strip()}"
                )

    def test_ser_summary_headings_absent(self):
        """Old SER Summary headings must remain absent."""
        content = self._read_app()
        assert "' SER Summary'" not in content, (
            "Old 'SER Summary' heading found in display code"
        )
        assert '" SER Summary"' not in content, (
            "Old 'SER Summary' heading found in display code"
        )


# ══════════════════════════════════════════════════════════════════════
# EXPORT SUMMARY TESTS — Phase 6 Export Summary
# ══════════════════════════════════════════════════════════════════════

class TestExportSummary:
    """Verify Export Summary section requirements."""

    @staticmethod
    def _read_app():
        return Path("app.py").read_text(encoding="utf-8")

    def test_export_summary_exists(self):
        """Export Summary metrics must exist in the Export section."""
        content = self._read_app()
        export_idx = content.find("# ── Export Card")
        assert export_idx != -1, "Export Card section not found"
        export_section = content[export_idx:]
        # Required summary fields displayed as metrics
        assert '"Call Duration"' in export_section, "Call Duration metric missing from Export"
        assert '"Review Status"' in export_section, "Review Status metric missing from Export"
        assert '"Manual Review Suggested"' in export_section, "Manual Review Suggested metric missing"
        assert '"Hold Candidates"' in export_section, "Hold Candidates metric missing from Export"
        assert '"Uncertain Hold Candidates"' in export_section, "Uncertain Hold Candidates metric missing"
        assert '"First Review Reason"' in export_section, "First Review Reason metric missing"

    def test_export_summary_uses_existing_payload(self):
        """Export Summary must read from the existing serialized payload."""
        content = self._read_app()
        export_idx = content.find("# ── Export Card")
        export_section = content[export_idx:]
        assert "export_data" in export_section, "export_data variable not used in Export section"
        assert 'export_data.get("hold_review"' in export_section, (
            "Export Summary must read hold_review from export_data"
        )
        assert 'export_data.get("review_reasons"' in export_section, (
            "Export Summary must read review_reasons from export_data"
        )

    def test_export_filename_displayed(self):
        """Export filename must be displayed in the summary."""
        content = self._read_app()
        export_idx = content.find("# ── Export Card")
        export_section = content[export_idx:]
        assert "Export Filename" in export_section, "Export Filename not displayed"
        assert "export_filename" in export_section, "export_filename variable not used"

    def test_full_json_preview_still_exists(self):
        """Full JSON Preview expander must still exist."""
        content = self._read_app()
        export_idx = content.find("# ── Export Card")
        export_section = content[export_idx:]
        assert "JSON Preview" in export_section, "JSON Preview expander removed"
        assert "st.code(export_json" in export_section, "st.code for JSON display removed"

    def test_download_button_still_exists(self):
        """Download button must still exist and use primary type."""
        content = self._read_app()
        export_idx = content.find("# ── Export Card")
        export_section = content[export_idx:]
        assert "st.download_button" in export_section, "Download button removed"
        assert 'type="primary"' in export_section, "Download button not using primary type"

    def test_export_order_summary_then_preview_then_download(self):
        """Export section must show: Summary → JSON Preview → Download."""
        content = self._read_app()
        export_idx = content.find("# ── Export Card")
        export_section = content[export_idx:]
        # Summary metrics come before JSON Preview
        summary_idx = export_section.find('"Call Duration"')
        preview_idx = export_section.find("JSON Preview")
        download_idx = export_section.find("st.download_button")
        assert summary_idx < preview_idx < download_idx, (
            "Export order must be: Summary → JSON Preview → Download"
        )

    def test_json_structure_unchanged(self):
        """JSON structure must use the existing export_data payload."""
        content = self._read_app()
        export_idx = content.find("# ── Export Card")
        export_section = content[export_idx:]
        # export_json is built from export_data (not modified)
        assert "export_json = json.dumps(export_data" in export_section, (
            "export_json must be built from unmodified export_data"
        )
