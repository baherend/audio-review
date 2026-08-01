"""
PDF Review Report Generator.

Produces a human-readable PDF review report from the serialized
FinalCallReview export payload. Intended for the Team Leader.

This module does NOT recalculate predictions, percentages, alerts,
hold results, or review status. It only formats the existing payload.

The JSON export remains the technical evidence artifact.
The PDF is evidence-only and not an official QA evaluation.
"""

import re
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Mapping, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ══════════════════════════════════════════════════════════════════════
# COLORS
# ══════════════════════════════════════════════════════════════════════

NAVY = colors.HexColor("#0a1628")
TEAL = colors.HexColor("#00897B")
HEADING_BG = colors.HexColor("#E0F2F1")
TABLE_HEADER_BG = colors.HexColor("#112240")
TABLE_HEADER_FG = colors.white
TABLE_ALT_ROW = colors.HexColor("#F5F5F5")
LIGHT_GRAY = colors.HexColor("#EEEEEE")
WARNING_AMBER = colors.HexColor("#FF8F00")
HIGH_RED = colors.HexColor("#D32F2F")
INFO_BLUE = colors.HexColor("#1565C0")


# ══════════════════════════════════════════════════════════════════════
# FORMATTING HELPERS
# ══════════════════════════════════════════════════════════════════════

def _fmt_pct(value: Any) -> str:
    """Format a ratio as percentage with one decimal place.

    The input may be 0.0–1.0 (ratio) or 0–100 (already percentage).
    Values <= 1.0 are treated as ratios and multiplied by 100.
    """
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if v <= 1.0:
        v *= 100.0
    return f"{v:.1f}%"


def _fmt_sec(seconds: Any) -> str:
    """Format seconds as human-readable duration."""
    if seconds is None:
        return "—"
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "—"
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(s, 60)
    return f"{int(m)}m {sec:.1f}s"


def _fmt_time(seconds: Any) -> str:
    """Format a time value in seconds."""
    if seconds is None:
        return "—"
    try:
        return f"{float(seconds):.1f}s"
    except (TypeError, ValueError):
        return "—"


def _humanize_snake(text: str) -> str:
    """Convert snake_case to Title Case for readability.

    Preserves the original meaning — does not upgrade classifications.
    """
    if not text:
        return "—"
    return text.replace("_", " ").title()


def _safe_get(d: Mapping, *keys, default=None) -> Any:
    """Safely traverse nested dicts."""
    current = d
    for key in keys:
        if isinstance(current, Mapping):
            current = current.get(key, default)
        else:
            return default
    return current


def _severity_color(severity: str) -> colors.Color:
    """Return a color for a severity level."""
    mapping = {
        "high": HIGH_RED,
        "warning": WARNING_AMBER,
        "info": INFO_BLUE,
    }
    return mapping.get(severity, colors.black)


# ══════════════════════════════════════════════════════════════════════
# STYLES
# ══════════════════════════════════════════════════════════════════════

def _build_styles() -> Dict[str, ParagraphStyle]:
    """Build the PDF paragraph styles."""
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontSize=18,
            textColor=NAVY,
            spaceAfter=4 * mm,
            alignment=TA_CENTER,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontSize=9,
            textColor=colors.gray,
            alignment=TA_CENTER,
            spaceAfter=2 * mm,
        ),
        "heading": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading1"],
            fontSize=13,
            textColor=NAVY,
            spaceBefore=8 * mm,
            spaceAfter=4 * mm,
            borderWidth=0,
            borderPadding=0,
        ),
        "subheading": ParagraphStyle(
            "SubHeading",
            parent=base["Heading2"],
            fontSize=11,
            textColor=TEAL,
            spaceBefore=4 * mm,
            spaceAfter=2 * mm,
        ),
        "body": ParagraphStyle(
            "BodyText",
            parent=base["Normal"],
            fontSize=9,
            leading=13,
            spaceAfter=2 * mm,
        ),
        "small": ParagraphStyle(
            "SmallText",
            parent=base["Normal"],
            fontSize=8,
            textColor=colors.gray,
            leading=10,
        ),
        "disclaimer": ParagraphStyle(
            "Disclaimer",
            parent=base["Normal"],
            fontSize=9,
            textColor=NAVY,
            backColor=colors.HexColor("#FFF3E0"),
            borderWidth=1,
            borderColor=WARNING_AMBER,
            borderPadding=8,
            leading=13,
            spaceBefore=4 * mm,
            spaceAfter=4 * mm,
        ),
        "info_box": ParagraphStyle(
            "InfoBox",
            parent=base["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#1B5E20"),
            backColor=colors.HexColor("#E8F5E9"),
            borderPadding=6,
            leading=12,
            spaceAfter=2 * mm,
        ),
        "channel_label": ParagraphStyle(
            "ChannelLabel",
            parent=base["Normal"],
            fontSize=9,
            textColor=TEAL,
            leading=12,
        ),
    }
    return styles


# ══════════════════════════════════════════════════════════════════════
# TABLE HELPERS
# ══════════════════════════════════════════════════════════════════════

def _make_kv_table(
    rows: List[Tuple[str, str]],
    col_widths: Optional[List[float]] = None,
) -> Table:
    """Create a key-value table with alternating row colors."""
    if col_widths is None:
        col_widths = [55 * mm, 100 * mm]

    table_data = []
    for key, value in rows:
        table_data.append([
            Paragraph(f"<b>{key}</b>", ParagraphStyle("kv_key", fontSize=9, leading=12)),
            Paragraph(str(value), ParagraphStyle("kv_val", fontSize=9, leading=12)),
        ])

    t = Table(table_data, colWidths=col_widths)
    style_cmds = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
    ]
    # Alternating row colors
    for i in range(len(table_data)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), TABLE_ALT_ROW))

    t.setStyle(TableStyle(style_cmds))
    return t


def _make_data_table(
    headers: List[str],
    rows: List[List[str]],
    col_widths: Optional[List[float]] = None,
) -> Table:
    """Create a data table with header styling."""
    header_style = ParagraphStyle(
        "th", fontSize=8, textColor=TABLE_HEADER_FG, leading=11
    )
    cell_style = ParagraphStyle("td", fontSize=8, leading=11)

    table_data = [[Paragraph(h, header_style) for h in headers]]
    for row in rows:
        table_data.append([Paragraph(str(c), cell_style) for c in row])

    if col_widths is None:
        available = 165 * mm
        col_widths = [available / len(headers)] * len(headers)

    t = Table(table_data, colWidths=col_widths)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), TABLE_HEADER_FG),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
    ]
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), TABLE_ALT_ROW))

    t.setStyle(TableStyle(style_cmds))
    return t


# ══════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════

def prepare_review_report_data(
    export_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Extract structured report data from the serialized export payload.

    This function does NOT recalculate any values. It only extracts
    and organizes the existing payload into a structure convenient
    for PDF rendering.

    Args:
        export_payload: The serialized FinalCallReview dict.

    Returns:
        A dict with keys for each report section.
    """
    report: Dict[str, Any] = {}

    # ── Section A: Header ────────────────────────────────────────────
    report["header"] = {
        "call_id": export_payload.get("call_id"),
        "call_duration_sec": export_payload.get("call_duration_sec"),
        "call_type": export_payload.get("call_type"),
    }

    # ── Section B: Call Overview ─────────────────────────────────────
    speaker_activity = _safe_get(export_payload, "speaker_activity", default={}) or {}
    report["overview"] = {
        "call_id": export_payload.get("call_id"),
        "call_duration_sec": export_payload.get("call_duration_sec"),
        "call_type": export_payload.get("call_type"),
        "agent_active_sec": speaker_activity.get("agent_active_duration_sec"),
        "customer_active_sec": speaker_activity.get("customer_active_duration_sec"),
        "overlap_sec": speaker_activity.get("overlap_duration_sec"),
        "both_inactive_sec": speaker_activity.get("both_inactive_duration_sec"),
        "total_intervals": speaker_activity.get("total_intervals"),
    }

    # ── Section C/D: SER Summary ─────────────────────────────────────
    ser_summary = _safe_get(export_payload, "ser_summary", default={}) or {}
    report["agent_ser"] = _extract_channel_ser(
        ser_summary.get("agent"), "Agent"
    )
    report["customer_ser"] = _extract_channel_ser(
        ser_summary.get("customer"), "Customer"
    )

    # ── Section E: Hold Review ───────────────────────────────────────
    hold_review = _safe_get(export_payload, "hold_review", default={}) or {}
    report["hold_review"] = {
        "total_candidates": hold_review.get("total_candidates", 0),
        "likely_candidates": hold_review.get("likely_candidates", 0),
        "uncertain_candidates": hold_review.get("uncertain_candidates", 0),
        "counted_hold_events": hold_review.get("counted_hold_events", 0),
        "count_exceeded": hold_review.get("count_exceeded", False),
        "count_limit": hold_review.get("count_limit", 2),
        "duration_violations": hold_review.get("duration_violations", []),
        "uncounted_over_duration": hold_review.get("uncounted_over_duration", []),
        "candidates": hold_review.get("candidates", []),
    }

    # ── Section F: Operational Alerts ────────────────────────────────
    alerts_summary = _safe_get(
        export_payload, "operational_alerts_summary", default={}
    ) or {}
    sorted_alerts = alerts_summary.get("sorted_alerts", [])
    report["alerts"] = {
        "total_alerts": alerts_summary.get("total_alerts", 0),
        "alerts_by_severity": alerts_summary.get("alerts_by_severity", {}),
        "sorted_alerts": sorted_alerts,
        "call_duration_result": alerts_summary.get("call_duration_result"),
    }

    # ── Section G: Final Call Review ─────────────────────────────────
    review_reasons = export_payload.get("review_reasons", []) or []
    report["review"] = {
        "review_status": export_payload.get("review_status", "unknown"),
        "manual_review_suggested": export_payload.get(
            "manual_review_suggested", False
        ),
        "review_reasons": review_reasons,
        "evidence_limitations": export_payload.get("evidence_limitations", []),
    }

    return report


def _extract_channel_ser(
    channel_data: Any, role: str
) -> Dict[str, Any]:
    """Extract SER data for one channel from the serialized payload."""
    if not channel_data or not isinstance(channel_data, Mapping):
        return {
            "role": role,
            "available": False,
            "dominant_label": None,
            "low_pct": None,
            "moderate_pct": None,
            "high_pct": None,
            "analyzed_windows": None,
            "uncertain_windows": None,
            "average_confidence": None,
        }

    return {
        "role": role,
        "available": True,
        "dominant_label": channel_data.get("dominant_label"),
        "low_pct": channel_data.get("low_emotion_ratio"),
        "moderate_pct": channel_data.get("neutral_emotion_ratio"),
        "high_pct": channel_data.get("high_emotion_ratio"),
        "analyzed_windows": channel_data.get("analyzed_windows"),
        "uncertain_windows": channel_data.get("uncertain_windows"),
        "average_confidence": channel_data.get("average_confidence"),
    }


# ══════════════════════════════════════════════════════════════════════
# PDF BUILDING
# ══════════════════════════════════════════════════════════════════════

def _add_page_footer(canvas, doc):
    """Draw page footer with page number and disclaimer."""
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.gray)
    # Page number
    canvas.drawCentredString(
        A4[0] / 2, 12 * mm,
        f"Page {doc.page}"
    )
    # Footer line
    canvas.setStrokeColor(colors.HexColor("#CCCCCC"))
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, 16 * mm, A4[0] - 20 * mm, 16 * mm)
    # Footer text
    canvas.setFont("Helvetica", 6)
    canvas.drawCentredString(
        A4[0] / 2, 8 * mm,
        "Audio Intelligence — Evidence-Only Review Report — "
        "Not an official QA score or employee evaluation"
    )
    canvas.restoreState()


def build_review_pdf(export_payload: Mapping[str, Any]) -> bytes:
    """Build a PDF review report from the serialized export payload.

    Args:
        export_payload: The serialized FinalCallReview dict.

    Returns:
        Valid PDF bytes.
    """
    report_data = prepare_review_report_data(export_payload)
    return _render_pdf(report_data)


def _render_pdf(report_data: Dict[str, Any]) -> bytes:
    """Render the structured report data into PDF bytes."""
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=25 * mm,
        title="Audio Intelligence — Call Review Report",
        author="Audio Intelligence System",
    )

    styles = _build_styles()
    story: List[Any] = []

    # ── Section A: Report Header ─────────────────────────────────────
    _section_header(story, report_data, styles)

    # ── Section B: Call Overview ──────────────────────────────────────
    _section_overview(story, report_data, styles)

    # ── Section C: Agent Vocal Activation Summary ────────────────────
    _section_ser(story, report_data["agent_ser"], "Agent", styles)

    # ── Section D: Customer Vocal Activation Summary ─────────────────
    _section_ser(story, report_data["customer_ser"], "Customer", styles)

    # ── Section E: Hold Review ───────────────────────────────────────
    _section_hold(story, report_data["hold_review"], styles)

    # ── Section F: Operational Alerts ────────────────────────────────
    _section_alerts(story, report_data["alerts"], styles)

    # ── Section G: Final Call Review ─────────────────────────────────
    _section_review(story, report_data["review"], styles)

    # ── Section H: Safety and Evidence Disclaimer ────────────────────
    _section_disclaimer(story, styles)

    doc.build(story, onFirstPage=_add_page_footer, onLaterPages=_add_page_footer)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════
# SECTION RENDERERS
# ══════════════════════════════════════════════════════════════════════

def _section_header(story: List, data: Dict, styles: Dict):
    """Section A: Report Header."""
    header = data["header"]
    story.append(Paragraph(
        "Audio Intelligence — Call Review Report", styles["title"]
    ))
    story.append(Paragraph(
        "Evidence-only audio-derived review report. "
        "Not an official QA score, pass/fail judgment, or employee evaluation.",
        styles["subtitle"],
    ))

    # Call identification
    call_id = header.get("call_id")
    if call_id:
        story.append(Paragraph(
            f"<b>Call ID:</b> {call_id}", styles["body"]
        ))

    # Channel mapping
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "<b>Channel Mapping:</b>", styles["body"]
    ))
    story.append(Paragraph(
        "Left Channel = Agent", styles["channel_label"]
    ))
    story.append(Paragraph(
        "Right Channel = Customer", styles["channel_label"]
    ))
    story.append(Spacer(1, 2 * mm))

    # Divider
    story.append(Table(
        [[""]],
        colWidths=[165 * mm],
        style=TableStyle([
            ("LINEBELOW", (0, 0), (-1, 0), 1, TEAL),
        ]),
    ))


def _section_overview(story: List, data: Dict, styles: Dict):
    """Section B: Call Overview."""
    overview = data["overview"]
    story.append(Paragraph("Call Overview", styles["heading"]))

    rows = [
        ("Call Duration", _fmt_sec(overview.get("call_duration_sec"))),
        ("Call Type", _humanize_snake(overview.get("call_type", "unknown"))),
        ("Agent Active Time", _fmt_sec(overview.get("agent_active_sec"))),
        ("Customer Active Time", _fmt_sec(overview.get("customer_active_sec"))),
        ("Both Active (Overlap)", _fmt_sec(overview.get("overlap_sec"))),
        ("Both Inactive", _fmt_sec(overview.get("both_inactive_sec"))),
    ]

    intervals = overview.get("total_intervals")
    if intervals is not None:
        rows.append(("Activity Intervals", str(intervals)))

    story.append(_make_kv_table(rows))
    story.append(Spacer(1, 4 * mm))


def _section_ser(
    story: List, ser_data: Dict, role: str, styles: Dict
):
    """Section C/D: Vocal Activation Summary for one channel."""
    story.append(Paragraph(
        f"{role} Vocal Activation Summary", styles["heading"]
    ))

    if not ser_data.get("available"):
        story.append(Paragraph(
            f"No {role} summary available.", styles["body"]
        ))
        return

    # Main metrics table
    rows = [
        ("Dominant Label", ser_data.get("dominant_label") or "—"),
        ("Low Vocal Activation", _fmt_pct(ser_data.get("low_pct"))),
        ("Moderate Vocal Activation", _fmt_pct(ser_data.get("moderate_pct"))),
        ("High Vocal Activation", _fmt_pct(ser_data.get("high_pct"))),
        ("Analyzed Windows", str(ser_data.get("analyzed_windows", "—"))),
        ("Uncertain Windows", str(ser_data.get("uncertain_windows", "—"))),
    ]

    avg_conf = ser_data.get("average_confidence")
    if avg_conf is not None:
        rows.append(("Average Confidence", f"{float(avg_conf):.3f}"))

    story.append(_make_kv_table(rows))
    story.append(Spacer(1, 4 * mm))


def _section_hold(story: List, hold: Dict, styles: Dict):
    """Section E: Hold Review."""
    story.append(Paragraph("Hold Review", styles["heading"]))

    # Summary metrics
    summary_rows = [
        ("Total Candidates", str(hold.get("total_candidates", 0))),
        ("Likely Candidates", str(hold.get("likely_candidates", 0))),
        ("Uncertain Candidates", str(hold.get("uncertain_candidates", 0))),
        ("Counted Hold Events", str(hold.get("counted_hold_events", 0))),
        ("Count Exceeded", "Yes" if hold.get("count_exceeded") else "No"),
        ("Count Limit", str(hold.get("count_limit", 2))),
    ]
    story.append(_make_kv_table(summary_rows))
    story.append(Spacer(1, 3 * mm))

    # Duration violations
    violations = hold.get("duration_violations", [])
    if violations:
        story.append(Paragraph("<b>Duration Violations</b>", styles["subheading"]))
        headers = ["Candidate ID", "Start", "End", "Duration", "Limit", "Excess"]
        rows = []
        for v in violations:
            rows.append([
                str(v.get("candidate_id", "—")),
                _fmt_time(v.get("start_time")),
                _fmt_time(v.get("end_time")),
                _fmt_sec(v.get("actual_duration_sec")),
                _fmt_sec(v.get("allowed_duration_sec")),
                _fmt_sec(v.get("excess_duration_sec")),
            ])
        story.append(_make_data_table(headers, rows))
        story.append(Spacer(1, 3 * mm))

    # Uncounted over-duration
    uncounted = hold.get("uncounted_over_duration", [])
    if uncounted:
        story.append(Paragraph(
            "<b>Uncounted Over-Duration Candidates</b>", styles["subheading"]
        ))
        headers = ["Candidate ID", "Classification", "Start", "End", "Duration", "Excess"]
        rows = []
        for u in uncounted:
            rows.append([
                str(u.get("candidate_id", "—")),
                _humanize_snake(u.get("classification", "")),
                _fmt_time(u.get("start_time")),
                _fmt_time(u.get("end_time")),
                _fmt_sec(u.get("duration_sec")),
                _fmt_sec(u.get("exceeds_duration_by_sec")),
            ])
        story.append(_make_data_table(headers, rows))
        story.append(Spacer(1, 3 * mm))

    # Candidate details table
    candidates = hold.get("candidates", [])
    if candidates:
        story.append(Paragraph("<b>Candidate Details</b>", styles["subheading"]))
        headers = ["ID", "Classification", "Start", "End", "Duration", "Evidence", "Counted"]
        rows = []
        for c in candidates:
            evidence = ", ".join(c.get("evidence_types", [])) or "—"
            counted = "Yes" if c.get("counted_by_policy") else "No"
            rows.append([
                str(c.get("candidate_id", "—")),
                _humanize_snake(c.get("classification", "")),
                _fmt_time(c.get("start_time")),
                _fmt_time(c.get("end_time")),
                _fmt_sec(c.get("duration_sec")),
                evidence,
                counted,
            ])
        story.append(_make_data_table(headers, rows))
    elif not violations and not uncounted:
        story.append(Paragraph(
            "No Hold candidates detected.", styles["body"]
        ))

    story.append(Spacer(1, 4 * mm))


def _section_alerts(story: List, alerts: Dict, styles: Dict):
    """Section F: Operational Alerts."""
    story.append(Paragraph("Operational Alerts", styles["heading"]))

    total = alerts.get("total_alerts", 0)
    if total == 0:
        story.append(Paragraph(
            "No operational alerts were generated for this call.",
            styles["info_box"],
        ))
        story.append(Spacer(1, 4 * mm))
        return

    # Severity summary
    severity_counts = alerts.get("alerts_by_severity", {})
    summary_rows = [
        ("Total Alerts", str(total)),
        ("High Severity", str(severity_counts.get("high", 0))),
        ("Warning", str(severity_counts.get("warning", 0))),
        ("Info", str(severity_counts.get("info", 0))),
    ]
    story.append(_make_kv_table(summary_rows))
    story.append(Spacer(1, 3 * mm))

    # Alert details
    sorted_alerts = alerts.get("sorted_alerts", [])
    if sorted_alerts:
        headers = ["Type", "Severity", "Start", "End", "Title"]
        rows = []
        for a in sorted_alerts:
            rows.append([
                _humanize_snake(a.get("alert_type", "")),
                (a.get("severity", "") or "").title(),
                _fmt_time(a.get("start_time")),
                _fmt_time(a.get("end_time")),
                a.get("title", "—"),
            ])
        story.append(_make_data_table(headers, rows))

    # Call duration result
    cdr = alerts.get("call_duration_result")
    if cdr and cdr.get("comparison") != "not_evaluated":
        story.append(Spacer(1, 3 * mm))
        cdr_rows = [
            ("Call Duration", _fmt_sec(cdr.get("call_duration_sec"))),
            ("Expected Range", f"{_fmt_sec(cdr.get('expected_min_sec'))} – {_fmt_sec(cdr.get('expected_max_sec'))}"),
            ("Comparison", _humanize_snake(cdr.get("comparison", ""))),
        ]
        story.append(_make_kv_table(cdr_rows))

    story.append(Spacer(1, 4 * mm))


def _section_review(story: List, review: Dict, styles: Dict):
    """Section G: Final Call Review."""
    story.append(Paragraph("Final Call Review", styles["heading"]))

    # Status
    status = review.get("review_status", "unknown")
    manual = review.get("manual_review_suggested", False)
    status_display = _humanize_snake(status)

    summary_rows = [
        ("Review Status", status_display),
        ("Manual Review Suggested", "Yes" if manual else "No"),
    ]
    story.append(_make_kv_table(summary_rows))
    story.append(Spacer(1, 3 * mm))

    # Review reasons
    reasons = review.get("review_reasons", [])
    if reasons:
        story.append(Paragraph("<b>Review Reasons</b>", styles["subheading"]))
        for r in reasons:
            title = r.get("title", "—")
            explanation = r.get("explanation", "")
            severity = r.get("severity", "info")
            sev_display = severity.title() if severity else ""
            story.append(Paragraph(
                f"<b>[{sev_display}] {title}</b>", styles["body"]
            ))
            if explanation:
                story.append(Paragraph(explanation, styles["small"]))
            story.append(Spacer(1, 1 * mm))
    else:
        story.append(Paragraph(
            "No review reasons generated.", styles["body"]
        ))

    # Evidence limitations
    limitations = review.get("evidence_limitations", [])
    if limitations:
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(
            "<b>Evidence Limitations</b>", styles["subheading"]
        ))
        for lim in limitations:
            code = lim.get("code", "")
            text = lim.get("text", "")
            story.append(Paragraph(
                f"<b>{_humanize_snake(code)}:</b> {text}",
                styles["small"],
            ))
            story.append(Spacer(1, 1 * mm))

    story.append(Spacer(1, 4 * mm))


def _section_disclaimer(story: List, styles: Dict):
    """Section H: Safety and Evidence Disclaimer."""
    story.append(Paragraph("Safety and Evidence Disclaimer", styles["heading"]))
    story.append(Paragraph(
        "This report contains audio-derived review evidence only. "
        "It is not an official QA score, pass/fail judgment, "
        "employee evaluation, or confirmation of agent actions. "
        "The Team Leader remains the final decision maker.",
        styles["disclaimer"],
    ))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "SER labels are coarse model-predicted classes "
        "(Low Vocal Activation, Moderate Vocal Activation, High Vocal Activation). "
        "They do not infer anger, satisfaction, professionalism, or any specific emotion.",
        styles["small"],
    ))
    story.append(Paragraph(
        "Hold candidates are inferred from audio features, not from "
        "a system Hold-button event log. They represent audio-derived "
        "evidence, not confirmed Hold events.",
        styles["small"],
    ))


# ══════════════════════════════════════════════════════════════════════
# FILENAME GENERATION
# ══════════════════════════════════════════════════════════════════════

def generate_pdf_filename(export_payload: Mapping[str, Any]) -> str:
    """Generate a safe PDF filename from the export payload.

    Uses Call ID or falls back to a generic name.
    Does not expose local filesystem paths.
    """
    call_id = export_payload.get("call_id")
    if call_id and isinstance(call_id, str):
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", call_id.strip())
        safe = re.sub(r"_+", "_", safe).strip("_")
        if safe:
            return f"{safe}_review_report.pdf"

    return "audio_review_report.pdf"
