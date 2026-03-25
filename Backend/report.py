"""
report.py — Generate a PDF incident report from AnalysisResult

Requires: reportlab
Install:  pip install reportlab --break-system-packages
"""

from pathlib import Path
from datetime import datetime
from schemas import AnalysisResult


def generate_pdf_report(result: AnalysisResult, output_path: str) -> str:
    """
    Build a clean PDF report summarising the analysis.
    Falls back gracefully if reportlab is not installed.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=2.5*cm, rightMargin=2.5*cm,
            topMargin=2.5*cm,  bottomMargin=2.5*cm,
        )

        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            "Title2", parent=styles["Title"],
            fontSize=20, textColor=colors.HexColor("#0d2b10"),
            spaceAfter=6,
        )
        subtitle_style = ParagraphStyle(
            "Sub", parent=styles["Normal"],
            fontSize=9, textColor=colors.HexColor("#4a6b4e"),
            spaceAfter=20,
        )
        heading_style = ParagraphStyle(
            "H2", parent=styles["Heading2"],
            fontSize=12, textColor=colors.HexColor("#1a3d1e"),
            spaceBefore=16, spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "Body2", parent=styles["Normal"],
            fontSize=9, leading=14,
            textColor=colors.HexColor("#2c3e2e"),
        )

        # Verdict colour
        verdict_colour = colors.HexColor("#c0392b") if result.littering_detected \
                    else colors.HexColor("#27ae60")

        verdict_style = ParagraphStyle(
            "Verdict", parent=styles["Normal"],
            fontSize=14, textColor=verdict_colour,
            fontName="Helvetica-Bold",
        )

        elements = []

        # ── Header ──────────────────────────────────────────────────────────
        elements.append(Paragraph("LitterWatch", title_style))
        elements.append(Paragraph("Smart Littering Detection System — Incident Report", subtitle_style))
        elements.append(HRFlowable(width="100%", thickness=1,
                                   color=colors.HexColor("#c8e6c9"), spaceAfter=16))

        # ── Verdict ──────────────────────────────────────────────────────────
        verdict_text = "⚠ LITTERING DETECTED" if result.littering_detected else "✔ NO LITTERING DETECTED"
        elements.append(Paragraph(verdict_text, verdict_style))
        elements.append(Spacer(1, 0.4*cm))

        # ── Summary table ────────────────────────────────────────────────────
        elements.append(Paragraph("Analysis Summary", heading_style))

        report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary_data = [
            ["Field", "Value"],
            ["Job ID",               result.job_id[:16] + "…"],
            ["Report Generated",     report_date],
            ["Video Duration",       f"{result.duration_s:.1f} seconds"],
            ["Resolution",           result.resolution],
            ["FPS",                  f"{result.fps:.1f}"],
            ["Total Frames",         str(result.total_frames)],
            ["Frames Analysed",      str(result.frames_analysed)],
            ["Persons Detected",     str(result.persons_detected)],
            ["Littering Events",     str(result.event_count)],
            ["Avg. Confidence",      f"{result.avg_confidence * 100:.1f}%"],
        ]

        t = Table(summary_data, colWidths=[7*cm, 10*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#1a3d1e")),
            ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, 0),  9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#f1f8f2"), colors.white]),
            ("FONTSIZE",     (0, 1), (-1, -1), 9),
            ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#c8e6c9")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING",   (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.5*cm))

        # ── Event log ────────────────────────────────────────────────────────
        elements.append(Paragraph("Detected Incidents", heading_style))

        if not result.events:
            elements.append(Paragraph(
                "No littering incidents were detected in this footage.", body_style))
        else:
            event_data = [["#", "Timestamp", "Frame", "Confidence", "Description"]]
            for i, ev in enumerate(result.events, 1):
                event_data.append([
                    str(i),
                    ev.timestamp_str,
                    str(ev.frame_number),
                    f"{ev.confidence * 100:.0f}%",
                    Paragraph(ev.description, body_style),
                ])

            et = Table(event_data, colWidths=[0.8*cm, 2.2*cm, 2*cm, 2.5*cm, 9.5*cm])
            et.setStyle(TableStyle([
                ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#c0392b")),
                ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
                ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("FONTSIZE",     (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#fff5f5"), colors.white]),
                ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#f5c6c6")),
                ("LEFTPADDING",  (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING",   (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
                ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
            ]))
            elements.append(et)

        # ── Footer ───────────────────────────────────────────────────────────
        elements.append(Spacer(1, 1*cm))
        elements.append(HRFlowable(width="100%", thickness=0.5,
                                   color=colors.HexColor("#c8e6c9")))
        elements.append(Spacer(1, 0.2*cm))
        elements.append(Paragraph(
            f"Generated by LitterWatch v1.0 · {report_date} · Job {result.job_id[:8]}",
            ParagraphStyle("Footer", parent=styles["Normal"],
                           fontSize=7, textColor=colors.HexColor("#7a9980"))
        ))

        doc.build(elements)
        print(f"[INFO] PDF report saved: {output_path}")
        return output_path

    except ImportError:
        # Write a plain-text fallback if reportlab is missing
        txt_path = output_path.replace(".pdf", ".txt")
        with open(txt_path, "w") as f:
            f.write(f"LITTERWATCH REPORT — {result.job_id}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Verdict          : {result.verdict}\n")
            f.write(f"Events Detected  : {result.event_count}\n")
            f.write(f"Frames Analysed  : {result.frames_analysed}\n")
            f.write(f"Duration         : {result.duration_s:.1f}s\n\n")
            for i, ev in enumerate(result.events, 1):
                f.write(f"[{i}] {ev.timestamp_str}  conf={ev.confidence:.2f}  {ev.description}\n")
        print(f"[WARN] reportlab not installed. Plain-text report saved: {txt_path}")
        return txt_path
