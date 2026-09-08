"""PDF report generation (ReportLab) for every recommendation type."""
from __future__ import annotations

import os
from datetime import datetime
from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

GREEN = colors.HexColor("#1b7f3b")
DARK = colors.HexColor("#123c1f")


def _styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontSize=20, textColor=GREEN, spaceAfter=4),
        "sub": ParagraphStyle("s", parent=ss["Normal"], fontSize=9, textColor=colors.grey, spaceAfter=10),
        "h": ParagraphStyle("h", parent=ss["Heading2"], fontSize=13, textColor=DARK, spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("b", parent=ss["Normal"], fontSize=10, leading=14, alignment=4),
        "small": ParagraphStyle("sm", parent=ss["Normal"], fontSize=8, textColor=colors.grey),
    }


def _table(rows: list[list[str]], col_widths=(65 * mm, 95 * mm), header=True) -> Table:
    t = Table(rows, colWidths=list(col_widths))
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8d6cc")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f8f4")]),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), GREEN), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    return t


def _image(path: str | None, width=70 * mm, height=55 * mm):
    if path and os.path.exists(path):
        try:
            img = Image(path, width=width, height=height)
            img.hAlign = "CENTER"
            return img
        except Exception:  # noqa: BLE001
            return None
    return None


def _doc(title: str, farmer_name: str | None, elements_fn) -> BytesIO:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title=title)
    st = _styles()
    el: list[Any] = [Paragraph(f"KrishiDisha - {title}", st["title"]),
                     Paragraph(f"Generated {datetime.now():%d %b %Y, %H:%M}"
                               + (f" for {farmer_name}" if farmer_name else ""), st["sub"])]
    elements_fn(el, st)
    el.append(Spacer(1, 14))
    el.append(Paragraph("KrishiDisha recommendations are decision-support estimates from machine-learning models trained "
                        "on public datasets. Validate with a local Krishi Vigyan Kendra or agronomist before large "
                        "investments.", st["small"]))
    doc.build(el)
    buf.seek(0)
    return buf


def crop_report(inputs: dict, result: dict, static_dir: str, farmer_name: str | None = None) -> BytesIO:
    def build(el, st):
        el.append(Paragraph("Soil and climate inputs", st["h"]))
        el.append(_table([["Parameter", "Value"]] + [[k, str(v)] for k, v in inputs.items()]))
        el.append(Paragraph("Recommendation", st["h"]))
        rows = [["Field", "Value"], ["Recommended crop", result["recommended_crop"].title()],
                ["Model confidence", f"{result['confidence'] * 100:.1f}%"],
                ["Alternatives", ", ".join(f"{a['crop']} ({a['probability'] * 100:.0f}%)" for a in result.get("alternatives", []))]]
        if result.get("revenue_per_acre"):
            rows += [["Indicative revenue / acre", f"Rs {result['revenue_per_acre']:,}"],
                     ["Indicative cost / acre", f"Rs {result['cost_per_acre']:,}"],
                     ["Indicative profit / acre", f"Rs {result['profit_per_acre']:,}"]]
        el.append(_table(rows))
        if result.get("guide"):
            el.append(Paragraph("Cultivation guide", st["h"]))
            g = result["guide"]
            el.append(_table([["Topic", "Guidance"]] + [[k.replace("_", " ").title(), Paragraph(str(v), st["body"])]
                                                        for k, v in g.items() if isinstance(v, str)][:10]))
        img = _image(os.path.join(static_dir, result["image"]) if result.get("image") else None)
        if img:
            el.append(Spacer(1, 8))
            el.append(img)
    return _doc("Crop Recommendation Report", farmer_name, build)


def fertilizer_report(inputs: dict, result: dict, static_dir: str, farmer_name: str | None = None) -> BytesIO:
    def build(el, st):
        el.append(Paragraph("Field inputs", st["h"]))
        el.append(_table([["Parameter", "Value"]] + [[k, str(v)] for k, v in inputs.items()]))
        el.append(Paragraph("Recommendation", st["h"]))
        rows = [["Field", "Value"], ["Recommended fertilizer", result["recommended_fertilizer"]],
                ["NPK grade", result.get("npk") or "-"], ["Model confidence", f"{result['confidence'] * 100:.1f}%"],
                ["How to use", Paragraph(result.get("usage_note") or "-", st["body"])]]
        el.append(_table(rows))
        if result.get("calculator") and "error" not in result["calculator"]:
            c = result["calculator"]
            el.append(Paragraph(f"Dose calculator ({c['area']} {c['unit']})", st["h"]))
            el.append(_table([["Fertilizer", "Quantity (kg)", "50 kg bags"]] +
                             [[k, str(v), str(c["bags_50kg"][k])] for k, v in c["fertilizers_kg"].items()],
                             col_widths=(55 * mm, 50 * mm, 50 * mm)))
            for s in c["schedule"]:
                el.append(Paragraph("- " + s, st["body"]))
        img = _image(os.path.join(static_dir, result["image"]) if result.get("image") else None)
        if img:
            el.append(Spacer(1, 8))
            el.append(img)
    return _doc("Fertilizer Recommendation Report", farmer_name, build)


def disease_report(result: dict, info: dict | None, uploaded_image: str | None, products: list[dict],
                   farmer_name: str | None = None) -> BytesIO:
    def build(el, st):
        top = result["top"]
        el.append(Paragraph("Diagnosis", st["h"]))
        rows = [["Field", "Value"], ["Crop", top["crop"]], ["Condition", top["condition"]],
                ["Confidence", f"{top['confidence'] * 100:.1f}%"], ["Model", result.get("model", "")],
                ["Other candidates", ", ".join(f"{p['name']} ({p['confidence'] * 100:.0f}%)" for p in result["predictions"][1:])]]
        el.append(_table(rows))
        img = _image(uploaded_image)
        if img:
            el.append(Spacer(1, 6))
            el.append(img)
        if info:
            el.append(Paragraph("About the disease", st["h"]))
            el.append(Paragraph(info.get("description", ""), st["body"]))
            el.append(Paragraph("Prevention and treatment", st["h"]))
            el.append(Paragraph(info.get("prevention", ""), st["body"]))
        if products:
            el.append(Paragraph("Recommended products (KrishiDisha marketplace)", st["h"]))
            el.append(_table([["Product", "Price", "Unit"]] + [[p["name"], f"Rs {p['price']}", p["unit"]] for p in products[:5]],
                             col_widths=(95 * mm, 30 * mm, 35 * mm)))
    return _doc("Disease Detection Report", farmer_name, build)


def yield_report(inputs: dict, result: dict, farmer_name: str | None = None) -> BytesIO:
    def build(el, st):
        el.append(Paragraph("Inputs", st["h"]))
        el.append(_table([["Parameter", "Value"]] + [[k, str(v)] for k, v in inputs.items()]))
        el.append(Paragraph("Prediction", st["h"]))
        el.append(_table([["Field", "Value"], ["Predicted yield", f"{result['predicted_yield']} {result['unit']}"],
                          ["Estimated production", f"{result['estimated_production']} tonnes"]]))
        for t in result.get("tips", []):
            el.append(Paragraph("- " + t, st["body"]))
    return _doc("Crop Yield Prediction Report", farmer_name, build)
