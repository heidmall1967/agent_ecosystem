"""Build the running itinerary PDF from plan.toml + live repo state (results/, notes/adr/).

Usage: uv run python itinerary/build.py [--out itinerary/itinerary.pdf]
"""
import argparse
import datetime as dt
import json
import random
import re
import tomllib
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")

pdfmetrics.registerFont(TTFont("Sans", FONT_DIR / "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("Sans-Bold", FONT_DIR / "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Mono", FONT_DIR / "DejaVuSansMono.ttf"))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans", boldItalic="Sans-Bold")

INK, MUTED, RULE = colors.HexColor("#1f2933"), colors.HexColor("#616e7c"), colors.HexColor("#d9dee4")
ACCENT = colors.HexColor("#2f5d8a")
STATUS_COLOR = {"done": colors.HexColor("#2e7d4f"), "in-progress": colors.HexColor("#b7791f"),
                "todo": colors.HexColor("#8a94a3")}

S = {
    "title": ParagraphStyle("title", fontName="Sans-Bold", fontSize=22, leading=27, textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("sub", fontName="Sans", fontSize=10, leading=14, textColor=MUTED),
    "h1": ParagraphStyle("h1", fontName="Sans-Bold", fontSize=16, leading=20, textColor=INK, spaceBefore=4, spaceAfter=2),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=11, leading=14, textColor=ACCENT, spaceBefore=9, spaceAfter=3),
    "body": ParagraphStyle("body", fontName="Sans", fontSize=9.2, leading=12.6, textColor=INK),
    "item": ParagraphStyle("item", fontName="Sans", fontSize=9.2, leading=12.6, textColor=INK, leftIndent=14, firstLineIndent=-14, spaceAfter=2),
    "opt": ParagraphStyle("opt", fontName="Sans", fontSize=9, leading=12, textColor=INK, leftIndent=30, firstLineIndent=-14),
    "small": ParagraphStyle("small", fontName="Sans", fontSize=8, leading=10.5, textColor=MUTED),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8.2, leading=10.4, textColor=INK),
    "cellc": ParagraphStyle("cellc", fontName="Sans", fontSize=8.2, leading=10.4, textColor=INK, alignment=TA_CENTER),
}


def md(text: str) -> str:
    """Escape for reportlab markup; `code` -> mono."""
    return re.sub(r"`([^`]+)`", r'<font name="Mono">\1</font>', escape(text))


def P(text, style="body"):
    return Paragraph(md(text), S[style])


def H(html, style="body"):
    """Paragraph from trusted markup (caller escapes any dynamic parts)."""
    return Paragraph(html, S[style])


def bullets(items, mark="•"):
    return [Paragraph(f"{mark}&nbsp;&nbsp;{md(t)}", S["item"]) for t in items]


def checklist(items):
    out = []
    for t in items:
        done = t.startswith("[x] ")
        out.append(Paragraph(f'{"☑" if done else "☐"}&nbsp;&nbsp;{md(t[4:] if done else t)}', S["item"]))
    return out


def shuffled(week_n: int, i: int, quiz: dict) -> tuple[list[str], int]:
    """Deterministic option shuffle so correct answers are not always in the same slot."""
    order = list(range(len(quiz["options"])))
    random.Random(f"{week_n}-{i}").shuffle(order)
    return [quiz["options"][j] for j in order], order.index(quiz["answer"])


def week_dates(start: dt.date, n: int) -> tuple[dt.date, dt.date]:
    a = start + dt.timedelta(weeks=n - 1)
    return a, a + dt.timedelta(days=6)


def progress_bar(frac: float, width=170 * mm, height=7 * mm) -> Drawing:
    d = Drawing(width, height + 4)
    d.add(Rect(0, 2, width, height, fillColor=RULE, strokeColor=None))
    d.add(Rect(0, 2, width * frac, height, fillColor=STATUS_COLOR["done"], strokeColor=None))
    d.add(String(width / 2, 2 + height / 2 - 3, f"{frac:.0%}", fontName="Sans-Bold", fontSize=8,
                 fillColor=INK, textAnchor="middle"))
    return d


def table(rows, widths, header=True, zebra=True):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.3, RULE),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), ACCENT), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    if zebra:
        style += [("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fa")])]
    t.setStyle(TableStyle(style))
    return t


def hdr(*cells):
    return [Paragraph(f'<font color="white"><b>{escape(c)}</b></font>', S["cell"]) for c in cells]


def latest_results() -> list[dict]:
    latest = {}
    for f in sorted((ROOT / "results").glob("*.json")):  # timestamped names sort chronologically
        d = json.loads(f.read_text())
        latest[d["pipeline"]] = d | {"file": f.name}
    return sorted(latest.values(), key=lambda d: d["file"])


def fmt(x, pct=True):
    return "–" if x is None else (f"{x:.0%}" if pct else f"{x:.1f}")


def cover(plan, weeks, today):
    start = plan["start_date"]
    n = len(weeks)
    done = sum(w["status"] == "done" for w in weeks)
    cal_week = min(max((today - start).days // 7 + 1, 1), n)
    expected_done = cal_week - 1
    hours = sum(w.get("hours", 0) for w in weeks)
    planned_hours = expected_done * plan["hours_per_week"] + plan["hours_per_week"] * min(((today - start).days % 7 + 1) / 7, 1)
    gap = expected_done - done
    pace = "on track" if gap <= 0 else f"behind by {gap} week{'s' * (gap > 1)}"
    current = next((w for w in weeks if w["status"] == "in-progress"), next((w for w in weeks if w["status"] == "todo"), weeks[-1]))

    story = [Paragraph(escape(plan["title"]), S["title"]),
             Paragraph(f"Generated {today:%a %d %b %Y} · {n} weeks · {plan['hours_per_week']} h/week · "
                       f"start {start:%d %b %Y}", S["sub"]), Spacer(1, 10),
             progress_bar(done / n), Spacer(1, 4),
             table([[H(f"<b>Weeks done</b><br/>{done} / {n}", "cell"), H(f"<b>Calendar week</b><br/>{cal_week}", "cell"),
                     H(f"<b>Pace</b><br/>{pace}", "cell"), H(f"<b>Hours logged</b><br/>{hours} / ~{planned_hours:.0f} planned to date", "cell")]],
                   [42.5 * mm] * 4, header=False, zebra=False),
             Spacer(1, 6),
             H(f"<b>Now:</b> Week {current['n']}: {md(current['title'])} ({md(current['levels'])})"),
             Paragraph("Weekly rhythm", S["h2"]), *bullets(plan["week_rhythm"])]

    story.append(Paragraph("Measured progress (latest run per pipeline)", S["h2"]))
    res = latest_results()
    if res:
        rows = [hdr("Pipeline", "Run", "n", "Answer acc", "Keyword recall", "hit@k", "p50 s", "p95 s")]
        for d in res:
            s = d["summary"]
            rows.append([P(f"`{d['pipeline']}`", "cell"), P(d["file"].rsplit("-", 2)[-2], "cell"), P(str(s["all"]["n"]), "cellc"),
                         P(fmt(s["all"]["answer_acc"]), "cellc"), P(fmt(s["all"]["keyword_recall"]), "cellc"),
                         P(fmt(s["all"]["hit@k"]), "cellc"), P(fmt(s["latency_p50_s"], False), "cellc"),
                         P(fmt(s["latency_p95_s"], False), "cellc")])
        story.append(table(rows, [32 * mm, 20 * mm, 10 * mm, 20 * mm, 24 * mm, 16 * mm, 16 * mm, 16 * mm]))
    else:
        story.append(P("No eval runs yet. Run `uv run python -m eval.harness baseline`.", "small"))

    story.append(Paragraph("Decision records", S["h2"]))
    adrs = sorted(p for p in (ROOT / "notes/adr").glob("*.md") if not p.name.startswith("0000"))
    if adrs:
        for p in adrs:
            first = p.read_text().splitlines()[0].lstrip("# ").strip() if p.read_text().strip() else p.stem
            story.append(Paragraph(f"☑&nbsp;&nbsp;{md(first)}", S["item"]))
    else:
        story.append(P("None yet. Copy `notes/adr/0000-template.md` at the end of each week.", "small"))

    story.append(Paragraph("Overview", S["h2"]))
    rows = [hdr("Wk", "Dates", "Focus", "Levels", "Status", "Quiz", "Hrs")]
    for w in weeks:
        a, b = week_dates(start, w["n"])
        c = STATUS_COLOR.get(w["status"], MUTED).hexval()[2:]
        rows.append([P(str(w["n"]), "cellc"), P(f"{a:%d %b}–{b:%d %b}", "cell"), P(w["title"], "cell"),
                     P(w["levels"], "cell"), Paragraph(f'<font color="#{c}"><b>{w["status"]}</b></font>', S["cell"]),
                     P(w.get("quiz_score") or "–", "cellc"), P(str(w.get("hours", 0)), "cellc")])
    story.append(table(rows, [10 * mm, 27 * mm, 56 * mm, 30 * mm, 23 * mm, 13 * mm, 13 * mm]))
    return story


def week_page(plan, w):
    a, b = week_dates(plan["start_date"], w["n"])
    c = STATUS_COLOR.get(w["status"], MUTED).hexval()[2:]
    meta = f'{w["phase"]} · {a:%a %d %b}–{b:%a %d %b %Y} · {w["levels"]} · <font color="#{c}"><b>{w["status"].upper()}</b></font>'
    story = [PageBreak(), Paragraph(f'Week {w["n"]}: {md(w["title"])}', S["h1"]), Paragraph(meta, S["sub"])]

    tracked = [f"Hours: {w.get('hours', 0)} / {plan['hours_per_week']}", f"Quiz: {w.get('quiz_score') or '–'}",
               f"Completed: {w.get('completed') or '–'}"]
    story += [Spacer(1, 3), P(" · ".join(tracked), "small")]
    if w.get("notes"):
        story.append(P(f"Notes: {w['notes']}", "small"))

    story += [Paragraph("Goals", S["h2"]), *bullets(w["goals"]),
              Paragraph("Reading", S["h2"]), *bullets(w["reading"], "▸"),
              Paragraph("Build checklist", S["h2"]), *checklist(w["tasks"]),
              KeepTogether([Paragraph("Assignment", S["h2"]), H(f"<b>{md(w['assignment'])}</b>"), Spacer(1, 3),
                            P("Acceptance criteria:", "small"), *checklist(w["acceptance"])])]

    story.append(Paragraph("Quiz (answer before reading; key at the end)", S["h2"]))
    for i, q in enumerate(w["quiz"], 1):
        opts, _ = shuffled(w["n"], i, q)
        story.append(KeepTogether([Paragraph(f"<b>Q{w['n']}.{i}</b>&nbsp;&nbsp;{md(q['q'])}", S["item"]),
                                   *[Paragraph(f"{'ABCD'[j]})&nbsp;&nbsp;{md(o)}", S["opt"]) for j, o in enumerate(opts)],
                                   Spacer(1, 3)]))

    story += [Paragraph("Architect discussion questions (write answers in the ADR)", S["h2"]), *bullets(w["questions"], "?")]
    return story


def answer_key(weeks):
    story = [PageBreak(), Paragraph("Answer key", S["h1"]), Spacer(1, 4)]
    rows = [hdr("Q", "Answer", "Why")]
    for w in weeks:
        for i, q in enumerate(w["quiz"], 1):
            opts, k = shuffled(w["n"], i, q)
            rows.append([P(f"{w['n']}.{i}", "cellc"), H(f"<b>{'ABCD'[k]}</b>", "cellc"),
                         P(f"{opts[k]}. {q['why']}", "cell")])
    story.append(table(rows, [12 * mm, 18 * mm, 142 * mm]))
    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "Running itinerary · edit itinerary/plan.toml and rebuild")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"{doc.page}")
    canvas.restoreState()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HERE / "itinerary.pdf", type=Path)
    args = ap.parse_args()

    plan = tomllib.loads((HERE / "plan.toml").read_text())
    weeks = plan["week"]
    for w in weeks:
        assert len(w["quiz"]) == 4 and all(0 <= q["answer"] < len(q["options"]) for q in w["quiz"]), f"week {w['n']} quiz"
        assert w["status"] in STATUS_COLOR, f"week {w['n']} status {w['status']!r}"

    today = dt.date.today()
    story = cover(plan, weeks, today)
    for w in weeks:
        story += week_page(plan, w)
    if plan.get("optional"):
        story += [Spacer(1, 6), Paragraph("Optional", S["h2"]), *checklist(plan["optional"]["items"])]
    story += answer_key(weeks)

    doc = SimpleDocTemplate(str(args.out), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=18 * mm, title=plan["title"], author="full_path")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
