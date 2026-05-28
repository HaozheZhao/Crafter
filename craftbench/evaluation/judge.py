"""CraftBench reference judge.

Each figure is scored ALONE (no side-by-side comparison, which removes the
position bias of pairwise judging) on a compact set of task- and content-type-
specific aspects rated 0-10. A weighted mean turns the per-aspect scores into a
single total per image. The candidate's margin over the human-drawn target then
yields a per-sample verdict in {Model, Tie, Human} under a calibrated tie band.

Aspects by task:
  * text-to-image           -> content_faithfulness, readability, and a
                               style-specific format aspect (academic / poster /
                               infographic).
  * the three reference-conditioned tasks replace the format aspect with an
    input_fidelity aspect tailored to how each task uses its conditioning input:
      - key-element composition: the specific provided elements must be reused;
      - mask-completion:         the blank region must be substantively filled
                                 while the rest is preserved;
      - sketch-conditioned:      the sketch layout must be followed AND polished
                                 into a clean figure (a near-copy is a failure).

content_faithfulness and input_fidelity carry the most weight (3.0); readability
and format the least (1.0-1.5). The source paper context is used for content
correctness only and does not raise input_fidelity. On academic text-to-image
inputs this reduces to a standard referenced judge.
"""
from __future__ import annotations
from .api import call_judge
from .images import img_part

MODEL = "google/gemini-3.5-flash"
SEED = 42
TIE_BAND = 0.30
MAX_SIDE = 1280

# ============================ text-to-image rubric ============================
ASPECTS = {
  "faithfulness": ("content_faithfulness",
    "Does it convey the content in the caption/brief ACCURATELY and COMPLETELY? Check the "
    "specific components, relationships and flow named in the caption are present and correct. "
    "CRITICAL: scrutinize the text in the figure — are labels REAL, specific and correct, or "
    "are some garbled / nonsensical / vague placeholders / AI-gibberish? Are any components "
    "fabricated or missing? Plausible-looking but wrong = low. 9-10 only if every caption "
    "element is present, correct, and all text is genuine and legible."),
  "readability": ("readability",
    "Is the information easy to extract: legible text, clear visual flow, sensible grouping, "
    "no clutter/overlap/spaghetti arrows? 9-10 only if a reader navigates it effortlessly."),
  "fmt_academic": ("format_academic",
    "Is this a CLEAN ACADEMIC FIGURE: precise single-figure diagram, restrained styling, "
    "crisp components/arrows, NO poster banner, NO infographic cartoons/decoration? Penalize "
    "decorative fluff, oversized titles, or clip-art that a top venue (NeurIPS/CVPR) would reject."),
  "fmt_poster": ("format_poster",
    "Is this a real CONFERENCE POSTER: large title banner, author/affiliation line, >=3 "
    "distinct sections with headers, AND rich content (results figures, tables, method "
    "diagrams)? Penalize a thin skeleton, a single plain figure, or missing banner/authors."),
  "fmt_infographic": ("format_infographic",
    "Is this a CASUAL EXPLAINER INFOGRAPHIC (Distill/Quanta/blog style): illustrative "
    "icons/characters or a visual metaphor, narrative callouts, friendly non-academic "
    "typography/color? Penalize a dry academic block diagram pretending to be an infographic."),
}
STYLE_ASPECT = {"academic": "fmt_academic", "poster": "fmt_poster", "infographic": "fmt_infographic"}
T2I_WEIGHTS = {"content_faithfulness": 3.0, "readability": 1.5,
               "format_academic": 1.0, "format_poster": 1.5, "format_infographic": 1.5}

T2I_SYSTEM = """\
You are a STRICT, skeptical reviewer of scientific figures. You will see the original \
request given to an illustrator, optionally an INPUT IMAGE they were told to work from, \
and ONE candidate figure. One of the candidates you review is a real human-made figure \
from a published paper/poster/blog; the other is AI-generated. AI figures often look \
glossy yet hide subtle flaws — garbled or fake text, plausible-but-wrong components, \
generic placeholders, or (for edit tasks) silently ignoring/altering the input image. \
Hunt for these.

Score each aspect 0-10 with CONSERVATIVE anchors:
  9-10 = flawless on this aspect after careful search (you found NO weakness)
  7-8  = good, minor issues
  5-6  = acceptable but clearly flawed
  3-4  = major problems
  0-2  = fails / unrelated / garbled
Do NOT default to 9-10. Most figures have at least minor flaws; if you are about to give \
9-10, re-examine and try to find a weakness first. Reward correctness and communication, \
not surface polish.

Aspects to score for THIS sample:
{aspects}

First list concrete weaknesses you actually observe (be specific; "none" only if truly \
flawless), then score. Return STRICT JSON only:
{{"weaknesses": ["<specific flaw>", ...], "scores": {{{keys}}}}}"""


def _t2i_brief(rec):
    cap = (rec.get("caption") or "").strip(); instr = (rec.get("instruction") or "").strip()
    pctx = (rec.get("paper_context") or "").strip()
    L = [f"Task: {rec['task']}  |  Requested figure style: {rec.get('style')}"]
    if cap: L.append(f"\nTarget figure caption:\n{cap}")
    if pctx: L.append(f"\nSOURCE PAPER CONTEXT (judge faithfulness against this):\n{pctx}")
    if instr: L.append(f"\nIllustrator's brief:\n{instr}")
    return "\n".join(L)


def _score_t2i(rec, img):
    keys = ["faithfulness", "readability", STYLE_ASPECT.get(rec.get("style"), "fmt_academic")]
    labels = [ASPECTS[k][0] for k in keys]
    asp_text = "\n".join(f"- {ASPECTS[k][0]}: {ASPECTS[k][1]}" for k in keys)
    sysp = T2I_SYSTEM.format(aspects=asp_text, keys=", ".join(f'"{l}": <0-10>' for l in labels))
    content = [{"type": "text", "text": "ORIGINAL REQUEST\n================\n" + _t2i_brief(rec)},
               {"type": "text", "text": "\nCandidate figure to score:"},
               img_part(img, max_side=MAX_SIDE)]
    out = call_judge(sysp, content, model=MODEL, temperature=0.0, seed=SEED)
    return _clean(out, labels)


# ======================= reference-conditioned rubrics =======================
EDIT_WEIGHTS = {"input_fidelity": 3.0, "content_faithfulness": 1.5, "readability": 1.0}

# key-element composition: the provided elements must actually be reused.
KEYELEMS_SYS = """\
You are STRICTLY evaluating a figure for an EDIT task whose input is a set of SPECIFIC visual \
elements (icons/photos/charts) to USE (the INPUT IMAGE).
The PRIMARY criterion is input_fidelity: compare the OUTPUT to the INPUT elements. Are the SPECIFIC provided elements actually \
REUSED (recognizably the same icons/photos/charts), not replaced by similar-but-different ones \
the model invented? A figure that ignores the provided elements or substitutes its own — even if \
polished or paper-faithful — MUST score LOW (0-4). 8-10 ONLY if most/all provided elements are clearly reused.

CRITICAL: the SOURCE PAPER CONTEXT (if shown) is provided ONLY so you can check the figure's
labels/content are technically correct (content_faithfulness). It must NOT raise input_fidelity:
following the input is judged PURELY by comparing the OUTPUT IMAGE to the INPUT IMAGE. A figure
that matches the paper but ignores the input image is a FAILED edit and scores low on input_fidelity.

Score 0-10:
- content_faithfulness: content is correct vs the caption{papernote}; text genuine/legible.
- readability: legible, clear, organized.
- input_fidelity: STRICT input-following as defined above (OUTPUT vs INPUT IMAGE only).

List concrete weaknesses, then score. STRICT JSON only:
{{"weaknesses": [...], "scores": {{"content_faithfulness": <n>, "readability": <n>, "input_fidelity": <n>}}}}"""

# mask-completion & sketch: the input is a brief to fill/polish, not to reproduce.
FILL_SYS = """\
You are a STRICT reviewer of a scientific-figure EDIT task. The author was given {inp} (the INPUT IMAGE).
Score 0-10:
- content_faithfulness: the content is correct vs the caption{papernote}; text genuine/legible.
- readability: legible, clear, organized.
- input_fidelity: {fid}
The SOURCE PAPER CONTEXT (if shown) is for content-correctness ONLY and must NOT raise input_fidelity.
List concrete weaknesses, then score. STRICT JSON only:
{{"weaknesses": [...], "scores": {{"content_faithfulness": <n>, "readability": <n>, "input_fidelity": <n>}}}}"""

FILL_CFG = {
  "sketch": dict(
    inp="a ROUGH SKETCH that specifies the intended LAYOUT / rough semantic structure (panels, boxes, labels, arrows, reading order)",
    fid=("input_fidelity scores TWO things together: (a) LAYOUT-FOLLOWING — the output keeps the "
         "sketch's panel arrangement, labels, arrow connections and reading order; (b) POLISH — the "
         "output is a genuinely REFINED, clean publication-quality figure (crisp shapes, real "
         "typography, proper icons/diagrams), NOT a near-copy that retains the rough sketch's hand-"
         "drawn/draft appearance. A NEAR-COPY of the rough sketch (little refinement) is a FAILED "
         "task and MUST score LOW (0-4) EVEN IF the layout matches perfectly — the sketch is a brief "
         "to polish, not a target to reproduce. 8-10 ONLY if layout is faithful AND clearly polished.")),
  "inpaint": dict(
    inp="a PARTIAL figure with ONE region blanked white that must be FILLED",
    fid=("input_fidelity scores TWO things together: (a) SUBSTANTIVE FILL — the output actually fills "
         "the blank region with substantive, appropriate content (panels/text/diagrams per the "
         "description); an output that leaves the region empty or merely REPRODUCES the input without "
         "filling is a FAILED inpaint and MUST score LOW (0-3); (b) PRESERVATION — the rest of the "
         "figure is kept unchanged. 8-10 ONLY if there is a correct substantive fill AND the rest is "
         "preserved. Do not reward a near-copy of the input that skips the fill.")),
}


def _edit_brief(rec):
    cap = (rec.get("caption") or "").strip(); region = (rec.get("masked_region") or "").strip()
    pctx = (rec.get("paper_context") or "").strip()
    L = [f"Target figure caption: {cap}"]
    if region: L.append(f"The blanked region should contain: {region}")
    if pctx: L.append(f"\nSOURCE PAPER CONTEXT (content-correctness only; NOT for input_fidelity):\n{pctx}")
    return "\n".join(L)


def _score_edit(rec, img):
    t = rec["task"]
    pn = " and paper context" if rec.get("paper_context") else ""
    if t == "keyelems":
        sysp = KEYELEMS_SYS.format(papernote=pn)
        intro = "\nINPUT IMAGE given to the author:"
        result_label = "\nResult to score (compare its structure/elements to the INPUT above):"
    else:
        c = FILL_CFG[t]
        sysp = FILL_SYS.format(inp=c["inp"], fid=c["fid"], papernote=pn)
        intro = "\nINPUT IMAGE (a brief to follow/fill — NOT to reproduce):"
        result_label = "\nResult to score:"
    content = [{"type": "text", "text": _edit_brief(rec)},
               {"type": "text", "text": intro}, img_part(rec["input_path"], max_side=MAX_SIDE),
               {"type": "text", "text": result_label}, img_part(img, max_side=MAX_SIDE)]
    out = call_judge(sysp, content, model=MODEL, temperature=0.0, seed=SEED)
    return _clean(out, ["content_faithfulness", "readability", "input_fidelity"])


# ================================ shared API ================================
def _clean(out, labels):
    sc = out.get("scores") or {}
    clean = {}
    for l in labels:
        try:
            clean[l] = max(0.0, min(10.0, float(sc.get(l))))
        except (TypeError, ValueError):
            clean[l] = None
    return {"scores": clean, "weaknesses": out.get("weaknesses", []), "err": out.get("_error")}


def score_image(rec, img):
    """Score ONE figure (``img`` = path) for the sample ``rec``."""
    return _score_t2i(rec, img) if rec["task"] == "t2i" else _score_edit(rec, img)


def weighted_total(scores):
    w = EDIT_WEIGHTS if "input_fidelity" in scores else T2I_WEIGHTS
    items = [(k, v) for k, v in scores.items() if v is not None]
    if not items:
        return None
    return sum(w.get(k, 1.0) * v for k, v in items) / sum(w.get(k, 1.0) for k, _ in items)


def verdict(margin, tie_band=TIE_BAND):
    """Margin = candidate_total - target_total -> {Model, Tie, Human}."""
    return "Tie" if abs(margin) <= tie_band else ("Model" if margin > 0 else "Human")
