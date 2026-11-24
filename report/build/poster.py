"""A1 Taste of Research poster on the UNSW Faculty template.
The template's yellow header, crest, background and title/by-line fonts are untouched; only their words change.
Body: Arial (UNSW body face, ships with Office) + Roboto Mono labels (the template's by-line face)."""
import copy
from pptx import Presentation
from pptx.util import Cm, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from PIL import Image

B = "/tmp/claude-0/brand/"
prs = Presentation("/tmp/claude-0/pt/base.pptx")
s = prs.slides[0]
INK = RGBColor(0x23, 0x1F, 0x20)
GREY = RGBColor(0x5A, 0x5A, 0x5A)
SOFT = RGBColor(0xCF, 0xCF, 0xCF)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
YEL = RGBColor(0xFF, 0xDC, 0x00)
PALE = RGBColor(0xFF, 0xF6, 0xC2)
LIGHT = RGBColor(0xF3, 0xF3, 0xF3)
SANS, MONO = "Arial", "Roboto Mono"

# ---------- template header: words only
tb = {sh.name: sh for sh in s.shapes}
def set_lines(shape, lines):
    tf = shape.text_frame
    p0 = tf.paragraphs[0]
    for _ in lines[1:]:
        tf._txBody.append(copy.deepcopy(p0._p))
    for p, t in zip(tf.paragraphs, lines):
        p.runs[0].text = t
title, byline = tb["TextBox 5"], tb["TextBox 6"]
set_lines(title, ["Automated bubble sizing"])
title.width = Cm(80.0) - title.left
set_lines(byline, ["Tristan Trieu  |  Supervisor: Prof Seher Ata",
                   "School of Minerals & Energy Resources Engineering"])
byline.width = Cm(82.0) - byline.left
byline.top = Cm(8.1)

L, W = 3.0, 78.1
COLW, GAP = 37.55, 3.0
XR = L + COLW + GAP

def tbox(x, y, w, h, paras, size=30, color=INK, bold=False, font=SANS, anchor=MSO_ANCHOR.TOP,
         align=PP_ALIGN.LEFT, after=0.45, spacing=None, name=None):
    bx = s.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(h))
    if name: bx.name = name
    tf = bx.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if spacing: p.line_spacing = spacing
        for t, o in (para if isinstance(para, list) else [(para, {})]):
            r = p.add_run(); r.text = t; f = r.font
            f.name = o.get("font", font); f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold); f.color.rgb = o.get("color", color)
        p.space_after = Pt(size * after)
    return bx

def rect(x, y, w, h, fill, shape=MSO_SHAPE.RECTANGLE, name=None, radius=None):
    b = s.shapes.add_shape(shape, Cm(x), Cm(y), Cm(w), Cm(h))
    b.fill.solid(); b.fill.fore_color.rgb = fill; b.line.fill.background(); b.shadow.inherit = False
    if radius is not None: b.adjustments[0] = radius
    if name: b.name = name
    return b

def section(x, y, w, num, label, headline, size=44):
    """Mono label ('01  WHY IT MATTERS') over a sentence headline that states the finding."""
    chip = rect(x, y + 0.05, 1.9, 1.15, YEL, name=f"Label {num}")
    tbox(x, y + 0.05, 1.9, 1.15, [num], size=24, bold=True, font=MONO, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
    tbox(x + 2.5, y + 0.05, w - 2.5, 1.15, [label.upper()], size=24, font=MONO, color=GREY, anchor=MSO_ANCHOR.MIDDLE)
    tbox(x, y + 1.75, w, 2.2, [headline], size=size, bold=True, spacing=0.95)
    return y + 4.4

def picture(path, x, y, w=None, h=None, name=None):
    iw, ih = Image.open(path).size
    if w is None: w = h * iw / ih
    if h is None: h = w * ih / iw
    p = s.shapes.add_picture(path, Cm(x), Cm(y), Cm(w), Cm(h))
    if name: p.name = name
    return w, h

def caption(x, y, w, t, h=2.2):
    tbox(x, y, w, h, [t], size=23, color=GREY, after=0)

# ---------- 1. the main finding (billboard band)
y0 = 16.4
rect(L, y0, W, 17.6, INK, name="Main finding band")
tbox(L + 2.0, y0 + 1.6, W - 4.0, 1.2, ["THE MAIN FINDING"], size=26, font=MONO, color=YEL)
tbox(L + 2.0, y0 + 3.3, W - 4.0, 6.0, [[
    ("A deep-learning tool now measures flotation bubble size ", {}),
    ("as accurately as a human expert", {"color": YEL}),
    (", even on operating conditions it has never seen.", {})]],
    size=60, bold=True, color=WHITE, spacing=0.95)
stats = [("0.99", "D32 ratio, model / hand,\nacross 8 unseen conditions"),
         ("0.62→0.90", "recall on the large bubbles\nthat dominate D32"),
         ("9%", "of detections were real\nmistakes after a hand audit"),
         ("< 1 min", "per froth frame; 86% of\nhand-outlined bubbles found")]
cw = (W - 4.0) / 4
for i, (big, small) in enumerate(stats):
    x = L + 2.0 + i * cw
    tbox(x, y0 + 10.4, cw - 0.8, 3.2, [big], size=80, bold=True, color=YEL)
    tbox(x, y0 + 13.7, cw - 0.8, 3.0, [small], size=24, color=SOFT, after=0)

# ---------- 2. why + rules
y = 36.0
yy = section(L, y, COLW, "01", "Why it matters", "Smaller bubbles catch more mineral")
tbox(L, yy, COLW, 6.6, [
    "Flotation recovers minerals by attaching them to rising air bubbles. For the same air, smaller bubbles carry more surface, so bubble size sets how much a cell can recover.",
    [("Engineers compare cells by the Sauter mean diameter, ", {}), ("D32 = Σd³/Σd²", {"bold": True}),
     (". Measuring it by hand from murky probe video is slow and subjective.", {})]], size=28)
yy = section(XR, y, COLW, "02", "Ground rules", "Measure only what an expert could")
tbox(XR, yy, COLW, 6.6, [
    [("Label: ", {"bold": True}), ("trace sharp bubbles, ignore part-hidden ones, leave blurred ones as background (383 labelled).", {})],
    [("Test: ", {"bold": True}), ("only on video clips the model never trained on, against an independent hand count.", {})],
    [("Benchmark: ", {"bold": True}), ("two hand passes by the same person agree on D32 to within 4%.", {})]], size=28)

# ---------- 3. how it works
y = 48.8
yy = section(L, y, W, "03", "How it works", "From probe video to D32 with one button")
steps = [("Probe video", "1024 × 1024 px, 30 fps; 8 rotor conditions"),
         ("Mask R-CNN", "round anchors, native resolution, ignore-aware loss"),
         ("Bubble outlines", "equivalent-area diameter of each in-focus bubble"),
         ("D32 + 95% CI", "keeps adding frames until D32 is known to ±5%"),
         ("BubbleMetrix", "desktop app; fine-tunes itself on new footage")]
sw, aw = 13.9, 2.15
for i, (h, b) in enumerate(steps):
    x = L + i * (sw + aw)
    rect(x, yy, sw, 6.0, LIGHT, MSO_SHAPE.ROUNDED_RECTANGLE, name=f"Step {i+1}", radius=0.06)
    c = rect(x + 0.8, yy + 0.8, 1.6, 1.6, YEL, MSO_SHAPE.OVAL, name=f"Step {i+1} number")
    tbox(x + 0.8, yy + 0.8, 1.6, 1.6, [str(i + 1)], size=26, bold=True, font=MONO, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
    tbox(x + 2.9, yy + 0.8, sw - 3.4, 1.6, [h], size=30, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    tbox(x + 0.8, yy + 2.9, sw - 1.6, 3.0, [b], size=23, color=GREY, after=0)
    if i < len(steps) - 1:
        a = s.shapes.add_shape(MSO_SHAPE.CHEVRON, Cm(x + sw + 0.6), Cm(yy + 2.3), Cm(aw - 1.2), Cm(1.4))
        a.fill.solid(); a.fill.fore_color.rgb = INK; a.line.fill.background(); a.name = f"Arrow {i+1}"

# ---------- 4/5. pulp results
y = 60.6
yy = section(L, y, COLW, "04", "Result · 8 unseen conditions", "D32 within 9% on every clip, 3% on six")
_, h = picture(B + "f_cv.png", L, yy + 0.2, w=COLW, name="Cross-validation chart")
caption(L, yy + 0.6 + h, COLW, "Leave-one-clip-out: train on 7 rotor clips, test on the 8th. Recall 0.92 ± 0.06, D32 ratio 0.99 ± 0.03, 377 bubbles.")
yy = section(XR, y, COLW, "05", "Result · what the model sees", "Every scored bubble found on a held-out frame")
_, h = picture(B + "pulp_gt_vs_model.png", XR, yy + 0.2, w=32.2, name="Hand labels vs model")
caption(XR, yy + 0.6 + h, COLW, "Hand labels (left) vs model (right), Fine inline frame 23. Extra outlines are mostly sharp bubbles nobody labelled.")

y = 85.6
yy = section(L, y, COLW, "06", "Result · hand audit", "Most “false positives” were labeller misses")
_, h = picture(B + "f_audit.png", L, yy + 0.2, w=34.0, name="False-positive audit chart")
caption(L, yy + 0.6 + h, COLW, "All 96 unmatched detections checked by hand. Precision rises from 0.40 against the labels to 0.80 once real bubbles count.")
yy = section(XR, y, COLW, "07", "Froth zone", "Cellpose separates packed froth bubbles")
_, h = picture(B + "froth.jpg", XR, yy + 0.2, w=30.0, name="Froth Cellpose outlines")
caption(XR, yy + 0.6 + h, COLW, "No froth training: recall 0.86 and precision 0.98 on 51 hand-outlined bubbles. D32 2.16 mm, 2.06–2.30 mm over 7 minutes.")

# ---------- 8. takeaways
y = 107.2
rect(L, y, W, 7.4, PALE, MSO_SHAPE.ROUNDED_RECTANGLE, name="Takeaways card", radius=0.08)
tbox(L + 1.5, y + 1.0, 18.0, 1.2, ["TAKEAWAYS"], size=26, font=MONO, color=INK, bold=True)
colw = (W - 3.0 - 2 * 1.5) / 3
items = [("Accurate.", "D32 matches a repeat human measurement on conditions the model never saw."),
         ("Honest.", "The inherited tool scored itself on its own labels; tested properly it found 5% of bubbles."),
         ("Next.", "Measure the probe scale with a target, add bright Venturi footage, record losslessly.")]
for i, (k, v) in enumerate(items):
    tbox(L + 1.5 + i * (colw + 1.5), y + 2.8, colw, 5.6, [[(k + " ", {"bold": True}), (v, {})]], size=28, after=0)

prs.save(B + "poster.pptx")
print("saved")
