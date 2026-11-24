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


def tile(x, y, w, h, big, small):
    rect(x, y, w, h, PALE, MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    tbox(x + 1.0, y + 0.7, w - 2.0, 2.9, [big], size=72, bold=True)
    tbox(x + 1.0, y + 3.7, w - 2.0, h - 4.0, [small], size=24, color=GREY, after=0)

# ---------- what we are doing
y0 = 16.4
rect(L, y0, W, 11.4, INK, name="Purpose band")
tbox(L + 2.0, y0 + 1.5, W - 4.0, 1.2, ["WHAT WE ARE DOING"], size=26, font=MONO, color=YEL)
tbox(L + 2.0, y0 + 3.2, W - 4.0, 7.0, [[
    ("Teaching a computer to ", {}), ("measure the bubbles", {"color": YEL}),
    (" in flotation video, so researchers no longer size them by hand.", {})]],
    size=60, bold=True, color=WHITE, spacing=0.95)
tbox(L, y0 + 12.6, W, 2.6, [[("Why it matters: ", {"bold": True}),
    ("flotation recovers minerals on air bubbles, and smaller bubbles catch more. Engineers compare cells by the Sauter mean diameter, ", {}),
    ("D32 = Σd³/Σd²", {"bold": True}), (".", {})]], size=30)

# ---------- how it works: three pictures
y = 33.6
yy = section(L, y, W, "01", "How it works", "Video in, bubble size out")
PH = 23.0
pics = [("step1.png", "Probe video frame: crowded, murky, mostly out of focus"),
        ("step2.png", "The AI outlines every in-focus bubble"),
        ("step3.png", "Bubble-size distribution and D32 for the whole clip")]
slot = (W - 2 * 3.5) / 3
for i, (img, cap) in enumerate(pics):
    x = L + i * (slot + 3.5)
    iw, ih = Image.open(B + img).size
    pw = min(slot, PH * iw / ih)
    picture(B + img, x + (slot - pw) / 2, yy + 0.2, h=PH, name=f"Step {i+1} picture")
    c = rect(x, yy + PH + 0.8, 1.5, 1.5, YEL, MSO_SHAPE.OVAL, name=f"Step {i+1} number")
    tbox(x, yy + PH + 0.8, 1.5, 1.5, [str(i + 1)], size=26, bold=True, font=MONO, anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
    tbox(x + 2.0, yy + PH + 0.75, slot - 2.0, 2.6, [cap], size=26)
    if i < 2:
        a = s.shapes.add_shape(MSO_SHAPE.CHEVRON, Cm(x + slot + 0.85), Cm(yy + 0.2 + PH / 2 - 1.1), Cm(1.8), Cm(2.2))
        a.fill.solid(); a.fill.fore_color.rgb = INK; a.line.fill.background(); a.name = f"Arrow {i+1}"

# ---------- results: pulp | froth
y = 66.2
tw = (COLW - 1.5) / 2
yy = section(L, y, COLW, "02", "Pulp zone · probe video", "As accurate as a human, on new conditions")
tile(L, yy, tw, 7.6, "0.99", "D32 vs hand measurement, average over 8 unseen conditions")
tile(L + tw + 1.5, yy, tw, 7.6, "92%", "of hand-labelled bubbles found")
_, h = picture(B + "f_cv.png", L, yy + 8.6, w=COLW, name="Pulp cross-validation chart")
caption(L, yy + 9.0 + h, COLW, "Each clip tested after training on the other seven. Shaded: within ±5% of the hand value.")

yy = section(XR, y, COLW, "03", "Froth zone · column video", "Packed froth bubbles, no extra training")
tile(XR, yy, tw, 7.6, "86%", "of hand-outlined bubbles found, 98% of detections correct")
tile(XR + tw + 1.5, yy, tw, 7.6, "2.16 mm", "froth D32, steady (2.06–2.30 mm) over 7 minutes")
_, h = picture(B + "froth.jpg", XR, yy + 8.6, w=COLW, name="Froth Cellpose outlines")
caption(XR, yy + 9.0 + h, COLW, "Froth frame (left) and the outlines found by Cellpose, a general-purpose segmenter (right).")

# ---------- takeaways + tool
y = 99.8
rect(L, y, W, 15.6, PALE, MSO_SHAPE.ROUNDED_RECTANGLE, name="Takeaways card", radius=0.05)
tbox(L + 1.5, y + 1.2, 40, 1.2, ["TAKEAWAYS"], size=26, font=MONO, bold=True)
tbox(L + 1.5, y + 3.2, 54.0, 11.5, [
    [("✓  ", {"bold": True}), ("Bubble size now comes from video in one click", {})],
    [("✓  ", {"bold": True}), ("Pulp D32 within 1–4% of hand measurement, on conditions never seen in training", {})],
    [("✓  ", {"bold": True}), ("Froth measured with no froth-specific training, under 1 minute per frame", {})],
    [("Next: ", {"bold": True}), ("test on 9 new froth videos; fine-tune only where a quick check fails", {})]], size=34, after=0.5)
gx = L + W - 1.5 - 12.4 * 1420 / 920
gw, gh = picture(B + "gui.png", gx, y + 1.2, h=12.4, name="BubbleMetrix screenshot")
caption(gx, y + 1.5 + gh, gw, "BubbleMetrix, the desktop tool built in this project", h=1.4)

prs.save(B + "poster.pptx")
print("saved")
