"""
frame_annotator.py - outline froth bubbles on a whole measured region.

The measured region of one video frame (what the GUI measures) is shown at 2x,
plain resize, no filtering. You outline bubbles freehand. The region is split
into a 4 x 4 grid of tiles: when every front-layer bubble in a tile is
outlined, mark the tile COMPLETE (T). Only complete tiles are scored, and
inside them the labels are complete, so recall, precision, size error and the
D32 of the same area can all be compared with Cellpose. Tiles you don't finish
cost nothing.

  Left-drag          outline a bubble (let go: the ends are joined)
  Right-click        delete the bubble under the cursor
  U                  next click toggles UNSURE on a bubble (not scored); D = back to drawing
  T                  toggle COMPLETE on the tile under the cursor
  Mouse wheel, + -   zoom            Middle-drag or arrow keys   pan
  Space (hold)       hide outlines   C   view: plain / enhanced / pixels
  F                  flicker the neighbouring video frames      [ ]  step a frame
  Ctrl+Z / Ctrl+Y    undo / redo     S   save (also saves every 10 bubbles and on close)

Draw on the middle of the dark rim; front-layer bubbles only (skip the ghost
rings seen through a bubble). A bubble belongs to the tile its centre is in.

Output in <folder>/labels/: <stem>_labels.json (format "regions",
label_mode "frame_tiles"), <stem>_instances.png (16-bit), <stem>_complete.png.

  python frame_annotator.py [folder]     (default: to_label_frame beside this file)
"""
import copy
import json
import math
import os
import sys
import tkinter as tk
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageTk
import cv2
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = 260
TILES = (4, 4)                  # columns, rows
VIEWS = ("plain", "enhanced", "pixels")
ZOOMS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
SUB = 16


# ---------------------------------------------------------------------------
def poly_mask_bbox(poly, H, W, ss=4):
    """(mask, x0, y0): pixels at least half covered by the polygon, bbox crop only."""
    p = np.asarray(poly, float)
    x0, y0 = max(int(math.floor(p[:, 0].min())), 0), max(int(math.floor(p[:, 1].min())), 0)
    x1, y1 = min(int(math.ceil(p[:, 0].max())), W), min(int(math.ceil(p[:, 1].max())), H)
    if x1 <= x0 or y1 <= y0:
        return None, 0, 0
    big = np.zeros(((y1 - y0) * ss, (x1 - x0) * ss), np.uint8)
    q = np.round(((p - [x0, y0]) * ss - 0.5) * SUB).astype(np.int32)
    cv2.fillPoly(big, [q], 1, lineType=cv2.LINE_8, shift=4)
    m = big.reshape(y1 - y0, ss, x1 - x0, ss).mean(axis=(1, 3)) >= 0.5
    return m, x0, y0


def owner_map(shapes, H, W):
    """1..N label map; where outlines overlap, a pixel goes to the bubble it is deepest inside."""
    own = np.zeros((H, W), np.int32)
    best = np.zeros((H, W), np.float32)
    for k, s in enumerate(shapes, start=1):
        m, x0, y0 = poly_mask_bbox(s["poly"], H, W)
        if m is None or not m.any():
            continue
        dep = ndimage.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1].astype(np.float32)
        h, w = m.shape
        bv, ov = best[y0:y0 + h, x0:x0 + w], own[y0:y0 + h, x0:x0 + w]
        take = m & (dep > bv)
        ov[take] = k
        bv[take] = dep[take]
    return own


def poly_centroid(poly):
    p = np.asarray(poly, float)
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    a = cr.sum() / 2
    if abs(a) < 1e-9:
        return float(x.mean()), float(y.mean())
    return float(((x + x1) * cr).sum() / (6 * a)), float(((y + y1) * cr).sum() / (6 * a))


def clean_stroke(stroke):
    p = np.asarray(stroke, float)
    if len(p) >= 2:
        p = p[np.r_[True, np.hypot(*np.diff(p, axis=0).T) > 0.15]]
    return p


# ---------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self, folder):
        super().__init__()
        self.folder = folder
        self.out_dir = os.path.join(folder, "labels")
        os.makedirs(self.out_dir, exist_ok=True)
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".png") and not f.startswith("_"))
        if not files:
            messagebox.showerror("Nothing to label", "No .png in %s" % folder)
            self.destroy()
            return
        self.file = files[0]
        self.stem = os.path.splitext(self.file)[0]
        self.img = Image.open(os.path.join(folder, self.file)).convert("RGB")
        self.W, self.H = self.img.size
        self.frames = {0: self.img}
        for k in range(-3, 4):
            p = os.path.join(folder, "_context", "%s_f%+d.png" % (self.stem, k))
            if k and os.path.exists(p):
                self.frames[k] = Image.open(p).convert("RGB")
        tw, th = TILES
        self.tx = np.linspace(0, self.W, tw + 1).round().astype(int)
        self.ty = np.linspace(0, self.H, th + 1).round().astype(int)
        self.view, self.zoom_i = "plain", ZOOMS.index(2.0)
        self.ox = self.oy = 0.0                       # image coords at the canvas top-left
        self.frame_off, self.flicker, self.hide = 0, False, False
        self.mode = "draw"
        self.stroke = None
        self.since_save = 0
        self._cache = {}
        self.title("Froth frame annotator - %s" % self.file)
        self.configure(bg="#1e293b")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.cw, self.ch = max(600, min(sw - 340, 1400)), max(500, min(sh - 140, 900))
        self._build()
        self._read()
        self._recompute()
        self._render()

    # ---------------------------------------------------------------- UI
    def _build(self):
        side = tk.Frame(self, bg="#1e293b", padx=10, pady=10, width=290)
        side.pack(side="right", fill="y")
        self.canvas = tk.Canvas(self, width=self.cw, height=self.ch, bg="#0f172a",
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side="left", fill="both", expand=True)

        def lab(t, **kw):
            o = dict(bg="#1e293b", fg="#e2e8f0", justify="left", anchor="w", wraplength=270)
            o.update(kw)
            return tk.Label(side, text=t, **o)
        lab(self.file, font=("Segoe UI", 10, "bold")).pack(fill="x")
        lab("%d x %d px, %d x %d tiles" % (self.W, self.H, TILES[0], TILES[1]), fg="#94a3b8").pack(fill="x")
        self.lbl_mode = lab("", fg="#f472b6", font=("Segoe UI", 10, "bold"))
        self.lbl_mode.pack(fill="x", pady=(8, 0))
        self.lbl_count = lab("", font=("Segoe UI", 11, "bold"), fg="#4ade80")
        self.lbl_count.pack(fill="x", pady=(8, 0))
        self.lbl_detail = lab("", fg="#cbd5e1")
        self.lbl_detail.pack(fill="x")
        lab("").pack()
        lab("Outline each bubble freehand on the\n"
            "middle of its dark rim; let go and the\n"
            "ends are joined. Front layer only.\n\n"
            "When EVERY bubble in a tile is done,\n"
            "hover it and press T (turns green).\n"
            "Only green tiles are scored.\n\n"
            "Right-click: delete    U: unsure\n"
            "Wheel / + -: zoom    middle-drag / arrows: pan\n"
            "Space: hide outlines    C: view\n"
            "F: flicker frames    [ ]: step frame\n"
            "Ctrl+Z / Ctrl+Y    S: save", fg="#94a3b8", font=("Segoe UI", 9)).pack(fill="x")
        self.lbl_view = lab("", fg="#7dd3fc")
        self.lbl_view.pack(fill="x", pady=(10, 0))
        self.lbl_status = lab("", fg="#fbbf24")
        self.lbl_status.pack(fill="x", pady=(10, 0))
        tk.Button(side, text="Save", command=self._save, width=10).pack(anchor="w", pady=8)

        c = self.canvas
        c.bind("<ButtonPress-1>", self._l_down)
        c.bind("<B1-Motion>", self._l_drag)
        c.bind("<ButtonRelease-1>", self._l_up)
        c.bind("<ButtonPress-3>", self._r_click)
        c.bind("<ButtonPress-2>", self._m_down)
        c.bind("<B2-Motion>", self._m_drag)
        c.bind("<MouseWheel>", self._wheel)                       # Windows / macOS
        c.bind("<Button-4>", lambda e: self._zoom_at(e, +1))      # X11
        c.bind("<Button-5>", lambda e: self._zoom_at(e, -1))
        c.bind("<Motion>", lambda e: setattr(self, "mouse", (e.x, e.y)))
        c.bind("<Configure>", self._resized)
        self.mouse = (0, 0)
        for key, fn in (("<Control-z>", self._undo), ("<Control-y>", self._redo), ("s", self._save),
                        ("c", self._cycle_view), ("f", self._toggle_flicker), ("t", self._toggle_tile)):
            self.bind(key, lambda e, fn=fn: fn())
        self.bind("d", lambda e: self._set_mode("draw"))
        self.bind("u", lambda e: self._set_mode("unsure"))
        self.bind("<KeyPress-space>", lambda e: self._set_hide(True))
        self.bind("<KeyRelease-space>", lambda e: self._set_hide(False))
        self.bind("bracketleft", lambda e: self._step(-1))
        self.bind("bracketright", lambda e: self._step(1))
        for k, d in (("plus", 1), ("equal", 1), ("minus", -1), ("KP_Add", 1), ("KP_Subtract", -1)):
            self.bind("<%s>" % k, lambda e, d=d: self._zoom_at(None, d))
        for k, (dx, dy) in (("Left", (-1, 0)), ("Right", (1, 0)), ("Up", (0, -1)), ("Down", (0, 1))):
            self.bind("<%s>" % k, lambda e, dx=dx, dy=dy: self._pan_keys(dx, dy))
        self.protocol("WM_DELETE_WINDOW", self._close)

    # ---------------------------------------------------------------- data
    def _paths(self):
        j = lambda s: os.path.join(self.out_dir, self.stem + s)
        return dict(json=j("_labels.json"), inst=j("_instances.png"), comp=j("_complete.png"), ign=j("_ignore.png"))

    def _read(self):
        self.shapes, self.complete = [], set()
        p = self._paths()["json"]
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8"))
            self.shapes = [dict(s) for s in d.get("shapes", [])]
            self.complete = {tuple(t) for t in d.get("complete_tiles", [])}
            self.lbl_status.configure(text="resumed %d bubbles, %d complete tiles" % (len(self.shapes), len(self.complete)))
        self.history, self.future = [], []
        self.dirty = False

    def _state(self):
        return copy.deepcopy(self.shapes), set(self.complete)

    def _snapshot(self):
        self.history.append(self._state())
        self.future.clear()
        if len(self.history) > 400:
            self.history.pop(0)
        self.dirty = True

    def _undo(self):
        if self.history:
            self.future.append(self._state())
            self.shapes, self.complete = self.history.pop()
            self.dirty = True
            self._recompute(); self._render()

    def _redo(self):
        if self.future:
            self.history.append(self._state())
            self.shapes, self.complete = self.future.pop()
            self.dirty = True
            self._recompute(); self._render()

    def _tile_of(self, x, y):
        i = int(np.searchsorted(self.tx, x, side="right") - 1)
        j = int(np.searchsorted(self.ty, y, side="right") - 1)
        return min(max(i, 0), TILES[0] - 1), min(max(j, 0), TILES[1] - 1)

    def _recompute(self):
        self.own = owner_map(self.shapes, self.H, self.W)
        self.cent = []
        for s in self.shapes:
            self.cent.append(poly_centroid(s["poly"]))
        n_done = sum(1 for (x, y), s in zip(self.cent, self.shapes)
                     if self._tile_of(x, y) in self.complete and not s.get("unsure"))
        self.lbl_count.configure(text="%d / %d bubbles in complete tiles" % (n_done, TARGET))
        self.lbl_detail.configure(text="%d drawn in total, %d unsure; %d of %d tiles complete" % (
            len(self.shapes), sum(1 for s in self.shapes if s.get("unsure")),
            len(self.complete), TILES[0] * TILES[1]))
        self.lbl_mode.configure(text="MODE: %s" % ("draw" if self.mode == "draw" else
                                                     "UNSURE - click a bubble (D to go back)"))

    # ---------------------------------------------------------------- view
    @property
    def z(self):
        return ZOOMS[self.zoom_i]

    def _resized(self, e):
        self.cw, self.ch = e.width, e.height
        self._clamp(); self._render()

    def _clamp(self):
        vw, vh = self.cw / self.z, self.ch / self.z
        self.ox = min(max(self.ox, 0.0), max(self.W - vw, 0.0)) if vw < self.W else (self.W - vw) / 2
        self.oy = min(max(self.oy, 0.0), max(self.H - vh, 0.0)) if vh < self.H else (self.H - vh) / 2

    def _base(self, k):
        key = (k, self.view, self.zoom_i)
        if key not in self._cache:
            im = self.frames.get(k, self.img)
            size = (int(round(self.W * self.z)), int(round(self.H * self.z)))
            if self.view == "pixels":
                out = im.resize(size, Image.NEAREST)
            elif self.view == "enhanced":
                lab = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2LAB)
                lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
                out = Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)).resize(size, Image.LANCZOS)
                out = out.filter(ImageFilter.UnsharpMask(radius=1.5, percent=60, threshold=2))
            else:
                out = im.resize(size, Image.BICUBIC)
            if len(self._cache) > 24:
                self._cache.clear()
            self._cache[key] = out
        return self._cache[key]

    def _scr(self, x, y):
        return (x - self.ox) * self.z, (y - self.oy) * self.z

    def _img_xy(self, e):
        return e.x / self.z + self.ox, e.y / self.z + self.oy

    def _render(self):
        base = self._base(self.frame_off)
        x0, y0 = int(round(self.ox * self.z)), int(round(self.oy * self.z))
        canvas_im = Image.new("RGB", (self.cw, self.ch), (15, 23, 42))
        vis = base.crop((max(x0, 0), max(y0, 0), min(x0 + self.cw, base.size[0]), min(y0 + self.ch, base.size[1])))
        canvas_im.paste(vis, (max(-x0, 0), max(-y0, 0)))
        d = ImageDraw.Draw(canvas_im, "RGBA")
        # tiles
        for i in range(TILES[0]):
            for j in range(TILES[1]):
                a = self._scr(self.tx[i], self.ty[j]); b = self._scr(self.tx[i + 1], self.ty[j + 1])
                if (i, j) in self.complete:
                    d.rectangle([a, b], fill=(34, 197, 94, 28), outline=(34, 197, 94, 230), width=3)
                else:
                    d.rectangle([a, b], outline=(56, 189, 248, 110), width=1)
        if not self.hide:
            w = 2 if self.z >= 2 else 1
            for s in self.shapes:
                col = (250, 204, 21, 255) if s.get("unsure") else (255, 60, 60, 235)
                pts = [self._scr(x, y) for x, y in s["poly"]]
                if len(pts) > 2:
                    d.line(pts + pts[:1], fill=col, width=w)
        if self.stroke is not None and len(self.stroke) > 1:
            d.line([self._scr(x, y) for x, y in self.stroke], fill=(56, 189, 248, 255), width=2)
        self.photo = ImageTk.PhotoImage(canvas_im)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.lbl_view.configure(text="view: %s | zoom %gx | frame %+d%s" % (
            self.view, self.z, self.frame_off, " [flicker]" if self.flicker else ""))

    def _zoom_at(self, e, d):
        i = min(max(self.zoom_i + d, 0), len(ZOOMS) - 1)
        if i == self.zoom_i:
            return
        sx, sy = (e.x, e.y) if e is not None else self.mouse
        ix, iy = sx / self.z + self.ox, sy / self.z + self.oy
        self.zoom_i = i
        self.ox, self.oy = ix - sx / self.z, iy - sy / self.z
        self._clamp(); self._render()

    def _wheel(self, e):
        self._zoom_at(e, 1 if e.delta > 0 else -1)

    def _m_down(self, e):
        self.pan0 = (e.x, e.y, self.ox, self.oy)

    def _m_drag(self, e):
        x, y, ox, oy = self.pan0
        self.ox, self.oy = ox - (e.x - x) / self.z, oy - (e.y - y) / self.z
        self._clamp(); self._render()

    def _pan_keys(self, dx, dy):
        self.ox += dx * 0.25 * self.cw / self.z
        self.oy += dy * 0.25 * self.ch / self.z
        self._clamp(); self._render()

    def _cycle_view(self):
        self.view = VIEWS[(VIEWS.index(self.view) + 1) % len(VIEWS)]
        self._render()

    def _set_hide(self, v):
        if self.hide != v:
            self.hide = v; self._render()

    def _set_mode(self, m):
        self.mode = m
        self._recompute()

    def _step(self, d):
        if self.frame_off + d in self.frames:
            self.frame_off += d; self._render()

    def _toggle_flicker(self):
        self.flicker = not self.flicker
        if self.flicker:
            self._seq = [k for k in (-2, -1, 0, 1, 2, 1, 0, -1) if k in self.frames] or [0]
            self._fi = 0
            self._flick()
        else:
            self.frame_off = 0; self._render()

    def _flick(self):
        if self.flicker:
            self.frame_off = self._seq[self._fi % len(self._seq)]
            self._fi += 1
            self._render()
            self.after(150, self._flick)

    def _back(self):
        if self.flicker or self.frame_off:
            self.flicker, self.frame_off = False, 0

    # ---------------------------------------------------------------- edit
    def _hit(self, e):
        x, y = self._img_xy(e)
        xi, yi = int(x), int(y)
        if 0 <= xi < self.W and 0 <= yi < self.H and self.own[yi, xi]:
            return int(self.own[yi, xi])
        best, bd = 0, 3.0
        for k, s in enumerate(self.shapes, start=1):
            p = np.asarray(s["poly"], float)
            dd = np.hypot(p[:, 0] - x, p[:, 1] - y).min()
            if dd < bd:
                best, bd = k, dd
        return best

    def _l_down(self, e):
        self._back()
        if self.mode == "unsure":
            k = self._hit(e)
            if k:
                self._snapshot()
                self.shapes[k - 1]["unsure"] = not self.shapes[k - 1].get("unsure", False)
                self._recompute()
            self._render()
            return
        self.stroke = [self._img_xy(e)]

    def _l_drag(self, e):
        if self.stroke is not None:
            self.stroke.append(self._img_xy(e))
            self._render()

    def _l_up(self, e):
        if self.stroke is None:
            return
        p, self.stroke = clean_stroke(self.stroke), None
        area = 0.0
        if len(p) >= 4:
            x, y = p[:, 0], p[:, 1]
            area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        if len(p) < 4 or area < 4:
            self.lbl_status.configure(text="too small - outline the whole bubble")
            self._render()
            return
        self._snapshot()
        self.shapes.append(dict(poly=p.round(2).tolist()))
        self.since_save += 1
        self._recompute()
        self.lbl_status.configure(text="bubble %d: d = %.1f px" % (len(self.shapes), 2 * math.sqrt(area / math.pi)))
        if self.since_save >= 10:
            self._save()
        self._render()

    def _r_click(self, e):
        self._back()
        k = self._hit(e)
        if k:
            self._snapshot()
            del self.shapes[k - 1]
            self._recompute()
            self.lbl_status.configure(text="deleted")
        self._render()

    def _toggle_tile(self):
        x, y = self.mouse[0] / self.z + self.ox, self.mouse[1] / self.z + self.oy
        if not (0 <= x < self.W and 0 <= y < self.H):
            return
        t = self._tile_of(x, y)
        self._snapshot()
        self.complete ^= {t}
        self._recompute(); self._render()
        self.lbl_status.configure(text="tile %s %s" % (t, "COMPLETE" if t in self.complete else "not complete"))

    # ---------------------------------------------------------------- save
    def _save(self):
        own = self.own
        inst = np.zeros(own.shape, np.uint16)
        comp = np.zeros(own.shape, np.uint8)
        for (i, j) in self.complete:
            comp[self.ty[j]:self.ty[j + 1], self.tx[i]:self.tx[i + 1]] = 255
        bubbles, shapes = [], []
        nid = 0
        for k, s in enumerate(self.shapes, start=1):
            m = own == k
            if not m.any():
                continue
            nid += 1
            inst[m] = nid
            ys, xs = np.nonzero(m)
            cx, cy = float(xs.mean()) + 0.5, float(ys.mean()) + 0.5
            area = float(m.sum())
            d = 2 * math.sqrt(area / math.pi)
            in_c = self._tile_of(cx, cy) in self.complete
            edge = bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
            bubbles.append(dict(id=nid, shape_type="outline", label="bubble", cx=round(cx, 2), cy=round(cy, 2),
                                r=round(d / 2, 3), d_eq_px=round(d, 2), area_px=area, touches_edge=edge,
                                in_complete_tile=in_c, marked_unsure=bool(s.get("unsure")),
                                unsure=bool(s.get("unsure") or edge), below_floor=False))
            shapes.append(dict(s, id=nid))
        p = self._paths()
        Image.fromarray(inst).save(p["inst"])
        Image.fromarray(comp).save(p["comp"])
        Image.fromarray(np.zeros(own.shape, np.uint8)).save(p["ign"])
        meta = {}
        fj = os.path.join(self.folder, "frame.json")
        if os.path.exists(fj):
            meta = json.load(open(fj, encoding="utf-8"))
        doc = dict(image_file=self.file, regime="FROTH (freehand outlines, complete tiles)",
                   width=self.W, height=self.H, schema_version=5, format="regions",
                   label_mode="frame_tiles", fully_labelled=False, floor_px=0.0,
                   edge_convention="middle of the dark rim", frame=meta,
                   tiles=dict(cols=TILES[0], rows=TILES[1], x=self.tx.tolist(), y=self.ty.tolist()),
                   complete_tiles=sorted([list(t) for t in self.complete]),
                   complete_png=os.path.basename(p["comp"]),
                   instances_png=os.path.basename(p["inst"]), ignore_png=os.path.basename(p["ign"]),
                   count=sum(1 for b in bubbles if b["in_complete_tile"] and not b["unsure"]),
                   bubbles=bubbles, shapes=shapes, ignore_regions=[])
        tmp = p["json"] + ".tmp"
        json.dump(doc, open(tmp, "w", encoding="utf-8"), indent=1)
        os.replace(tmp, p["json"])
        self.dirty = False
        self.since_save = 0
        self.lbl_status.configure(text="saved: %d bubbles (%d in complete tiles)" % (len(bubbles), doc["count"]))

    def _close(self):
        if self.dirty:
            self._save()
        self.destroy()


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "to_label_frame")
    app = App(folder)
    if app.winfo_exists():
        app.mainloop()
