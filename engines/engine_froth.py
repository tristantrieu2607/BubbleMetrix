"""engines.engine_froth
Froth detection engine: Cellpose (zero-shot, default pretrained model).

Validated on the hand-labelled froth crop (Froth/eval, case f299_partial):
26/26 labelled bubbles found, mean mask IoU 0.80, diameter ratio 0.99.
BubSAM3, which this engine replaces, scored 23/26, IoU 0.71.

How a frame is measured
  1. Region: on a full froth frame, the in-focus band of the column is found
     automatically (Froth/roi/crop_measurable.py - the same detector every
     validated run used). A crop or a small image is measured whole.
  2. Scale: the tube is 55 mm wide inside (inner glass surfaces). On a full frame the
     two outer glass walls are found and mm/px = 55 / (their distance in px).
     Where the walls are not in view (crops), the scale of the last full frame
     is used, or 55 mm / 471 px (inner width, mean over 180WW.MP4).
  3. Segmentation: Cellpose runs in its own environment (Froth/.venv) through
     Froth/cellpose/gui_worker.py, started once and kept alive. That isolation
     is deliberate: Cellpose needs opencv-python-headless, the GUI needs the
     windowed OpenCV, and the two cannot share one environment.
  4. Records: one per Cellpose instance, equivalent-area diameter from the pixel
     count (the same measurand as the pulp engine and the labels), the outline
     for drawing, and `touches_image_edge` set when the bubble is cut by the
     measured region's border, so the metrology edge rule drops it.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
FROTH_DIR = os.path.join(PROJECT_ROOT, "Froth")
WORKER = os.path.join(FROTH_DIR, "cellpose", "gui_worker.py")
ROI_CODE_DIR = os.path.join(FROTH_DIR, "roi")
LOG_DIR = os.path.join(FROTH_DIR, "cellpose", "results")

# 55 mm is the INNER width of the tube (confirmed by Tristan, 2026-11-20). The inner glass
# surfaces sit ~15 px inside the outer edges: on 24 frames spread over 180WW.MP4 the
# outer span is 501 px (SD 0.5) and the inner span 471 px (SD 1.7).
TUBE_WIDTH_MM = 55.0
REFERENCE_WALL_SPAN_PX = 502.0          # outer walls on frame_0000 of 180WW.MP4 (wall search guard)
REFERENCE_INNER_SPAN_PX = 471.0         # inner glass surfaces, mean over the clip
DEFAULT_MM_PER_PX = TUBE_WIDTH_MM / REFERENCE_INNER_SPAN_PX   # 0.11677

# Wall search band: rows below the top fitting and above the base (as BubSAM3 used).
WALL_BAND_Y = (330, 1150)


def froth_python() -> str:
    """Froth/.venv interpreter (made by Froth/setup.ps1)."""
    for c in (os.path.join(FROTH_DIR, ".venv", "Scripts", "python.exe"),
              os.path.join(FROTH_DIR, ".venv", "bin", "python")):
        if os.path.exists(c):
            return c
    raise RuntimeError(
        "Cellpose is not installed yet.\n\n"
        "Double-click  Froth\\RUN_CELLPOSE_VIDEO.bat  once (or run Froth\\setup.ps1).\n"
        "It creates Froth\\.venv with Cellpose; after that the Froth mode works.")


# ---------------------------------------------------------------------------
# Region and scale
# ---------------------------------------------------------------------------
def is_full_frame(img: np.ndarray) -> bool:
    """A whole camera frame of the column (portrait, tube walls in view) rather
    than a crop. 180WW.MP4 frames are 720 x 1280."""
    h, w = img.shape[:2]
    return h >= 900 and w >= 500 and h > w


def find_outer_walls(gray: np.ndarray) -> Optional[Tuple[int, int]]:
    """Outer glass walls by column darkness - the same rule as run_bubsam.find_walls,
    which measure_bubbles.py uses for the 55 mm calibration."""
    y0, y1 = min(WALL_BAND_Y[0], gray.shape[0]), min(WALL_BAND_Y[1], gray.shape[0])
    colmean = gray[y0:y1].mean(axis=0)
    mid = len(colmean) // 2
    thresh = colmean.mean() - 0.5 * colmean.std()
    left_c = np.nonzero(colmean[:mid] < thresh)[0]
    right_c = np.nonzero(colmean[mid:] < thresh)[0]
    if not len(left_c) or not len(right_c):
        return None
    left, right = int(left_c[-1]), int(right_c[0] + mid)
    if right - left < 0.5 * len(colmean):          # implausible: not the tube
        return None
    return left, right


def find_inner_walls(gray: np.ndarray, outer: Tuple[int, int],
                     lo: int = 8, hi: int = 36) -> Optional[Tuple[int, int]]:
    """Inner glass surfaces: the darkest column 8-36 px inside each outer wall (the
    glass shows as a bright band with a dark line on its inner face)."""
    y0, y1 = min(WALL_BAND_Y[0], gray.shape[0]), min(WALL_BAND_Y[1], gray.shape[0])
    col = gray[y0:y1].mean(axis=0).astype(np.float32)
    col = cv2.GaussianBlur(col.reshape(1, -1), (5, 1), 0).ravel()
    L, R = outer
    if R - hi < L + hi:
        return None
    li = L + lo + int(np.argmin(col[L + lo:L + hi]))
    ri = R - hi + int(np.argmin(col[R - hi:R - lo]))
    return li, ri


def in_focus_region(img_rgb: np.ndarray) -> Tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the in-focus froth band on a full frame."""
    if ROI_CODE_DIR not in sys.path:
        sys.path.insert(0, ROI_CODE_DIR)
    from crop_measurable import detect_measurable_roi
    (x0, y0, x1, y1), _ = detect_measurable_roi(img_rgb, wall_inset_px=8, focus_thresh_frac=0.58)
    return int(x0), int(y0), int(x1), int(y1)


# ---------------------------------------------------------------------------
# Label map -> bubble records
# ---------------------------------------------------------------------------
def labels_to_bubbles(labels: np.ndarray, offset: Tuple[int, int], frame_idx: int = 0) -> List[Dict[str, Any]]:
    from scipy import ndimage
    ox, oy = offset
    H, W = labels.shape
    out: List[Dict[str, Any]] = []
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        m = (labels[sl] == k).astype(np.uint8)
        area = float(m.sum())
        if area < 1:
            continue
        sy, sx = sl[0].start, sl[1].start
        pad = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        cs, _ = cv2.findContours(pad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)
        carea = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        hull = cv2.contourArea(cv2.convexHull(c))
        pts = (c.reshape(-1, 2) - 1 + np.array([sx + ox, sy + oy])).astype(float)
        ys, xs = np.nonzero(m)
        d = 2.0 * math.sqrt(area / math.pi)
        out.append({
            "id": len(out) + 1,
            "frame_idx": frame_idx,
            "cx": round(float(xs.mean() + sx + ox), 1),
            "cy": round(float(ys.mean() + sy + oy), 1),
            "r": round(d / 2.0, 2),
            "diameter_px": round(d, 2),
            "area_px": area,
            "circularity": round(float(4 * math.pi * carea / peri ** 2) if peri > 0 else 0.0, 3),
            "solidity": round(float(carea / hull) if hull > 0 else 0.0, 3),
            "score": 1.0,
            "touches_image_edge": bool(sl[0].start == 0 or sl[1].start == 0
                                       or sl[0].stop == H or sl[1].stop == W),
            "outline": pts.round(1).tolist(),
        })
    return out


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class FrothDetector:
    """Cellpose behind a persistent worker process. Thread-safe: one frame at a time."""

    def __init__(self, log_cb: Optional[Callable[[str], None]] = None, model_path: Optional[str] = None):
        self.log = log_cb or (lambda s: None)
        self.model_path = model_path or None      # None = Cellpose's default (validated zero-shot)
        self.proc: Optional[subprocess.Popen] = None
        self.info: Dict[str, Any] = {}
        self.lock = threading.Lock()
        self.tmp = tempfile.mkdtemp(prefix="froth_cellpose_")
        self.mm_per_px = DEFAULT_MM_PER_PX
        self.scale_source = "default: 55 mm inner width / 471 px (180WW.MP4)"
        self.last_roi: Optional[Tuple[int, int, int, int]] = None
        self.last_seconds = 0.0

    # -- worker lifecycle ---------------------------------------------------
    def set_model(self, model_path: Optional[str]) -> None:
        """Switch between the default model and a fine-tuned one (restarts the worker)."""
        model_path = model_path or None
        if model_path != self.model_path:
            with self.lock:
                self.close()
                self.model_path = model_path

    def _start(self):
        if self.proc and self.proc.poll() is None:
            return
        py = froth_python()
        os.makedirs(LOG_DIR, exist_ok=True)
        self._log_path = os.path.join(LOG_DIR, "gui_worker.log")
        self._log_fh = open(self._log_path, "a", encoding="utf-8")
        self._log_fh.write("\n==== %s start ====\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
        self._log_fh.flush()
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        self.log("[Froth] Starting Cellpose (first time: loading the model, ~20-60 s)...")
        self.proc = subprocess.Popen(
            [py, "-u", WORKER] + ([self.model_path] if self.model_path else []), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._log_fh, text=True, encoding="utf-8", bufsize=1,
            creationflags=flags, env=env, cwd=os.path.dirname(WORKER))
        msg = self._read_reply(timeout_s=600)
        if not msg.get("ready"):
            raise RuntimeError("Cellpose failed to start: %s\n(details: %s)"
                               % (msg.get("error", "unknown"), self._log_path))
        self.info = msg
        self.log("[Froth] Cellpose %s ready (model %s, %s)" % (
            msg.get("cellpose_version"), msg.get("model"), msg.get("device")))

    def _read_reply(self, timeout_s: float) -> Dict[str, Any]:
        deadline = time.time() + timeout_s
        while True:
            if time.time() > deadline:
                raise RuntimeError("Cellpose did not answer within %d s (log: %s)" % (timeout_s, self._log_path))
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError("Cellpose stopped unexpectedly. See the log:\n%s" % self._log_path)
            if line.startswith("@@ "):
                return json.loads(line[3:])

    def close(self):
        p = self.proc
        self.proc = None
        if p and p.poll() is None:
            try:
                p.stdin.write(json.dumps({"quit": True}) + "\n")
                p.stdin.flush()
                p.wait(timeout=5)
            except Exception:
                p.kill()

    # -- measurement --------------------------------------------------------
    def calibrate(self, img_bgr: np.ndarray) -> float:
        """Update mm/px from the tube's inner width (55 mm) if the walls are in view."""
        if is_full_frame(img_bgr):
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            walls = find_outer_walls(gray)
            if walls:
                span = walls[1] - walls[0]
                inner = find_inner_walls(gray, walls)
                ispan = (inner[1] - inner[0]) if inner else 0
                # Guards: same camera set-up -> outer ~502 px, inner ~471 px.
                if (0.8 * REFERENCE_WALL_SPAN_PX <= span <= 1.2 * REFERENCE_WALL_SPAN_PX
                        and 0.85 * REFERENCE_INNER_SPAN_PX <= ispan <= 1.15 * REFERENCE_INNER_SPAN_PX):
                    self.mm_per_px = TUBE_WIDTH_MM / ispan
                    self.scale_source = "55 mm inner width = %d px (walls detected)" % ispan
                    return self.mm_per_px
                self.log("[Froth] Wall spans %d / %d px are far from the expected ~502 / 471 px; keeping %.4f mm/px"
                         % (span, ispan, self.mm_per_px))
        return self.mm_per_px

    def detect_frame(self, img_bgr: np.ndarray, frame_idx: int = 0) -> List[Dict[str, Any]]:
        with self.lock:
            self._start()
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if is_full_frame(img_bgr):
                x0, y0, x1, y1 = in_focus_region(rgb)
            else:
                x0, y0, x1, y1 = 0, 0, rgb.shape[1], rgb.shape[0]
            self.last_roi = (x0, y0, x1, y1)
            self.calibrate(img_bgr)

            crop = os.path.join(self.tmp, "roi.png")
            out = os.path.join(self.tmp, "labels.npy")
            cv2.imwrite(crop, img_bgr[y0:y1, x0:x1])
            if os.path.exists(out):
                os.remove(out)
            self.proc.stdin.write(json.dumps({"image": crop, "out": out}) + "\n")
            self.proc.stdin.flush()
            msg = self._read_reply(timeout_s=900)
            if not msg.get("ok"):
                raise RuntimeError("Cellpose error: %s" % msg.get("error"))
            self.last_seconds = float(msg.get("seconds", 0))
            labels = np.load(out)
            return labels_to_bubbles(labels, (x0, y0), frame_idx)
