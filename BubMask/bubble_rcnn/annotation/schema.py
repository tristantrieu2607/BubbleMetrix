"""Label schema v2 -- the one place the annotation guide is encoded.

v1 recorded a shape and nothing else, so three rules the guide confirms could not
be written down at all.  Everything below exists because of a specific rule in
``claude/pulp-annotation-guide.md`` (v1.2), and the section is named next to it.

  S3  floor 12 px equivalent diameter (40 um).  Set by the debris -- the largest
      window speck is 10.5 px -- not by preference.  A stamp below the floor is
      refused, not silently kept.
  S3  ``at_limit`` for 12-20 px.  +-17% on diameter is +-59% on volume and D32 is
      volume-weighted, so this band has to stay droppable without relabelling.
  S7  ``touches_edge`` for anything crossing the image rectangle, and
      ``touches_fov`` for anything crossing the circular field boundary.  These are
      different boundaries and only the second is the one the guide names: a 512 px
      crop border is an arbitrary line we drew, while the FOV circle is where the
      probe stops seeing.  Full 1024 px frames have both; a centre crop usually has
      only the first.  Either way the bubble is kept for training -- otherwise the
      model learns the border is empty -- and excluded from the size distribution,
      because a partly visible bubble has no known diameter.
  S4  ``ignore_regions``, written with COCO ``iscrowd = 1``.  An occluded bubble is
      not traced and not left unlabelled -- left unlabelled it teaches "crowded
      means empty water", which is how the original BubMask reached 5% recall.
  S6  ``unsure``.  Forcing a binary on a borderline object puts noise straight into
      the training signal.
  --  ``confirmed_empty``.  An empty crop is a real false-positive test only when a
      person says it is empty.  "Skipped" is not a label, and the two must not be
      the same file on disk.
  S10 ``view``.  Focus is judged on raw pixels; CLAHE makes defocused blobs look
      sharp.  Recording which view a call was made on makes that auditable.

The measurand is the **equivalent-area diameter**, matching what the Fiji ruler
records, so that model and human numbers are the same quantity.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCHEMA_VERSION = 2
FLOOR_PX = 12.0        # guide S3 (pulp)
# Froth has no floor (2026-10-07). The 12 px floor is set by window debris in the
# pulp probe; the froth camera has no such debris, and froth bubbles under 12 px
# (~1.3 mm) are real and carry ~20% of D32. Pulp is unchanged.
FROTH_FLOOR_PX = 0.0


def floor_for_regime(regime: str) -> float:
    return FROTH_FLOOR_PX if "froth" in (regime or "").lower() else FLOOR_PX
AT_LIMIT_PX = 20.0     # guide S3


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def semi_axes(b: Dict[str, Any]) -> Tuple[float, float]:
    """Semi-axes in px.  A circle is the isotropic case."""
    if b.get("shape_type") == "ellipse":
        r2 = float(b.get("r", 0.0)) * 2.0
        return float(b.get("axis_w", r2)) / 2.0, float(b.get("axis_h", r2)) / 2.0
    r = float(b["r"])
    return r, r


def equiv_diameter_px(b: Dict[str, Any]) -> float:
    """Equivalent-area diameter: d_eq = 2*sqrt(area/pi) = 2*sqrt(a*b) for an ellipse."""
    a, c = semi_axes(b)
    return 2.0 * float(np.sqrt(a * c))


def below_floor(b: Dict[str, Any]) -> bool:
    return equiv_diameter_px(b) < FLOOR_PX


def apply_flags(b: Dict[str, Any], width: float, height: float,
                fov: Optional[Tuple[float, float, float]] = None) -> Dict[str, Any]:
    """Set the derived flags.  ``unsure`` is a human call and is never derived.

    ``fov`` is (centre_x, centre_y, radius) in the coordinates of *this image*.
    Pass it for a full frame, where the circular field boundary is visible; a crop
    cut from the middle of a frame usually never reaches it.
    """
    d = equiv_diameter_px(b)
    a, c = semi_axes(b)
    cx, cy = float(b["cx"]), float(b["cy"])
    b["d_eq_px"] = round(d, 2)
    b["at_limit"] = bool(FLOOR_PX <= d < AT_LIMIT_PX)
    b["touches_edge"] = bool(cx - a < 0 or cy - c < 0 or cx + a > width or cy + c > height)
    if fov is not None:
        fx, fy, fr = fov
        b["touches_fov"] = bool(float(np.hypot(cx - fx, cy - fy)) + max(a, c) > fr)
    else:
        b.setdefault("touches_fov", False)
    b.setdefault("unsure", False)
    return b


def fov_for_image(width: float, height: float,
                  centre: Tuple[float, float] = (524.0, 508.0),
                  radius: float = 601.0) -> Optional[Tuple[float, float, float]]:
    """The probe's FOV circle, but only where it is meaningful.

    The circle is measured in full-frame coordinates.  A 512 px crop carries no
    record here of where in the frame it came from, so the circle cannot be placed
    and the flag is left alone rather than guessed.
    """
    if int(width) == 1024 and int(height) == 1024:
        return (centre[0], centre[1], radius)
    return None


def make_bubble(shape_type: str, cx: float, cy: float, *, r: Optional[float] = None,
                axis_w: Optional[float] = None, axis_h: Optional[float] = None,
                angle: float = 0.0, view: str = "raw",
                width: float = 512.0, height: float = 512.0) -> Dict[str, Any]:
    if shape_type == "ellipse":
        aw = float(axis_w or 0.0)
        ah = float(axis_h or 0.0)
        b = {"shape_type": "ellipse", "cx": float(cx), "cy": float(cy),
             "axis_w": aw, "axis_h": ah, "angle": float(angle),
             "r": float(np.sqrt(aw * ah) / 2.0), "label": "bubble", "view": view}
    else:
        b = {"shape_type": "circle", "cx": float(cx), "cy": float(cy),
             "r": float(r or 0.0), "label": "bubble", "view": view}
    return apply_flags(b, width, height, fov_for_image(width, height))


def make_ignore_region(x1: float, y1: float, x2: float, y2: float) -> Dict[str, Any]:
    """A rough box over an occluded bubble or an unreadable patch (guide S4).

    Deliberately rough: the guide asks for a blob, not a trace, because tracing a
    hidden edge means inventing it.  ``iscrowd`` is carried explicitly so the
    COCO semantics survive a round trip through the file.
    """
    return {"type": "rect", "label": "ignore", "iscrowd": 1,
            "x1": float(min(x1, x2)), "y1": float(min(y1, y2)),
            "x2": float(max(x1, x2)), "y2": float(max(y1, y2))}


# --------------------------------------------------------------------------- #
# rasterisation
# --------------------------------------------------------------------------- #
def rasterise_instances(bubbles: List[Dict[str, Any]], h: int, w: int) -> np.ndarray:
    """16-bit instance mask, 0 = background, i = the i-th bubble (1-based)."""
    import cv2
    inst = np.zeros((int(h), int(w)), dtype=np.uint16)
    for i, b in enumerate(bubbles, start=1):
        a, c = semi_axes(b)
        centre = (int(round(float(b["cx"]))), int(round(float(b["cy"]))))
        if b.get("shape_type") == "ellipse":
            cv2.ellipse(inst, centre, (max(1, int(round(a))), max(1, int(round(c)))),
                        float(b.get("angle", 0.0)), 0, 360, int(i), -1)
        else:
            cv2.circle(inst, centre, max(1, int(round(a))), int(i), -1)
    return inst


def rasterise_ignore(regions: List[Dict[str, Any]], h: int, w: int) -> np.ndarray:
    """Binary ignore mask.  Anchors mostly inside it are dropped from the loss."""
    import cv2
    m = np.zeros((int(h), int(w)), dtype=np.uint8)
    for reg in regions or []:
        x1 = int(round(float(reg["x1"]))); y1 = int(round(float(reg["y1"])))
        x2 = int(round(float(reg["x2"]))); y2 = int(round(float(reg["y2"])))
        cv2.rectangle(m, (x1, y1), (x2, y2), 1, -1)
    return m


# --------------------------------------------------------------------------- #
# file format
# --------------------------------------------------------------------------- #
def build_document(*, image_file: str, relative_path: str, regime: str,
                   width: int, height: int, bubbles: List[Dict[str, Any]],
                   ignore_regions: List[Dict[str, Any]],
                   confirmed_empty: bool,
                   floor_px: Optional[float] = None) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "image_file": image_file,
        "relative_path": relative_path,
        "regime": regime,
        "width": int(width),
        "height": int(height),
        "floor_px": FLOOR_PX if floor_px is None else float(floor_px),
        "count": len(bubbles),
        "confirmed_empty": bool(confirmed_empty),
        "bubbles": bubbles,
        "ignore_regions": ignore_regions,
    }


def is_annotated(doc: Dict[str, Any]) -> bool:
    """A crop is done when someone made a call on it -- including "it is empty".

    v1 treated a saved zero-bubble file as unannotated, so ``confirmed_empty`` was
    unreachable and the crop was re-offered for ever.
    """
    return bool(doc.get("bubbles")) or bool(doc.get("confirmed_empty")) \
        or bool(doc.get("ignore_regions"))
