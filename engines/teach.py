"""engines.teach
"Teach the model": new footage -> hand-checked labels -> fine-tuned model, guarded.

Shared by the GUI (BubMask venv) and by both trainers (pulp: same venv;
froth: Froth/.venv). Imports numpy, cv2 and json only, so it loads in both.

A teaching set is a folder:

  teaching/<regime>/<set name>/
      set.json                      what is in the set (videos, frames, done flags)
      <stem>.png                    the image that was labelled (pulp: whole frame;
                                    froth: a BOX_PX square cut from the in-focus band)
      <stem>_labels.json            the labels, in the pulp label schema (circles), plus
                                    "outline" polygons for froth pre-labels

Labels are pre-filled by the model in use; the user deletes wrong ones, adds missed
ones and ticks "every bubble is marked". Only ticked items are used.

Fine-tuned models live in models/<regime>/<id>/ and models/registry.json says which
one is in use. Nothing under BubMask/runs/ is touched, so the GUI's choice of
the base pulp model does not change behind anyone's back.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import random
import shutil
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
TEACH_ROOT = os.path.join(PROJECT_ROOT, "teaching")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
REGISTRY = os.path.join(MODELS_ROOT, "registry.json")

# ---------------------------------------------------------------------------
# Minimum labels before "Teach the model" unlocks (see build notes 2026-11-20)
#
# pulp : 15 frames and 150 bubbles. Our own runs: 65 labelled bubbles trained the
#        first model to recall 0.52 / D32 ratio 0.80; ~330 per fold gave recall 0.91
#        / D32 0.99 (leave-one-clip-out). Fine-tuning starts from the trained model
#        and replays the old labels, so it needs less than that; 150 is the floor,
#        300 is better. Pulp frames hold ~10 bubbles each (412 over 43 full frames).
# froth: 4 boxes and 300 bubbles. Cellpose 2.0 (Pachitariu & Stringer, Nat Methods
#        2022) fine-tunes with 100-200 corrected objects when the model pre-labels;
#        one box in four is held back for the check, so ~225 train + ~75 check.
# ---------------------------------------------------------------------------
MINIMUMS = {
    "pulp": {"items": 15, "bubbles": 150, "check_items": 3},
    "froth": {"items": 4, "bubbles": 300, "check_items": 1},
}
RECOMMENDED_BUBBLES = {"pulp": 300, "froth": 500}

FRAMES_PER_ADD = {"pulp": 8, "froth": 2}     # added per press of "Add frames from this video"
PULP_FRAME_GAP = 5                           # config.rotor_frame_gap: decorrelated pulp frames
BOX_PX = 192                                 # froth labelling box (~120 bubbles)
PULP_FLOOR_PX = 12.0


# ---------------------------------------------------------------------------
# small file helpers
# ---------------------------------------------------------------------------
def _atomic_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def _slug(s: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_" else "_" for c in s)
    return keep.strip("_")[:60] or "video"


def spread_order(n: int) -> List[int]:
    """Bit-reversed order: every prefix is spread over 0..n-1."""
    if n <= 0:
        return []
    bits = max(1, (n - 1).bit_length())
    out, seen = [], set()
    for i in range(1 << bits):
        r = int(format(i, "0%db" % bits)[::-1], 2)
        if r < n and r not in seen:
            seen.add(r)
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
def make_circle(cx: float, cy: float, r: float, w: int, h: int, source: str = "user",
                outline: Optional[List[List[float]]] = None) -> Dict[str, Any]:
    """A label entry in the pulp schema (schema_version 2). `outline` (froth
    pre-labels) is kept for drawing and for the froth label map."""
    b = {"shape_type": "circle", "cx": round(float(cx), 2), "cy": round(float(cy), 2),
         "r": round(float(r), 2), "label": "bubble", "d_eq_px": round(2.0 * float(r), 2),
         "at_limit": bool(PULP_FLOOR_PX <= 2 * r < 20.0),
         "touches_edge": bool(cx - r < 0 or cy - r < 0 or cx + r > w or cy + r > h),
         "touches_fov": False, "unsure": False, "view": "unknown", "source": source}
    if outline:
        b["outline"] = [[round(float(x), 1), round(float(y), 1)] for x, y in outline]
    return b


def bubbles_from_detections(dets: Sequence[Dict[str, Any]], offset: Tuple[float, float],
                            w: int, h: int) -> List[Dict[str, Any]]:
    """Model detections (GUI bubble records, full-frame coords) -> label entries in
    the item's own coords. Equivalent-area circle, the same measurand as the labels."""
    ox, oy = offset
    out = []
    for d in dets:
        dpx = float(d.get("diameter_px", 2 * float(d.get("r", 0))))
        if dpx <= 0:
            continue
        cx, cy = float(d["cx"]) - ox, float(d["cy"]) - oy
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        ol = d.get("outline")
        if ol:
            ol = [[p[0] - ox, p[1] - oy] for p in ol]
        out.append(make_circle(cx, cy, dpx / 2.0, w, h, source="model", outline=ol))
    return out


def instance_map(bubbles: Sequence[Dict[str, Any]], h: int, w: int) -> np.ndarray:
    """int32 label image, 0 = background. Outline polygons where present, otherwise
    circles. Where two overlap, the later-drawn one wins on the overlap only."""
    lab = np.zeros((h, w), np.int32)
    k = 0
    for b in bubbles:
        if b.get("unsure"):
            continue
        k += 1
        m = np.zeros((h, w), np.uint8)
        ol = b.get("outline")
        if ol and len(ol) >= 3:
            cv2.fillPoly(m, [np.round(np.asarray(ol, np.float64)).astype(np.int32)], 1)
        else:
            cv2.circle(m, (int(round(b["cx"])), int(round(b["cy"]))), max(1, int(round(b["r"]))), 1, -1)
        lab[m > 0] = k
    return lab


def gt_records(bubbles: Sequence[Dict[str, Any]], floor_px: float = 0.0) -> List[Dict[str, Any]]:
    """Labels -> records that engine_metrology's matcher reads (cx, cy, r, diameter_px)."""
    out = []
    for i, b in enumerate(bubbles, 1):
        if b.get("unsure"):
            continue
        d = float(b.get("d_eq_px", 2 * b["r"]))
        if d < floor_px:
            continue
        out.append({"id": i, "cx": float(b["cx"]), "cy": float(b["cy"]), "r": d / 2.0,
                    "diameter_px": d, "touches_image_edge": bool(b.get("touches_edge"))})
    return out


# ---------------------------------------------------------------------------
# the teaching set
# ---------------------------------------------------------------------------
class TeachingSet:
    def __init__(self, folder: str):
        self.folder = folder
        self.path = os.path.join(folder, "set.json")
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as fh:
                self.doc = json.load(fh)
        else:
            raise FileNotFoundError(self.path)

    # -- create / list ---------------------------------------------------
    @staticmethod
    def create(regime: str, name: Optional[str] = None, root: str = TEACH_ROOT) -> "TeachingSet":
        name = name or _dt.datetime.now().strftime("set_%Y-%m-%d_%H%M")
        folder = os.path.join(root, regime, _slug(name))
        os.makedirs(folder, exist_ok=True)
        _atomic_json(os.path.join(folder, "set.json"),
                     {"regime": regime, "name": name, "created": _now(), "items": [], "trainings": []})
        return TeachingSet(folder)

    @staticmethod
    def latest(regime: str, root: str = TEACH_ROOT) -> Optional["TeachingSet"]:
        d = os.path.join(root, regime)
        if not os.path.isdir(d):
            return None
        cands = [os.path.join(d, f) for f in os.listdir(d) if os.path.exists(os.path.join(d, f, "set.json"))]
        if not cands:
            return None
        return TeachingSet(max(cands, key=lambda p: os.path.getmtime(os.path.join(p, "set.json"))))

    @property
    def regime(self) -> str:
        return self.doc["regime"]

    @property
    def items(self) -> List[Dict[str, Any]]:
        return self.doc["items"]

    def save(self) -> None:
        _atomic_json(self.path, self.doc)

    # -- paths -------------------------------------------------------------
    def image_path(self, item: Dict[str, Any]) -> str:
        return os.path.join(self.folder, item["stem"] + ".png")

    def labels_path(self, item: Dict[str, Any]) -> str:
        return os.path.join(self.folder, item["stem"] + "_labels.json")

    # -- adding footage ------------------------------------------------------
    def add_from_video(self, video_path: str, n: Optional[int] = None,
                       roi_fn: Optional[Callable[[np.ndarray], Tuple[int, int, int, int]]] = None,
                       seed: int = 0) -> List[Dict[str, Any]]:
        """Cut n new, spread-out frames (pulp) or boxes (froth) from a video.

        Frames already in the set are skipped, so pressing the button again on the
        same video gives different frames. Pulp frames are at least PULP_FRAME_GAP
        apart (decorrelated); froth frames are spread over the whole clip and each
        gets its box at a different place inside the in-focus band (roi_fn)."""
        regime = self.regime
        n = n or FRAMES_PER_ADD[regime]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video: %s" % video_path)
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vname = os.path.basename(video_path)
        used = {it["frame_idx"] for it in self.items if it.get("video") == vname}

        if regime == "pulp":
            cands = list(range(0, tot, PULP_FRAME_GAP))
        else:
            # froth: one candidate per ~2 s of video, so boxes come from different moments
            cands = sorted(set(int(round(x)) for x in np.linspace(0, max(tot - 1, 0), max(2, min(tot, 200)))))
        order = [cands[i] for i in spread_order(len(cands))]
        rng = random.Random(seed + len(self.items) * 7919)
        added = []
        for f_idx in order:
            if len(added) >= n:
                break
            if f_idx in used:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            stem = "teach_%s_f%06d" % (_slug(os.path.splitext(vname)[0]), f_idx)
            box = None
            img = frame
            if regime == "froth":
                H, W = frame.shape[:2]
                x0, y0, x1, y1 = roi_fn(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if roi_fn else (0, 0, W, H)
                if (x1 - x0) < BOX_PX or (y1 - y0) < BOX_PX:
                    continue
                bx = rng.randint(x0, x1 - BOX_PX)
                by = rng.randint(y0, y1 - BOX_PX)
                box = [bx, by, bx + BOX_PX, by + BOX_PX]
                img = frame[by:by + BOX_PX, bx:bx + BOX_PX]
                stem += "_x%d_y%d" % (bx, by)
            cv2.imwrite(os.path.join(self.folder, stem + ".png"), img)
            item = {"stem": stem, "video": vname, "video_path": video_path, "frame_idx": int(f_idx),
                    "box": box, "done": False, "prelabelled": False, "added": _now()}
            self.items.append(item)
            used.add(f_idx)
            added.append(item)
            self.write_labels(item, [])
        cap.release()
        self.save()
        return added

    # -- labels --------------------------------------------------------------
    def read_labels(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        p = self.labels_path(item)
        if not os.path.exists(p):
            return []
        with open(p, encoding="utf-8") as fh:
            return json.load(fh).get("bubbles", [])

    def write_labels(self, item: Dict[str, Any], bubbles: List[Dict[str, Any]]) -> None:
        img = cv2.imread(self.image_path(item), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2] if img is not None else (0, 0)
        floor = PULP_FLOOR_PX if self.regime == "pulp" else 0.0
        doc = {"schema_version": 2, "image_file": item["stem"] + ".png",
               "relative_path": item["stem"] + ".png",
               "regime": ("PULP: teaching set" if self.regime == "pulp" else "FROTH: teaching set"),
               "width": w, "height": h, "floor_px": floor, "count": len(bubbles),
               # "done with nothing on it" is a real negative; otherwise the trainer skips it
               "confirmed_empty": bool(item.get("done") and not bubbles),
               "fully_labelled": bool(item.get("done")),
               "source_video": item.get("video"), "frame_idx": item.get("frame_idx"),
               "box_in_frame": item.get("box"),
               "bubbles": bubbles, "ignore_regions": []}
        _atomic_json(self.labels_path(item), doc)

    # -- what the model in use predicted (kept to score it before any fine-tuning) --
    def model_path(self, item: Dict[str, Any]) -> str:
        return os.path.join(self.folder, item["stem"] + "_model.json")

    def save_model_predictions(self, item: Dict[str, Any], dets: Sequence[Dict[str, Any]],
                               model_name: str) -> None:
        keep = ("cx", "cy", "r", "diameter_px", "score", "touches_image_edge", "outline")
        _atomic_json(self.model_path(item), {"model": model_name,
                                             "detections": [{k: d[k] for k in keep if k in d} for d in dets]})

    def read_model_predictions(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        p = self.model_path(item)
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)

    def current_model_score(self, model_name: str) -> Optional[Dict[str, Any]]:
        """How good is the model in use on this footage, before any fine-tuning?

        Its own predictions (saved when it pre-labelled each item) against the user's
        corrected labels on every checked item, through the GUI's gates. Checked items
        are complete, so precision counts. Items whose saved predictions came from a
        different model are skipped (`missing` says how many need re-running)."""
        from engine_metrology import summarize_bubbles
        per, missing = [], 0
        for it in self.items:
            if not it.get("done"):
                continue
            pred = self.read_model_predictions(it)
            if not pred or pred.get("model") != model_name:
                missing += 1
                continue
            img = cv2.imread(self.image_path(it), cv2.IMREAD_GRAYSCALE)
            h, w = img.shape[:2]
            if self.regime == "pulp":
                fov = (524.0, 508.0, 601.0) if (w, h) == (1024, 1024) else None
                acc = summarize_bubbles(pred["detections"], um_per_px=None, min_diam_px=PULP_FLOOR_PX,
                                        max_diam_px=250.0, apply_fov_gate=fov is not None,
                                        exclude_edge=True)["accepted_bubbles"]
                per.append(score_image(acc, self.read_labels(it), PULP_FLOOR_PX, w, h, fov))
            else:
                acc = summarize_bubbles(pred["detections"], um_per_px=None, exclude_edge=True)["accepted_bubbles"]
                per.append(score_image(acc, self.read_labels(it), 0.0, w, h))
        if not per:
            return {"items": 0, "missing": missing}
        out = pool_scores(per)
        out["items"] = len(per)
        out["missing"] = missing
        return out

    def set_done(self, item: Dict[str, Any], done: bool) -> None:
        item["done"] = bool(done)
        self.write_labels(item, self.read_labels(item))     # refresh confirmed_empty
        self.save()

    # -- readiness -------------------------------------------------------------
    def counts(self) -> Dict[str, int]:
        done = [it for it in self.items if it.get("done")]
        nb = 0
        for it in done:
            nb += sum(1 for b in self.read_labels(it) if not b.get("unsure"))
        return {"items": len(self.items), "done": len(done), "bubbles": nb}

    def readiness(self) -> Tuple[bool, str]:
        m = MINIMUMS[self.regime]
        c = self.counts()
        unit = "frames" if self.regime == "pulp" else "boxes"
        need_i = max(0, m["items"] - c["done"])
        need_b = max(0, m["bubbles"] - c["bubbles"])
        if need_i or need_b:
            parts = []
            if need_i:
                parts.append("%d more checked %s" % (need_i, unit))
            if need_b:
                parts.append("%d more bubbles" % need_b)
            return False, "Need " + " and ".join(parts) + "."
        rec = RECOMMENDED_BUBBLES[self.regime]
        tail = "" if c["bubbles"] >= rec else " (%d+ bubbles gives a more reliable result)" % rec
        return True, "Ready: %d %s, %d bubbles.%s" % (c["done"], unit, c["bubbles"], tail)

    def split(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """(train, check). Every 4th checked item, spread over the set, is held back
        to compare old and new model on footage neither trained on."""
        done = [it for it in self.items if it.get("done")]
        k = max(MINIMUMS[self.regime]["check_items"], len(done) // 4)
        idx = set(spread_order(len(done))[1::2][:k]) if len(done) > 1 else set()
        if len(idx) < k:
            for i in spread_order(len(done)):
                if len(idx) >= k:
                    break
                idx.add(i)
        check = [done[i] for i in sorted(idx)]
        train = [it for i, it in enumerate(done) if i not in idx]
        return train, check


# ---------------------------------------------------------------------------
# model registry
# ---------------------------------------------------------------------------
def load_registry() -> Dict[str, Any]:
    if os.path.exists(REGISTRY):
        try:
            with open(REGISTRY, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return {"pulp": {"active": None, "history": []}, "froth": {"active": None, "history": []}}


def active_model(regime: str) -> Optional[str]:
    """Path of the fine-tuned model in use, or None for the validated base model."""
    p = load_registry().get(regime, {}).get("active")
    return p if p and os.path.exists(p) else None


def active_label(regime: str) -> str:
    reg = load_registry().get(regime, {})
    p = reg.get("active")
    if not p or not os.path.exists(p):
        return "base model (validated)"
    for h in reg.get("history", []):
        if h.get("path") == p:
            return "fine-tuned %s (%s)" % (h.get("date", ""), h.get("set", ""))
    return "fine-tuned: " + os.path.basename(os.path.dirname(p))


def record_training(regime: str, entry: Dict[str, Any], make_active: bool) -> None:
    reg = load_registry()
    r = reg.setdefault(regime, {"active": None, "history": []})
    r.setdefault("history", []).append(entry)
    if make_active:
        r["active"] = entry["path"]
    _atomic_json(REGISTRY, reg)


def set_active(regime: str, path: Optional[str]) -> None:
    reg = load_registry()
    reg.setdefault(regime, {"active": None, "history": []})["active"] = path
    _atomic_json(REGISTRY, reg)


def new_model_dir(regime: str) -> str:
    d = os.path.join(MODELS_ROOT, regime, _dt.datetime.now().strftime("ft_%Y%m%d_%H%M%S"))
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# scoring and the adopt / keep-old rule
# ---------------------------------------------------------------------------
def pool_scores(per_image: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pool per-image match results (from engine_metrology.evaluate_predictions_vs_
    ground_truth) into one score. D32 is pooled over all bubbles, not averaged."""
    tp = sum(r["tp_count"] for r in per_image)
    npred = sum(r["pred_count"] for r in per_image)
    ngt = sum(r["gt_count"] for r in per_image)
    ious = [m["iou"] for r in per_image for m in r.get("matches", [])]
    pd = [d for r in per_image for d in r.get("_pred_diams", [])]
    gd = [d for r in per_image for d in r.get("_gt_diams", [])]

    def d32(x):
        x = np.asarray(x, float)
        return float((x ** 3).sum() / (x ** 2).sum()) if len(x) and (x ** 2).sum() > 0 else 0.0

    p = tp / npred if npred else 0.0
    rc = tp / ngt if ngt else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc > 0 else 0.0
    ratio = d32(pd) / d32(gd) if d32(gd) > 0 else float("nan")
    return {"images": len(per_image), "gt": ngt, "pred": npred, "tp": tp,
            "precision": round(p, 4), "recall": round(rc, 4), "f1": round(f1, 4),
            "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
            "d32_ratio": round(ratio, 4) if math.isfinite(ratio) else None,
            "d32_error_pct": round(abs(ratio - 1) * 100, 2) if math.isfinite(ratio) else None}


# Tolerances for the adopt rule. A new model must be better on the user's held-back
# footage and no meaningfully worse on the frozen benchmark it never trained on.
GAIN_F1 = 0.02          # counts as better: F1 up by this much ...
GAIN_D32 = 2.0          # ... or D32 error down by this many percentage points
SLACK_F1 = 0.02         # and the other one may not get worse by more than this
SLACK_D32 = 2.0
BENCH_RECALL_DROP = 0.05
BENCH_D32_RISE = 3.0


def decide(old_check: Dict[str, Any], new_check: Dict[str, Any],
           old_bench: Optional[Dict[str, Any]], new_bench: Optional[Dict[str, Any]],
           bench_partial: bool = False) -> Tuple[bool, List[str]]:
    """Return (adopt, reasons). Reasons are plain sentences for the GUI."""
    why: List[str] = []

    def e(s):
        v = s.get("d32_error_pct")
        return float("inf") if v is None else float(v)

    df1 = new_check["f1"] - old_check["f1"]
    dd32 = e(old_check) - e(new_check)        # positive = error went down
    better = df1 >= GAIN_F1 or dd32 >= GAIN_D32
    not_worse = df1 >= -SLACK_F1 and dd32 >= -SLACK_D32
    why.append("On your held-back footage: F1 %.2f -> %.2f, D32 error %s -> %s."
               % (old_check["f1"], new_check["f1"], _pct(old_check), _pct(new_check)))
    ok = better and not_worse
    if not better:
        why.append("Not clearly better on your footage (needs F1 +%.2f or D32 error -%.0f points)."
                   % (GAIN_F1, GAIN_D32))
    elif not not_worse:
        why.append("Better on one measure but worse on the other.")

    if old_bench and new_bench and old_bench.get("gt"):
        drop = old_bench["recall"] - new_bench["recall"]
        rise = (e(new_bench) - e(old_bench)) if not bench_partial else 0.0
        line = "On the frozen benchmark: recall %.2f -> %.2f" % (old_bench["recall"], new_bench["recall"])
        if not bench_partial:
            line += ", D32 error %s -> %s" % (_pct(old_bench), _pct(new_bench))
        why.append(line + ".")
        if drop > BENCH_RECALL_DROP or rise > BENCH_D32_RISE:
            ok = False
            why.append("Got worse on the benchmark it never trained on, so it may have forgotten "
                       "the old footage.")
    else:
        why.append("No frozen benchmark available; judged on your footage only.")
    return ok, why


def _pct(s: Dict[str, Any]) -> str:
    v = s.get("d32_error_pct")
    return "n/a" if v is None else "%.1f%%" % v


def score_image(pred_accepted: Sequence[Dict[str, Any]], label_bubbles: Sequence[Dict[str, Any]],
                floor_px: float, w: int, h: int,
                fov: Optional[Tuple[float, float, float]] = None) -> Dict[str, Any]:
    """Match one image. `pred_accepted` = bubbles the GUI would keep (already through
    summarize_bubbles). The labels get the same gates: floor, image edge, and the
    pulp field of view, so both sides are judged on the same population."""
    from engine_metrology import evaluate_predictions_vs_ground_truth
    gt = []
    for g in gt_records(label_bubbles, floor_px):
        r = g["r"]
        if g["cx"] - r < 0 or g["cy"] - r < 0 or g["cx"] + r > w or g["cy"] + r > h:
            continue
        if fov and math.hypot(g["cx"] - fov[0], g["cy"] - fov[1]) + r > fov[2]:
            continue
        gt.append(g)
    res = evaluate_predictions_vs_ground_truth(list(pred_accepted), gt, iou_threshold=0.5)
    res["_pred_diams"] = [float(p["diameter_px"]) for p in pred_accepted]
    res["_gt_diams"] = [float(g["diameter_px"]) for g in gt]
    for k in ("tagged_predictions", "tagged_ground_truth"):
        res.pop(k, None)
    return res


def current_model_verdict(sc: Optional[Dict[str, Any]], regime: str) -> str:
    """One plain sentence on whether fine-tuning is worth it."""
    if not sc or not sc.get("items"):
        return "Check some frames first; the current model is then scored on them."
    unit = "frames" if regime == "pulp" else "boxes"
    err = sc.get("d32_error_pct")
    line = ("Current model on your %d checked %s: recall %.2f, precision %.2f, D32 %s."
            % (sc["items"], unit, sc["recall"], sc["precision"],
               "n/a" if err is None else "off by %.1f%%" % err))
    if err is not None and err <= 5.0 and sc["f1"] >= 0.8:
        line += " It already works well on this footage; fine-tuning is optional."
    elif err is not None and err > 10.0 or sc["f1"] < 0.6:
        line += " It struggles here; fine-tuning should help."
    else:
        line += " Fine-tuning may help."
    return line
