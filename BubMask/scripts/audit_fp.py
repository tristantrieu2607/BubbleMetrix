"""Step through every false positive from an eval run and tag what it really is.

Why: on full frames the labels are not exhaustive, so some "false positives" are real
bubbles nobody stamped. Editing the TEST labels after seeing the model's output would
bias them toward the model. Auditing the false positives instead leaves the labels
untouched and lets precision be reported two ways: against the labels, and after audit.

Run from BubMask\\ after an eval:
    .venv\\Scripts\\python.exe scripts\\audit_fp.py                     # runs/eval_test
    .venv\\Scripts\\python.exe scripts\\audit_fp.py --eval runs/eval_x

Keys:  1 real bubble (in focus, unstamped)   2 defocused / soft
       3 debris or speck                     4 cut by the image edge
       5 duplicate / fragment of a stamped bubble
       6 unsure / other
       7 occluded -- part-hidden bubble the model drew a full circle over
       B back   V raw <-> CLAHE   H hide/show all outlines (see the original)   Q save and quit
Progress is saved after every key (fp_audit.json); re-running resumes.
Cyan = this prediction, green = your stamps, orange = ignore boxes.
"""
import argparse, collections, json, os, sys
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bubble_rcnn.annotation import schema

def tag_key(r):
    """Stable across eval re-runs (fp_id is not): image + rounded box."""
    return f"{r['crop']}|" + ",".join(str(int(round(v))) for v in r["box"])


TAGS = {ord("1"): "real", ord("2"): "defocused", ord("3"): "debris",
        ord("4"): "edge_cut", ord("5"): "duplicate", ord("6"): "unsure",
        ord("7"): "occluded"}
DISPLAY = 640


def crop_view(rec, img, doc, clahe_on, show_overlay=True):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = rec["box"]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(110.0, 1.6 * max(x2 - x1, y2 - y1))
    a, b = int(max(0, cx - half)), int(max(0, cy - half))
    c, d = int(min(w, cx + half)), int(min(h, cy + half))
    base = img
    if clahe_on:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        base = cv2.cvtColor(cv2.createCLAHE(2.0, (8, 8)).apply(g), cv2.COLOR_GRAY2BGR)
    view = base.copy()
    for r in (doc.get("ignore_regions", []) or []) if show_overlay else []:
        if all(k in r for k in ("x1", "y1", "x2", "y2")):
            cv2.rectangle(view, (int(r["x1"]), int(r["y1"])), (int(r["x2"]), int(r["y2"])), (0, 140, 255), 1)
    for bb in doc.get("bubbles", []) if show_overlay else []:
        ra, rc = schema.semi_axes(bb)
        cv2.ellipse(view, (int(round(bb["cx"])), int(round(bb["cy"]))),
                    (max(1, int(round(ra))), max(1, int(round(rc)))),
                    float(bb.get("angle", 0.0)), 0, 360, (0, 255, 0), 1)
    if show_overlay and rec["contour"]:
        cv2.polylines(view, [np.array(rec["contour"], np.int32)], True, (255, 255, 0), 2)
    win = view[b:d, a:c]
    s = DISPLAY / max(win.shape[:2])
    return cv2.resize(win, (int(win.shape[1] * s), int(win.shape[0] * s)), interpolation=cv2.INTER_NEAREST)


def summarise(fps, tags, summary_path):
    cnt = collections.Counter(tags.get(tag_key(r)) for r in fps)
    s = json.load(open(summary_path, encoding="utf-8"))
    tp, kept = s["true_positives"], s["total_pred_bubbles"]
    done = sum(v for k, v in cnt.items() if k)
    print(f"\naudited {done}/{len(fps)} false positives")
    for k in ["real", "occluded", "defocused", "debris", "edge_cut", "duplicate", "unsure"]:
        print(f"  {k:10s} {cnt.get(k, 0)}")
    print(f"precision vs labels      {tp / max(1, kept):.3f}  ({tp}/{kept})")
    if done:
        real = cnt.get("real", 0)
        print(f"precision after audit    {(tp + real) / max(1, kept):.3f}  ({tp}+{real} real)/{kept}"
              + ("" if done == len(fps) else "   [partial]"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="runs/eval_test")
    a = ap.parse_args()
    ev = os.path.join(ROOT, a.eval)
    fps = json.load(open(os.path.join(ev, "false_positives.json"), encoding="utf-8"))
    out = os.path.join(ev, "fp_audit.json")
    tags = json.load(open(out, encoding="utf-8")) if os.path.exists(out) else {}
    if not fps:
        print("no false positives recorded -- re-run eval first"); return

    cache = {}
    i = next((k for k, r in enumerate(fps) if tag_key(r) not in tags), len(fps))
    clahe_on = False
    show_overlay = True
    win = "FP audit"
    while 0 <= i < len(fps):
        r = fps[i]
        if r["image"] not in cache:
            p = r["image"] if os.path.isabs(r["image"]) else os.path.join(ROOT, r["image"])
            cache = {r["image"]: (cv2.imread(p), json.load(open(
                r["labels"] if os.path.isabs(r["labels"]) else os.path.join(ROOT, r["labels"]), encoding="utf-8")))}
        img, doc = cache[r["image"]]
        v = crop_view(r, img, doc, clahe_on, show_overlay)
        head = np.zeros((48, v.shape[1], 3), np.uint8)
        cur = tags.get(tag_key(r), "-")
        cv2.putText(head, f"{i + 1}/{len(fps)}  {r['crop'][:38]}  d={r['d_px']:.0f}px  score={r['score']:.2f}"
                    f"{'  EDGE' if r['touches_image_edge'] else ''}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(head, f"tag: {cur}  {'' if show_overlay else '[OUTLINES HIDDEN] '}1 real 2 defoc 3 debris 4 edge 5 dup 6 unsure 7 occl | B V H Q",
                    (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.imshow(win, np.vstack([head, v]))
        k = cv2.waitKey(0) & 0xFF
        if k in (ord("q"), ord("Q"), 27):
            break
        if k in (ord("v"), ord("V")):
            clahe_on = not clahe_on; continue
        if k in (ord("h"), ord("H")):
            show_overlay = not show_overlay; continue
        if k in (ord("b"), ord("B")):
            i = max(0, i - 1); continue
        if k in TAGS:
            tags[tag_key(r)] = TAGS[k]
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(tags, fh, indent=1)
            i += 1
    cv2.destroyAllWindows()
    summarise(fps, tags, os.path.join(ev, "eval_summary.json"))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
