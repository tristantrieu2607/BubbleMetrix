"""
run_cellpose.py - Cellpose on froth, ZERO-SHOT by default.

Cellpose (Stringer et al., 2021; Pachitariu & Stringer, 2022) is a U-Net that
predicts, for every pixel, whether it belongs to an object and a vector
pointing at that object's centre. Pixels whose vectors converge on the same
centre become one object. That is a SEPARATION mechanism, which is what froth
needs: every pixel is bubble, and the question is where one ends.

This script runs the pretrained default model with its default settings and no
training, so the comparison with BubSAM3 is like for like (both zero-shot).
Every setting that is changed from the default is written to <prefix>_run.json.

Outputs mirror roi/run_bubsam3.py, so everything downstream is shared:

  <prefix>_masks.csv       same columns as BubSAM3 (pixels only). Measure it with
                           ..\\roi\\measure_bubbles.py, exactly like BubSAM3
  <prefix>_instances.npz   every outline as a float polygon, same format as
                           BubSAM3's - this is what eval\\froth_eval.py scores
  <prefix>_labels.npy      Cellpose's own label map (0 = background), untouched
  <prefix>_overlay.png     outlines on the unenhanced ROI
  <prefix>_roi_crop.png    the ROI that was segmented
  <prefix>_run.json        Cellpose version, model, settings, runtime

The ROI is taken by the SAME code BubSAM3 uses (roi/run_bubsam.py
load_image), so the two methods always see the same pixels.

Usage (from Froth\\, with Froth\\.venv):
  .venv\\Scripts\\python.exe cellpose\\run_cellpose.py                       # frame_0000, static ROI
  .venv\\Scripts\\python.exe cellpose\\run_cellpose.py --image X.png --no-roi # a crop, as is
  .venv\\Scripts\\python.exe cellpose\\run_cellpose.py --diameter 14          # NOT zero-shot any more
First run downloads the pretrained weights (needs normal internet).
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
FROTH = os.path.dirname(HERE)
ROI_DIR = os.path.join(FROTH, "roi")
DEFAULT_IMAGE = os.path.join(ROI_DIR, "frames", "frame_0000.png")
OUT_DIR = os.path.join(HERE, "results")

# Same column order as run_bubsam3.CSV_COLS, plus two Cellpose-only columns at
# the end. measure_bubbles.py reads by name, so extra columns are harmless.
CSV_COLS = ["id", "source", "area_px2", "equiv_diam_px", "major_px", "minor_px",
            "angle_deg", "perimeter_px", "circularity", "solidity",
            "defect_ratio", "covered_frac", "edge_support", "score",
            "centroid_x_roi", "centroid_y_roi",
            "centroid_x_full", "centroid_y_full",
            "touches_roi_edge", "passes_gates"]


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# --------------------------------------------------------------------------
# Image + ROI: BubSAM3's own loader, so both methods see identical pixels
# --------------------------------------------------------------------------
def load_roi(args):
    if args.no_roi:
        im = np.array(Image.open(args.image).convert("RGB"))
        log("loaded %s %s (no ROI)" % (os.path.basename(args.image), im.shape))
        return im, (0, 0)
    sys.path.insert(0, ROI_DIR)
    try:
        from run_bubsam import load_image          # needs torch + segment-anything
    except ImportError as e:
        sys.exit("ROI selection reuses roi/run_bubsam.py, which could not be "
                 "imported (%s). Use Froth\\.venv, or pass --no-roi." % e)
    return load_image(args.image, True, auto_roi=args.auto_roi,
                      focus_thresh=args.focus_thresh, manual_y=args.manual_y)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_model(args):
    import torch
    import cellpose
    from cellpose import models
    gpu = bool(args.gpu and torch.cuda.is_available())
    kw = dict(gpu=gpu)
    if args.model:
        kw["pretrained_model"] = args.model
    try:                       # cellpose 4: bfloat16 is the default; fp32 is safer on CPU
        model = models.CellposeModel(use_bfloat16=gpu, **kw)
    except TypeError:          # older cellpose has no use_bfloat16
        model = models.CellposeModel(**kw)
    version = getattr(cellpose, "version", getattr(cellpose, "__version__", "unknown"))
    name = str(getattr(model, "pretrained_model", args.model or "default"))
    log("cellpose %s, model %s, device %s" % (version, os.path.basename(name),
                                               "cuda" if gpu else "cpu"))
    return model, str(version), name, gpu


def run_model(model, rgb, args):
    kw = dict(flow_threshold=args.flow_threshold,
              cellprob_threshold=args.cellprob_threshold,
              min_size=args.min_size)
    if args.diameter:
        kw["diameter"] = args.diameter
    out = model.eval(rgb, **kw)
    labels = np.asarray(out[0]).astype(np.int32)
    cellprob = None
    try:                       # flows[2] is the cell-probability logit map
        cp = np.asarray(out[1][2], dtype=np.float32)
        if cp.shape == labels.shape:
            cellprob = 1.0 / (1.0 + np.exp(-cp))
    except Exception:
        pass
    return labels, cellprob


# --------------------------------------------------------------------------
# Label map -> records with BubSAM3's statistics
# --------------------------------------------------------------------------
def instance_records(labels, cellprob):
    """One record per Cellpose instance.

    Area and equivalent diameter come from the PIXEL COUNT - Cellpose's answer
    is a set of pixels, so that is its honest area. Circularity, solidity,
    ellipse and the concavity measure come from the outer contour, the same
    way run_bubsam3.contour_record computes them.
    """
    H, W = labels.shape
    recs = []
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        y0, x0 = sl[0].start, sl[1].start
        m = (labels[sl] == k).astype(np.uint8)
        area = float(m.sum())
        if area < 1:
            continue
        pad = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        cs, _ = cv2.findContours(pad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)
        poly = (c.reshape(-1, 2) - 1 + np.array([x0, y0])).astype(np.float32)
        carea = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        d = 2.0 * np.sqrt(area / np.pi)
        rec = dict(poly=poly, area=area, diam=float(d),
                   perim=float(peri),
                   circ=float(4 * np.pi * carea / peri ** 2) if peri > 0 else 0.0,
                   solidity=float(carea / hull_area) if hull_area > 0 else 0.0,
                   major=float("nan"), minor=float("nan"), angle=float("nan"),
                   defect=0.0, source="cellpose")
        ys, xs = np.nonzero(m)
        rec["cx"], rec["cy"] = float(xs.mean() + x0), float(ys.mean() + y0)
        if len(c) >= 5:
            (_, _), (a1, a2), ang = cv2.fitEllipse(c)
            rec["major"], rec["minor"], rec["angle"] = float(max(a1, a2)), float(min(a1, a2)), float(ang)
        if len(c) >= 4:
            try:
                hull = cv2.convexHull(c, returnPoints=False)
                dfx = cv2.convexityDefects(c, hull)
                if dfx is not None and len(dfx):
                    # OpenCV 4 returns (N, 1, 4), OpenCV 5 returns (N, 4) - reshape,
                    # exactly as run_bubsam3.max_defect_depth does
                    depth = float(np.asarray(dfx).reshape(-1, 4)[:, 3].max()) / 256.0
                    rec["defect"] = depth / max(d / 2.0, 1e-6)
            except cv2.error:
                pass
        rec["score"] = float(cellprob[sl][m > 0].mean()) if cellprob is not None else float("nan")
        rec["touches_edge"] = bool(sl[0].start == 0 or sl[1].start == 0 or
                                   sl[0].stop == H or sl[1].stop == W)
        recs.append(rec)
    return recs


def passes_gates(r, args):
    """BubSAM3's stage-B shape gates, same defaults. Applied only with --gates."""
    aspect = r["major"] / r["minor"] if r["minor"] and r["minor"] > 0 else 1.0
    return (args.min_diam <= r["diam"] <= args.max_diam and
            r["circ"] >= args.min_circularity and
            r["solidity"] >= args.min_solidity and
            aspect <= args.max_aspect)


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------
def save_csv(recs, offset, path):
    import csv
    ox, oy = offset
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        for i, r in enumerate(recs):
            w.writerow([i, r["source"], round(r["area"], 1), round(r["diam"], 2),
                        round(r["major"], 2), round(r["minor"], 2),
                        round(r["angle"], 1), round(r["perim"], 2),
                        round(r["circ"], 4), round(r["solidity"], 4),
                        round(r["defect"], 4), 0.0, "nan", round(r["score"], 4),
                        round(r["cx"], 1), round(r["cy"], 1),
                        round(r["cx"] + ox, 1), round(r["cy"] + oy, 1),
                        int(r["touches_edge"]), int(r["gate"])])
    log("mask table saved: %s (%d rows)" % (path, len(recs)))


def save_instances(recs, offset, shape, path):
    """Same format as run_bubsam3.save_instances - one reader scores both."""
    polys = np.empty(len(recs), dtype=object)
    for i, r in enumerate(recs):
        polys[i] = r["poly"]
    np.savez_compressed(
        path, polys=polys,
        source=np.array([r["source"] for r in recs]),
        diam=np.array([r["diam"] for r in recs], dtype=np.float32),
        offset=np.array(offset, dtype=np.int32),
        shape=np.array(shape[:2], dtype=np.int32),
        method=np.array("cellpose"))
    log("instances saved: %s (%d outlines)" % (path, len(recs)))


def save_overlay(rgb, recs, path):
    o = rgb.copy()
    for r in recs:
        col = (0, 200, 255) if r["gate"] else (255, 60, 60)
        cv2.polylines(o, [np.round(r["poly"]).astype(np.int32)], True, col, 1)
    Image.fromarray(o).save(path)
    log("overlay saved: %s" % path)


def report(recs):
    d = np.array([r["diam"] for r in recs if r["gate"]])
    if not len(d):
        print("\n  no bubbles kept\n"); return
    print("\n  bubbles         : %d" % len(d))
    print("  diameter px     : min %.1f  median %.1f  mean %.1f  p95 %.1f  max %.1f"
          % (d.min(), np.median(d), d.mean(), np.percentile(d, 95), d.max()))
    print("  D32 (Sauter)    : %.1f px" % ((d ** 3).sum() / (d ** 2).sum()))
    print("\n  A count and a size distribution - NOT accuracy. Score it with eval\\froth_eval.py.\n")


# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Cellpose zero-shot froth segmentation")
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--no-roi", action="store_true", help="segment the whole image as given")
    p.add_argument("--auto-roi", action="store_true", help="BubSAM3's in-focus ROI detector")
    p.add_argument("--focus-thresh", type=float, default=0.58)
    p.add_argument("--manual-y", type=int, nargs=2, default=None)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--out-prefix", default="cellpose")
    g = p.add_argument_group("model (defaults = zero-shot)")
    g.add_argument("--model", default=None,
                   help="pretrained model name or path; default = Cellpose's default")
    g.add_argument("--diameter", type=float, default=None,
                   help="expected diameter in px; default None = model decides")
    g.add_argument("--flow-threshold", type=float, default=0.4)
    g.add_argument("--cellprob-threshold", type=float, default=0.0)
    g.add_argument("--min-size", type=int, default=15, help="px area; Cellpose default")
    g.add_argument("--gpu", action="store_true")
    g = p.add_argument_group("optional: BubSAM3's shape gates (off = raw Cellpose)")
    g.add_argument("--gates", action="store_true")
    g.add_argument("--min-diam", type=float, default=4.0)
    g.add_argument("--max-diam", type=float, default=80.0)
    g.add_argument("--min-circularity", type=float, default=0.60)
    g.add_argument("--min-solidity", type=float, default=0.88)
    g.add_argument("--max-aspect", type=float, default=3.0)
    return p


DEFAULTS = {"diameter": None, "flow_threshold": 0.4, "cellprob_threshold": 0.0,
            "min_size": 15, "model": None, "gates": False}


def main(argv=None):
    args = build_parser().parse_args(argv)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, args.out_prefix)

    rgb, offset = load_roi(args)
    model, version, model_name, gpu = build_model(args)
    t0 = time.time()
    labels, cellprob = run_model(model, rgb, args)
    secs = time.time() - t0
    log("Cellpose took %.0f s, %d instances" % (secs, int(labels.max())))
    # save the model's answer first: nothing after this line can lose a finished run
    np.save(stem + "_labels.npy", labels)
    Image.fromarray(rgb).save(stem + "_roi_crop.png")

    recs = instance_records(labels, cellprob)
    n_split = sum(1 for k, sl in enumerate(ndimage.find_objects(labels), 1)
                  if sl is not None and ndimage.label(labels[sl] == k)[1] > 1)
    if n_split:
        log("note: %d instances are in more than one piece; their outline is the "
            "largest piece. eval/froth_eval.py scores <prefix>_labels.npy, which "
            "is exact, so this affects the CSV statistics only" % n_split)
    for r in recs:
        r["gate"] = passes_gates(r, args) if args.gates else True
    kept = [r for r in recs if r["gate"]]
    if args.gates:
        log("shape gates: %d -> %d" % (len(recs), len(kept)))

    save_overlay(rgb, recs, stem + "_overlay.png")
    save_csv(kept, offset, stem + "_masks.csv")
    save_instances(kept, offset, rgb.shape, stem + "_instances.npz")

    changed = {k: getattr(args, k) for k, v in DEFAULTS.items() if getattr(args, k) != v}
    model_changed = {k: v for k, v in changed.items() if k != "gates"}
    run = dict(method="cellpose", cellpose_version=version, model=model_name,
               zero_shot=not model_changed,          # model + settings untouched
               gates_applied=bool(args.gates),       # post-filter only
               changed_from_default=changed,
               image=os.path.abspath(args.image), roi_offset=list(offset),
               roi_shape=list(rgb.shape[:2]), device="cuda" if gpu else "cpu",
               seconds=round(secs, 1), n_instances=len(recs), n_kept=len(kept),
               date=time.strftime("%Y-%m-%d %H:%M"))
    json.dump(run, open(stem + "_run.json", "w"), indent=2)
    log("run record: %s" % (stem + "_run.json"))
    report(recs)


if __name__ == "__main__":
    main()
