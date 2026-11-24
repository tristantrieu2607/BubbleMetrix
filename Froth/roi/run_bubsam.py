"""
run_bubsam.py - a BubSAM-INSPIRED bubble segmentation pipeline.

*** THIS IS NOT A REIMPLEMENTATION OF BubSAM. ***

BubSAM (Xu, Feng et al., AIChE J 2024, doi:10.1002/aic.18570) is paywalled and
its method section was not available when this file was written. All that could
be confirmed from the abstract is:
  - it is SAM-based, and claims ~30% better segmentation than vanilla SAM
  - it has a "bubble shape reconstruction" (BSR) step claimed ~30% better than
    ellipse fitting
  - the domain is *bubbly flow* (dispersed bubbles in liquid), NOT packed froth

Every design choice below is therefore either (a) justified by the measured
output of run_sam.py on this exact frame, or (b) marked INFERRED. If you obtain
the PDF, rewrite the marked parts to match the paper and delete this notice.

What this fixes relative to run_sam.py, based on its measured 647-mask output:
  1. ROI crop      - run_sam.py segmented the whole 720x1280 frame including the
                     top tube, base and glass walls. first_exp.py:37 restricts
                     the watershed baseline to Y0,Y1 = 330,1150 inside the walls.
                     Matching that ROI makes the two comparable and cuts the
                     pixel count by more than half.
  2. Seeded prompts- run_sam.py's uniform 64x64 grid is the known failure mode
                     for small objects. Seeding SAM with detected bubble centres
                     puts a prompt on each bubble instead of hoping the grid
                     lands on one. [INFERRED as BubSAM's likely "+30% over SAM";
                     unverified.]
  3. Shape filter  - run_sam.py's D32 came out 219 px because 14 masks >80 px
                     (max 435 px) are column/background, not bubbles.
                     Circularity, solidity and area gates remove them.
  4. Progress      - run_sam.py printed nothing for 23 minutes. This reports per
                     stage so a long run is distinguishable from a hang.
  5. Single encode - SamPredictor encodes the image ONCE. run_sam.py's
                     crop_n_layers=1 forced 5 full 1024x1024 ViT passes.

2026-08-18 addition - `--mode tiled`. The measured defect is neither the prompts
nor the thresholds but the SCALE of the pixels reaching SAM: it resizes to 1024
and decodes 256x256 logits (stride 4), while 47% of hand-counted bubbles are
under 10 px, i.e. under 3.2 decoder cells. Tiled mode upscales each window
before inference. See segment_tiled().

Usage:
  .venv/Scripts/python.exe run_bubsam.py                    # ROI of frame_0000
  .venv/Scripts/python.exe run_bubsam.py --image path.tif --no-roi
  .venv/Scripts/python.exe run_bubsam.py --mode grid        # vanilla-SAM control

  # the scale test, against the only ground truth that exists:
  .venv/Scripts/python.exe run_bubsam.py --mode tiled --no-roi \
      --image "../bubmask-fiji/validation/real_tiff_samples/benchmark_bubbles_800x534_8bit.tif" \
      --points-per-side 32 --out-prefix bubsam_tiled
  .venv/Scripts/python.exe score_handcount.py results/bubsam_tiled_masks.csv --image-size 800x534
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE = os.path.join(HERE, "frames", "frame_0000.png")
DEFAULT_CKPT = os.path.join(HERE, "checkpoints", "sam_vit_b_01ec64.pth")
OUT_DIR = os.path.join(HERE, "results")

# Vertical ROI from first_exp.py:37 - excludes the top tube and the base.
ROI_Y0, ROI_Y1 = 330, 1150
WALL_INSET_PX = 8


def log(msg):
    """Unbuffered stage logging - run_sam.py's silence for 23 min is the bug."""
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# --------------------------------------------------------------------------
# 1. Load + ROI
# --------------------------------------------------------------------------
def find_walls(gray):
    """Locate the inner glass walls by column-mean darkness.

    Ported in spirit from first_exp.py:64. Returns (left, right) inner-edge
    columns. Falls back to the full width if the walls are not obvious.
    """
    colmean = gray.mean(axis=0)
    mid = len(colmean) // 2
    thresh = colmean.mean() - 0.5 * colmean.std()
    left_candidates = np.nonzero(colmean[:mid] < thresh)[0]
    right_candidates = np.nonzero(colmean[mid:] < thresh)[0]
    left = int(left_candidates[-1]) if len(left_candidates) else 0
    right = int(right_candidates[0] + mid) if len(right_candidates) else len(colmean) - 1
    if right - left < 0.3 * len(colmean):        # detection failed, use full width
        return 0, len(colmean) - 1
    return left, right


def load_image(path, use_roi, auto_roi=False, focus_thresh=0.58, manual_y=None):
    im = np.array(Image.open(path).convert("RGB"))
    log("loaded %s %s" % (os.path.basename(path), im.shape))
    if not use_roi:
        return im, (0, 0)
    if auto_roi or manual_y is not None:
        try:
            from crop_measurable import detect_measurable_roi
            (x0, y0, x1, y1), _ = detect_measurable_roi(
                im, wall_inset_px=WALL_INSET_PX,
                focus_thresh_frac=focus_thresh,
                manual_y_range=manual_y
            )
            roi = im[y0:y1, x0:x1]
            pct = 100.0 * roi.shape[0] * roi.shape[1] / (im.shape[0] * im.shape[1])
            log("Measurable In-Focus ROI y[%d:%d] x[%d:%d] -> %s (%.0f%% of frame)"
                % (y0, y1, x0, x1, roi.shape, pct))
            return roi, (x0, y0)
        except Exception as e:
            log("Auto-ROI failed (%s), falling back to static ROI" % e)
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    y0, y1 = min(ROI_Y0, im.shape[0]), min(ROI_Y1, im.shape[0])
    left, right = find_walls(gray[y0:y1])
    x0, x1 = left + WALL_INSET_PX, right - WALL_INSET_PX
    roi = im[y0:y1, x0:x1]
    pct = 100.0 * roi.shape[0] * roi.shape[1] / (im.shape[0] * im.shape[1])
    log("ROI y[%d:%d] x[%d:%d] -> %s (%.0f%% of frame)"
        % (y0, y1, x0, x1, roi.shape, pct))
    return roi, (x0, y0)


def preprocess(rgb, clahe_clip):
    """CLAHE on the L channel. The froth is backlit and low-contrast; SAM was
    trained on natural images. [INFERRED - BubSAM's preprocessing is unknown.]"""
    if clahe_clip <= 0:
        return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=clahe_clip,
                                   tileGridSize=(8, 8)).apply(lab[:, :, 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    log("CLAHE applied (clip=%s)" % clahe_clip)
    return out


# --------------------------------------------------------------------------
# 2. Seed detection -> point prompts
# --------------------------------------------------------------------------
def detect_seeds(rgb, min_radius, max_seeds):
    """One point prompt per candidate bubble centre.

    Distance-transform peaks of the thresholded image - the same seeding idea
    first_exp.py uses for watershed, reused here as SAM prompts. This is the
    part meant to recover the small bubbles a uniform grid misses.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, -4)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dist = cv2.distanceTransform(th, cv2.DIST_L2, 5)

    # A peak is a pixel equal to its local max and far enough from background.
    k = max(3, int(2 * min_radius) | 1)
    peaks = (dist >= cv2.dilate(dist, np.ones((k, k), np.uint8))) & (dist >= min_radius)
    ys, xs = np.nonzero(peaks)
    log("seed detection: %d candidate centres (min_radius=%s)" % (len(ys), min_radius))

    if len(ys) > max_seeds:                      # keep the strongest peaks
        order = np.argsort(dist[ys, xs])[::-1][:max_seeds]
        ys, xs = ys[order], xs[order]
        log("  capped to %d strongest" % max_seeds)
    return np.stack([xs, ys], axis=1).astype(np.float32)


# --------------------------------------------------------------------------
# 3. Shape gates + measurement
# --------------------------------------------------------------------------
def contour_stats(mask):
    """Area, circularity, solidity and fitted ellipse for one binary mask.

    Ellipse fitting is included because BubSAM's abstract benchmarks its BSR
    *against* ellipse fitting - so this is the baseline it claims to beat, not
    BSR itself. True BSR (concave-point splitting of merged bubbles) is NOT
    implemented here; the method is not public to us.
    """
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    if area <= 0 or peri <= 0:
        return None
    hull_area = cv2.contourArea(cv2.convexHull(c))
    stats = {
        "area": float(area),
        "circularity": float(4.0 * np.pi * area / (peri * peri)),
        "solidity": float(area / hull_area) if hull_area > 0 else 0.0,
        "major": float("nan"), "minor": float("nan"),
    }
    if len(c) >= 5:
        (_, _), (ax1, ax2), _ = cv2.fitEllipse(c)
        stats["major"], stats["minor"] = float(max(ax1, ax2)), float(min(ax1, ax2))
    return stats


def passes_gates(s, args):
    """Reject the column/background masks that wrecked run_sam.py's D32."""
    d = 2.0 * np.sqrt(s["area"] / np.pi)
    return (args.min_diam <= d <= args.max_diam
            and s["circularity"] >= args.min_circularity
            and s["solidity"] >= args.min_solidity)


def locate(mask):
    """Bounding box and centroid, computed once so dedupe and save_csv reuse it."""
    ys, xs = np.nonzero(mask)
    return {"bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            "cx": float(xs.mean()), "cy": float(ys.mean())}


def boxes_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def mask_iou(a, b, box_a, box_b):
    """IoU over the union bounding box only.

    Comparing full ROI-sized arrays here is what makes naive NMS quadratic in
    pixels as well as in masks; bubbles are small and local, so the union box is
    a tiny fraction of the ROI.
    """
    x0 = min(box_a[0], box_b[0])
    y0 = min(box_a[1], box_b[1])
    x1 = max(box_a[2], box_b[2]) + 1
    y1 = max(box_a[3], box_b[3]) + 1
    sa, sb = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
    inter = np.logical_and(sa, sb).sum()
    if inter == 0:
        return 0.0
    return inter / float(np.logical_or(sa, sb).sum())


# --------------------------------------------------------------------------
# 4. Segmentation backends
# --------------------------------------------------------------------------
def segment_seeded(sam, rgb, args):
    """SamPredictor with one point prompt per detected centre.

    Encodes the image ONCE (run_sam.py paid for 5 encoder passes), then decodes
    prompts in batches, filtering each batch immediately so full-resolution
    masks never accumulate - run_sam.py peaked at 3 GB doing exactly that.
    """
    seeds = detect_seeds(rgb, args.min_radius, args.max_seeds)
    if len(seeds) == 0:
        log("WARNING: no seeds found; falling back to grid mode")
        return segment_grid(sam, rgb, args)

    predictor = SamPredictor(sam)
    t0 = time.time()
    predictor.set_image(rgb)
    log("image encoded once in %.1fs" % (time.time() - t0))

    kept, B = [], args.batch
    device = predictor.device
    for i in range(0, len(seeds), B):
        chunk = seeds[i:i + B]
        pts = torch.as_tensor(chunk, dtype=torch.float, device=device)[:, None, :]
        lbl = torch.ones((len(chunk), 1), dtype=torch.int, device=device)
        tpts = predictor.transform.apply_coords_torch(pts, rgb.shape[:2])
        with torch.no_grad():
            masks, ious, _ = predictor.predict_torch(
                point_coords=tpts, point_labels=lbl, multimask_output=True)
        # multimask_output gives 3 nested candidates; take the best-scoring one.
        best = ious.argmax(dim=1)
        for j in range(len(chunk)):
            m = masks[j, best[j]].cpu().numpy()
            s = contour_stats(m)
            if s and passes_gates(s, args):
                s["iou"] = float(ious[j, best[j]])
                s.update(locate(m))
                kept.append((m, s))
        log("  decoded %d/%d seeds, %d kept"
            % (min(i + B, len(seeds)), len(seeds), len(kept)))

    return dedupe(kept, args.nms_iou)


def segment_grid(sam, rgb, args):
    """Vanilla SamAutomaticMaskGenerator - the control, to measure any gain."""
    gen = SamAutomaticMaskGenerator(
        sam, points_per_side=args.points_per_side,
        crop_n_layers=0,                     # ROI already crops; no crop layers
        pred_iou_thresh=0.86, stability_score_thresh=0.92,
        min_mask_region_area=10)
    log("grid mode: %d^2 points, this takes a while..." % args.points_per_side)
    out = []
    for m in gen.generate(rgb):
        s = contour_stats(m["segmentation"])
        if s and passes_gates(s, args):
            s["iou"] = float(m["predicted_iou"])
            s.update(locate(m["segmentation"]))
            out.append((m["segmentation"], s))
    return out


def _starts(n, tile, step):
    """Tile origins along one axis, last one clamped flush to the far edge."""
    if n <= tile:
        return [0]
    s = list(range(0, n - tile + 1, step))
    if s[-1] != n - tile:
        s.append(n - tile)
    return s


def segment_tiled(sam, rgb, args):
    """Sliding window, each tile UPSCALED before it reaches SAM.

    This is the scale fix. SAM resizes its input to 1024 on the long side and
    its decoder emits 256x256 logits - an effective output stride of 4. Feeding
    the whole 800x534 benchmark image means a 10 px bubble (the hand-counted
    median) is 3.2 decoder cells across and a 4.5 px one is 1.4. Half the real
    population is thinner than the smallest thing the decoder can represent, so
    no threshold, seed or gate can recover it: the masks are never proposed.

    A 256 px tile upscaled 4x fills the same 1024 input, putting that median
    bubble at 40 px / 10 cells. Same model, same weights, ~15x more resolution
    where it matters.

    Measurement is done in UPSCALED space and divided back down - a 5 px bubble
    is 8 pixels of area in the original and 128 after 4x, so area quantisation
    stops dominating the small end.
    """
    from segment_anything.utils.amg import rle_to_mask

    H, W = rgb.shape[:2]
    t, ov, up = args.tile, args.tile_overlap, args.tile_upscale
    step = max(1, t - ov)
    xs, ys = _starts(W, t, step), _starts(H, t, step)
    tw, th = min(t, W), min(t, H)
    log("tiling %dx%d into %d tiles of %dx%d (overlap %d), upscale %dx -> %dx%d input"
        % (W, H, len(xs) * len(ys), tw, th, ov, up, tw * up, th * up))
    if ov < args.max_diam:
        log("  WARNING: overlap %d < max_diam %.0f - a bubble that big can be cut "
            "by every tile that sees it" % (ov, args.max_diam))

    # Thresholds below SAM's 0.86/0.92 defaults: run_bubsam2.py measured small
    # bubbles scoring lower (mean predicted_iou 0.889 at 5-12 px vs 0.928 at
    # 16-25 px), so the stock cut is size-biased. Shape gates do the rejecting.
    gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.points_per_side,
        crop_n_layers=0,
        pred_iou_thresh=args.tile_iou,
        stability_score_thresh=args.tile_stab,
        min_mask_region_area=int(0.5 * np.pi / 4 * (args.min_diam * up) ** 2),
        output_mode="uncompressed_rle",     # never materialise full-res bools
    )

    kept, n_tile, n_edge = [], 0, 0
    total = len(xs) * len(ys)
    for y0 in ys:
        for x0 in xs:
            n_tile += 1
            if args.max_tiles and n_tile > args.max_tiles:
                log("  stopping at --max-tiles %d" % args.max_tiles)
                return dedupe_boxes(kept, args.nms_iou)
            sub = rgb[y0:y0 + th, x0:x0 + tw]
            big = cv2.resize(sub, (tw * up, th * up), interpolation=cv2.INTER_CUBIC)
            t0 = time.time()
            recs = gen.generate(big)
            before = len(kept)
            for m in recs:
                bx, by, bw, bh = m["bbox"]
                # A bubble clipped by an interior tile edge is measured too small.
                # Drop it here; the overlapping neighbour tile sees it whole.
                if ((bx <= 1 and x0 > 0)
                        or (by <= 1 and y0 > 0)
                        or (bx + bw >= tw * up - 1 and x0 + tw < W)
                        or (by + bh >= th * up - 1 and y0 + th < H)):
                    n_edge += 1
                    continue
                mb = rle_to_mask(m["segmentation"])
                s = contour_stats(mb)
                if s is None:
                    continue
                s["area"] /= float(up * up)          # back to original px
                s["major"] /= up
                s["minor"] /= up
                if not passes_gates(s, args):        # gates are in original px
                    continue
                cnts, _ = cv2.findContours(mb.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    continue
                c = max(cnts, key=cv2.contourArea).astype(np.float64) / up
                c[:, :, 0] += x0
                c[:, :, 1] += y0
                s["contour"] = np.round(c).astype(np.int32)
                s["iou"] = float(m["predicted_iou"])
                s["bbox"] = (x0 + bx / up, y0 + by / up,
                             x0 + (bx + bw) / up, y0 + (by + bh) / up)
                s["cx"] = x0 + (bx + bw / 2.0) / up
                s["cy"] = y0 + (by + bh / 2.0) / up
                kept.append((None, s))
            log("  tile %d/%d at (%d,%d): %d raw -> %d kept, %.0fs"
                % (n_tile, total, x0, y0, len(recs), len(kept) - before,
                   time.time() - t0))

    log("tile-edge rejections: %d" % n_edge)
    return dedupe_boxes(kept, args.nms_iou)


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def dedupe_boxes(kept, nms_iou):
    """NMS on bounding boxes, for tiled mode.

    Overlap bands see the same bubble twice. Pixel-level IoU is unavailable here
    by design - tiled mode never keeps full-resolution masks, only contours - and
    for small near-convex blobs box IoU tracks mask IoU closely enough to dedupe.
    """
    t0 = time.time()
    kept.sort(key=lambda t: t[1]["iou"], reverse=True)
    out = []
    for item in kept:
        bb = item[1]["bbox"]
        if not any(box_iou(bb, o[1]["bbox"]) >= nms_iou for o in out):
            out.append(item)
    log("NMS: %d -> %d unique masks (box iou>%s) in %.1fs"
        % (len(kept), len(out), nms_iou, time.time() - t0))
    return out


def save_contour_overlay(rgb, results, path):
    """Overlay for tiled mode, which carries contours rather than masks."""
    canvas = rgb.copy()
    rng = np.random.default_rng(0)
    for _, s in results:
        cv2.drawContours(canvas, [s["contour"]], -1,
                         tuple(int(v) for v in rng.integers(60, 255, 3)), 1)
    Image.fromarray(canvas).save(path)
    log("overlay saved: %s" % path)


def dedupe(kept, nms_iou):
    """Neighbouring seeds land on the same bubble; keep the best-scoring copy.

    Bounding boxes are screened first so the pixel-level IoU only runs on masks
    that actually overlap. Without that screen this is ~1M full-ROI array
    comparisons for a few thousand seeds, which is slower than the segmentation.
    """
    t0 = time.time()
    kept.sort(key=lambda t: t[1]["iou"], reverse=True)
    out = []
    for m, s in kept:
        bb = s["bbox"]
        if not any(boxes_overlap(bb, o["bbox"]) and mask_iou(m, om, bb, o["bbox"]) >= nms_iou
                   for om, o in out):
            out.append((m, s))
    log("NMS: %d -> %d unique masks (iou>%s) in %.1fs"
        % (len(kept), len(out), nms_iou, time.time() - t0))
    return out


# --------------------------------------------------------------------------
# 5. Output
# --------------------------------------------------------------------------
def save_boundary_overlay(rgb, results, path):
    """Boundaries, not filled interiors.

    run_sam.py:48 promised boundaries but filled whole masks at 45% opacity,
    which on packed froth hides whether bubbles were actually separated - the
    one thing you need to see.
    """
    canvas = rgb.copy()
    rng = np.random.default_rng(0)
    for m, _ in results:
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        colour = tuple(int(v) for v in rng.integers(60, 255, 3))
        cv2.drawContours(canvas, cnts, -1, colour, 1)
    Image.fromarray(canvas).save(path)
    log("overlay saved: %s" % path)


def save_csv(results, offset, path):
    ox, oy = offset
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["area_px2", "equiv_diam_px", "major_px", "minor_px",
                         "circularity", "solidity", "predicted_iou",
                         "centroid_x_full", "centroid_y_full"])
        for _, s in results:
            writer.writerow([round(s["area"], 1),
                             round(2.0 * np.sqrt(s["area"] / np.pi), 2),
                             round(s["major"], 2), round(s["minor"], 2),
                             round(s["circularity"], 4), round(s["solidity"], 4),
                             round(s["iou"], 4),
                             int(s["cx"]) + ox, int(s["cy"]) + oy])
    log("mask table saved: %s" % path)


def report(results):
    if not results:
        log("NO MASKS SURVIVED THE GATES - loosen --min-circularity/--min-solidity")
        return
    d = np.array([2.0 * np.sqrt(s["area"] / np.pi) for _, s in results])
    print()
    print("  masks kept      : %d" % len(d))
    print("  diameter px     : min %.1f  median %.1f  p95 %.1f  max %.1f"
          % (d.min(), np.median(d), np.percentile(d, 95), d.max()))
    # D32 on gated masks only. run_sam.py's ungated D32 was 219 px, meaningless.
    print("  D32             : %.1f px" % ((d ** 3).sum() / (d ** 2).sum()))
    print()
    print("  NOTE: this is a count and a size distribution, NOT a recall figure.")
    print("  HANDOFF.md:37 - decide by recall against a hand-count, not an overlay.")


def main():
    p = argparse.ArgumentParser(description="BubSAM-inspired bubble segmentation")
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--model-type", default="vit_b")
    p.add_argument("--mode", choices=["seeded", "grid", "tiled"], default="seeded",
                   help="seeded = point prompts at detected centres (fast); "
                        "grid = vanilla SamAutomaticMaskGenerator (control); "
                        "tiled = upscaled sliding window (the scale fix)")
    p.add_argument("--no-roi", action="store_true", help="segment the full frame")
    p.add_argument("--auto-roi", action="store_true",
                   help="automatically detect and crop in-focus measurable froth region")
    p.add_argument("--clahe", type=float, default=2.0, help="0 disables")
    p.add_argument("--min-radius", type=float, default=3.0,
                   help="min distance-transform peak, px")
    p.add_argument("--max-seeds", type=int, default=4000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--points-per-side", type=int, default=64,
                   help="grid mode; also tiled (32 over a 4x tile = 8 px spacing "
                        "in original coords, below the 10 px hand-counted median)")
    # Tiled mode. 256 px tile at 4x fills SAM's 1024 input exactly.
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--tile-overlap", type=int, default=64,
                   help="must exceed the largest bubble or big ones get clipped "
                        "in every tile; hand-count max was 57 px")
    p.add_argument("--tile-upscale", type=int, default=4)
    p.add_argument("--tile-iou", type=float, default=0.80,
                   help="pred_iou floor in tiled mode; SAM's 0.86 default is "
                        "size-biased against small bubbles (measured)")
    p.add_argument("--tile-stab", type=float, default=0.85)
    p.add_argument("--max-tiles", type=int, default=0,
                   help="stop after N tiles - smoke test before the full run")
    # Gates. Defaults from the measured run_sam.py distribution: real bubbles sat
    # at 10-40 px, and everything above 80 px was column/background.
    p.add_argument("--min-diam", type=float, default=4.0)
    p.add_argument("--max-diam", type=float, default=80.0)
    p.add_argument("--min-circularity", type=float, default=0.55)
    p.add_argument("--min-solidity", type=float, default=0.85)
    p.add_argument("--nms-iou", type=float, default=0.7)
    p.add_argument("--out-prefix", default="bubsam")
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(args.checkpoint):
        sys.exit("checkpoint not found: %s" % args.checkpoint)

    rgb, offset = load_image(args.image, not args.no_roi, auto_roi=args.auto_roi)
    rgb = preprocess(rgb, args.clahe)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log("device: %s  threads: %d" % (device, torch.get_num_threads()))
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)

    t0 = time.time()
    backend = {"grid": segment_grid, "tiled": segment_tiled}.get(
        args.mode, segment_seeded)
    results = backend(sam, rgb, args)
    log("%s segmentation took %.1fs" % (args.mode, time.time() - t0))

    stem = os.path.join(OUT_DIR, args.out_prefix)
    overlay = (save_contour_overlay if args.mode == "tiled"
               else save_boundary_overlay)
    overlay(rgb, results, stem + "_overlay.png")
    save_csv(results, offset, stem + "_masks.csv")
    report(results)


if __name__ == "__main__":
    main()
