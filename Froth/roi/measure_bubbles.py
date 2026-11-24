"""
measure_bubbles.py - turn a run's mask CSV into physical bubble sizes.

Takes the pixel geometry any of the run_bubsam* scripts writes, applies a
mm/px scale derived from the 55 mm cell, and produces

  <prefix>_bubbles_mm.csv    one row per bubble, in mm
  <prefix>_summary.csv       ONE row for this frame - mean, median, D32, D43,
                             area fraction, number density
  <prefix>_summary.json      the same row, for programmatic use
  <prefix>_distribution.png  size histogram + cumulative volume curve

The summary is deliberately one row carrying `frame` and `time_s` columns, so
the video stage is "for each sampled frame, append one row" and D32(t) is then
just a column of that file. Nothing here has to change to get there - see
measure_sequence() at the bottom, which is that loop, already written.

CALIBRATION
-----------
The cell is 55 mm wide as the WHOLE TUBE - the outer glass faces. The froth
itself occupies a narrower channel inside the glass; on frame_0000 that froth
band is 452 px, which would give 55/452 = 0.12168 mm/px if the 55 mm referred
to it. It does not. The 55 mm is the outer wall-to-wall span of the tube, so
`--wall-mode outer` is the default and gives 55/502 = 0.10956 mm/px.

  --wall-mode outer   glass dark-to-bright transitions, 502 px on frame_0000
                      -> 0.10956 mm/px          <- default, the 55 mm tube
  --wall-mode inner   froth/glass boundary, 452 px
                      -> 0.12168 mm/px          <- 10% larger diameters; for
                                                 reference only

What is still an ESTIMATE is where the outer edges sit, because find_walls()
locates them from the glass dark-to-bright transition rather than from a
ruler. The transition is soft-edged, so treat 502 px as good to a few pixels,
i.e. ~1% on diameter. If you want it exact, measure the cell in the image
against a known target and pass `--mm-per-px`, which overrides detection
entirely. `--calibration-overlay` draws both candidate pairs so the detected
edges can be checked by eye.

Usage:
  .venv/Scripts/python.exe measure_bubbles.py results/bubsam3_masks.csv
  .venv/Scripts/python.exe measure_bubbles.py results/bubsam3_masks.csv \
      --wall-mode outer --calibration-overlay
  .venv/Scripts/python.exe measure_bubbles.py results/bubsam3_masks.csv \
      --mm-per-px 0.1120 --time-s 0 --frame frame_0000
  .venv/Scripts/python.exe measure_bubbles.py results/bubsam3_masks.csv \
      --exclude-source ws          # SAM masks only
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

from run_bubsam import ROI_Y0, ROI_Y1, WALL_INSET_PX, DEFAULT_IMAGE, \
    find_walls, log

CELL_WIDTH_MM = 55.0


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def find_froth_edges(gray_band, outer, smooth=9, frac=0.6):
    """Inner walls by TEXTURE rather than by darkness.

    find_walls() locates the dark-to-bright transitions of the glass, i.e. the
    OUTER pair. The froth is the only strongly textured thing in frame, so it
    can be found as the band where texture energy is high.

    The measure is mean |d(gray)/dy| per column, NOT the column variance. A
    vertical glass edge is constant down a column, so it contributes nothing to
    a vertical derivative - but it dominates the column variance, which is why
    a variance-based version of this returned a froth band 715 px wide, wider
    than the 502 px outer walls containing it.

    The search is bounded by `outer` and takes the longest contiguous run above
    threshold, so the dim far-wall reflection outside the cell cannot extend it.
    Returns (left, right) columns of the froth, or None.
    """
    band = gray_band.astype(np.float32)
    energy = np.abs(cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3)).mean(axis=0)
    if smooth > 1:
        energy = np.convolve(energy, np.ones(smooth) / smooth, mode="same")

    lo, hi = (0, len(energy) - 1) if not outer else outer
    if hi - lo < 10:
        return None
    span = hi - lo
    ref = np.median(energy[lo + int(0.2 * span):hi - int(0.2 * span)])
    if not np.isfinite(ref) or ref <= 0:
        return None

    above = np.zeros(len(energy), bool)
    above[lo:hi + 1] = energy[lo:hi + 1] > frac * ref
    # Longest contiguous run of texture, so an isolated bright fleck outside
    # the froth cannot pull an edge outward.
    best, cur_start, best_len = None, None, 0
    for i, v in enumerate(above):
        if v and cur_start is None:
            cur_start = i
        elif not v and cur_start is not None:
            if i - cur_start > best_len:
                best, best_len = (cur_start, i - 1), i - cur_start
            cur_start = None
    if cur_start is not None and len(above) - cur_start > best_len:
        best = (cur_start, len(above) - 1)
    return best


def calibrate(image_path, cell_mm, mode):
    """Returns (mm_per_px or None, outer_pair, inner_pair)."""
    im = np.array(Image.open(image_path).convert("RGB"))
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    y0, y1 = min(ROI_Y0, im.shape[0]), min(ROI_Y1, im.shape[0])
    band = gray[y0:y1]

    outer = find_walls(band)
    inner = find_froth_edges(band, outer)
    pair = outer if mode == "outer" else inner
    if pair is None or pair[1] <= pair[0]:
        return None, outer, inner
    return cell_mm / float(pair[1] - pair[0]), outer, inner


def calibration_overlay(image_path, outer, inner, path):
    """Draw both candidate wall pairs, so the ambiguity is looked at rather
    than argued about."""
    im = np.array(Image.open(image_path).convert("RGB")).copy()
    y0, y1 = min(ROI_Y0, im.shape[0]), min(ROI_Y1, im.shape[0])
    for pair, col, name, dy in ((outer, (255, 60, 60), "outer", 30),
                                (inner, (60, 160, 255), "inner", 60)):
        if pair is None:
            continue
        for x in pair:
            cv2.line(im, (int(x), y0), (int(x), y1), col, 2)
        cv2.putText(im, "%s %d px -> %.5f mm/px"
                    % (name, pair[1] - pair[0],
                       CELL_WIDTH_MM / (pair[1] - pair[0])),
                    (10, y0 + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    Image.fromarray(im).save(path)
    log("calibration overlay saved: %s" % path)


# --------------------------------------------------------------------------
# Reading a run
# --------------------------------------------------------------------------
def load_run(path, exclude_sources=()):
    """Read any run_bubsam* mask CSV into a list of dicts.

    Only equiv_diam_px is required; area, ellipse axes, circularity, centroid
    and the source tag are used when present. That is how one reader serves
    bubsam1 (has ellipse axes), bubsam2 (has neither axes nor source) and
    bubsam3 (has everything).
    """
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("no rows in %s" % path)
    if "equiv_diam_px" not in rows[0]:
        sys.exit("%s has no equiv_diam_px column; not a run output" % path)

    def num(r, key):
        try:
            return float(r.get(key, ""))
        except (TypeError, ValueError):
            return float("nan")

    xkey = "centroid_x_roi" if "centroid_x_roi" in rows[0] else "centroid_x_full"
    ykey = "centroid_y_roi" if "centroid_y_roi" in rows[0] else "centroid_y_full"

    out, skipped = [], 0
    for i, r in enumerate(rows):
        src = r.get("source", "sam")
        if src in exclude_sources:
            skipped += 1
            continue
        d = num(r, "equiv_diam_px")
        if not np.isfinite(d) or d <= 0:
            skipped += 1
            continue
        a = num(r, "area_px2")
        if not np.isfinite(a) or a <= 0:
            a = np.pi / 4.0 * d * d
        out.append({
            "id": r.get("id", i), "source": src,
            "diam_px": d, "area_px": a,
            "major_px": num(r, "major_px"), "minor_px": num(r, "minor_px"),
            "circ": num(r, "circularity"),
            "x_px": num(r, xkey), "y_px": num(r, ykey),
        })
    if skipped:
        log("skipped %d rows (excluded source, or unusable diameter)" % skipped)
    return out


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def moment_diameter(d, p, q):
    """Generalised mean diameter D[p,q] = (sum d^p / sum d^q)^(1/(p-q)).

    D32 (Sauter) is the volume-to-surface ratio and is THE number for a froth:
    interfacial area per unit gas volume is 6/D32, which is what mass transfer
    scales with. D43 (De Brouckere) is volume-weighted and is what a laser
    diffractometer reports, so it is the one to quote against instrument data.
    """
    d = np.asarray(d, dtype=float)
    num, den = (d ** p).sum(), (d ** q).sum()
    if den <= 0:
        return float("nan")
    return float((num / den) ** (1.0 / (p - q)))


def summarise(bubbles, mm_per_px, roi_area_px, frame, time_s):
    d_mm = np.array([b["diam_px"] for b in bubbles]) * mm_per_px
    a_mm2 = np.array([b["area_px"] for b in bubbles]) * mm_per_px ** 2
    roi_mm2 = roi_area_px * mm_per_px ** 2

    by_src = {}
    for b in bubbles:
        by_src[b["source"]] = by_src.get(b["source"], 0) + 1
    ar = [b["major_px"] / b["minor_px"] for b in bubbles
          if np.isfinite(b["major_px"]) and np.isfinite(b["minor_px"])
          and b["minor_px"] > 0]
    circ = [b["circ"] for b in bubbles if np.isfinite(b["circ"])]

    return {
        "frame": frame,
        "time_s": round(float(time_s), 4),
        "n": len(d_mm),
        "mm_per_px": round(float(mm_per_px), 6),
        "roi_area_mm2": round(float(roi_mm2), 2),
        "d_min_mm": round(float(d_mm.min()), 4),
        "d_mean_mm": round(float(d_mm.mean()), 4),
        "d_median_mm": round(float(np.median(d_mm)), 4),
        "d_std_mm": round(float(d_mm.std(ddof=1)) if len(d_mm) > 1 else 0.0, 4),
        "d10_mm": round(float(np.percentile(d_mm, 10)), 4),
        "d90_mm": round(float(np.percentile(d_mm, 90)), 4),
        "d_max_mm": round(float(d_mm.max()), 4),
        "D32_mm": round(moment_diameter(d_mm, 3, 2), 4),
        "D43_mm": round(moment_diameter(d_mm, 4, 3), 4),
        # Dimensionless width of the distribution, so it survives the
        # calibration ambiguity intact - worth watching over time for that
        # reason alone.
        "span": round(float((np.percentile(d_mm, 90) - np.percentile(d_mm, 10))
                            / np.median(d_mm)), 4),
        # Fraction of the ROI the DETECTED bubbles cover. Detection-limited,
        # NOT gas holdup: everything the detector missed is counted as liquid.
        "detected_area_fraction": round(float(a_mm2.sum() / roi_mm2), 4),
        "number_density_per_mm2": round(float(len(d_mm) / roi_mm2), 4),
        "total_area_mm2": round(float(a_mm2.sum()), 3),
        "aspect_median": round(float(np.median(ar)), 4) if ar else float("nan"),
        "circularity_median": (round(float(np.median(circ)), 4) if circ
                               else float("nan")),
        "by_source": ";".join("%s=%d" % kv for kv in sorted(by_src.items())),
    }


BUBBLE_COLS = ["id", "source", "frame", "time_s", "area_px2", "equiv_diam_px",
               "equiv_diam_mm", "area_mm2", "major_mm", "minor_mm", "aspect",
               "circularity", "x_mm", "y_mm"]


def write_bubbles(bubbles, mm_per_px, frame, time_s, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(BUBBLE_COLS)
        for b in bubbles:
            ar = (b["major_px"] / b["minor_px"]
                  if np.isfinite(b["major_px"]) and np.isfinite(b["minor_px"])
                  and b["minor_px"] > 0 else float("nan"))
            w.writerow([
                b["id"], b["source"], frame, round(float(time_s), 4),
                round(b["area_px"], 1), round(b["diam_px"], 2),
                round(b["diam_px"] * mm_per_px, 4),
                round(b["area_px"] * mm_per_px ** 2, 5),
                round(b["major_px"] * mm_per_px, 4),
                round(b["minor_px"] * mm_per_px, 4),
                round(ar, 3), round(b["circ"], 4),
                round(b["x_px"] * mm_per_px, 3),
                round(b["y_px"] * mm_per_px, 3)])
    log("per-bubble table saved: %s (%d rows)" % (path, len(bubbles)))


def append_summary(summary, path):
    """Append one row per frame.

    The video stage calls this in a loop and the resulting file IS D32(t) - no
    separate time-series format is needed, and no frame has to be re-measured
    to add another one.
    """
    cols = list(summary.keys())
    new = not os.path.exists(path)
    if not new:
        with open(path) as f:
            if next(csv.reader(f), None) != cols:
                log("summary schema changed; rewriting %s" % path)
                new = True
    with open(path, "w" if new else "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(cols)
        w.writerow([summary[c] for c in cols])
    log("summary saved: %s" % path)


def plot_distribution(bubbles, mm_per_px, summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.array([b["diam_px"] for b in bubbles]) * mm_per_px
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.hist(d, bins=40, color="#4477aa", edgecolor="white", linewidth=0.4)
    for v, c, lbl in ((summary["d_median_mm"], "#333333", "median"),
                      (summary["D32_mm"], "#cc3311", "D32")):
        ax1.axvline(v, color=c, ls="--", lw=1.4, label="%s %.3f mm" % (lbl, v))
    ax1.set_xlabel("equivalent diameter (mm)")
    ax1.set_ylabel("count")
    ax1.set_title("n = %d,  %.5f mm/px" % (summary["n"], mm_per_px))
    ax1.legend(fontsize=8)

    # Cumulative VOLUME fraction, not count. A froth's count distribution is
    # dominated by bubbles carrying almost none of the gas; the volume curve is
    # where D32 and D43 actually live, and where a shift over time will show.
    ds = np.sort(d)
    ax2.plot(ds, 100 * np.cumsum(ds ** 3) / (ds ** 3).sum(),
             color="#228833", lw=1.6, label="cumulative volume")
    ax2.plot(ds, 100 * np.arange(1, len(ds) + 1) / len(ds),
             color="#999999", lw=1.2, ls=":", label="cumulative count")
    ax2.axhline(50, color="#dddddd", lw=0.8)
    ax2.set_xlabel("equivalent diameter (mm)")
    ax2.set_ylabel("cumulative %")
    ax2.set_title("volume vs count")
    ax2.legend(fontsize=8, loc="lower right")

    for ax in (ax1, ax2):
        ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    log("distribution plot saved: %s" % path)


def print_report(summary, mode, outer, inner):
    print()
    print("  MEASUREMENT  frame %s   t = %s s"
          % (summary["frame"], summary["time_s"]))
    print("  " + "-" * 60)
    print("  scale               : %.6f mm/px   (%s)"
          % (summary["mm_per_px"], mode))
    if outer and inner and outer[1] > outer[0] and inner[1] > inner[0]:
        print("    outer pair %3d px -> %.5f mm/px"
              % (outer[1] - outer[0], CELL_WIDTH_MM / (outer[1] - outer[0])))
        print("    inner pair %3d px -> %.5f mm/px   (%.0f%% apart)"
              % (inner[1] - inner[0], CELL_WIDTH_MM / (inner[1] - inner[0]),
                 100.0 * abs((outer[1] - outer[0])
                             / float(inner[1] - inner[0]) - 1)))
    print("  bubbles measured    : %d   (%s)" % (summary["n"],
                                                 summary["by_source"]))
    print("  ROI area            : %.1f mm2" % summary["roi_area_mm2"])
    print()
    print("  equivalent diameter (mm)")
    print("    min %.3f   d10 %.3f   median %.3f   d90 %.3f   max %.3f"
          % (summary["d_min_mm"], summary["d10_mm"], summary["d_median_mm"],
             summary["d90_mm"], summary["d_max_mm"]))
    print("    mean %.3f +/- %.3f       span %.2f"
          % (summary["d_mean_mm"], summary["d_std_mm"], summary["span"]))
    print()
    print("  D32 (Sauter)        : %.4f mm     <- the froth number"
          % summary["D32_mm"])
    print("  D43 (De Brouckere)  : %.4f mm" % summary["D43_mm"])
    print("  specific area 6/D32 : %.2f 1/mm" % (6.0 / summary["D32_mm"]))
    print()
    print("  number density      : %.3f bubbles/mm2"
          % summary["number_density_per_mm2"])
    print("  detected area frac  : %.3f   NOT gas holdup - whatever the"
          % summary["detected_area_fraction"])
    print("                        detector missed is counted as liquid")
    print()
    if mode == "outer":
        print("  Scale uses the 55 mm outer tube span. The edge POSITIONS are")
        print("  still detected from the glass transition, not measured;")
        print("  pass --mm-per-px to override. See measure_bubbles.py header.")
    else:
        print("  NOT the 55 mm span: --wall-mode inner uses the froth/glass")
        print("  boundary (452 px), which is a narrower channel inside the")
        print("  55 mm tube. Diameters here will be ~10% larger than the")
        print("  55 mm reference. Re-run with --wall-mode outer (default).")
    print()


# --------------------------------------------------------------------------
# The video stage, already written
# --------------------------------------------------------------------------
def measure_sequence(csv_paths, times, mm_per_px, roi_area_px, out_summary):
    """Measure many frames into one summary file - D32(t) as a column.

    Not wired to a CLI yet on purpose: the frames do not exist yet, and how to
    sample them is an open question. At 30 fps the same bubble appears in
    hundreds of consecutive frames, so a distribution built from every frame is
    meaningless without tracking or sparse sampling (HANDOFF.md; first_exp.py
    used SAMPLE_EVERY_S = 30). Segment the sampled frames with run_bubsam3.py,
    then call this with their CSVs and their timestamps.
    """
    rows = []
    for path, t in zip(csv_paths, times):
        bubbles = load_run(path)
        s = summarise(bubbles, mm_per_px, roi_area_px,
                      os.path.basename(path), t)
        append_summary(s, out_summary)
        rows.append(s)
    return rows


def main():
    p = argparse.ArgumentParser(
        description="Measure bubbles from a run's mask CSV, in mm")
    p.add_argument("csv", help="a *_masks.csv from run_bubsam / 2 / 3")
    p.add_argument("--image", default=DEFAULT_IMAGE,
                   help="the frame the run segmented; used for calibration")
    p.add_argument("--cell-width-mm", type=float, default=CELL_WIDTH_MM)
    p.add_argument("--wall-mode", choices=["outer", "inner"], default="outer",
                   help="which pair of edges is --cell-width-mm apart. "
                        "outer (default) = the 55 mm outer span of the tube "
                        "(glass dark-to-bright transitions, 502 px on frame_0000)")
    p.add_argument("--mm-per-px", type=float, default=None,
                   help="measured scale; overrides wall detection entirely")
    p.add_argument("--roi-size", default=None,
                   help="WxH of the segmented ROI, e.g. 486x820. Default: "
                        "derived from --image as run_bubsam does it")
    p.add_argument("--auto-roi", action="store_true",
                   help="use the in-focus measurable bed ROI for bed area calculations")
    p.add_argument("--focus-thresh", type=float, default=0.58)
    p.add_argument("--manual-y", type=int, nargs=2, default=None)
    p.add_argument("--frame", default=None, help="label for the frame column")
    p.add_argument("--time-s", type=float, default=0.0,
                   help="timestamp of this frame; becomes the x axis once the "
                        "video stage appends one row per frame")
    p.add_argument("--exclude-source", nargs="*", default=[],
                   help="drop rows by source tag, e.g. --exclude-source ws "
                        "to measure SAM masks only")
    p.add_argument("--out-prefix", default=None,
                   help="default: the input CSV path with _masks.csv stripped")
    p.add_argument("--calibration-overlay", action="store_true",
                   help="write an image with both candidate wall pairs drawn")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        sys.exit("not found: %s" % args.csv)

    stem = args.out_prefix or args.csv.replace("_masks.csv", "")
    frame = args.frame or os.path.splitext(os.path.basename(args.image))[0]

    mm_per_px, outer, inner = calibrate(args.image, args.cell_width_mm,
                                        args.wall_mode)
    mode = args.wall_mode
    if args.mm_per_px is not None:
        mm_per_px, mode = args.mm_per_px, "manual"
        log("scale overridden: %.6f mm/px" % mm_per_px)
    elif mm_per_px is None:
        sys.exit("wall detection failed and no --mm-per-px given")

    if args.calibration_overlay:
        calibration_overlay(args.image, outer, inner, stem + "_calibration.png")

    if args.roi_size:
        w, h = (int(v) for v in args.roi_size.lower().split("x"))
    elif args.auto_roi or args.manual_y is not None:
        try:
            from crop_measurable import detect_measurable_roi
            im = np.array(Image.open(args.image).convert("RGB"))
            (x0, y0, x1, y1), _ = detect_measurable_roi(
                im, wall_inset_px=WALL_INSET_PX,
                focus_thresh_frac=args.focus_thresh,
                manual_y_range=args.manual_y
            )
            w, h = (x1 - x0), (y1 - y0)
        except Exception:
            w, h = 486, 820
    else:
        crop_p = stem + "_roi_crop.png"
        if os.path.exists(crop_p):
            im_c = Image.open(crop_p)
            w, h = im_c.size
        else:
            im = np.array(Image.open(args.image).convert("RGB"))
            gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            y0, y1 = min(ROI_Y0, im.shape[0]), min(ROI_Y1, im.shape[0])
            left, right = find_walls(gray[y0:y1])
            w = (right - WALL_INSET_PX) - (left + WALL_INSET_PX)
            h = y1 - y0
    roi_area_px = float(w * h)
    log("ROI taken as %dx%d = %.0f px2" % (w, h, roi_area_px))

    bubbles = load_run(args.csv, set(args.exclude_source))
    if not bubbles:
        sys.exit("no measurable bubbles in %s" % args.csv)

    summary = summarise(bubbles, mm_per_px, roi_area_px, frame, args.time_s)
    write_bubbles(bubbles, mm_per_px, frame, args.time_s,
                  stem + "_bubbles_mm.csv")
    append_summary(summary, stem + "_summary.csv")
    with open(stem + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if not args.no_plot:
        plot_distribution(bubbles, mm_per_px, summary,
                          stem + "_distribution.png")
    print_report(summary, mode, outer, inner)


if __name__ == "__main__":
    main()
