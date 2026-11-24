"""bubble_ui.engine_metrology
Core metrology and validation engine for Bubble Size Measurement.
Compliant with ASTM E112 and ISO 13322-1 standards.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2


def compute_sauter_d32(diameters: List[float]) -> float:
    """Volume-weighted Sauter Mean Diameter:
    D32 = sum(d_i^3) / sum(d_i^2)
    """
    if not diameters:
        return 0.0
    d = np.asarray(diameters, dtype=np.float64)
    d2 = np.sum(d ** 2)
    if d2 <= 1e-9:
        return 0.0
    return float(np.sum(d ** 3) / d2)


def compute_debrouckere_d43(diameters: List[float]) -> float:
    """De Brouckere Mean Diameter:
    D43 = sum(d_i^4) / sum(d_i^3)
    """
    if not diameters:
        return 0.0
    d = np.asarray(diameters, dtype=np.float64)
    d3 = np.sum(d ** 3)
    if d3 <= 1e-9:
        return 0.0
    return float(np.sum(d ** 4) / d3)


def compute_distribution_histogram(diameters: List[float], n_bins: int = 20) -> Dict[str, Any]:
    """Generates histogram bins and cumulative volume distribution (Q3)."""
    if not diameters:
        return {
            "bins": [],
            "counts": [],
            "densities": [],
            "cumulative_pct": [],
            "d10": 0.0,
            "d50": 0.0,
            "d90": 0.0,
            "span": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    d = np.sort(np.asarray(diameters, dtype=np.float64))
    d_min = float(np.min(d))
    d_max = float(np.max(d))

    if d_min == d_max:
        bins = [round(d_min * 0.9, 1), round(d_max * 1.1, 1)]
        counts = [len(d)]
        cumulative = [100.0]
    else:
        counts, bin_edges = np.histogram(d, bins=n_bins)
        bins = [round(float(b), 2) for b in bin_edges[:-1]]
        counts = [int(c) for c in counts]
        
        # Q3 = cumulative VOLUME distribution (each bubble weighted by d^3), which is
        # what the GUI labels it and what D32 is built on. (It was a count fraction, Q0.)
        vol = np.histogram(d, bins=bin_edges, weights=d ** 3)[0]
        cum_v = np.cumsum(vol) / max(float(vol.sum()), 1e-12) * 100.0
        cumulative = [round(float(c), 1) for c in cum_v]

    d10 = float(np.percentile(d, 10))
    d50 = float(np.percentile(d, 50))
    d90 = float(np.percentile(d, 90))
    span = float((d90 - d10) / d50) if d50 > 1e-9 else 0.0

    return {
        "bins": bins,
        "counts": counts,
        "cumulative_pct": cumulative,
        "d10": round(d10, 2),
        "d50": round(d50, 2),
        "d90": round(d90, 2),
        "span": round(span, 3),
        "min": round(d_min, 2),
        "max": round(d_max, 2),
    }


def summarize_bubbles(
    bubbles: List[Dict[str, Any]],
    um_per_px: Optional[float] = None,
    min_diam_px: float = 0.0,
    max_diam_px: float = 1e6,
    apply_fov_gate: bool = False,
    fov_center_xy: Tuple[float, float] = (524.0, 508.0),   # measured probe FOV
    fov_radius_px: float = 601.0,                          # not min(w,h)*0.44
    exclude_edge: bool = True,   # drop bubbles cut by the image border -- their size is truncated
    unit_name: str = "um",       # unit that `um_per_px` is expressed in ("um" pulp, "mm" froth)
    calibration_note: Optional[str] = None,

) -> Dict[str, Any]:
    """Applies quality gates, computes metrology metrics, and prepares distribution tables.

    `um_per_px` is the physical size of one pixel in `unit_name` units. The name is kept
    for backward compatibility; froth passes mm per px with unit_name="mm".
    """
    calibrated = um_per_px is not None and um_per_px > 0
    scale = float(um_per_px) if calibrated else 1.0

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    diams_px: List[float] = []
    diams_val: List[float] = []

    for b in bubbles:
        if "diameter_px" in b:
            d_px = float(b["diameter_px"])
        elif "d_eq_px" in b:
            d_px = float(b["d_eq_px"])
        elif "r" in b:
            d_px = float(b["r"] * 2.0)
        elif "area" in b and b["area"] > 0:
            d_px = float(2.0 * math.sqrt(b["area"] / math.pi))
        else:
            d_px = 0.0

        if d_px <= 0:
            continue

        cx = float(b.get("cx", b.get("center_x_px", b.get("center_xy", [0, 0])[0])))
        cy = float(b.get("cy", b.get("center_y_px", b.get("center_xy", [0, 0])[1])))

        # Gate 1: Size bounds
        if d_px < min_diam_px or d_px > max_diam_px:
            b_copy = dict(b)
            b_copy["rejection_reason"] = f"size_out_of_bounds ({d_px:.1f}px)"
            rejected.append(b_copy)
            continue

        # Gate 2: FOV clearance
        if apply_fov_gate:
            dist = math.hypot(cx - fov_center_xy[0], cy - fov_center_xy[1])
            if (dist + (d_px / 2.0)) > fov_radius_px:
                b_copy = dict(b)
                b_copy["rejection_reason"] = "touches_fov_boundary"
                rejected.append(b_copy)
                continue

        # Gate 3: edge rule -- a bubble cut by the image border has a truncated size
        if exclude_edge and b.get("touches_image_edge"):
            b_copy = dict(b)
            b_copy["rejection_reason"] = "touches_image_edge"
            rejected.append(b_copy)
            continue

        d_unit = d_px * scale if calibrated else d_px

        b_acc = dict(b)
        b_acc["cx"] = round(cx, 1)
        b_acc["cy"] = round(cy, 1)
        b_acc["diameter_px"] = round(d_px, 2)
        b_acc["diameter_val"] = round(d_unit, 2)
        b_acc["unit"] = unit_name if calibrated else "px"

        accepted.append(b_acc)
        diams_px.append(d_px)
        diams_val.append(d_unit)

    d32 = compute_sauter_d32(diams_val)
    d43 = compute_debrouckere_d43(diams_val)
    d32_px = compute_sauter_d32(diams_px)
    d43_px = compute_debrouckere_d43(diams_px)
    mean_d = float(np.mean(diams_val)) if diams_val else 0.0
    median_d = float(np.median(diams_val)) if diams_val else 0.0
    std_d = float(np.std(diams_val)) if diams_val else 0.0

    n_edge = sum(1 for b in rejected if b.get("rejection_reason") == "touches_image_edge")
    psd = compute_distribution_histogram(diams_val)

    return {
        "n_total": len(bubbles),
        "n_accepted": len(accepted),
        # D32 = sum(d^3)/sum(d^2) is volume-weighted, so a handful of large masks set
        # it.  On a single frame the count is routinely under 20, and a D32 quoted
        # from that is a number about two bubbles, not a size distribution.  The
        # caller gets told rather than left to notice.
        "sample_size_warning": (
            None if len(accepted) >= 30 else
            f"D32 computed from {len(accepted)} bubble(s). D32 is volume-weighted, so "
            f"a few large masks set it -- this is not a size distribution. Pool "
            f"decorrelated frames before quoting it."
        ),
        # 3.291 um/px comes from a field size quoted by the video provider, not from a
        # target imaged through this probe.  Every micron figure below inherits that.
        "calibration_status": (calibration_note if calibration_note else
                               ("provisional -- 3.291 um/px is derived, not measured; "
                                "pixels are the honest unit") if calibrated else "none -- pixels"),
        "n_rejected": len(rejected),
        "unit": unit_name if calibrated else "px",
        "um_per_px": um_per_px if calibrated else None,
        "scale_per_px": um_per_px if calibrated else None,
        "d32": round(d32, 2),
        "n_edge_excluded": n_edge,
        "edge_rule": exclude_edge,
        "d43": round(d43, 2),
        "d32_px": round(d32_px, 2),
        "d43_px": round(d43_px, 2),
        "mean_d": round(mean_d, 2),
        "median_d": round(median_d, 2),
        "std_d": round(std_d, 2),
        "distribution": psd,
        "accepted_bubbles": accepted,
        "rejected_bubbles": rejected,
    }


def circle_iou(c1: Tuple[float, float, float], c2: Tuple[float, float, float]) -> float:
    """Exact sub-pixel Intersection-over-Union between two circles (cx, cy, r)."""
    x1, y1, r1 = c1
    x2, y2, r2 = c2
    d = math.hypot(x1 - x2, y1 - y2)

    if d >= r1 + r2:
        return 0.0

    if d <= abs(r1 - r2):
        r_inner = min(r1, r2)
        r_outer = max(r1, r2)
        inter_area = math.pi * r_inner * r_inner
        union_area = math.pi * r_outer * r_outer
        return inter_area / union_area if union_area > 0 else 0.0

    r1_sq = r1 * r1
    r2_sq = r2 * r2
    alpha = math.acos(max(-1.0, min(1.0, (d * d + r1_sq - r2_sq) / (2.0 * d * r1))))
    beta = math.acos(max(-1.0, min(1.0, (d * d + r2_sq - r1_sq) / (2.0 * d * r2))))

    inter_area = (
        r1_sq * alpha
        + r2_sq * beta
        - 0.5 * math.sqrt(max(0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)))
    )
    union_area = math.pi * r1_sq + math.pi * r2_sq - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def evaluate_predictions_vs_ground_truth(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    iou_threshold: float = 0.50,
    um_per_px: Optional[float] = None,
    unit_name: str = "um",
) -> Dict[str, Any]:
    """Greedy bipartite matching by IoU to validate model detections against ground truth.
    Categorizes into True Positives (TP), False Positives (FP), and False Negatives (FN).
    Computes Precision, Recall, F1, Mean IoU, Model D32 vs GT D32, and D32 Error %.
    """
    preds = []
    for i, p in enumerate(predictions):
        cx = float(p.get("cx", p.get("center_x_px", 0)))
        cy = float(p.get("cy", p.get("center_y_px", 0)))
        d = float(p.get("diameter_px", p.get("d_eq_px", p.get("r", 0) * 2.0)))
        r = d / 2.0
        preds.append({"id": p.get("id", i + 1), "cx": cx, "cy": cy, "r": r, "diam": d, "orig": p})

    gts = []
    for j, g in enumerate(ground_truth):
        cx = float(g.get("cx", g.get("center_x_px", 0)))
        cy = float(g.get("cy", g.get("center_y_px", 0)))
        d = float(g.get("diameter_px", g.get("d_eq_px", g.get("r", 0) * 2.0)))
        r = d / 2.0
        gts.append({"id": g.get("id", j + 1), "cx": cx, "cy": cy, "r": r, "diam": d, "orig": g})

    iou_pairs = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            iou = circle_iou((p["cx"], p["cy"], p["r"]), (g["cx"], g["cy"], g["r"]))
            if iou >= iou_threshold:
                iou_pairs.append((iou, pi, gi))

    iou_pairs.sort(key=lambda x: x[0], reverse=True)
    matched_p = set()
    matched_g = set()
    matches = []

    for iou, pi, gi in iou_pairs:
        if pi not in matched_p and gi not in matched_g:
            matched_p.add(pi)
            matched_g.add(gi)
            matches.append({
                "iou": round(iou, 4),
                "pred_id": preds[pi]["id"],
                "gt_id": gts[gi]["id"],
                "pred_diam": preds[pi]["diam"],
                "gt_diam": gts[gi]["diam"],
                "cx": preds[pi]["cx"],
                "cy": preds[pi]["cy"],
                "r": preds[pi]["r"],
            })

    tp_count = len(matches)
    fp_count = len(preds) - tp_count
    fn_count = len(gts) - tp_count

    precision = float(tp_count / len(preds)) if preds else (1.0 if not gts else 0.0)
    recall = float(tp_count / len(gts)) if gts else 1.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean([m["iou"] for m in matches])) if matches else 0.0

    scale = um_per_px if um_per_px and um_per_px > 0 else 1.0
    unit = unit_name if (um_per_px and um_per_px > 0) else "px"

    pred_diams = [p["diam"] * scale for p in preds]
    gt_diams = [g["diam"] * scale for g in gts]

    pred_d32 = compute_sauter_d32(pred_diams)
    gt_d32 = compute_sauter_d32(gt_diams)
    d32_ratio = float(pred_d32 / gt_d32) if gt_d32 > 1e-9 else 1.0
    d32_error_pct = abs(d32_ratio - 1.0) * 100.0

    tagged_preds = []
    for pi, p in enumerate(preds):
        p_dict = dict(p["orig"])
        if pi in matched_p:
            p_dict["validation_status"] = "TP"
            m = next(m for m in matches if m["pred_id"] == p["id"])
            p_dict["matched_iou"] = m["iou"]
            p_dict["matched_gt_id"] = m["gt_id"]
        else:
            p_dict["validation_status"] = "FP"
            p_dict["matched_iou"] = 0.0
        tagged_preds.append(p_dict)

    tagged_gts = []
    for gi, g in enumerate(gts):
        g_dict = dict(g["orig"])
        if gi in matched_g:
            g_dict["validation_status"] = "TP_GT"
            m = next(m for m in matches if m["gt_id"] == g["id"])
            g_dict["matched_iou"] = m["iou"]
        else:
            g_dict["validation_status"] = "FN"
            g_dict["matched_iou"] = 0.0
        tagged_gts.append(g_dict)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou": round(mean_iou, 4),
        "tp_count": tp_count,
        "fp_count": fp_count,
        "fn_count": fn_count,
        "pred_count": len(preds),
        "gt_count": len(gts),
        "pred_d32": round(pred_d32, 2),
        "gt_d32": round(gt_d32, 2),
        "d32_ratio": round(d32_ratio, 4),
        "d32_error_pct": round(d32_error_pct, 2),
        "unit": unit,
        "matches": matches,
        "tagged_predictions": tagged_preds,
        "tagged_ground_truth": tagged_gts,
    }
