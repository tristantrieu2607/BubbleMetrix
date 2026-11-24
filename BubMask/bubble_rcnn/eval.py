"""Evaluation against hand labels.

Rewritten because v1 reported a number that was not the number it was compared to.

* **Mask IoU, not box IoU.**  v1 compared axis-aligned boxes around circles.  For
  two near-concentric circles that is systematically higher than mask IoU and is
  blind to shape, so "93% IoU" was not the same quantity as the 0.875 mask-IoU
  human ceiling it appeared to beat.
* **Equivalent-area diameter on both sides.**  v1 took the ground truth as the
  stamped radius and the prediction as ``minEnclosingCircle`` of the convex hull.
  Those are different measurands, and the second is the worst available estimator
  for a mask that has swallowed a cluster.  Both sides now use d_eq = 2*sqrt(A/pi),
  which is what the Fiji ruler records.
* **Match before filtering.**  A prediction that matches a hand-labelled bubble is
  never discarded for overlapping an ignore region -- a person traced that bubble.
  Only *unmatched* predictions face the ignore filter.  This is COCO's crowd rule.
* **Missed entirely vs touched but unmatched.**  "Never proposed" is an
  architecture problem; "found but fragmented" is a post-processing one, and one
  recall number hides which you have.
* **A circularity gate on the measurement path only.**  D32 is volume-weighted, so
  a handful of cluster masks set it.  Recall and precision stay ungated so they
  describe the raw detector.

Nothing here applies the 12 px floor to recall: the floor is a labelling rule, and
the labels already obey it.
"""

import os
import json
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from bubble_rcnn.annotation import schema
from bubble_rcnn.config import BubbleConfig
from bubble_rcnn.models.mask_rcnn_v2 import load_bubble_maskrcnn
from bubble_rcnn.sizing.metrology import BubbleMetrology

CIRCULARITY_GATE = 0.70          # measurement path only
SIZE_BANDS_PX = [(12, 40), (40, 70), (70, 100), (100, 140), (140, 1e9)]


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.count_nonzero(a & b))
    if inter == 0.0:
        return 0.0
    union = float(np.count_nonzero(a | b))
    return inter / union if union > 0 else 0.0


def mask_d_eq(m: np.ndarray) -> float:
    area = float(np.count_nonzero(m))
    return 2.0 * float(np.sqrt(area / np.pi)) if area > 0 else 0.0


def mask_circularity(m: np.ndarray) -> float:
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    per = cv2.arcLength(c, True)
    if per <= 0:
        return 0.0
    return float(4.0 * np.pi * cv2.contourArea(c) / (per * per))


def _band(d: float) -> str:
    for lo, hi in SIZE_BANDS_PX:
        if lo <= d < hi:
            return f"{lo:.0f}-{hi:.0f}px" if hi < 1e8 else f"{lo:.0f}+px"
    return "unbanded"


def evaluate_bubble_model(
    weights_path: str,
    test_crops_dir: str,
    test_labels_dir: str,
    output_dir: str,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.72,
    config: Optional[BubbleConfig] = None,
    device: Optional[str] = None,
    split_note: str = "unspecified -- state whether these crops were seen in training",
    restrict_to: Optional[set] = None,
    edge_rule: bool = True,
) -> Dict[str, Any]:
    config = config or BubbleConfig()
    os.makedirs(output_dir, exist_ok=True)
    overlays_dir = os.path.join(output_dir, "eval_overlays")
    os.makedirs(overlays_dir, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, ckpt = load_bubble_maskrcnn(weights_path, config=config, device=device)
    model.eval()
    print(f"[Eval] anchors: {getattr(model, 'anchor_scheme', '?')}")

    metrology = BubbleMetrology(config)

    samples = []
    for root, _, files in os.walk(test_crops_dir):
        for f in sorted(files):
            if not f.endswith(".png") or f.endswith(("_view.png", "_mask.png", "_ignore.png")):
                continue
            stem = os.path.splitext(f)[0]
            if restrict_to is not None and stem not in restrict_to:
                continue
            rel = os.path.relpath(os.path.join(root, f), test_crops_dir)
            lf = os.path.join(test_labels_dir, os.path.dirname(rel), f"{stem}_labels.json")
            if not os.path.exists(lf):
                lf = os.path.join(test_labels_dir, f"{stem}_labels.json")
            if os.path.exists(lf):
                samples.append((os.path.join(root, f), lf, f))

    tot_gt = tot_pred = tot_tp = 0
    tot_ignored_pred = 0
    n_edge_gt = n_edge_pred = 0   # edge rule: bubbles cut by the image border are not scored
    n_subfloor_pred = 0      # predictions under the 12 px floor -- same rule as the labels
    fp_records: List[Dict[str, Any]] = []   # unmatched kept predictions, for scripts/audit_fp.py
    missed_entirely = touched_unmatched = 0
    matched_ious: List[float] = []
    matched_box_ious: List[float] = []
    gt_d: List[float] = []
    pred_d_all: List[float] = []
    pred_d_gated: List[float] = []
    n_gated_out = 0
    band_gt: Dict[str, int] = {}
    band_tp: Dict[str, int] = {}
    crop_reports = []

    for img_path, label_file, crop_name in samples:
        with open(label_file, "r", encoding="utf-8") as f:
            doc = json.load(f)
        gt_bubbles = doc.get("bubbles", [])

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        ignore = schema.rasterise_ignore(doc.get("ignore_regions", []), h, w).astype(bool)
        side = label_file.replace("_labels.json", "_ignore.png")
        if os.path.exists(side):
            extra = cv2.imread(side, cv2.IMREAD_GRAYSCALE)
            if extra is not None and extra.shape[:2] == (h, w):
                ignore |= extra > 127

        gt = []
        for b in gt_bubbles:
            a, c = schema.semi_axes(b)
            if edge_rule and (float(b["cx"]) - a < 1 or float(b["cy"]) - c < 1
                              or float(b["cx"]) + a > w - 1 or float(b["cy"]) + c > h - 1):
                # edge rule: a bubble cut by the image border has a truncated size --
                # not scored, and its area joins the ignore mask
                n_edge_gt += 1
                em = np.zeros((h, w), np.uint8)
                cv2.circle(em, (int(round(float(b["cx"]))), int(round(float(b["cy"])))), max(1, int(round(a))), 1, -1)
                ignore |= em.astype(bool)
                continue
            m = np.zeros((h, w), dtype=np.uint8)
            centre = (int(round(float(b["cx"]))), int(round(float(b["cy"]))))
            if b.get("shape_type") == "ellipse":
                cv2.ellipse(m, centre, (max(1, int(round(a))), max(1, int(round(c)))),
                            float(b.get("angle", 0.0)), 0, 360, 1, -1)
            else:
                cv2.circle(m, centre, max(1, int(round(a))), 1, -1)
            d = schema.equiv_diameter_px(b)
            gt.append({"mask": m.astype(bool), "d": d,
                       "box": [float(b["cx"]) - a, float(b["cy"]) - c,
                               float(b["cx"]) + a, float(b["cy"]) + c],
                       "at_limit": bool(b.get("at_limit", False)),
                       "touches_edge": bool(b.get("touches_edge", False))})
            gt_d.append(d)
            band_gt[_band(d)] = band_gt.get(_band(d), 0) + 1
        tot_gt += len(gt)

        img_t = torch.as_tensor(img_rgb.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        with torch.no_grad():
            out = model([img_t.to(device)])[0]

        scores = out["scores"].cpu().numpy()
        pmasks = out["masks"].squeeze(1).cpu().numpy()
        preds = []
        for i in range(len(scores)):
            if scores[i] < score_threshold:
                continue
            m = (pmasks[i] >= 0.5)
            if not m.any():
                continue
            d_eq = mask_d_eq(m)
            if d_eq < schema.FLOOR_PX:
                # labels cannot go below the floor, so neither can predictions
                n_subfloor_pred += 1
                continue
            if edge_rule:
                ys_, xs_ = np.nonzero(m)
                if xs_.min() <= 0 or ys_.min() <= 0 or xs_.max() >= w - 1 or ys_.max() >= h - 1:
                    n_edge_pred += 1
                    continue
            ys, xs = np.nonzero(m)
            preds.append({"mask": m, "d": d_eq, "score": float(scores[i]),
                          "circ": mask_circularity(m),
                          "box": [float(xs.min()), float(ys.min()),
                                  float(xs.max() + 1), float(ys.max() + 1)]})

        # ---- greedy one-to-one by IoU, highest first ----
        pairs = []
        for pi, p in enumerate(preds):
            for gi, g in enumerate(gt):
                v = mask_iou(p["mask"], g["mask"])
                if v > 0:
                    pairs.append((v, pi, gi))
        pairs.sort(reverse=True)
        used_p, used_g = set(), set()
        touched_g = set(gi for _, _, gi in pairs)
        crop_tp = 0
        for v, pi, gi in pairs:
            if v < iou_threshold or pi in used_p or gi in used_g:
                continue
            used_p.add(pi); used_g.add(gi)
            matched_ious.append(v)
            xa = max(preds[pi]["box"][0], gt[gi]["box"][0]); ya = max(preds[pi]["box"][1], gt[gi]["box"][1])
            xb = min(preds[pi]["box"][2], gt[gi]["box"][2]); yb = min(preds[pi]["box"][3], gt[gi]["box"][3])
            inter = max(0.0, xb - xa) * max(0.0, yb - ya)
            a1 = (preds[pi]["box"][2] - preds[pi]["box"][0]) * (preds[pi]["box"][3] - preds[pi]["box"][1])
            a2 = (gt[gi]["box"][2] - gt[gi]["box"][0]) * (gt[gi]["box"][3] - gt[gi]["box"][1])
            matched_box_ious.append(inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0)
            band_tp[_band(gt[gi]["d"])] = band_tp.get(_band(gt[gi]["d"])) or 0
            band_tp[_band(gt[gi]["d"])] += 1
            crop_tp += 1

        for gi in range(len(gt)):
            if gi in used_g:
                continue
            if gi in touched_g:
                touched_unmatched += 1
            else:
                missed_entirely += 1

        # ---- ignore filter, applied only to unmatched predictions ----
        kept_pred = 0
        for pi, p in enumerate(preds):
            if pi not in used_p and ignore.any():
                inside = float(np.count_nonzero(p["mask"] & ignore))
                if inside / max(1.0, float(np.count_nonzero(p["mask"]))) >= 0.5:
                    tot_ignored_pred += 1
                    continue
            kept_pred += 1
            if pi not in used_p:
                cs, _ = cv2.findContours(p["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cpts = max(cs, key=cv2.contourArea).reshape(-1, 2) if cs else np.zeros((0, 2), int)
                fp_records.append({
                    "fp_id": len(fp_records), "crop": crop_name, "image": img_path, "labels": label_file,
                    "score": p["score"], "d_px": p["d"], "box": p["box"],
                    "max_iou_with_gt": float(max((v for v, pj, _ in pairs if pj == pi), default=0.0)),
                    "touches_image_edge": bool(p["box"][0] <= 1 or p["box"][1] <= 1
                                               or p["box"][2] >= w - 1 or p["box"][3] >= h - 1),
                    "contour": cpts[::max(1, len(cpts) // 60)].tolist(),
                })
            pred_d_all.append(p["d"])
            if p["circ"] >= CIRCULARITY_GATE:
                pred_d_gated.append(p["d"])
            else:
                n_gated_out += 1

        tot_pred += kept_pred
        tot_tp += crop_tp

        overlay = img_bgr.copy()
        if ignore.any():
            overlay[ignore] = (0.72 * overlay[ignore] + 0.28 * np.array([0, 140, 220])).astype(np.uint8)
        for p in preds:
            cs, _ = cv2.findContours(p["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cs, -1, (255, 255, 0), 2)
        for g in gt:
            cs, _ = cv2.findContours(g["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cs, -1, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(overlays_dir, f"{os.path.splitext(crop_name)[0]}_eval_overlay.png"), overlay)

        crop_reports.append({"crop": crop_name, "gt": len(gt), "pred": kept_pred, "tp": crop_tp})

    precision = tot_tp / max(1, tot_pred)
    recall = tot_tp / max(1, tot_gt)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    um = config.um_per_px
    d32_gt = metrology.compute_sauter_d32([d * um for d in gt_d])
    d32_raw = metrology.compute_sauter_d32([d * um for d in pred_d_all])
    d32_gated = metrology.compute_sauter_d32([d * um for d in pred_d_gated])

    results = {
        "weights": weights_path,
        "split_note": split_note,
        "crops_evaluated": len(crop_reports),
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "iou_kind": "mask",
        "diameter_measurand": "equivalent-area diameter (2*sqrt(A/pi)), both sides",
        "total_gt_bubbles": tot_gt,
        "total_pred_bubbles": tot_pred,
        "predictions_dropped_inside_ignore": tot_ignored_pred,
        "predictions_dropped_below_floor": n_subfloor_pred,
        "false_positives": len(fp_records),
        "edge_rule": edge_rule,
        "gt_dropped_touching_edge": n_edge_gt,
        "predictions_dropped_touching_edge": n_edge_pred,
        "gt_d_px": [round(d, 2) for d in gt_d],
        "pred_d_px": [round(d, 2) for d in pred_d_all],
        "true_positives": tot_tp,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mean_mask_iou_matched": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "mean_box_iou_matched_for_reference": float(np.mean(matched_box_ious)) if matched_box_ious else 0.0,
        "missed_entirely": missed_entirely,
        "touched_but_unmatched": touched_unmatched,
        "recall_by_size_band": {b: {"gt": band_gt[b], "tp": band_tp.get(b, 0),
                                    "recall": band_tp.get(b, 0) / band_gt[b]} for b in sorted(band_gt)},
        "d32_gt_um": d32_gt,
        "d32_pred_um_ungated": d32_raw,
        "d32_pred_um_circularity_gated": d32_gated,
        "d32_ratio_gated": (d32_gated / d32_gt) if d32_gt > 0 else 0.0,
        "d32_ratio_ungated": (d32_raw / d32_gt) if d32_gt > 0 else 0.0,
        "masks_rejected_by_circularity_gate": n_gated_out,
        "calibration_status": "provisional -- um are 3.291 um/px unconfirmed; pixels are the honest unit",
        "human_ceiling_reference": {"count_agreement": 0.82, "mean_mask_iou": 0.875, "d32_ratio": 0.96},
        "per_crop": crop_reports,
    }

    print("\n=================== EVALUATION ===================")
    print(f"split: {split_note}")
    print(f"crops {len(crop_reports)} | GT {tot_gt} | pred {tot_pred} "
          f"({tot_ignored_pred} dropped inside ignore, {n_subfloor_pred} below the 12 px floor) | TP {tot_tp} | FP {len(fp_records)}")
    print(f"recall {recall:.4f}   precision {precision:.4f}   F1 {f1:.4f}")
    print(f"mean MASK IoU {results['mean_mask_iou_matched']:.4f}   "
          f"(box IoU for reference {results['mean_box_iou_matched_for_reference']:.4f}; human ceiling 0.875)")
    print(f"missed entirely {missed_entirely} | touched but unmatched {touched_unmatched}")
    print(f"edge rule {'on' if edge_rule else 'off'}: {n_edge_gt} labelled and {n_edge_pred} predicted "
          f"bubble(s) cut by the image border not scored")
    for b, v in results["recall_by_size_band"].items():
        print(f"   {b:>10s}  n={v['gt']:3d}  recall {v['recall']:.2f}")
    print(f"D32 GT {d32_gt:.1f} um | pred gated {d32_gated:.1f} (ratio {results['d32_ratio_gated']:.4f}) "
          f"| ungated {d32_raw:.1f} (ratio {results['d32_ratio_ungated']:.4f})")
    print(f"{n_gated_out} mask(s) rejected by the {CIRCULARITY_GATE} circularity gate")
    print("um are provisional; the honest unit is pixels")
    print("==================================================")

    with open(os.path.join(output_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(output_dir, "false_positives.json"), "w", encoding="utf-8") as f:
        json.dump(fp_records, f)
    return results
