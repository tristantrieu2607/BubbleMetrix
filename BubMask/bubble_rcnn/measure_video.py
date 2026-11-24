"""Full video measurement and sizing pipeline producing calibrated bubble size distributions."""

import os
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional

from bubble_rcnn.config import BubbleConfig
from bubble_rcnn.models.mask_rcnn_v2 import load_bubble_maskrcnn
from bubble_rcnn.preprocessing.video_reader import VideoReader
from bubble_rcnn.preprocessing.fouling_detector import OpticalConditioner
from bubble_rcnn.reconstruction.circle_fit import ShapeReconstructor
from bubble_rcnn.sizing.metrology import BubbleMetrology
from bubble_rcnn import calibration as calib

def measure_video_clip(
    video_path: str,
    weights_path: str,
    output_dir: str,
    score_threshold: float = 0.5,
    sample_stride: Optional[int] = None,
    config: Optional[BubbleConfig] = None,
    device: Optional[str] = None,
    allow_provisional_um: bool = False,
    exclude_edge: bool = True,
) -> Dict[str, Any]:
    """Measure a clip.

    Units follow handover principle 6: microns are emitted only for a calibration
    marked ``measured``.  Pass ``allow_provisional_um=True`` to force microns from
    the provisional 3.291 um/px -- the number will then carry that word everywhere
    it appears.  The froth column has no calibration at all, so it is pixels only
    whatever this flag says.
    """
    config = config or BubbleConfig()
    os.makedirs(output_dir, exist_ok=True)
    overlays_dir = os.path.join(output_dir, "overlays")
    os.makedirs(overlays_dir, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Model
    model, checkpoint = load_bubble_maskrcnn(weights_path, config=config, device=device)
    model.eval()

    reader = VideoReader(video_path, config)
    conditioner = OpticalConditioner()
    metrology = BubbleMetrology(config)

    # ---- units, decided once, up front -------------------------------------
    is_froth = reader.is_froth
    regime = "froth" if is_froth else "pulp"
    cal = calib.load(regime)
    if cal.may_emit_microns:
        effective_um_per_px = cal.um_per_px
        unit_note = f"measured calibration: {cal.describe()}"
    elif allow_provisional_um and cal.um_per_px is not None:
        effective_um_per_px = cal.um_per_px
        unit_note = f"PROVISIONAL calibration, not publishable: {cal.describe()}"
    else:
        effective_um_per_px = None
        unit_note = f"pixels only -- {cal.describe()}"
    print(f"[Measure] units: {unit_note}")

    # Frame indices to sample
    if sample_stride is not None:
        frame_indices = list(range(0, reader.total_frames, sample_stride))
    else:
        frame_indices = reader.get_decorrelated_frame_indices()

    print(f"[Measure] Analyzing {len(frame_indices)} decorrelated frames from {video_path}...")

    all_reconstructed_records = []
    n_edge = 0
    csv_rows = ["frame_idx,bubble_id,center_x_px,center_y_px,diameter_px,diameter_um,score"]
    # one scale for the CSV and the summary; v1 used a different one in each

    for f_idx in frame_indices:
        frame_bgr = reader.read_frame(f_idx)
        if frame_bgr is None:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_t = torch.as_tensor(frame_rgb.transpose(2, 0, 1), dtype=torch.float32) / 255.0

        with torch.no_grad():
            preds = model([img_t.to(device)])[0]

        scores = preds["scores"].cpu().numpy()
        masks = preds["masks"].squeeze(1).cpu().numpy()
        keep = scores >= score_threshold

        overlay = frame_bgr.copy()
        frame_records = []

        reconstruction_method = "polygon_facet" if is_froth else "min_enclosing_circle"

        for b_id, idx in enumerate(range(len(scores)), start=1):
            if not keep[idx]:
                continue
            bin_mask = (masks[idx] >= 0.5).astype(np.uint8)
            if exclude_edge and not is_froth and (bin_mask[0, :].any() or bin_mask[-1, :].any()
                                                  or bin_mask[:, 0].any() or bin_mask[:, -1].any()):
                n_edge += 1   # cut by the image border: truncated size
                continue
            if is_froth:
                rec = ShapeReconstructor.reconstruct_from_binary_mask(bin_mask, method=reconstruction_method)
            else:
                # equivalent-area diameter: the measurand of the labels and of eval
                area = float(bin_mask.sum())
                if area <= 0:
                    continue
                ys, xs = np.nonzero(bin_mask)
                rec = {"center_xy": (float(xs.mean()), float(ys.mean())),
                       "diameter_px": 2.0 * float(np.sqrt(area / np.pi))}
            if rec is None:
                continue

            cx, cy = int(round(rec["center_xy"][0])), int(round(rec["center_xy"][1]))

            # Vertical ROI gate for froth: ignore top coalescence and bottom liquid
            if is_froth and not (380 <= cy <= 950):
                continue

            rec["frame_idx"] = f_idx
            rec["bubble_id"] = b_id
            rec["score"] = float(scores[idx])
            frame_records.append(rec)

            r = int(round(rec["diameter_px"] / 2.0))
            d_um = (rec["diameter_px"] * effective_um_per_px) if effective_um_per_px else None
            tag = f"{d_um:.0f}um" if d_um is not None else f"{rec['diameter_px']:.0f}px"

            # Draw overlay with ID and size
            cv2.circle(overlay, (cx, cy), max(1, r), (0, 255, 0), 2)
            cv2.putText(overlay, tag, (cx - r, max(12, cy - r - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            um_field = f"{d_um:.2f}" if d_um is not None else ""
            csv_rows.append(f"{f_idx},{b_id},{cx},{cy},{rec['diameter_px']:.2f},{um_field},{rec['score']:.4f}")

        # Save sample overlay
        cv2.imwrite(os.path.join(overlays_dir, f"frame_{f_idx:03d}_measured.png"), overlay)
        all_reconstructed_records.extend(frame_records)

    # 2. Metrology analysis
    summary = metrology.analyze_instances(all_reconstructed_records,
                                          um_per_px=effective_um_per_px,
                                          apply_fov_gate=not is_froth)
    summary["video_file"] = os.path.basename(video_path)
    summary["frames_analyzed"] = len(frame_indices)
    summary["regime"] = regime
    summary["calibration"] = cal.describe()
    summary["calibration_status"] = cal.status
    summary["units_note"] = unit_note
    summary["fov_gate_applied"] = not is_froth
    summary["edge_rule"] = exclude_edge
    summary["n_excluded_touching_edge"] = n_edge
    if summary.get("n_accepted", 0) < 30:
        summary["sample_size_warning"] = (
            f"D32 computed from {summary.get('n_accepted', 0)} bubbles. D32 is "
            f"volume-weighted, so a handful of large masks set it; this is not a "
            f"size distribution.")

    # 3. Save CSV Table
    csv_path = os.path.join(output_dir, "bubble_measurements.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_rows))

    # 4. Generate & Save Histogram Plot
    use_um = effective_um_per_px is not None
    unit_lbl = "um" if use_um else "px"
    accepted_diameters = [(b["diameter_um"] if use_um else b["d_eq_px"])
                          for b in summary["accepted_bubbles"]]
    d32_plot = summary["sauter_mean_diameter_d32_um"] if use_um else summary["sauter_mean_diameter_d32_px"]
    mean_plot = summary["mean_diameter_um"] if use_um else summary["mean_diameter_px"]
    if accepted_diameters:
        plt.figure(figsize=(8, 5))
        plt.hist(accepted_diameters, bins=25, color="skyblue", edgecolor="black", alpha=0.7)
        plt.axvline(d32_plot, color="red", linestyle="--", linewidth=2, label=f"D32 = {d32_plot:.1f} {unit_lbl}")
        plt.axvline(mean_plot, color="blue", linestyle=":", linewidth=2, label=f"Mean = {mean_plot:.1f} {unit_lbl}")
        plt.title(f"Bubble Size Distribution: {os.path.basename(video_path)}  (n={len(accepted_diameters)})")
        plt.xlabel(f"Equivalent Diameter ({unit_lbl})")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        hist_path = os.path.join(output_dir, "bubble_size_distribution.png")
        plt.savefig(hist_path, dpi=200)
        plt.close()

    # 5. Save Summary JSON
    summary_path = os.path.join(output_dir, "measurement_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in summary.items() if k not in ("accepted_bubbles", "rejected_bubbles")}, f, indent=2)

    print(f"\n[Measure Complete] Video: {os.path.basename(video_path)}")
    print(f"Total Detected: {summary['total_detections']} | Accepted: {summary['accepted_count']}")
    print(f"D32: {d32_plot:.1f} {unit_lbl} | Mean: {mean_plot:.1f} {unit_lbl} | units: {unit_note}")
    print(f"Edge rule {'on' if exclude_edge else 'off'}: {n_edge} bubble(s) cut by the image border excluded")
    if "sample_size_warning" in summary:
        print(f"WARNING: {summary['sample_size_warning']}")
    print(f"Artifacts saved to: {output_dir}")

    return summary
