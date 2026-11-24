"""Calibrated metrology module computing equivalent diameters, Sauter D32, and PSD."""

import numpy as np
from typing import List, Dict, Any, Optional
from bubble_rcnn.config import BubbleConfig

class BubbleMetrology:
    def __init__(self, config: Optional[BubbleConfig] = None):
        self.config = config or BubbleConfig()

    def compute_sauter_d32(self, diameters_um: List[float]) -> float:
        """
        Computes volume-weighted Sauter Mean Diameter D32:
        D32 = sum(d^3) / sum(d^2)
        """
        if not diameters_um:
            return 0.0
        d = np.asarray(diameters_um, dtype=np.float64)
        sum_d2 = np.sum(d ** 2)
        if sum_d2 == 0:
            return 0.0
        return float(np.sum(d ** 3) / sum_d2)

    def compute_debrouckere_d43(self, diameters_um: List[float]) -> float:
        """
        Computes De Brouckere Mean Diameter D43:
        D43 = sum(d^4) / sum(d^3)
        """
        if not diameters_um:
            return 0.0
        d = np.asarray(diameters_um, dtype=np.float64)
        sum_d3 = np.sum(d ** 3)
        if sum_d3 == 0:
            return 0.0
        return float(np.sum(d ** 4) / sum_d3)

    def analyze_instances(self, reconstructed_records: List[Dict[str, Any]],
                          um_per_px: Optional[float] = None,
                          apply_fov_gate: bool = True) -> Dict[str, Any]:
        """Statistics across reconstructed records, in whatever unit is honest.

        ``um_per_px`` must be passed by the caller.  v1 always used the pulp
        constant here while ``measure_video`` used a different constant for froth
        when writing the CSV, so the two outputs of a single froth run disagreed by
        34x.  Passing ``None`` means "no calibration": everything is reported in
        pixels and the micron fields are null.

        ``apply_fov_gate`` exists because the FOV circle in the config is the *pulp
        probe's* -- centre (524, 508), r 601.  Applying it to a froth column is
        meaningless, so the caller says whether it applies.
        """
        calibrated = um_per_px is not None
        scale = float(um_per_px) if calibrated else 1.0
        valid_diameters_px = []
        valid_diameters_um = []
        accepted_records = []
        rejected_records = []

        for rec in reconstructed_records:
            d_px = rec["diameter_px"]
            cx, cy = rec["center_xy"]

            # Filter 1: Size floor & ceiling (12 px <= d <= 120 px)
            if d_px < self.config.min_diameter_px or d_px > self.config.max_diameter_px:
                rec["rejection_reason"] = f"size_out_of_bounds ({d_px:.1f}px)"
                rejected_records.append(rec)
                continue

            # Filter 2: Circular FOV boundary clearance (avoid clipped corner bubbles)
            # Distance from FOV center -- pulp probe geometry only
            if not apply_fov_gate:
                rec["diameter_um"] = float(d_px * scale) if calibrated else None
                rec["d_eq_px"] = float(d_px)
                valid_diameters_px.append(d_px)
                valid_diameters_um.append(d_px * scale)
                accepted_records.append(rec)
                continue
            fov_cx, fov_cy = self.config.fov_center_xy
            dist_from_center = np.hypot(cx - fov_cx, cy - fov_cy)
            if (dist_from_center + (d_px / 2.0)) > self.config.fov_radius_px:
                rec["rejection_reason"] = "touches_fov_boundary"
                rejected_records.append(rec)
                continue

            d_um = d_px * scale
            rec["diameter_um"] = float(d_um) if calibrated else None
            rec["d_eq_px"] = float(d_px)
            
            valid_diameters_px.append(d_px)
            valid_diameters_um.append(d_um)
            accepted_records.append(rec)

        d32 = self.compute_sauter_d32(valid_diameters_um)
        d43 = self.compute_debrouckere_d43(valid_diameters_um)
        mean_d = float(np.mean(valid_diameters_um)) if valid_diameters_um else 0.0
        median_d = float(np.median(valid_diameters_um)) if valid_diameters_um else 0.0
        std_d = float(np.std(valid_diameters_um)) if valid_diameters_um else 0.0

        d32_px = self.compute_sauter_d32(valid_diameters_px)
        d43_px = self.compute_debrouckere_d43(valid_diameters_px)

        return {
            "total_detections": len(reconstructed_records),
            "accepted_count": len(accepted_records),
            "rejected_count": len(rejected_records),
            "calibrated": calibrated,
            "um_per_px": float(um_per_px) if calibrated else None,
            "sauter_mean_diameter_d32_px": d32_px,
            "debrouckere_mean_diameter_d43_px": d43_px,
            "mean_diameter_px": float(np.mean(valid_diameters_px)) if valid_diameters_px else 0.0,
            "median_diameter_px": float(np.median(valid_diameters_px)) if valid_diameters_px else 0.0,
            "std_diameter_px": float(np.std(valid_diameters_px)) if valid_diameters_px else 0.0,
            "n_accepted": len(valid_diameters_px),
            "sauter_mean_diameter_d32_um": d32 if calibrated else None,
            "debrouckere_mean_diameter_d43_um": d43 if calibrated else None,
            "mean_diameter_um": mean_d if calibrated else None,
            "median_diameter_um": median_d if calibrated else None,
            "std_diameter_um": std_d if calibrated else None,
            "accepted_bubbles": accepted_records,
            "rejected_bubbles": rejected_records
        }
