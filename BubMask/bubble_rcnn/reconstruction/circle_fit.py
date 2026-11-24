"""Bubble Shape Reconstruction module implementing Welzl's Min Enclosing Circle and Fitzgibbon's Direct Ellipse."""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional

class ShapeReconstructor:
    """
    Implements the geometric shape reconstruction methodology from Cui et al. (2022).
    - Sub-millimeter bubbles under high surface tension are physically spherical.
    - Fits Minimum Enclosing Circle (Welzl O(N) algorithm) to eliminate false internal specular outlines
      and reconstruct occluded spherical boundaries.
    - Fits Fitzgibbon's Direct Least-Squares Ellipse (cv2.fitEllipseDirect) for deformed bubbles.
    """
    @staticmethod
    def reconstruct_from_binary_mask(mask: np.ndarray, method: str = "min_enclosing_circle") -> Optional[Dict[str, Any]]:
        """
        Takes a single instance binary mask (uint8), extracts the exterior contour,
        and fits the reconstructed geometry.
        """
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None

        # Choose the largest contour by area
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 3:
            return None

        # Filter internal specular noise via convex hull
        hull = cv2.convexHull(contour)

        if method == "min_enclosing_circle":
            (cx, cy), radius = cv2.minEnclosingCircle(hull)
            diameter = 2.0 * radius
            area = np.pi * (radius ** 2)
            return {
                "shape_type": "circle",
                "center_xy": (float(cx), float(cy)),
                "radius_px": float(radius),
                "diameter_px": float(diameter),
                "area_px": float(area),
                "reconstructed_mask_generator": lambda h, w: cv2.circle(np.zeros((h, w), dtype=np.uint8), (int(round(cx)), int(round(cy))), int(round(radius)), 1, -1)
            }
        
        elif method == "fit_ellipse":
            # Hierarchical safeguard: fitEllipseDirect requires >= 5 points
            if len(hull) >= 5:
                try:
                    # cv2.fitEllipseDirect implements Fitzgibbon et al. (1999) Direct Least Squares
                    # returning: ((cx, cy), (width, height), angle_deg)
                    box = cv2.fitEllipseDirect(hull)
                    (cx, cy), (d1, d2), angle = box
                    major_d = max(d1, d2)
                    minor_d = min(d1, d2)

                    # Clamp physically implausible eccentricities
                    if minor_d > 0 and (major_d / minor_d) <= 3.0:
                        equiv_d = np.sqrt(major_d * minor_d)
                        area = np.pi * (major_d / 2.0) * (minor_d / 2.0)
                        return {
                            "shape_type": "ellipse",
                            "center_xy": (float(cx), float(cy)),
                            "major_axis_px": float(major_d),
                            "minor_axis_px": float(minor_d),
                            "diameter_px": float(equiv_d),
                            "area_px": float(area),
                            "angle_deg": float(angle),
                            "reconstructed_mask_generator": lambda h, w: cv2.ellipse(np.zeros((h, w), dtype=np.uint8), (int(round(cx)), int(round(cy))), (int(round(d1/2)), int(round(d2/2))), angle, 0, 360, 1, -1)
                        }
                except cv2.error:
                    pass

            # Fallback to Minimum Enclosing Circle if points < 5 or ellipse degenerates
            return ShapeReconstructor.reconstruct_from_binary_mask(mask, method="min_enclosing_circle")

        elif method in ("polygon_facet", "direct_facet"):
            # For 2D froth foam facets: compute centroid and exact polygon contour area (no circle bias)
            area = float(cv2.contourArea(contour))
            if area <= 0:
                area = float(np.sum(mask))
            diameter = 2.0 * np.sqrt(area / np.pi)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
            else:
                (cx, cy), _ = cv2.minEnclosingCircle(contour)
            return {
                "shape_type": "polygon_facet",
                "center_xy": (float(cx), float(cy)),
                "diameter_px": float(diameter),
                "area_px": float(area)
            }

        else:
            raise ValueError(f"Unknown shape reconstruction method: {method}")
