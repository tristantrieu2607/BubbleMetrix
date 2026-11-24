"""Optical preprocessing: Stationary fouling filter and CLAHE contrast enhancer."""

import cv2
import numpy as np
from typing import Tuple

class OpticalConditioner:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Applies CLAHE to enhance boundary contrast in over-exposed Venturi frames."""
        if len(image.shape) == 3:
            # Convert to LAB to equalize L channel without shifting hue
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            return self.clahe.apply(image)

    @staticmethod
    def compute_temporal_fouling_mask(frames: list, std_thresh: float = 8.0, dark_thresh: float = 60.0) -> np.ndarray:
        """
        Computes static fouling mask across a series of clip frames.
        Moving bubbles exhibit high temporal variance (std > 20).
        Static probe window fouling exhibits low variance (std < std_thresh) and dark intensity.
        """
        if not frames:
            raise ValueError("Empty frames list passed to compute_temporal_fouling_mask")
        
        # Stack frames along time axis (grayscale)
        gray_frames = []
        for f in frames:
            if len(f.shape) == 3:
                gray_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
            else:
                gray_frames.append(f)
        
        stack = np.stack(gray_frames, axis=0).astype(np.float32)
        std_map = np.std(stack, axis=0)
        mean_map = np.mean(stack, axis=0)

        # Fouling = persistently static AND dark
        fouling_mask = (std_map < std_thresh) & (mean_map < dark_thresh)
        fouling_mask = fouling_mask.astype(np.uint8) * 255

        # Morphological opening to eliminate isolated noise pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fouling_mask = cv2.morphologyEx(fouling_mask, cv2.MORPH_OPEN, kernel)
        return fouling_mask
