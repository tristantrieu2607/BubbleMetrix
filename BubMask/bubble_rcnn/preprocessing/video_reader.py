"""Video extraction module supporting 1024x1024 rawvideo AVI and portrait MP4."""

import os
import cv2
import numpy as np
from typing import Generator, Tuple, List, Optional
from bubble_rcnn.config import BubbleConfig

class VideoReader:
    def __init__(self, video_path: str, config: Optional[BubbleConfig] = None):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        self.video_path = video_path
        self.config = config or BubbleConfig()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        self.is_froth = "froth" in video_path.lower() or "180ww" in os.path.basename(video_path).lower()
        self.is_venturi = "venturi" in os.path.basename(video_path).lower() or "dilute" in os.path.basename(video_path).lower()
        self.frame_gap = self.config.venturi_frame_gap if self.is_venturi else self.config.rotor_frame_gap

    def get_decorrelated_frame_indices(self) -> List[int]:
        """Returns frame indices spaced >= frame_gap to guarantee statistically independent bubble populations."""
        return list(range(0, self.total_frames, self.frame_gap))

    def read_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Reads a specific frame index."""
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None
        return frame

    def iterate_decorrelated_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Iterates over independent frames with minimal seek overhead."""
        cap = cv2.VideoCapture(self.video_path)
        indices = set(self.get_decorrelated_frame_indices())
        current_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if current_idx in indices:
                yield current_idx, frame
            current_idx += 1
        cap.release()

    def get_fov_mask(self, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Returns circular FOV aperture mask to eliminate corner vignette."""
        h = shape[0] if shape else self.height
        w = shape[1] if shape else self.width
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = self.config.fov_center_xy
        r = self.config.fov_radius_px
        cv2.circle(mask, (cx, cy), r, 255, -1)
        return mask
