"""bubble_ui.engine_pulp
Pulp detection engine powered by BubMask (Mask R-CNN ResNet-50-FPN-V2).
Provides high-accuracy sub-pixel bubble detection, geometric fitting, and video streaming.
"""

from __future__ import annotations
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import cv2
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
PULP_DIR = os.path.join(PROJECT_ROOT, "BubMask")

if PULP_DIR not in sys.path:
    sys.path.insert(0, PULP_DIR)

from bubble_rcnn.config import BubbleConfig
from bubble_rcnn.models.mask_rcnn_v2 import load_bubble_maskrcnn
from bubble_rcnn.reconstruction.circle_fit import ShapeReconstructor

DEFAULT_WEIGHTS = os.path.join(PULP_DIR, "runs", "rr_full_e20", "best.pt")
PROVISIONAL_UM_PER_PX = 3.291  # 303.858 px/mm optical calibration


class PulpDetector:
    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = BubbleConfig()
        self.weights_path = weights_path or DEFAULT_WEIGHTS
        self.model: Optional[torch.nn.Module] = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.weights_path):
            # Fallback search in runs
            fallback = os.path.join(PULP_DIR, "runs", "rr_full_e20", "best.pt")
            if os.path.exists(fallback):
                self.weights_path = fallback
            else:
                raise FileNotFoundError(f"Pulp model checkpoint not found: {self.weights_path}")

        model, checkpoint = load_bubble_maskrcnn(self.weights_path, config=self.config, device=self.device)
        model.eval()
        self.model = model

    def set_weights(self, new_weights_path: str):
        if os.path.exists(new_weights_path):
            self.weights_path = new_weights_path
            self._load_model()
        else:
            raise FileNotFoundError(f"Weights file not found: {new_weights_path}")

    def detect_frame(
        self,
        img_bgr: np.ndarray,
        score_thresh: float = 0.40,
        reconstruction_method: str = "equivalent_area",   # same measurand as labels and eval; MEC runs ~9% high
        frame_idx: int = 0
    ) -> List[Dict[str, Any]]:
        """Runs Mask R-CNN on an image and fits analytical geometry."""
        if self.model is None:
            self._load_model()

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_t = torch.as_tensor(img_rgb.transpose(2, 0, 1), dtype=torch.float32) / 255.0

        with torch.no_grad():
            preds = self.model([img_t.to(self.device)])[0]

        scores = preds["scores"].cpu().numpy()
        masks = preds["masks"].squeeze(1).cpu().numpy()
        boxes = preds["boxes"].cpu().numpy()

        bubbles: List[Dict[str, Any]] = []
        b_id = 1

        for idx in range(len(scores)):
            score = float(scores[idx])
            if score < score_thresh:
                continue

            bin_mask = (masks[idx] >= 0.5).astype(np.uint8)
            if reconstruction_method == "equivalent_area":
                area = float(bin_mask.sum())
                if area <= 0:
                    continue
                ys, xs = np.nonzero(bin_mask)
                rec = {"center_xy": (float(xs.mean()), float(ys.mean())),
                       "diameter_px": 2.0 * float(np.sqrt(area / np.pi))}
            else:
                rec = ShapeReconstructor.reconstruct_from_binary_mask(bin_mask, method=reconstruction_method)
            if rec is None:
                continue

            cx = float(rec["center_xy"][0])
            cy = float(rec["center_xy"][1])
            d_px = float(rec["diameter_px"])
            r = d_px / 2.0

            # Contour for client rendering
            cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            approx_cnt = []
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                # Subsample contour points for json efficiency
                pts = c.squeeze().tolist()
                if isinstance(pts[0], list):
                    approx_cnt = pts[:: max(1, len(pts) // 30)]

            bubble_rec = {
                "id": b_id,
                "frame_idx": frame_idx,
                "touches_image_edge": bool(bin_mask[0, :].any() or bin_mask[-1, :].any()
                                           or bin_mask[:, 0].any() or bin_mask[:, -1].any()),
                "cx": round(cx, 1),
                "cy": round(cy, 1),
                "r": round(r, 1),
                "diameter_px": round(d_px, 2),
                "score": round(score, 4),
                "circularity": round(float(rec.get("circularity", 1.0)), 3),
                "solidity": round(float(rec.get("solidity", 1.0)), 3),
                "major": round(float(rec.get("major_axis_px", d_px)), 2),
                "minor": round(float(rec.get("minor_axis_px", d_px)), 2),
                "angle": round(float(rec.get("angle_deg", 0.0)), 1),
                "bbox": [int(boxes[idx][0]), int(boxes[idx][1]), int(boxes[idx][2]), int(boxes[idx][3])],
                "contour": approx_cnt,
            }
            bubbles.append(bubble_rec)
            b_id += 1

        return bubbles

    def detect_video(
        self,
        video_path: str,
        stride: int = 5,
        max_frames: int = 50,
        score_thresh: float = 0.40,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[int, str]]:
        """Processes video stream across frames and collects all bubble instances."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = list(range(0, total_frames, stride))
        if max_frames and len(frame_indices) > max_frames:
            frame_indices = frame_indices[:max_frames]

        all_bubbles: List[Dict[str, Any]] = []
        sample_frames: Dict[int, str] = {}  # Base64 or frame preview

        global_id = 1
        for step, f_idx in enumerate(frame_indices, start=1):
            if progress_cb:
                progress_cb(step, len(frame_indices), f"Processing frame {f_idx}/{total_frames}...")

            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame_bubbles = self.detect_frame(frame, score_thresh=score_thresh, frame_idx=f_idx)
            for b in frame_bubbles:
                b["id"] = global_id
                global_id += 1
                all_bubbles.append(b)

        cap.release()
        return all_bubbles, sample_frames
