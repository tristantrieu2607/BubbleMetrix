"""Crop sampler module: extracts stratified 512x512 crops from real video clips across both pulp and froth regimes."""

import os
import cv2
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from bubble_rcnn.config import BubbleConfig
from bubble_rcnn.preprocessing.video_reader import VideoReader
from bubble_rcnn.preprocessing.fouling_detector import OpticalConditioner

def find_all_videos(video_dir: str) -> List[str]:
    """Finds all .avi and .mp4 video files recursively or in the specified directory."""
    videos = []
    if os.path.isfile(video_dir) and (video_dir.lower().endswith(".avi") or video_dir.lower().endswith(".mp4")):
        return [video_dir]
    for root, _, files in os.walk(video_dir):
        for f in files:
            if f.lower().endswith(".avi") or f.lower().endswith(".mp4"):
                videos.append(os.path.join(root, f))
    return sorted(videos)

def extract_stratified_crops(
    video_dir: str,
    output_dir: str,
    crop_size: int = 512,
    crops_per_clip: int = 3,
    config: Optional[BubbleConfig] = None
) -> List[Dict[str, Any]]:
    """
    Extracts stratified 512x512 crops from all video clips in video_dir (handling both pulp and froth).
    - Pulp videos (1024x1024): sampled inside circular FOV window.
    - Froth videos (720x1280): sampled inside the rectangular foam column between glass walls.
    """
    config = config or BubbleConfig()
    os.makedirs(output_dir, exist_ok=True)
    manifest = []

    video_paths = find_all_videos(video_dir)
    if not video_paths:
        raise FileNotFoundError(f"No .avi or .mp4 video clips found in {video_dir}")

    conditioner = OpticalConditioner()

    for v_path in video_paths:
        v_name = os.path.basename(v_path)
        reader = VideoReader(v_path, config)
        clip_stem = os.path.splitext(v_name)[0]
        
        is_froth = reader.is_froth
        is_venturi = reader.is_venturi

        if is_froth:
            regime = "froth_mosaic"
            # Froth video (180WW) has 14,095 frames (~470 seconds).
            # Sample across time: e.g. t = 10s (f300), 30s (f900), 60s (f1800), 120s (f3600), 240s (f7200)
            candidate_times_sec = [10, 30, 60, 120, 240, 360]
            selected_frames = [
                min(int(t * reader.fps), reader.total_frames - 1)
                for t in candidate_times_sec[:crops_per_clip]
            ]
            # Froth column: Width 720, cell inner walls at x ~ 150 to 640 (width ~490).
            # Center a 512 crop at X: [104, 616]
            # Height 1280: Y: [380, 892] (middle foam) and Y: [680, 1192] (lower foam)
            crop_origins = [
                (104, 400),
                (104, 680),
                (104, 300)
            ]
        else:
            regime = "venturi_bright" if is_venturi else "rotor_dark"
            frames_indices = reader.get_decorrelated_frame_indices()
            step = max(1, len(frames_indices) // crops_per_clip)
            selected_frames = [frames_indices[i * step] for i in range(min(crops_per_clip, len(frames_indices)))]
            # Pulp FOV center is (524, 508), radius 601. Safe inner window: [250, 750]
            crop_origins = [
                (256, 256),  # Center
                (128, 256),  # Left-center
                (256, 128)   # Top-center
            ]

        for f_count, f_idx in enumerate(selected_frames):
            frame = reader.read_frame(f_idx)
            if frame is None:
                continue
            
            origin = crop_origins[f_count % len(crop_origins)]
            x, y = origin

            # Ensure crop stays within image boundary
            h, w = frame.shape[:2]
            x_end = min(w, x + crop_size)
            y_end = min(h, y + crop_size)
            x_start = max(0, x_end - crop_size)
            y_start = max(0, y_end - crop_size)

            crop = frame[y_start:y_end, x_start:x_end]
            if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
                continue

            crop_id = f"{clip_stem}_f{f_idx:05d}_x{x_start}y{y_start}"
            crop_filename = f"{crop_id}.png"
            crop_path = os.path.join(output_dir, crop_filename)
            
            # Write raw crop
            cv2.imwrite(crop_path, crop)

            # Generate view-enhanced crop (CLAHE) for inspection
            enhanced_crop = conditioner.enhance_contrast(crop)
            view_path = os.path.join(output_dir, f"{crop_id}_view.png")
            cv2.imwrite(view_path, enhanced_crop)

            entry = {
                "crop_id": crop_id,
                "video_file": v_name,
                "regime": regime,
                "frame_idx": f_idx,
                "origin_xy": (x_start, y_start),
                "crop_size": crop_size,
                "raw_file": crop_filename,
                "view_file": f"{crop_id}_view.png"
            }
            manifest.append(entry)

    # Save manifest
    manifest_path = os.path.join(output_dir, "crop_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Extracted {len(manifest)} crops into {output_dir}")
    return manifest
