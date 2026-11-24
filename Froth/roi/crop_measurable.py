"""
crop_measurable.py - Automatic detection and cropping of measurable froth frames.

In froth flotation columns (e.g. 180WW.MP4), the top of the column often contains
defocused, blurry froth, bursting films, and fixtures that cannot be reliably
measured. Tiling SAM across that top band wastes CPU time (up to 40% of tiles)
and generates spurious or distorted masks.

This module analyzes the focus, sharpness, and high-frequency edge texture along
the column to automatically detect the in-focus, measurable bubble bed and crop
out both the unmeasurable upper froth and the bottom liquid/interface.

Usage:
  # Detect and visualize measurable ROI on a single frame
  python crop_measurable.py --image frames/frame_0000.png --save-vis --save-crop

  # Batch extract measurable frames from a video
  python crop_measurable.py --video ../samples/froth/180WW.MP4 --sample-every 30 --out-dir frames_measurable/
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np
from PIL import Image

try:
    from scipy.ndimage import gaussian_filter1d
except ImportError:
    gaussian_filter1d = None


def find_column_walls(gray, inset_px=8):
    """Locate the outer and inner glass column walls.

    Uses column-mean darkness transitions to detect glass walls.
    Returns:
        x0 (int): Left inner wall column with inset applied
        x1 (int): Right inner wall column with inset applied
        w_left (int): Raw detected left wall
        w_right (int): Raw detected right wall
    """
    H, W = gray.shape[:2]
    colmean = gray.mean(axis=0)
    mid = W // 2
    thresh = colmean.mean() - 0.5 * colmean.std()
    left_candidates = np.nonzero(colmean[:mid] < thresh)[0]
    right_candidates = np.nonzero(colmean[mid:] < thresh)[0]
    w_left = int(left_candidates[-1]) if len(left_candidates) else 0
    w_right = int(right_candidates[0] + mid) if len(right_candidates) else W - 1

    if w_right - w_left < 0.3 * W:  # Detection failed, use full width
        w_left, w_right = 0, W - 1

    x0 = max(0, w_left + inset_px)
    x1 = min(W, w_right - inset_px)
    return x0, x1, w_left, w_right


def compute_focus_profile(
    inner_gray,
    strip_h=32,
    step=8,
    metric="combined"
):
    """Calculates focus and texture metrics along vertical strips of the column.

    Args:
        inner_gray: Grayscale 2D array of the column interior between glass walls.
        strip_h: Height of each sliding strip in pixels.
        step: Step size in pixels between consecutive strips.
        metric: Focus metric ('combined', 'laplacian', 'sobel', 'canny').

    Returns:
        ys: Array of center Y coordinates for each strip.
        scores: Array of focus/texture scores corresponding to each Y.
        metrics_dict: Detailed metrics (laps, gmags, cannys, stds).
    """
    H, W = inner_gray.shape[:2]
    ys, laps, gmags, cannys, stds = [], [], [], [], []

    for y in range(0, max(1, H - strip_h), step):
        strip = inner_gray[y : y + strip_h, :]
        if strip.size == 0:
            continue

        # 1. Laplacian variance (high-frequency sharpness)
        lap_var = float(cv2.Laplacian(strip, cv2.CV_64F).var())

        # 2. Sobel gradient magnitude (edge energy)
        gx = cv2.Sobel(strip.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(strip.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        gmag = float(np.sqrt(gx**2 + gy**2).mean())

        # 3. Canny edge density (bubble boundary concentration)
        blur = cv2.GaussianBlur(strip, (3, 3), 1.0)
        canny = cv2.Canny(blur, 30, 80)
        canny_density = float(canny.mean() / 255.0)

        # 4. Standard deviation (local contrast)
        contrast = float(strip.std())

        ys.append(y + strip_h // 2)
        laps.append(lap_var)
        gmags.append(gmag)
        cannys.append(canny_density)
        stds.append(contrast)

    ys = np.array(ys, dtype=np.int32)
    laps = np.array(laps, dtype=np.float32)
    gmags = np.array(gmags, dtype=np.float32)
    cannys = np.array(cannys, dtype=np.float32)
    stds = np.array(stds, dtype=np.float32)

    if metric == "laplacian":
        scores = laps
    elif metric == "sobel":
        scores = gmags
    elif metric == "canny":
        scores = cannys
    else:  # 'combined' - multi-scale robust texture and sharpness index
        scores = gmags * np.sqrt(np.maximum(1.0, laps)) * (1.0 + 2.0 * cannys)

    return ys, scores, {
        "laps": laps,
        "gmags": gmags,
        "cannys": cannys,
        "stds": stds
    }


def detect_measurable_roi(
    img,
    focus_metric="combined",
    focus_thresh_frac=0.58,
    min_consecutive_strips=3,
    wall_inset_px=8,
    strip_h=32,
    step=8,
    smooth_sigma=2.0,
    min_height_px=100,
    manual_y_range=None
):
    """Detects the bounding box of the measurable, in-focus froth region.

    Args:
        img: RGB or BGR numpy array or file path.
        focus_metric: 'combined', 'laplacian', 'sobel', 'canny'.
        focus_thresh_frac: Fraction of peak in-focus score used as cutoff threshold (0.0 to 1.0).
        min_consecutive_strips: Minimum contiguous strips required for a valid bed.
        wall_inset_px: Inset in pixels from detected outer walls.
        strip_h: Sliding window height in pixels.
        step: Sliding window step in pixels.
        smooth_sigma: Gaussian filter sigma for smoothing vertical profile.
        min_height_px: Minimum height required for a valid ROI.

    Returns:
        roi: Tuple (x0, y0, x1, y1) in original image pixel coordinates.
        info: Dictionary with diagnostic profile curves and metadata.
    """
    if isinstance(img, str):
        img_arr = np.array(Image.open(img).convert("RGB"))
    else:
        img_arr = img

    if img_arr is None:
        raise ValueError("Invalid image provided")

    if len(img_arr.shape) == 3:
        gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_arr

    H, W = gray.shape[:2]

    # 1. Locate column walls (horizontal bounds)
    x0, x1, w_left, w_right = find_column_walls(gray, inset_px=wall_inset_px)
    inner_gray = gray[:, x0:x1]

    # 2. Compute vertical focus profile down the column
    ys, raw_scores, metrics = compute_focus_profile(
        inner_gray, strip_h=strip_h, step=step, metric=focus_metric
    )

    if len(ys) == 0:
        # Fallback if image is extremely small
        return (x0, 0, x1, H), {"fallback": True}

    # Smooth the profile to prevent single-pixel noise spikes
    if gaussian_filter1d is not None and smooth_sigma > 0:
        smoothed_scores = gaussian_filter1d(raw_scores, sigma=smooth_sigma)
    else:
        smoothed_scores = raw_scores

    # Focus peak search: focus on the middle 70% of the column to avoid top/bottom fixtures
    valid_range = (ys >= 0.15 * H) & (ys <= 0.88 * H)
    if not np.any(valid_range):
        valid_range = np.ones_like(ys, dtype=bool)

    peak_score = float(smoothed_scores[valid_range].max())
    cutoff_thresh = float(focus_thresh_frac * peak_score)

    # 3. Find the longest contiguous run of in-focus strips
    in_focus = smoothed_scores >= cutoff_thresh
    best_start, best_end, best_len = None, None, 0
    cur_start = None

    for i, is_foc in enumerate(in_focus):
        if is_foc and cur_start is None:
            cur_start = i
        elif not is_foc and cur_start is not None:
            cur_len = i - cur_start
            if cur_len > best_len:
                best_start, best_end, best_len = cur_start, i - 1, cur_len
            cur_start = None

    if cur_start is not None and (len(in_focus) - cur_start) > best_len:
        best_start, best_end, best_len = cur_start, len(in_focus) - 1, len(in_focus) - cur_start

    if manual_y_range is not None and len(manual_y_range) == 2:
        y0 = max(0, min(H, int(manual_y_range[0])))
        y1 = max(y0 + 10, min(H, int(manual_y_range[1])))
    elif best_start is not None and best_len >= min_consecutive_strips:
        y0 = max(0, int(ys[best_start] - strip_h // 2))
        y1 = min(H, int(ys[best_end] + strip_h // 2))
        if y1 - y0 < min_height_px:
            # Expand slightly if too tight
            y0 = max(0, int(ys[best_start] - strip_h))
            y1 = min(H, int(ys[best_end] + strip_h))
    else:
        # Fallback to standard default ROI
        y0 = min(330, H)
        y1 = min(1150, H)

    roi = (int(x0), int(y0), int(x1), int(y1))
    info = {
        "roi": roi,
        "width": roi[2] - roi[0],
        "height": roi[3] - roi[1],
        "ys": ys.tolist(),
        "raw_scores": raw_scores.tolist(),
        "smoothed_scores": smoothed_scores.tolist(),
        "peak_score": peak_score,
        "cutoff_thresh": cutoff_thresh,
        "walls_raw": (w_left, w_right),
        "walls_inset": (x0, x1),
        "metrics": {k: v.tolist() for k, v in metrics.items()},
        "image_shape": (H, W),
    }
    return roi, info


def crop_measurable_frame(img, roi=None, **kwargs):
    """Crops an image to its measurable froth region.

    Args:
        img: Numpy array or image path.
        roi: Optional pre-computed (x0, y0, x1, y1) tuple. If None, detected automatically.
        **kwargs: Arguments forwarded to detect_measurable_roi if roi is None.

    Returns:
        cropped_img: Cropped image slice.
        roi: Tuple (x0, y0, x1, y1) in original image coordinates.
        info: Diagnostic info dictionary.
    """
    if isinstance(img, str):
        img_arr = np.array(Image.open(img).convert("RGB"))
    else:
        img_arr = img

    info = {}
    if roi is None:
        roi, info = detect_measurable_roi(img_arr, **kwargs)

    x0, y0, x1, y1 = roi
    cropped = img_arr[y0:y1, x0:x1]
    return cropped, roi, info


def visualize_measurable_roi(img, roi, info, output_path):
    """Generates and saves a detailed diagnostic visualization of the measurable crop.

    Panel 1: Full frame with ROI bounding box, outer/inner walls, and exclusion zones.
    Panel 2: Cropped measurable frame.
    Panel 3: Focus & texture score curve down the column height with threshold lines.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if isinstance(img, str):
        img_arr = np.array(Image.open(img).convert("RGB"))
    else:
        img_arr = img

    H, W = img_arr.shape[:2]
    x0, y0, x1, y1 = roi

    fig, axes = plt.subplots(1, 3, figsize=(15, 8), gridspec_kw={"width_ratios": [1.2, 1, 1.2]})

    # Panel 1: Full frame with guides
    overlay = img_arr.copy()
    w_left, w_right = info.get("walls_raw", (x0, x1))

    # Draw shaded unmeasurable top/bottom bands
    fig_ax0 = axes[0]
    fig_ax0.imshow(overlay)
    # Shaded top
    if y0 > 0:
        fig_ax0.axhspan(0, y0, color="red", alpha=0.25, label="Excluded Top Froth")
    # Shaded bottom
    if y1 < H:
        fig_ax0.axhspan(y1, H, color="gray", alpha=0.25, label="Excluded Base / Liquid")

    # ROI Bounding Box
    rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="#00FF66", facecolor="none",
                         linewidth=2.5, label=f"Measurable ROI ({x1-x0}x{y1-y0}px)")
    fig_ax0.add_patch(rect)

    # Walls
    fig_ax0.axvline(w_left, color="#00BFFF", linestyle=":", alpha=0.7, label="Outer Glass Walls")
    fig_ax0.axvline(w_right, color="#00BFFF", linestyle=":", alpha=0.7)

    fig_ax0.set_title("Full Frame & Detected Region", fontsize=12, fontweight="bold")
    fig_ax0.legend(loc="lower left", fontsize=8, framealpha=0.8)
    fig_ax0.set_xlim(0, W)
    fig_ax0.set_ylim(H, 0)
    fig_ax0.axis("off")

    # Panel 2: Cropped measurable frame
    cropped = img_arr[y0:y1, x0:x1]
    axes[1].imshow(cropped)
    axes[1].set_title(f"Cropped Measurable Bed\n[{x1-x0} x {y1-y0} px]", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Vertical focus profile plot
    ys = info.get("ys", [])
    raw_scores = info.get("raw_scores", [])
    smoothed_scores = info.get("smoothed_scores", [])
    cutoff = info.get("cutoff_thresh", 0.0)

    if ys and smoothed_scores:
        ax3 = axes[2]
        ax3.plot(raw_scores, ys, color="lightgray", linestyle="--", label="Raw Focus Index")
        ax3.plot(smoothed_scores, ys, color="#FF5500", linewidth=2.0, label="Smoothed Focus Curve")
        ax3.axvline(cutoff, color="red", linestyle=":", label=f"Cutoff Thresh ({cutoff:.1f})")
        ax3.axhline(y0, color="#00AA44", linestyle="-", linewidth=1.8, label=f"Y0 (Top: {y0}px)")
        ax3.axhline(y1, color="#00AA44", linestyle="-", linewidth=1.8, label=f"Y1 (Bot: {y1}px)")

        ax3.axhspan(y0, y1, color="#00FF66", alpha=0.15)
        ax3.set_ylim(H, 0)
        ax3.set_xlabel("Focus & Texture Score", fontsize=10)
        ax3.set_ylabel("Frame Y Coordinate (px)", fontsize=10)
        ax3.set_title("Vertical Sharpness Profile", fontsize=12, fontweight="bold")
        ax3.legend(loc="lower right", fontsize=8)
        ax3.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to: {output_path}")


def extract_measurable_video_frames(
    video_path,
    out_dir,
    sample_every_s=30,
    save_vis=False,
    **kwargs
):
    """Extracts and crops in-focus measurable frames across a video clip.

    Args:
        video_path: Path to video file (e.g. 180WW.MP4).
        out_dir: Output directory where cropped frames and summaries will be written.
        sample_every_s: Interval in seconds between sampled frames.
        save_vis: If True, saves visual diagnostic plots alongside frames.
        **kwargs: Arguments forwarded to detect_measurable_roi.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_s = frame_count / fps if fps > 0 else 0.0

    print(f"Processing video: {os.path.basename(video_path)}")
    print(f"Duration: {duration_s:.1f}s, FPS: {fps:.2f}, Total Frames: {int(frame_count)}")
    print(f"Sampling every {sample_every_s}s -> approx {int(duration_s // sample_every_s) + 1} frames")

    timestamps = list(range(0, int(duration_s), sample_every_s))
    records = []

    for t_s in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000)
        ret, frame_bgr = cap.read()
        if not ret:
            print(f"Failed to read frame at t={t_s}s")
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        cropped, roi, info = crop_measurable_frame(frame_rgb, **kwargs)

        stem = f"frame_t{t_s:04d}s"
        crop_path = os.path.join(out_dir, f"{stem}_crop.png")
        Image.fromarray(cropped).save(crop_path)

        if save_vis:
            vis_path = os.path.join(out_dir, f"{stem}_vis.png")
            visualize_measurable_roi(frame_rgb, roi, info, vis_path)

        rec = {
            "time_s": t_s,
            "crop_path": crop_path,
            "roi_x0": roi[0],
            "roi_y0": roi[1],
            "roi_x1": roi[2],
            "roi_y1": roi[3],
            "crop_width": roi[2] - roi[0],
            "crop_height": roi[3] - roi[1],
            "peak_focus_score": info.get("peak_score", 0.0),
        }
        records.append(rec)
        print(f"  t={t_s:4d}s -> Measurable ROI: X=[{roi[0]}, {roi[2]}], Y=[{roi[1]}, {roi[3]}] ({rec['crop_width']}x{rec['crop_height']} px)")

    cap.release()

    meta_path = os.path.join(out_dir, "measurable_frames_summary.json")
    with open(meta_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved video extraction summary to: {meta_path}")
    return records


def extract_frame_from_video(video_path, time_s=0.0):
    """Extracts an RGB frame from a video at a specific timestamp (in seconds)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, float(time_s) * 1000.0)
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not read frame at t={time_s}s from {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser(
        description="crop_measurable.py - Detect and crop measurable in-focus froth region"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to input image frame")
    group.add_argument("--video", help="Path to input video file")

    parser.add_argument("--timestamp", type=float, default=None,
                        help="Specific timestamp in seconds when using --video")
    parser.add_argument("--out-dir", default="results", help="Directory to save output files")
    parser.add_argument("--sample-every", type=int, default=None,
                        help="Interval in seconds for video sequence sampling")
    parser.add_argument("--save-crop", action="store_true", help="Save the cropped measurable frame")
    parser.add_argument("--save-vis", action="store_true", help="Save the diagnostic visual inspection plot")
    parser.add_argument("--save-json", action="store_true", help="Save detection metadata JSON")
    parser.add_argument("--focus-thresh", type=float, default=0.58,
                        help="Fraction of peak focus score for threshold (default 0.58)")
    parser.add_argument("--manual-y", type=int, nargs=2, default=None,
                        help="Manual Y0 and Y1 crop bounds, e.g. --manual-y 630 1040")
    parser.add_argument("--metric", choices=["combined", "laplacian", "sobel", "canny"], default="combined",
                        help="Focus / texture metric")
    parser.add_argument("--wall-inset", type=int, default=8, help="Wall inset pixels")
    parser.add_argument("--prefix", default="measurable", help="Output filename prefix")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.video and args.sample_every is not None:
        extract_measurable_video_frames(
            args.video,
            args.out_dir,
            sample_every_s=args.sample_every,
            save_vis=args.save_vis,
            focus_metric=args.metric,
            focus_thresh_frac=args.focus_thresh,
            wall_inset_px=args.wall_inset
        )
    else:
        if args.video:
            t_s = args.timestamp if args.timestamp is not None else 0.0
            print(f"Extracting frame from video at t={t_s}s: {args.video}")
            img_arr = extract_frame_from_video(args.video, time_s=t_s)
            source_name = f"{os.path.basename(args.video)}_t{t_s:04.1f}s"
        else:
            img_arr = np.array(Image.open(args.image).convert("RGB"))
            source_name = os.path.basename(args.image)

        roi, info = detect_measurable_roi(
            img_arr,
            focus_metric=args.metric,
            focus_thresh_frac=args.focus_thresh,
            wall_inset_px=args.wall_inset,
            manual_y_range=args.manual_y
        )
        x0, y0, x1, y1 = roi
        w, h = x1 - x0, y1 - y0
        print(f"\n[crop_measurable] Results for {source_name}:")
        print(f"  Input frame size : {img_arr.shape[1]} x {img_arr.shape[0]} px")
        print(f"  Measurable ROI   : X=[{x0}, {x1}] (w={w} px), Y=[{y0}, {y1}] (h={h} px)")
        print(f"  Area retained    : {w*h / (img_arr.shape[0]*img_arr.shape[1])*100:.1f}% of frame")
        print(f"  Peak focus score : {info['peak_score']:.1f} (cutoff = {info['cutoff_thresh']:.1f})")

        stem = os.path.join(args.out_dir, f"{args.prefix}_roi")
        if args.save_crop:
            cropped = img_arr[y0:y1, x0:x1]
            crop_path = f"{stem}_crop.png"
            Image.fromarray(cropped).save(crop_path)
            print(f"  Cropped frame    : {crop_path}")

        if args.save_vis:
            vis_path = f"{stem}_vis.png"
            visualize_measurable_roi(img_arr, roi, info, vis_path)

        if args.save_json:
            json_path = f"{stem}_meta.json"
            with open(json_path, "w") as f:
                json.dump(info, f, indent=2)
            print(f"  Metadata JSON    : {json_path}")


if __name__ == "__main__":
    main()
