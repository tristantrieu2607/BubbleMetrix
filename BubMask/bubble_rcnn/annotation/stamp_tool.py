"""Interactive Multi-Regime Stamping Tool with Zoom, Pan, and Resizable High-DPI Display."""

import os
import cv2
import json
import math
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from bubble_rcnn.annotation import schema

class StampAnnotationTool:
    """
    Advanced Multi-Regime Annotation GUI:
    - High-Resolution Display: Defaults to 960x960 resizable window (nearly 2x larger on screen).
    - Zoom & Pan Engine:
      * Mouse Scroll Wheel: Zoom in / zoom out centered at mouse cursor.
      * Hotkeys '+' / '=' (zoom in), '-' (zoom out), 'R' / '0' (reset zoom to 1.0x).
      * Middle-click drag OR Right-click drag: Pan smoothly across zoomed image.
      * Arrow keys: Pan Up / Down / Left / Right.
    - Stamping Modes:
      * 'M': Toggle between Circle Mode (center + radius) and Ellipse Mode (bounding axes).
      * 'V': Toggle between Raw camera pixels and CLAHE-enhanced high-contrast view.
      * 'F': Filter crops between ALL, FROTH, ALL PULP, ROTOR RAW, ROTOR DILUTED, VENTURI AIR.
      * 'Z': Undo last stamped bubble.
    - Mouse Actions:
      * Left-drag: Stamp bubble instance (at sub-pixel image coordinates).
      * Right-click (quick tap): Delete nearest bubble.
    - Navigation:
      * 'S': Save annotations.
      * Space / 'N': Save and next crop.
      * 'P': Previous crop.
      * 'Q' / Esc: Save and quit.
    """
    def __init__(self, crops_dir: str, labels_dir: str, display_size: int = 960):
        self.crops_dir = crops_dir
        self.labels_dir = labels_dir
        self.display_size = display_size
        os.makedirs(labels_dir, exist_ok=True)

        self.all_crop_files = []
        for root, _, files in os.walk(crops_dir):
            for f in files:
                if f.endswith(".png") and not f.endswith("_view.png") and not f.endswith("_mask.png"):
                    rel = os.path.relpath(os.path.join(root, f), crops_dir).replace("\\", "/")
                    self.all_crop_files.append(rel)
        self.all_crop_files.sort()

        if not self.all_crop_files:
            raise RuntimeError(f"No raw PNG crops found in {crops_dir}")

        self.sidebar_width = 340
        self.filter_mode = "ALL"
        self._apply_filter()

        self.stamp_mode = "circle"
        self.show_enhanced = False
        self.bubbles = []

        # Pass 1 stamps bubbles; pass 2 marks occluded bubbles as ignore regions
        # (guide S4).  They are different objects with different loss semantics,
        # so they are different modes, not one list with a flag.
        self.pass_mode = "bubble"          # "bubble" | "ignore"
        self.ignore_regions: List[Dict[str, Any]] = []
        self.confirmed_empty = False
        self._warn_text = ""
        self._warn_until = 0.0

        # Zoom & Pan states
        self.zoom = 1.0  # 1.0 to 6.0
        self.center_view = [256.0, 256.0]  # center of view in image coordinates

        # Mouse interaction states
        self.drawing = False
        self.p_start_win = (0, 0)
        self.p_current_win = (0, 0)

        self.panning = False
        self.pan_start_win = (0, 0)
        self.pan_start_center = [256.0, 256.0]
        self.right_down_pos = (0, 0)

        # Action history stacks for full Undo and Redo
        self.undo_stack: List[Dict[str, Any]] = []
        self.redo_stack: List[Dict[str, Any]] = []

        self.window_name = "BubMask - Annotation Tool (Zoom & Pan Enabled)"
        self.current_idx = self._first_unlabelled()

    def _first_unlabelled(self) -> int:
        for idx, cf in enumerate(self.crop_files):
            stem = os.path.splitext(os.path.basename(cf))[0]
            sub_dir = os.path.dirname(cf)
            label_file = os.path.join(self.labels_dir, sub_dir, f"{stem}_labels.json")
            if not os.path.exists(label_file):
                label_file = os.path.join(self.labels_dir, f"{stem}_labels.json")
            if not os.path.exists(label_file):
                return idx
            # A crop is done once a call has been made on it -- including the call
            # "this crop is empty".  v1 treated a saved zero-bubble file as
            # unannotated, which made confirmed_empty unreachable and re-offered
            # the crop for ever.
            try:
                with open(label_file, "r", encoding="utf-8") as fp:
                    d = json.load(fp)
                    if not schema.is_annotated(d):
                        return idx
            except Exception:
                return idx
        return 0

    def _apply_filter(self):
        fm = self.filter_mode
        if fm == "FROTH":
            self.crop_files = [f for f in self.all_crop_files if "180ww" in f.lower() or "froth" in f.lower()]
        elif fm == "ROTOR RAW":
            self.crop_files = [f for f in self.all_crop_files if ("rotcoarse" in f.lower() or "rotfine" in f.lower() or "rotmed" in f.lower() or "rotor_raw" in f.lower()) and "dilute" not in f.lower()]
        elif fm == "ROTOR DILUTED":
            self.crop_files = [f for f in self.all_crop_files if "dilute" in f.lower()]
        elif fm == "VENTURI AIR":
            self.crop_files = [f for f in self.all_crop_files if "venturi" in f.lower()]
        elif fm == "ALL PULP":
            self.crop_files = [f for f in self.all_crop_files if "180ww" not in f.lower() and "froth" not in f.lower()]
        else:
            self.crop_files = self.all_crop_files

        if not self.crop_files:
            self.crop_files = self.all_crop_files
            self.filter_mode = "ALL"

    def _get_regime_name(self, filename: str) -> str:
        """Regime label, with the size range measured from the labels on disk.

        v1 announced "Rotor Diluted (30-70px)" over a regime whose stamped bubbles
        run 87-187 px, and "Venturi (10-35px)" over one where most candidates sit
        at or under the 12 px floor.  The annotator reads this line while deciding
        what to stamp, so a wrong range here biases the dataset.
        """
        fn = filename.lower()
        if "180ww" in fn or "froth" in fn:
            return "FROTH facets (stamped 17-55 px, median 24) -- 'M' for ellipse"
        elif "venturi" in fn:
            return "PULP: Venturi (12-30 px, median 15 -- MOST ARE AT THE FLOOR)"
        elif "dilute" in fn:
            return "PULP: Rotor Diluted (87-187 px, median 125 -- large and sparse)"
        else:
            return "PULP: Rotor Raw (46-98 px, median 73 -- dense, use 'I' for occluded)"

    def _get_viewport_bounds(self, h_img: int, w_img: int) -> Tuple[float, float, float, float]:
        """Calculates visible viewport bounding box in image coordinates."""
        w_vis = w_img / self.zoom
        h_vis = h_img / self.zoom

        x0 = max(0.0, min(self.center_view[0] - w_vis / 2.0, w_img - w_vis))
        y0 = max(0.0, min(self.center_view[1] - h_vis / 2.0, h_img - h_vis))
        
        # update center to clamped position
        self.center_view = [x0 + w_vis / 2.0, y0 + h_vis / 2.0]
        return x0, y0, w_vis, h_vis

    def _win_to_img(self, x_win: float, y_win: float, h_img: int, w_img: int) -> Tuple[float, float]:
        """Converts window pixel coordinates to base image pixel coordinates."""
        x0, y0, w_vis, h_vis = self._get_viewport_bounds(h_img, w_img)
        x_img = x0 + (x_win / float(self.display_size)) * w_vis
        y_img = y0 + (y_win / float(self.display_size)) * h_vis
        return max(0.0, min(x_img, float(w_img))), max(0.0, min(y_img, float(h_img)))

    def _img_to_win(self, x_img: float, y_img: float, h_img: int, w_img: int) -> Tuple[float, float]:
        """Converts base image pixel coordinates to window pixel coordinates."""
        x0, y0, w_vis, h_vis = self._get_viewport_bounds(h_img, w_img)
        x_win = ((x_img - x0) / w_vis) * float(self.display_size)
        y_win = ((y_img - y0) / h_vis) * float(self.display_size)
        return x_win, y_win

    def _load_crop(self, idx: int):
        self.current_idx = max(0, min(idx, len(self.crop_files) - 1))
        crop_rel = self.crop_files[self.current_idx]
        crop_path = os.path.join(self.crops_dir, crop_rel)
        crop_dir = os.path.dirname(crop_path)
        crop_base = os.path.basename(crop_rel)
        
        self.img_raw = cv2.imread(crop_path)
        if self.img_raw is None:
            raise FileNotFoundError(f"Cannot read: {crop_path}")
        # open centred on the image (was fixed at 256,256 -- top-left quadrant of a 1024 frame)
        _h0, _w0 = self.img_raw.shape[:2]
        self.center_view = [_w0 / 2.0, _h0 / 2.0]

        # Check for enhanced view
        view_name = f"{os.path.splitext(crop_base)[0]}_view.png"
        view_path = os.path.join(crop_dir, view_name)
        if os.path.exists(view_path):
            self.img_view = cv2.imread(view_path)
        else:
            gray = cv2.cvtColor(self.img_raw, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enh = clahe.apply(gray)
            self.img_view = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR)

        # Smart default mode:
        if "froth" in self._get_regime_name(crop_base).lower():
            self.stamp_mode = "ellipse"
        else:
            self.stamp_mode = "circle"

        # Load existing labels
        label_stem = os.path.splitext(crop_base)[0]
        sub_dir = os.path.dirname(crop_rel)
        target_labels_dir = os.path.join(self.labels_dir, sub_dir)
        label_path = os.path.join(target_labels_dir, f"{label_stem}_labels.json")
        if not os.path.exists(label_path):
            label_path = os.path.join(self.labels_dir, f"{label_stem}_labels.json")

        self.bubbles = []
        self.ignore_regions = []
        self.confirmed_empty = False
        self.pass_mode = "bubble"
        self.undo_stack.clear()
        self.redo_stack.clear()
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.bubbles = data.get("bubbles", [])
                self.ignore_regions = data.get("ignore_regions", [])
                self.confirmed_empty = bool(data.get("confirmed_empty", False))

    def _save_labels(self):
        crop_rel = self.crop_files[self.current_idx]
        crop_base = os.path.basename(crop_rel)
        label_stem = os.path.splitext(crop_base)[0]
        sub_dir = os.path.dirname(crop_rel)
        target_labels_dir = os.path.join(self.labels_dir, sub_dir)
        os.makedirs(target_labels_dir, exist_ok=True)

        label_path = os.path.join(target_labels_dir, f"{label_stem}_labels.json")
        mask_path = os.path.join(target_labels_dir, f"{label_stem}_mask.png")

        regime = self._get_regime_name(crop_base)
        h, w = self.img_raw.shape[:2]

        # Re-derive the flags on save so a bubble that was moved, or loaded from a
        # v1 file, cannot keep a stale at_limit / touches_edge.
        fov = schema.fov_for_image(float(w), float(h))
        for b in self.bubbles:
            schema.apply_flags(b, float(w), float(h), fov)

        data = schema.build_document(
            image_file=crop_base,
            relative_path=crop_rel,
            regime=regime,
            width=w,
            height=h,
            bubbles=self.bubbles,
            ignore_regions=self.ignore_regions,
            confirmed_empty=self.confirmed_empty,
            floor_px=schema.floor_for_regime(regime),
        )
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        cv2.imwrite(mask_path, schema.rasterise_instances(self.bubbles, h, w))
        ign = schema.rasterise_ignore(self.ignore_regions, h, w)
        ignore_path = label_path.replace("_labels.json", "_ignore.png")
        if self.ignore_regions:
            cv2.imwrite(ignore_path, (ign * 255).astype(np.uint8))
        elif os.path.exists(ignore_path):
            # no regions left: blank it rather than leave a stale mask behind
            cv2.imwrite(ignore_path, np.zeros((h, w), dtype=np.uint8))

        flag_n = sum(1 for b in self.bubbles if b.get("at_limit") or b.get("touches_edge") or b.get("unsure"))
        print(f"[Saved] {len(self.bubbles)} bubbles ({flag_n} flagged), {len(self.ignore_regions)} ignore regions, "
              f"confirmed_empty={self.confirmed_empty} for {crop_base} ({regime}) -> {label_path}")

    def _find_bubble_at(self, img_x: float, img_y: float) -> Optional[int]:
        """Finds index of bubble under the given image coordinates, checking inside disc then proximity."""
        if not self.bubbles:
            return None

        target_idx = None
        min_d = float("inf")

        # 1. Direct hit on circle disc or ellipse interior (+ tolerance for rim click)
        for i, b in enumerate(self.bubbles):
            r = b.get("r", 15.0)
            st = b.get("shape_type", "circle")
            if st == "ellipse":
                ax_w = max(1.0, b.get("axis_w", r * 2.0) / 2.0)
                ax_h = max(1.0, b.get("axis_h", r * 2.0) / 2.0)
                dx = img_x - b["cx"]
                dy = img_y - b["cy"]
                norm_d = (dx / ax_w) ** 2 + (dy / ax_h) ** 2
                if norm_d <= 1.44:  # Within 1.2x radius boundary
                    if norm_d < min_d:
                        min_d = norm_d
                        target_idx = i
            else:
                d_center = math.hypot(b["cx"] - img_x, b["cy"] - img_y)
                tol = 15.0 / self.zoom + 10.0
                if d_center <= (r + tol):
                    if d_center < min_d:
                        min_d = d_center
                        target_idx = i

        # 2. Proximity fallback: nearest bubble center within generous reach
        if target_idx is None:
            dists = [math.hypot(b["cx"] - img_x, b["cy"] - img_y) for b in self.bubbles]
            best_i = int(np.argmin(dists))
            r_best = self.bubbles[best_i].get("r", 15.0)
            reach = r_best + (35.0 / self.zoom) + 12.0
            if dists[best_i] <= reach:
                target_idx = best_i

        return target_idx

    def _delete_bubble_at(self, img_x: float, img_y: float) -> bool:
        """Deletes the bubble at the given image coordinates if found."""
        target_idx = self._find_bubble_at(img_x, img_y)
        if target_idx is not None and 0 <= target_idx < len(self.bubbles):
            removed = self.bubbles.pop(target_idx)
            self.undo_stack.append({'action': 'delete', 'bubble': removed, 'index': target_idx})
            self.redo_stack.clear()
            print(f"[Deleted] Bubble #{target_idx + 1} at ({int(round(removed['cx']))}, {int(round(removed['cy']))}) | Remaining: {len(self.bubbles)}")
            return True
        return False

    def undo(self) -> bool:
        """Undoes the most recent action (restores deleted bubbles or removes stamped bubbles)."""
        if self.undo_stack and self.undo_stack[-1]['action'] in ('add_ignore', 'delete_ignore'):
            act = self.undo_stack.pop()
            if act['action'] == 'add_ignore':
                if self.ignore_regions:
                    self.ignore_regions.pop()
            else:
                self.ignore_regions.insert(act['index'], act['region'])
            self.redo_stack.append(act)
            return True
        if not self.undo_stack:
            # Fallback if no history recorded yet (e.g. pre-existing bubbles on disk): pop last bubble
            if self.bubbles:
                popped = self.bubbles.pop()
                self.redo_stack.append({'action': 'add', 'bubble': popped})
                print(f"[Undo] Removed last bubble at ({int(round(popped['cx']))}, {int(round(popped['cy']))}) | Remaining: {len(self.bubbles)}")
                return True
            print("[Undo] Nothing to undo.")
            return False

        cmd = self.undo_stack.pop()
        if cmd['action'] == 'add':
            bubble = cmd['bubble']
            if bubble in self.bubbles:
                idx = self.bubbles.index(bubble)
                self.bubbles.remove(bubble)
                self.redo_stack.append({'action': 'add', 'bubble': bubble, 'index': idx})
                print(f"[Undo] Removed added bubble at ({int(round(bubble['cx']))}, {int(round(bubble['cy']))}) | Remaining: {len(self.bubbles)}")
                return True
            elif self.bubbles:
                popped = self.bubbles.pop()
                self.redo_stack.append({'action': 'add', 'bubble': popped})
                print(f"[Undo] Removed last bubble | Remaining: {len(self.bubbles)}")
                return True
        elif cmd['action'] == 'delete':
            bubble = cmd['bubble']
            idx = max(0, min(cmd.get('index', len(self.bubbles)), len(self.bubbles)))
            self.bubbles.insert(idx, bubble)
            self.redo_stack.append({'action': 'delete', 'bubble': bubble, 'index': idx})
            print(f"[Undo] Restored deleted bubble at ({int(round(bubble['cx']))}, {int(round(bubble['cy']))}) | Remaining: {len(self.bubbles)}")
            return True
        return False

    def redo(self) -> bool:
        """Redoes the most recently undone action."""
        if not self.redo_stack:
            print("[Redo] Nothing to redo.")
            return False

        cmd = self.redo_stack.pop()
        if cmd['action'] == 'add':
            bubble = cmd['bubble']
            idx = max(0, min(cmd.get('index', len(self.bubbles)), len(self.bubbles)))
            self.bubbles.insert(idx, bubble)
            self.undo_stack.append({'action': 'add', 'bubble': bubble})
            print(f"[Redo] Re-added bubble at ({int(round(bubble['cx']))}, {int(round(bubble['cy']))}) | Remaining: {len(self.bubbles)}")
            return True
        elif cmd['action'] == 'delete':
            bubble = cmd['bubble']
            if bubble in self.bubbles:
                idx = self.bubbles.index(bubble)
                self.bubbles.remove(bubble)
                self.undo_stack.append({'action': 'delete', 'bubble': bubble, 'index': idx})
                print(f"[Redo] Re-deleted bubble at ({int(round(bubble['cx']))}, {int(round(bubble['cy']))}) | Remaining: {len(self.bubbles)}")
                return True
        return False

    def _mouse_callback(self, event, x, y, flags, param):
        if x >= self.display_size:
            return
        h_img, w_img = self.img_raw.shape[:2]
        self.p_current_win = (x, y)

        # 1. Mouse Scroll Wheel -> Zoom In / Zoom Out centered at cursor
        if event == cv2.EVENT_MOUSEWHEEL:
            cursor_img_x, cursor_img_y = self._win_to_img(x, y, h_img, w_img)
            old_zoom = self.zoom

            # Detect wheel direction
            zoom_in = (flags > 0)
            if zoom_in:
                self.zoom = min(6.0, self.zoom * 1.25)
            else:
                self.zoom = max(1.0, self.zoom / 1.25)

            if abs(self.zoom - old_zoom) > 1e-4:
                # Adjust center view so cursor position remains stable under mouse
                w_vis = w_img / self.zoom
                h_vis = h_img / self.zoom
                new_x0 = cursor_img_x - (x / float(self.display_size)) * w_vis
                new_y0 = cursor_img_y - (y / float(self.display_size)) * h_vis
                new_x0 = max(0.0, min(new_x0, w_img - w_vis))
                new_y0 = max(0.0, min(new_y0, h_img - h_vis))
                self.center_view = [new_x0 + w_vis / 2.0, new_y0 + h_vis / 2.0]
            return

        # 2. Middle Button Drag OR Right Button Drag -> Pan Viewport
        if event == cv2.EVENT_MBUTTONDOWN:
            self.panning = True
            self.pan_start_win = (x, y)
            self.pan_start_center = list(self.center_view)
            return
        elif event == cv2.EVENT_MBUTTONUP:
            self.panning = False
            return

        if event == cv2.EVENT_RBUTTONDOWN:
            self.right_down_pos = (x, y)
            self.pan_start_win = (x, y)
            self.pan_start_center = list(self.center_view)
            return
        elif event == cv2.EVENT_RBUTTONUP:
            dist_moved = math.hypot(x - self.right_down_pos[0], y - self.right_down_pos[1])
            # Generous tolerance (25px) so trackpad click movement isn't mistaken for pan
            if dist_moved <= 25.0:
                img_x, img_y = self._win_to_img(x, y, h_img, w_img)
                if self.pass_mode == "ignore":
                    self._delete_ignore_at(img_x, img_y)
                else:
                    self._delete_bubble_at(img_x, img_y)
            self.panning = False
            return

        # 3. Mouse Movement
        if event == cv2.EVENT_MOUSEMOVE:
            # Check if panning with middle button or holding right drag with significant movement (> 15px)
            if self.panning or (flags & cv2.EVENT_FLAG_RBUTTON and math.hypot(x - self.right_down_pos[0], y - self.right_down_pos[1]) > 15.0):
                w_vis = w_img / self.zoom
                h_vis = h_img / self.zoom
                dx_win = x - self.pan_start_win[0]
                dy_win = y - self.pan_start_win[1]
                dx_img = dx_win * (w_vis / float(self.display_size))
                dy_img = dy_win * (h_vis / float(self.display_size))

                new_cx = self.pan_start_center[0] - dx_img
                new_cy = self.pan_start_center[1] - dy_img
                x0 = max(0.0, min(new_cx - w_vis / 2.0, w_img - w_vis))
                y0 = max(0.0, min(new_cy - h_vis / 2.0, h_img - h_vis))
                self.center_view = [x0 + w_vis / 2.0, y0 + h_vis / 2.0]
                return
            return

        # 4. Left Button -> Stamp Bubble
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.p_start_win = (x, y)
            self.p_current_win = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                x0_img, y0_img = self._win_to_img(self.p_start_win[0], self.p_start_win[1], h_img, w_img)
                x1_img, y1_img = self._win_to_img(x, y, h_img, w_img)

                if self.pass_mode == "ignore":
                    # Pass 2: a rough box over an occluded bubble (guide S4).  Rough
                    # on purpose -- tracing a hidden edge means inventing it.
                    if abs(x1_img - x0_img) >= 4.0 and abs(y1_img - y0_img) >= 4.0:
                        reg = schema.make_ignore_region(x0_img, y0_img, x1_img, y1_img)
                        self.ignore_regions.append(reg)
                        self.undo_stack.append({'action': 'add_ignore', 'region': reg})
                        self.redo_stack.clear()
                    return

                view = "clahe" if self.show_enhanced else "raw"
                if self.stamp_mode == "circle":
                    radius = math.hypot(x1_img - x0_img, y1_img - y0_img)
                    if radius >= 1.0:
                        b = schema.make_bubble("circle", x0_img, y0_img, r=radius,
                                               view=view, width=w_img, height=h_img)
                        self._accept_stamp(b)
                else:  # Ellipse bounding mode
                    cx = (x0_img + x1_img) / 2.0
                    cy = (y0_img + y1_img) / 2.0
                    axis_w = abs(x1_img - x0_img)
                    axis_h = abs(y1_img - y0_img)
                    if axis_w >= 2.0 and axis_h >= 2.0:
                        b = schema.make_bubble("ellipse", cx, cy, axis_w=axis_w, axis_h=axis_h,
                                               view=view, width=w_img, height=h_img)
                        self._accept_stamp(b)

    def _warn(self, text: str, seconds: float = 2.5):
        self._warn_text = text
        self._warn_until = time.time() + seconds
        print(f"[Refused] {text}")

    def _accept_stamp(self, b: Dict[str, Any]) -> bool:
        """Apply the 12 px floor (guide S3) before a stamp is allowed in.

        The floor is set by the debris: window specks measure 3.2 px median and
        10.5 px at the largest, so anything under 12 px cannot be told apart from
        one.  Refusing at stamp time is the only place this is cheap -- afterwards
        it is a relabelling job.
        """
        d = schema.equiv_diameter_px(b)
        crop_base = os.path.basename(self.crop_files[self.current_idx]) if self.crop_files else ""
        floor = schema.floor_for_regime(self._get_regime_name(crop_base))
        twitch = 6.0 if floor > 0 else 2.0       # froth: small bubbles are real
        if d < twitch:
            # A click with no drag, or a 1-2 px twitch.  Not a judgement about the
            # image, so it gets no judgement-shaped message.
            print(f"[Ignored] {d:.1f} px -- click with no drag; hold and drag from the "
                  f"centre to the rim to stamp")
            return False
        if d < floor:
            self._warn(f"{d:.1f} px is below the {floor:.0f} px floor "
                       f"({d * 3.291:.0f} um) -- at this size a bubble cannot be told "
                       f"apart from window debris")
            return False
        self.bubbles.append(b)
        self.undo_stack.append({'action': 'add', 'bubble': b})
        self.redo_stack.clear()
        if b.get("at_limit"):
            print(f"[at_limit] {d:.1f} px -- kept and flagged; the 12-20 px band can be dropped later")
        return True

    def _find_ignore_at(self, img_x: float, img_y: float) -> Optional[int]:
        for i in range(len(self.ignore_regions) - 1, -1, -1):
            r = self.ignore_regions[i]
            if r["x1"] <= img_x <= r["x2"] and r["y1"] <= img_y <= r["y2"]:
                return i
        return None

    def _delete_ignore_at(self, img_x: float, img_y: float) -> bool:
        i = self._find_ignore_at(img_x, img_y)
        if i is None:
            return False
        removed = self.ignore_regions.pop(i)
        self.undo_stack.append({'action': 'delete_ignore', 'region': removed, 'index': i})
        self.redo_stack.clear()
        print(f"[Deleted] ignore region #{i + 1} | Remaining: {len(self.ignore_regions)}")
        return True

    def _toggle_unsure_at(self, img_x: float, img_y: float) -> bool:
        """Guide S6: forcing a binary on a borderline object puts noise into training."""
        i = self._find_bubble_at(img_x, img_y)
        if i is None:
            return False
        self.bubbles[i]["unsure"] = not self.bubbles[i].get("unsure", False)
        print(f"[unsure] bubble #{i + 1} -> {self.bubbles[i]['unsure']}")
        return True

    def _render_sidebar(self, h: int, w: int) -> np.ndarray:
        sb = np.full((h, w, 3), (22, 24, 28), dtype=np.uint8)
        cv2.line(sb, (0, 0), (0, h), (60, 65, 75), 2)
        
        y = 28
        def draw_header(title, col=(0, 240, 255)):
            nonlocal y
            cv2.rectangle(sb, (10, y - 18), (w - 10, y + 6), (35, 38, 48), -1)
            cv2.putText(sb, title, (16, y), cv2.FONT_HERSHEY_DUPLEX, 0.46, col, 1)
            y += 24

        def draw_line(text, col=(220, 220, 220), font_scale=0.36, bold=False):
            nonlocal y
            thick = 2 if bold else 1
            cv2.putText(sb, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, thick)
            y += 18

        # These lines are the confirmed guide (claude/pulp-annotation-guide.md v1.2),
        # not a paraphrase of it.  The v1 sidebar told the annotator to extrapolate
        # occluded bubbles and to skip border bubbles -- both the opposite of S4 and
        # S7, and both of them inflate D32.
        draw_header("WHAT TO STAMP  (guide S2, S3)", (0, 255, 255))
        y += 4
        draw_line("[+] STAMP -- in focus AND measurable:", (0, 255, 120), bold=True)
        draw_line(" * sharp rim, distinct dark annulus", (210, 255, 210))
        draw_line(" * bright specular core", (210, 255, 210))
        draw_line(" * full outline visible", (210, 255, 210))
        draw_line(" * d_eq >= 12 px (40 um)", (210, 255, 210))
        draw_line("   below 12 px is REFUSED: that is debris", (140, 240, 255))
        y += 6
        draw_line("[-] BACKGROUND -- do not stamp:", (80, 80, 255), bold=True)
        draw_line(" * soft edge fading over 10+ px", (180, 180, 255))
        draw_line(" * no rim, no core, washed-out grey", (180, 180, 255))
        draw_line("   (defocus images it LARGER than it is)", (140, 200, 255))
        draw_line(" * stationary blobs / dark specks", (180, 180, 255))
        y += 6
        draw_line("[I] IGNORE -- press 'I', drag a box:", (0, 200, 255), bold=True)
        draw_line(" * outline part-hidden behind another", (200, 235, 255))
        draw_line("   DO NOT guess the hidden edge (S4)", (140, 240, 255))
        draw_line("   and DO NOT leave it unlabelled --", (140, 240, 255))
        draw_line("   that teaches 'crowded = empty water'", (140, 240, 255))
        y += 6
        draw_line("Edge bubbles: STAMP them (S7).", (255, 220, 120), bold=True)
        draw_line(" auto-flagged touches_edge; they train,", (255, 235, 180))
        draw_line(" they are excluded from the PSD.", (255, 235, 180))
        y += 6
        draw_line("Can't decide? hover + 'T' = unsure (S6).", (255, 160, 255))
        draw_line("Empty crop? press 'E' to confirm it.", (255, 160, 255))

        y += 10
        draw_header("RING COLOURS", (200, 255, 200))
        y += 4
        draw_line(" green   : clean measurable bubble", (0, 255, 0))
        draw_line(" amber   : at_limit (12-20 px band)", (0, 220, 255))
        draw_line(" cyan    : touches_edge", (255, 200, 0))
        draw_line(" magenta : unsure", (255, 0, 255))
        draw_line(" orange  : hovered -> [DEL]", (0, 140, 255))

        y += 10
        draw_header("REGIME HINTS", (255, 200, 80))
        y += 4
        draw_line(" * Judge focus on RAW ('V' toggles).", (255, 235, 180))
        draw_line("   CLAHE makes defocus look sharp --", (100, 220, 255))
        draw_line("   use it to FIND, never to DECIDE.", (100, 220, 255))
        draw_line(" * Venturi: over-exposed, not small.", (255, 235, 180))
        draw_line(" * Rotor raw: dense; stamp the sharp", (255, 235, 180))
        draw_line("   layer, box the rest as ignore.", (255, 235, 180))
        draw_line(" * Froth: 'M' for ellipse mode.", (255, 235, 180))

        y += 10
        draw_header("CONTROLS", (200, 160, 255))
        y += 4
        draw_line(" Left-Drag : stamp / drag ignore box", (240, 240, 240))
        draw_line(" Right-Tap : delete under cursor", (240, 240, 240))
        draw_line(" 'D' / 'X' : delete under cursor", (240, 240, 240))
        draw_line(" 'I'       : bubble <-> ignore pass", (0, 200, 255), bold=True)
        draw_line(" 'T'       : flag hovered bubble unsure", (255, 160, 255))
        draw_line(" 'E'       : confirm crop is empty", (255, 160, 255))
        draw_line(" Ctrl+Z / Z: undo    Ctrl+Y / Y: redo", (0, 255, 255))
        draw_line(" Scroll    : zoom at cursor (1x - 6x)", (240, 240, 240))
        draw_line(" Mid-Drag / Arrows : pan", (240, 240, 240))
        draw_line(" 'V' raw/CLAHE   'M' circle/ellipse", (240, 240, 240))
        draw_line(" 'F' regime filter   'R'/'0' reset zoom", (240, 240, 240))
        draw_line(" Space / N : save & next", (0, 255, 255), bold=True)
        draw_line(" 'P' prev   'S' save   'Q'/Esc quit", (240, 240, 240))

        return sb

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_size + self.sidebar_width, self.display_size)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        self._load_crop(self.current_idx)

        while True:
            base = self.img_view if self.show_enhanced else self.img_raw
            h_img, w_img = base.shape[:2]
            crop_rel = self.crop_files[self.current_idx]
            crop_base = os.path.basename(crop_rel)
            regime = self._get_regime_name(crop_base)

            # 1. Extract visible viewport slice & scale to display size
            x0, y0, w_vis, h_vis = self._get_viewport_bounds(h_img, w_img)
            x_int = max(0, min(int(round(x0)), w_img - 1))
            y_int = max(0, min(int(round(y0)), h_img - 1))
            w_int = max(1, min(int(round(w_vis)), w_img - x_int))
            h_int = max(1, min(int(round(h_vis)), h_img - y_int))

            viewport = base[y_int : y_int + h_int, x_int : x_int + w_int]
            display = cv2.resize(viewport, (self.display_size, self.display_size), interpolation=cv2.INTER_LINEAR)

            scale_factor = float(self.display_size) / w_vis

            # 2. Draw confirmed bubbles transformed to display space
            hover_idx = None
            if not self.drawing:
                img_cur_x, img_cur_y = self._win_to_img(self.p_current_win[0], self.p_current_win[1], h_img, w_img)
                if 0 <= self.p_current_win[0] < self.display_size and 0 <= self.p_current_win[1] < self.display_size:
                    hover_idx = self._find_bubble_at(img_cur_x, img_cur_y)

            # Ignore regions first, so bubbles stamped inside one stay readable.
            if self.ignore_regions:
                shade = display.copy()
                for reg in self.ignore_regions:
                    ax, ay = self._img_to_win(reg["x1"], reg["y1"], h_img, w_img)
                    bx, by = self._img_to_win(reg["x2"], reg["y2"], h_img, w_img)
                    cv2.rectangle(shade, (int(round(ax)), int(round(ay))),
                                  (int(round(bx)), int(round(by))), (0, 140, 220), -1)
                cv2.addWeighted(shade, 0.28, display, 0.72, 0, display)
                for reg in self.ignore_regions:
                    ax, ay = self._img_to_win(reg["x1"], reg["y1"], h_img, w_img)
                    bx, by = self._img_to_win(reg["x2"], reg["y2"], h_img, w_img)
                    cv2.rectangle(display, (int(round(ax)), int(round(ay))),
                                  (int(round(bx)), int(round(by))), (0, 170, 255), 2)
                    cv2.putText(display, "IGNORE", (int(round(ax)) + 4, int(round(ay)) + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1)

            for idx, b in enumerate(self.bubbles, start=1):
                st = b.get("shape_type", "circle")
                cx_w, cy_w = self._img_to_win(b["cx"], b["cy"], h_img, w_img)

                # Skip if outside display boundary
                if cx_w < -50 or cx_w > self.display_size + 50 or cy_w < -50 or cy_w > self.display_size + 50:
                    continue

                is_hovered = (hover_idx is not None and (idx - 1) == hover_idx)
                if is_hovered:
                    color = (0, 80, 255)
                elif b.get("unsure"):
                    color = (255, 0, 255)          # magenta -- pass 3, decide later
                elif b.get("at_limit"):
                    color = (0, 220, 255)          # amber -- 12-20 px, droppable band
                elif b.get("touches_edge"):
                    color = (255, 200, 0)          # cyan-ish -- trains, but not sized
                else:
                    color = (0, 255, 0)
                thick = 3 if is_hovered else 2

                if st == "ellipse":
                    ax_w_win = int(round(b.get("axis_w", b.get("r", 10.0)*2.0) / 2.0 * scale_factor))
                    ax_h_win = int(round(b.get("axis_h", b.get("r", 10.0)*2.0) / 2.0 * scale_factor))
                    cv2.ellipse(display, (int(round(cx_w)), int(round(cy_w))), (max(1, ax_w_win), max(1, ax_h_win)), float(b.get("angle", 0)), 0, 360, color, thick)
                else:
                    r_win = int(round(b["r"] * scale_factor))
                    cv2.circle(display, (int(round(cx_w)), int(round(cy_w))), max(1, r_win), color, thick)

                cv2.circle(display, (int(round(cx_w)), int(round(cy_w))), 2, (0, 0, 255), -1)
                label_txt = f"{idx} [DEL]" if is_hovered else str(idx)
                label_col = (0, 180, 255) if is_hovered else (255, 255, 0)
                cv2.putText(display, label_txt, (int(round(cx_w)) + 4, int(round(cy_w)) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_col, 1)

            # 3. Draw active drag preview
            if self.drawing and self.pass_mode == "ignore":
                x0_w, y0_w = self.p_start_win
                x1_w, y1_w = self.p_current_win
                cv2.rectangle(display, (x0_w, y0_w), (x1_w, y1_w), (0, 170, 255), 2)
            elif self.drawing:
                x0_w, y0_w = self.p_start_win
                x1_w, y1_w = self.p_current_win
                if self.stamp_mode == "circle":
                    r_drag = int(round(math.hypot(x1_w - x0_w, y1_w - y0_w)))
                    cv2.circle(display, (x0_w, y0_w), max(1, r_drag), (0, 255, 255), 2)
                    cv2.circle(display, (x0_w, y0_w), 2, (0, 0, 255), -1)
                else:
                    cx_d, cy_d = int((x0_w + x1_w) / 2), int((y0_w + y1_w) / 2)
                    ax_w_d, ax_h_d = int(abs(x1_w - x0_w) / 2), int(abs(y1_w - y0_w) / 2)
                    cv2.ellipse(display, (cx_d, cy_d), (max(1, ax_w_d), max(1, ax_h_d)), 0, 0, 360, (0, 255, 255), 2)

            # 4. HUD Header Banner
            view_status = "ENHANCED (CLAHE)" if self.show_enhanced else "RAW"
            hud_line1 = f"[{self.current_idx+1}/{len(self.crop_files)}] {crop_rel} | {regime}"
            n_lim = sum(1 for b in self.bubbles if b.get("at_limit"))
            n_uns = sum(1 for b in self.bubbles if b.get("unsure"))
            hud_line2 = (f"Pass: {self.pass_mode.upper()} ('I') | Shape: {self.stamp_mode.upper()} ('M') | "
                         f"View: {view_status} ('V') | Zoom: {self.zoom:.2f}x | "
                         f"Undo({len(self.undo_stack)}) Redo({len(self.redo_stack)})")
            hud_line3 = (f"Bubbles: {len(self.bubbles)}  at_limit: {n_lim}  unsure: {n_uns}  "
                         f"ignore: {len(self.ignore_regions)}  "
                         f"empty-confirmed: {'YES' if self.confirmed_empty else 'no'} ('E')")

            cv2.rectangle(display, (0, 0), (display.shape[1], 62), (20, 20, 20), -1)
            if self.pass_mode == "ignore":
                cv2.rectangle(display, (0, 0), (display.shape[1], 62), (0, 70, 110), -1)
            cv2.putText(display, hud_line1, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            cv2.putText(display, hud_line2, (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
            cv2.putText(display, hud_line3, (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 255, 180), 1)

            if time.time() < self._warn_until:
                tw = display.shape[1]
                cv2.rectangle(display, (0, 66), (tw, 96), (0, 0, 140), -1)
                cv2.putText(display, self._warn_text, (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)

            # 5. Composite Canvas: Display Viewport (left) + Rules Sidebar (right)
            canvas = np.zeros((self.display_size, self.display_size + self.sidebar_width, 3), dtype=np.uint8)
            canvas[:, :self.display_size] = display
            canvas[:, self.display_size:] = self._render_sidebar(self.display_size, self.sidebar_width)

            cv2.imshow(self.window_name, canvas)
            key_full = cv2.waitKeyEx(20)
            key = key_full & 0xFF

            # Arrow keys / WASD panning
            step = 40.0 / self.zoom
            # Windows waitKeyEx: Left: 2424832, Up: 2490368, Right: 2555904, Down: 2621440
            # Linux: Left: 65361, Up: 65362, Right: 65363, Down: 65364
            if key_full in (2490368, 65362):  # Up
                self.center_view[1] -= step
            elif key_full in (2621440, 65364):  # Down
                self.center_view[1] += step
            elif key_full in (2424832, 65361):  # Left
                self.center_view[0] -= step
            elif key_full in (2555904, 65363):  # Right
                self.center_view[0] += step

            # Key actions
            if key in (ord('q'), ord('Q'), 27):
                self._save_labels()
                break
            elif key in (ord('s'), ord('S')):
                self._save_labels()
            elif key in (ord('n'), ord('N'), 32):
                self._save_labels()
                if self.current_idx < len(self.crop_files) - 1:
                    self._load_crop(self.current_idx + 1)
            elif key in (ord('p'), ord('P')):
                self._save_labels()
                if self.current_idx > 0:
                    self._load_crop(self.current_idx - 1)
            elif key in (ord('m'), ord('M')):
                self.stamp_mode = "ellipse" if self.stamp_mode == "circle" else "circle"
            elif key in (ord('v'), ord('V')):
                self.show_enhanced = not self.show_enhanced
            elif key in (ord('i'), ord('I')):
                self.pass_mode = "ignore" if self.pass_mode == "bubble" else "bubble"
                print(f"[Pass] {self.pass_mode.upper()} "
                      f"({'rough box over an occluded bubble' if self.pass_mode == 'ignore' else 'stamp measurable bubbles'})")
            elif key in (ord('e'), ord('E')):
                if self.bubbles and not self.confirmed_empty:
                    self._warn("cannot confirm empty: this crop has stamped bubbles")
                else:
                    self.confirmed_empty = not self.confirmed_empty
                    print(f"[confirmed_empty] {self.confirmed_empty} -- an empty crop only "
                          f"tests false positives when someone says it is empty")
            elif key in (ord('t'), ord('T')):
                img_x, img_y = self._win_to_img(self.p_current_win[0], self.p_current_win[1], h_img, w_img)
                if not self._toggle_unsure_at(img_x, img_y):
                    print("[Info] Hover over a bubble and press 'T' to flag it unsure.")
            elif key in (ord('d'), ord('D'), ord('x'), ord('X'), 8, 127) or key_full in (3014656, 65535):
                img_x, img_y = self._win_to_img(self.p_current_win[0], self.p_current_win[1], h_img, w_img)
                deleted = self._delete_bubble_at(img_x, img_y)
                if not deleted and self.bubbles:
                    print(f"[Info] Hover cursor over a bubble and press 'D' to delete, or press Ctrl+Z to undo.")
            elif key in (26, ord('z'), ord('Z'), ord('u'), ord('U')):
                self.undo()
            elif key in (25, ord('y'), ord('Y')):
                self.redo()
            elif key in (ord('+'), ord('=')):
                self.zoom = min(6.0, self.zoom * 1.25)
            elif key in (ord('-'), ord('_')):
                self.zoom = max(1.0, self.zoom / 1.25)
            elif key in (ord('r'), ord('R'), ord('0')):
                self.zoom = 1.0
                self.center_view = [256.0, 256.0]
            elif key in (ord('f'), ord('F')):
                modes = ["ALL", "FROTH", "ALL PULP", "ROTOR RAW", "ROTOR DILUTED", "VENTURI AIR"]
                curr = self.filter_mode if self.filter_mode in modes else "ALL"
                idx_f = modes.index(curr)
                self.filter_mode = modes[(idx_f + 1) % len(modes)]
                self._apply_filter()
                self.current_idx = self._first_unlabelled()
                self._load_crop(self.current_idx)

        cv2.destroyAllWindows()
