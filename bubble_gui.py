"""bubble_gui.py
BubbleMetrix - Desktop GUI for Bubble Size Measurement & Metrology.
Clean, native scientific desktop interface (Tkinter + Pillow + Matplotlib).
No local web server needed. Runs standalone via launch_gui.bat or gui.ps1.

Operating Regimes:
  1. 💧 Pulp  - Mask R-CNN (BubMask) at 3.291 um/px.
  2. 🫧 Froth - Cellpose, zero-shot (validated: 26/26 hand-labelled bubbles, IoU 0.80).
               Runs in Froth/.venv through a helper process; scale from the 55 mm tube.

No tuning controls on purpose: every threshold is fixed at the value it was validated
with, so a scientist picks a video, presses one button and gets a D32.

Core Modules:
  - Measurement Module: Full video scrubber, frame detection, D32, D43, BSD histogram & Q3 curve.
  - Validation Module: Ground truth comparison, split view, Precision, Recall, F1, Mean IoU,
                       D32 Error %, and Interactive Self-Test Sandbox.
  - Interactive Bubble Editor: Direct canvas editing to delete false positives and add missing
                               real bubbles with live real-time D32 recalculation.
  - D32 Convergence: Measure Full Video stops once D32 is known to +/-5% (95%), and plots
                     running D32 against the number of bubbles measured (engines/convergence.py).
  - Fine-tune Model: add frames from new footage, correct the model's pre-labels, press one
                     button; the fine-tuned model is used only if it beats the old one on
                     held-back frames and on the frozen benchmark (bubble_gui_teach.py).

Detections are stored per frame: switching frames shows that frame's own result or none,
never the previous frame's contours.
"""

from __future__ import annotations
import csv
import json
import math
import os
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont

# Set up matplotlib backend for Tkinter
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Setup search paths
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = HERE

PULP_DIR = os.path.join(PROJECT_ROOT, "BubMask")
FROTH_DIR = os.path.join(PROJECT_ROOT, "Froth")
ENGINES_DIR = os.path.join(PROJECT_ROOT, "engines")

for p in [PROJECT_ROOT, PULP_DIR, FROTH_DIR, ENGINES_DIR]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from engine_metrology import (
    compute_sauter_d32,
    compute_debrouckere_d43,
    compute_distribution_histogram,
    evaluate_predictions_vs_ground_truth,
    summarize_bubbles,
)
from engine_pulp import PulpDetector
from engine_froth import FrothDetector, DEFAULT_MM_PER_PX
from convergence import ConvergenceTracker, spread_order, TARGET as D32_TARGET
import teach
from bubble_gui_teach import TeachTabMixin, ConvergenceTabMixin

def _latest_pulp_weights() -> str:
    """Newest trained best.pt under runs/ (skips smoke tests and CV folds)."""
    import glob
    cands = [p for p in glob.glob(os.path.join(PULP_DIR, "runs", "*", "best.pt"))
             if not os.path.basename(os.path.dirname(p)).startswith(("_", "cv"))]
    return max(cands, key=os.path.getmtime) if cands else os.path.join(PULP_DIR, "runs", "rr_full_e20", "best.pt")


DEFAULT_PULP_WEIGHTS = _latest_pulp_weights()
DEFAULT_PULP_CALIB = 3.291  # um/px

# Fixed, validated settings (no longer exposed in the UI).
PULP_CONF = 0.40          # operating point chosen for D32 (handover s8)
PULP_MIN_D_PX = 12.0      # label floor
PULP_MAX_D_PX = 250.0
# Measure Full Video: candidate frames, visited in a spread order (start, end, middle,
# quarters ...) and stopped once D32 is stable (engines/convergence.py).
FROTH_VIDEO_MAX_FRAMES = 40   # evenly spaced over the clip (decorrelated froth samples)
PULP_FRAME_GAP = 5            # BubbleConfig.rotor_frame_gap: frames this far apart are decorrelated
PULP_VIDEO_MAX = 120


class BubbleMetrixApp(TeachTabMixin, ConvergenceTabMixin, tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BubbleMetrix - Multiphase Bubble Size Measurement Studio")
        self.geometry("1420x920")
        self.minsize(1100, 720)

        self._setup_theme()

        # Engine instances
        self.pulp_detector = None
        self.froth_detector = None

        # State variables
        self.regime = tk.StringVar(value="pulp")
        self.scale_var = tk.DoubleVar(value=DEFAULT_PULP_CALIB)   # μm/px (pulp) or mm/px (froth)
        self.unit_var = tk.StringVar(value="μm/px")
        # Scale: None = default (pulp 3.291 μm/px; froth measured from the 55 mm tube on each
        # frame); a number = set by the user in the Scale box, used for every frame.
        self.scale_manual: Dict[str, Optional[float]] = {"pulp": None, "froth": None}
        self.scale_entry_var = tk.StringVar(value=f"{DEFAULT_PULP_CALIB:g}")
        self._suppress_tab_sync = False
        self.method_info_var = tk.StringVar(value="Method: Mask R-CNN")
        self.froth_roi = None    # (x0, y0, x1, y1) measured region on the last froth frame
        # Pulp model: the fine-tuned one if the Fine-tune tab adopted one, else the base model
        self.default_pulp_weights = DEFAULT_PULP_WEIGHTS
        self.weights_path = tk.StringVar(value=teach.active_model("pulp") or DEFAULT_PULP_WEIGHTS)
        self.status_var = tk.StringVar(value="Ready")

        # Input & Video State
        self.input_type = tk.StringVar(value="video")  # 'video' | 'crop' | 'image'
        self.selected_source_path = ""
        self.video_cap = None
        self.total_video_frames = 0
        self.current_frame_idx = tk.IntVar(value=0)
        self.frame_str_var = tk.StringVar(value="Frame 0 / 0")

        # Partially labelled ground truth (precision not computable), from Froth/eval/cases.json
        self.partial_label_files = self._load_partial_label_files()

        # Image & Canvas Viewport State
        self.current_raw_img: Optional[Image.Image] = None
        self.current_display_img: Optional[Image.Image] = None
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.is_auto_fit = True
        self._drag_start = None

        # Measured bubbles dataset
        self.measured_bubbles: List[Dict[str, Any]] = []        # bubbles drawn on THIS frame
        self.metrology_summary: Optional[Dict[str, Any]] = None
        # Results kept per frame, so switching frames never shows another frame's contours:
        # (source path, frame index) -> {"bubbles", "summary" (single-frame runs), "roi"}
        self.frame_results: Dict[Tuple[str, int], Dict[str, Any]] = {}
        # Last Measure Full Video result: (video path, summary over all sampled frames)
        self.video_result: Optional[Tuple[str, Dict[str, Any]]] = None
        self.stop_when_converged = tk.BooleanVar(value=True)

        # View mode toggles
        self.view_mode = tk.StringVar(value="overlay")  # 'raw' | 'overlay' | 'validation'
        self.show_labels = tk.BooleanVar(value=False)   # per-bubble numbers clutter a field of hundreds

        # Editor State
        self.editor_mode = tk.StringVar(value="inspect")  # 'inspect' | 'delete' | 'circle' | 'ellipse'
        self.editor_bubbles: List[Dict[str, Any]] = []
        self.editor_original_raw: List[Dict[str, Any]] = []
        self.editor_history: List[List[Dict[str, Any]]] = []
        self.editor_redo_history: List[List[Dict[str, Any]]] = []
        self.editor_stamp_start: Optional[Tuple[float, float]] = None
        self.editor_stamp_current: Optional[Tuple[float, float]] = None
        self.editor_added_count = 0
        self.editor_deleted_count = 0

        # Validation State
        self.val_ground_truth: List[Dict[str, Any]] = []
        self.val_predictions: List[Dict[str, Any]] = []
        self.val_metrics: Optional[Dict[str, Any]] = None
        self.val_tool_mode = tk.StringVar(value="off")  # "off", "draw", "delete"
        self.val_draw_start: Optional[Tuple[float, float]] = None
        self.val_draw_current: Optional[Tuple[float, float]] = None
        self.val_gt_history: List[List[Dict[str, Any]]] = []
        self.val_gt_redo_history: List[List[Dict[str, Any]]] = []
        self._pending_validation: bool = False

        # Async queue
        self.task_queue = queue.Queue()

        self._init_teach_state()
        self._build_ui()
        self._bind_events()
        self._discover_data()
        self.after(100, self._check_async_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -------------------------------------------------------------------------
    # Units, scale and housekeeping
    # -------------------------------------------------------------------------
    def _is_froth(self) -> bool:
        return self.regime.get() == "froth"

    def _unit(self) -> str:
        """Display unit for diameters."""
        return "mm" if self._is_froth() else "μm"

    def _unit_name(self) -> str:
        """Unit tag stored in records / summaries."""
        return "mm" if self._is_froth() else "um"

    def _fmt(self, v: float) -> str:
        """Diameters: mm to 2 decimals (froth ~1-3 mm), μm to 1."""
        return f"{v:.2f}" if self._is_froth() else f"{v:.1f}"

    def _calibration_note(self) -> str:
        if self.scale_manual[self.regime.get()]:
            return f"set by the user: {self.scale_var.get():g} {'mm' if self._is_froth() else 'um'}/px"
        if self._is_froth():
            src = self.froth_detector.scale_source if self.froth_detector else "55 mm inner width / 471 px"
            return f"55 mm tube width -> {self.scale_var.get():.5f} mm/px ({src})"
        return "provisional -- 3.291 um/px is derived, not measured"

    def _refresh_scale_info(self):
        self.method_info_var.set("Method: Cellpose" if self._is_froth() else "Method: Mask R-CNN")
        self.unit_var.set("mm/px" if self._is_froth() else "μm/px")
        v = self.scale_var.get()
        self.scale_entry_var.set(f"{v:.5f}" if self._is_froth() else f"{v:g}")

    # ---------------- adjustable scale ----------------
    def _default_scale(self) -> float:
        if self._is_froth():
            return self.froth_detector.mm_per_px if self.froth_detector else DEFAULT_MM_PER_PX
        return DEFAULT_PULP_CALIB

    def _apply_scale_entry(self, event=None):
        try:
            v = float(self.scale_entry_var.get().replace(",", "."))
            if not (v > 0 and math.isfinite(v)):
                raise ValueError
        except ValueError:
            messagebox.showwarning("Scale", f"Enter the size of one pixel as a positive number ({self.unit_var.get()}).")
            self._refresh_scale_info()
            return
        reg = self.regime.get()
        self.scale_manual[reg] = v
        self.scale_var.set(v)
        self._refresh_scale_info()
        self._log(f"[Scale] {reg}: set to {v:g} {self.unit_var.get()} for all frames.")
        self._rescale_results()

    def _reset_scale(self):
        reg = self.regime.get()
        self.scale_manual[reg] = None
        self.scale_var.set(self._default_scale())
        self._refresh_scale_info()
        self._log(f"[Scale] {reg}: back to the default "
                  f"({'measured from the 55 mm tube on each frame' if reg == 'froth' else '3.291 μm/px'}).")
        self._rescale_results()

    def _rescale_summary(self, summ: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Re-express a result in the scale now in use. Detection is in pixels, so only
        the conversion changes; the accepted bubbles and their gates stay the same."""
        if not summ:
            return summ
        reg = "froth" if summ.get("unit") == "mm" else "pulp"
        target = self.scale_manual[reg] or summ.get("auto_scale_per_px") or (
            DEFAULT_MM_PER_PX if reg == "froth" else DEFAULT_PULP_CALIB)
        old = summ.get("scale_per_px") or target
        if abs(old - target) < 1e-12:
            return summ
        new = summarize_bubbles(summ.get("accepted_bubbles", []), um_per_px=target,
                                unit_name="mm" if reg == "froth" else "um", exclude_edge=False,
                                calibration_note=("set by the user: %g per px" % target) if self.scale_manual[reg]
                                else None)
        f = target / old
        for k, v in summ.items():
            if k not in new or k in ("n_total", "n_rejected", "n_edge_excluded", "rejected_bubbles"):
                new[k] = v
        new["auto_scale_per_px"] = summ.get("auto_scale_per_px")
        conv = summ.get("convergence")
        if conv:
            conv = dict(conv)
            for k in ("d32", "lo", "hi"):
                if k in conv:
                    conv[k] = conv[k] * f
            conv["history"] = [dict(h, d32=h["d32"] * f, lo=h["lo"] * f, hi=h["hi"] * f) for h in conv.get("history", [])]
            new["convergence"] = conv
        if "mm_per_px" in new:
            new["mm_per_px"] = target
        return new

    def _rescale_results(self):
        if self.video_result:
            self.video_result = (self.video_result[0], self._rescale_summary(self.video_result[1]))
        for key, e in self.frame_results.items():
            if e.get("summary"):
                e["summary"] = self._rescale_summary(e["summary"])
        # per-bubble values in every stored frame
        for key, e in self.frame_results.items():
            reg = "froth" if any(b.get("unit") == "mm" for b in e.get("bubbles", [])) else "pulp"
            src = e.get("summary") or (self.video_result[1] if self.video_result and self.video_result[0] == key[0] else None)
            sc = (src or {}).get("scale_per_px")
            if sc:
                for b in e.get("bubbles", []):
                    b["diameter_val"] = round(float(b.get("diameter_px", 0)) * sc, 2)
        if self.metrology_summary:
            self.metrology_summary = self._rescale_summary(self.metrology_summary)
        if self.input_type.get() == "video":
            self._restore_frame_result()
        elif self.metrology_summary:
            self._update_kpis(self.metrology_summary)
            self._plot_histogram(self.metrology_summary.get("distribution"))
            self._populate_table(self.metrology_summary.get("accepted_bubbles", []))
        self._recalculate_editor_live_math()
        if self.video_result and self.video_result[1].get("convergence"):
            self.conv_unit = "mm" if self.video_result[1].get("unit") == "mm" else "μm"
            self._conv_finish(self.video_result[1]["convergence"], self.video_result[1])
        self._redraw_current_canvas()

    def _load_partial_label_files(self):
        """Label files that are only partly labelled -> precision is meaningless on them."""
        out = set()
        try:
            cases = json.load(open(os.path.join(FROTH_DIR, "eval", "cases.json"), encoding="utf-8"))
            for c in cases.values():
                if not c.get("fully_labelled"):
                    out.add(os.path.basename(c.get("labels", "")))
        except Exception:
            pass
        out.add("180WW_f00299_x104y400_labels.json")   # 26 stamped circles out of several hundred
        return out

    def _log_async(self, text: str):
        self.task_queue.put(("log", text))

    def _on_close(self):
        try:
            if self.froth_detector:
                self.froth_detector.close()
        finally:
            self.destroy()

    def _setup_theme(self):
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        self.font_main = ("Segoe UI", 9)
        self.font_bold = ("Segoe UI", 9, "bold")
        self.font_title = ("Segoe UI", 10, "bold")
        self.font_mono = ("Consolas", 9)

        # Scientific desktop palette
        self.configure(bg="#eef2f6")
        self.style.configure("TFrame", background="#eef2f6")
        self.style.configure("TLabelframe", background="#eef2f6", padding=6)
        self.style.configure("TLabelframe.Label", font=self.font_title, foreground="#1e293b", background="#eef2f6")
        self.style.configure("TLabel", background="#eef2f6", foreground="#334155", font=self.font_main)
        self.style.configure("TButton", font=self.font_main, padding=(6, 3))
        self.style.configure("TRadiobutton", background="#eef2f6", font=self.font_main)
        self.style.configure("TCheckbutton", background="#eef2f6", font=self.font_main)

        # Regime switcher button style
        self.style.configure("Regime.TRadiobutton", font=("Segoe UI", 10, "bold"), padding=(8, 4))

        # Action Buttons
        self.style.configure("Primary.TButton", font=("Segoe UI", 9, "bold"),
                             foreground="#ffffff", background="#1976d2", padding=(10, 5))
        self.style.map("Primary.TButton", background=[("active", "#1565c0"), ("disabled", "#b0bec5")])

        self.style.configure("Success.TButton", font=("Segoe UI", 9, "bold"),
                             foreground="#ffffff", background="#059669", padding=(8, 4))
        self.style.map("Success.TButton", background=[("active", "#047857"), ("disabled", "#b0bec5")])

        self.style.configure("Warn.TButton", font=("Segoe UI", 9, "bold"),
                             foreground="#ffffff", background="#dc2626", padding=(8, 4))
        self.style.map("Warn.TButton", background=[("active", "#b91c1c"), ("disabled", "#b0bec5")])

    # -------------------------------------------------------------------------
    # Main Layout
    # -------------------------------------------------------------------------
    def _build_ui(self):
        root_pane = ttk.Frame(self, padding=6)
        root_pane.pack(fill="both", expand=True)
        root_pane.columnconfigure(1, weight=1)
        root_pane.rowconfigure(0, weight=1)

        # Left Column: Controls (Scrollable)
        left_container = ttk.Frame(root_pane, width=410)
        left_container.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left_container.rowconfigure(0, weight=1)
        left_container.columnconfigure(0, weight=1)

        self._build_left_panel(left_container)

        # Right Column: Multi-tab Viewer & Metrology
        right_container = ttk.Frame(root_pane)
        right_container.grid(row=0, column=1, sticky="nsew")
        right_container.rowconfigure(0, weight=1)
        right_container.columnconfigure(0, weight=1)

        self._build_right_notebook(right_container)

        # Bottom Status Bar
        statusbar = ttk.Frame(root_pane, padding=(4, 4, 4, 2))
        statusbar.grid(row=1, column=0, columnspan=2, sticky="ew")
        statusbar.columnconfigure(0, weight=1)

        status_lbl = ttk.Label(statusbar, textvariable=self.status_var, font=self.font_mono)
        status_lbl.grid(row=0, column=0, sticky="w")

        self.progress_bar = ttk.Progressbar(statusbar, mode="determinate", length=260)
        self.progress_bar.grid(row=0, column=1, sticky="e", padx=(8, 0))

    def _build_left_panel(self, container):
        canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0, width=400, bg="#eef2f6")
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scroll_content = ttk.Frame(canvas, padding=(0, 0, 4, 0))

        scroll_content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_win = canvas.create_window((0, 0), window=scroll_content, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_win, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # 1. Operating Regime Header Card
        regime_box = ttk.LabelFrame(scroll_content, text="1. Operating Regime", padding=6)
        regime_box.pack(fill="x", pady=(0, 6))

        r_row = ttk.Frame(regime_box)
        r_row.pack(fill="x", pady=2)
        ttk.Radiobutton(r_row, text="💧 Pulp", value="pulp",
                        variable=self.regime, style="Regime.TRadiobutton",
                        command=self._on_regime_changed).pack(side="left", padx=(0, 10))
        ttk.Radiobutton(r_row, text="🫧 Froth", value="froth",
                        variable=self.regime, style="Regime.TRadiobutton",
                        command=self._on_regime_changed).pack(side="left")

        ttk.Label(regime_box, textvariable=self.method_info_var, foreground="#475569").pack(anchor="w", pady=(4, 0))
        sc_row = ttk.Frame(regime_box)
        sc_row.pack(fill="x", pady=(4, 0))
        ttk.Label(sc_row, text="Scale:").pack(side="left")
        sc_entry = ttk.Entry(sc_row, textvariable=self.scale_entry_var, width=10, font=self.font_mono)
        sc_entry.pack(side="left", padx=(4, 2))
        sc_entry.bind("<Return>", self._apply_scale_entry)
        ttk.Label(sc_row, textvariable=self.unit_var).pack(side="left", padx=(0, 6))
        ttk.Button(sc_row, text="Set", width=5, command=self._apply_scale_entry).pack(side="left")
        ttk.Button(sc_row, text="Default", width=8, command=self._reset_scale).pack(side="left", padx=(2, 0))

        # 2. Input Source Selection
        in_box = ttk.LabelFrame(scroll_content, text="2. Input Target & Source", padding=6)
        in_box.pack(fill="x", pady=(0, 6))

        in_type_row = ttk.Frame(in_box)
        in_type_row.pack(fill="x", pady=(1, 4))
        ttk.Radiobutton(in_type_row, text="🎬 Video", value="video",
                        variable=self.input_type, command=self._sync_input_type).pack(side="left", padx=(0, 8))
        ttk.Radiobutton(in_type_row, text="📁 Benchmark Crop", value="crop",
                        variable=self.input_type, command=self._sync_input_type).pack(side="left", padx=(0, 8))
        # (an image opened with Browse... still works; it just has no button of its own)

        # Dropdowns
        self.video_combo_var = tk.StringVar()
        self.video_combo = ttk.Combobox(in_box, textvariable=self.video_combo_var, state="readonly")
        self.video_combo.bind("<<ComboboxSelected>>", lambda e: self._on_source_combo_selected())
        self.video_combo.pack(fill="x", pady=2)

        # File browse row
        browse_row = ttk.Frame(in_box)
        browse_row.pack(fill="x", pady=(2, 2))
        self.source_path_entry = ttk.Entry(browse_row)
        self.source_path_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(browse_row, text="Browse...", width=9, command=self._browse_source_file).pack(side="left", padx=(4, 0))

        # Video Frame Scrubber Frame
        self.video_scrubber_frame = ttk.LabelFrame(scroll_content, text="3. Video Scrubber & Stepper", padding=6)
        self.video_scrubber_frame.pack(fill="x", pady=(0, 6))

        f_hdr = ttk.Frame(self.video_scrubber_frame)
        f_hdr.pack(fill="x")
        ttk.Label(f_hdr, text="Scrub Frame:").pack(side="left")
        ttk.Label(f_hdr, textvariable=self.frame_str_var, font=self.font_mono, foreground="#1976d2").pack(side="right")

        self.frame_scale = ttk.Scale(self.video_scrubber_frame, from_=0, to=100,
                                      variable=self.current_frame_idx, orient="horizontal",
                                      command=self._on_slider_scrubbed)
        self.frame_scale.pack(fill="x", pady=(3, 3))

        step_btn_row = ttk.Frame(self.video_scrubber_frame)
        step_btn_row.pack(fill="x")
        ttk.Button(step_btn_row, text="⏮ -5", width=5, command=lambda: self._step_frame(-5)).pack(side="left", padx=1)
        ttk.Button(step_btn_row, text="◀ -1", width=5, command=lambda: self._step_frame(-1)).pack(side="left", padx=1)
        ttk.Button(step_btn_row, text="+1 ▶", width=5, command=lambda: self._step_frame(1)).pack(side="left", padx=1)
        ttk.Button(step_btn_row, text="+5 ⏭", width=5, command=lambda: self._step_frame(5)).pack(side="left", padx=1)

        # 4. Pipeline Execution Actions
        action_box = ttk.LabelFrame(scroll_content, text="4. Execution & Pipeline", padding=6)
        action_box.pack(fill="x", pady=(0, 6))
        self.action_box = action_box

        self.btn_measure_video = ttk.Button(action_box, text="🎬 Measure Full Video (size distribution)",
                                            style="Primary.TButton", command=self._action_measure_video)
        self.btn_measure_video.pack(fill="x", pady=2)
        self.chk_stop = ttk.Checkbutton(action_box, text="Stop when D32 is stable (±5%, 95% confidence)",
                                        variable=self.stop_when_converged)
        self.chk_stop.pack(anchor="w", pady=(0, 4))
        self._sync_stop_checkbox()

        self.btn_measure_frame = ttk.Button(action_box, text="⚡ Measure Current Frame",
                                            command=self._action_measure_frame)
        self.btn_measure_frame.pack(fill="x", pady=2)

        self.btn_val_benchmark = ttk.Button(action_box, text="🎯 Run Validation vs Ground Truth",
                                            style="Success.TButton", command=self._action_run_validation)
        self.btn_val_benchmark.pack(fill="x", pady=2)



    # -------------------------------------------------------------------------
    # Right Column: Multi-tab Notebook
    # -------------------------------------------------------------------------
    def _build_right_notebook(self, container):
        self.notebook = ttk.Notebook(container)
        self.notebook.pack(fill="both", expand=True)

        # Tab 1: Image & Overlay Viewer
        self.tab_viewer = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(self.tab_viewer, text="👁️ Viewer")
        self._build_viewer_tab(self.tab_viewer)

        # Tab 2: Interactive Bubble Editor
        self.tab_editor = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(self.tab_editor, text="✏️ Editor")
        self._build_editor_tab(self.tab_editor)

        # Tab 3: Size Distribution (PSD Histogram)
        self.tab_histogram = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(self.tab_histogram, text="📊 BSD")
        self._build_histogram_tab(self.tab_histogram)

        # Tab 4: Validation & Benchmark
        self.tab_validation = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(self.tab_validation, text="🎯 Validation")
        self._build_validation_tab(self.tab_validation)

        # Tab 5: D32 convergence (how many bubbles until D32 stops changing)
        self.tab_convergence = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(self.tab_convergence, text="📈 D32 Convergence")
        self._build_convergence_tab(self.tab_convergence)

        # Tab 6: Fine-tune the model on new footage
        self.tab_teach = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(self.tab_teach, text="🧠 Fine-tune Model")
        self._build_teach_tab(self.tab_teach)

        # Tab 5: Data Table & Export
        self.tab_table = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(self.tab_table, text="📋 Table")
        self._build_table_tab(self.tab_table)

        # Tab 6: Console Log
        self.tab_log = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(self.tab_log, text="📝 Log")
        self._build_log_tab(self.tab_log)

    def _build_viewer_tab(self, parent):
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        # Top Control Bar
        toolbar = ttk.Frame(parent)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        ttk.Label(toolbar, text="View Mode:").pack(side="left", padx=(0, 4))
        ttk.Radiobutton(toolbar, text="Detection Overlay", value="overlay",
                        variable=self.view_mode, command=self._redraw_current_canvas).pack(side="left", padx=2)
        ttk.Radiobutton(toolbar, text="Raw Image", value="raw",
                        variable=self.view_mode, command=self._redraw_current_canvas).pack(side="left", padx=2)
        ttk.Radiobutton(toolbar, text="Validation Match", value="validation",
                        variable=self.view_mode, command=self._redraw_current_canvas).pack(side="left", padx=2)

        ttk.Checkbutton(toolbar, text="Show Labels", variable=self.show_labels,
                        command=self._redraw_current_canvas).pack(side="left", padx=(8, 4))

        # Canvas Zoom Tools
        ttk.Button(toolbar, text="Fit Image", command=self._canvas_fit_image).pack(side="right", padx=2)
        ttk.Button(toolbar, text="Zoom -", command=lambda: self._canvas_zoom(0.8)).pack(side="right", padx=2)
        ttk.Button(toolbar, text="Zoom +", command=lambda: self._canvas_zoom(1.25)).pack(side="right", padx=2)

        # Canvas Center Area
        canvas_frame = ttk.Frame(parent, borderwidth=1, relief="sunken")
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas_viewer = tk.Canvas(canvas_frame, bg="#000000", highlightthickness=0)
        self.canvas_viewer.grid(row=0, column=0, sticky="nsew")

        # KPI Ribbon at bottom of viewer
        self.kpi_ribbon = ttk.Frame(parent, padding=(2, 4))
        self.kpi_ribbon.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        self._build_kpi_cards(self.kpi_ribbon)

    def _build_kpi_cards(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(2, weight=1)
        parent.columnconfigure(3, weight=1)

        self.kpi_labels = {}
        items = [
            ("d32", "Sauter Mean (D32)", "0.0", "Volume-weighted mean"),
            ("d43", "De Brouckere (D43)", "0.0", "Mass moment mean"),
            ("count", "In-Focus Bubbles", "0", "Accepted instances"),
            ("d50", "Median (D50)", "0.0", "Span: 0.00"),
        ]

        for col, (key, title, def_val, sub) in enumerate(items):
            card = ttk.LabelFrame(parent, text=title, padding=(6, 4))
            card.grid(row=0, column=col, sticky="nsew", padx=2)

            val_lbl = ttk.Label(card, text=def_val, font=("Segoe UI", 13, "bold"), foreground="#1976d2")
            val_lbl.pack(anchor="w")

            sub_lbl = ttk.Label(card, text=sub, font=("Segoe UI", 7), foreground="#64748b")
            sub_lbl.pack(anchor="w")

            self.kpi_labels[key] = (val_lbl, sub_lbl)

    def _build_editor_tab(self, parent):
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        # Editor Toolbar
        toolbar = ttk.Frame(parent)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        ttk.Label(toolbar, text="Editor Tool:", font=self.font_bold).pack(side="left", padx=(0, 6))

        ttk.Radiobutton(toolbar, text="👆 Inspect", value="inspect",
                        variable=self.editor_mode, command=self._on_editor_tool_changed).pack(side="left", padx=2)
        ttk.Radiobutton(toolbar, text="❌ Delete False Bubble", value="delete",
                        variable=self.editor_mode, command=self._on_editor_tool_changed).pack(side="left", padx=2)
        ttk.Radiobutton(toolbar, text="⚪ Add Circle Stamp", value="circle",
                        variable=self.editor_mode, command=self._on_editor_tool_changed).pack(side="left", padx=2)
        ttk.Radiobutton(toolbar, text="⬭ Add Ellipse", value="ellipse",
                        variable=self.editor_mode, command=self._on_editor_tool_changed).pack(side="left", padx=2)

        # Action Buttons
        ttk.Button(toolbar, text="↩ Undo (Ctrl+Z)", command=self._editor_undo).pack(side="left", padx=(8, 2))
        ttk.Button(toolbar, text="↪ Redo", command=self._editor_redo).pack(side="left", padx=2)
        ttk.Button(toolbar, text="↺ Reset Raw", command=self._editor_reset_raw).pack(side="left", padx=2)

        ttk.Button(toolbar, text="💾 Finalize & Save Results", style="Success.TButton",
                   command=self._editor_finalize_results).pack(side="right", padx=2)

        # Canvas Area for Editor
        canvas_frame = ttk.Frame(parent, borderwidth=1, relief="sunken")
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas_editor = tk.Canvas(canvas_frame, bg="#000000", highlightthickness=0, cursor="crosshair")
        self.canvas_editor.grid(row=0, column=0, sticky="nsew")

        # Live Math Ribbon
        live_ribbon = ttk.Frame(parent, padding=(4, 4))
        live_ribbon.grid(row=2, column=0, sticky="ew", pady=(4, 0))

        ttk.Label(live_ribbon, text="⚡ Live Recalculated Values:  ", font=self.font_bold, foreground="#1976d2").pack(side="left")
        self.lbl_editor_live_d32 = ttk.Label(live_ribbon, text="D32: 0.0 μm", font=("Segoe UI", 10, "bold"), foreground="#059669")
        self.lbl_editor_live_d32.pack(side="left", padx=(0, 16))

        self.lbl_editor_live_count = ttk.Label(live_ribbon, text="Bubbles: 0", font=("Segoe UI", 10, "bold"))
        self.lbl_editor_live_count.pack(side="left", padx=(0, 16))

        self.lbl_editor_changes = ttk.Label(live_ribbon, text="(0 added, 0 deleted)", font=self.font_main, foreground="#64748b")
        self.lbl_editor_changes.pack(side="left")

    def _build_histogram_tab(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        self.fig_psd, self.ax_psd = plt.subplots(figsize=(8, 5), dpi=100)
        self.fig_psd.patch.set_facecolor("#f8fafc")
        self.ax_psd.set_facecolor("#ffffff")

        self.canvas_psd = FigureCanvasTkAgg(self.fig_psd, master=parent)
        self.canvas_psd.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _build_validation_tab(self, parent):
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        # Validation Header Toolbar
        val_toolbar = ttk.Frame(parent)
        val_toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        ttk.Button(val_toolbar, text="🎯 Run Benchmark vs Labels", style="Success.TButton",
                   command=self._action_run_validation).pack(side="left", padx=2)

        ttk.Separator(val_toolbar, orient="vertical").pack(side="left", fill="y", padx=8)

        ttk.Label(val_toolbar, text="Self-Annotation:", font=self.font_bold).pack(side="left", padx=(0, 4))
        ttk.Radiobutton(val_toolbar, text="Off", value="off", variable=self.val_tool_mode,
                        command=self._on_val_tool_changed).pack(side="left", padx=2)
        ttk.Radiobutton(val_toolbar, text="✏️ Stamp", value="draw", variable=self.val_tool_mode,
                        command=self._on_val_tool_changed).pack(side="left", padx=2)
        ttk.Radiobutton(val_toolbar, text="🗑️ Delete", value="delete", variable=self.val_tool_mode,
                        command=self._on_val_tool_changed).pack(side="left", padx=2)

        ttk.Button(val_toolbar, text="↩ Undo (Ctrl+Z)", command=self._val_gt_undo).pack(side="left", padx=(6, 2))
        ttk.Button(val_toolbar, text="↪ Redo (Ctrl+Y)", command=self._val_gt_redo).pack(side="left", padx=2)
        ttk.Button(val_toolbar, text="Clear All", command=self._clear_self_test_gt).pack(side="left", padx=2)

        self.lbl_val_gt_count = ttk.Label(val_toolbar, text="(0 bubbles)", foreground="#64748b")
        self.lbl_val_gt_count.pack(side="left", padx=6)

        # Validation KPIs Card Row
        self.val_kpi_frame = ttk.Frame(parent)
        self.val_kpi_frame.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        self._build_validation_kpis(self.val_kpi_frame)

        # Validation Canvas
        canvas_frame = ttk.Frame(parent, borderwidth=1, relief="sunken")
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas_val = tk.Canvas(canvas_frame, bg="#000000", highlightthickness=0)
        self.canvas_val.grid(row=0, column=0, sticky="nsew")

    def _build_validation_kpis(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(2, weight=1)
        parent.columnconfigure(3, weight=1)
        parent.columnconfigure(4, weight=1)

        self.val_kpi_labels = {}
        items = [
            ("recall", "Recall", "0.0%", "Recovered bubbles"),
            ("precision", "Precision", "0.0%", "Accurate predictions"),
            ("f1", "F1 Score", "0.000", "Harmonic mean"),
            ("iou", "Mean Mask IoU", "0.0%", "Contour alignment"),
            ("d32_error", "D32 Accuracy", "0.0%", "Error: 0.0%"),
        ]

        for col, (key, title, def_val, sub) in enumerate(items):
            card = ttk.LabelFrame(parent, text=title, padding=(6, 4))
            card.grid(row=0, column=col, sticky="nsew", padx=2)

            val_lbl = ttk.Label(card, text=def_val, font=("Segoe UI", 12, "bold"), foreground="#059669")
            val_lbl.pack(anchor="w")

            sub_lbl = ttk.Label(card, text=sub, font=("Segoe UI", 7), foreground="#64748b")
            sub_lbl.pack(anchor="w")

            self.val_kpi_labels[key] = (val_lbl, sub_lbl)

    def _build_table_tab(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        cols = ("id", "cx", "cy", "r", "diam_px", "diam_unit", "circ", "solidity", "score", "status")
        self.tree = ttk.Treeview(parent, columns=cols, show="headings")

        self.tree.heading("id", text="ID")
        self.tree.heading("cx", text="Center X")
        self.tree.heading("cy", text="Center Y")
        self.tree.heading("r", text="Radius")
        self.tree.heading("diam_px", text="Diam (px)")
        self.tree.heading("diam_unit", text="Diam (Calibrated)")
        self.tree.heading("circ", text="Circularity")
        self.tree.heading("solidity", text="Solidity")
        self.tree.heading("score", text="Score")
        self.tree.heading("status", text="Status / Gate")

        for c in cols:
            self.tree.column(c, width=80, anchor="center")
        self.tree.column("id", width=45)

        sb = ttk.Scrollbar(parent, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        # Export row
        exp_row = ttk.Frame(parent)
        exp_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(exp_row, text="Export CSV Table", command=self._export_csv).pack(side="left", padx=2)
        ttk.Button(exp_row, text="Export JSON Summary", command=self._export_json).pack(side="left", padx=2)

    def _build_log_tab(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        self.log_text = tk.Text(parent, wrap="word", bg="#0f172a", fg="#f8fafc",
                                font=self.font_mono, insertbackground="#38bdf8")
        self.log_text.grid(row=0, column=0, sticky="nsew")

        sb = ttk.Scrollbar(parent, command=self.log_text.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=sb.set)

    # -------------------------------------------------------------------------
    # Event Bindings & Canvas Interactivity
    # -------------------------------------------------------------------------
    def _bind_events(self):
        # Viewer Canvas Events
        self.canvas_viewer.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas_viewer.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas_viewer.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas_viewer.bind("<ButtonPress-3>", self._on_pan_start)
        self.canvas_viewer.bind("<B3-Motion>", self._on_pan_drag)
        self.canvas_viewer.bind("<MouseWheel>", self._on_mousewheel)

        self.canvas_viewer.bind("<Configure>", self._on_canvas_configure)

        # Editor Canvas Events
        self.canvas_editor.bind("<ButtonPress-1>", self._on_editor_press)
        self.canvas_editor.bind("<B1-Motion>", self._on_editor_drag)
        self.canvas_editor.bind("<ButtonRelease-1>", self._on_editor_release)
        self.canvas_editor.bind("<ButtonPress-3>", self._on_pan_start)
        self.canvas_editor.bind("<B3-Motion>", self._on_pan_drag)
        self.canvas_editor.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas_editor.bind("<Configure>", self._on_canvas_configure)

        # Validation Canvas Events (for Self-Test GT drawing)
        self.canvas_val.bind("<ButtonPress-1>", self._on_val_canvas_press)
        self.canvas_val.bind("<B1-Motion>", self._on_val_canvas_drag)
        self.canvas_val.bind("<ButtonRelease-1>", self._on_val_canvas_release)
        self.canvas_val.bind("<ButtonPress-3>", self._on_pan_start)
        self.canvas_val.bind("<B3-Motion>", self._on_pan_drag)
        self.canvas_val.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas_val.bind("<Configure>", self._on_canvas_configure)

        # Notebook tab change
        self.notebook.bind("<<NotebookTabChanged>>", lambda e: self._on_tab_changed())

        # Hotkeys
        self.bind("<Control-z>", lambda e: self._on_global_undo())
        self.bind("<Control-y>", lambda e: self._on_global_redo())
        self.bind("<Delete>", lambda e: self._on_global_delete())
        self.bind("<BackSpace>", lambda e: self._on_global_delete())

    # -------------------------------------------------------------------------
    # Regime Switching & Data Discovery
    # -------------------------------------------------------------------------
    def _sync_stop_checkbox(self):
        """Pulp annotates every frame, so there is nothing to stop early; froth (~40 s a
        frame on CPU) samples frames and can stop once D32 is stable."""
        if not hasattr(self, "chk_stop"):
            return
        if self._is_froth():
            self.chk_stop.pack(anchor="w", pady=(0, 4), after=self.btn_measure_video)
        else:
            self.chk_stop.pack_forget()

    def _on_tab_changed(self):
        """Validation works on labelled Benchmark Crops and fine-tuning on video, so
        opening those tabs switches the input on the left to match."""
        if not self._suppress_tab_sync:
            want = None
            if self.notebook.select() == str(self.tab_validation):
                want = "crop"
            elif self.notebook.select() == str(self.tab_teach):
                want = "video"
            if want and self.input_type.get() != want:
                self.input_type.set(want)
                self._populate_sources()
                self._log(f"[Input] Switched to {'Benchmark Crop' if want == 'crop' else 'Video'} for this tab.")
        self._redraw_current_canvas()

    def _on_regime_changed(self):
        reg = self.regime.get()
        self.froth_roi = None
        self.scale_var.set(self.scale_manual[reg] or self._default_scale())
        self._log(f"[Regime] {'Froth: Cellpose' if reg == 'froth' else 'Pulp: Mask R-CNN'}, "
                  f"scale {self.scale_var.get():g} {'mm/px' if reg == 'froth' else 'μm/px'}")
        self._refresh_scale_info()
        self._sync_stop_checkbox()
        self._teach_on_regime()

        self._populate_sources()

    def _sync_input_type(self):
        self._populate_sources()

    def _discover_data(self):
        self._populate_sources()

    def _populate_sources(self):
        reg = self.regime.get()
        itype = self.input_type.get()
        items = []

        if itype == "video":
            # keep it in its place (step 3), above Execution, when it comes back
            self.video_scrubber_frame.pack(fill="x", pady=(0, 6), before=self.action_box)
            if reg == "pulp":
                vdir = os.path.join(PROJECT_ROOT, "samples", "pulp")
            else:
                vdir = os.path.join(PROJECT_ROOT, "samples", "froth")

            if os.path.exists(vdir):
                for f in sorted(os.listdir(vdir)):
                    if f.endswith((".avi", ".mp4", ".MP4")):
                        items.append(os.path.join(vdir, f))
        elif itype == "crop":
            self.video_scrubber_frame.pack_forget()
            cdir = os.path.join(PULP_DIR, "data", "crops", reg)
            if os.path.exists(cdir):
                for root, _, files in os.walk(cdir):
                    for f in sorted(files):
                        if f.endswith(".png") and not f.endswith(("_view.png", "_mask.png")):
                            items.append(os.path.join(root, f))
            if reg == "froth":
                # Fully labelled froth patches - listed only once their labels exist, so
                # nobody sees a detection overlay on a patch before labelling it.
                tdir = os.path.join(FROTH_DIR, "eval", "to_label")
                if os.path.isdir(tdir):
                    for f in sorted(os.listdir(tdir)):
                        if f.endswith(".png") and self._find_ground_truth_for_source(os.path.join(tdir, f)):
                            items.append(os.path.join(tdir, f))
        else:
            self.video_scrubber_frame.pack_forget()

        self.video_combo["values"] = [os.path.basename(p) for p in items]
        self._available_paths = items

        if items:
            self.video_combo.current(0)
            self._on_source_combo_selected()

    def _on_source_combo_selected(self):
        idx = self.video_combo.current()
        if idx >= 0 and idx < len(self._available_paths):
            self.selected_source_path = self._available_paths[idx]
            self.source_path_entry.delete(0, tk.END)
            self.source_path_entry.insert(0, self.selected_source_path)

            if self.input_type.get() == "video":
                self._open_video_file(self.selected_source_path)
            else:
                self._load_image_file(self.selected_source_path)

    def _browse_source_file(self):
        f = filedialog.askopenfilename(
            title="Select Video or Image File",
            filetypes=[("Media Files", "*.avi *.mp4 *.MP4 *.png *.tif *.jpg"), ("All Files", "*.*")]
        )
        if f:
            self.selected_source_path = f
            self.source_path_entry.delete(0, tk.END)
            self.source_path_entry.insert(0, f)
            if f.lower().endswith((".avi", ".mp4")):
                self.input_type.set("video")
                self._open_video_file(f)
            else:
                self.input_type.set("image")
                self._load_image_file(f)

    # -------------------------------------------------------------------------
    # Image & Video Loading
    # -------------------------------------------------------------------------
    def _clear_frame_results(self, clear_self_test=True):
        self.measured_bubbles = []
        self.metrology_summary = None
        self.froth_roi = None       # the measured region belongs to one frame as well
        self.val_predictions = []
        self.val_metrics = None
        self.editor_bubbles = []
        self.editor_original_raw = []
        self.editor_added_count = 0
        self.editor_deleted_count = 0
        self.val_draw_start = None
        self.val_draw_current = None
        if clear_self_test:
            self.val_ground_truth = []
            self.val_gt_history = []
            self.val_gt_redo_history = []
            if hasattr(self, "lbl_val_gt_count"):
                self.lbl_val_gt_count.configure(text="(0 bubbles)")
        self._update_kpis({"unit": self._unit_name(), "d32": 0.0, "d43": 0.0, "n_accepted": 0, "median_d": 0.0})
        self._update_validation_kpis({})

    def _open_video_file(self, path):
        if self.video_cap:
            self.video_cap.release()

        self.video_cap = cv2.VideoCapture(path)
        if not self.video_cap.isOpened():
            messagebox.showerror("Error", f"Could not open video file: {path}")
            return

        self._clear_frame_results(clear_self_test=True)
        self.total_video_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_scale.configure(to=max(1, self.total_video_frames - 1))
        self.current_frame_idx.set(0)
        self.frame_str_var.set(f"Frame 0 / {self.total_video_frames}")
        self._read_and_display_frame(0)
        self.is_auto_fit = True
        self.after(50, self._canvas_fit_image)

    def _load_image_file(self, path):
        if not os.path.exists(path):
            return
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            return
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.current_raw_img = Image.fromarray(img_rgb)
        self._clear_frame_results(clear_self_test=True)
        self._canvas_fit_image()

    def _read_and_display_frame(self, frame_idx):
        if not self.video_cap:
            return
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.video_cap.read()
        if ret and frame is not None:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_raw_img = Image.fromarray(img_rgb)
            self._clear_frame_results(clear_self_test=True)
            self._restore_frame_result()
            self._redraw_current_canvas()

    # -------------------------------------------------------------------------
    # Per-frame results (no contours carried from one frame to another)
    # -------------------------------------------------------------------------
    def _frame_key(self) -> Tuple[str, int]:
        idx = int(self.current_frame_idx.get()) if self.input_type.get() == "video" else 0
        return (self.selected_source_path, idx)

    def _show_frame_entry(self, entry: Dict[str, Any], summary: Optional[Dict[str, Any]]):
        """Put one frame's bubbles on screen; KPIs from `summary` if given."""
        self.measured_bubbles = [dict(b) for b in entry.get("bubbles", [])]
        self.froth_roi = entry.get("roi")
        self.editor_bubbles = [dict(b) for b in self.measured_bubbles]
        self.editor_original_raw = [dict(b) for b in self.measured_bubbles]
        self.editor_history, self.editor_redo_history = [], []
        self.editor_added_count = self.editor_deleted_count = 0
        self._recalculate_editor_live_math()
        if summary:
            self.metrology_summary = summary
            self._update_kpis(summary)
            self._plot_histogram(summary.get("distribution"))
            self._populate_table(summary.get("accepted_bubbles", []))

    def _restore_frame_result(self):
        """After a frame change: show this frame's own result, if it has one."""
        key = self._frame_key()
        entry = self.frame_results.get(key)
        vid = self.video_result[1] if (self.video_result and self.video_result[0] == key[0]) else None
        if entry:
            self._show_frame_entry(entry, entry.get("summary") or vid)
        elif vid:
            # the whole-video numbers stay on the KPI cards; no contours on an unsampled frame
            self.metrology_summary = vid
            self._update_kpis(vid)
            self.status_var.set(f"Frame {key[1]} was not one of the measured frames - no detections drawn.")

    def _on_slider_scrubbed(self, val):
        idx = int(float(val))
        self.frame_str_var.set(f"Frame {idx} / {self.total_video_frames}")
        self._read_and_display_frame(idx)

    def _step_frame(self, delta):
        new_idx = self.current_frame_idx.get() + delta
        if new_idx < 0:
            new_idx = 0
        if new_idx >= self.total_video_frames:
            new_idx = self.total_video_frames - 1
        self.current_frame_idx.set(new_idx)
        self.frame_str_var.set(f"Frame {new_idx} / {self.total_video_frames}")
        self._read_and_display_frame(new_idx)


    # -------------------------------------------------------------------------
    # Measurement Pipeline
    # -------------------------------------------------------------------------
    def _action_measure_frame(self):
        if self.current_raw_img is None:
            messagebox.showwarning("Warning", "Please select or open an image/video frame first.")
            return

        self.status_var.set("Running bubble detection...")
        self.btn_measure_frame.configure(state="disabled")
        # Read every Tk value here, on the main thread; the worker thread touches no Tk state.
        reg = self.regime.get()
        key = self._frame_key()
        frame_idx = key[1]
        weights = self.weights_path.get()
        manual = self.scale_manual[reg]
        img_np = np.array(self.current_raw_img)

        def worker():
            try:
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                if reg == "pulp":
                    if self.pulp_detector is None:
                        self._log_async("[Pulp] Loading Mask R-CNN ResNet-50-FPN-V2...")
                        self.pulp_detector = PulpDetector(weights)
                    raw_bubbles = self.pulp_detector.detect_frame(
                        img_bgr, score_thresh=PULP_CONF, frame_idx=frame_idx
                    )
                    summary = summarize_bubbles(
                        raw_bubbles, um_per_px=manual or DEFAULT_PULP_CALIB,
                        min_diam_px=PULP_MIN_D_PX, max_diam_px=PULP_MAX_D_PX,
                        apply_fov_gate=True, exclude_edge=True,
                        calibration_note=f"set by the user: {manual:g} um/px" if manual else None,
                    )
                    summary["auto_scale_per_px"] = DEFAULT_PULP_CALIB
                else:
                    det = self._get_froth_detector()
                    self.task_queue.put(("progress", (5, "Cellpose: segmenting this frame (about 30-60 s on CPU)...")))
                    raw_bubbles = det.detect_frame(img_bgr, frame_idx=frame_idx)
                    summary = summarize_bubbles(
                        raw_bubbles, um_per_px=manual or det.mm_per_px, unit_name="mm",
                        exclude_edge=True, calibration_note=(f"set by the user: {manual:g} mm/px" if manual
                                                             else f"{det.scale_source}"),
                    )
                    summary["auto_scale_per_px"] = det.mm_per_px
                    summary["mm_per_px"] = manual or det.mm_per_px
                    summary["measured_region"] = list(det.last_roi) if det.last_roi else None
                    self._log_async(f"[Froth] Cellpose took {det.last_seconds:.0f} s on region {det.last_roi}")

                summary["frame_key"] = [key[0], key[1]]
                self.task_queue.put(("measure_done", summary))
            except Exception as e:
                self.task_queue.put(("error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _get_froth_detector(self) -> FrothDetector:
        if self.froth_detector is None:
            self.froth_detector = FrothDetector(log_cb=self._log_async,
                                                model_path=teach.active_model("froth"))
        return self.froth_detector

    def _action_measure_video(self):
        if not self.selected_source_path or not self.selected_source_path.lower().endswith((".avi", ".mp4")):
            messagebox.showwarning("Warning", "Please select a video file to run full video measurement.")
            return

        self.status_var.set("Processing full video stream...")
        self.btn_measure_video.configure(state="disabled")
        reg = self.regime.get()
        weights = self.weights_path.get()
        video_path = self.selected_source_path

        stop_early = bool(self.stop_when_converged.get()) and reg == "froth"
        manual = self.scale_manual[reg]
        self._conv_reset(reg)

        def worker():
            try:
                cap = cv2.VideoCapture(video_path)
                tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if reg == "pulp":
                    cands = list(range(0, tot, PULP_FRAME_GAP))[:PULP_VIDEO_MAX]
                else:
                    # Froth at 30 fps: consecutive frames show the same bubbles, so candidates
                    # are spread evenly over the whole clip (decorrelated frames).
                    cands = sorted(set(int(round(i)) for i in
                                       np.linspace(0, max(tot - 1, 0), min(max(tot, 1), FROTH_VIDEO_MAX_FRAMES))))
                # Visit start, end, middle, quarters, ... so stopping early still covers the clip
                order = [cands[i] for i in spread_order(len(cands))]

                if reg == "pulp" and self.pulp_detector is None:
                    self.pulp_detector = PulpDetector(weights)
                det = self._get_froth_detector() if reg == "froth" else None
                if det is not None:
                    self.task_queue.put(("progress", (1, f"Cellpose: up to {len(order)} frames spread over the video (~30-60 s each)...")))

                # Pulp: every frame is annotated, but D32 uses only the decorrelated frames
                # (every 5th): at 30 fps one bubble appears in several frames running, and
                # counting it each time would make the error bar too small.
                display_only = [f for f in range(tot) if f not in set(cands)] if reg == "pulp" else []
                n_work = len(order) + len(display_only)
                tracker = ConvergenceTracker()
                all_bubbles, rois, visited = [], {}, []
                display_bubbles: Dict[str, List[Dict[str, Any]]] = {}
                g_id = 1
                for step, f_idx in enumerate(order, start=1):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        continue

                    if reg == "pulp":
                        b_list = self.pulp_detector.detect_frame(frame, score_thresh=PULP_CONF, frame_idx=f_idx)
                        fs = summarize_bubbles(b_list, um_per_px=manual or DEFAULT_PULP_CALIB,
                                               min_diam_px=PULP_MIN_D_PX, max_diam_px=PULP_MAX_D_PX,
                                               apply_fov_gate=True, exclude_edge=True)
                    else:
                        b_list = det.detect_frame(frame, frame_idx=f_idx)
                        rois[f_idx] = list(det.last_roi) if det.last_roi else None
                        fs = summarize_bubbles(b_list, um_per_px=manual or det.mm_per_px, unit_name="mm",
                                               exclude_edge=True)
                    visited.append(f_idx)

                    for b in b_list:
                        b["id"] = g_id
                        g_id += 1
                        all_bubbles.append(b)

                    # convergence uses exactly the bubbles the final D32 uses (same gates)
                    pt = tracker.add_frame([b["diameter_val"] for b in fs["accepted_bubbles"]], f_idx)
                    self.task_queue.put(("convergence_point", dict(pt)))
                    rel = f"±{pt['rel']*100:.1f}%" if math.isfinite(pt["rel"]) else "±?"
                    pct = int(100.0 * step / n_work)
                    self.task_queue.put(("progress", (pct, f"Frame {f_idx} ({step}/{n_work}): D32 {pt['d32']:.3g} "
                                                           f"{rel} from {pt['n']} bubbles")))
                    if stop_early and tracker.converged:
                        break

                # Pulp: the remaining frames, for the overlay only (same detector and gates)
                for k, f_idx in enumerate(display_only, start=1):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        continue
                    b_list = self.pulp_detector.detect_frame(frame, score_thresh=PULP_CONF, frame_idx=f_idx)
                    fs = summarize_bubbles(b_list, um_per_px=manual or DEFAULT_PULP_CALIB,
                                           min_diam_px=PULP_MIN_D_PX, max_diam_px=PULP_MAX_D_PX,
                                           apply_fov_gate=True, exclude_edge=True)
                    for b in fs["accepted_bubbles"]:
                        b["id"] = g_id
                        g_id += 1
                    display_bubbles[str(f_idx)] = fs["accepted_bubbles"]
                    step = len(order) + k
                    self.task_queue.put(("progress", (int(100.0 * step / n_work),
                                                      f"Annotating frame {f_idx} ({step}/{n_work})...")))

                cap.release()

                if reg == "pulp":
                    summary = summarize_bubbles(
                        all_bubbles, um_per_px=manual or DEFAULT_PULP_CALIB,
                        min_diam_px=PULP_MIN_D_PX, max_diam_px=PULP_MAX_D_PX,
                        apply_fov_gate=True, exclude_edge=True,
                        calibration_note=f"set by the user: {manual:g} um/px" if manual else None,
                    )
                    summary["auto_scale_per_px"] = DEFAULT_PULP_CALIB
                else:
                    summary = summarize_bubbles(
                        all_bubbles, um_per_px=manual or det.mm_per_px, unit_name="mm", exclude_edge=True,
                        calibration_note=(f"set by the user: {manual:g} mm/px" if manual
                                          else f"{det.scale_source}"),
                    )
                    summary["auto_scale_per_px"] = det.mm_per_px
                    summary["mm_per_px"] = manual or det.mm_per_px
                summary["frame_indices"] = sorted(visited)
                summary["frame_rois"] = {str(k): v for k, v in rois.items()}
                summary["video_file"] = os.path.basename(video_path)
                summary["video_path"] = video_path
                summary["frames_analyzed"] = len(visited)
                summary["frames_available"] = len(order)
                summary["frames_annotated"] = len(visited) + len(display_bubbles)
                summary["_display_bubbles"] = display_bubbles      # popped by the GUI, not exported
                summary["convergence"] = tracker.report()
                summary["convergence"]["stopped_early"] = bool(stop_early and tracker.converged
                                                               and len(visited) < len(order))
                self.task_queue.put(("measure_video_done", summary))
            except Exception as e:
                self.task_queue.put(("error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    # -------------------------------------------------------------------------
    # Validation & Benchmark
    # -------------------------------------------------------------------------
    def _action_run_validation(self):
        if self.current_raw_img is None:
            messagebox.showwarning("Warning", "Please select or open an image or video frame first.")
            return

        # Single-click execution: if detections haven't run on this frame, run them automatically first!
        if not self.measured_bubbles:
            self._pending_validation = True
            self._log("[Validation] Running detection on current frame before benchmarking...")
            self._action_measure_frame()
            return

        self._run_validation_scoring()

    def _run_validation_scoring(self):
        gt = []
        gt_file = None
        is_self_test = (len(self.val_ground_truth) > 0 and self.val_tool_mode.get() != "off")

        if is_self_test:
            gt = self.val_ground_truth
            self._log(f"[Validation] Scoring against {len(gt)} User Self-Annotation bubbles...")
        else:
            gt_file = self._find_ground_truth_for_source(self.selected_source_path)
            if gt_file and os.path.exists(gt_file):
                self._log(f"[Validation] Loading ground truth: {os.path.basename(gt_file)}")
                with open(gt_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for i, b in enumerate(data.get("bubbles", []), start=1):
                    if b.get("unsure"):
                        continue   # unsure / cut by the patch edge / below floor: not scored
                    r = float(b.get("r", b.get("d_eq_px", 0) / 2.0))
                    gt.append({
                        "id": i,
                        "cx": float(b.get("cx", 0)),
                        "cy": float(b.get("cy", 0)),
                        "r": r,
                        "diameter_px": r * 2.0,
                    })
            else:
                if self.input_type.get() == "video":
                    messagebox.showinfo("Video Mode Notice",
                                        "This video frame does not have a pre-computed ground truth file.\n\n"
                                        "To validate against ground truth:\n"
                                        "• Select '📁 Benchmark Crop' in Step 2 to test human-labelled crops, OR\n"
                                        "• Use '✏️ Stamp' in the toolbar to draw reference bubbles on this frame and benchmark against them!")
                else:
                    messagebox.showinfo("No Ground Truth Found",
                                        "No paired ground truth JSON was found for this image.\n\n"
                                        "Tip: Use '✏️ Stamp' to draw reference bubbles and benchmark against them!")
                return

        res = evaluate_predictions_vs_ground_truth(
            predictions=self.measured_bubbles,
            ground_truth=gt,
            iou_threshold=0.50,
            um_per_px=self.scale_var.get(),
            unit_name=self._unit_name(),
        )
        # Partly labelled ground truth: an unmatched detection may be a real bubble nobody
        # stamped, so precision / F1 / D32 ratio are not meaningful. Say so instead.
        if not is_self_test and gt_file and os.path.basename(gt_file) in self.partial_label_files:
            res["partial_labels"] = True
            self._log("[Validation] These labels are PARTIAL (only some bubbles stamped): "
                      "recall and IoU are valid, precision is not shown.")

        self.val_metrics = res
        self.val_predictions = res["tagged_predictions"]
        self.val_ground_truth = res["tagged_ground_truth"]

        self._update_validation_kpis(res)
        self.view_mode.set("validation")
        self._redraw_current_canvas()
        self._suppress_tab_sync = True        # keep the frame just scored (it may be a video frame)
        self.notebook.select(self.tab_validation)
        self.after(300, lambda: setattr(self, "_suppress_tab_sync", False))
        prec = "n/a (partial labels)" if res.get("partial_labels") else f"{res['precision']*100:.1f}%"
        self._log(f"[Validation] Recall: {res['recall']*100:.1f}%, Precision: {prec}, Mean IoU: {res['mean_iou']*100:.1f}%")

    def _find_ground_truth_for_source(self, path: str) -> Optional[str]:
        stem = os.path.splitext(os.path.basename(path))[0]
        for labels_dir in (os.path.join(PULP_DIR, "data", "labels"),
                           os.path.join(FROTH_DIR, "eval", "to_label", "labels")):
            for root, _, files in os.walk(labels_dir):
                for f in files:
                    if f == f"{stem}_labels.json":
                        return os.path.join(root, f)
        return None

    def _on_val_tool_changed(self):
        mode = self.val_tool_mode.get()
        if mode == "draw":
            self.status_var.set("Self-Annotation: Left-click and drag on canvas to stamp a reference bubble.")
            self.canvas_val.configure(cursor="crosshair")
        elif mode == "delete":
            self.status_var.set("Self-Annotation: Click any reference bubble to delete it.")
            self.canvas_val.configure(cursor="X_cursor")
        else:
            self.status_var.set("Self-Annotation turned off.")
            self.canvas_val.configure(cursor="")
        self._redraw_current_canvas()

    def _push_val_gt_history(self):
        self.val_gt_history.append([dict(b) for b in self.val_ground_truth])
        self.val_gt_redo_history = []

    def _val_gt_undo(self):
        if self.val_gt_history:
            self.val_gt_redo_history.append([dict(b) for b in self.val_ground_truth])
            self.val_ground_truth = self.val_gt_history.pop()
            self._update_val_gt_ui()
            self._redraw_current_canvas()
            self._log("[Self-Annotation] Undo performed.")

    def _val_gt_redo(self):
        if self.val_gt_redo_history:
            self.val_gt_history.append([dict(b) for b in self.val_ground_truth])
            self.val_ground_truth = self.val_gt_redo_history.pop()
            self._update_val_gt_ui()
            self._redraw_current_canvas()
            self._log("[Self-Annotation] Redo performed.")

    def _clear_self_test_gt(self):
        if self.val_ground_truth:
            self._push_val_gt_history()
            self.val_ground_truth = []
            self._update_val_gt_ui()
            self._redraw_current_canvas()
            self._log("[Self-Annotation] Cleared all reference bubbles.")

    def _update_val_gt_ui(self):
        if hasattr(self, "lbl_val_gt_count"):
            self.lbl_val_gt_count.configure(text=f"({len(self.val_ground_truth)} bubbles)")

    def _on_global_undo(self):
        current_tab_idx = self.notebook.index(self.notebook.select())
        if self._teach_tab_active():
            self._teach_undo()
        elif current_tab_idx == 1:
            self._editor_undo()
        elif current_tab_idx == 3:
            self._val_gt_undo()

    def _on_global_redo(self):
        current_tab_idx = self.notebook.index(self.notebook.select())
        if current_tab_idx == 1:
            self._editor_redo()
        elif current_tab_idx == 3:
            self._val_gt_redo()

    def _on_global_delete(self):
        if isinstance(self.focus_get(), (tk.Entry, ttk.Entry)):
            return      # Backspace while typing a path must not delete a bubble
        current_tab_idx = self.notebook.index(self.notebook.select())
        if current_tab_idx == 3 and self.val_ground_truth:
            self._push_val_gt_history()
            removed = self.val_ground_truth.pop()
            self._update_val_gt_ui()
            self._redraw_current_canvas()
            self._log(f"[Self-Annotation] Deleted last bubble #{removed.get('id', '')}")

    # -------------------------------------------------------------------------
    # Interactive Bubble Editor Logic
    # -------------------------------------------------------------------------
    def _on_editor_tool_changed(self):
        tool = self.editor_mode.get()
        if tool == "delete":
            self.status_var.set("Delete Mode: Click any bubble to delete it.")
        elif tool == "circle":
            self.status_var.set("Circle Mode: Left-drag to stamp a new circular bubble.")
        elif tool == "ellipse":
            self.status_var.set("Ellipse Mode: Left-drag to stamp an elliptical bubble.")
        else:
            self.status_var.set("Inspect Mode: Hover or click bubbles to inspect.")

    def _on_editor_press(self, event):
        img_x, img_y = self._canvas_to_image_coords(event.x, event.y)
        tool = self.editor_mode.get()

        if tool == "delete":
            # Find nearest bubble
            idx = self._find_bubble_at(img_x, img_y, self.editor_bubbles)
            if idx != -1:
                self._push_editor_history()
                removed = self.editor_bubbles.pop(idx)
                self.editor_deleted_count += 1
                self._recalculate_editor_live_math()
                self._redraw_current_canvas()
                self._log(f"[Editor] Deleted bubble #{removed.get('id', idx+1)} at ({img_x:.0f}, {img_y:.0f})")
        elif tool in ["circle", "ellipse"]:
            self.editor_stamp_start = (img_x, img_y)
            self.editor_stamp_current = (img_x, img_y)

    def _on_editor_drag(self, event):
        if self.editor_stamp_start:
            img_x, img_y = self._canvas_to_image_coords(event.x, event.y)
            self.editor_stamp_current = (img_x, img_y)
            self._redraw_current_canvas()

    def _on_editor_release(self, event):
        if self.editor_stamp_start and self.editor_stamp_current:
            x0, y0 = self.editor_stamp_start
            x1, y1 = self.editor_stamp_current
            r = math.hypot(x1 - x0, y1 - y0)

            if r >= 3.0:
                self._push_editor_history()
                d_px = r * 2.0
                scale = self.scale_var.get()
                new_b = {
                    "id": len(self.editor_bubbles) + 1,
                    "cx": round(x0, 1),
                    "cy": round(y0, 1),
                    "r": round(r, 1),
                    "diameter_px": round(d_px, 2),
                    "diameter_val": round(d_px * scale, 2),
                    "unit": self._unit_name(),
                    "circularity": 1.0,
                    "solidity": 1.0,
                    "score": 1.0,
                    "manually_added": True,
                }
                self.editor_bubbles.append(new_b)
                self.editor_added_count += 1
                self._recalculate_editor_live_math()
                self._log(f"[Editor] Added bubble #{new_b['id']} (d={d_px:.1f}px)")

            self.editor_stamp_start = None
            self.editor_stamp_current = None
            self._redraw_current_canvas()

    def _push_editor_history(self):
        self.editor_history.append([dict(b) for b in self.editor_bubbles])
        self.editor_redo_history = []

    def _editor_undo(self):
        if self.editor_history:
            self.editor_redo_history.append([dict(b) for b in self.editor_bubbles])
            self.editor_bubbles = self.editor_history.pop()
            self._recalculate_editor_live_math()
            self._redraw_current_canvas()
            self._log("[Editor] Undo performed.")

    def _editor_redo(self):
        if self.editor_redo_history:
            self.editor_history.append([dict(b) for b in self.editor_bubbles])
            self.editor_bubbles = self.editor_redo_history.pop()
            self._recalculate_editor_live_math()
            self._redraw_current_canvas()
            self._log("[Editor] Redo performed.")

    def _editor_reset_raw(self):
        if messagebox.askyesno("Confirm Reset", "Reset bubbles back to original raw model detections?"):
            self._push_editor_history()
            self.editor_bubbles = [dict(b) for b in self.editor_original_raw]
            self.editor_added_count = 0
            self.editor_deleted_count = 0
            self._recalculate_editor_live_math()
            self._redraw_current_canvas()
            self._log("[Editor] Reset back to raw detections.")

    def _recalculate_editor_live_math(self):
        scale = self.scale_var.get()
        unit = self._unit()
        diams = [b.get("diameter_px", b.get("r", 0) * 2.0) * scale for b in self.editor_bubbles]

        d32 = compute_sauter_d32(diams)
        self.lbl_editor_live_d32.configure(text=f"D32: {self._fmt(d32)} {unit}")
        self.lbl_editor_live_count.configure(text=f"Bubbles: {len(self.editor_bubbles)}")
        self.lbl_editor_changes.configure(text=f"({self.editor_added_count} added, {self.editor_deleted_count} deleted)")

    def _editor_finalize_results(self):
        if not self.editor_bubbles:
            messagebox.showwarning("Warning", "No bubbles to finalize.")
            return

        self.measured_bubbles = [dict(b) for b in self.editor_bubbles]
        summary = summarize_bubbles(self.measured_bubbles, um_per_px=self.scale_var.get(),
                                    unit_name=self._unit_name(), calibration_note=self._calibration_note())
        self.metrology_summary = summary
        self.frame_results[self._frame_key()] = {"bubbles": [dict(b) for b in self.measured_bubbles],
                                                 "summary": summary, "roi": self.froth_roi}

        self._update_kpis(summary)
        self._plot_histogram(summary.get("distribution"))
        self._populate_table(summary.get("accepted_bubbles", []))

        messagebox.showinfo("Results Finalized",
                            f"Successfully finalized {len(self.measured_bubbles)} bubbles!\n"
                            f"Updated Sauter Mean D32: {summary.get('d32')} {summary.get('unit')}")
        self._log(f"[Editor] Finalized {len(self.measured_bubbles)} bubbles. D32 = {summary.get('d32')}")
        self.notebook.select(self.tab_viewer)

    # -------------------------------------------------------------------------
    # Canvas Rendering & Zoom / Pan
    # -------------------------------------------------------------------------
    def _get_active_canvas(self):
        try:
            current_tab_idx = self.notebook.index(self.notebook.select())
            if current_tab_idx == 1:
                return self.canvas_editor
            elif current_tab_idx == 3:
                return self.canvas_val
        except Exception:
            pass
        return self.canvas_viewer

    def _on_canvas_configure(self, event):
        if getattr(self, "is_auto_fit", True) and self.current_raw_img is not None:
            if event.width > 20 and event.height > 20:
                self._canvas_fit_image()

    def _canvas_fit_image(self):
        if self.current_raw_img is None:
            return
        canvas = self._get_active_canvas()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw <= 20 or ch <= 20:
            self.update_idletasks()
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
            if cw <= 20 or ch <= 20:
                cw, ch = 800, 600

        iw, ih = self.current_raw_img.size
        scale_w = cw / iw
        scale_h = ch / ih
        self.zoom_level = min(scale_w, scale_h)
        self.pan_x = int((cw - iw * self.zoom_level) / 2)
        self.pan_y = int((ch - ih * self.zoom_level) / 2)
        self.is_auto_fit = True
        self._redraw_current_canvas()

    def _canvas_zoom(self, factor, center_x=None, center_y=None):
        if self.current_raw_img is None:
            return
        self.is_auto_fit = False
        canvas = self._get_active_canvas()
        cw = canvas.winfo_width() or 800
        ch = canvas.winfo_height() or 600

        if center_x is None:
            center_x = cw / 2.0
        if center_y is None:
            center_y = ch / 2.0

        ix = (center_x - self.pan_x) / self.zoom_level
        iy = (center_y - self.pan_y) / self.zoom_level

        new_zoom = max(0.05, min(20.0, self.zoom_level * factor))
        self.zoom_level = new_zoom

        self.pan_x = int(center_x - ix * new_zoom)
        self.pan_y = int(center_y - iy * new_zoom)
        self._redraw_current_canvas()

    def _on_mousewheel(self, event):
        factor = 1.15 if event.delta > 0 else 0.85
        self._canvas_zoom(factor, center_x=event.x, center_y=event.y)

    def _on_pan_start(self, event):
        self._drag_start = (event.x, event.y)

    def _on_pan_drag(self, event):
        if self._drag_start:
            dx = event.x - self._drag_start[0]
            dy = event.y - self._drag_start[1]
            if abs(dx) > 1 or abs(dy) > 1:
                self.is_auto_fit = False
            self.pan_x += dx
            self.pan_y += dy
            self._drag_start = (event.x, event.y)
            self._redraw_current_canvas()

    def _canvas_to_image_coords(self, cx, cy):
        ix = (cx - self.pan_x) / self.zoom_level
        iy = (cy - self.pan_y) / self.zoom_level
        return ix, iy

    def _image_to_canvas_coords(self, ix, iy):
        cx = ix * self.zoom_level + self.pan_x
        cy = iy * self.zoom_level + self.pan_y
        return cx, cy

    def _find_bubble_at(self, ix, iy, bubbles):
        for idx, b in enumerate(bubbles):
            d = math.hypot(b["cx"] - ix, b["cy"] - iy)
            if d <= b["r"] + 3:
                return idx
        return -1

    def _redraw_current_canvas(self):
        if self.current_raw_img is None:
            return

        current_tab_idx = self.notebook.index(self.notebook.select())

        if current_tab_idx == 1:
            # Editor tab
            canvas = self.canvas_editor
            bubbles_to_draw = self.editor_bubbles
            is_editor = True
            is_val = False
        elif current_tab_idx == 3:
            # Validation tab
            canvas = self.canvas_val
            bubbles_to_draw = self.val_predictions
            is_editor = False
            is_val = True
        else:
            # Viewer tab
            canvas = self.canvas_viewer
            bubbles_to_draw = self.measured_bubbles
            is_editor = False
            is_val = self.view_mode.get() == "validation"

        # Generate composited image
        rendered_img = self.current_raw_img.copy()
        draw = ImageDraw.Draw(rendered_img)

        # Only bubbles that belong to the frame on screen (a video run holds many frames)
        if self.input_type.get() == "video":
            cur = int(self.current_frame_idx.get())
            bubbles_to_draw = [b for b in bubbles_to_draw if int(b.get("frame_idx", cur)) == cur]

        # Draw overlays if mode permits
        if is_editor or self.view_mode.get() in ["overlay", "validation"]:
            for b in bubbles_to_draw:
                cx, cy, r = b["cx"], b["cy"], b["r"]

                if is_val:
                    status = b.get("validation_status", "TP")
                    if status == "TP":
                        color = "#10b981"  # Green True Positive
                    elif status == "FP":
                        color = "#ef4444"  # Red False Positive
                    else:
                        color = "#00e5ff"
                elif b.get("manually_added"):
                    color = "#10b981"      # Green for user added
                else:
                    color = "#00e5ff"      # Cyan for model bubbles

                outline = b.get("outline")
                if outline and len(outline) >= 3:
                    # Froth: Cellpose's own outline (packed bubbles are polygons, not circles)
                    draw.polygon([tuple(p) for p in outline], outline=color, width=1)
                else:
                    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=2)
                    draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill="#ffffff")

                if self.show_labels.get() and not is_val:
                    dv = b.get('diameter_val', b.get('diameter_px', 0))
                    d_tag = f"{dv:.2f}" if self._is_froth() else f"{dv:.0f}"
                    draw.text((cx - r, max(0, cy - r - 10)), d_tag, fill="#ffff00")

        # A video frame with no result: say so (drawn on the canvas below, readable at any zoom),
        # so an empty frame is not read as "no bubbles here"
        unmeasured_msg = None
        if (self.input_type.get() == "video" and not is_editor and self._frame_key() not in self.frame_results
                and self.view_mode.get() != "raw"):
            unmeasured_msg = ("This frame has not been measured yet - measuring now..."
                              if str(self.btn_measure_frame["state"]) == "disabled" else
                              "This frame has not been measured. Press Measure Current Frame.")

        # Froth: show the region that was measured (in-focus band of the column)
        if self._is_froth() and self.froth_roi and (is_editor or self.view_mode.get() in ["overlay", "validation"]):
            x0, y0, x1, y1 = self.froth_roi
            iw0, ih0 = rendered_img.size
            if (x1 - x0) < iw0 or (y1 - y0) < ih0:
                draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline="#f97316", width=3)
                draw.text((x0 + 6, y0 + 4), "measured region (in focus)", fill="#f97316")

        # Draw Ground Truth in Validation Mode
        if is_val and self.val_ground_truth:
            for g in self.val_ground_truth:
                cx, cy, r = g["cx"], g["cy"], g["r"]
                # Yellow for False Negative / Ground Truth
                status = g.get("validation_status")
                col = "#f59e0b" if status == "FN" else "#10b981"
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=2)

        # Draw rubber-band circle stamp during dragging
        if is_editor and self.editor_stamp_start and self.editor_stamp_current:
            x0, y0 = self.editor_stamp_start
            x1, y1 = self.editor_stamp_current
            r = math.hypot(x1 - x0, y1 - y0)
            draw.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], outline="#f59e0b", width=2)

        if is_val and self.val_tool_mode.get() == "draw" and self.val_draw_start and self.val_draw_current:
            x0, y0 = self.val_draw_start
            x1, y1 = self.val_draw_current
            r = math.hypot(x1 - x0, y1 - y0)
            draw.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], outline="#f59e0b", width=2)
            draw.ellipse([x0 - 2, y0 - 2, x0 + 2, y0 + 2], fill="#f59e0b")

        # Resize for zoom
        iw, ih = rendered_img.size
        nw = max(10, int(iw * self.zoom_level))
        nh = max(10, int(ih * self.zoom_level))
        scaled_img = rendered_img.resize((nw, nh), Image.Resampling.BILINEAR)

        self.photo = ImageTk.PhotoImage(scaled_img)
        canvas.delete("all")
        canvas.create_image(self.pan_x, self.pan_y, anchor="nw", image=self.photo)
        if unmeasured_msg:
            t = canvas.create_text(14, 12, anchor="nw", text=unmeasured_msg, fill="#fbbf24",
                                   font=("Segoe UI", 10, "bold"))
            x0, y0, x1, y1 = canvas.bbox(t)
            r = canvas.create_rectangle(x0 - 8, y0 - 5, x1 + 8, y1 + 5, fill="#111827", outline="#fbbf24")
            canvas.tag_raise(t, r)

    def _on_canvas_press(self, event):
        pass

    def _on_canvas_drag(self, event):
        pass

    def _on_canvas_release(self, event):
        pass

    def _on_val_canvas_press(self, event):
        tool = self.val_tool_mode.get()
        if tool == "off":
            return
        img_x, img_y = self._canvas_to_image_coords(event.x, event.y)
        if tool == "delete":
            self._delete_val_bubble_at(img_x, img_y)
        elif tool == "draw":
            self.val_draw_start = (img_x, img_y)
            self.val_draw_current = (img_x, img_y)

    def _on_val_canvas_drag(self, event):
        tool = self.val_tool_mode.get()
        if tool == "draw" and self.val_draw_start:
            img_x, img_y = self._canvas_to_image_coords(event.x, event.y)
            self.val_draw_current = (img_x, img_y)
            self._redraw_current_canvas()

    def _on_val_canvas_release(self, event):
        tool = self.val_tool_mode.get()
        if tool == "draw" and self.val_draw_start:
            img_x, img_y = self._canvas_to_image_coords(event.x, event.y)
            r = math.hypot(img_x - self.val_draw_start[0], img_y - self.val_draw_start[1])
            if r >= 3.0:
                self._push_val_gt_history()
                self.val_ground_truth.append({
                    "id": len(self.val_ground_truth) + 1,
                    "cx": round(self.val_draw_start[0], 1),
                    "cy": round(self.val_draw_start[1], 1),
                    "r": round(r, 1),
                    "diameter_px": round(r * 2.0, 2),
                    "validation_status": "FN",
                })
                self._update_val_gt_ui()
                self._log(f"[Self-Annotation] Added bubble #{len(self.val_ground_truth)} (d={r*2:.1f}px)")
            self.val_draw_start = None
            self.val_draw_current = None
            self._redraw_current_canvas()

    def _delete_val_bubble_at(self, img_x, img_y):
        idx = self._find_bubble_at(img_x, img_y, self.val_ground_truth)
        if idx >= 0:
            self._push_val_gt_history()
            removed = self.val_ground_truth.pop(idx)
            self._update_val_gt_ui()
            self._redraw_current_canvas()
            self._log(f"[Self-Annotation] Deleted bubble #{removed.get('id', idx+1)}")

    # -------------------------------------------------------------------------
    # KPI Cards & Matplotlib Plot Updates
    # -------------------------------------------------------------------------
    def _update_kpis(self, summary: Dict[str, Any]):
        unit = {"um": "μm"}.get(summary.get("unit", "um"), summary.get("unit", "um"))
        d32 = summary.get("d32", 0.0)
        d43 = summary.get("d43", 0.0)
        cnt = summary.get("n_accepted", 0)
        d50 = summary.get("median_d", 0.0)
        rej = summary.get("n_rejected", 0)

        f = (lambda v: f"{v:.2f}") if unit == "mm" else (lambda v: f"{v:.1f}")
        self.kpi_labels["d32"][0].configure(text=f"{f(d32)} {unit}")
        self.kpi_labels["d43"][0].configure(text=f"{f(d43)} {unit}")
        self.kpi_labels["count"][0].configure(text=str(cnt))
        edge = summary.get("n_edge_excluded", 0)
        self.kpi_labels["count"][1].configure(text=f"{edge} cut by the region edge, excluded" if edge else f"{rej} excluded")
        self.kpi_labels["d50"][0].configure(text=f"{f(d50)} {unit}")
        span = (summary.get("distribution") or {}).get("span")
        self.kpi_labels["d50"][1].configure(text=f"Span (d90-d10)/d50: {span:.2f}" if span else "Span: -")
        warn = summary.get("sample_size_warning")
        conv = summary.get("convergence")
        if conv and conv.get("history") and math.isfinite(conv.get("rel_half_width", float("inf"))):
            state = "stable" if conv.get("converged") else "not yet stable"
            self.kpi_labels["d32"][1].configure(text=f"±{conv['rel_half_width']*100:.1f}% (95%), {state}")
        else:
            self.kpi_labels["d32"][1].configure(text="Few bubbles: pool more frames" if warn else "Volume-weighted mean")

    def _update_validation_kpis(self, metrics: Dict[str, Any]):
        if not metrics or "recall" not in metrics:
            for k in self.val_kpi_labels:
                self.val_kpi_labels[k][0].configure(text="--")
                self.val_kpi_labels[k][1].configure(text="--")
            return

        self.val_kpi_labels["recall"][0].configure(text=f"{metrics['recall']*100:.1f}%")
        self.val_kpi_labels["recall"][1].configure(text=f"{metrics['tp_count']} / {metrics['gt_count']} recovered")

        if metrics.get("partial_labels"):
            for k in ("precision", "f1", "d32_error"):
                self.val_kpi_labels[k][0].configure(text="n/a")
                self.val_kpi_labels[k][1].configure(text="partial labels")
            self.val_kpi_labels["iou"][0].configure(text=f"{metrics['mean_iou']*100:.1f}%")
            return
        self.val_kpi_labels["precision"][0].configure(text=f"{metrics['precision']*100:.1f}%")
        self.val_kpi_labels["precision"][1].configure(text=f"{metrics['tp_count']} / {metrics['pred_count']} accurate")

        self.val_kpi_labels["f1"][0].configure(text=f"{metrics['f1']:.3f}")
        self.val_kpi_labels["iou"][0].configure(text=f"{metrics['mean_iou']*100:.1f}%")

        self.val_kpi_labels["d32_error"][0].configure(text=f"{metrics['d32_ratio']*100:.1f}%")
        self.val_kpi_labels["d32_error"][1].configure(text=f"Error: {metrics['d32_error_pct']:.1f}%")

    def _plot_histogram(self, psd: Optional[Dict[str, Any]]):
        if not psd or not psd.get("bins"):
            return

        self.ax_psd.clear()
        if getattr(self, "ax_psd_twin", None) is not None:
            self.ax_psd_twin.remove()          # otherwise every run stacks another Q3 axis
            self.ax_psd_twin = None
        bins = psd["bins"]
        counts = psd["counts"]
        cum = psd.get("cumulative_pct", [])
        unit = self._unit()

        # Frequency Bars
        x = np.arange(len(bins))
        width = 0.65
        bars = self.ax_psd.bar(x, counts, width, color="#0284c7", alpha=0.75, label="Bubble Count", edgecolor="#0369a1")

        self.ax_psd.set_xlabel(f"Equivalent Diameter ({unit})", fontsize=10, fontweight="bold", color="#334155")
        self.ax_psd.set_ylabel("Frequency (Count)", fontsize=10, fontweight="bold", color="#0284c7")
        self.ax_psd.set_xticks(x)
        self.ax_psd.set_xticklabels([f"{b:.2f}" if unit == "mm" else f"{b:.0f}" for b in bins], rotation=35, fontsize=8)
        self.ax_psd.grid(axis="y", linestyle="--", alpha=0.4)

        # Cumulative line on twin axis
        if cum:
            ax2 = self.ax_psd_twin = self.ax_psd.twinx()
            ax2.plot(x, cum, color="#6366f1", linewidth=2, marker="o", markersize=4, label="Cumulative Q3 (%)")
            ax2.set_ylabel("Cumulative Volume Q3 (%)", fontsize=10, fontweight="bold", color="#6366f1")
            ax2.set_ylim(0, 105)

        self.fig_psd.tight_layout()
        self.canvas_psd.draw()

    def _populate_table(self, bubbles: List[Dict[str, Any]]):
        for item in self.tree.get_children():
            self.tree.delete(item)

        unit = self._unit()
        for b in bubbles:
            self.tree.insert("", "end", values=(
                b.get("id"),
                f"{b.get('cx', 0):.1f}",
                f"{b.get('cy', 0):.1f}",
                f"{b.get('r', 0):.1f}",
                f"{b.get('diameter_px', 0):.1f}",
                f"{self._fmt(b.get('diameter_val', b.get('diameter_px', 0)))} {unit}",
                f"{b.get('circularity', 1.0):.2f}",
                f"{b.get('solidity', 1.0):.2f}",
                f"{b.get('score', 1.0):.2f}",
                b.get("rejection_reason", "Accepted"),
            ))

    def _export_csv(self):
        rows = (self.metrology_summary or {}).get("accepted_bubbles") or self.measured_bubbles
        if not rows:
            messagebox.showinfo("Export", "No bubble measurements to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            initialfile=f"bubble_measurements_{self.regime.get()}.csv"
        )
        if path:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "frame", "cx", "cy", "r", "diameter_px", "diameter_calibrated", "unit", "circularity", "solidity", "score"])
                for b in rows:
                    writer.writerow([
                        b.get("id"), b.get("frame_idx", ""), b.get("cx"), b.get("cy"), b.get("r"),
                        b.get("diameter_px"), b.get("diameter_val", b.get("diameter_px")),
                        b.get("unit", "px"), b.get("circularity", ""), b.get("solidity", ""), b.get("score", "")
                    ])
            self._log(f"[Export] Saved CSV to {path}")
            messagebox.showinfo("Export Complete", f"Exported {len(rows)} bubbles to:\n{path}")

    def _export_json(self):
        if not self.metrology_summary:
            messagebox.showinfo("Export", "No summary data to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            initialfile=f"metrology_summary_{self.regime.get()}.json"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.metrology_summary, f, indent=2)
            self._log(f"[Export] Saved JSON to {path}")
            messagebox.showinfo("Export Complete", f"Exported metrology report to:\n{path}")

    def _log(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{ts}] {text}\n")
        self.log_text.see(tk.END)

    def _check_async_queue(self):
        try:
            while True:
                msg_type, data = self.task_queue.get_nowait()
                if msg_type == "log":
                    self._log(data)
                    continue
                if (msg_type in ("measure_done", "measure_video_done") and self._is_froth() and self.froth_detector
                        and not self.scale_manual["froth"]):
                    self.scale_var.set(self.froth_detector.mm_per_px)
                    self._refresh_scale_info()
                if msg_type == "measure_done":
                    self.btn_measure_frame.configure(state="normal")
                    self.progress_bar["value"] = 100
                    key = tuple(data.get("frame_key") or self._frame_key())
                    entry = {"bubbles": data.get("accepted_bubbles", []), "summary": data,
                             "roi": data.get("measured_region")}
                    self.frame_results[key] = entry
                    if key != self._frame_key():
                        # the user moved on while this frame was running: keep the result
                        # for that frame, draw nothing here
                        self.status_var.set(f"Frame {key[1]} finished ({len(entry['bubbles'])} bubbles) - "
                                            f"shown when you go back to it.")
                        self._log(f"[Measure] Frame {key[1]} finished after you moved to another frame; "
                                  f"result kept for frame {key[1]} only.")
                        self._pending_validation = False
                        continue
                    self.status_var.set("Measurement Complete!")
                    self._show_frame_entry(entry, data)
                    self._redraw_current_canvas()
                    self._log(f"[Measure] Complete: {len(self.measured_bubbles)} bubbles. D32 = {data.get('d32')} {data.get('unit')}")
                    self._log(f"[Measure] Edge rule {'on' if data.get('edge_rule', True) else 'off'}: "
                              f"{data.get('n_edge_excluded', 0)} bubble(s) cut by the image border excluded")

                    if getattr(self, "_pending_validation", False):
                        self._pending_validation = False
                        self.after(50, self._run_validation_scoring)

                elif msg_type == "convergence_point":
                    self._conv_add_point(data)

                elif msg_type == "measure_video_done":
                    self.btn_measure_video.configure(state="normal")
                    self.status_var.set("Video Measurement Complete!")
                    self.progress_bar["value"] = 100
                    vpath = data.get("video_path", self.selected_source_path)
                    self.video_result = (vpath, data)
                    # store each measured frame's own bubbles; draw only the current frame's
                    rois = data.get("frame_rois", {})
                    by_frame: Dict[int, List[Dict[str, Any]]] = {f: [] for f in data.get("frame_indices", [])}
                    for b in data.get("accepted_bubbles", []):
                        by_frame.setdefault(int(b.get("frame_idx", -1)), []).append(b)
                    for f_idx, bl in by_frame.items():
                        self.frame_results[(vpath, f_idx)] = {"bubbles": bl, "summary": None,
                                                              "roi": rois.get(str(f_idx))}
                    for f_s, bl in data.pop("_display_bubbles", {}).items():
                        self.frame_results[(vpath, int(f_s))] = {"bubbles": bl, "summary": None, "roi": None}
                    self.metrology_summary = data
                    self._update_kpis(data)
                    self._plot_histogram(data.get("distribution"))
                    self._populate_table(data.get("accepted_bubbles", []))
                    if self._frame_key() in self.frame_results:
                        self._show_frame_entry(self.frame_results[self._frame_key()], None)
                    self._redraw_current_canvas()
                    conv = data.get("convergence", {})
                    self._conv_finish(conv, data)
                    verdict = self._conv_verdict_text(conv, data)
                    self._log(f"[Video Measure] Analyzed {data.get('frames_analyzed')} frames. Aggregated D32 = "
                              f"{data.get('d32')} {data.get('unit')}. {verdict}")
                    if data.get("unit") == "mm":
                        where = (f"Contours are drawn on the {data.get('frames_analyzed')} measured frames "
                                 f"(froth takes ~40 s a frame, so it samples the clip).")
                    else:
                        where = (f"All {data.get('frames_annotated')} frames have contours. D32 uses every "
                                 f"{PULP_FRAME_GAP}th frame ({data.get('frames_analyzed')} frames), because one "
                                 f"bubble appears in several frames in a row.")
                    messagebox.showinfo("Video Measure Complete",
                                        f"Frames used for D32: {data.get('frames_analyzed')} of {data.get('frames_available')}\n"
                                        f"Bubbles measured: {data.get('n_accepted')}\n"
                                        f"Sauter mean D32: {data.get('d32')} {data.get('unit')}\n"
                                        f"De Brouckere D43: {data.get('d43')} {data.get('unit')}\n\n"
                                        f"{verdict}\n\n"
                                        f"Edge rule: {data.get('n_edge_excluded', 0)} bubble(s) cut by the image border excluded.\n"
                                        f"{where}")

                elif msg_type.startswith("teach_"):
                    self._teach_on_message(msg_type, data)

                elif msg_type == "progress":
                    pct, text = data
                    self.progress_bar["value"] = pct
                    self.status_var.set(text)

                elif msg_type == "error":
                    self.btn_measure_frame.configure(state="normal")
                    self.btn_measure_video.configure(state="normal")
                    self.status_var.set("Error occurred")
                    self._log(f"[ERROR] {data}")
                    messagebox.showerror("Execution Error", data)

        except queue.Empty:
            pass
        finally:
            try:
                if self.winfo_exists():
                    self.after(100, self._check_async_queue)
            except tk.TclError:
                pass


if __name__ == "__main__":
    app = BubbleMetrixApp()
    app.mainloop()
