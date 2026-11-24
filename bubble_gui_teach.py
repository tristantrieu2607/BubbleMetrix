"""bubble_gui_teach.py - the two tabs added to BubbleMetrix on 2026-11-20.

  ConvergenceTabMixin  "D32 Convergence": running D32 with its 95% band against the
                       number of bubbles measured, filled live by Measure Full Video,
                       and a plain verdict (stable / not yet, and what is still needed).

  TeachTabMixin        "Fine-tune Model": fine-tuning for people who should not have to
                       think about fine-tuning. The current model is kept and adjusted with
                       a few corrected frames of the new footage; before that, step 3 scores
                       the current model on those same frames, so the user sees whether
                       fine-tuning is needed at all.
                         1. Add frames from the video picked on the left (spread over the
                            clip, decorrelated). The model in use pre-labels them.
                         2. Fix the labels (drag = add, right-click = delete) and tick
                            "Every bubble here is marked".
                         3. Read the current model's score on the checked frames.
                         4. Press Fine-tune. Training runs in a separate process; afterwards the
                            old and new model are compared on frames held back from training
                            and on the frozen benchmark, and the new model is only switched
                            in if it is better (engines/teach.decide). One click goes back.

Both are mixins on BubbleMetrixApp (bubble_gui.py), so they share its queue, log,
detectors and status bar. Worker threads never touch Tk; they post to task_queue.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

import teach

HERE = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# D32 convergence tab
# =============================================================================
class ConvergenceTabMixin:
    def _build_convergence_tab(self, parent):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        parent.rowconfigure(2, weight=1)
        parent.columnconfigure(0, weight=1)
        self.conv_verdict = ttk.Label(parent, text="Run  Measure Full Video  to see when D32 stops changing.",
                                      font=("Segoe UI", 11, "bold"), foreground="#1e293b",
                                      wraplength=900, justify="left")
        self.conv_verdict.grid(row=0, column=0, sticky="w", pady=(2, 0))
        self.conv_detail = ttk.Label(parent, foreground="#475569", wraplength=900, justify="left",
                                     text="D32 counts as stable when its 95% confidence interval is within "
                                          "±5% of the value, from at least 3 frames, at two frames in a row. "
                                          "The interval allows for frame-to-frame differences, not just bubble "
                                          "count, so a few frames with many bubbles are not over-trusted.")
        self.conv_detail.grid(row=1, column=0, sticky="w", pady=(2, 6))

        self.fig_conv, self.ax_conv = plt.subplots(figsize=(8, 4.6), dpi=100)
        self.fig_conv.patch.set_facecolor("#f8fafc")
        self.canvas_conv = FigureCanvasTkAgg(self.fig_conv, master=parent)
        self.canvas_conv.get_tk_widget().grid(row=2, column=0, sticky="nsew")
        self.conv_points: List[Dict[str, Any]] = []
        self.conv_final: Optional[Dict[str, Any]] = None
        self.conv_unit = "μm"
        self._conv_draw()

    def _conv_reset(self, regime: str):
        self.conv_points = []
        self.conv_final = None
        self.conv_unit = "mm" if regime == "froth" else "μm"
        self.conv_verdict.configure(text="Measuring... D32 and its 95% band update after every frame.",
                                    foreground="#1e293b")
        self._conv_draw()

    def _conv_add_point(self, pt: Dict[str, Any]):
        self.conv_points.append(pt)
        if math.isfinite(pt.get("rel", float("inf"))):
            self.conv_verdict.configure(
                text=f"So far: D32 {self._conv_fmt(pt['d32'])} {self.conv_unit} ±{pt['rel']*100:.1f}% "
                     f"from {pt['n']} bubbles in {pt['k']} frames"
                     + ("  -  stable" if pt.get("converged") else ""))
        self._conv_draw()

    def _conv_finish(self, conv: Dict[str, Any], summary: Dict[str, Any]):
        self.conv_final = conv
        if conv.get("history"):
            self.conv_points = [dict(h, converged=False) for h in conv["history"]]
        ok = conv.get("converged")
        self.conv_verdict.configure(text=self._conv_verdict_text(conv, summary),
                                    foreground="#047857" if ok else "#b45309")
        self._conv_draw()

    def _conv_fmt(self, v: float) -> str:
        return f"{v:.2f}" if self.conv_unit == "mm" else f"{v:.1f}"

    def _conv_verdict_text(self, conv: Dict[str, Any], summary: Dict[str, Any]) -> str:
        if not conv or not conv.get("n"):
            return "No bubbles were measured, so there is no D32."
        unit = "mm" if summary.get("unit") == "mm" else "μm"
        f = (lambda v: f"{v:.2f}") if unit == "mm" else (lambda v: f"{v:.1f}")
        rel = conv.get("rel_half_width", float("inf"))
        rel_s = f"±{rel*100:.1f}%" if math.isfinite(rel) else "(interval needs at least 2 frames)"
        if conv.get("converged"):
            txt = (f"D32 is stable: {f(conv['d32'])} {unit} {rel_s} (95% confidence). It settled after "
                   f"{conv.get('converged_after_bubbles')} bubbles from {conv.get('converged_after_frames')} frames.")
            if conv.get("stopped_early"):
                txt += (f" Measurement stopped there ({summary.get('frames_analyzed')} of "
                        f"{summary.get('frames_available')} frames); more frames would not move D32 by more than 5%.")
            return txt
        txt = (f"D32 is not yet stable: {f(conv['d32'])} {unit} {rel_s} from {conv['n']} bubbles in "
               f"{conv['frames']} frames (target ±{conv.get('target_rel', 0.05)*100:.0f}%).")
        need = conv.get("estimated_bubbles_needed")
        if need:
            txt += (f" Roughly {need} bubbles (~{conv.get('estimated_frames_needed')} frames like these) "
                    f"are needed. Pool another clip of the same condition.")
        return txt

    def _conv_draw(self):
        ax = self.ax_conv
        ax.clear()
        pts = [p for p in self.conv_points if p.get("n")]
        unit = self.conv_unit
        ax.set_xlabel("Bubbles measured", fontsize=10, color="#334155")
        ax.set_ylabel(f"D32 so far ({unit})", fontsize=10, color="#334155")
        ax.grid(linestyle="--", alpha=0.35)
        if pts:
            n = np.array([p["n"] for p in pts], float)
            d = np.array([p["d32"] for p in pts], float)
            lo = np.array([p["lo"] if math.isfinite(p.get("rel", float("inf"))) else np.nan for p in pts])
            hi = np.array([p["hi"] if math.isfinite(p.get("rel", float("inf"))) else np.nan for p in pts])
            ax.fill_between(n, lo, hi, color="#0284c7", alpha=0.18, label="95% interval", step=None)
            ax.plot(n, d, color="#0369a1", marker="o", markersize=4, linewidth=1.6, label="D32 so far")
            ref = d[-1]
            ax.axhspan(ref * 0.95, ref * 1.05, facecolor="none", edgecolor="#64748b", linestyle=":",
                       linewidth=1, label="±5% of current D32")
            fin = self.conv_final or {}
            if fin.get("converged_after_bubbles"):
                ax.axvline(fin["converged_after_bubbles"], color="#047857", linestyle="--", linewidth=1.2,
                           label=f"stable after {fin['converged_after_bubbles']} bubbles")
            for p in pts:
                ax.annotate(str(p["k"]), (p["n"], p["d32"]), textcoords="offset points", xytext=(0, 6),
                            fontsize=7, color="#64748b", ha="center")
            finite = np.isfinite(lo) & np.isfinite(hi)
            if finite.any():
                span_lo = min(np.nanmin(lo[finite]), d.min())
                span_hi = max(np.nanmax(hi[finite]), d.max())
                pad = 0.05 * (span_hi - span_lo + 1e-9)
                ax.set_ylim(max(0, span_lo - pad), span_hi + pad)
            ax.set_xlim(0, n.max() * 1.05 + 1)
            ax.legend(fontsize=8, loc="upper right")
            ax.set_title("Numbers above the points: frames measured", fontsize=8, color="#64748b", loc="left")
        else:
            ax.text(0.5, 0.5, "No measurement yet", ha="center", va="center", color="#94a3b8",
                    transform=ax.transAxes)
        self.fig_conv.tight_layout()
        self.canvas_conv.draw_idle()


# =============================================================================
# Fine-tune Model tab
# =============================================================================
class TeachTabMixin:
    # ---------------------------------------------------------------- state
    def _init_teach_state(self):
        self.teach_set: Optional[teach.TeachingSet] = None
        self.teach_idx = 0
        self.teach_labels: List[Dict[str, Any]] = []
        self.teach_undo_stack: List[List[Dict[str, Any]]] = []
        self.teach_tool = tk.StringVar(value="add")
        self.teach_zoom = None          # None = fit to canvas
        self.teach_pan = [0.0, 0.0]
        self.teach_drag = None
        self.teach_pan_start = None
        self.teach_img: Optional[Image.Image] = None
        self.teach_photo = None
        self.teach_proc: Optional[subprocess.Popen] = None
        self.teach_busy = False         # adding / pre-labelling frames
        self.teach_running_regime: Optional[str] = None
        self.teach_model_var = tk.StringVar(value="")
        self.teach_item_var = tk.StringVar(value="No teaching set yet.")
        self.teach_ready_var = tk.StringVar(value="")
        self.teach_status_var = tk.StringVar(value="")
        self.teach_current_var = tk.StringVar(value="Check some frames first; the current model is then scored on them.")
        self.teach_current_score: Optional[Dict[str, Any]] = None

    def _teach_regime(self) -> str:
        return self.regime.get()

    def _teach_tab_active(self) -> bool:
        try:
            return self.notebook.select() == str(self.tab_teach)
        except Exception:
            return False

    # ---------------------------------------------------------------- layout
    def _build_teach_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        head = ttk.Frame(parent)
        head.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Label(head, text="Model in use:", font=self.font_bold).pack(side="left")
        ttk.Label(head, textvariable=self.teach_model_var, foreground="#1976d2",
                  font=self.font_bold).pack(side="left", padx=(4, 12))
        self.btn_teach_revert = ttk.Button(head, text="↺ Go back to the base model", command=self._teach_revert)
        self.btn_teach_revert.pack(side="left")
        ttk.Button(head, text="Start a new set of frames", command=self._teach_new_set).pack(side="right")

        steps = ttk.Frame(parent)
        steps.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        for c in range(4):
            steps.columnconfigure(c, weight=1, uniform="steps")

        s1 = ttk.LabelFrame(steps, text="① Add footage", padding=6)
        s1.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        self.btn_teach_add = ttk.Button(s1, text="➕ Add frames from this video",
                                        style="Primary.TButton", command=self._teach_add_frames)
        self.btn_teach_add.pack(fill="x")
        ttk.Label(s1, foreground="#64748b", wraplength=220, justify="left",
                  text="Uses the video picked on the left. Frames are spread over the clip; press again for "
                       "more, or pick another video of the new conditions and press again.").pack(anchor="w", pady=(4, 0))

        s2 = ttk.LabelFrame(steps, text="② Fix the labels", padding=6)
        s2.grid(row=0, column=1, sticky="nsew", padx=3)
        nav = ttk.Frame(s2)
        nav.pack(fill="x")
        ttk.Button(nav, text="◀", width=3, command=lambda: self._teach_goto(self.teach_idx - 1)).pack(side="left")
        ttk.Button(nav, text="▶", width=3, command=lambda: self._teach_goto(self.teach_idx + 1)).pack(side="left", padx=(2, 6))
        ttk.Button(nav, text="Next unchecked ⏭", command=self._teach_next_unchecked).pack(side="left")
        ttk.Label(s2, textvariable=self.teach_item_var, font=self.font_mono, foreground="#334155",
                  wraplength=230, justify="left").pack(anchor="w", pady=(4, 2))
        tools = ttk.Frame(s2)
        tools.pack(fill="x")
        ttk.Radiobutton(tools, text="Add (drag)", value="add", variable=self.teach_tool).pack(side="left")
        ttk.Radiobutton(tools, text="Delete (click)", value="delete", variable=self.teach_tool).pack(side="left", padx=4)
        ttk.Button(tools, text="↩ Undo", command=self._teach_undo).pack(side="left", padx=(4, 0))
        self.btn_teach_done = ttk.Button(s2, text="✔ Every bubble here is marked", command=self._teach_toggle_done)
        self.btn_teach_done.pack(fill="x", pady=(4, 0))

        s3c = ttk.LabelFrame(steps, text="③ How good is the current model?", padding=6)
        s3c.grid(row=0, column=2, sticky="nsew", padx=3)
        ttk.Label(s3c, textvariable=self.teach_current_var, wraplength=230, justify="left",
                  foreground="#334155").pack(anchor="w")
        ttk.Button(s3c, text="Re-check", command=self._teach_recheck_current).pack(anchor="w", pady=(4, 0))

        s3 = ttk.LabelFrame(steps, text="④ Fine-tune", padding=6)
        s3.grid(row=0, column=3, sticky="nsew", padx=(3, 0))
        brow = ttk.Frame(s3)
        brow.pack(fill="x")
        self.btn_teach_go = ttk.Button(brow, text="🧠 Fine-tune the model", style="Success.TButton",
                                       command=self._teach_start_training)
        self.btn_teach_go.pack(side="left", fill="x", expand=True)
        self.btn_teach_cancel = ttk.Button(brow, text="Stop", style="Warn.TButton", command=self._teach_cancel)
        self.btn_teach_cancel.pack(side="left", padx=(4, 0))
        ttk.Label(s3, textvariable=self.teach_ready_var, wraplength=230, justify="left",
                  foreground="#334155").pack(anchor="w", pady=(4, 0))
        ttk.Label(s3, textvariable=self.teach_status_var, wraplength=230, justify="left",
                  foreground="#1976d2", font=self.font_bold).pack(anchor="w", pady=(2, 0))

        cf = ttk.Frame(parent, borderwidth=1, relief="sunken")
        cf.grid(row=2, column=0, sticky="nsew")
        cf.rowconfigure(0, weight=1)
        cf.columnconfigure(0, weight=1)
        self.canvas_teach = tk.Canvas(cf, bg="#000000", highlightthickness=0, cursor="crosshair")
        self.canvas_teach.grid(row=0, column=0, sticky="nsew")

        ttk.Label(parent, foreground="#64748b", wraplength=1000, justify="left",
                  text="Fine-tuning keeps the current model and adjusts it with your corrected frames from the new "
                       "footage. Drag from a bubble's centre to its edge to add it; right-click removes one. Wheel "
                       "zooms, middle-drag pans. Cyan = the model's suggestion, green = added by you. Mark every "
                       "bubble before ticking: unmarked bubbles are learned as background.").grid(row=3, column=0, sticky="w", pady=(4, 0))

        c = self.canvas_teach
        c.bind("<ButtonPress-1>", self._teach_press)
        c.bind("<B1-Motion>", self._teach_motion)
        c.bind("<ButtonRelease-1>", self._teach_release)
        c.bind("<ButtonPress-3>", lambda e: self._teach_delete_at(*self._teach_c2i(e.x, e.y)))
        c.bind("<ButtonPress-2>", lambda e: setattr(self, "teach_pan_start", (e.x, e.y)))
        c.bind("<B2-Motion>", self._teach_pan_motion)
        c.bind("<MouseWheel>", self._teach_wheel)
        c.bind("<Button-4>", lambda e: self._teach_zoom_at(1.15, e.x, e.y))
        c.bind("<Button-5>", lambda e: self._teach_zoom_at(1 / 1.15, e.x, e.y))
        c.bind("<Configure>", lambda e: self._teach_redraw())
        self.notebook.bind("<<NotebookTabChanged>>", lambda e: self._teach_redraw() if self._teach_tab_active() else None,
                           add="+")
        self._teach_on_regime()

    # ---------------------------------------------------------------- set handling
    def _teach_on_regime(self):
        """Called at start-up and when the Pulp/Froth switch changes."""
        reg = self._teach_regime()
        self.teach_set = teach.TeachingSet.latest(reg)
        self.teach_idx = 0
        self.teach_model_var.set(f"{'Pulp' if reg == 'pulp' else 'Froth'}: {teach.active_label(reg)}")
        self._teach_load_item()
        self._teach_refresh_counts()
        self._teach_update_current_score()

    def _teach_new_set(self):
        if self.teach_set and self.teach_set.items and not messagebox.askyesno(
                "New set of frames", "Start an empty set of frames? The current one stays on disk "
                                    f"({self.teach_set.folder})."):
            return
        self.teach_set = teach.TeachingSet.create(self._teach_regime())
        self.teach_idx = 0
        self._teach_load_item()
        self._teach_refresh_counts()
        self._log(f"[Fine-tune] New teaching set: {self.teach_set.folder}")

    def _teach_item(self) -> Optional[Dict[str, Any]]:
        if not self.teach_set or not self.teach_set.items:
            return None
        self.teach_idx = max(0, min(self.teach_idx, len(self.teach_set.items) - 1))
        return self.teach_set.items[self.teach_idx]

    def _teach_load_item(self):
        it = self._teach_item()
        self.teach_undo_stack = []
        self.teach_zoom = None
        if it is None:
            self.teach_img = None
            self.teach_labels = []
            self.teach_item_var.set("No frames yet. Pick a video on the left and press ①.")
        else:
            self.teach_img = Image.open(self.teach_set.image_path(it)).convert("RGB")
            self.teach_labels = self.teach_set.read_labels(it)
            self._teach_item_text()
        self._teach_redraw()

    def _teach_item_text(self):
        it = self._teach_item()
        if it is None:
            return
        n = len(self.teach_set.items)
        st = "✔ checked" if it.get("done") else ("waiting for the model's suggestions..." if not it.get("prelabelled")
                                                  else "not checked yet")
        where = f"frame {it['frame_idx']}" + (f", box at {it['box'][0]},{it['box'][1]}" if it.get("box") else "")
        self.teach_item_var.set(f"{self.teach_idx + 1} / {n}  ·  {len(self.teach_labels)} bubbles  ·  {st}\n"
                                f"{it['video']}, {where}")
        self.btn_teach_done.configure(style="Success.TButton" if it.get("done") else "TButton",
                                      text="✔ Checked (click to undo)" if it.get("done")
                                      else "✔ Every bubble here is marked")

    def _teach_refresh_counts(self):
        ts = self.teach_set
        busy = self.teach_busy or self.teach_proc is not None
        if ts is None or not ts.items:
            self.teach_ready_var.set(self._teach_minimum_text() + "\nNo frames added yet.")
            self.btn_teach_go.configure(state="disabled")
        else:
            ok, msg = ts.readiness()
            c = ts.counts()
            unit = "frames" if ts.regime == "pulp" else "boxes"
            self.teach_ready_var.set(f"{c['done']} of {c['items']} {unit} checked, {c['bubbles']} bubbles.\n{msg}")
            self.btn_teach_go.configure(state="normal" if ok and not busy else "disabled")
        self.btn_teach_add.configure(state="disabled" if busy else "normal")
        self.btn_teach_cancel.configure(state="normal" if self.teach_proc is not None else "disabled")
        self.btn_teach_revert.configure(state="disabled" if (teach.active_model(self._teach_regime()) is None or busy)
                                        else "normal")

    def _teach_minimum_text(self) -> str:
        reg = self._teach_regime()
        m = teach.MINIMUMS[reg]
        unit = "frames" if reg == "pulp" else "boxes"
        return (f"Minimum to teach: {m['items']} checked {unit} and {m['bubbles']} bubbles "
                f"({teach.RECOMMENDED_BUBBLES[reg]}+ is better).")

    # ---------------------------------------------------------------- ① add footage
    def _teach_add_frames(self):
        path = self.selected_source_path
        if not path or not path.lower().endswith((".avi", ".mp4")):
            messagebox.showinfo("Fine-tune the model", "Pick a video on the left first.")
            return
        reg = self._teach_regime()
        if self.teach_set is None or self.teach_set.regime != reg:
            self.teach_set = teach.TeachingSet.create(reg)
            self._log(f"[Fine-tune] New teaching set: {self.teach_set.folder}")
        ts = self.teach_set
        weights = self.weights_path.get()
        model_name = self._teach_model_name(reg)
        self.teach_busy = True
        self._teach_refresh_counts()
        self.teach_status_var.set("Cutting frames from the video...")

        def worker():
            try:
                roi_fn = None
                if reg == "froth":
                    from engine_froth import in_focus_region
                    roi_fn = in_focus_region
                added = ts.add_from_video(path, roi_fn=roi_fn)
                self.task_queue.put(("teach_added", len(added)))
                todo = [it for it in ts.items if not it.get("prelabelled")]
                for i, it in enumerate(todo, 1):
                    self.task_queue.put(("teach_status", f"The model is suggesting labels: {i} / {len(todo)}..."))
                    img = cv2.imread(ts.image_path(it))
                    if reg == "pulp":
                        if self.pulp_detector is None:
                            from engine_pulp import PulpDetector
                            self.pulp_detector = PulpDetector(weights)
                        dets = self.pulp_detector.detect_frame(img, score_thresh=0.40, frame_idx=it["frame_idx"])
                    else:
                        dets = self._get_froth_detector().detect_frame(img, frame_idx=it["frame_idx"])
                    labels = teach.bubbles_from_detections(dets, (0, 0), img.shape[1], img.shape[0])
                    ts.save_model_predictions(it, dets, model_name)
                    ts.write_labels(it, labels)
                    it["prelabelled"] = True
                    ts.save()
                    self.task_queue.put(("teach_prelabelled", it["stem"]))
                self.task_queue.put(("teach_add_done", len(added)))
            except Exception as e:
                self.task_queue.put(("teach_add_done", "error: %s" % e))

        threading.Thread(target=worker, daemon=True).start()

    # ---------------------------------------------------------------- ③ current model
    def _teach_model_name(self, reg: str) -> str:
        if reg == "pulp":
            return os.path.abspath(self.weights_path.get())
        return teach.active_model("froth") or "cellpose-default"

    def _teach_update_current_score(self):
        ts = self.teach_set
        if ts is None:
            self.teach_current_score = None
            self.teach_current_var.set(teach.current_model_verdict(None, self._teach_regime()))
            return
        sc = ts.current_model_score(self._teach_model_name(ts.regime))
        self.teach_current_score = sc
        txt = teach.current_model_verdict(sc, ts.regime)
        if sc and sc.get("missing"):
            txt += f" ({sc['missing']} checked item(s) not scored yet: press Re-check.)"
        self.teach_current_var.set(txt)

    def _teach_recheck_current(self):
        """Run the model in use on checked items whose saved predictions are missing or
        came from another model, then score it again."""
        ts = self.teach_set
        if ts is None or self.teach_busy:
            return
        reg = ts.regime
        name = self._teach_model_name(reg)
        todo = [it for it in ts.items if it.get("done")
                and (ts.read_model_predictions(it) or {}).get("model") != name]
        if not todo:
            self._teach_update_current_score()
            return
        weights = self.weights_path.get()
        self.teach_busy = True
        self._teach_refresh_counts()

        def worker():
            try:
                for i, it in enumerate(todo, 1):
                    self.task_queue.put(("teach_status", f"Running the current model: {i} / {len(todo)}..."))
                    img = cv2.imread(ts.image_path(it))
                    if reg == "pulp":
                        if self.pulp_detector is None:
                            from engine_pulp import PulpDetector
                            self.pulp_detector = PulpDetector(weights)
                        dets = self.pulp_detector.detect_frame(img, score_thresh=0.40, frame_idx=it["frame_idx"])
                    else:
                        dets = self._get_froth_detector().detect_frame(img, frame_idx=it["frame_idx"])
                    ts.save_model_predictions(it, dets, name)
                self.task_queue.put(("teach_recheck_done", None))
            except Exception as e:
                self.task_queue.put(("teach_recheck_done", "error: %s" % e))

        threading.Thread(target=worker, daemon=True).start()

    # ---------------------------------------------------------------- ② navigation + labelling
    def _teach_goto(self, idx: int):
        if not self.teach_set or not self.teach_set.items:
            return
        self.teach_idx = max(0, min(idx, len(self.teach_set.items) - 1))
        self._teach_load_item()

    def _teach_next_unchecked(self):
        if not self.teach_set:
            return
        n = len(self.teach_set.items)
        for k in range(1, n + 1):
            j = (self.teach_idx + k) % n
            if not self.teach_set.items[j].get("done"):
                self._teach_goto(j)
                return
        messagebox.showinfo("Fine-tune the model", "Every frame in the set is checked.")

    def _teach_toggle_done(self):
        it = self._teach_item()
        if it is None:
            return
        if not it.get("prelabelled"):
            messagebox.showinfo("Fine-tune the model", "Wait until the model's suggestions are in for this frame.")
            return
        self.teach_set.set_done(it, not it.get("done"))
        self._teach_item_text()
        self._teach_refresh_counts()
        self._teach_update_current_score()
        self._teach_redraw()

    def _teach_save_labels(self):
        it = self._teach_item()
        if it is None:
            return
        self.teach_set.write_labels(it, self.teach_labels)
        self._teach_item_text()
        self._teach_refresh_counts()
        if it.get("done"):
            self._teach_update_current_score()

    def _teach_push_undo(self):
        self.teach_undo_stack.append([dict(b) for b in self.teach_labels])
        self.teach_undo_stack = self.teach_undo_stack[-50:]

    def _teach_undo(self):
        if self.teach_undo_stack:
            self.teach_labels = self.teach_undo_stack.pop()
            self._teach_save_labels()
            self._teach_redraw()

    def _teach_hit(self, ix: float, iy: float) -> int:
        """Index of the smallest bubble under the point, or -1."""
        best, best_r = -1, float("inf")
        tol = 2.0 / self._teach_scale()
        for i, b in enumerate(self.teach_labels):
            if math.hypot(b["cx"] - ix, b["cy"] - iy) <= b["r"] + tol and b["r"] < best_r:
                best, best_r = i, b["r"]
        return best

    def _teach_delete_at(self, ix: float, iy: float):
        if self._teach_item() is None:
            return
        i = self._teach_hit(ix, iy)
        if i >= 0:
            self._teach_push_undo()
            self.teach_labels.pop(i)
            self._teach_save_labels()
            self._teach_redraw()

    def _teach_press(self, e):
        if self._teach_item() is None or self.teach_img is None:
            return
        ix, iy = self._teach_c2i(e.x, e.y)
        if self.teach_tool.get() == "delete":
            self._teach_delete_at(ix, iy)
        else:
            self.teach_drag = (ix, iy, ix, iy)

    def _teach_motion(self, e):
        if self.teach_drag:
            ix, iy = self._teach_c2i(e.x, e.y)
            self.teach_drag = (self.teach_drag[0], self.teach_drag[1], ix, iy)
            self._teach_redraw()

    def _teach_release(self, e):
        if not self.teach_drag:
            return
        x0, y0, x1, y1 = self.teach_drag
        self.teach_drag = None
        r = math.hypot(x1 - x0, y1 - y0)
        if r >= 1.5 and self.teach_img is not None:
            w, h = self.teach_img.size
            self._teach_push_undo()
            self.teach_labels.append(teach.make_circle(x0, y0, r, w, h, source="user"))
            self._teach_save_labels()
        self._teach_redraw()

    # ---------------------------------------------------------------- canvas geometry
    def _teach_scale(self) -> float:
        if self.teach_img is None:
            return 1.0
        if self.teach_zoom is None:
            cw = max(50, self.canvas_teach.winfo_width())
            ch = max(50, self.canvas_teach.winfo_height())
            iw, ih = self.teach_img.size
            z = min(cw / iw, ch / ih)
            self.teach_pan = [(cw - iw * z) / 2, (ch - ih * z) / 2]
            return z
        return self.teach_zoom

    def _teach_c2i(self, cx, cy):
        z = self._teach_scale()
        return (cx - self.teach_pan[0]) / z, (cy - self.teach_pan[1]) / z

    def _teach_zoom_at(self, f, cx, cy):
        if self.teach_img is None:
            return
        z = self._teach_scale()
        ix, iy = (cx - self.teach_pan[0]) / z, (cy - self.teach_pan[1]) / z
        nz = max(0.1, min(40.0, z * f))
        self.teach_zoom = nz
        self.teach_pan = [cx - ix * nz, cy - iy * nz]
        self._teach_redraw()

    def _teach_wheel(self, e):
        self._teach_zoom_at(1.15 if e.delta > 0 else 1 / 1.15, e.x, e.y)

    def _teach_pan_motion(self, e):
        if self.teach_pan_start:
            self._teach_scale()
            if self.teach_zoom is None:
                self.teach_zoom = self._teach_scale()
            self.teach_pan[0] += e.x - self.teach_pan_start[0]
            self.teach_pan[1] += e.y - self.teach_pan_start[1]
            self.teach_pan_start = (e.x, e.y)
            self._teach_redraw()

    def _teach_redraw(self):
        c = getattr(self, "canvas_teach", None)
        if c is None:
            return
        c.delete("all")
        if self.teach_img is None:
            c.create_text(max(200, c.winfo_width() // 2), max(100, c.winfo_height() // 2), fill="#94a3b8",
                          text="No frames yet.\nPick a video on the left, then press ① Add frames.",
                          font=("Segoe UI", 12), justify="center")
            return
        z = self._teach_scale()
        iw, ih = self.teach_img.size
        disp = self.teach_img.resize((max(1, int(iw * z)), max(1, int(ih * z))), Image.Resampling.BICUBIC)
        self.teach_photo = ImageTk.PhotoImage(disp)
        px, py = self.teach_pan
        c.create_image(px, py, anchor="nw", image=self.teach_photo)
        it = self._teach_item()
        if it and it.get("done"):
            c.create_rectangle(px, py, px + iw * z, py + ih * z, outline="#10b981", width=3)
        for b in self.teach_labels:
            col = "#10b981" if b.get("source") == "user" else "#00e5ff"
            ol = b.get("outline")
            if ol and len(ol) >= 3:
                pts = [v for p in ol for v in (px + p[0] * z, py + p[1] * z)]
                c.create_polygon(pts, outline=col, fill="", width=1)
            else:
                r = b["r"] * z
                cx, cy = px + b["cx"] * z, py + b["cy"] * z
                c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=col, width=1.5)
        if self.teach_drag:
            x0, y0, x1, y1 = self.teach_drag
            r = math.hypot(x1 - x0, y1 - y0) * z
            cx, cy = px + x0 * z, py + y0 * z
            c.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#f59e0b", width=2)

    # ---------------------------------------------------------------- ③ training
    def _teach_estimate_minutes(self, regime: str, n_train: int) -> str:
        if regime == "pulp":
            # 6 epochs x (9 windows per new frame + as many replayed) / batch 2, ~5 s per step on
            # the lab laptop (rr_full_e20: 20 epochs x 107 steps in ~3 h), + ~5 min for the check
            m = 6 * 9 * n_train * 5 / 60 + 5
        else:
            # 40 epochs x 8 crops of 256 px, last 4 ViT blocks trained: ~6 s/step on a 2-core test VM
            m = 40 * 8 * 6 / 60 + 5
        lo, hi = int(m * 0.7) // 5 * 5 + 5, int(m * 1.5) // 5 * 5 + 5
        return f"about {lo}-{hi} minutes on this computer's CPU"

    def _teach_start_training(self):
        ts = self.teach_set
        if ts is None or self.teach_proc is not None:
            return
        ok, msg = ts.readiness()
        if not ok:
            messagebox.showinfo("Fine-tune the model", msg)
            return
        reg = ts.regime
        train_items, check_items = ts.split()
        unit = "frames" if reg == "pulp" else "boxes"
        self._teach_update_current_score()
        if not messagebox.askyesno(
                "Fine-tune the model",
                f"{self.teach_current_var.get()}\n\n"
                f"Train on {len(train_items)} {unit}; {len(check_items)} more are held back to check the result.\n\n"
                f"This takes {self._teach_estimate_minutes(reg, len(train_items))}. It runs in the background: "
                f"keep BubbleMetrix open and the computer plugged in. You can keep measuring meanwhile "
                f"(more slowly).\n\nAt the end the new model is used only if it does better than the "
                f"current one. Start?"):
            return

        out = teach.new_model_dir(reg)
        if reg == "pulp":
            py = sys.executable
            cmd = [py, "-u", os.path.join(HERE, "engines", "finetune_pulp.py"), "--set", ts.folder,
                   "--base", self.weights_path.get(), "--out", out]
        else:
            from engine_froth import froth_python
            try:
                py = froth_python()
            except Exception as e:
                messagebox.showerror("Fine-tune the model", str(e))
                return
            cmd = [py, "-u", os.path.join(HERE, "Froth", "cellpose", "finetune_cellpose.py"), "--set", ts.folder,
                   "--out", out, "--base", teach.active_model("froth") or ""]
        cmd += os.environ.get("BUBBLEMETRIX_TEACH_EXTRA_ARGS", "").split()   # test runs only
        log_fh = open(os.path.join(out, "train.log"), "a", encoding="utf-8")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        self.teach_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log_fh, text=True, encoding="utf-8",
                                           bufsize=1, creationflags=flags, env=env, cwd=HERE)
        self.teach_running_regime = reg
        self.teach_out = out
        self._keep_awake(True)
        self._log(f"[Fine-tune] Training started ({reg}): {' '.join(cmd)}")
        self.teach_status_var.set("Starting...")
        self._teach_refresh_counts()
        proc = self.teach_proc

        def reader():
            last = None
            for line in proc.stdout:
                if line.startswith("@@ "):
                    try:
                        last = json.loads(line[3:])
                        self.task_queue.put(("teach_progress", last))
                    except ValueError:
                        pass
            rc = proc.wait()
            res_p = os.path.join(out, "result.json")
            if rc == 0 and os.path.exists(res_p):
                with open(res_p, encoding="utf-8") as fh:
                    self.task_queue.put(("teach_done", json.load(fh)))
            else:
                err = (last or {}).get("error") or f"stopped (exit code {rc})"
                self.task_queue.put(("teach_failed", f"{err}\nLog: {os.path.join(out, 'train.log')}"))
            log_fh.close()

        threading.Thread(target=reader, daemon=True).start()

    def _teach_cancel(self):
        if self.teach_proc and messagebox.askyesno("Fine-tune the model", "Stop the training? Nothing changes; "
                                                                      "the current model stays in use."):
            try:
                self.teach_proc.terminate()
            except Exception:
                pass

    def _keep_awake(self, on: bool):
        """Windows: stop the laptop sleeping mid-training (it killed a run before)."""
        try:
            import ctypes
            ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
        except Exception:
            pass

    def _teach_apply_model(self, regime: str, path: Optional[str]):
        """Switch the detector in use; old detections came from the old model, so drop them."""
        if regime == "pulp":
            self.weights_path.set(path or self.default_pulp_weights)
            self.pulp_detector = None           # reloads with the new weights on next use
        elif self.froth_detector is not None:
            self.froth_detector.set_model(path)
        self.frame_results = {}
        self.video_result = None
        self._refresh_scale_info()
        self.teach_model_var.set(f"{'Pulp' if regime == 'pulp' else 'Froth'}: {teach.active_label(regime)}")
        self._teach_refresh_counts()

    def _teach_revert(self):
        reg = self._teach_regime()
        if teach.active_model(reg) is None:
            return
        if messagebox.askyesno("Fine-tune the model", "Go back to the validated base model? The fine-tuned model "
                                                  "stays on disk and is listed in models/registry.json."):
            teach.set_active(reg, None)
            self._teach_apply_model(reg, None)
            self._log(f"[Fine-tune] {reg}: back to the base model.")

    # ---------------------------------------------------------------- queue messages (main thread)
    def _teach_on_message(self, kind: str, data: Any):
        if kind == "teach_status":
            self.teach_status_var.set(data)
        elif kind == "teach_added":
            if self.teach_set and self.teach_set.items:
                self.teach_idx = len(self.teach_set.items) - data if data else self.teach_idx
                self._teach_load_item()
            self._log(f"[Fine-tune] Added {data} new {'frames' if self._teach_regime() == 'pulp' else 'boxes'}.")
        elif kind == "teach_prelabelled":
            it = self._teach_item()
            if it and it["stem"] == data:
                self._teach_load_item()
        elif kind == "teach_add_done":
            self.teach_busy = False
            if isinstance(data, str) and data.startswith("error"):
                self.teach_status_var.set("Could not add frames: " + data[7:])
                messagebox.showerror("Fine-tune the model", data)
            else:
                self.teach_status_var.set(f"{data} added. Fix the labels, then tick each one." if data
                                          else "No new frames left in this video. Pick another video.")
            self._teach_item_text()
            self._teach_refresh_counts()
        elif kind == "teach_recheck_done":
            self.teach_busy = False
            self.teach_status_var.set("" if data is None else "Re-check failed: " + str(data)[7:])
            self._teach_update_current_score()
            self._teach_refresh_counts()
        elif kind == "teach_progress":
            st = data.get("stage")
            if st == "train":
                eta = data.get("eta_s")
                left = "" if eta is None else (f" · about {max(1, round(eta / 60))} min left")
                self.teach_status_var.set(f"Training: epoch {data.get('epoch')}/{data.get('of')} · "
                                          f"step {data.get('step')}/{data.get('steps')}{left}")
                if data.get("steps"):
                    self.progress_bar["value"] = int(90 * data["step"] / max(1, data["steps"]))
            elif st == "check":
                self.teach_status_var.set(f"Comparing old and new model: {data.get('done')}/{data.get('of')} "
                                          f"({data.get('what', '')})")
                self.progress_bar["value"] = 90 + int(10 * data.get("done", 0) / max(1, data.get("of", 1)))
            elif st == "setup":
                self._log(f"[Fine-tune] {data.get('train')} training / {data.get('check')} held-back items, "
                          f"device {data.get('device')}")
        elif kind in ("teach_done", "teach_failed"):
            reg = self.teach_running_regime or self._teach_regime()
            self.teach_proc = None
            self.teach_running_regime = None
            self._keep_awake(False)
            self.progress_bar["value"] = 100
            if kind == "teach_failed":
                self.teach_status_var.set("Training stopped. The current model is still in use.")
                self._log(f"[Fine-tune] Training stopped: {data}")
                messagebox.showwarning("Fine-tune the model", f"Training did not finish.\n\n{data}\n\n"
                                                          f"The current model is still in use.")
            else:
                adopt = bool(data.get("adopt"))
                entry = {"path": data["model_path"], "date": time.strftime("%Y-%m-%d %H:%M"),
                         "set": os.path.basename(data.get("teaching_set", "")), "adopted": adopt,
                         "base": data.get("base"), "check_old": data.get("check_old"),
                         "check_new": data.get("check_new"), "bench_old": data.get("bench_old"),
                         "bench_new": data.get("bench_new"), "result": os.path.join(self.teach_out, "result.json")}
                teach.record_training(reg, entry, make_active=adopt)
                if self.teach_set:
                    self.teach_set.doc.setdefault("trainings", []).append(
                        {"date": entry["date"], "model": entry["path"], "adopted": adopt})
                    self.teach_set.save()
                reasons = "\n".join(data.get("reasons", []))
                self._log(f"[Fine-tune] Finished. Adopted: {adopt}. " + " ".join(data.get("reasons", [])))
                if adopt:
                    self._teach_apply_model(reg, data["model_path"])
                    self.teach_status_var.set("New model is in use.")
                    messagebox.showinfo("Fine-tune the model", "The new model did better and is now in use.\n\n"
                                        f"{reasons}\n\n'Go back to the base model' undoes this.")
                else:
                    self.teach_status_var.set("Kept the current model (the new one was not better).")
                    messagebox.showinfo("Fine-tune the model", "The current model stays in use: the new one was not "
                                        f"better.\n\n{reasons}\n\nAdd and check more frames, then press Fine-tune again.")
            self._teach_refresh_counts()
