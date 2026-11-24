"""engines.convergence
When has D32 stopped changing?  Used by Measure Full Video to stop early and to say
how sure the D32 is.

D32 = sum(d^3) / sum(d^2) is a ratio of two sums, so its standard error comes from
linearising the ratio (the delta method, the same maths as a ratio estimator in
survey sampling):

    z_j = (d_j^3 - D32 * d_j^2) / sum(d^2)

Two error estimates are computed and the larger one is used:

  * bubble level - every bubble independent:  SE^2 = n/(n-1) * sum_j z_j^2
  * frame level  - frames as clusters, so frame-to-frame drift of the bubble
    population (and bubbles that share a frame) are counted:
                    SE^2 = k/(k-1) * sum_i (sum_{j in frame i} z_j)^2,
    with a Student-t multiplier on k-1 degrees of freedom, which is large for
    few frames and so stops the rule trusting 2-3 frames.

D32 counts as converged when the 95% half-width is <= TARGET of D32, at least
MIN_FRAMES frames are in, and that was already true at the previous frame (so one
lucky frame does not end the run).

Checked against the bootstrap in Froth/eval/d32_convergence.py on the 9 saved
Cellpose froth frames (see build notes, 2026-11-20).  numpy only.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

TARGET = 0.05          # 95% CI half-width as a fraction of D32
MIN_FRAMES = 3         # never call it converged from fewer frames
CONSECUTIVE = 2        # the rule must hold at this many successive checks

# two-sided 95% Student t quantiles, df 1..30 (above that 1.96 is close enough)
_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t95(df: int) -> float:
    if df < 1:
        return float("inf")
    return _T95[df - 1] if df <= len(_T95) else 1.96


def d32_interval(frames: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """95% interval for the pooled D32 of a list of per-frame diameter lists."""
    groups = [np.asarray(f, dtype=np.float64) for f in frames if len(f)]
    k = len(groups)
    if not k:
        return {"n": 0, "k": 0, "d32": 0.0, "half_width": float("inf"), "rel": float("inf"),
                "lo": 0.0, "hi": 0.0, "rel_bubble": float("inf"), "rel_frame": float("inf")}
    d = np.concatenate(groups)
    n = len(d)
    s2 = float((d ** 2).sum())
    D = float((d ** 3).sum() / s2) if s2 > 0 else 0.0
    if n < 2 or D <= 0:
        return {"n": n, "k": k, "d32": D, "half_width": float("inf"), "rel": float("inf"),
                "lo": D, "hi": D, "rel_bubble": float("inf"), "rel_frame": float("inf")}

    z = (d ** 3 - D * d ** 2) / s2
    se_b = math.sqrt(n / (n - 1) * float((z ** 2).sum()))
    hw_b = 1.96 * se_b

    if k >= 2:
        Z, i = [], 0
        for g in groups:
            Z.append(float(z[i:i + len(g)].sum()))
            i += len(g)
        Z = np.asarray(Z)
        se_c = math.sqrt(k / (k - 1) * float((Z ** 2).sum()))
        hw_c = t95(k - 1) * se_c
    else:
        hw_c = float("inf")

    hw = max(hw_b, hw_c)
    return {"n": int(n), "k": int(k), "d32": D, "half_width": hw, "rel": hw / D,
            "lo": D - hw, "hi": D + hw, "rel_bubble": hw_b / D, "rel_frame": hw_c / D}


class ConvergenceTracker:
    """Feed it one frame of accepted diameters at a time; it says when to stop."""

    def __init__(self, target: float = TARGET, min_frames: int = MIN_FRAMES,
                 consecutive: int = CONSECUTIVE):
        self.target = target
        self.min_frames = min_frames
        self.consecutive = consecutive
        self.frames: List[np.ndarray] = []
        self.frame_ids: List[int] = []
        self.history: List[Dict[str, Any]] = []
        self._streak = 0
        self.converged_at: Optional[Dict[str, Any]] = None

    def add_frame(self, diameters: Sequence[float], frame_idx: int = -1) -> Dict[str, Any]:
        self.frames.append(np.asarray(diameters, dtype=np.float64))
        self.frame_ids.append(int(frame_idx))
        ci = d32_interval(self.frames)
        ok = ci["k"] >= self.min_frames and ci["rel"] <= self.target
        self._streak = self._streak + 1 if ok else 0
        ci["frame_idx"] = int(frame_idx)
        ci["converged"] = self._streak >= self.consecutive
        self.history.append(ci)
        if ci["converged"] and self.converged_at is None:
            self.converged_at = dict(ci)
        return ci

    @property
    def converged(self) -> bool:
        return self.converged_at is not None

    def report(self) -> Dict[str, Any]:
        """Plain summary for the GUI and the JSON export."""
        last = self.history[-1] if self.history else d32_interval([])
        out = {
            "target_rel": self.target,
            "min_frames": self.min_frames,
            "converged": self.converged,
            "n": last["n"], "frames": last["k"], "d32": last["d32"],
            "rel_half_width": last["rel"], "lo": last["lo"], "hi": last["hi"],
            "history": [{"n": h["n"], "k": h["k"], "d32": h["d32"], "lo": h["lo"], "hi": h["hi"],
                         "rel": h["rel"], "frame_idx": h["frame_idx"]} for h in self.history],
        }
        if self.converged_at:
            out["converged_after_bubbles"] = self.converged_at["n"]
            out["converged_after_frames"] = self.converged_at["k"]
        elif last["n"] > 0 and math.isfinite(last["rel"]):
            # error shrinks ~ 1/sqrt(sample); estimate what is still needed
            f = (last["rel"] / self.target) ** 2
            out["estimated_bubbles_needed"] = int(math.ceil(last["n"] * f))
            out["estimated_frames_needed"] = max(self.min_frames, int(math.ceil(last["k"] * f)))
        return out


def spread_order(n: int) -> List[int]:
    """0..n-1 in bit-reversed (van der Corput) order: any prefix is spread over the
    whole range, so stopping early still samples the start, middle and end."""
    if n <= 0:
        return []
    bits = max(1, (n - 1).bit_length())
    seen, out = set(), []
    for i in range(1 << bits):
        r = int(format(i, "0%db" % bits)[::-1], 2)
        if r < n and r not in seen:
            seen.add(r)
            out.append(r)
    return out
