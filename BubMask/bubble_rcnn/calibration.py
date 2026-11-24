"""Calibration, and the refusal.

Handover principle 6: refuse to emit millimetres for any clip without a *measured*
calibration.  Pixels are honest; a wrong scale is not, and the previous tool's
183 px/mm -- stamped everywhere as an assumption and never replaced -- made every
diameter in its report 66% too large.

3.291 um/px comes from a 1024 px = 3.37 mm field quoted by the video provider.
That is a derived number, not a measured one, so it is marked `provisional` here
and stays provisional until somebody images a target through the same probe at the
same working distance and writes `measured` into calibration.csv.

The froth column is a different optical path and has **no calibration at all**.
v1 carried a hard-coded 1000/8.910 um/px in ``measure_video`` with no source
anywhere in the repo or the docs, while the summary written next to it used the
pulp 3.291 -- a 34x disagreement between the CSV and the JSON of the same run.
Both are refused until a real number exists.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(os.path.dirname(REPO), "calibration.csv")


@dataclass(frozen=True)
class Calibration:
    regime: str
    px_per_mm: Optional[float]
    status: str          # "measured" | "provisional" | "none"
    source: str

    @property
    def um_per_px(self) -> Optional[float]:
        return (1000.0 / self.px_per_mm) if self.px_per_mm else None

    @property
    def may_emit_microns(self) -> bool:
        return self.status == "measured" and self.px_per_mm is not None

    def describe(self) -> str:
        if self.px_per_mm is None:
            return f"{self.regime}: no calibration ({self.source}) -- pixels only"
        return (f"{self.regime}: {self.px_per_mm:.3f} px/mm = {self.um_per_px:.3f} um/px "
                f"[{self.status}] {self.source}")


_BUILT_IN: Dict[str, Calibration] = {
    "pulp": Calibration("pulp", 303.858, "provisional",
                        "1024 px = 3.37 mm field, quoted by the video provider; not measured through the probe"),
    "froth": Calibration("froth", None, "none",
                         "no target ever imaged through the froth column optics"),
}


def load(regime: str, csv_path: Optional[str] = None) -> Calibration:
    """Built-in defaults, overridden by calibration.csv when one exists.

    CSV columns: regime, px_per_mm, status, source.
    """
    cal = _BUILT_IN.get(regime, Calibration(regime, None, "none", "unknown regime"))
    path = csv_path or DEFAULT_CSV
    if os.path.exists(path):
        try:
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if (row.get("regime") or "").strip().lower() == regime:
                        ppm = row.get("px_per_mm")
                        cal = Calibration(regime,
                                          float(ppm) if ppm else None,
                                          (row.get("status") or "provisional").strip().lower(),
                                          (row.get("source") or path).strip())
        except Exception:
            pass
    return cal


def require_microns(regime: str, csv_path: Optional[str] = None) -> float:
    cal = load(regime, csv_path)
    if not cal.may_emit_microns:
        raise ValueError(
            f"refusing to emit microns for '{regime}': calibration status is "
            f"'{cal.status}'. {cal.describe()}  Report pixels, or image a target "
            f"through this optical path and set status=measured in calibration.csv."
        )
    return cal.um_per_px
