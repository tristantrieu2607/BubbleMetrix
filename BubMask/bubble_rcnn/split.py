"""The clip-level split, read from one file by both train and eval.

Handover principle 3: split by clip, never by frame.  Frames inside a clip are
near-duplicates -- same fouling on the window, same lighting, the same bubbles
drifting -- so a frame-level split reports partly-memorised performance.  The
previous split put frame 015 of a clip in train and frame 065 of the same clip in
val, for every clip.

Keeping the split in a file rather than in a random seed means it is auditable,
it survives a re-run, and it cannot be quietly re-drawn after somebody sees a
number they dislike.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "split_clips.json")

# _x<x>y<y> origin tag added 2026-09-27: without it the 26 uniform-random crops
# resolved to "unassigned" and would have been silently dropped from training.
_FRAME_SUFFIX = re.compile(r"_f\d+(_x\d+y\d+)?(_crop\d+|_full)?$")


def clip_of(stem: str) -> str:
    """Clip name for a crop stem: drop the frame and crop suffix."""
    return _FRAME_SUFFIX.sub("", os.path.splitext(os.path.basename(stem))[0])


def load(path: Optional[str] = None) -> Dict[str, List[str]]:
    with open(path or DEFAULT_PATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    return {"test": list(d.get("test_clips", [])),
            "train": list(d.get("train_clips", [])),
            "val": list(d.get("val_clips", []))}


def side_of(stem: str, split: Optional[Dict[str, List[str]]] = None) -> str:
    """'test', 'val', 'train', or 'unassigned' for a crop stem."""
    split = split or load()
    c = clip_of(stem)
    for side in ("test", "val", "train"):
        if c in split.get(side, []):
            return side
    return "unassigned"


def filter_dir(crops_dir: str, side: str, split: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """Crop stems in ``crops_dir`` belonging to ``side``."""
    split = split or load()
    out = []
    for root, _, files in os.walk(crops_dir):
        for f in sorted(files):
            if not f.endswith(".png") or f.endswith(("_view.png", "_mask.png", "_ignore.png")):
                continue
            stem = os.path.splitext(f)[0]
            if side_of(stem, split) == side:
                out.append(stem)
    return out


# ---------------------------------------------------------------------------
# Leave-one-clip-out cross-validation (added 2026-09-29).
# There is no more footage: 8 rotor clips is the whole dataset. One fixed 3-clip
# test set wastes data and gives one noisy number. Instead every clip is the test
# clip once; the next clip in the list is that fold's val clip (checkpoint
# selection only); the other 6 train. Report mean +/- spread over the 8 folds.
# ---------------------------------------------------------------------------

def all_clips(split: Optional[Dict[str, List[str]]] = None) -> List[str]:
    split = split or load()
    return sorted(set(split["train"]) | set(split["val"]) | set(split["test"]))


def make_fold(k: int, split: Optional[Dict[str, List[str]]] = None,
              with_val: bool = True) -> Dict[str, List[str]]:
    """Fold k: clip k is test. with_val=False puts every other clip in train --
    use it with a fixed epoch count and last.pt (no checkpoint selection needed)."""
    clips = all_clips(split)
    n = len(clips)
    if not 0 <= k < n:
        raise ValueError(f"fold {k} out of range 0..{n - 1}")
    test = clips[k]
    if not with_val:
        return {"test": [test], "val": [], "train": [c for c in clips if c != test]}
    val = clips[(k + 1) % n]
    return {"test": [test], "val": [val], "train": [c for c in clips if c not in (test, val)]}


def resolve(path: Optional[str] = None, fold: Optional[int] = None,
            with_val: bool = True) -> Dict[str, List[str]]:
    """The split file, or fold ``fold`` of leave-one-clip-out built from it."""
    base = load(path)
    return base if fold is None else make_fold(fold, base, with_val=with_val)
