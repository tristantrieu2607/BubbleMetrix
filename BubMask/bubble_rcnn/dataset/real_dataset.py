"""Dataset for real human-stamped annotations, with the ignore path actually wired.

Three things changed from v1, all of them consequences of the annotation guide:

1. **``ignore_mask`` is produced.**  ``models/ignore_loss.py`` reads
   ``targets[i]["ignore_mask"]``; nothing used to write it, so ``IgnoreAwareRPN`` and
   ``IgnoreAwareRoIHeads`` were installed and did nothing.  Every anchor over an
   occluded bubble was still trained as background -- the exact signal the guide
   spends section 4 preventing, and the reason the original BubMask reached 5%
   recall on dense footage.

2. **An unconfirmed empty crop is not a background image.**  "Skipped" and "there
   are no bubbles here" are different statements and used to be the same file.
   Only ``confirmed_empty`` crops are trained as negatives; the rest are dropped
   from the index, loudly.

3. **The floor is enforced, as ignore rather than background.**  A sub-floor
   object is usually debris, but it is never *water*, so punishing a detection
   there teaches the model to suppress the appearance it must find.

Augmentation now transforms the ignore mask with the image.  In v1 a horizontal
flip moved the boxes and masks and left the ignore mask where it was; that was
harmless only because the mask never existed.
"""

import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Any, Tuple, List, Optional

from bubble_rcnn.annotation import schema
from bubble_rcnn.config import BubbleConfig

_FLOOR_PX = BubbleConfig().min_diameter_px


class RealBubbleDataset(Dataset):
    def __init__(self, crops_dir: str, labels_dir: str, is_train: bool = False,
                 min_bubbles: int = 0, transforms: Optional[Any] = None,
                 require_confirmed_empty: bool = True, verbose: bool = True,
                 restrict_to: Optional[set] = None,
                 scale_range: Optional[Tuple[float, float]] = None,
                 scale_prob: float = 0.0):
        self.crops_dir = crops_dir
        self.labels_dir = labels_dir
        self.is_train = is_train
        self.min_bubbles = min_bubbles
        self.transforms = transforms
        self.require_confirmed_empty = require_confirmed_empty
        # When set, only these crop stems are used -- this is how the clip-level
        # split reaches the loader, instead of a random seed nobody can audit.
        self.restrict_to = set(restrict_to) if restrict_to is not None else None
        # scale jitter -- training only, see config.scale_jitter_*
        self.scale_range = tuple(scale_range) if scale_range else None
        self.scale_prob = float(scale_prob) if (scale_range and is_train) else 0.0

        self.samples: List[Tuple[str, str]] = []
        self.skipped_unconfirmed: List[str] = []

        label_map = {}
        for root, _, files in os.walk(labels_dir):
            for f in files:
                if f.endswith("_labels.json"):
                    label_map[f[: -len("_labels.json")]] = os.path.join(root, f)

        for root, _, files in os.walk(crops_dir):
            for f in sorted(files):
                if not f.endswith(".png") or f.endswith("_view.png") \
                        or f.endswith("_mask.png") or f.endswith("_ignore.png"):
                    continue
                stem = os.path.splitext(f)[0]
                if self.restrict_to is not None and stem not in self.restrict_to:
                    continue
                if stem not in label_map:
                    continue
                label_path = label_map[stem]
                try:
                    with open(label_path, "r", encoding="utf-8") as fp:
                        doc = json.load(fp)
                except Exception:
                    continue

                n = len(doc.get("bubbles", []))
                if n < self.min_bubbles:
                    continue
                if n == 0 and self.require_confirmed_empty and not doc.get("confirmed_empty", False):
                    # Saved with nothing on it and never confirmed empty: this is a
                    # crop somebody walked past, not evidence of empty water.
                    self.skipped_unconfirmed.append(stem)
                    continue
                self.samples.append((os.path.join(root, f), label_path))

        if verbose and self.skipped_unconfirmed:
            print(f"[Dataset] {len(self.skipped_unconfirmed)} crop(s) skipped: zero bubbles and "
                  f"not confirmed empty. Press 'E' in the annotator to confirm one as a "
                  f"true negative, or stamp it.")
            for stem in self.skipped_unconfirmed[:10]:
                print(f"          - {stem}")
            if len(self.skipped_unconfirmed) > 10:
                print(f"          ... and {len(self.skipped_unconfirmed) - 10} more")

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------ #
    def _build_ignore(self, doc: Dict[str, Any], label_file: str,
                      h: int, w: int, sub_floor: List[Dict[str, Any]]) -> np.ndarray:
        """Ignore = hand-drawn regions + any generated ignore mask + sub-floor objects."""
        ign = schema.rasterise_ignore(doc.get("ignore_regions", []), h, w)

        side = label_file.replace("_labels.json", "_ignore.png")
        if os.path.exists(side):
            extra = cv2.imread(side, cv2.IMREAD_GRAYSCALE)
            if extra is not None and extra.shape[:2] == (h, w):
                ign = np.maximum(ign, (extra > 127).astype(np.uint8))

        for b in sub_floor:
            a, c = schema.semi_axes(b)
            centre = (int(round(float(b["cx"]))), int(round(float(b["cy"]))))
            if b.get("shape_type") == "ellipse":
                cv2.ellipse(ign, centre, (max(1, int(round(a))), max(1, int(round(c)))),
                            float(b.get("angle", 0.0)), 0, 360, 1, -1)
            else:
                cv2.circle(ign, centre, max(1, int(round(a))), 1, -1)
        return ign

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path, label_file = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        with open(label_file, "r", encoding="utf-8") as f:
            doc = json.load(f)
        bubbles = doc.get("bubbles", [])

        boxes, masks, areas, sub_floor = [], [], [], []

        for b in bubbles:
            if schema.below_floor(b):
                sub_floor.append(b)
                continue
            a, c = schema.semi_axes(b)
            cx, cy = float(b["cx"]), float(b["cy"])
            x1, y1 = max(0.0, cx - a), max(0.0, cy - c)
            x2, y2 = min(float(w), cx + a), min(float(h), cy + c)
            if (x2 - x1) < 4.0 or (y2 - y1) < 4.0:
                continue
            m = np.zeros((h, w), dtype=np.uint8)
            centre = (int(round(cx)), int(round(cy)))
            if b.get("shape_type") == "ellipse":
                cv2.ellipse(m, centre, (max(1, int(round(a))), max(1, int(round(c)))),
                            float(b.get("angle", 0.0)), 0, 360, 1, -1)
            else:
                cv2.circle(m, centre, max(1, int(round(a))), 1, -1)
            boxes.append([x1, y1, x2, y2])
            masks.append(m)
            areas.append(float(np.pi * a * c))

        ignore = self._build_ignore(doc, label_file, h, w, sub_floor)

        # ---- augmentation: image, instance masks and ignore move together ----
        if self.is_train:
            if np.random.rand() > 0.5:
                img = np.fliplr(img).copy()
                ignore = np.fliplr(ignore).copy()
                boxes = [[w - bb[2], bb[1], w - bb[0], bb[3]] for bb in boxes]
                masks = [np.fliplr(m).copy() for m in masks]
            if np.random.rand() > 0.5:
                img = np.flipud(img).copy()
                ignore = np.flipud(ignore).copy()
                boxes = [[bb[0], h - bb[3], bb[2], h - bb[1]] for bb in boxes]
                masks = [np.flipud(m).copy() for m in masks]
            if np.random.rand() > 0.5:
                gamma = float(np.random.uniform(0.85, 1.15))
                table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
                img = cv2.LUT(img, table)

        # ---- scale jitter: gives the 100+ px band examples it otherwise lacks ----
        if self.scale_prob > 0 and np.random.rand() < self.scale_prob:
            lo, hi = self.scale_range
            s = float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))
            nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
            interp = cv2.INTER_LINEAR if s > 1.0 else cv2.INTER_AREA
            img = cv2.resize(img, (nw, nh), interpolation=interp)
            ignore = cv2.resize(ignore, (nw, nh), interpolation=cv2.INTER_NEAREST)
            kb, km, ka = [], [], []
            for bb, m, ar in zip(boxes, masks, areas):
                m2 = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
                ar2 = ar * s * s
                if 2.0 * np.sqrt(ar2 / np.pi) < _FLOOR_PX or not m2.any():
                    # shrunk below the floor: not a target, and not background either
                    ignore = np.maximum(ignore, m2)
                    continue
                kb.append([min(bb[0] * s, nw), min(bb[1] * s, nh),
                           min(bb[2] * s, nw), min(bb[3] * s, nh)])
                km.append(m2)
                ka.append(ar2)
            boxes, masks, areas = kb, km, ka
            h, w = nh, nw

        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            masks_t = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)
            labels_t = torch.ones((len(boxes),), dtype=torch.int64)
            areas_t = torch.as_tensor(areas, dtype=torch.float32)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            masks_t = torch.zeros((0, h, w), dtype=torch.uint8)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            areas_t = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": areas_t,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "ignore_mask": torch.as_tensor(ignore, dtype=torch.uint8),
        }

        img_t = torch.as_tensor(img.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        if self.transforms is not None:
            img_t, target = self.transforms(img_t, target)
        return img_t, target


def collate_fn(batch):
    return tuple(zip(*batch))


class TiledDataset(Dataset):
    """Slice oversized frames into native tiles without downsampling.

    A bubble cut by a tile boundary is moved into the ignore mask rather than
    dropped.  Dropping it made the seam look like empty water, which is the same
    mistake as leaving an occluded bubble unlabelled, one scale down.
    """

    def __init__(self, base: Dataset, tile: int = 512, stride: Optional[int] = None,
                 min_visible: float = 0.6, random_tiles: bool = False):
        self.base = base
        self.tile = int(tile)
        self.stride = int(stride) if stride else int(tile) // 2
        self.min_visible = float(min_visible)
        # Training: draw a fresh window per item. Scale jitter changes the image
        # size, so the grid precomputed from the file on disk no longer fits it.
        self.random_tiles = bool(random_tiles)
        self.index: List[Tuple[int, int, int]] = []

        for i in range(len(base)):
            img_p = None
            if hasattr(base, "samples"):
                img_p, _ = base.samples[i]
            elif hasattr(base, "dataset") and hasattr(base.dataset, "samples"):
                img_p, _ = base.dataset.samples[base.indices[i]]

            if img_p is not None:
                img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                h, w = img.shape[:2]
            else:
                img_t, _ = base[i]
                h, w = img_t.shape[1:]

            if h <= self.tile and w <= self.tile:
                self.index.append((i, 0, 0))
                continue
            for y in self._origins(h):
                for x in self._origins(w):
                    self.index.append((i, x, y))

    def _origins(self, n: int) -> List[int]:
        if n <= self.tile:
            return [0]
        xs = list(range(0, n - self.tile + 1, self.stride))
        if xs[-1] != n - self.tile:
            xs.append(n - self.tile)
        return xs

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int):
        i, x, y = self.index[k]
        img_t, target = self.base[i]
        _, h, w = img_t.shape
        if h <= self.tile and w <= self.tile:
            return img_t, target

        if self.random_tiles:
            x = int(np.random.randint(0, max(w - self.tile, 0) + 1))
            y = int(np.random.randint(0, max(h - self.tile, 0) + 1))
        x = min(x, max(w - self.tile, 0))
        y = min(y, max(h - self.tile, 0))
        sub_img = img_t[:, y: y + self.tile, x: x + self.tile]

        ign_full = target.get("ignore_mask")
        if ign_full is None:
            ign = np.zeros((self.tile, self.tile), dtype=np.uint8)
        else:
            ign = ign_full[y: y + self.tile, x: x + self.tile].numpy().copy()

        masks = target["masks"].numpy()
        keep_masks, keep_boxes = [], []
        for m in masks:
            full = float(m.sum())
            if full <= 0:
                continue
            tile_m = m[y: y + self.tile, x: x + self.tile]
            vis = float(tile_m.sum())
            if vis <= 0:
                continue
            if vis / full < self.min_visible:
                # cut by the seam: not a bubble the model can be scored on here,
                # and not background either
                ign = np.maximum(ign, tile_m)
                continue
            ys, xs = np.nonzero(tile_m)
            x1, x2 = float(xs.min()), float(xs.max() + 1)
            y1, y2 = float(ys.min()), float(ys.max() + 1)
            if max(x2 - x1, y2 - y1) < 4.0:
                ign = np.maximum(ign, tile_m)
                continue
            keep_masks.append(tile_m)
            keep_boxes.append([x1, y1, x2, y2])

        out = {k2: v for k2, v in target.items()
               if k2 not in ("boxes", "labels", "masks", "area", "iscrowd", "ignore_mask")}
        if keep_boxes:
            boxes = torch.as_tensor(keep_boxes, dtype=torch.float32)
            out["boxes"] = boxes
            out["masks"] = torch.as_tensor(np.stack(keep_masks), dtype=torch.uint8)
            out["labels"] = torch.ones((len(keep_boxes),), dtype=torch.int64)
            out["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            out["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            out["masks"] = torch.zeros((0, self.tile, self.tile), dtype=torch.uint8)
            out["labels"] = torch.zeros((0,), dtype=torch.int64)
            out["area"] = torch.zeros((0,), dtype=torch.float32)
        out["iscrowd"] = torch.zeros((len(out["boxes"]),), dtype=torch.int64)
        out["ignore_mask"] = torch.as_tensor(ign, dtype=torch.uint8)
        return sub_img, out
