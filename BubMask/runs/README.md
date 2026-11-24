# runs/ — what is in here, and what it is worth

Only results produced by the **fixed** pipeline live here. Everything from the
earlier pipeline was moved to `_quarantine/superseded/` on 2026-09-07, because it
was produced under conditions that make its numbers unusable:

* scored on crops the model trained on;
* the rotor_diluted set counted 8 images twice, so "held-out" crops were
  byte-identical copies of training crops;
* the pulp split was by frame, not by clip, so every clip appeared on both sides;
* IoU was **box** IoU, which reads far higher than mask IoU on near-concentric
  circles — the reported 93% was not the same quantity as the 0.875 human ceiling
  it appeared to beat;
* predicted diameter was min-enclosing-circle of the convex hull, ground truth was
  the stamped radius — two different measurands;
* microns were emitted from a calibration marked provisional.

Nothing in `_quarantine/superseded/` should be quoted. It is kept only so the
history is auditable.

---

## rotor_raw — the one verified result

**`rotor_raw/`** — model trained 2026-09-07 under the clip-level split in
`data/split_clips.json`. 16 training crops (65 instances), 4 validation crops
(1 clip, held out of training for checkpoint selection only), 3 clips reserved and
never seen. `best.pt` selected on validation loss; `last.pt` is epoch 20.

**`eval_rr_base/`** — scored on the 3 reserved clips, 10 crops, 48 hand-labelled
bubbles. Mask IoU, equivalent-area diameter on both sides, ignore regions applied
to unmatched predictions only.

| | |
|---|---|
| recall | **0.521** |
| precision | **0.641** |
| F1 | 0.575 |
| mean mask IoU | **0.853** (human repeatability 0.875) |
| D32 ratio | **0.798** (provisional calibration — pixels are the honest unit) |
| missed entirely / touched but unmatched | 14 / 9 |

Recall by size: 0.64 (40–70 px), 0.62 (70–100 px), **0.14 (100–140 px)**, 0.00 (140+).

The failure is the large end, and it carries the D32: D32 is volume-weighted, so
missing big bubbles is what makes predicted D32 20% low. Mean IoU 0.853 against a
human 0.875 says segmentation is fine — this is a detection problem. 14 of the 23
misses were never proposed at all, which points at the anchor ladder rather than
post-processing.

## rotor_diluted — no verified result exists

There is no honest score for rotor_diluted and there cannot be one from that data
alone: it is a **single clip**, so no clip-level split is possible. Its 8 images
and 29 labelled bubbles are good **training** data. Any recall or IoU figure
computed on it is a fit-to-training number.

The earlier "98.1% recall / 92.5% IoU on held-out test crops" came from that
regime and is in `_quarantine/superseded/runs_eval_rotor_diluted/`. Do not use it.
