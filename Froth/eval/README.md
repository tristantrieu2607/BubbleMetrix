# Froth evaluation

One harness, `froth_eval.py`, scores every method against hand labels by the same rules as the pulp evaluation:

- Matching is greedy one-to-one by **mask IoU ≥ 0.5**.
- Ground truth with no overlapping prediction is "missed entirely". Ground truth that a prediction overlaps but doesn't match is "touched but unmatched". The first points at the detector, the second at the outline.
- Predictions below the label floor (`floor_px`), or mostly inside an ignore region, are not counted as false positives.
- Bubbles marked `unsure` are left out of recall, and a prediction that matches one isn't a false positive.

```powershell
.\run_eval.ps1                                  # run + score everything on f299_partial
..\.venv\Scripts\python.exe froth_eval.py score --case f299_partial --pred <file> --name <label>
..\.venv\Scripts\python.exe froth_eval.py table --case f299_partial
```

`--pred` accepts an `*_instances.npz` (BubSAM3, Cellpose) or an integer label map `*.npy`.

## Cases (`cases.json`)

| case | image | labels | precision? |
|---|---|---|---|
| `f299_partial` | 384 px crop of frame 299 | 26 stamped circles, all ≥ 17 px | **no**: partly labelled |

A partly labelled case gives recall, IoU and size bias only. An unmatched prediction might be a real bubble nobody stamped, so precision is never computed for one.

## Adding a fully labelled case (the next step)

1. Cut a patch (around 128 px) from a frame not used before (not 0, not 299). **Label it before looking at any overlay of it.**
2. Label it with the stamp tool. Stamp every bubble you can trace down to the floor, ignore-box anything you'd have to guess, and use the ellipse stamp (`M`). Put the edge on the middle of the dark rim.
3. Copy the patch PNG and its `_labels.json` into `data\`, then add an entry to `cases.json` with `"fully_labelled": true`. Set `label_offset` to `[0, 0]` if the labels were drawn on the patch itself.
4. `.\run_eval.ps1 -Case <name>`. Precision, F1 and the whole-image D32 ratio now appear in the table.

Four such patches, about 250 bubbles, is the target.

Results go in `results\<case>\`: one `*_score.json` and `*_overlay.png` per method, plus `comparison.md`. Method outputs for a case go in `results\<case>\runs\`, never in `roi\results` or `cellpose\results`.
