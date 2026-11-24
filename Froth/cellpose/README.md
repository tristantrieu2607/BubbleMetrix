# Cellpose on froth

[Cellpose](https://github.com/MouseLand/cellpose) (Stringer et al., 2021, *Nature Methods*) predicts, for every pixel, whether it belongs to an object and which way that object's centre lies. Pixels that point to the same centre form one object. It was built for packed, touching cells, which is the froth problem: separating, not detecting.

The default pretrained model in Cellpose 4 is Cellpose-SAM. This script uses it **as is**: no training, no diameter hint, default thresholds. That makes it a zero-shot baseline, directly comparable with BubSAM3.

```powershell
.\run.ps1                                     # frame_0000, BubSAM3's static ROI, then measure
.\run.ps1 -Image <png> -NoROI -MmPerPx 0.10956 # a crop
.\run.ps1 -Gates                              # + BubSAM3's shape gates (post-filter only)
.\run.ps1 -Diameter 14                        # tuned, NOT zero-shot - recorded in _run.json
```

## Outputs (`results\`)

| file | contents |
|---|---|
| `<prefix>_masks.csv` | same columns as BubSAM3, plus `touches_roi_edge` and `passes_gates` |
| `<prefix>_instances.npz` | outlines, same format as BubSAM3 |
| `<prefix>_labels.npy` | Cellpose's own label map, untouched. The evaluation scores this, because it is exact |
| `<prefix>_overlay.png` | outlines on the ROI |
| `<prefix>_run.json` | version, model, every setting changed from default, runtime |

The ROI comes from BubSAM3's own loader (`..\roi\run_bubsam.py`), so both methods always segment the same pixels. Measurement is BubSAM3's `measure_bubbles.py`, so the calibration and the D32 formula are shared too.

## Fine-tuning (not done yet)

Only if zero-shot Cellpose scores badly on a **fully labelled** case. Train on corrected patches that are **not** in `eval\`. A bubble used for training can never be used for testing.
