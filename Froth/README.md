# Froth — bubble segmentation for the froth zone

Everything for the froth footage (`samples\froth\180WW.MP4`) in one place.

**Method: Cellpose, zero-shot** (pretrained, no training, default settings). On the hand-labelled crop it found 26/26 bubbles, mean IoU 0.80, diameter ratio 0.99. BubSAM3 scored 23/26 and IoU 0.71 and is retired (now in `progress\Froth\roi\`); `roi\` keeps only the in-focus-region and measurement code Cellpose reuses.

Froth is a different problem from pulp. Every pixel in frame is bubble, so the job is **separating** packed bubbles, not **detecting** sparse ones.

```
Froth\
  setup.ps1, requirements.txt   one environment: Froth\.venv (torch CPU, cellpose 4.x)
  RUN_CELLPOSE_VIDEO.bat        double-click: setup + Cellpose on 5 sampled frames + the labelled crop
  LABEL_FROTH.bat               double-click: whole-frame annotator (eval\to_label_frame)
  cellpose\                     run_cellpose.py (batch), gui_worker.py (used by the GUI)
  eval\                         froth_eval.py scoring, frame_annotator.py, cases.json, d32_convergence.py
  roi\                          crop_measurable.py (in-focus band), run_bubsam.py, measure_bubbles.py
```

## Using it (scientists)

Run the main GUI as before (`launch_gui.bat` at the drive root):

1. Choose **🫧 Froth** and pick the video.
2. Press **Measure Full Video**.

That's all. The GUI segments up to 40 frames spread over the clip and stops once D32 is known to ±5%, and reports D32, D43, median and the volume distribution in **mm**. There's nothing to tune. Scale comes from the 55 mm inner tube width, detected in every frame. Only the in-focus band is measured (orange box), and bubbles cut by its edge are excluded. Cellpose runs in its own environment (`Froth\.venv`) in the background, and the first run takes about 30–60 s to load the model.

## Setup (once)

```powershell
cd "D:\Taste of Research\Froth"
.\setup.ps1
```

This creates `Froth\.venv`, which every script here uses. It downloads about 1 GB from PyPI. Cellpose fetches its model weights (about 1 GB) the first time it runs. CPU only.

## Run from the command line

```powershell
cellpose\run.ps1               # one frame, same region and measurement as the GUI
```

## Compare methods

```powershell
eval\run_eval.ps1               # BubSAM3 + Cellpose + bubseg on the labelled crop, then a table
```

This writes `eval\results\<case>\comparison.md` and an overlay per method. Scoring rules: one-to-one matching at mask IoU ≥ 0.5, the same as the pulp harness. See `eval\README.md`.

**Current result** (case `f299_partial`: 26 hand labels, all ≥ 17 px, partly labelled, so no precision):

| method | found | recall (95% CI) | mean IoU | diameter ratio |
|---|---|---|---|---|
| BubSAM3 | 23/26 | 0.88 (0.71–0.96) | 0.71 | 0.93 |
| bubseg (ROI = whole crop) | 13/26 | 0.50 (0.32–0.68) | 0.61 | 0.88 |
| Cellpose (zero-shot) | *run `eval\run_eval.ps1 -Only cellpose`* | | | |

## What has NOT moved, and why

`old_stuff\bubSAM_froth\` is still the copy the main GUIs use. `bubble_gui.py`, `engines\engine_froth.py` and the old web server load its checkpoint by path. It stays until those paths are repointed here. **New froth work goes in `Froth\`**; treat `old_stuff\bubSAM_froth\` as frozen.

`handcount.csv` / `handcount_rois.zip` were not copied. They come from a different image (`benchmark_bubbles_800x534_8bit.tif`) and are not froth ground truth.

Build notes: `claude/froth-build-notes.md` in the project.
