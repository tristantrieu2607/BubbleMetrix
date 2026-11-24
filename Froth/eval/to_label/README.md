# Froth patches to label (target: 260 bubbles)

Five 128 × 128 px patches from frames of `samples\froth\180WW.MP4` that you have never seen a model overlay of. **Label them before you look at any overlay of these frames.**

| # | patch | where |
|---|---|---|
| 1 | `180WW_froth_t150s_x440y880` | lit, lower right |
| 2 | `180WW_froth_t240s_x360y767` | centre of the lit patch |
| 3 | `180WW_froth_t300s_x270y880` | lit, lower left |
| 4 | `180WW_froth_t360s_x185y640` | in focus, outside the lit patch |
| 5 | `180WW_froth_t420s_x300y584` | straddles the hazy band |

`_where_the_patches_are.jpg` shows each patch on its frame. The sidebar keeps a running total (**All patches: N / 260**).

## How (froth_annotator.py, 2026-10-12)

Double-click **`Froth\LABEL_FROTH.bat`**.

Each bubble here is a dome lit from one side: a sharp dark crescent on one side, a soft edge on the other.

1. **Trace the sharp part of the rim, on the middle of the dark line.** Let go and the tool fits the circle (an ellipse if you traced more than about 200°). A third of the rim is the minimum; half is better.
2. **Odd shape?** Trace all the way round so the ends meet, and your outline is kept as drawn.
3. **Label only bubbles you're sure of.** You don't need every bubble. These labels measure recall, outline accuracy and size accuracy. Precision comes from the separate check below.
4. **Front layer only.** Skip the ghost rings you see through a bubble's face.
5. **Only a sliver visible?** Draw it, then press `U` and click it (unsure, not scored).

| | |
|---|---|
| left-drag | trace a rim |
| right-click a bubble | delete it |
| `U`, then click | toggle unsure (`D` back to drawing) |
| `F` | flicker the neighbouring frames (rims stand out in motion) |
| `[` `]` | step one frame |
| `C` | view: enhanced / smooth / raw pixels |
| hold `Space` | hide overlay |
| `Ctrl+Z` / `Ctrl+Y` | undo / redo |
| `→` / `←` | save and go to next / previous patch |

Labels you drew with the old border tool are converted the first time you open the new tool. Closed loops are kept, and open arcs are circle-fitted where they cover enough of the rim. Check them, and right-click any that came out wrong. The old files are in `labels\_border_mode_backup\`.

## Precision check (10–15 min)

Double-click **`Froth\CHECK_PRECISION.bat`**. It shows 210 Cellpose masks from other frames, one at a time: the raw crop on the left, the outline on the right. Press:

- `Y` = one real bubble, outline roughly on its edge
- `N` = not a bubble
- `M` = covers 2+ bubbles
- `P` = only part of a bubble
- `U` = can't tell

Every answer saves immediately, so you can stop any time.

When both are done, tell Claude.
