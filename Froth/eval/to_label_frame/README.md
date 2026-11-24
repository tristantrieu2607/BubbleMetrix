# Outline bubbles on one whole frame

`180WW_t150s_roi.png` is the measured (in-focus) region of `samples\froth\180WW.MP4` at t = 150 s, exactly what the GUI measures. You haven't seen a model overlay of this frame, so keep it that way until you're done.

Double-click **`Froth\LABEL_FROTH.bat`**.

1. Outline each bubble freehand, on the middle of its dark rim. Let go and the ends are joined.
2. Front-layer bubbles only. Skip the ghost rings you see through a bubble.
3. The region is split into a 4 × 4 grid. When **every** bubble in a tile is outlined, hover over the tile and press **T**; it turns green. Only green tiles are scored. Inside them, the labels are complete, which gives precision, recall, size error and the D32 of the same area.
4. About 260 bubbles in green tiles is enough, which is roughly 6–7 tiles. Spread them over the frame (centre, edges, top, bottom) instead of taking 7 neighbouring tiles.
5. A bubble belongs to the tile its centre is in. Outline bubbles that cross a tile border completely.

| | |
|---|---|
| left-drag | outline a bubble |
| right-click | delete a bubble |
| `U`, then click | unsure (not scored); `D` back to drawing |
| `T` | tile under the cursor complete / not complete |
| wheel, `+` `-` | zoom |
| middle-drag, arrow keys | pan |
| hold `Space` | hide outlines |
| `C` | view: plain / enhanced / pixels |
| `F` | flicker the neighbouring frames |
| `Ctrl+Z` / `Ctrl+Y` | undo / redo |
| `S` | save (it also saves every 10 bubbles and on close) |
