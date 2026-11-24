# Froth patches to label

Four 128 × 128 px patches from frames **nobody has run a method on**: t = 240, 300, 360 and 420 s of `samples\froth\180WW.MP4`. **Label them before you look at any overlay of these frames.**

| patch | where | why |
|---|---|---|
| `180WW_froth_t240s_x360y767` | centre of the lit patch | best-case froth |
| `180WW_froth_t300s_x270y880` | lit patch, lower left | best case, different frame |
| `180WW_froth_t360s_x185y640` | in focus, outside the lit patch | dimmer, softer; bubseg doesn't measure here |
| `180WW_froth_t420s_x300y584` | straddles the hazy band | where BubSAM3 invented masks |

`_where_the_patches_are.jpg` shows each patch on its frame.

## How: draw the walls, not circles

Double-click **`Froth\LABEL_FROTH.bat`**.

1. **Trace the dark wall between two bubbles** (left-drag). Every area the walls close off fills with a colour, and each colour is one bubble. If two bubbles share a colour, a wall has a gap: close it.
2. **Haze, glass, anything you can't call:** press `I`, then click that area. It turns grey and is never scored.
3. **Maybe-a-bubble:** press `U`, then click it (yellow). It isn't counted for or against a method.
4. **Right-drag** erases walls. `Ctrl+Z` undoes. Hold `Space` to see the image without your walls.
5. **When it's too blurry to see a wall:** press `F` to flicker through the neighbouring video frames. The froth drifts a little each frame, and walls stand out in motion. `[` and `]` step one frame at a time. Drawing always snaps back to the labelled frame. `C` cycles the view: enhanced (default), smooth, raw pixels. The dimmed border around the patch is context only.
6. **If you still can't see a wall, don't guess.** Mark that area `U` (unsure) or `I` (ignore). Ground truth can only contain what a person can resolve.
7. `→` saves and moves to the next patch.

You don't need to do anything about:

- **Bubbles cut by the patch edge.** They show faint and aren't scored.
- **The exact edge position.** Each wall is shared between its two bubbles, so edges land on the middle of the wall automatically.

Expect 15–30 min per patch. Labels save to `to_label\labels\` and the annotator reopens where you left off.

When all four are done, tell Claude. They get added as fully labelled cases, Cellpose is scored on them (precision included), and they appear in the GUI under **Froth → Benchmark Crop**.
