## Abstract {.abstract-h}

<div class="abstract" markdown="1">

Bubble size controls how much mineral flotation recovers, yet it is still measured from probe video largely by hand. This project built and validated a deep-learning pipeline that outlines in-focus bubbles in pulp-zone video from a novel flotation cell and reports the Sauter mean diameter (D32).

A labelling protocol separated measurable from unmeasurable bubbles (focus gate, 12 px size floor, ignore regions for part-hidden bubbles), and 383 bubbles were hand-labelled on 35 full frames. A Mask R-CNN with bubble-shaped anchors, no image resizing and an ignore-aware loss was always scored on clips it had never seen.

Against the previous model, recall on large bubbles (100–140 px) rose from 0.62 to 0.90 and D32 error fell from 18% to 4%, level with the agreement between two hand-labelling passes (D32 ratio 0.96). In leave-one-clip-out cross-validation over all 8 rotor conditions (377 bubbles), recall averaged 0.92 ± 0.06 and the D32 ratio 0.99 ± 0.03. A manual audit showed that only 16% of unmatched detections were genuine model errors; most were real bubbles missed during labelling.

In the froth zone, Cellpose-SAM used without any froth training found 86% of 51 hand-outlined bubbles with 98% precision in the labelled area. It runs in under a minute per frame and gives a D32 of 2.16 mm (2.06–2.30 mm per frame over 7 minutes). Both pipelines are packaged in BubbleMetrix, a desktop tool that stops measuring once D32 is known to ±5% and lets researchers fine-tune the models on new footage with a built-in before/after check.

</div>

<div class="keyfig" markdown="1">
<div><b>0.99 ± 0.03</b><span>D32 ratio, model / hand, on 8 unseen conditions</span></div>
<div><b>0.62 → 0.90</b><span>recall on large (100–140 px) bubbles</span></div>
<div><b>16%</b><span>of unmatched detections were real model errors</span></div>
<div><b>2.16 mm</b><span>froth D32, zero-shot Cellpose, &lt; 1 min per frame</span></div>
</div>

## 1 Introduction

Flotation separates valuable minerals from waste rock by attaching the mineral particles to rising air bubbles. Smaller bubbles carry more surface area per unit of gas, so bubble size directly sets how much mineral a cell can capture.

Engineers summarise a bubble population with the Sauter mean diameter, the diameter of a bubble with the same volume-to-surface ratio as the whole population:

<div class="eq">D<sub>32</sub> = Σ d<sub>i</sub><sup>3</sup> / Σ d<sub>i</sub><sup>2</sup></div>

The project's novel flotation cell is filmed by a probe in the pulp and by a camera on the froth column. Measuring bubbles from that video by hand is slow and subjective. Most pulp bubbles are out of focus or overlapping, and only sharp ones can be sized accurately.

The project inherited BubMask-Fiji, a Mask R-CNN tool built by a previous student. It had never been tested against independent hand counts on this footage (Section 4).

**Aims**

1. Automate bubble sizing from pulp-zone probe video.
2. Match hand measurement for D32, tested on operating conditions the model has never seen.
3. Extend the method to the froth zone.
4. Deliver a tool researchers can use without any machine-learning knowledge.

## 2 Background

**Why D32 matters.** The flotation rate constant is close to linear in the bubble surface area flux, S<sub>b</sub> = 6 J<sub>g</sub> / D32, where J<sub>g</sub> is the superficial gas velocity (Gorain et al., 1997, 1999; Finch et al., 2000). At a fixed air rate, halving D32 doubles the bubble surface available to collect particles. That is why D32, rather than a mean or median diameter, is the number flotation researchers compare.

**How bubble size is measured.** Image analysis of a sampled bubble stream is the most common method (Gomez & Finch, 2007; Grau & Heiskanen, 2005). Bubble Analyser, an open-source classical pipeline, reports under 5% D32 error against manual segmentation on clean images (Mesa et al., 2022); this project uses that as the bar for acceptable D32 error. Machine vision is also established for monitoring flotation froth (Aldrich et al., 2010).

**Deep learning for bubbles.** Thresholding and watershed (Vincent & Soille, 1991) fail on dense, overlapping bubbly flows. CNN detection and instance segmentation are now the main direction (Cerqueira & Paladino, 2021; Wang et al., 2021; He et al., 2022; Hessenkemper et al., 2022). Mask R-CNN (He et al., 2017) underlies BubMask (Kim & Park, 2021) and has handled sub-millimetre bubbles at up to 20% gas holdup (Cui et al., 2022).

**General-purpose segmenters.** Segment Anything (SAM; Kirillov et al., 2023) and Cellpose (Stringer et al., 2021; Pachitariu et al., 2025) segment objects they were never trained on. Cellpose was built for packed cells, which makes it a natural candidate for packed froth bubbles.

## 3 Data

### 3.1 Pulp footage

Thirteen probe clips were supplied: 8 rotor clips, 1 diluted rotor clip and 4 Venturi clips. No further footage exists, so these clips are the whole dataset.

| Property | Value |
| --- | --- |
| Frames | 1024 × 1024, 8-bit greyscale, uncompressed, 101 frames per clip at 30 fps |
| Field of view | circle, centre (524, 508), radius 601 px (measured) |
| Scale | 303.858 px/mm = 3.291 µm/px, from the provider's quoted 3.37 mm field width; **provisional**, never measured through the probe |
| Frame independence | rotor clips decorrelate by 5 frames; Venturi and diluted clips are still correlated at 10 |
| Imaging regimes | rotor: dark and crowded (median grey 29–48); Venturi and diluted: bright, 6–54% of pixels clipped at 255 |
| Typical rotor bubble | hand-measured median 73.5 px ≈ 244 µm (range 49.5–111 px) |

The 8 rotor clips, all at 90 ppm frother, vary rotor type (Coarse, Fine, Med), rotor speed (1260 or 1440 rpm), air setting (200 or 300), water setting (900 or 800) and an inline configuration. Short names used below: Coarse 200air, Coarse 300air, Fine 200air, Fine inline, Fine 300air, Fine 1440rpm, Med 200air and Med 300air.

### 3.2 Froth footage

One froth video (720 × 1280 portrait, 30 fps, about 8 minutes, H.264 from a phone) was supplied by Imtiaz Shah. Only about 15% of the column is well lit, and only a band of each frame is in focus.

The scale comes from the column itself: 55 mm is the tube's inner width. The inner glass surfaces are found on every frame, about 471 px apart (468–472 over 24 frames), giving 0.1168 mm/px.

### 3.3 Labelled pulp dataset

The rotor data are split by clip, never by frame, in a fixed file.

| Split | Clips | Images | Bubbles |
| --- | --- | --- | --- |
| Train | Coarse 200air, Fine 200air, Fine 1440rpm, Med 200air | 16 crops + 22 full frames | 328 |
| Validation | Fine 300air | 4 crops + 4 full frames | 58 |
| Test | Coarse 300air, Fine inline, Med 300air | 9 full frames (+10 retired crops) | 76 (124 with the crops) |

The 35 full frames were labelled between 27 Sep and 1 Oct 2026: 383 bubbles and 124 ignore boxes. Test frames were labelled blind, before any model output was viewed, and at least 5 frames from anything viewed earlier.

## 4 Audit of the inherited tool

The inherited tool reported an F1 of 0.996, but an independent hand count found 5% recall and a D32 2.3 times too high. The score had been computed against labels the same model produced, so it measured the model against itself. Every problem below was found before any new model was trained, and each fix carried into the new pipeline.

| Problem | Effect | Fix |
| --- | --- | --- |
| Model pre-labels used as ground truth for the same model | F1 0.996 reported; real recall 0.05, D32 2.3× high | Score only against independent hand labels |
| Scale wrong by 1.66× (183 px/mm assumed vs 303.858) | Every diameter 66% too large | Report pixels until a scale is measured |
| Images resized, COCO anchors (smallest ≈ 67 px in original pixels) | Most bubbles smaller than any anchor | No resizing; bubble-sized anchors |
| Ignore regions never reached the loss | Part-hidden bubbles trained as background | Ignore-aware loss |
| Box IoU reported as IoU | 93% claimed against a mask-IoU human ceiling of 0.875 | Mask IoU |
| Two different diameter definitions | 8.9% bias | Equivalent-area diameter on both sides |
| Duplicated images and frame-level splits | Test images leaked into training | Clip-level split in a fixed file |
| No validation loop | Checkpoint chosen on training loss | Held-out validation clip |
| Reported figures differed from files on disk (99.1% vs 73.8% recall) | Unverifiable claims | Numbers only from run outputs |

The principles kept throughout: split by clip, hand-count before looking at overlays, report D32 and not just IoU, and refuse microns without a measured scale.

## 5 Pulp method

### 5.1 Labelling protocol

The rule: stamp what you can trace, ignore what you would have to guess, and leave defocus alone.

| Rule | Decision | Reason |
| --- | --- | --- |
| Focus gate | In focus (sharp edge, dark rim, bright centre) → label; defocused → background | Defocused bubbles image larger; including them inflated D32 2.3× before |
| Size floor | 12 px = 40 µm minimum | Set by window debris (largest speck 10.5 px) |
| At-limit flag | 12–20 px flagged | ±1 px on the radius is ±17% on diameter and ±59% on volume |
| Occlusion | Sharp but part-hidden → ignore region, never traced or completed | Neither measurable nor background |
| Empty images | Must be confirmed empty | "Skipped" is not "no bubbles" |
| Focus judged on the raw image | Contrast-enhanced view only for finding candidates | Enhancement makes defocus look sharp |

The same annotator labelled the same 4 crops twice to set the ceiling any model is compared against. Count agreement was 0.82, mean mask IoU on matched bubbles 0.875, median per-bubble diameter disagreement 3.2% and the D32 ratio between passes 0.96. Choosing which bubbles count was the unstable part; sizing was stable once a bubble was picked.

### 5.2 Model

| Component | Choice | Reason |
| --- | --- | --- |
| Architecture | torchvision Mask R-CNN, ResNet-50-FPN-v2, COCO-pretrained (Lin et al., 2014; Li et al., 2021) | Instance masks give a per-bubble outline and so a diameter |
| Input scale | No internal resizing | Resizing breaks the pixel scale and shrinks small bubbles below the anchors |
| Anchors | Aspect ratio 1 only, 3 sizes per level at √2 spacing, 8–256 px | Bubbles are round; worst-case anchor IoU about 0.71 instead of 0.50 |
| Ignore-aware loss | Anchors and proposals mostly inside ignore regions removed from the background loss | Otherwise part-hidden bubbles train as background |
| Augmentation | Flips, gamma, scale jitter 0.75–2.0× (p = 0.5) | Supplies the large-bubble examples the labels lack |
| Training | 512 px random tiles, AdamW (Loshchilov & Hutter, 2019), lr 1e-3, cosine schedule, batch 2, early backbone frozen, CPU | Full frames on a laptop CPU |
| Checkpoint | Lowest validation loss on the held-out clip (epoch 16 of 20) | Never selected on training loss |

### 5.3 Evaluation

- Split by clip, never by frame: frames within a clip are near-duplicates.
- A match needs mask IoU ≥ 0.5.
- Diameter is the equivalent-area diameter, d = 2√(A/π), for labels and predictions alike.
- The headline quantity is the D32 ratio, model D32 divided by hand D32.
- Predictions are matched before any filtering, and the 12 px floor applies to predictions too.
- Bubbles cut by the image border are not scored on either side (edge rule).
- Only full frames are scored, at confidence 0.40, chosen for D32 fidelity.

### 5.4 Cross-validation

Leave-one-clip-out over the 8 rotor clips: train on 7, test on the 8th, so every clip is tested once (Roberts et al., 2017). The recipe was frozen beforehand: 12 epochs, confidence 0.40, the same anchors, scale jitter and edge rule, and the same COCO starting point for every fold. Each fold took 2.5–3.5 hours on CPU. Results are reported as the mean ± SD across clips and pooled over bubbles.

## 6 Pulp results

The final model sizes bubbles as consistently as a repeat hand label: on unseen clips its D32 is within 1.5% of the hand value on average. Precision against the labels is low, but the audit in 6.3 shows the labels, not the model, account for most of that gap.

### 6.1 Previous and final model on the same 124 held-out bubbles

| | Previous model (7 Sep) | Final model | Human repeatability |
| --- | --- | --- | --- |
| Recall | 0.734 | **0.847** | 0.82 |
| Precision (against labels) | 0.417 | **0.461** | — |
| Mean mask IoU | 0.835 | **0.891** | 0.875 |
| D32 ratio | 0.819 | **0.957** | 0.96 |

The final model is better on every measure, precision included, so the recall gain does not come from predicting more. The large-bubble failure is fixed (Figure 1): recall on 100–140 px bubbles rose from 0.62 to 0.90. Because D32 weights by volume, that moved the D32 error from 18% low to 4% low. Three changes were made together (full-frame labels, scale jitter, new anchors), so the gain is reported as their combined effect.

<figure><img src="f_size.png"><figcaption><b>Figure 1.</b> Recall by bubble size, previous vs final model, on the same 124 held-out bubbles. The highlighted band is the size range that moved D32 most.</figcaption></figure>

For context, earlier approaches scored far lower on the first test crops: watershed reached recall 0.03 and a D32 ratio of 0.43; SAM with automatic masks reached recall 0.52 and a D32 ratio of 0.77, because it split large bubbles into fragments.

### 6.2 Cross-validation over all 8 rotor conditions

Across the 8 held-out clips (377 bubbles), recall was 0.921 ± 0.058, mean mask IoU 0.888 ± 0.011 and the D32 ratio 0.985 ± 0.035. Pooled over bubbles, recall was 0.910 and the D32 ratio 0.992. The D32 ratio stayed within 9% of the hand value on every clip and within 3% on six of eight (Figure 2). The worst clip was Fine inline at 0.91. Recall never fell below 0.83 (Med 300air, the smallest test set).

<figure><img src="f_cv.png"><figcaption><b>Figure 2.</b> Leave-one-clip-out results: each clip was held out while the model trained on the other seven. The shaded band is ±5% of the hand-measured D32.</figcaption></figure>

Pooled recall by size was 0.74 for 12–40 px, 0.89 for 40–70 px, 0.93 for 70–100 px and 0.97 for 100–140 px. Precision against labels sat between 0.38 and 0.50 on every clip (pooled F1 0.60), consistent with the audit below. The cross-validation figures are higher than the fixed split because each fold trains on 7 clips instead of 4 and the edge rule removes truncated bubbles. Both are legitimate differences.

### 6.3 What the "false positives" really were

All 96 unmatched detections on the 9 test frames were inspected by hand, with the labels left untouched (Figure 3).

<figure><img src="f_audit.png"><figcaption><b>Figure 3.</b> Hand audit of the 96 detections that matched no label. Only the defocused group is a genuine model error.</figcaption></figure>

Counting missed in-focus bubbles as correct raises precision from 0.40 to 0.67, and also counting part-hidden bubbles raises it to 0.80. Only 15 of 159 detections (9%) were real mistakes. A higher confidence threshold would not help: at 0.8 it removes 11 of the 15 defocused errors but also 16 of the 44 real bubbles.

<figure class="wide"><img src="pulp_gt_vs_model.png"><figcaption><b>Figure 4.</b> Hand labels (left) and model outlines (right) on Fine inline frame 23, a clip held out in cross-validation. Most extra model outlines are sharp bubbles that were not labelled.</figcaption></figure>

### 6.4 How many bubbles are enough?

D32's 95% confidence interval is estimated by the delta method for a ratio, taking the larger of a bubble-level and a frame-level (clustered) error. A measurement counts as stable when the interval is within ±5% of D32, from at least 3 frames, at two frames in a row. On the RotMed 300air clip, D32 was stable after 137 bubbles from 8 frames (254.8 µm ±4.7%). The Venturi 500 air, 60 ppm clip was not stable after all 21 sampled frames (355 bubbles, 446 µm ±9.7%); the model was never trained on Venturi footage.

## 7 Froth segmentation

Cellpose-SAM, used zero-shot with default settings, finds froth bubbles reliably and gives a stable D32 of 2.16 mm. Froth is a different problem from pulp: the bubbles are packed edge to edge, so the task is separating neighbours rather than finding isolated objects.

### 7.1 Method

- Cellpose-SAM (model cpsam_v2, cellpose 4.2.1.1), default settings, no froth training and nothing tuned.
- Only the in-focus band of each frame is measured, found automatically (for example 489 × 424 px at t = 0 s).
- Scale from the 55 mm inner tube width, measured on every frame (Section 3.2).
- Same rules as pulp: edge rule, equivalent-area diameter, matching at mask IoU ≥ 0.5.
- Ground truth: hand-drawn outlines; unresolved areas marked unsure and not scored.

<figure class="wide"><img src="froth.jpg"><figcaption><b>Figure 5.</b> Froth frame at t = 0 s (left) and Cellpose outlines in the in-focus band (right), 886 bubbles, no froth training.</figcaption></figure>

### 7.2 Accuracy against hand labels

| Test | Bubbles | Recall | Precision | Mean IoU | D32 ratio |
| --- | --- | --- | --- | --- | --- |
| Fully labelled patch, t = 240 s | 51 | 0.86 (95% CI 0.74–0.93) | 0.98 local / 0.68 strict | 0.67 | 1.10 |
| Partly labelled crop, frame 299 | 26 | 1.00 (0.87–1.00) | not computable | 0.80 | 0.97 |

Every labelled bubble on the patch is at least 80% covered by its own Cellpose mask. The 7 counted as missed are 6.6–10 px bubbles whose Cellpose mask is larger, so IoU falls below 0.5. Strict precision counts every unlabelled detection as an error; most of those are real bubbles in the unresolved area, so local precision (unresolved area ignored) is the fairer figure.

There is a constant boundary offset of +1.2 px in radius: the labels were traced on the bright cap, while Cellpose places the edge mid-film. It matters most for small bubbles (diameter ratio 1.31 under 9 px, 1.05 over 17 px). With the labels grown by 1 px, all 51 match, IoU rises to 0.77 and the diameter ratio to 1.08. This is a labelling convention to settle, not a detection error.

The earlier froth approaches were weaker on the same crop: watershed (bubseg) found 13 of 26 bubbles, and tiled SAM (BubSAM3) found 23 of 26 with IoU 0.71 at 27–40 minutes per frame.

### 7.3 Consistency over the video

<figure><img src="f_froth.png"><figcaption><b>Figure 6.</b> Froth D32 on 9 frames spread over 7 minutes (6,075 bubbles). The dashed line is the pooled D32; the shaded band is ±5% of it.</figcaption></figure>

Over 9 frames spread across 7 minutes, the pooled D32 is 2.16 mm and every frame lies within 2.06–2.30 mm. Cellpose takes 30–48 s per frame on a laptop CPU, 35–80 times faster than tiled SAM.

A bootstrap over the pooled bubbles puts the 95% interval at ±11% for 200 bubbles, ±5% for about 1,100 and ±2.1% for all 6,075. Frame-to-frame spread (4.4%) is slightly above what sampling alone explains (3.2%, p = 0.08), so the stopping rule also uses a frame-level interval. In the order the software visits frames, it stops after 6–7 frames with D32 within 2% of the 9-frame value. A whole t = 150 s frame with 160 more hand outlines is ready to score; it needs one Cellpose run on that frame.

## 8 Software: BubbleMetrix

BubbleMetrix wraps both pipelines in a Windows desktop app so a researcher picks a video and presses one button. It runs on a laptop CPU with no settings to tune; the user manual covers installation and every feature.

<figure class="wide"><img src="gui.png"><figcaption><b>Figure 7.</b> BubbleMetrix after measuring a pulp clip: outlines on the current frame, and D32 with its 95% interval in the result cards.</figcaption></figure>

| Feature | What it does |
| --- | --- |
| Pulp or froth | Mask R-CNN for pulp; Cellpose for froth, run in its own environment through a helper process. Settings are fixed at their validated values. |
| Scale | Adjustable. Pulp default 3.291 µm/px; froth measured from the 55 mm inner tube width on every frame. Changing it re-expresses results without re-running detection. |
| Measure Full Video | Pulp: every frame outlined, D32 from every 5th frame (bubbles repeat between consecutive frames). Froth: up to 40 frames spread over the clip, stopping once D32 is stable. |
| D32 Convergence | Running D32 with its 95% band against bubbles measured, and either "stable after N bubbles" or an estimate of how many more are needed. |
| Editor and Validation | Correct detections by hand; score the model against hand labels (recall, precision, IoU, D32 ratio). |
| Fine-tune Model | Add frames from new footage, correct the model's suggestions, see the current model's accuracy, then fine-tune in the background (30–60 min on CPU). The new model is used only if it beats the old one on held-back frames and is no worse on a fixed benchmark; one click reverts. |
| Exports | CSV per bubble (with frame number) and a JSON summary including the convergence result. |

The fine-tuning minimums are 15 frames and 150 bubbles for pulp, and 4 boxes and 300 bubbles for froth. The froth figure follows Cellpose 2.0, where 100–200 corrected objects suffice when the model pre-labels the images (Pachitariu & Stringer, 2022). The pulp figure comes from this project's own runs: 65 labelled bubbles gave recall 0.52, while about 330 per cross-validation fold gave 0.91. Both are floors; the held-back comparison decides.

## 9 Discussion

**Sizing matches a human; selection is the open question.** Mask IoU (0.89) and the D32 ratio (0.96 fixed split, 0.99 cross-validated) sit at the level of two hand-labelling passes by the same person (0.875 and 0.96). Where model and labels disagree, it is about which bubbles count, exactly as the repeatability study predicted.

**The model exposed a labelling bias.** On full 1024 px frames, small or faint bubbles with a clear edge were easy to skip. The model, trained only on in-focus round bubbles, found them anyway, which suggests it learned what a bubble looks like rather than copying the annotator. The standard fix is a blind completeness pass over the labels followed by retraining.

**Part-hidden bubbles are included on purpose.** The labels exclude them because they cannot be traced by hand, but leaving them out biases the distribution small, since large bubbles are overlapped most. An automatic occlusion flag was tried and removed: it caught only 1 of 20 audited cases.

**The operating point is chosen for D32, not F1.** Because D32 weights by volume, a missed 130 px bubble costs about 17 times what a spurious 50 px bubble adds. That is why the confidence threshold is 0.40 rather than the higher value that would maximise precision.

**Against the literature.** Bubble Analyser reports under 5% D32 error with classical processing on clean images (Mesa et al., 2022). This pipeline reaches 1–4% on crowded, murky pulp footage where watershed gave a D32 ratio of 0.43.

**Knowing when to stop.** The stopping rule turns "how many frames?" into a stated precision. Froth needs about 6–7 frames for ±5%; pulp settled after 137 bubbles on one rotor clip because its size spread is narrower. The interval covers frame-to-frame variation within one clip, not between repeat runs of a condition.

## 10 Limitations and imaging recommendations

### 10.1 Limitations

- Only 8 rotor clips exist, so generalisation is shown across these conditions only.
- One annotator labelled everything, and the human ceiling comes from 4 crops.
- The pulp scale is provisional: micron values derive from a quoted field width, not a target imaged through the probe. Pixels are the primary unit; microns are indicative.
- Without telecentric optics and with a large depth of field, magnification depends on bubble depth, and the measurement volume is unknown, so number concentration cannot be reported.
- The bright regime (diluted and Venturi clips) is not in the model's training data, and the Venturi D32 was not yet stable from one clip.
- Test labels are incomplete (Section 6.3), so precision against labels understates true precision, and recall may be slightly optimistic.
- Froth: 77 hand-labelled bubbles have been scored so far, and the boundary convention (bright cap or film centre) shifts D32 by about 10% on bubbles this small.
- CPU-only training (about 3 hours per run) limited the number of ablations.

### 10.2 Imaging-rig recommendations

1. Set exposure so the background is about 200–220 grey with no clipped pixels; the Venturi clips had 6–54% clipped.
2. Use telecentric optics so magnification does not depend on bubble depth.
3. Use a shallower depth of field, or a known measurement volume, which would also allow number concentration.
4. Record one bubble-free flat-field frame per session.
5. Image a calibration target through the probe at working distance so micron values become publishable.
6. Light the whole froth column, not about 15% of it.
7. Record losslessly: a lossy re-encode of a pulp clip made the model miss every bubble in testing.

## 11 Conclusion and future work

All four aims were met.

1. **Automated sizing.** Bubble size and D32 come from probe video with one button, with no parameters to tune.
2. **Human-level D32 on unseen conditions.** Across all 8 rotor conditions held out in turn, recall was 0.83–1.00 per clip and the D32 ratio 0.99 ± 0.03, matching two hand passes by the same annotator (0.96).
3. **Froth.** Zero-shot Cellpose recovered 86% of hand-outlined bubbles with 98% local precision and gave a stable D32 of 2.16 mm in under a minute per frame.
4. **A usable tool.** BubbleMetrix reports D32 with its confidence interval, stops when ±5% is reached, and lets researchers fine-tune on new footage with a guarded before/after check.

The project also showed that the inherited tool's reported accuracy came from circular validation, and that most of the new model's "errors" were bubbles missed by the human labeller.

**Future work**

- A blind completeness pass over the test labels, then retraining.
- Fine-tune on the bright regime (Venturi and diluted clips) using the software's fine-tune tab.
- Measure the pulp scale with a target imaged through the probe.
- Score the labelled t = 150 s froth frame and settle the froth boundary convention.
- Repeat runs of each condition to measure run-to-run spread of D32.

## References {.refs-h}

<div class="refs" markdown="1">

Aldrich, C., Marais, C., Shean, B.J. & Cilliers, J.J. (2010). Online monitoring and control of froth flotation systems with machine vision: a review. *International Journal of Mineral Processing*, 96, 1–13.

Cerqueira, R.F.L. & Paladino, E.E. (2021). Development of a deep learning-based image processing technique for bubble pattern recognition and shape reconstruction in dense bubbly flows. *Chemical Engineering Science*, 230, 116163.

Cui, Y., Li, C., Zhang, W., Ning, X., Shi, X., Gao, J. & Lan, X. (2022). A deep learning-based image processing method for bubble detection, segmentation, and shape reconstruction in high gas holdup sub-millimeter bubbly flows. *Chemical Engineering Journal*, 449, 137859.

Finch, J.A., Xiao, J., Hardie, C. & Gomez, C.O. (2000). Gas dispersion properties: bubble surface area flux and gas holdup. *Minerals Engineering*, 13(4), 365–372.

Gomez, C.O. & Finch, J.A. (2007). Gas dispersion measurements in flotation cells. *International Journal of Mineral Processing*, 84(1–4), 51–58.

Gorain, B.K., Franzidis, J.-P. & Manlapig, E.V. (1997). Studies on impeller type, impeller speed and air flow rate in an industrial scale flotation cell. Part 4: Effect of bubble surface area flux on flotation performance. *Minerals Engineering*, 10(4), 367–379.

Gorain, B.K., Franzidis, J.-P. & Manlapig, E.V. (1999). The empirical prediction of bubble surface area flux in mechanical flotation cells from cell design and operating data. *Minerals Engineering*, 12(3), 309–322.

Grau, R.A. & Heiskanen, K. (2005). Bubble size distribution in laboratory scale flotation cells. *Minerals Engineering*, 18(12), 1164–1172.

He, K., Gkioxari, G., Dollár, P. & Girshick, R. (2017). Mask R-CNN. *Proceedings of the IEEE International Conference on Computer Vision*, 2961–2969.

He, Y., Hu, C., Li, H., Jiang, B., Hu, X., Wang, K. & Tang, D. (2022). A flexible image processing technique for measuring bubble parameters based on a neural network. *Chemical Engineering Journal*, 429, 132138.

Hessenkemper, H., Starke, S., Atassi, Y., Ziegenhein, T. & Lucas, D. (2022). Bubble identification from images with machine learning methods. *International Journal of Multiphase Flow*, 155, 104169.

Kim, Y. & Park, H. (2021). Deep learning-based automated and universal bubble detection and mask extraction in complex two-phase flows. *Scientific Reports*, 11, 8940.

Kirillov, A. et al. (2023). Segment Anything. *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 4015–4026.

Li, Y., Xie, S., Chen, X., Dollár, P., He, K. & Girshick, R. (2021). Benchmarking detection transfer learning with vision transformers. arXiv:2111.11429.

Lin, T.-Y. et al. (2014). Microsoft COCO: Common objects in context. *European Conference on Computer Vision*, 740–755.

Loshchilov, I. & Hutter, F. (2019). Decoupled weight decay regularization. *International Conference on Learning Representations*.

Mesa, D., Quintanilla, P. & Reyes, F. (2022). Bubble Analyser — an open-source software for bubble size measurement using image analysis. *Minerals Engineering*, 180, 107497.

Pachitariu, M. & Stringer, C. (2022). Cellpose 2.0: how to train your own model. *Nature Methods*, 19, 1634–1641.

Pachitariu, M., Rariden, M. & Stringer, C. (2025). Cellpose-SAM: superhuman generalization for cellular segmentation. *bioRxiv*, doi:10.1101/2025.04.28.651001.

Roberts, D.R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40, 913–929.

Stringer, C., Wang, T., Michaelos, M. & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular segmentation. *Nature Methods*, 18, 100–106.

Vincent, L. & Soille, P. (1991). Watersheds in digital spaces: an efficient algorithm based on immersion simulations. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 13(6), 583–598.

Wang, Q., Li, X., Xu, C., Yan, T. & Li, Y. (2021). Bubble recognizing and tracking in a plate heat exchanger by using image processing and convolutional neural network. *International Journal of Multiphase Flow*, 138, 103593.

</div>
