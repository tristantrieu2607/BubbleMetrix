# Taste of Research — report & poster kit

**Tristan Trieu (legal name Quoc Anh Trieu) · zID z5793457**
**UNSW Faculty of Engineering Taste of Research, 2026 Term 2–Term 3**
**Project:** Bubble Size Measurement in a Novel Flotation Technology
**Supervisor:** Prof Seher Ata · **School:** School of Minerals and Energy Resources Engineering

Compiled 6 Oct 2026; updated 10 Oct (cross-validation, Cellpose froth) and **20 Nov 2026** (deadlines and IDs from email, 55 mm = inner tube width, D32 convergence, BubbleMetrix software). Every number comes from a file on disk or a run log. Where something still needs checking, it says so.

Figures: `report/figures/`, overlays: `report/overlays/` (300 dpi PNG). Faculty files: `report/faculty/` (poster template `.potx`, recipient guidelines PDF). Results-only slide deck: `report/TOR_results_deck.pptx`.

---

## 0. Contents

1. Deadlines, IDs and deliverable rules
2. Abstract draft
3. Background and motivation
4. Data
5. Pulp method
6. Pulp results
7. Discussion
8. Limitations
9. Froth segmentation
10. Imaging-rig recommendations
11. What went wrong before, and how it was found
12. Poster plan
13. Final-report outline
14. Key-numbers cheat sheet
15. References
16. File locations
17. Software: BubbleMetrix

---

## 1. Deadlines, IDs and deliverable rules

### 1.1 From email (checked 20 Nov 2026)

| | |
|---|---|
| zID | **z5793457** (activated 26 Jun 2026) |
| Name for the poster filename | Faculty wants your **full name**; the scholarship is under **Quoc Anh Trieu** (you go by Tristan Trieu). Suggested: `Quoc_Anh_Tristan_Trieu_z5793457_TOR_poster.pdf` |
| Project start | 6 Aug 2026 |
| **Project end** | **10 Dec 2026** (end of Term 3, confirmed by Sonia Kao, 12 Oct) |
| **Poster due** | **at the conclusion of the project → by 10 Dec 2026**, submitted to the Faculty Office as PDF |
| Poster Showcase | the annual Taste of Research Poster Showcase **in 2027** (date not announced yet) |
| Written report | format, length and due date **set by the supervisor**; nothing in email yet → **ask Prof Ata** |
| Contact | Sonia Kao, scholarships@eng.unsw.edu.au |

### 1.2 Offer letter (18 May 2026)

60-day FTE project, $6,000 stipend. To stay eligible: take part in research activities, stay enrolled full-time, submit progress reports as agreed with the supervisor, **submit a project poster** at the end, **present at the Poster Showcase**.

### 1.3 Poster rules (2026 TOR Recipient Guidelines)

**Content:** Aims · Background & motivation (for a non-specialist) · Method & results (concise, convincing) · Conclusions (back to the aims; why it matters).

**Format:**
- Use the Faculty PowerPoint template (`report/faculty/Project poster template_portrait.potx`). **Do not change its fonts, background or format**; edit text and add figures only.
- A1 portrait, Page Setup "Custom" — **do not change slide dimensions**.
- Include **supervisor's name and school**.
- Save as **PDF with all fonts and images embedded**, filename with **full name and zID**.
- Print an **A4 test copy on a laser printer** first; check layout, typos, missing elements.
- The Faculty prints it at A1 in colour.

**Template:** yellow UNSW header with crest; a title box ("Title goes here", Clancy 115 pt) and a by-line box ("By line goes here", Roboto Mono 60 pt); everything below the header is blank. ⚠️ Clancy and Roboto Mono must be installed or PowerPoint substitutes them silently and the PDF won't embed them — check Acrobat → Properties → Fonts before submitting.

---

## 2. Abstract draft

> Bubble size controls how efficiently flotation recovers valuable minerals, but measuring it from probe video is still done largely by hand. This project built and evaluated a deep-learning pipeline (Mask R-CNN) that finds and outlines in-focus bubbles in pulp-zone video from a novel flotation cell and reports the Sauter mean diameter (D32). A labelling protocol separated measurable from unmeasurable bubbles (focus gate, 12 px size floor, ignore regions for part-hidden bubbles), and 383 bubbles were hand-labelled on 35 full video frames plus earlier crops. The model was always scored on video clips it had never seen. Compared with an earlier version, recall on large bubbles (100–140 px) rose from 0.62 to 0.90 and the D32 error fell from 18% to 4%, matching the agreement between two hand-labelling passes by the same person (D32 ratio 0.96). In leave-one-clip-out cross-validation over all 8 operating conditions, recall averaged 0.92 ± 0.06 and the D32 ratio 0.99 ± 0.03 (377 hand-labelled bubbles). A manual audit showed most "false positives" were real bubbles missed during labelling; only 16% were genuine model errors. In the froth zone, a general-purpose cell segmenter (Cellpose) used without any froth training recovered 86% of 51 hand-outlined bubbles with 98% precision in the labelled area, ran in under a minute per frame, and gave a D32 of 2.16 mm (2.06–2.30 mm per frame over 7 minutes of video). Both are packaged in a desktop tool, BubbleMetrix, which stops measuring once D32 is known to ±5% and lets researchers fine-tune the models on new footage with a built-in before/after check.

---

## 3. Background and motivation

### 3.1 For a general audience (poster)

- Flotation separates valuable minerals from waste rock by attaching them to air bubbles that float to the top.
- **Smaller bubbles have more surface area per volume**, so bubble size directly affects how much mineral is captured.
- Engineers summarise bubble size with the **Sauter mean diameter, D32 = Σd³ / Σd²** — the diameter of a bubble with the same volume-to-surface ratio as the whole population.
- Measuring bubbles by hand from video is slow and subjective. Automating it lets researchers compare spargers, rotor speeds, air rates and frother doses quickly.
- The difficulty: the pulp is crowded and murky; most bubbles are out of focus or overlapping, and only sharp ones can be measured accurately.

### 3.2 For the technical report

- Bubble size distribution (BSD) governs gas–liquid interfacial area and flotation kinetics: the flotation rate constant scales with bubble surface area flux S_b = 6·J_g / D32 (Gorain et al., 1997; Finch et al., 2000). Image analysis is the most common BSD measurement (Gomez & Finch, 2007; Mesa et al., 2022).
- Classical segmentation (thresholding, watershed) fails in dense, overlapping bubbly flows; CNN-based detection and instance segmentation are the current direction (Cerqueira & Paladino, 2021; He et al., 2022; Cui et al., 2022; Wang et al., 2021).
- Mask R-CNN has been used for sub-millimetre bubbles at gas holdup up to 20%, with a shape-reconstruction step for overlapping bubbles (Cui et al., 2022).
- Bubble Analyser reports < 5% D32 error against manual segmentation for a classical pipeline on clean images (Mesa et al., 2022) — a useful benchmark for "acceptable" D32 error.
- This project inherited **BubMask-Fiji** (a Fiji/ImageJ port of BubMask, the Mask R-CNN bubble model of Kim & Park, 2021, built by a previous student, Arman), which had never been honestly validated on this footage (§11).

---

## 4. Data

### 4.1 Pulp footage

| Property | Value |
|---|---|
| Clips | 13 AVI: **8 rotor**, 1 rotor diluted, 4 Venturi |
| Frames | 1024 × 1024, 8-bit greyscale, uncompressed, 101 frames per clip (30 fps) |
| Field of view | circle, centre (524, 508), radius 601 px (measured) |
| Calibration | 303.858 px/mm = **3.291 µm/px**, from the provider's quoted 1024 px = 3.37 mm field. **Provisional** — never measured through the probe. Report pixels as the primary unit. (Adjustable in BubbleMetrix.) |
| Frame independence | rotor clips decorrelate by frame 5; Venturi/dilute still correlated at frame 10 → sample ≥ 5 (rotor) / ≥ 20 (bright) frames apart |
| Two imaging regimes | rotor: dark and crowded (median grey 29–48); Venturi/dilute: bright, **over-exposed** (6–54% of pixels clipped at 255) |
| Typical rotor bubble | hand-measured median **73.5 px ≈ 244 µm** (range 49.5–111 px) |
| Bubble speed | Venturi 500 air: the largest bubbles move ~90 px per frame (measured, frames 94–100) |
| No new footage | confirmed late Sept 2026 — these clips are the whole dataset |

**The 8 rotor clips** (all 90 ppm frother):

| Short name | Clip |
|---|---|
| Coarse 200air | RotCoarse_1260rpm_90ppm_200air_900water |
| Coarse 300air | RotCoarse_1260rpm_90ppm_300air_800water |
| Fine 200air | RotFine_1260rpm_90ppm_200air_900water |
| Fine inline | RotFine_1260rpm_90ppm_200air_900water_inline |
| Fine 300air | RotFine_1260rpm_90ppm_300air_800water |
| Fine 1440rpm | RotFine_1440rpm_90ppm_200air_900water |
| Med 200air | RotMed_1260rpm_90ppm_200air_900water |
| Med 300air | RotMed_1260rpm_90ppm_300air_800water |

Variables: sparger/rotor type (Coarse/Fine/Med), rotor speed (1260/1440 rpm), air setting (200/300), water setting (900/800), inline configuration. *Units for "air" and "water" are not in any email — check with Prof Ata before printing them.*

Others: `RotFine_..._inline_6dilute` (diluted inline clip); Venturi 200/500 air × 60/90 ppm.

### 4.2 Froth footage

- `samples/froth/180WW.MP4`: 720 × 1280 portrait, 30 fps, ~8 min (14,095 frames), H.264 from an Apple device (lossy compression affects boundary accuracy).
- Strong static illumination gradient; only ~15% of the column is lit.
- **Calibration: 55 mm is the inner width of the tube (confirmed 20 Nov).** The inner glass surfaces are found on every frame: **471 px** on average (468–472 over 24 frames spread across the clip; outer edges 501 px) → **0.1168 mm/px**. This replaces the earlier outer-width scale (0.1096 mm/px), so every froth diameter in mm is **6.6% larger** than in earlier drafts.
- Only a band of each frame is in focus; measurements use that band (found automatically).
- Supplied by Imtiaz Shah (team member).

### 4.3 Labelled dataset (rotor pulp)

| Split | Clips | Images | Bubbles |
|---|---|---|---|
| Train | Coarse 200air, Fine 200air, Fine 1440rpm, Med 200air | 16 crops (512 px) + 22 full frames | 65 + 263 = **328** |
| Val | Fine 300air | 4 crops + 4 full frames | 14 + 44 = **58** |
| Test | Coarse 300air, Fine inline, Med 300air | 9 full frames (+10 old crops, now retired) | **76** (124 incl. retired crops) |

- 35 full frames labelled 27 Sept – 1 Oct; 383 bubbles, 124 ignore boxes, none under the 12 px floor.
- Test frames labelled **blind** (no model output viewed first), ≥ 5 frames from anything previously viewed.
- Old 512 px test crops retired on 4 Oct (borders cut bubbles into half-circles). Training still uses the train/val crops.
- **Rotor diluted:** 8 full frames, 29 bubbles, median 125 px — training only (one clip).
- **Venturi:** 24 crops, 101 stamps; not in any training run.

---

## 5. Pulp method

### 5.1 Labelling protocol (the "ruler")

One rule: **stamp what you can trace, ignore what you'd have to guess, leave defocus alone.**

| Rule | Decision | Why |
|---|---|---|
| **Focus gate** | in-focus (sharp edge, dark rim, bright specular centre) → label; defocused → background | defocused bubbles image **larger**; including them inflated D32 2.3× in the previous work |
| **Size floor** | 12 px = 40 µm minimum | set by window debris (largest speck 10.5 px) |
| **At-limit flag** | 12–20 px flagged | ±1 px on radius = ±17% diameter = ±59% volume |
| **Occlusion** | sharp but part-hidden → **ignore region**, never traced or completed | neither measurable nor background |
| **Empty images** | must be explicitly confirmed empty | "skipped" ≠ "no bubbles" |
| **Focus judged on raw** | CLAHE view only for finding candidates | contrast enhancement makes defocus look sharp |

**Human repeatability** (same annotator, same 4 crops, two passes) — the ceiling any model is compared against:

| | value |
|---|---|
| count agreement | 0.82 |
| mean mask IoU on matched bubbles | 0.875 |
| per-bubble diameter disagreement | median 3.2% |
| D32 ratio between passes | **0.96** |

Selection (which bubbles count) is the unstable part; sizing is stable once a bubble is picked.

### 5.2 Model

| Component | Choice | Reason |
|---|---|---|
| Architecture | torchvision **Mask R-CNN ResNet-50-FPN-v2**, COCO-pretrained (He et al., 2017; Lin et al., 2017; Li et al., 2021) | instance masks give per-bubble outlines → diameter |
| Input scale | **no internal resizing** | resizing destroys the calibration and shrinks small bubbles below the anchors |
| Anchors | "round3": aspect ratio 1.0, 3 sizes per FPN level at √2 spacing, **8–256 px** | bubbles are round; worst-case anchor IoU ~0.71 vs ~0.50 |
| Ignore-aware loss | anchors/proposals mostly inside ignore regions removed from the background loss (COCO "crowd" logic) | otherwise part-hidden bubbles train as "background" (cause of the old 5% recall) |
| Augmentation | flips, gamma, **scale jitter 0.75–2.0×** (p = 0.5) | gives the model large-bubble examples |
| Tiling | 512 px tiles; random windows in training | full frames on CPU |
| Training | AdamW, lr 1e-3, cosine, batch 2, backbone layers 1–2 frozen, CPU | |
| Epochs | 20; best validation loss at **epoch 16** (plateau from ~11) | `fig6_training_curve.png` |
| Checkpoint | selected on **validation** loss (held-out clip) | |

Main model: `BubMask/runs/rr_full_e20/best.pt`, ~3 h for 20 epochs on CPU.

### 5.3 Evaluation protocol

- **Split by clip, never by frame.**
- **Mask IoU ≥ 0.5** to count a match (not box IoU).
- **Equivalent-area diameter d = 2√(A/π)** on both sides (min-enclosing circle reads 8.9% larger).
- **D32 ratio** = model D32 / hand D32 — the headline quantity.
- **Match before filtering**; **12 px floor** on predictions too; **edge rule** (bubbles cut by the image border not scored, either side).
- **Scored on full frames only**; confidence **0.40** (chosen for D32 fidelity).
- Recall by size band; "missed entirely" vs "touched but unmatched".

### 5.4 Cross-validation protocol

- **Leave-one-clip-out, 8 folds**: train on 7 rotor clips, test on the 8th.
- Recipe frozen beforehand: 12 epochs, conf 0.40, round3 anchors, scale jitter, edge rule, full frames scored; final-epoch model; same COCO start each fold; 2.5–3.5 h per fold on CPU.
- Reported as mean ± SD across clips and pooled over bubbles.

---

## 6. Pulp results

### 6.1 Timeline of honest results

| Date | Model / method | Test set | Recall | Precision | Mask IoU | D32 ratio |
|---|---|---|---|---|---|---|
| Aug | original BubMask (inherited) | independent hand count | **0.05** | — | — | **2.3× too high** |
| 27 Aug | watershed baseline | 6 crops / 33 bubbles | 0.03 | 0.00 | 0.09 | 0.43 |
| 27 Aug | SAM ViT-B, automatic masks | 6 crops / 33 bubbles | 0.52 | 0.08 | 0.72 | 0.77 |
| 7 Sep | BubMask2 v1 (conf 0.72) | 10 crops / 48 bubbles | 0.52 | 0.64 | 0.85 | 0.80 |
| 7 Sep | BubMask2 v1 (conf 0.40) | 10 crops / 48 bubbles | 0.69 | 0.50 | 0.85 | 0.82 |
| 3 Oct | **rr_full_e20** | 19 images / 124 bubbles | **0.85** | 0.46 | **0.89** | **0.96** |

SAM fragments large bubbles (recall 0.17 for 300+ µm), which is also why its D32 was 23% low.

### 6.2 Previous vs new model — same 124 held-out bubbles

| | Previous (`rotor_raw`, 7 Sep) | **New (`rr_full_e20`)** | Human repeatability |
|---|---|---|---|
| Recall | 0.734 | **0.847** | 0.82 |
| Precision (vs labels) | 0.417 | **0.461** | — |
| Mask IoU | 0.835 | **0.891** | 0.875 |
| D32 ratio | 0.819 | **0.957** | 0.96 |
| Missed entirely / touched-unmatched | 19 / 14 | **12 / 7** | — |

| Band | n | Previous | New |
|---|---|---|---|
| 12–40 px | 4 | 0.25 | 0.75 |
| 40–70 px | 29 | 0.86 | 0.86 |
| 70–100 px | 68 | 0.76 | 0.82 |
| **100–140 px** | 21 | 0.62 | **0.90** |
| **140+ px** | 2 | 0.00 | **1.00** |

→ `fig1_recall_by_size_old_vs_new.png`, `fig2_headline_old_vs_new.png`

Better on every metric including precision, so the recall gain is not from predicting more. D32 moved from 18% low to 4% low — level with human repeatability. *Caveat:* three things changed together (more full-frame labels, scale jitter, new anchors); report the combined effect. Mask IoU 0.89 vs human 0.875 is **at** the ceiling, not above it.

### 6.3 Full frames only (9 test frames, 76 bubbles, before edge rule)

Recall 0.829 · precision 0.396 · mask IoU 0.889 · **D32 ratio 0.962**. Re-scored through the GUI's gates (FOV + edge on both sides, 20 Nov): recall 0.82, D32 ratio 0.96 on 72 bubbles.

### 6.4 False-positive audit — the precision story

The 96 unmatched detections on the 9 test frames were each inspected by hand, labels left untouched.

| What it really was | n | share |
|---|---|---|
| In-focus bubble **missed during labelling** | 44 | 46% |
| Part-hidden (occluded) bubble | 20 | 21% |
| Cut by the image edge | 17 | 18% |
| **Defocused — a genuine model error** | 15 | **16%** |
| Debris / duplicate | 0 | — |

| Precision | value |
|---|---|
| Against original labels | **0.40** |
| Counting missed in-focus bubbles as correct | **0.67** |
| Also counting occluded bubbles (included by design) | **0.80** |

Recall against labels + audited bubbles: 0.89. Only 15 of 159 detections (9%) were real mistakes. → `fig3_false_positive_audit.png`, `fig4_precision_before_after_audit.png`

Raising the threshold would not help: at conf 0.8 it removes 11 of the 15 defocused errors but also 16 of the 44 real bubbles. *Caveat:* recall is measured on labelled bubbles, which skew easier, so it may be slightly optimistic.

### 6.5 Leave-one-clip-out cross-validation (all 8 folds)

| Fold | Held-out clip | Frames | Bubbles | Recall | Precision | Mask IoU | D32 ratio |
|---|---|---|---|---|---|---|---|
| 0 | Coarse 200air | 8 | 66 | 0.894 | 0.504 | 0.896 | 1.014 |
| 1 | Coarse 300air | 3 | 10 | 1.000 | 0.435 | 0.879 | 0.966 |
| 2 | Fine 200air | 4 | 48 | 0.938 | 0.388 | 0.890 | 0.979 |
| 3 | Fine inline | 3 | 34 | 1.000 | 0.459 | 0.907 | 0.914 |
| 4 | Fine 300air | 4 | 42 | 0.905 | 0.384 | 0.888 | 1.010 |
| 5 | Fine 1440rpm | 4 | 50 | 0.920 | 0.414 | 0.896 | 0.983 |
| 6 | Med 200air | 6 | 98 | 0.888 | 0.503 | 0.877 | 1.023 |
| 7 | Med 300air | 3 | 29 | 0.828 | 0.407 | 0.873 | 0.992 |
| **Mean ± SD** | | | | **0.921 ± 0.058** | **0.437 ± 0.048** | **0.888 ± 0.011** | **0.985 ± 0.035** |
| **Pooled** | | | **377** | **0.910** | **0.444** | | **0.992** |

F1 per clip 0.591 ± 0.045; pooled 0.60. Pooled recall by band: 12–40 px 0.74 (n=31) · 40–70 0.89 (112) · 70–100 0.93 (199) · **100–140 0.97 (33)** · 140+ 1.00 (2). → `fig5_cross_validation_per_clip.png`

Across every unseen condition D32 is within 9% of hand labels (worst: Fine inline 0.91) and within 3% for six of eight clips; recall never below 0.83; IoU SD 0.01. CV numbers are higher than the fixed split because each model trains on 7 clips (vs 4) and the edge rule removes truncated bubbles — both legitimate; say so.

### 6.6 Training curve

Validation loss 0.50 → ~0.42 by epoch 7, plateau 0.39–0.42 from epoch 11, minimum 0.391 at epoch 16; training loss keeps falling (0.73 → 0.26). → `fig6_training_curve.png`

### 6.7 Example output

`overlays/pulp_gt_vs_model.png` — hand labels (left) vs model (right), Fine inline frame 23 (CV fold 3). Use this on the poster. `fig7_example_overlay_inline_f023.png` — both on one image.

### 6.8 How many bubbles are enough? (D32 convergence, new)

D32's 95% confidence interval comes from the delta method for a ratio estimator, taking the larger of a bubble-level and a frame-level (clustered, Student-t) error. **Stable** = half-width ≤ 5% of D32 from ≥ 3 frames, at two frames in a row. BubbleMetrix applies this live.

| Clip | Frames used (every 5th) | Bubbles | D32 | 95% CI | Stable? |
|---|---|---|---|---|---|
| RotMed 300air (rotor) | 8 of 21 | 137 | 254.8 µm | ±4.7% | yes |
| Venturi 500 air 60 ppm | 21 of 21 | 355 | 446 µm | ±9.7% | no (model never trained on Venturi) |

Pulp settles on far fewer bubbles than froth because its size spread is narrower. The interval covers frame-to-frame variation **within one clip**, not between repeat runs.

---

## 7. Discussion points

1. **Sizing is solved to human level; selection is the open question.** IoU and D32 match human repeatability; disagreement is about *which* bubbles count.
2. **Annotation bias is a finding, not a failure.** The model exposed systematic under-labelling on full frames; fix = blind completeness pass and retrain (future work).
3. **Occlusion was a deliberate decision.** Labels exclude part-hidden bubbles, but the model outlines them; excluding them biases the distribution small, so they are **included by default**. An automatic occlusion flag caught only 1 of 20 audited cases and was removed.
4. **Edge rule** keeps D32 honest (truncated bubbles dropped both sides).
5. **Large bubbles drive D32.** A missed 130 px bubble costs ~17× what a spurious 50 px one adds, so the operating point is chosen for D32 fidelity, not F1.
6. **Literature:** Bubble Analyser < 5% D32 error on clean images (Mesa et al., 2022); this pipeline reaches ~1–4% on crowded, murky pulp where watershed scored D32 0.43.
7. **Knowing when to stop.** A stopping rule turns "how many frames?" into a stated precision (±5%) and cuts froth measurement from 12+ frames to 6–7.

---

## 8. Limitations

- **8 rotor clips** — generalisation shown across these conditions only; no further footage.
- **One annotator**; human ceiling from 4 crops.
- **Pulp calibration provisional** — derived from a quoted field width; report pixels, µm as indicative.
- **No telecentric optics, large depth of field** — magnification depends on depth; measurement volume unknown, so no number concentration.
- Bright regime (dilute, Venturi) not in the model; Venturi D32 not yet stable from one clip.
- Test labels incomplete → precision vs labels understates; recall may be slightly optimistic.
- CPU-only training (~3 h per run) limited ablations.
- **Froth:** 77 hand-labelled bubbles scored so far (51 on one patch + 26 on one crop); 160 more outlined on the t = 150 s frame but not yet scored; boundary convention (§9.5).

---

## 9. Froth segmentation

Froth is a different problem from pulp: the task is **separating** neighbouring bubbles rather than **detecting** isolated ones. Footage from Imtiaz. *Confirm with Prof Ata how much froth goes in the poster and report.*

### 9.1 Method

- **Cellpose zero-shot** (Cellpose-SAM `cpsam_v2`, cellpose 4.2.1.1; Stringer et al., 2021; Pachitariu et al., 2025), **default settings**, no froth training.
- **Region:** in-focus band of each frame, found automatically (e.g. 489 × 424 px at t = 0 s).
- **Scale:** 55 mm inner tube width = inner glass surfaces found per frame (~471 px) → **0.1168 mm/px** (§4.2).
- **Edge rule**, equivalent-area diameter, D32 = Σd³/Σd², matching at mask IoU ≥ 0.5 — same as pulp.
- **Ground truth:** hand-drawn outlines (closed walls / freehand on the mid-rim); unresolved areas marked unsure and not scored.
- In BubbleMetrix: Froth = Cellpose, up to 40 frames spread over the clip, stopping when D32 is stable.

Earlier approaches (BubSAM3 = tiled SAM, bubseg = watershed) are superseded (§9.4).

### 9.2 Results against ground truth

**Fully labelled patch** — t = 240 s, 128 px, a frame no method had seen before labelling; 51 bubbles (6.6–23 px, median 10.7 px).

| Metric | Value (95% CI) |
|---|---|
| Recall (IoU ≥ 0.5) | **0.86** (0.74–0.93), 44 / 51 |
| Precision, strict | 0.68 (0.56–0.78) |
| Precision, local (unresolved area ignored) | **0.98** (0.88–1.00) |
| F1, strict / local | 0.76 / **0.92** |
| Mean mask IoU (matched) | 0.67 |
| D32, hand labels (all 51) | 14.0 px ≈ **1.64 mm** |
| D32, matched 44: hand vs Cellpose | 14.4 vs 15.9 px → **ratio 1.10** |
| D32 ratio, whole patch (strict / local) | 1.08 / 1.14 |

- **Every labelled bubble is found** (each ≥ 80% covered by its own mask); the 7 "misses" are 6.6–10 px bubbles where Cellpose's mask is larger, so IoU < 0.5.
- **Systematic boundary offset, +1.2 px radius:** labels traced on the bright cap, Cellpose edges mid-film. Constant, so it matters most for small bubbles (diameter ratio 1.31 under 9 px, 1.05 over 17 px). With labels grown 1 px: 51/51 matched, IoU 0.77, diameter ratio 1.08. A labelling-convention question, not a detection error.

**Partly labelled crop** — frame 299, 26 hand circles ≥ 17 px: recall **26/26** (CI 0.87–1.00), mean mask IoU 0.80, mean diameter ratio 0.99, **D32 ratio 0.97** (matched).

**Whole frame t = 150 s (new, not yet scored):** 160 freehand outlines on the mid-rim (14 Oct); no tile marked complete, so it gives recall and sizing, not precision. Needs one Cellpose run on that frame (`Froth\eval\RUN_CELLPOSE_ON_PATCHES.bat`), then `froth_eval.py` case `frame_t150s`.

### 9.3 Results over the video (no labels — consistency only)

Cellpose on 9 frames spread over 7 minutes, edge rule applied, 55 mm inner width:

| Frame | Bubbles | D32 (mm) |
|---|---|---|
| 0 s | 869 | 2.12 |
| 30 s | 645 | 2.30 |
| 60 s | 614 | 2.25 |
| 120 s | 667 | 2.09 |
| 180 s | 677 | 2.06 |
| 240 s | 671 | 2.08 |
| 300 s | 702 | 2.08 |
| 360 s | 606 | 2.25 |
| 420 s | 624 | 2.23 |
| **All 9** | **6,075** | **2.16** (per frame 2.06–2.30) |

- 30–48 s per frame on CPU (35–80× faster than the earlier SAM pipeline at 27–40 min).
- Figures `froth_f3_d32_stability_and_sizes.png` and `froth_f4_runtime.png` were drawn with the old outer-width scale — **multiply their mm axis by 1.066** or regenerate before use.

### 9.4 Earlier approaches (one line each)

- **bubseg** (DoG seeds → marker watershed; Vincent & Soille, 1991): 13/26 labelled bubbles.
- **BubSAM3** (tiled SAM ViT-B; Kirillov et al., 2023): 23/26, IoU 0.71, D32 ratio 0.91; 27–40 min per frame.
- Cellpose replaced both: 26/26, IoU 0.80, D32 ratio 0.97, under a minute per frame.

### 9.5 How many bubbles for a stable froth D32? (new)

Pooling the 9 Cellpose frames (6,075 bubbles):

- Bubble-level bootstrap: 95% CI ±27% at 25 bubbles, ±11% at ~200, ±7% at ~480, **±5% at ~1,100**, ±2.1% at 6,075.
- Frame-to-frame spread of D32 4.4% vs 3.2% expected from sampling alone (p = 0.08): borderline, so the GUI also uses a frame-level interval.
- With that rule, measurement **stops after 6–7 frames (≈4,100–4,800 bubbles)**, D32 within 2% of the 9-frame value. → `Froth/eval/results/d32_convergence/convergence.png` (regenerate for the inner-width scale; relative widths unchanged).
- This measures precision of the measurement, not accuracy.

### 9.6 Limitations and next steps

- **Labels:** 77 scored; t = 150 s (160 outlines) to score; boundary convention (bright cap vs film centre) shifts D32 ~10% on bubbles this small — fix before more labelling.
- **Lighting:** only ~15% of the column is lit; the in-focus band limits each frame.
- If Cellpose falls short on new labels: fine-tune it in BubbleMetrix (§17), or StarDist / μSAM through the same harness.

### 9.7 For the poster

One panel, "Froth zone — separating packed bubbles": `overlays/cellpose_yellow_side_by_side.png`, one method line ("Cellpose, zero-shot, no froth training"), three numbers: **recall 0.86 / local precision 0.98 on 51 hand-outlined bubbles**, **D32 = 2.16 mm (2.06–2.30 per frame over 7 min)**, **< 1 min per frame**.

---

## 10. Imaging-rig recommendations

1. **Exposure:** background ~200–220 grey, zero clipped pixels (rotor 0%, Venturi 6–54% clipped).
2. **Telecentric optics** so magnification doesn't depend on bubble depth.
3. **Shallower depth of field** / known measurement volume (also enables number concentration).
4. **Flat-field reference frame** per session.
5. **Calibration target** imaged through the probe so µm values become publishable.
6. **Froth lighting:** light the whole column, not ~15%.
7. **Lossless recording** (uncompressed/FFV1 rather than H.264 or MJPG): a lossy re-encode of a pulp clip made the model miss every bubble in testing.

---

## 11. What went wrong before, and how it was found

| Problem found | Effect | Fix |
|---|---|---|
| **Circular validation** — the inherited model's pre-labels used as ground truth for the same model | reported F1 0.996; independent hand count: **5% recall, D32 2.3× too high** | never score a model against labels a model produced |
| **Calibration wrong by 1.66×** (183 px/mm assumed vs 303.858) | every diameter 66% too large | report pixels until calibration is measured |
| **Resize + COCO anchors** (smallest ≈ 67 px in original pixels) | most bubbles smaller than any anchor | no resizing; bubble-sized anchors |
| **Ignore regions had no effect in the loss** | part-hidden bubbles trained as background | ignore-aware loss |
| **Box IoU reported as IoU** | 93% claimed vs mask-IoU ceiling 0.875 | mask IoU |
| **Two different diameters** | 8.9% bias | equivalent-area diameter both sides |
| **Duplicated images, frame-level splits** | test leaked into training | clip-level split in a fixed file |
| **No validation loop** | "best" chosen on training loss | validation clip |
| **Reported ≠ on disk** (99.1% recall claimed; file said 73.8%) | | numbers only from run outputs |
| **Split bug (27 Sep)** — 26 crops silently excluded | | fixed in `split.py` |
| **Froth scale used the outer tube width** (found 20 Nov) | froth diameters 6.6% low | inner glass surfaces detected per frame |

Principles: split by clip; hand-count before looking at overlays; report D32, not just IoU; refuse microns without a measured calibration; report n as independent frames.

---

## 12. Poster plan (A1 portrait, below the yellow header)

**Title options:** "Teaching a computer to measure bubbles in flotation" · "Automated bubble sizing in a novel flotation cell with deep learning" · "Measuring bubbles the way an expert does — automatically"

**By-line:** Tristan Trieu · Supervisor: Prof Seher Ata · School of Minerals and Energy Resources Engineering, UNSW

| Row | Left column | Right column |
|---|---|---|
| 1 | **Aims** — automate bubble sizing from probe video; match hand-measurement accuracy for D32; test on unseen conditions; make it usable by researchers | **Background** — §3.1 + small schematic (bubble → mineral attachment; "smaller bubbles = more surface") |
| 2 | **Method** — pipeline strip: frame → hand labels → Mask R-CNN → outlines → D32 | **The ruler** — sharp vs defocused vs part-hidden, 3 crops with ✓/✗ |
| 3 | **Result 1: large bubbles fixed** — `fig1` | **Result 2: D32 at human level** — `fig2` + "D32 within 4%" |
| 4 | **Result 3: every unseen condition** — `fig5` | **Result 4: most "errors" were missed labels** — `fig3` |
| 5 | **Example** — `overlays/pulp_gt_vs_model.png` | **Froth** — §9.7 panel |
| 6 | **Conclusions** — ✓ automated, ✓ D32 within 1–4% (human 4%), ✓ recall 0.83–1.00 on all unseen clips; model revealed labelling bias; froth 0.86 recall / 0.98 local precision | **Impact & next steps** — BubbleMetrix (stops at ±5% D32, one-click fine-tuning with a before/after check), rig fixes (§10), measured calibration, bright regime |

**Hero numbers:** **0.14 → 0.90** large-bubble recall · **D32 within 4%** (fixed split) / **0.99 ± 0.03** (CV, 8 unseen clips) · **only 16%** of unmatched detections were model errors · froth **26/26** found, **< 1 min** per frame · **±5% D32 after ~6 froth frames**.

**Checklist:**
- [ ] Fonts Clancy / Roboto Mono installed and embedded
- [ ] Template header, fonts, background unchanged; size A1 Custom
- [ ] Supervisor name and school present
- [ ] Figures readable at A1 (≥ ~24 pt); legend on overlays
- [ ] No µm values presented as final for pulp (mark provisional or use px); froth mm uses the 55 mm inner width
- [ ] A4 laser test print checked
- [ ] PDF filename with full name + z5793457, submitted by **10 Dec 2026**

---

## 13. Final-report outline (adapt to Prof Ata's format)

1. **Introduction** — flotation, bubble size, D32; aims.
2. **Literature** — why bubble size matters (Gorain 1997, 1999; Finch 2000; Nesset 2006); measurement (Gomez & Finch 2007; Grau & Heiskanen 2005; Mesa 2022); froth vision (Aldrich 2010); deep-learning bubble detection (Kim & Park 2021; Hessenkemper 2022; Cerqueira & Paladino 2021; Wang 2021; He 2022; Cui 2022); Mask R-CNN; SAM; Cellpose; watershed.
3. **Data** — §4.
4. **Audit of the inherited tool** — §11.
5. **Method** — §5.
6. **Results** — §6 incl. convergence (§6.8).
7. **Froth** — §9.
8. **Software** — §17 (BubbleMetrix; manual in `docs/MANUAL.md`).
9. **Discussion** — §7.
10. **Limitations and rig recommendations** — §8, §10.
11. **Conclusion and future work** — blind label completeness pass + retrain; bright-regime fine-tune; measured pulp calibration; score t = 150 s froth frame and fix the boundary convention; Fiji integration.

---

## 14. Key-numbers cheat sheet

| Quantity | Value | Source |
|---|---|---|
| Pulp calibration (provisional) | 303.858 px/mm = 3.291 µm/px | provider field width |
| Froth calibration | **55 mm inner width = 471 px → 0.1168 mm/px** | inner glass surfaces, 24 frames |
| Typical rotor bubble | 73.5 px ≈ 244 µm | hand measurement |
| Size floor | 12 px = 40 µm | debris max 10.5 px |
| Human repeatability | count 0.82 · IoU 0.875 · D32 0.96 | 4 crops, two passes |
| Original BubMask | recall 0.05, D32 2.3× high | independent hand count |
| First honest model (7 Sep) | recall 0.52 / 0.69 (conf 0.72 / 0.40), large-bubble recall 0.14 | 48 bubbles |
| Final model, fixed split | recall 0.85, precision 0.46, IoU 0.89, D32 0.96 | 124 bubbles |
| Same, old model | recall 0.73, precision 0.42, IoU 0.83, D32 0.82 | 124 bubbles |
| Large-bubble recall (100–140 px) | 0.62 → **0.90** (fixed split); **0.97** pooled CV | |
| Precision after audit | 0.40 → 0.67 → 0.80 | 96 audited detections |
| Genuine model errors | 15 / 159 detections (9%); 16% of unmatched | audit |
| CV (8 folds) | recall 0.921 ± 0.058, precision 0.437 ± 0.048, F1 0.591 ± 0.045, IoU 0.888 ± 0.011, D32 0.985 ± 0.035; pooled recall 0.910, precision 0.444, D32 0.992 | 377 bubbles |
| Pulp D32 convergence | RotMed 300air: stable ±4.7% after 137 bubbles / 8 frames; Venturi 500/60: ±9.7% after 355, not stable | GUI, 20 Nov |
| Froth, labelled patch | recall 0.86, precision 0.68 strict / 0.98 local, F1 0.76 / 0.92; D32 ratio 1.10 | 51 bubbles |
| Froth, labelled crop | 26/26, IoU 0.80, D32 ratio 0.97 | 26 bubbles |
| Froth, video | **D32 2.16 mm** (2.06–2.30 per frame), 6,075 bubbles, 30–48 s/frame | 9 frames, 0–420 s |
| Froth D32 convergence | ±5% at ~1,100 bubbles (bubble-level); GUI rule stops after 6–7 frames | 9 frames |
| Best epoch | 16 (val loss 0.391) | rr_full_e20 |
| Labelled data | 383 bubbles on 35 full frames + crops (rotor); froth 77 scored + 160 to score | |

---

## 15. References

**From `readings/`** (verified from the PDFs):
- Cerqueira, R.F.L. & Paladino, E.E. (2021). Development of a deep learning-based image processing technique for bubble pattern recognition and shape reconstruction in dense bubbly flows. *Chemical Engineering Science*, 230, 116163.
- Cui, Y., Li, C., Zhang, W., Ning, X., Shi, X., Gao, J. & Lan, X. (2022). A deep learning-based image processing method for bubble detection, segmentation, and shape reconstruction in high gas holdup sub-millimeter bubbly flows. *Chemical Engineering Journal*, 449, 137859.
- He, Y., Hu, C., Li, H., Jiang, B., Hu, X., Wang, K. & Tang, D. (2022). A flexible image processing technique for measuring bubble parameters based on a neural network. *Chemical Engineering Journal*, 429, 132138.
- Mesa, D., Quintanilla, P. & Reyes, F. (2022). Bubble Analyser — An open-source software for bubble size measurement using image analysis. *Minerals Engineering*, 180, 107497.
- Wang, Q., Li, X., Xu, C., Yan, T. & Li, Y. (2021). Bubble recognizing and tracking in a plate heat exchanger by using image processing and convolutional neural network. *International Journal of Multiphase Flow*, 138, 103593.

**Flotation and bubble size:**
- Gorain, B.K., Franzidis, J.-P. & Manlapig, E.V. (1997). Studies on impeller type, impeller speed and air flow rate in an industrial scale flotation cell. Part 4: Effect of bubble surface area flux on flotation performance. *Minerals Engineering*, 10(4), 367–379.
- Gorain, B.K., Franzidis, J.-P. & Manlapig, E.V. (1999). The empirical prediction of bubble surface area flux in mechanical flotation cells from cell design and operating data. *Minerals Engineering*, 12(3), 309–322. (Confirm exact title.)
- Finch, J.A., Xiao, J., Hardie, C. & Gomez, C.O. (2000). Gas dispersion properties: bubble surface area flux and gas holdup. *Minerals Engineering*, 13(4), 365–372.
- Nesset, J.E., Hernandez-Aguilar, J.R., Acuna, C., Gomez, C.O. & Finch, J.A. (2006). Some gas dispersion characteristics of mechanical flotation machines. *Minerals Engineering*, 19(6–8), 807–815.
- Gomez, C.O. & Finch, J.A. (2007). Gas dispersion measurements in flotation cells. *International Journal of Mineral Processing*, 84(1–4), 51–58.
- Grau, R.A. & Heiskanen, K. (2005). Bubble size distribution in laboratory scale flotation cells. *Minerals Engineering*, 18(12), 1164–1172.
- Aldrich, C., Marais, C., Shean, B.J. & Cilliers, J.J. (2010). Online monitoring and control of froth flotation systems with machine vision: a review. *International Journal of Mineral Processing*, 96. (Pages 1–13 — confirm.)

**Bubble detection with deep learning:**
- Kim, Y. & Park, H. (2021). Deep learning-based automated and universal bubble detection and mask extraction in complex two-phase flows. *Scientific Reports*, 11, 8940. doi:10.1038/s41598-021-88334-0.
- Hessenkemper, H., Starke, S., Atassi, Y., Ziegenhein, T. & Lucas, D. (2022). Bubble identification from images with machine learning methods. *International Journal of Multiphase Flow*, 155, 104169.

**Models and training** (check page numbers before submitting):
- He, K., Gkioxari, G., Dollár, P. & Girshick, R. (2017). Mask R-CNN. *ICCV*, 2961–2969.
- Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B. & Belongie, S. (2017). Feature pyramid networks for object detection. *CVPR*, 2117–2125.
- Lin, T.-Y. et al. (2014). Microsoft COCO: Common objects in context. *ECCV*, 740–755.
- Li, Y., Xie, S., Chen, X., Dollár, P., He, K. & Girshick, R. (2021). Benchmarking detection transfer learning with vision transformers. arXiv:2111.11429.
- Loshchilov, I. & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*.
- Kirillov, A. et al. (2023). Segment Anything. *ICCV*, 4015–4026.
- Stringer, C., Wang, T., Michaelos, M. & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular segmentation. *Nature Methods*, 18, 100–106.
- Pachitariu, M. & Stringer, C. (2022). Cellpose 2.0: how to train your own model. *Nature Methods*, 19, 1634–1641. (Human-in-the-loop fine-tuning; basis of the GUI's minimum-label rule.)
- Pachitariu, M., Rariden, M. & Stringer, C. (2025). Cellpose-SAM: superhuman generalization for cellular segmentation. *bioRxiv*, doi:10.1101/2025.04.28.651001. (Check for a journal version.)
- Vincent, L. & Soille, P. (1991). Watersheds in digital spaces. *IEEE TPAMI*, 13(6), 583–598.
- Beucher, S. & Meyer, F. (1993). The morphological approach to segmentation: the watershed transformation. In *Mathematical Morphology in Image Processing*, Marcel Dekker, 433–481.

**Froth alternatives a reviewer may ask about:**
- Schmidt, U., Weigert, M., Broaddus, C. & Myers, G. (2018). Cell detection with star-convex polygons. *MICCAI*, LNCS 11071, 265–273. (StarDist.)
- Archit, A. et al. (2025). Segment Anything for Microscopy. *Nature Methods*, 22(3), 579–591. (μSAM.)

**Evaluation and statistics:**
- Roberts, D.R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40, 913–929.
- Wilson, E.B. (1927). Probable inference, the law of succession, and statistical inference. *JASA*, 22(158), 209–212.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153–157.
- Efron, B. & Tibshirani, R.J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall. (Bootstrap CIs for D32 convergence.)

**Not yet verified:** Zhang et al. (2019), froth watershed method.

**Software:** PyTorch / torchvision; Cellpose; OpenCV; Tkinter; Fiji/ImageJ; BubMask-Fiji (Armansyah, prior UNSW student).

---

## 16. File locations (`D:\Taste of Research\`, after the 20 Nov tidy-up; folders renamed 23 Nov: `Gemini_BubMask2` → `BubMask`, `Froth\bubsam3` → `Froth\roi`)

| What | Where |
|---|---|
| GUI | `bubble_gui.py`, `bubble_gui_teach.py`, `launch_gui.bat`, `engines\` |
| Manual | `docs\MANUAL.md` (+ `docs\img\`) |
| Pulp model code / final model | `BubMask\bubble_rcnn\`, `BubMask\runs\rr_full_e20\best.pt` |
| Fixed-split evaluation, audit | `BubMask\runs\eval_test\` |
| Cross-validation | `BubMask\runs\cv\` |
| Labels / images | `BubMask\data\` |
| Froth | `Froth\cellpose\` (runs, fine-tuning), `Froth\roi\` (in-focus band finder, `crop_measurable.py`), `Froth\eval\` (scoring, labels, results, D32 convergence) |
| Fine-tuning sets / models | `teaching\`, `models\` |
| Report material | `report\` (this kit, figures, overlays, deck, Faculty template + guidelines, readings) |
| Superseded work | `progress\` |
| Code on GitHub | see `README.md` |

---

## 17. Software: BubbleMetrix

Desktop tool (Python/Tkinter, Windows, CPU) wrapping both pipelines so a researcher picks a video and presses one button. Manual: `docs/MANUAL.md`.

| Feature | What it does |
|---|---|
| Pulp / Froth | Mask R-CNN (pulp) or Cellpose (froth, own environment via a helper process); settings fixed at their validated values |
| Scale | adjustable; pulp default 3.291 µm/px, froth measured from the 55 mm inner tube width on every frame |
| Measure Full Video | pulp: every frame outlined, D32 from every 5th (decorrelated) frame; froth: up to 40 spread frames, stops when D32 is stable |
| D32 Convergence tab | running D32 with its 95% band vs bubbles measured; "stable after N bubbles" or how many more are needed |
| Editor / Validation | fix detections by hand; score against hand labels (recall, precision, IoU, D32 ratio) |
| Fine-tune Model | add frames from new footage → correct the model's suggestions → see the current model's accuracy → fine-tune (background, ~30–60 min CPU). The new model is used only if it beats the old one on held-back frames **and** isn't worse on the frozen benchmark; one click reverts |
| Minimum labels | pulp 15 frames / 150 bubbles; froth 4 boxes / 300 bubbles (Cellpose 2.0: 100–200 corrected objects suffice when the model pre-labels) |
| Exports | CSV (per bubble, with frame) and JSON (summary incl. convergence) |
