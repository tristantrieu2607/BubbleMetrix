"""
froth_eval.py - score any froth method against hand labels, with one set of rules.

Same rules as the pulp harness (BubMask):
  * greedy one-to-one matching by MASK IoU >= 0.5
  * ground truth that a prediction overlaps but does not match is reported
    separately ("touched but unmatched") from ground truth nothing overlaps
  * nothing is tuned here, and nothing here should be tuned on these labels

A "case" is one labelled image, defined in cases.json. A case is either
  partial         only some bubbles were stamped -> recall, IoU and sizing only.
                  An unmatched prediction may be a real bubble nobody stamped,
                  so precision is NOT computed and is never printed.
  fully_labelled  every bubble >= the label floor was stamped -> precision, F1
                  and the whole-image D32 ratio as well.

Label conventions honoured (annotation/schema.py of the stamp tool):
  circles and ellipses; ignore_regions (rectangles) - a prediction mostly
  inside one is neither TP nor FP; bubbles flagged `unsure` - excluded from
  recall, but a prediction that matches one is not a false positive either;
  floor_px - predictions below the label floor are not false positives.

Predictions: either an *_instances.npz (BubSAM3 and Cellpose both write one)
or an integer label map *.npy (0 = background). The method must have been run
on the case image itself (with --no-roi, or with a ROI - the npz offset is
applied), OR, for a patch cut from a full frame (case key "frame_origin":
[x0, y0]), on that full frame: the prediction is then cropped to the patch
(a label map uses roi_offset from the <prefix>_run.json next to it). Patch
pixels outside the method's ROI are ignore, and GT mostly there is not counted:
the GUI never measures there either.

Patch-edge rule (crops): a prediction cut by the case image border can't be
judged, so it is never a false positive (it may still match). Ground truth cut
by the border is already `unsure` in border-annotator labels.

--unsure-as-ignore ("local" precision): unsure GT regions also count as ignore
for false positives, i.e. predictions lying mostly in area the annotator left
unresolved are not counted against the method. Reported alongside the strict
number, never instead of it.

  python eval\\froth_eval.py score --case f299_partial --pred X_instances.npz --name BubSAM3
  python eval\\froth_eval.py table --case f299_partial
  python eval\\froth_eval.py audit                   (precision from the mask audit)
"""
import argparse
import glob
import json
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, "cases.json")
RESULTS = os.path.join(HERE, "results")
IOU_THR = 0.5
SUB = 16                       # sub-pixel rasterisation (cv2 shift=4)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_case(name):
    cases = json.load(open(CASES))
    if name not in cases:
        sys.exit("unknown case %r; cases.json has: %s" % (name, ", ".join(cases)))
    c = dict(cases[name])
    c["name"] = name
    for k in ("image", "labels"):
        c[k] = os.path.join(HERE, c[k])
    img = cv2.imread(c["image"])
    if img is None:
        sys.exit("cannot read case image %s" % c["image"])
    c["img"] = img
    c["shape"] = img.shape[:2]
    return c


def _paste(arr, ox, oy, H, W):
    """Place a label-space array into case-image space (image = label - offset)."""
    out = np.zeros((H, W), arr.dtype)
    h, w = arr.shape
    x0, y0 = -ox, -oy                      # where label (0,0) lands in the image
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    ww, hh = min(w - sx0, W - dx0), min(h - sy0, H - dy0)
    if ww > 0 and hh > 0:
        out[dy0:dy0 + hh, dx0:dx0 + ww] = arr[sy0:sy0 + hh, sx0:sx0 + ww]
    return out


def load_gt_regions(doc, case):
    """Border-drawn labels (border_annotator.py): a 16-bit instance map, exact masks."""
    from PIL import Image
    d = os.path.dirname(case["labels"])
    ox, oy = case.get("label_offset", [0, 0])
    H, W = case["shape"]
    inst = _paste(np.array(Image.open(os.path.join(d, doc["instances_png"]))).astype(np.int32), ox, oy, H, W)
    ign = np.zeros((H, W), bool)
    if doc.get("ignore_png") and os.path.exists(os.path.join(d, doc["ignore_png"])):
        ign = _paste(np.array(Image.open(os.path.join(d, doc["ignore_png"]))) > 0, ox, oy, H, W)
    # frame_annotator.py: only tiles marked complete are labelled completely; the
    # rest of the image is ignore, and GT mostly outside them is not counted.
    outside = None
    if doc.get("complete_png") and os.path.exists(os.path.join(d, doc["complete_png"])):
        outside = _paste(np.array(Image.open(os.path.join(d, doc["complete_png"]))) > 0, ox, oy, H, W) == 0
        ign = ign | outside
    gts = []
    for b in doc.get("bubbles", []):
        m = inst == int(b["id"])
        if not m.any():
            continue
        uns = bool(b.get("unsure", False))
        if outside is not None and (m & outside).sum() > 0.5 * m.sum():
            uns = True
        gts.append(dict(mask=m, d=2 * math.sqrt(m.sum() / math.pi), unsure=uns))
    return gts, ign, float(doc.get("floor_px", 5.0)), bool(doc.get("confirmed_empty", False))


def load_gt(case):
    doc = json.load(open(case["labels"]))
    if doc.get("format") == "regions":
        return load_gt_regions(doc, case)
    ox, oy = case.get("label_offset", [0, 0])
    H, W = case["shape"]
    gts = []
    for b in doc.get("bubbles", []):
        m = np.zeros((H, W), np.uint8)
        centre = (int(round((b["cx"] - ox) * SUB)), int(round((b["cy"] - oy) * SUB)))
        if b.get("shape_type") == "ellipse":
            r2 = float(b.get("r", 0.0)) * 2
            a, c = float(b.get("axis_w", r2)) / 2, float(b.get("axis_h", r2)) / 2
            cv2.ellipse(m, centre, (int(round(a * SUB)), int(round(c * SUB))),
                        float(b.get("angle", 0.0)), 0, 360, 1, -1, shift=4)
        else:
            cv2.circle(m, centre, int(round(float(b["r"]) * SUB)), 1, -1, shift=4)
        if not m.any():
            continue                                   # outside this image
        gts.append(dict(mask=m.astype(bool), d=2 * math.sqrt(m.sum() / math.pi),
                        unsure=bool(b.get("unsure", False))))
    ign = np.zeros((H, W), bool)
    for r in doc.get("ignore_regions", []) or []:
        x1, y1 = int(round(r["x1"] - ox)), int(round(r["y1"] - oy))
        x2, y2 = int(round(r["x2"] - ox)), int(round(r["y2"] - oy))
        ign[max(y1, 0):max(y2 + 1, 0), max(x1, 0):max(x2 + 1, 0)] = True
    return gts, ign, float(doc.get("floor_px", 12.0)), bool(doc.get("confirmed_empty", False))


ROI_OUTSIDE = {}      # pred path -> patch pixels outside the method's ROI (scored as ignore)


def load_pred(path, shape, exclude_sources=(), frame_origin=None):
    H, W = shape
    preds = []
    fx, fy = frame_origin if frame_origin else (0, 0)
    if path.endswith(".npy"):
        lab = np.load(path)
        if frame_origin is not None:
            # label map covers the method's ROI of the full frame -> crop the patch
            rj = path[:-len("_labels.npy")] + "_run.json" if path.endswith("_labels.npy") else None
            if not rj or not os.path.exists(rj):
                sys.exit("need %s for the label map's roi_offset" % rj)
            rx, ry = [int(v) for v in json.load(open(rj)).get("roi_offset", [0, 0])]
            # Part of the patch outside the method's ROI (e.g. the hazy band above the
            # in-focus region) is never measured by the GUI: it becomes ignore.
            inside = _paste(np.ones(lab.shape, bool), fx - rx, fy - ry, H, W)
            lab = _paste(lab, fx - rx, fy - ry, H, W)
            ROI_OUTSIDE[path] = ~inside
        if lab.shape != (H, W):
            sys.exit("label map %s is %s, case image is %s - run the method on the "
                     "case image with --no-roi" % (path, lab.shape, (H, W)))
        for k in np.unique(lab):
            if k > 0:
                preds.append(dict(mask=(lab == k), source="labelmap"))
    else:
        z = np.load(path, allow_pickle=True)
        ox, oy = [int(v) for v in z["offset"]] if "offset" in z else (0, 0)
        ox, oy = ox - fx, oy - fy
        srcs = z["source"] if "source" in z else ["?"] * len(z["polys"])
        for p, s in zip(z["polys"], srcs):
            if str(s) in exclude_sources:
                continue
            m = np.zeros((H, W), np.uint8)
            pts = np.round((np.asarray(p, np.float64) + [ox, oy]) * SUB).astype(np.int32)
            cv2.fillPoly(m, [pts], 1, shift=4)
            if m.any():
                preds.append(dict(mask=m.astype(bool), source=str(s)))
    for q in preds:
        mk = q["mask"]
        q["edge"] = bool(mk[0].any() or mk[-1].any() or mk[:, 0].any() or mk[:, -1].any())
        q["area"] = int(q["mask"].sum())
        q["d"] = 2 * math.sqrt(q["area"] / math.pi)
    return preds


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 3), round((c + h) / d, 3))


def d32(ds):
    ds = np.asarray(ds, float)
    return float((ds ** 3).sum() / (ds ** 2).sum()) if len(ds) else float("nan")


def match(gts, preds):
    ious = np.zeros((len(gts), len(preds)))
    for i, g in enumerate(gts):
        ys, xs = np.nonzero(g["mask"])
        ga = len(ys)
        for j, p in enumerate(preds):
            inter = int(p["mask"][ys, xs].sum())
            if inter:
                ious[i, j] = inter / (ga + p["area"] - inter)
    pairs = sorted(((ious[i, j], i, j) for i, j in zip(*np.nonzero(ious >= IOU_THR))),
                   reverse=True)
    gi, pj, m = set(), set(), {}
    for v, i, j in pairs:
        if i in gi or j in pj:
            continue
        gi.add(i); pj.add(j); m[i] = (j, float(v))
    return m, ious


def score(case, preds, name, unsure_as_ignore=False, outside_roi=None):
    gts, ign, floor, confirmed_empty = load_gt(case)
    n_outside = 0
    if outside_roi is not None and outside_roi.any():
        # GT mostly outside the method's ROI is not counted (the method never looked there)
        n_outside = sum(1 for g in gts if not g["unsure"] and (g["mask"] & outside_roi).sum() > 0.5 * g["mask"].sum())
        for g in gts:
            if (g["mask"] & outside_roi).sum() > 0.5 * g["mask"].sum():
                g["unsure"] = True
        ign = ign | outside_roi
    if unsure_as_ignore:
        ign = ign.copy()
        for g in gts:
            if g["unsure"]:
                ign |= g["mask"]
    m, ious = match(gts, preds)
    counted = [i for i, g in enumerate(gts) if not g["unsure"]]
    tp = [i for i in counted if i in m]
    fn = [i for i in counted if i not in m]
    pairs = [(gts[i]["d"], preds[m[i][0]]["d"], m[i][1]) for i in tp]

    out = dict(method=name, case=case["name"], fully_labelled=bool(case.get("fully_labelled")),
               n_gt=len(counted), n_gt_unsure=len(gts) - len(counted), n_pred=len(preds),
               tp=len(tp), recall=len(tp) / len(counted) if counted else float("nan"),
               recall_ci95=wilson(len(tp), len(counted)),
               mean_iou=float(np.mean([p[2] for p in pairs])) if pairs else float("nan"),
               mean_diam_ratio=float(np.mean([p[1] / p[0] for p in pairs])) if pairs else float("nan"),
               median_abs_diam_err=float(np.median([abs(p[1] / p[0] - 1) for p in pairs])) if pairs else float("nan"),
               d32_ratio_matched=d32([p[1] for p in pairs]) / d32([p[0] for p in pairs]) if pairs else float("nan"),
               missed_entirely=sum(1 for i in fn if ious[i].max(initial=0) == 0),
               touched_unmatched=sum(1 for i in fn if ious[i].max(initial=0) > 0),
               label_floor_px=floor, n_gt_outside_method_roi=n_outside)
    bands = [(0, 12), (12, 17), (17, 25), (25, 1e9)]
    out["recall_by_band"] = {
        ("%d+" % a if b > 1e8 else "%d-%d" % (a, b)):
        "%d/%d" % (sum(1 for i in tp if a <= gts[i]["d"] < b),
                   sum(1 for i in counted if a <= gts[i]["d"] < b))
        for a, b in bands}

    if case.get("fully_labelled"):
        matched_p = {m[i][0] for i in m}                      # incl. matches to `unsure`
        judged = [j for j, p in enumerate(preds) if not p["edge"]
                  and p["d"] >= floor and (p["mask"] & ign).sum() < 0.5 * p["area"]]
        fp = [j for j in judged if j not in matched_p]
        out["unsure_as_ignore"] = bool(unsure_as_ignore)
        out["n_pred_judged"] = len(judged)
        out["n_pred_edge"] = sum(1 for p in preds if p["edge"])
        out["fp_d_median"] = float(np.median([preds[j]["d"] for j in fp])) if fp else float("nan")
        n_tp = len(tp)
        out["fp"] = len(fp)
        out["precision"] = n_tp / (n_tp + len(fp)) if (n_tp + len(fp)) else float("nan")
        out["precision_ci95"] = wilson(n_tp, n_tp + len(fp))
        r, pr = out["recall"], out["precision"]
        out["f1"] = 2 * r * pr / (r + pr) if (r + pr) > 0 else float("nan")
        kept = [preds[j]["d"] for j in judged]
        out["d32_ratio_image"] = d32(kept) / d32([gts[i]["d"] for i in counted]) if counted else float("nan")
        out["count_ratio_image"] = len(kept) / len(counted) if counted else float("nan")
    else:
        out["precision"] = None
        out["precision_note"] = "not computable: partially labelled case"
    return out, gts, m


def overlay(case, gts, preds, m, path):
    Z = max(3, 768 // max(case["shape"]))
    img = cv2.resize(case["img"], None, fx=Z, fy=Z, interpolation=cv2.INTER_CUBIC)
    matched_p = {m[i][0] for i in m}
    for j, p in enumerate(preds):
        cs, _ = cv2.findContours(cv2.resize(p["mask"].astype(np.uint8), None, fx=Z, fy=Z,
                                            interpolation=cv2.INTER_NEAREST),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, (255, 200, 0) if j in matched_p else (0, 165, 255), 1)
    for i, g in enumerate(gts):
        cs, _ = cv2.findContours(cv2.resize(g["mask"].astype(np.uint8), None, fx=Z, fy=Z,
                                            interpolation=cv2.INTER_NEAREST),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        col = (160, 160, 160) if g["unsure"] else ((0, 220, 0) if i in m else (0, 0, 255))
        cv2.drawContours(img, cs, -1, col, 2)
    cv2.putText(img, "GT: green matched, red missed | pred: blue matched, orange unmatched",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imwrite(path, img)


# --------------------------------------------------------------------------
# Table
# --------------------------------------------------------------------------
def fmt(v, nd=2):
    if v is None:
        return "n/a"
    if isinstance(v, float) and math.isnan(v):
        return "-"
    return ("%%.%df" % nd) % v if isinstance(v, float) else str(v)


def table(case_name):
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(RESULTS, case_name, "*_score.json")))]
    if not rows:
        sys.exit("no scores yet under results/%s" % case_name)
    full = any(r.get("fully_labelled") for r in rows)
    head = ["method", "found / GT", "recall (95% CI)", "mean IoU", "diam ratio", "D32 ratio (matched)",
            "missed / touched", "n pred"]
    if full:
        head[3:3] = ["precision (95% CI)", "F1", "D32 ratio (image)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in sorted(rows, key=lambda r: -r["recall"]):
        cells = [r["method"], "%d/%d" % (r["tp"], r["n_gt"]),
                 "%s (%s-%s)" % (fmt(r["recall"]), fmt(r["recall_ci95"][0]), fmt(r["recall_ci95"][1])),
                 fmt(r["mean_iou"]), fmt(r["mean_diam_ratio"]), fmt(r["d32_ratio_matched"]),
                 "%d / %d" % (r["missed_entirely"], r["touched_unmatched"]), str(r["n_pred"])]
        if full:
            pc = r.get("precision_ci95") or [float("nan")] * 2
            cells[3:3] = ["%s (%s-%s)" % (fmt(r.get("precision")), fmt(pc[0]), fmt(pc[1])),
                          fmt(r.get("f1")), fmt(r.get("d32_ratio_image"))]
        lines.append("| " + " | ".join(cells) + " |")
    note = ("" if full else "\nPartially labelled case: precision is not computable and is not shown.\n")
    md = "\n".join(lines) + "\n" + note
    open(os.path.join(RESULTS, case_name, "comparison.md"), "w").write(md)
    print(md)


# --------------------------------------------------------------------------
# Precision audit (make_audit.py + audit_masks.py)
# --------------------------------------------------------------------------
def audit_summary(folder=None, n_boot=2000, seed=0):
    """Precision from the yes/no audit of sampled Cellpose masks.

    The sample is stratified by size (thirds of each frame's masks), so each
    stratum's precision is re-weighted to the population: by count (how many
    masks are right) and by volume d^3 (what D32 sees). 95% CI by bootstrap
    within strata."""
    folder = folder or os.path.join(HERE, "audit")
    doc = json.load(open(os.path.join(folder, "items.json")))
    ans = json.load(open(os.path.join(folder, "answers.json"))).get("answers", {})
    n_s, v_s = np.zeros(3), np.zeros(3)
    for st in doc["strata"].values():
        n_s += st["n"]; v_s += st["vol"]
    per = {b: [] for b in range(3)}
    tally = {k: 0 for k in "YNMPU"}
    for it in doc["items"]:
        a = ans.get(it["id"])
        if a is None:
            continue
        tally[a] += 1
        if a != "U":
            per[it["stratum"]].append(1.0 if a == "Y" else 0.0)
    judged = sum(len(v) for v in per.values())
    if judged == 0:
        sys.exit("no answers yet in %s" % os.path.join(folder, "answers.json"))

    def est(samples, w):
        ps = np.array([np.mean(x) if len(x) else np.nan for x in samples])
        ok = ~np.isnan(ps)
        return float((ps[ok] * w[ok]).sum() / w[ok].sum())

    samples = [np.array(per[b]) for b in range(3)]
    rng = np.random.default_rng(seed)
    boots_n, boots_v = [], []
    for _ in range(n_boot):
        rs = [x[rng.integers(0, len(x), len(x))] if len(x) else x for x in samples]
        boots_n.append(est(rs, n_s)); boots_v.append(est(rs, v_s))
    tp = tally["Y"]
    out = dict(
        n_items=len(doc["items"]), n_answered=sum(tally.values()), n_judged=judged, answers=tally,
        precision_raw=tp / judged, precision_raw_ci95=wilson(tp, judged),
        precision_count_weighted=est(samples, n_s),
        precision_count_ci95=[round(float(np.percentile(boots_n, q)), 3) for q in (2.5, 97.5)],
        precision_volume_weighted=est(samples, v_s),
        precision_volume_ci95=[round(float(np.percentile(boots_v, q)), 3) for q in (2.5, 97.5)],
        by_size_third={("small", "middle", "large")[b]: "%d/%d" % (int(sum(per[b])), len(per[b])) for b in range(3)},
        error_types={"not a bubble": tally["N"], "merged": tally["M"], "part (split)": tally["P"]},
        population_share_count=(n_s / n_s.sum()).round(3).tolist(),
        population_share_volume=(v_s / v_s.sum()).round(3).tolist(),
        method=doc.get("method"), rule=doc.get("rule"))
    od = os.path.join(RESULTS, "audit")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "summary.json"), "w"), indent=2, default=float)
    md = ("| precision | value | 95%% CI |\n|---|---|---|\n"
          "| by count (population-weighted) | %.2f | %.2f-%.2f |\n"
          "| by volume, what D32 sees | %.2f | %.2f-%.2f |\n"
          "| raw sample | %.2f | %.2f-%.2f |\n\n"
          "Judged %d of %d masks (%d can't tell). Errors: %d not a bubble, %d merged, %d part. "
          "By size third (small/middle/large): %s.\n" % (
              out["precision_count_weighted"], *out["precision_count_ci95"],
              out["precision_volume_weighted"], *out["precision_volume_ci95"],
              out["precision_raw"], *out["precision_raw_ci95"],
              judged, len(doc["items"]), tally["U"], tally["N"], tally["M"], tally["P"],
              " / ".join(out["by_size_third"].values())))
    open(os.path.join(od, "summary.md"), "w").write(md)
    print(md)
    return out


# --------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--case", required=True)
    s.add_argument("--pred", required=True, help="*_instances.npz or label map *.npy")
    s.add_argument("--name", required=True, help="method name for the table")
    s.add_argument("--exclude-source", nargs="*", default=[])
    s.add_argument("--unsure-as-ignore", action="store_true",
                   help="local precision: predictions mostly inside unsure GT regions are not FP")
    t = sub.add_parser("table")
    t.add_argument("--case", required=True)
    au = sub.add_parser("audit", help="precision from the audit_masks.py answers")
    au.add_argument("--folder", default=None)
    a = p.parse_args(argv)

    if a.cmd == "audit":
        return audit_summary(a.folder)
    if a.cmd == "table":
        return table(a.case)
    case = load_case(a.case)
    preds = load_pred(a.pred, case["shape"], tuple(a.exclude_source), case.get("frame_origin"))
    out, gts, m = score(case, preds, a.name, a.unsure_as_ignore, ROI_OUTSIDE.get(a.pred))
    try:                                    # relative, so results survive moving the drive
        out["pred_file"] = os.path.relpath(os.path.abspath(a.pred), HERE)
    except ValueError:                      # different drive letter on Windows
        out["pred_file"] = os.path.abspath(a.pred)
    od = os.path.join(RESULTS, case["name"])
    os.makedirs(od, exist_ok=True)
    slug = "_".join("".join(ch if ch.isalnum() else " " for ch in a.name).split()).lower()
    json.dump(out, open(os.path.join(od, slug + "_score.json"), "w"), indent=2, default=float)
    overlay(case, gts, preds, m, os.path.join(od, slug + "_overlay.png"))
    print(json.dumps({k: v for k, v in out.items() if k != "pred_file"}, indent=1, default=float))


if __name__ == "__main__":
    main()
