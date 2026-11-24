"""Leave-one-clip-out cross-validation over the 8 rotor clips.

For each fold k: train on the other 7 clips for a fixed number of epochs, keep the
final model (last.pt), score on the held-out clip. The epoch count comes from the
fixed split, where a validation clip exists -- no clip is spent on checkpoint
selection here. --with-val restores the 6 train / 1 val / 1 test layout. Then aggregate: mean and spread per metric across folds.
This is the number to report -- there is no more footage, so every clip has to
serve as test once.

Run from BubMask\\ (only after settings are frozen on the fixed split):
    .venv\\Scripts\\python.exe scripts\\run_cv.py              # all 8 folds
    .venv\\Scripts\\python.exe scripts\\run_cv.py --folds 0 3  # a subset
Finished folds are skipped on re-run, so it can be stopped and resumed.
"""
import argparse, json, os, statistics, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bubble_rcnn import split as splitmod

METRICS = ["recall", "precision", "f1_score", "mean_mask_iou_matched", "d32_ratio_gated"]


def run(cmd):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--out", default="runs/cv")
    ap.add_argument("--with-val", action="store_true", help="6 train / 1 val / 1 test, score best.pt")
    a = ap.parse_args()

    clips = splitmod.all_clips()
    folds = a.folds if a.folds else list(range(len(clips)))
    py = sys.executable
    rows = []
    pool = []
    for k in folds:
        f = splitmod.make_fold(k, with_val=a.with_val)
        ckpt = "best.pt" if a.with_val else "last.pt"
        d = os.path.join(a.out, f"fold{k}")
        summ = os.path.join(d, "eval", "eval_summary.json")
        print(f"\n=== fold {k}: test={f['test'][0]}  val={f['val'][0] if f['val'] else '(none)'}")
        # last.pt is written after every epoch, so its existence does not mean the fold
        # finished -- the manifest's status does. An interrupted fold resumes.
        man = os.path.join(ROOT, d, "train_manifest.json")
        done = os.path.exists(man) and json.load(open(man, encoding="utf-8")).get("status") == "complete"
        if not done:
            cmd = [py, "-m", "bubble_rcnn.cli", "train", "--regime", "rotor_raw",
                   "--fold", str(k), "--epochs", str(a.epochs), "--out", d] + ([] if a.with_val else ["--no-val"])
            if os.path.exists(os.path.join(ROOT, d, "last.pt")):
                print(f"   fold {k} was interrupted -- resuming from last.pt")
                cmd += ["--resume", os.path.join(d, "last.pt")]
            run(cmd)
        if not os.path.exists(os.path.join(ROOT, summ)):
            run([py, "-m", "bubble_rcnn.cli", "eval", "--weights", os.path.join(d, ckpt),
                 "--split", "test", "--fold", str(k), "--conf", str(a.conf),
                 "--out", os.path.join(d, "eval")])
        r = json.load(open(os.path.join(ROOT, summ), encoding="utf-8"))
        pool.append(r)
        rows.append({"fold": k, "test_clip": f["test"][0], "n_gt": r.get("total_gt_bubbles"),
                     **{m: r.get(m) for m in METRICS}})

    agg = {}
    for m in METRICS:
        v = [r[m] for r in rows if r[m] is not None]
        if v:
            agg[m] = {"mean": statistics.mean(v), "sd": statistics.stdev(v) if len(v) > 1 else 0.0,
                      "min": min(v), "max": max(v)}
    # pooled: every test bubble counted once, so clips with more bubbles weigh more
    tp = sum(r["true_positives"] for r in pool)
    pr = sum(r["total_pred_bubbles"] for r in pool)
    gtn = sum(r["total_gt_bubbles"] for r in pool)
    gd = [d for r in pool for d in r.get("gt_d_px", [])]
    pdl = [d for r in pool for d in r.get("pred_d_px", [])]
    d32 = lambda ds: (sum(x ** 3 for x in ds) / sum(x ** 2 for x in ds)) if ds else 0.0
    bands = {}
    for r in pool:
        for b, v in r.get("recall_by_size_band", {}).items():
            bb = bands.setdefault(b, {"gt": 0, "tp": 0})
            bb["gt"] += v["gt"]; bb["tp"] += v["tp"]
    for v in bands.values():
        v["recall"] = v["tp"] / v["gt"] if v["gt"] else 0.0
    pooled = {"gt": gtn, "pred": pr, "tp": tp, "recall": tp / max(1, gtn), "precision": tp / max(1, pr),
              "d32_ratio_px": (d32(pdl) / d32(gd)) if gd else None, "recall_by_size_band": bands}
    out = {"pooled": pooled, "conf": a.conf, "epochs": a.epochs, "checkpoint": ckpt, "with_val": a.with_val, "folds": rows, "aggregate": agg}
    with open(os.path.join(ROOT, a.out, "cv_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("\nfold  test clip                                          n_gt  recall  prec   IoU    D32")
    for r in rows:
        print(f"{r['fold']:>4}  {r['test_clip']:<50} {r['n_gt'] or 0:>4}  "
              + "  ".join(f"{(r[m] or 0):.3f}" for m in ["recall", "precision", "mean_mask_iou_matched", "d32_ratio_gated"]))
    print("mean  " + " " * 56 + "  ".join(f"{agg[m]['mean']:.3f}" for m in ["recall", "precision", "mean_mask_iou_matched", "d32_ratio_gated"] if m in agg))
    print("sd    " + " " * 56 + "  ".join(f"{agg[m]['sd']:.3f}" for m in ["recall", "precision", "mean_mask_iou_matched", "d32_ratio_gated"] if m in agg))
    print(f"\npooled over all folds: GT {gtn} | pred {pr} | TP {tp} | recall {pooled['recall']:.3f} | "
          f"precision {pooled['precision']:.3f} | D32 ratio {pooled['d32_ratio_px'] or 0:.3f}")
    for b in sorted(bands):
        print(f"   {b:>10s}  n={bands[b]['gt']:3d}  recall {bands[b]['recall']:.2f}")
    print(f"\nwritten {os.path.join(a.out, 'cv_summary.json')}")


if __name__ == "__main__":
    main()
