"""finetune_cellpose.py - fine-tune Cellpose on a froth teaching set (GUI button).

Runs in Froth\\.venv (Cellpose lives there; see gui_worker.py for why). Started by
the GUI; nothing to set:

  python finetune_cellpose.py --set <teaching set folder> --out <model folder> [--base <cellpose model file>]

Recipe (Cellpose 2.0 / Cellpose-SAM fine-tuning defaults, fewer epochs for CPU):
  * start from the model in use (cpsam_v2 unless a fine-tune is active)
  * train on the checked 192 px boxes, 256 px crops, rotation/scale augmentation
    inside Cellpose; AdamW lr 1e-5, weight decay 0.1, EPOCHS epochs x 8 images
  * only the last TRAIN_BLOCKS ViT blocks + neck + head train (CPU memory and time;
    full training of ~300 M parameters ran out of 7 GB RAM in testing)
  * keep the result, then the check: old and new model on (a) the held-back boxes
    (fully labelled: precision, recall, F1, D32) and (b) the frozen f299 crop
    (26 hand circles, partial: recall and IoU only), scored as the GUI measures.
    engines/teach.decide() says adopt or keep the old model.

Progress: stdout lines starting with "@@ " (JSON). result.json in --out.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FROTH_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(FROTH_DIR)
ENGINES = os.path.join(ROOT, "engines")
if ENGINES not in sys.path:
    sys.path.insert(0, ENGINES)

import numpy as np
from PIL import Image

import teach
from engine_metrology import summarize_bubbles
from engine_froth import labels_to_bubbles

EPOCHS = 40
NIMG_PER_EPOCH = 8
LR = 1e-5
WEIGHT_DECAY = 0.1
TRAIN_BLOCKS = 4          # last 4 of 24 ViT-L blocks + neck + head train; the rest stay frozen (CPU RAM/time)
EVAL_KW = dict(flow_threshold=0.4, cellprob_threshold=0.0, min_size=15)   # = gui_worker.py


def say(**kw):
    sys.stdout.write("@@ " + json.dumps(kw) + "\n")
    sys.stdout.flush()


def rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def accepted(model, img):
    lab = np.asarray(model.eval(img, **EVAL_KW)[0]).astype(np.int32)
    recs = labels_to_bubbles(lab, (0, 0))
    return summarize_bubbles(recs, um_per_px=None, exclude_edge=True)["accepted_bubbles"]


def score(model, pairs, tag, tick):
    per = []
    for img_p, bubbles in pairs:
        img = rgb(img_p)
        h, w = img.shape[:2]
        per.append(teach.score_image(accepted(model, img), bubbles, 0.0, w, h))
        tick(tag)
    return teach.pool_scores(per)


def benchmark_pairs():
    """The f299 crop: 26 hand circles from before any froth model was run on it."""
    cases_p = os.path.join(FROTH_DIR, "eval", "cases.json")
    if not os.path.exists(cases_p):
        return []
    case = json.load(open(cases_p, encoding="utf-8")).get("f299_partial")
    if not case:
        return []
    img_p = os.path.join(FROTH_DIR, "eval", case["image"])
    lab_p = os.path.join(FROTH_DIR, "eval", case["labels"])
    if not (os.path.exists(img_p) and os.path.exists(lab_p)):
        return []
    ox, oy = case.get("label_offset", [0, 0])
    out = []
    for b in json.load(open(lab_p, encoding="utf-8")).get("bubbles", []):
        b = dict(b)
        b["cx"] = float(b["cx"]) - ox
        b["cy"] = float(b["cy"]) - oy
        out.append(b)
    return [(img_p, out)]


def load_model(path, gpu):
    from cellpose import models
    kw = {"gpu": gpu}
    try:
        return models.CellposeModel(pretrained_model=path or "cpsam_v2", use_bfloat16=gpu, **kw)
    except TypeError:
        return models.CellposeModel(pretrained_model=path or "cpsam_v2", **kw)


def freeze_early_encoder(net, train_blocks: int) -> int:
    """Train only the last `train_blocks` ViT blocks, the neck and the output head.

    Full Cellpose-SAM training (~300 M parameters) needs > 7 GB of RAM on CPU for the
    gradients and AdamW state and is slow; the early blocks hold generic image features
    that a few hundred froth bubbles cannot improve. AdamW keeps no state for frozen
    parameters, so memory and time drop roughly in proportion. train_blocks <= 0 trains
    everything (the Cellpose default, for a GPU)."""
    enc = getattr(net, "encoder", None)
    if enc is None or train_blocks <= 0 or not hasattr(enc, "blocks"):
        return sum(p.numel() for p in net.parameters())
    for p in enc.parameters():
        p.requires_grad = False
    for blk in list(enc.blocks)[-train_blocks:]:
        for p in blk.parameters():
            p.requires_grad = True
    if hasattr(enc, "neck"):
        for p in enc.neck.parameters():
            p.requires_grad = True
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-blocks", type=int, default=TRAIN_BLOCKS,
                    help="ViT blocks to train (0 = all, the GPU default)")
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--nimg", type=int, default=NIMG_PER_EPOCH)
    ap.add_argument("--no-bench", action="store_true", help="testing only")
    a = ap.parse_args()

    import torch
    from cellpose import train
    gpu = bool(torch.cuda.is_available())
    os.makedirs(a.out, exist_ok=True)

    ts = teach.TeachingSet(a.set)
    train_items, check_items = ts.split()
    say(stage="setup", train=len(train_items), check=len(check_items), device="cuda" if gpu else "cpu")

    def data(items):
        X, Y = [], []
        for it in items:
            img = rgb(ts.image_path(it))
            X.append(img)
            Y.append(teach.instance_map(ts.read_labels(it), img.shape[0], img.shape[1]))
        return X, Y

    Xtr, Ytr = data(train_items)
    Xte, Yte = data(check_items)
    if not Xtr:
        raise SystemExit("No checked training boxes in the teaching set.")

    base = a.base or None
    model = load_model(base, gpu)
    trainable = freeze_early_encoder(model.net, a.train_blocks)
    say(stage="setup", trainable_params=trainable)

    # progress: count the augmentation calls train_seg makes (one per batch)
    n_train_calls = a.epochs * a.nimg
    test_epochs = [e for e in range(a.epochs) if e == 5 or e % 10 == 0] if Xte else []
    total = n_train_calls + len(test_epochs) * len(Xte)
    orig = train.random_rotate_and_resize
    state = {"n": 0, "t0": time.time()}

    def counted(*args, **kw):
        state["n"] += 1
        el = time.time() - state["t0"]
        n = state["n"]
        say(stage="train", step=n, steps=total, epoch=min(a.epochs, 1 + (n - 1) // max(1, a.nimg)),
            of=a.epochs, eta_s=round(el / n * max(0, total - n)))
        return orig(*args, **kw)

    train.random_rotate_and_resize = counted
    say(stage="train", step=0, steps=total, epoch=0, of=a.epochs, eta_s=None)
    t0 = time.time()
    path, tr_loss, te_loss = train.train_seg(
        model.net, train_data=Xtr, train_labels=Ytr,
        test_data=Xte or None, test_labels=Yte or None,
        channel_axis=-1, load_files=False, normalize=True,
        batch_size=1, learning_rate=LR, weight_decay=WEIGHT_DECAY,
        n_epochs=a.epochs, nimg_per_epoch=a.nimg, bsize=256, min_train_masks=1,
        save_path=a.out, model_name="froth_finetuned")
    train.random_rotate_and_resize = orig
    train_min = (time.time() - t0) / 60
    cand = os.path.join(a.out, "froth_finetuned")
    if os.path.abspath(str(path)) != os.path.abspath(cand):
        shutil.copy2(str(path), cand)

    # ---------------- the check: old vs new ----------------
    chk_pairs = [(ts.image_path(it), ts.read_labels(it)) for it in check_items]
    bench = [] if a.no_bench else benchmark_pairs()
    n_all = 2 * (len(chk_pairs) + len(bench))
    cnt = [0]

    def tick(tag):
        cnt[0] += 1
        say(stage="check", done=cnt[0], of=max(1, n_all), what=tag)

    del model                   # one ViT-L in memory at a time (laptop RAM)
    import gc
    gc.collect()
    old_model = load_model(base, gpu)
    old_chk = score(old_model, chk_pairs, "old model, your boxes", tick)
    old_b = score(old_model, bench, "old model, benchmark", tick) if bench else None
    del old_model
    gc.collect()
    new_model = load_model(cand, gpu)
    new_chk = score(new_model, chk_pairs, "new model, your boxes", tick)
    new_b = score(new_model, bench, "new model, benchmark", tick) if bench else None
    adopt, reasons = teach.decide(old_chk, new_chk, old_b, new_b, bench_partial=True)

    result = {"regime": "froth", "adopt": adopt, "reasons": reasons,
              "model_path": os.path.abspath(cand), "base": base or "cpsam_v2",
              "teaching_set": os.path.abspath(ts.folder),
              "train_items": [it["stem"] for it in train_items],
              "check_items": [it["stem"] for it in check_items],
              "train_minutes": round(train_min, 1),
              "train_loss": [float(x) for x in np.asarray(tr_loss).ravel()],
              "check_old": old_chk, "check_new": new_chk, "bench_old": old_b, "bench_new": new_b,
              "bench": "f299_partial (26 hand circles; recall/IoU only)" if bench else None,
              "recipe": {"epochs": a.epochs, "nimg_per_epoch": a.nimg, "lr": LR, "train_blocks": a.train_blocks,
                         "trainable_params": trainable,
                         "weight_decay": WEIGHT_DECAY, "bsize": 256, "eval": EVAL_KW}}
    with open(os.path.join(a.out, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    say(stage="done", result=os.path.join(a.out, "result.json"), adopt=adopt)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        say(stage="error", error="%s: %s" % (type(e).__name__, e))
        sys.exit(1)
