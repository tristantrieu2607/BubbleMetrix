"""engines/finetune_pulp.py - fine-tune the pulp Mask R-CNN on a teaching set.

Started by the GUI as a separate process (same Python as the GUI). Nothing to set:
the recipe is fixed so a scientist only presses one button.

  python engines/finetune_pulp.py --set <teaching set folder> --base <weights .pt> --out <model folder>

Recipe
  * start from the model in use (not from COCO), backbone layers 1-2 frozen as in train.py
  * every epoch: ~9 random 512 windows per new training frame (flips, gamma,
    scale jitter) + an equal number of windows replayed from the original training
    clips (data/split_clips.json train + val), so the old footage is not forgotten
  * 6 epochs, AdamW lr 2e-4 -> 1e-6 cosine; keep the epoch with the lowest loss on
    the held-back teaching frames
Then the check: old and new model are both run on (a) the held-back teaching frames
and (b) the frozen benchmark (the 3 reserved test clips, full frames), scored the
way the GUI measures, and engines.teach.decide() says adopt or keep the old one.

Progress goes to stdout as lines starting with "@@ " (JSON); everything else is log.
result.json in --out holds the scores and the decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PULP_DIR = os.path.join(ROOT, "BubMask")
for p in (HERE, PULP_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from bubble_rcnn.config import BubbleConfig
from bubble_rcnn.dataset.real_dataset import RealBubbleDataset, TiledDataset, collate_fn
from bubble_rcnn.models.mask_rcnn_v2 import load_bubble_maskrcnn

import teach
from engine_metrology import summarize_bubbles

EPOCHS = 6
LR = 2e-4
NEW_REPEAT = 1
CONF = 0.40
MIN_D, MAX_D = 12.0, 250.0
FOV = (524.0, 508.0, 601.0)
_STEM = re.compile(r"^(.*)_f(\d+)(_full)?$")


def say(**kw):
    sys.stdout.write("@@ " + json.dumps(kw) + "\n")
    sys.stdout.flush()


def clip_of(stem: str) -> str:
    m = _STEM.match(stem)
    return m.group(1) if m else stem


def label_index(labels_dir):
    out = {}
    for r, _, fs in os.walk(labels_dir):
        for f in fs:
            if f.endswith("_labels.json"):
                out[f[:-12]] = os.path.join(r, f)
    return out


def image_index(crops_dir):
    out = {}
    for r, _, fs in os.walk(crops_dir):
        for f in fs:
            if f.endswith(".png") and not f.endswith(("_view.png", "_mask.png", "_ignore.png")):
                out[f[:-4]] = os.path.join(r, f)
    return out


# ---------------------------------------------------------------------------
# inference + scoring, the way the GUI measures
# ---------------------------------------------------------------------------
def accepted_bubbles(model, img_bgr, device):
    """Same steps as engine_pulp.PulpDetector.detect_frame + the GUI's pulp gates."""
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.as_tensor(rgb.transpose(2, 0, 1), dtype=torch.float32) / 255.0
    with torch.no_grad():
        p = model([t.to(device)])[0]
    recs = []
    for s, m in zip(p["scores"].cpu().numpy(), p["masks"].squeeze(1).cpu().numpy()):
        if s < CONF:
            continue
        bm = m >= 0.5
        a = float(bm.sum())
        if a <= 0:
            continue
        ys, xs = np.nonzero(bm)
        d = 2.0 * math.sqrt(a / math.pi)
        recs.append({"cx": float(xs.mean()), "cy": float(ys.mean()), "r": d / 2, "diameter_px": d,
                     "touches_image_edge": bool(bm[0].any() or bm[-1].any() or bm[:, 0].any() or bm[:, -1].any())})
    s = summarize_bubbles(recs, um_per_px=None, min_diam_px=MIN_D, max_diam_px=MAX_D,
                          apply_fov_gate=True, exclude_edge=True)
    return s["accepted_bubbles"]


def score_set(model, device, pairs, tag, on_each=None):
    """pairs: [(image path, label bubbles)]"""
    per = []
    for i, (img_p, bubbles) in enumerate(pairs, 1):
        img = cv2.imread(img_p)
        if img is None:
            continue
        h, w = img.shape[:2]
        fov = FOV if (w, h) == (1024, 1024) else None   # probe geometry only
        acc = accepted_bubbles(model, img, device)
        per.append(teach.score_image(acc, bubbles, MIN_D, w, h, fov))
        if on_each:
            on_each(i, len(pairs), tag)
    return teach.pool_scores(per)


def benchmark_pairs():
    """Full frames of the reserved test clips: never trained on by the base model
    or by any fine-tune (they are excluded from replay below)."""
    with open(os.path.join(PULP_DIR, "data", "split_clips.json"), encoding="utf-8") as fh:
        split = json.load(fh)
    test = set(split.get("test_clips", []))
    labs = label_index(os.path.join(PULP_DIR, "data", "labels", "pulp"))
    imgs = image_index(os.path.join(PULP_DIR, "data", "crops", "pulp"))
    pairs = []
    for stem in sorted(labs):
        if not stem.endswith("_full") or clip_of(stem) not in test or stem not in imgs:
            continue
        with open(labs[stem], encoding="utf-8") as fh:
            pairs.append((imgs[stem], json.load(fh).get("bubbles", [])))
    return pairs, sorted(test)


def replay_stems():
    with open(os.path.join(PULP_DIR, "data", "split_clips.json"), encoding="utf-8") as fh:
        split = json.load(fh)
    keep = set(split.get("train_clips", [])) | set(split.get("val_clips", []))
    labs = label_index(os.path.join(PULP_DIR, "data", "labels", "pulp"))
    return {s for s in labs if clip_of(s) in keep}


def bench_cache_key(weights, pairs):
    h = hashlib.sha1()
    h.update(os.path.abspath(weights).encode())
    h.update(str(os.path.getmtime(weights)).encode())
    for p, b in pairs:
        h.update(p.encode()); h.update(str(len(b)).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-steps", type=int, default=0, help="testing only: stop each epoch early")
    ap.add_argument("--no-bench", action="store_true", help="testing only")
    a = ap.parse_args()

    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(14, os.cpu_count() or 4))
    os.makedirs(a.out, exist_ok=True)

    ts = teach.TeachingSet(a.set)
    train_items, check_items = ts.split()
    say(stage="setup", train=len(train_items), check=len(check_items), device=device)

    cfg = BubbleConfig()
    new_ds = RealBubbleDataset(ts.folder, ts.folder, is_train=True,
                               restrict_to={it["stem"] for it in train_items},
                               scale_range=cfg.scale_jitter_range, scale_prob=cfg.scale_jitter_prob)
    if len(new_ds) == 0:
        raise SystemExit("No checked training frames found in the teaching set.")
    new_tiles = TiledDataset(new_ds, tile=512, random_tiles=True)
    rep_ds = RealBubbleDataset(os.path.join(PULP_DIR, "data", "crops", "pulp"),
                               os.path.join(PULP_DIR, "data", "labels", "pulp"), is_train=True,
                               restrict_to=replay_stems(), verbose=False,
                               scale_range=cfg.scale_jitter_range, scale_prob=cfg.scale_jitter_prob)
    rep_tiles = TiledDataset(rep_ds, tile=512, random_tiles=True) if len(rep_ds) else None
    chk_ds = RealBubbleDataset(ts.folder, ts.folder, is_train=False,
                               restrict_to={it["stem"] for it in check_items}, verbose=False)
    chk_loader = DataLoader(TiledDataset(chk_ds, tile=512), batch_size=2, shuffle=False,
                            num_workers=0, collate_fn=collate_fn) if len(chk_ds) else None

    model, base_ck = load_bubble_maskrcnn(a.base, config=cfg, device=device)
    scheme = getattr(model, "anchor_scheme", base_ck.get("anchor_scheme", "v1") if isinstance(base_ck, dict) else "v1")
    for name, prm in model.backbone.body.named_parameters():
        if any(l in name for l in ("conv1", "bn1", "layer1", "layer2")):
            prm.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-6)

    n_new = len(new_tiles) * NEW_REPEAT
    steps_per_epoch = math.ceil((n_new + min(n_new, len(rep_tiles) if rep_tiles else 0)) / 2)
    if a.max_steps:
        steps_per_epoch = min(steps_per_epoch, a.max_steps)
    total_steps = steps_per_epoch * a.epochs
    say(stage="train", epoch=0, of=a.epochs, step=0, steps=total_steps, eta_s=None,
        new_windows=n_new, replay_windows=min(n_new, len(rep_tiles) if rep_tiles else 0))

    best, best_ep, hist = float("inf"), 0, []
    t0, done_steps = time.time(), 0
    cand = os.path.join(a.out, "best.pt")
    for ep in range(1, a.epochs + 1):
        parts = [new_tiles] * NEW_REPEAT
        if rep_tiles is not None:
            k = min(n_new, len(rep_tiles))
            parts.append(Subset(rep_tiles, random.sample(range(len(rep_tiles)), k)))
        loader = DataLoader(ConcatDataset(parts), batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
        model.train()
        tot, nb = 0.0, 0
        for step, (imgs, tgts) in enumerate(loader, 1):
            if a.max_steps and step > a.max_steps:
                break
            imgs = [i.to(device) for i in imgs]
            tgts = [{k2: v.to(device) for k2, v in t.items()} for t in tgts]
            loss = sum(l for l in model(imgs, tgts).values())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            tot += float(loss.item()); nb += 1; done_steps += 1
            el = time.time() - t0
            say(stage="train", epoch=ep, of=a.epochs, step=done_steps, steps=total_steps,
                eta_s=round(el / done_steps * (total_steps - done_steps)), loss=round(float(loss.item()), 4))
        sched.step()

        val = None
        if chk_loader is not None:
            vt, vn = 0.0, 0
            with torch.no_grad():
                for imgs, tgts in chk_loader:
                    imgs = [i.to(device) for i in imgs]
                    tgts = [{k2: v.to(device) for k2, v in t.items()} for t in tgts]
                    vt += float(sum(l for l in model(imgs, tgts).values()).item()); vn += 1
            val = vt / max(1, vn)
        score = val if val is not None else tot / max(1, nb)
        hist.append({"epoch": ep, "loss": tot / max(1, nb), "check_loss": val})
        if score < best:
            best, best_ep = score, ep
            torch.save({"epoch": ep, "model_state_dict": model.state_dict(), "loss": score,
                        "selected_on": "teaching check frames" if val is not None else "train",
                        "anchor_scheme": scheme, "fine_tuned_from": os.path.abspath(a.base),
                        "teaching_set": os.path.abspath(ts.folder)}, cand)
        say(stage="epoch", epoch=ep, of=a.epochs, loss=round(tot / max(1, nb), 4),
            check_loss=None if val is None else round(val, 4))

    train_min = (time.time() - t0) / 60

    # ---------------- the check: old vs new ----------------
    model.eval()
    new_model, _ = load_bubble_maskrcnn(cand, config=cfg, device=device); new_model.eval()
    old_model, _ = load_bubble_maskrcnn(a.base, config=cfg, device=device); old_model.eval()
    chk_pairs = [(ts.image_path(it), ts.read_labels(it)) for it in check_items]
    bench, test_clips = ([], []) if a.no_bench else benchmark_pairs()
    total = 2 * len(chk_pairs) + len(bench) * 2
    cnt = [0]

    def tick(i, n, tag):
        cnt[0] += 1
        say(stage="check", done=cnt[0], of=max(1, total), what=tag)

    old_chk = score_set(old_model, device, chk_pairs, "old model, your frames", tick)
    new_chk = score_set(new_model, device, chk_pairs, "new model, your frames", tick)

    old_b = new_b = None
    if bench:
        cache_p = os.path.join(teach.MODELS_ROOT, "bench_cache.json")
        cache = {}
        if os.path.exists(cache_p):
            try:
                cache = json.load(open(cache_p, encoding="utf-8"))
            except Exception:
                cache = {}
        key = "pulp|" + bench_cache_key(a.base, bench)
        if key in cache:
            old_b = cache[key]; cnt[0] += len(bench)
        else:
            old_b = score_set(old_model, device, bench, "old model, benchmark", tick)
            cache[key] = old_b
            os.makedirs(teach.MODELS_ROOT, exist_ok=True)
            json.dump(cache, open(cache_p, "w", encoding="utf-8"), indent=1)
        new_b = score_set(new_model, device, bench, "new model, benchmark", tick)

    adopt, reasons = teach.decide(old_chk, new_chk, old_b, new_b)
    result = {"regime": "pulp", "adopt": adopt, "reasons": reasons,
              "model_path": os.path.abspath(cand), "base": os.path.abspath(a.base),
              "teaching_set": os.path.abspath(ts.folder),
              "train_items": [it["stem"] for it in train_items],
              "check_items": [it["stem"] for it in check_items],
              "best_epoch": best_ep, "history": hist, "train_minutes": round(train_min, 1),
              "check_old": old_chk, "check_new": new_chk,
              "bench_old": old_b, "bench_new": new_b, "bench_clips": test_clips,
              "recipe": {"epochs": a.epochs, "lr": LR, "new_repeat": NEW_REPEAT, "replay": "train+val clips",
                         "frozen": "conv1, bn1, layer1, layer2"}}
    with open(os.path.join(a.out, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    say(stage="done", result=os.path.join(a.out, "result.json"), adopt=adopt)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        say(stage="error", error="%s: %s" % (type(e).__name__, e))
        sys.exit(1)
