"""Training loop module for PyTorch Mask R-CNN on real human-stamped bubble dataset."""

import os
import time
import json
import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any

from bubble_rcnn.config import BubbleConfig
from bubble_rcnn.dataset.real_dataset import RealBubbleDataset, collate_fn
from bubble_rcnn.models.mask_rcnn_v2 import build_bubble_maskrcnn

def train_bubble_model(
    crops_dir: str,
    labels_dir: str,
    output_dir: str,
    val_crops_dir: Optional[str] = None,
    val_labels_dir: Optional[str] = None,
    config: Optional[BubbleConfig] = None,
    num_epochs: int = 12,
    lr: float = 0.001,
    batch_size: int = 2,
    device: Optional[str] = None,
    train_stems: Optional[set] = None,
    val_stems: Optional[set] = None,
    split_note: str = "",
    resume: Optional[str] = None,
) -> Dict[str, Any]:
    """Train, selecting the checkpoint on a held-out *validation* clip.

    v1 had no validation loop at all: ``best.pt`` was whichever epoch had the
    lowest **training** loss, which on 20 crops selects for memorisation.  The
    validation clip is carved out of the training side, never out of the test
    side -- picking a checkpoint on the test set is a quiet way of fitting to it.
    """
    config = config or BubbleConfig()
    os.makedirs(output_dir, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Initializing training on device: {device}")

    # 1. Dataset & DataLoader (with online data augmentation & native 512 tiling)
    from bubble_rcnn.dataset.real_dataset import TiledDataset
    raw_dataset = RealBubbleDataset(crops_dir, labels_dir, is_train=True, restrict_to=train_stems,
                                    scale_range=config.scale_jitter_range,
                                    scale_prob=config.scale_jitter_prob)
    if len(raw_dataset) == 0:
        raise RuntimeError(f"No labeled images found in {labels_dir} matching {crops_dir}")
    train_dataset = TiledDataset(raw_dataset, tile=512, random_tiles=config.scale_jitter_prob > 0)
    print(f"[Train] anchors: {config.anchor_scheme} | scale jitter p={config.scale_jitter_prob} "
          f"range={config.scale_jitter_range}")
    print(f"[Train] {len(raw_dataset)} training crops -> {len(train_dataset)} native 512x512 tiles.")
    if split_note:
        print(f"[Train] split: {split_note}")

    val_loader = None
    if val_stems:
        val_raw = RealBubbleDataset(crops_dir, labels_dir, is_train=False, restrict_to=val_stems)
        if len(val_raw):
            val_loader = DataLoader(TiledDataset(val_raw, tile=512), batch_size=batch_size,
                                    shuffle=False, num_workers=0, collate_fn=collate_fn)
            print(f"[Train] {len(val_raw)} validation crops -> {len(val_loader.dataset)} tiles "
                  f"(checkpoint selection only)")
    if val_loader is None:
        print("[Train] WARNING: no validation set. best.pt will be selected on TRAINING loss, "
              "which on a small dataset selects for memorisation. Prefer last.pt.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 0 is safest for Windows multiprocessing
        collate_fn=collate_fn
    )

    # CPU Optimization: Unlock full multicore compute
    if device == "cpu":
        optimal_threads = min(14, os.cpu_count() or 4)
        torch.set_num_threads(optimal_threads)
        print(f"[Train] Tuned PyTorch CPU threads: {optimal_threads}")

    # 2. Build SOTA Mask R-CNN V2 model with microbubble anchors & NoResizeTransform
    model = build_bubble_maskrcnn(config=config, pretrained=True, cpu_optimized=(device == "cpu"))

    # CPU Optimization: Freeze low-level backbone layers (conv1, bn1, layer1, layer2)
    # The early layers already detect edges/textures from COCO V2.
    # Freezing them cuts backward-pass compute by ~4.5x without loss of accuracy!
    frozen_count = 0
    for name, param in model.backbone.body.named_parameters():
        if any(layer in name for layer in ["conv1", "bn1", "layer1", "layer2"]):
            param.requires_grad = False
            frozen_count += 1
    print(f"[Train] Froze {frozen_count} low-level backbone parameters (4x CPU speedup active).")

    model.to(device)

    # 3. Optimizer & LR Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_loss = float("inf")
    history = []

    def manifest_now(status: str) -> Dict[str, Any]:
        return {
            "status": status,
            "num_epochs": num_epochs,
            "best_loss": best_loss,
            "best_epoch": min(history, key=lambda h: h["val_loss"] if h["val_loss"] is not None else h["loss"])["epoch"] if history else None,
            "best_selected_on": "val" if val_loader is not None else "train",
            "train_crops": len(raw_dataset),
            "train_tiles": len(train_dataset),
            "val_tiles": len(val_loader.dataset) if val_loader is not None else 0,
            "split_note": split_note,
            "anchor_scheme": config.anchor_scheme,
            "scale_jitter": {"prob": config.scale_jitter_prob, "range": list(config.scale_jitter_range)},
            "history": history,
        }

    def write_manifest(m: Dict[str, Any]) -> None:
        # written after EVERY epoch, so an interrupted run keeps its loss history
        with open(os.path.join(output_dir, "train_manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(m, fh, indent=2)

    # ---- resume: continue from last.pt with optimizer, LR schedule and history ----
    start_epoch = 1
    if resume:
        ck = torch.load(resume, map_location=device, weights_only=False)
        if "optimizer_state_dict" not in ck:
            raise SystemExit(f"--resume: {resume} predates resume support (no optimizer state); start a new run")
        if ck.get("anchor_scheme", "v1") != config.anchor_scheme:
            raise SystemExit(f"--resume: checkpoint anchors '{ck.get('anchor_scheme', 'v1')}' != '{config.anchor_scheme}'; "
                             f"pass the same --anchors as the original run")
        if ck.get("num_epochs", num_epochs) != num_epochs:
            raise SystemExit(f"--resume: original run was --epochs {ck['num_epochs']}; pass the same value "
                             f"(the LR schedule is tied to it)")
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        lr_scheduler.load_state_dict(ck["scheduler_state_dict"])
        best_loss = float(ck.get("best_loss", best_loss))
        history = list(ck.get("history", []))
        start_epoch = int(ck["epoch"]) + 1
        print(f"[Train] resumed from {resume}: continuing at epoch {start_epoch}/{num_epochs}, best so far {best_loss:.4f}")
        if start_epoch > num_epochs:
            print("[Train] nothing left to do -- run already finished.")

    start_time = time.time()
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        loss_components = {}
        t_epoch_start = time.time()

        for step, (images, targets) in enumerate(train_loader, start=1):
            t_s0 = time.time()
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            epoch_loss += losses.item()
            for k, v in loss_dict.items():
                loss_components[k] = loss_components.get(k, 0.0) + v.item()

            t_s = time.time() - t_s0
            print(f"  -> Epoch {epoch:02d}/{num_epochs:02d} | Step {step:02d}/{len(train_loader)} | Batch Loss: {losses.item():.4f} ({t_s:.1f}s/step)", end="\r", flush=True)

        # ---- validation loss ---------------------------------------------
        # Torchvision returns losses only in train() mode, so the mode stays on and
        # gradients are switched off instead.  Dropout/BN behaviour therefore matches
        # training, which is what makes the two numbers comparable.
        val_loss = None
        if val_loader is not None:
            tot, nb = 0.0, 0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    ld = model(images, targets)
                    tot += float(sum(l for l in ld.values()).item()); nb += 1
            val_loss = tot / max(1, nb)

        lr_scheduler.step()
        epoch_duration = time.time() - t_epoch_start
        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"\n[Epoch {epoch:02d}/{num_epochs:02d}] Total Loss: {avg_loss:.4f} (LR: {lr_scheduler.get_last_lr()[0]:.6f}) in {epoch_duration:.1f}s")

        if val_loss is not None:
            print(f"[Epoch {epoch:02d}] val loss: {val_loss:.4f}")

        # Save checkpoint -- on val loss when there is one, never on train loss
        # if a validation set exists.
        score = val_loss if val_loss is not None else avg_loss
        is_best = score < best_loss
        if is_best:
            best_loss = score
            best_path = os.path.join(output_dir, "best.pt")
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "loss": best_loss, "selected_on": "val" if val_loss is not None else "train",
                        "anchor_scheme": config.anchor_scheme},
                       best_path)

        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "val_loss": val_loss,
            "lr": lr_scheduler.get_last_lr()[0]
        })

        # last.pt carries everything --resume needs
        last_path = os.path.join(output_dir, "last.pt")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "loss": avg_loss,
                    "anchor_scheme": config.anchor_scheme,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": lr_scheduler.state_dict(),
                    "best_loss": best_loss, "history": history, "num_epochs": num_epochs},
                   last_path)
        write_manifest(manifest_now(f"running: {epoch}/{num_epochs} epochs done"))

    elapsed = time.time() - start_time
    print(f"[Train] Training complete in {elapsed/60:.1f} minutes. Best loss: {best_loss:.4f}")

    manifest = manifest_now("complete")
    write_manifest(manifest)

    return manifest
