"""Central Command-Line Interface for BubMask."""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        prog="bubble_rcnn",
        description="BubMask: SOTA Mask R-CNN Bubble Detection and Metrology System"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # Command: extract-crops
    p_extract = subparsers.add_parser("extract-crops", help="Extract stratified 512x512 crops from videos")
    p_extract.add_argument("--videos", required=True, help="Directory containing .avi/.mp4 video clips")
    p_extract.add_argument("--out", default="data/crops", help="Directory to save extracted crops")
    p_extract.add_argument("--crops-per-clip", type=int, default=3, help="Number of crops per video clip")

    # Command: annotate
    p_annot = subparsers.add_parser("annotate", help="Launch interactive 1-click bubble stamping tool")
    p_annot.add_argument("--regime", choices=["froth", "pulp", "rotor_diluted", "rotor_raw", "venturi_air", "all"], default=None, help="Select regime to annotate")
    p_annot.add_argument("--crops", default=None, help="Directory containing image crops (auto-routed if --regime is set)")
    p_annot.add_argument("--labels", default=None, help="Directory to save annotations (auto-routed if --regime is set)")

    # Command: train
    p_train = subparsers.add_parser("train", help="Train Mask R-CNN on real human-stamped bubble dataset")
    p_train.add_argument("--regime", choices=["froth", "pulp", "rotor_diluted", "rotor_raw", "venturi_air"], default=None, help="Train regime")
    p_train.add_argument("--crops", default=None, help="Directory containing image crops")
    p_train.add_argument("--labels", default=None, help="Directory containing labels JSON")
    p_train.add_argument("--out", default=None, help="Directory to save checkpoints")
    p_train.add_argument("--epochs", type=int, default=12, help="Number of training epochs (val bottoms ~10-12)")
    p_train.add_argument("--anchors", choices=["round3", "v1"], default="round3", help="Anchor scheme (see models/mask_rcnn_v2.ANCHOR_SCHEMES)")
    p_train.add_argument("--no-scale-jitter", action="store_true", help="Disable training-time scale jitter (for ablation)")
    p_train.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    p_train.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    p_train.add_argument("--resume", type=str, default=None, help="Continue an interrupted run from its last.pt (same --epochs/--anchors/--fold)")
    p_train.add_argument("--split-file", default=None, help="Alternate split JSON (default: data/split_clips.json)")
    p_train.add_argument("--fold", type=int, default=None, help="Leave-one-clip-out fold 0-7 (overrides the fixed split)")
    p_train.add_argument("--no-val", action="store_true", help="With --fold: no validation clip, the val clip trains too; use last.pt")
    p_train.add_argument("--no-split", action="store_true", help="Ignore data/split_clips.json and train on every labelled crop (leaks the test clips)")

    # Command: eval
    p_eval = subparsers.add_parser("eval", help="Evaluate model against held-out real test crops")
    p_eval.add_argument("--weights", required=True, help="Path to trained model weights (.pt)")
    p_eval.add_argument("--regime", choices=["froth", "pulp", "rotor_diluted", "rotor_raw", "venturi_air"], default=None, help="Evaluation regime")
    p_eval.add_argument("--test-crops", default=None, help="Directory containing test crops")
    p_eval.add_argument("--test-labels", default=None, help="Directory containing test labels")
    p_eval.add_argument("--out", default=None, help="Directory to save evaluation reports")
    p_eval.add_argument("--iou", type=float, default=0.5, help="IoU matching threshold (default: 0.5)")
    p_eval.add_argument("--conf", type=float, default=0.40, help="Confidence score threshold (default: 0.40 -- best D32; 0.72 misses the large bubbles)")
    p_eval.add_argument("--split-file", default=None, help="Alternate split JSON (default: data/split_clips.json)")
    p_eval.add_argument("--keep-edge", action="store_true", help="Score bubbles cut by the image border too (default: edge rule on)")
    p_eval.add_argument("--include-crops", action="store_true", help="Also score old 512 crops (default: full frames only -- crop borders cut bubbles)")
    p_eval.add_argument("--fold", type=int, default=None, help="Leave-one-clip-out fold 0-7 (overrides the fixed split)")
    p_eval.add_argument("--split", choices=["test", "val", "train", "all"], default="test", help="Which side of data/split_clips.json to score on (default: test)")

    # Command: measure
    p_meas = subparsers.add_parser("measure", help="Measure bubbles in full video and export calibrated D32")
    p_meas.add_argument("--video", required=True, help="Path to input .avi or .mp4 video")
    p_meas.add_argument("--weights", required=True, help="Path to trained model weights (.pt)")
    p_meas.add_argument("--out", default="results/measured", help="Directory to save measurement artifacts")
    p_meas.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    p_meas.add_argument("--stride", type=int, default=None, help="Frame sample stride (optional)")
    p_meas.add_argument("--keep-edge", action="store_true", help="Keep bubbles cut by the image border (default: excluded -- their size is truncated)")
    p_meas.add_argument("--allow-provisional-um", action="store_true", help="Emit microns from the provisional calibration (labelled as provisional everywhere)")

    args = parser.parse_args()

    if args.command == "extract-crops":
        from bubble_rcnn.preprocessing.crop_sampler import extract_stratified_crops
        extract_stratified_crops(
            video_dir=args.videos,
            output_dir=args.out,
            crops_per_clip=args.crops_per_clip
        )

    elif args.command == "annotate":
        from bubble_rcnn.annotation.stamp_tool import StampAnnotationTool
        if args.regime == "rotor_diluted":
            crops_dir = args.crops or "data/crops/pulp/rotor_diluted"
            labels_dir = args.labels or "data/labels/pulp/rotor_diluted"
        elif args.regime == "rotor_raw":
            crops_dir = args.crops or "data/crops/pulp/rotor_raw"
            labels_dir = args.labels or "data/labels/pulp/rotor_raw"
        elif args.regime == "venturi_air":
            crops_dir = args.crops or "data/crops/pulp/venturi_air"
            labels_dir = args.labels or "data/labels/pulp/venturi_air"
        elif args.regime == "froth":
            crops_dir = args.crops or "data/crops/froth"
            labels_dir = args.labels or "data/labels/froth"
        elif args.regime == "pulp":
            crops_dir = args.crops or "data/crops/pulp"
            labels_dir = args.labels or "data/labels/pulp"
        else:
            crops_dir = args.crops or "data/crops"
            labels_dir = args.labels or "data/labels"

        tool = StampAnnotationTool(crops_dir=crops_dir, labels_dir=labels_dir)
        print(f"[Stamper] Launching Annotation GUI for: {crops_dir} -> {labels_dir}")
        print("[Stamper] Controls: Left-drag=Stamp | Right-click=Delete | 'D'/'X'=Delete | Ctrl+Z=Undo | Ctrl+Y=Redo | 'M'=Mode | 'V'=Contrast | 'S'=Save | Space=Next")
        tool.run()

    elif args.command == "train":
        if args.regime == "rotor_diluted":
            # rotor_diluted is one clip: no clip-level split exists, so it cannot be
            # trained or scored on its own. It joins the rotor clips as training data.
            raise SystemExit("rotor_diluted is training-only data. Use:\n"
                             "  train --regime pulp --split-file data/split_clips_with_dilute.json --out runs/rr_dil")

        from bubble_rcnn.train import train_bubble_model
        if args.regime == "rotor_raw":
            crops_dir = args.crops or "data/crops/pulp/rotor_raw"
            labels_dir = args.labels or "data/labels/pulp/rotor_raw"
            out_dir = args.out or "runs/rotor_raw"
        elif args.regime == "venturi_air":
            crops_dir = args.crops or "data/crops/pulp/venturi_air"
            labels_dir = args.labels or "data/labels/pulp/venturi_air"
            out_dir = args.out or "runs/venturi_air"
        elif args.regime == "froth":
            crops_dir = args.crops or "data/crops/froth"
            labels_dir = args.labels or "data/labels/froth"
            out_dir = args.out or "runs/froth"
        elif args.regime == "pulp":
            crops_dir = args.crops or "data/crops/pulp"
            labels_dir = args.labels or "data/labels/pulp"
            out_dir = args.out or "runs/pulp"
        else:
            crops_dir = args.crops or "data/crops"
            labels_dir = args.labels or "data/labels"
            out_dir = args.out or "runs/exp1"

        train_stems = val_stems = None
        split_note = "no split file applied -- every labelled crop is in training"
        if not args.no_split:
            try:
                from bubble_rcnn import split as splitmod
                sp = splitmod.resolve(args.split_file, args.fold, with_val=not args.no_val)
                tr = set(splitmod.filter_dir(crops_dir, "train", sp))
                va = set(splitmod.filter_dir(crops_dir, "val", sp))
                te = splitmod.filter_dir(crops_dir, "test", sp)
                un = splitmod.filter_dir(crops_dir, "unassigned", sp)
                if tr:
                    train_stems, val_stems = tr, va
                    split_note = (f"clip-level split: {len(tr)} train / {len(va)} val / "
                                  f"{len(te)} test crops held out; test clips = {', '.join(sp['test'])}")
                    if un:
                        print(f"[Train] {len(un)} crop(s) are in no split group and are EXCLUDED: {un[:5]}")
                else:
                    print("[Train] split file matches no crops here; training on everything.")
            except FileNotFoundError:
                print("[Train] no data/split_clips.json; training on everything.")

        from bubble_rcnn.config import BubbleConfig
        cfg = BubbleConfig()
        cfg.anchor_scheme = args.anchors
        if args.no_scale_jitter:
            cfg.scale_jitter_prob = 0.0
        if args.resume and not args.out:
            out_dir = os.path.dirname(os.path.abspath(args.resume))   # keep writing into the same run
        # never overwrite a trained model by default -- runs/rotor_raw is the baseline
        elif not args.out and os.path.exists(os.path.join(out_dir, "best.pt")):
            import time as _time
            out_dir = f"{out_dir}_{_time.strftime('%Y%m%d_%H%M')}"
            print(f"[Train] default run dir already holds a model; writing to {out_dir}")

        train_bubble_model(
            resume=args.resume,
            config=cfg,
            crops_dir=crops_dir,
            labels_dir=labels_dir,
            output_dir=out_dir,
            num_epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            train_stems=train_stems,
            val_stems=val_stems,
            split_note=split_note
        )

    elif args.command == "eval":
        from bubble_rcnn.eval import evaluate_bubble_model
        crops_dir = args.test_crops or "data/crops/pulp/rotor_raw"
        labels_dir = args.test_labels or "data/labels/pulp/rotor_raw"
        out_dir = args.out or f"runs/eval_{args.split}"
        restrict = None
        note = f"all crops under {crops_dir}"
        if args.split != "all":
            from bubble_rcnn import split as splitmod
            sp = splitmod.resolve(args.split_file, args.fold)
            restrict = set(splitmod.filter_dir(crops_dir, args.split, sp))
            if not args.include_crops:
                n_before = len(restrict)
                restrict = {st for st in restrict if "_crop" not in st}
                if n_before != len(restrict):
                    print(f"[Eval] scoring full frames only: {n_before - len(restrict)} 512-crop image(s) skipped "
                          f"(--include-crops to score them)")
            clips = sp.get(args.split, [])
            note = (f"{args.split} side of data/split_clips.json: {len(restrict)} crops "
                    f"from {len(clips)} held-out clip(s) -- {', '.join(clips)}")
            if not restrict:
                raise SystemExit(f"No crops on the '{args.split}' side under {crops_dir}.")
        evaluate_bubble_model(
            weights_path=args.weights,
            test_crops_dir=crops_dir,
            test_labels_dir=labels_dir,
            output_dir=out_dir,
            iou_threshold=args.iou,
            score_threshold=args.conf,
            split_note=note,
            restrict_to=restrict,
            edge_rule=not args.keep_edge,
        )

    elif args.command == "measure":
        from bubble_rcnn.measure_video import measure_video_clip
        measure_video_clip(
            video_path=args.video,
            weights_path=args.weights,
            output_dir=args.out,
            score_threshold=args.conf,
            sample_stride=args.stride,
            allow_provisional_um=args.allow_provisional_um,
            exclude_edge=not args.keep_edge,
        )

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
