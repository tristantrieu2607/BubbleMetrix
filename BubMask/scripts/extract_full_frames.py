"""Export full 1024x1024 rotor_raw frames for labelling, replacing 512 crops.

Why full frames (decided 2026-09-27):
* 512 crops cut large bubbles at the crop border -- the 100+ px band is the gap.
* The original 30 crops all sit at origin (256, 256); the outer field and the FOV
  edge have never been labelled.
* measure_video runs on full frames, so training on centre crops is a train /
  inference mismatch. The FOV-circle flag (touches_fov) only works on full frames.
Training is unaffected: TiledDataset already cuts images into 512 tiles.

What it writes, into data/crops/pulp/rotor_raw/ (so train/eval/split see it):
  <clip>_f<NNN>_full.png        raw frame
  <clip>_f<NNN>_full_view.png   CLAHE view for the stamp tool's V toggle
  frame_manifest.json           provenance: clip, frame, side, why chosen

Frames chosen:
* train/val -- the parent frames of the 26 uniform-random crops extracted
  2026-09-07 (crop_manifest_new.json). Those crops were never labelled and are
  superseded by their frames.
* test -- TEST_PER_CLIP new frames per reserved clip, >= FRAME_GAP from every frame
  already cropped on that clip (overlays have been viewed on those).
  LABEL THESE BEFORE RUNNING ANY MODEL ON THEM.

Re-running is safe: existing frames are skipped, the seed is fixed.
"""
import json, os, random
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(os.path.dirname(ROOT), "samples", "pulp")
OUT = os.path.join(ROOT, "data", "crops", "pulp", "rotor_raw")
PENDING_MANIFEST = os.path.join(OUT, "crop_manifest_new.json")
SPLIT = os.path.join(ROOT, "data", "split_clips.json")
FRAME_GAP = 5            # rotor decorrelates by frame 5
TEST_PER_CLIP = 3        # 9 frames x ~20 bubbles ~= the 150-200 test bubbles needed
SEED = 20260927


def read_frame(clip, idx):
    cap = cv2.VideoCapture(os.path.join(SAMPLES, clip + ".avi"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return (frame if ok else None), n


def frames_used(clip):
    """Every frame index already present for a clip, as crop or full frame."""
    used = set()
    for f in os.listdir(OUT):
        if f.startswith(clip + "_f"):
            tok = f[len(clip) + 2:].split("_")[0]
            if tok.isdigit():
                used.add(int(tok))
    return used


def write(clip, idx, frame, side, why, clahe, manifest):
    stem = f"{clip}_f{idx:03d}_full"
    raw = os.path.join(OUT, stem + ".png")
    if os.path.exists(raw):
        print(f"   skip (exists) {stem}")
        return
    cv2.imwrite(raw, frame)
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    cv2.imwrite(os.path.join(OUT, stem + "_view.png"),
                cv2.cvtColor(clahe.apply(g), cv2.COLOR_GRAY2BGR))
    manifest.append({"image_id": stem, "clip": clip, "video_file": clip + ".avi",
                     "frame_idx": idx, "origin_xy": [0, 0], "size": list(frame.shape[:2]),
                     "side": side, "chosen_because": why, "seed": SEED})
    print(f"   + {stem}  [{side}]")


def main():
    rng = random.Random(SEED)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    split = json.load(open(SPLIT, encoding="utf-8"))
    manifest = []

    print("train/val: parent frames of the 26 pending crops")
    for m in json.load(open(PENDING_MANIFEST, encoding="utf-8")):
        frame, _ = read_frame(m["clip"], m["frame_idx"])
        if frame is None:
            print(f"!! unreadable {m['clip']} f{m['frame_idx']}"); continue
        side = "val" if m["clip"] in split["val_clips"] else "train"
        write(m["clip"], m["frame_idx"], frame, side,
              f"parent frame of unlabelled crop {m['crop_id']}", clahe, manifest)

    print(f"test: {TEST_PER_CLIP} new frames per reserved clip -- label blind")
    for clip in split["test_clips"]:
        used = frames_used(clip)
        _, n = read_frame(clip, 0)
        cands = [f for f in range(n) if all(abs(f - u) >= FRAME_GAP for u in used)]
        rng.shuffle(cands)
        picked = []
        for f in cands:
            if len(picked) >= TEST_PER_CLIP:
                break
            if any(abs(f - p) < FRAME_GAP for p in picked):
                continue
            frame, _ = read_frame(clip, f)
            if frame is None:
                continue
            write(clip, f, frame, "test",
                  f">= {FRAME_GAP} frames from every previously cropped frame", clahe, manifest)
            picked.append(f)

    mpath = os.path.join(OUT, "frame_manifest.json")
    old = json.load(open(mpath, encoding="utf-8")) if os.path.exists(mpath) else []
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(old + manifest, fh, indent=2)
    print(f"\n{len(manifest)} frames written; provenance in {os.path.relpath(mpath, ROOT)}")


if __name__ == "__main__":
    main()
