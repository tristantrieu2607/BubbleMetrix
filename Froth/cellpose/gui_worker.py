"""
gui_worker.py - Cellpose as a long-lived helper process for the BubbleMetrix GUI.

The GUI runs in BubMask\\.venv (Mask R-CNN, OpenCV with windows). Cellpose
lives in Froth\\.venv and brings opencv-python-headless with it; installing both in
one environment breaks `import cv2` for whichever was installed first. So the GUI
starts this script with Froth\\.venv's Python once, the model loads once, and every
frame after that is one request.

Protocol (one JSON object per line):
  stdin   {"image": "<png path>", "out": "<npy path>"}      or   {"quit": true}
  stdout  @@ {"ready": true, "cellpose_version": ..., "model": ..., "device": ...}
          @@ {"ok": true, "n": <instances>, "seconds": <float>, "out": "<npy path>"}
          @@ {"ok": false, "error": "<message>"}
Only lines starting with "@@ " are protocol; anything else Cellpose prints is ignored
by the GUI. Cellpose's own logging goes to stderr, which the GUI sends to a log file.

Model: argv[1] if given (a fine-tuned model file made by finetune_cellpose.py from
the GUI's "Teach the model" tab), otherwise Cellpose's default pretrained model.

Settings are Cellpose's defaults, the same as the validated zero-shot runs
(cellpose/run_cellpose.py with no options): default pretrained model, no diameter,
flow_threshold 0.4, cellprob_threshold 0.0, min_size 15. Nothing here is tunable on
purpose - the GUI is for measuring, not for tuning.
"""
import json
import os
import sys
import time
import traceback


def reply(obj):
    sys.stdout.write("@@ " + json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    try:
        import numpy as np
        import torch
        import cellpose
        from cellpose import models
        from PIL import Image
        gpu = bool(torch.cuda.is_available())
        kw = {"gpu": gpu}
        if len(sys.argv) > 1 and sys.argv[1]:
            if not os.path.exists(sys.argv[1]):
                raise FileNotFoundError("fine-tuned model not found: %s" % sys.argv[1])
            kw["pretrained_model"] = sys.argv[1]
        try:
            model = models.CellposeModel(use_bfloat16=gpu, **kw)   # fp32 on CPU
        except TypeError:
            model = models.CellposeModel(**kw)
        version = str(getattr(cellpose, "version", getattr(cellpose, "__version__", "?")))
        name = os.path.basename(str(getattr(model, "pretrained_model", "default")))
        reply({"ready": True, "cellpose_version": version, "model": name,
               "device": "cuda" if gpu else "cpu"})
    except Exception as e:
        reply({"ready": False, "error": "%s: %s" % (type(e).__name__, e),
               "trace": traceback.format_exc()})
        return 1

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("quit"):
                break
            rgb = np.array(Image.open(req["image"]).convert("RGB"))
            t0 = time.time()
            out = model.eval(rgb, flow_threshold=0.4, cellprob_threshold=0.0, min_size=15)
            labels = np.asarray(out[0]).astype(np.int32)
            np.save(req["out"], labels)
            reply({"ok": True, "n": int(labels.max()), "seconds": round(time.time() - t0, 1),
                   "out": req["out"]})
        except Exception as e:
            reply({"ok": False, "error": "%s: %s" % (type(e).__name__, e),
                   "trace": traceback.format_exc()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
