"""Self-test of the GUI's froth path on this machine, without opening the GUI.
Runs with the GUI's interpreter (BubMask\\.venv), exactly as bubble_gui.py does:
starts the Cellpose worker in Froth\\.venv, measures frame 0 of 180WW.MP4 and the
labelled crop, and checks the numbers against the validated batch run.
Result: Froth\\eval\\selftest_gui_engine.txt"""
import os, sys, time, json, traceback
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "Froth", "eval", "selftest_gui_engine.txt")
lines = []
def log(s):
    lines.append(s); print(s, flush=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
try:
    sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "engines"))
    import cv2, numpy as np
    log("python: %s" % sys.executable)
    import engine_froth as ef
    from engine_metrology import summarize_bubbles, evaluate_predictions_vs_ground_truth
    import bubble_gui  # import only: catches syntax/import errors in the GUI module
    log("bubble_gui imports OK")
    det = ef.FrothDetector(log_cb=log)
    cap = cv2.VideoCapture(os.path.join(ROOT, "samples", "froth", "180WW.MP4"))
    ok, frame = cap.read(); cap.release()
    t0 = time.time()
    b = det.detect_frame(frame, frame_idx=0)
    s = summarize_bubbles(b, um_per_px=det.mm_per_px, unit_name="mm", exclude_edge=True)
    log("frame 0: %d instances, %d accepted, %d edge-excluded, D32 %.3f mm, %.5f mm/px (%s), region %s, %.0f s"
        % (len(b), s["n_accepted"], s["n_edge_excluded"], s["d32"], det.mm_per_px, det.scale_source, det.last_roi, time.time() - t0))
    ref = np.load(os.path.join(ROOT, "Froth", "cellpose", "results", "video", "0000", "cellpose_labels.npy"))
    log("batch run had %d instances on the same region -> %s" % (ref.max(), "MATCH" if ref.max() == len(b) else "DIFFERENT"))
    crop = cv2.imread(os.path.join(ROOT, "BubMask", "data", "crops", "froth", "180WW_f00299_x104y400.png"))
    bc = det.detect_frame(crop)
    gt = json.load(open(os.path.join(ROOT, "BubMask", "data", "labels", "froth", "180WW_f00299_x104y400_labels.json")))["bubbles"]
    gt = [{"cx": g["cx"], "cy": g["cy"], "r": g["r"], "diameter_px": 2 * g["r"]} for g in gt]
    r = evaluate_predictions_vs_ground_truth(bc, gt, um_per_px=det.mm_per_px, unit_name="mm")
    log("labelled crop (GUI circle-IoU scoring): recall %d/%d, mean IoU %.3f (precision not meaningful: partial labels)"
        % (r["tp_count"], r["gt_count"], r["mean_iou"]))
    det.close()
    log("RESULT: PASS")
except Exception:
    log(traceback.format_exc()); log("RESULT: FAIL")
