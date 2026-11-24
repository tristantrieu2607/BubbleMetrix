"""d32_convergence.py - how many bubbles until D32 stops changing (froth, Cellpose).

Uses the saved Cellpose runs on 9 full-frame in-focus ROIs (0-180 s video runs and the
240-420 s patch-frame runs). Applies the GUI's rule: bubbles cut by the ROI edge are
dropped. Diameters in mm at 55 mm inner / 471 px.

Outputs (next to this script, in results/d32_convergence/):
  summary.txt          per-frame D32, pooled D32, n* for +/-5% and +/-2%, frame-effect test
  convergence.png      running D32 vs n (random orders) + bootstrap 95% band
"""
import csv, glob, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results", "d32_convergence")
MM_PER_PX = 55.0 / 471.0   # 55 mm = inner tube width
RNG = np.random.default_rng(20261105)
B = 2000

files = sorted(glob.glob(os.path.join(ROOT, "cellpose", "results", "video", "*", "cellpose_masks.csv")) +
               glob.glob(os.path.join(HERE, "results", "patch_runs", "*", "cellpose_masks.csv")))

def d32(d):
    d = np.asarray(d, float)
    return float((d ** 3).sum() / (d ** 2).sum())

frames = {}
for f in files:
    name = os.path.basename(os.path.dirname(f))
    name = "t0000s" if name == "0000" else name
    with open(f) as fh:
        rows = list(csv.DictReader(fh))
    d = [float(r["equiv_diam_px"]) * MM_PER_PX for r in rows if r["touches_roi_edge"] in ("0", "False", "false")]
    frames[name] = np.array(d)
names = sorted(frames)
pool = np.concatenate([frames[k] for k in names])
frame_id = np.concatenate([[i] * len(frames[k]) for i, k in enumerate(names)])
D = d32(pool)
N = len(pool)

lines = []
P = lambda s: (print(s), lines.append(s))
P("Froth D32 convergence (Cellpose zero-shot, edge-cut bubbles excluded, 55 mm inner / 471 px)")
P("")
P("frame    n     D32 (mm)")
for k in names:
    P("%-7s %4d   %.3f" % (k, len(frames[k]), d32(frames[k])))
P("pooled  %4d   %.3f   (%d frames)" % (N, D, len(names)))
P("")

# 1. Bubble-level: bootstrap D32 from n bubbles drawn from the pool
ns = np.unique(np.round(np.logspace(np.log10(25), np.log10(N), 40)).astype(int))
band = []
for n in ns:
    s = np.array([d32(pool[RNG.integers(0, N, n)]) for _ in range(B)])
    lo, hi = np.percentile(s, [2.5, 97.5])
    band.append((n, lo, hi, (hi - lo) / 2 / D))
band = np.array(band)

def n_star(tol):
    ok = band[:, 3] <= tol
    for i in range(len(ok)):
        if ok[i:].all():
            return int(band[i, 0])
    return None

P("1) Sampling only (bubbles treated as independent)")
P("   n     95% CI of D32 (mm)   half-width")
for n, lo, hi, hw in band[:: max(1, len(band) // 12)]:
    P("   %-5d %.3f - %.3f          +/-%.1f%%" % (n, lo, hi, 100 * hw))
for tol in (0.10, 0.05, 0.02):
    s = n_star(tol)
    P("   n* for +/-%d%%: %s bubbles" % (100 * tol, s if s else "> %d (not reached)" % N))
P("")

# 2. Is the frame-to-frame spread larger than sampling noise? (parametric bootstrap under H0:
#    every frame is a random draw from the same pooled population)
obs_sd = np.std([d32(frames[k]) for k in names], ddof=1)
null_sd = []
for _ in range(B):
    null_sd.append(np.std([d32(pool[RNG.integers(0, N, len(frames[k]))]) for k in names], ddof=1))
null_sd = np.array(null_sd)
p = (1 + (null_sd >= obs_sd).sum()) / (B + 1)
P("2) Frame effect: SD of per-frame D32 = %.3f mm (%.1f%% of D32)" % (obs_sd, 100 * obs_sd / D))
P("   Expected from sampling alone: %.3f mm (95%% of null runs below %.3f)" % (null_sd.mean(), np.percentile(null_sd, 95)))
P("   p = %.4f  -> %s" % (p, "frames genuinely differ: sample over time, not just more bubbles"
                           if p < 0.05 else "no evidence frames differ: bubble count is what matters"))
P("")

# 3. Frame-level (cluster) bootstrap: CI of D32 vs number of frames
P("3) Resampling whole frames (the honest CI when frames differ)")
P("   frames  ~bubbles  95% CI half-width")
for k in range(1, len(names) + 1):
    s = []
    for _ in range(B):
        pick = RNG.integers(0, len(names), k)
        s.append(d32(np.concatenate([frames[names[i]] for i in pick])))
    lo, hi = np.percentile(s, [2.5, 97.5])
    P("   %-6d  %-8d  +/-%.1f%%" % (k, int(round(k * N / len(names))), 100 * (hi - lo) / 2 / D))
P("   (with 9 frames this band is itself rough; it is the trend that matters)")

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "summary.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i in range(30):
        order = RNG.permutation(N)
        d = pool[order]
        run = np.cumsum(d ** 3) / np.cumsum(d ** 2)
        ax.plot(np.arange(1, N + 1), run, color="0.6", lw=0.6, alpha=0.6)
    ax.fill_between(band[:, 0], band[:, 1], band[:, 2], color="tab:blue", alpha=0.25, label="95% band (bootstrap)")
    ax.axhline(D, color="k", lw=1, label="pooled D32 = %.2f mm" % D)
    for tol, ls in ((0.05, "--"), (0.02, ":")):
        ax.axhspan(D * (1 - tol), D * (1 + tol), color="none", ec="tab:red", ls=ls, lw=0.8)
        s = n_star(tol)
        if s:
            ax.axvline(s, color="tab:red", ls=ls, lw=1, label="n* (+/-%d%%) = %d" % (100 * tol, s))
    ax.set_xscale("log"); ax.set_xlabel("bubbles measured (n)"); ax.set_ylabel("running D32 (mm)")
    ax.set_title("Froth: D32 vs number of bubbles (grey: 30 random orders)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "convergence.png"), dpi=150)
    print("\nplot: results/d32_convergence/convergence.png")
except ImportError:
    print("\n(matplotlib not available: no plot)")
