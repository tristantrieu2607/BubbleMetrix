import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({"font.family": "Liberation Sans", "font.size": 26, "axes.edgecolor": "#BBBBBB",
                 "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                 "xtick.color": "#555555", "ytick.color": "#231F20", "axes.labelcolor": "#555555",
                 "savefig.dpi": 200, "figure.facecolor": "white"})
INK, GREY, LG, YEL = "#231F20", "#9A9A9A", "#D9D9D9", "#FFDC00"
O = "/tmp/claude-0/brand/"

# 1. cross-validation: recall and D32 ratio per held-out clip
clips = ["Coarse 200air","Coarse 300air","Fine 200air","Fine inline","Fine 300air","Fine 1440rpm","Med 200air","Med 300air"]
n = [66,10,48,34,42,50,98,29]
rec = [0.894,1.000,0.938,1.000,0.905,0.920,0.888,0.828]
d32 = [1.014,0.966,0.979,0.914,1.010,0.983,1.023,0.992]
fig, (a, b) = plt.subplots(1, 2, figsize=(14.8, 7.2), sharey=True, gridspec_kw={"wspace": 0.08})
ys = list(range(len(clips)))[::-1]
for ax, vals, ref, lo, hi, ttl in [(a, rec, None, 0.75, 1.05, "Recall"), (b, d32, 1.0, 0.85, 1.10, "D32 ratio (model / hand)")]:
    if ref:
        ax.axvspan(0.95, 1.05, color=YEL, alpha=0.35, lw=0)
        ax.axvline(ref, color=INK, lw=1.5)
        ax.text(1.047, -0.5, "±5% band", ha="right", va="bottom", fontsize=17, color="#555555")
    for y, v in zip(ys, vals):
        ax.plot([lo, v], [y, y], color=LG, lw=3, zorder=1)
        ax.scatter([v], [y], s=260, color=INK, zorder=3)
        ax.text(v, y + 0.32, f"{v:.2f}", ha="center", va="bottom", fontsize=20, color=INK)
    ax.set_xlim(lo, hi); ax.set_title(ttl, loc="left", fontsize=26, fontweight="bold", color=INK, pad=14)
    ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", labelsize=20)
    ax.set_ylim(-0.6, len(clips) - 0.2)
a.set_yticks(ys); a.set_yticklabels([f"{c} (n={k})" for c, k in zip(clips, n)], fontsize=21)
fig.savefig(O + "f_cv.png", bbox_inches="tight", pad_inches=0.15); plt.close(fig)

# 2. recall by size, previous vs final model
bands = ["12–40", "40–70", "70–100", "100–140", "140+"]
nb = [4, 29, 68, 21, 2]
old = [0.25, 0.86, 0.76, 0.62, 0.00]; new = [0.75, 0.86, 0.82, 0.90, 1.00]
fig, ax = plt.subplots(figsize=(14.8, 7.0))
x = range(len(bands)); w = 0.38
ax.bar([i - w/2 for i in x], old, w, color=LG, label="Previous model (Sep)")
ax.bar([i + w/2 for i in x], new, w, color=INK, label="Final model")
for i in x:
    ax.text(i - w/2, old[i] + 0.02, f"{old[i]:.2f}", ha="center", fontsize=19, color="#555555")
    ax.text(i + w/2, new[i] + 0.02, f"{new[i]:.2f}", ha="center", fontsize=19, color=INK, fontweight="bold")
ax.add_patch(plt.Rectangle((2.5, 0), 1.0, 1.13, color=YEL, alpha=0.35, lw=0, zorder=0))
ax.set_xticks(list(x)); ax.set_xticklabels([f"{b} px\n(n={k})" for b, k in zip(bands, nb)], fontsize=20)
ax.set_ylim(0, 1.15); ax.set_yticks([0, 0.5, 1.0]); ax.tick_params(axis="y", labelsize=20, length=0)
ax.grid(axis="y", color="#EEEEEE"); ax.set_axisbelow(True); ax.set_ylabel("Recall", fontsize=22)
ax.legend(frameon=False, fontsize=20, loc="upper left", ncol=2, bbox_to_anchor=(0, 1.12))
fig.savefig(O + "f_size.png", bbox_inches="tight", pad_inches=0.15); plt.close(fig)

# 3. false-positive audit
cats = ["In-focus bubble\nmissed in labelling", "Part-hidden bubble\n(kept by design)", "Cut by the\nimage edge", "Defocused:\ngenuine model error"]
cnt = [44, 20, 17, 15]
fig, ax = plt.subplots(figsize=(14.8, 6.6))
ys = list(range(4))[::-1]
cols = [GREY, GREY, GREY, INK]
ax.barh(ys, cnt, color=cols, height=0.62)
for y, c in zip(ys, cnt):
    ax.text(c + 0.8, y, f"{c}  ({c/96:.0%})", va="center", fontsize=22, color=INK)
ax.set_yticks(ys); ax.set_yticklabels(cats, fontsize=21); ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 56); ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
ax.set_xlabel("Unmatched detections on 9 held-out frames (96 in total)", fontsize=20)
fig.savefig(O + "f_audit.png", bbox_inches="tight", pad_inches=0.15); plt.close(fig)

# 4. froth D32 per frame
t = [0, 30, 60, 120, 180, 240, 300, 360, 420]
d = [2.123, 2.301, 2.254, 2.085, 2.064, 2.077, 2.081, 2.254, 2.227]
fig, ax = plt.subplots(figsize=(14.8, 6.0))
ax.axhspan(2.16 * 0.95, 2.16 * 1.05, color=YEL, alpha=0.35, lw=0)
ax.axhline(2.16, color=INK, lw=1.5, ls="--")
ax.text(425, 2.16 + 0.008, "pooled 2.16 mm", ha="right", va="bottom", fontsize=19, color="#555555")
ax.plot(t, d, color=LG, lw=3, zorder=1); ax.scatter(t, d, s=220, color=INK, zorder=3)
for ti, di in zip(t, d):
    ax.text(ti, di + (0.018 if di >= 2.16 else -0.034), f"{di:.2f}", ha="center", fontsize=18, color=INK)
ax.set_ylim(1.95, 2.38); ax.set_xticks(t); ax.tick_params(labelsize=19, length=0)
ax.set_xlabel("Time in video (s)", fontsize=20); ax.set_ylabel("D32 (mm)", fontsize=20)
ax.grid(axis="y", color="#EEEEEE"); ax.set_axisbelow(True)
fig.savefig(O + "f_froth.png", bbox_inches="tight", pad_inches=0.15); plt.close(fig)
print("ok")
