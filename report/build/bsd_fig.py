import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({"font.family": "Liberation Sans", "axes.spines.top": False, "axes.spines.right": False,
                 "axes.spines.left": False, "axes.edgecolor": "#BBBBBB"})
J = json.load(open("/tmp/claude-0/brand/bsd.json")); d = np.array([b["diameter_val"] for b in J["bubbles"]]); d32 = J["d32"]
fig, ax = plt.subplots(figsize=(7.9, 7.9))
bins = np.arange(0, 500, 25)
ax.hist(d, bins=bins, color="#9A9A9A", edgecolor="white", linewidth=1.5)
ax.axvline(d32, color="#231F20", lw=3)
ymax = ax.get_ylim()[1]
ax.text(d32 + 8, ymax * 0.93, f"D32 = {d32:.0f} µm", fontsize=30, fontweight="bold", color="#231F20", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFDC00", ec="none"))
ax.set_xlabel("Bubble diameter (µm)", fontsize=24, color="#555555"); ax.set_ylabel("Bubbles", fontsize=24, color="#555555")
ax.tick_params(labelsize=20, length=0); ax.grid(axis="y", color="#EEEEEE"); ax.set_axisbelow(True)
ax.text(0.0, 1.02, f"{len(d)} bubbles from one 3-second clip", transform=ax.transAxes, fontsize=20, color="#555555")
fig.savefig("/tmp/claude-0/brand/step3.png", dpi=200, bbox_inches="tight", pad_inches=0.15, facecolor="white")
print(len(d), d32)
