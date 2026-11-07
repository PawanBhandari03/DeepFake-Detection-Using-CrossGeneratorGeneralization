"""The layered architecture of the first version of the paper, redrawn in plain black and white.

    python paper/make_layered_bw.py   ->  paper/figures/fig_arch_bw.png

Solid boxes = implemented; dashed boxes = designed.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent / "figures"
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 6.5})


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.9, color="black", shrinkA=0, shrinkB=0))


def b(ax, x, y, w, h, text, dashed=False, bold=False, fs=6.2):
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec="black", lw=0.9, ls=(0, (4, 2)) if dashed else "-"))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.3)


fig, ax = plt.subplots(figsize=(7.1, 3.3))
ax.set_xlim(0, 100)
ax.set_ylim(0, 47)
ax.axis("off")
b(ax, 2, 39, 28, 6.5, "User upload\nimage, video or document")
b(ax, 37, 39, 40, 6.5, "Validation and routing\n(type read from the bytes, size limit)")
arrow(ax, (30, 42.2), (37, 42.2))
specs = ((2, "IMAGE ANALYSER (implemented)", "native-resolution patches\nfrozen CLIP + trained head\npatch heatmap", False),
         (35, "VIDEO ANALYSER (designed)", "sampled frames, face crops\nframe scores, top-fraction\naggregation, timeline", True),
         (68, "DOCUMENT ANALYSER (designed)", "OCR, ELA / SRM / DCT views\ndual-stream U-Net mask\narithmetic and PDF checks", True))
for x, t, d, dashed in specs:
    b(ax, x, 17, 30, 17, "", dashed=dashed)
    ax.text(x + 15, 31.2, t, ha="center", va="center", fontsize=5.8, fontweight="bold")
    ax.text(x + 15, 24, d, ha="center", va="center", fontsize=5.7, linespacing=1.4)
    arrow(ax, (57, 39), (x + 15, 34.1))
b(ax, 2, 3.5, 30, 8, "Evidence records\nscore, map, cues, flags")
b(ax, 38, 3.5, 25, 8, "Calibration and fusion\np = sigmoid(a·s + c)")
b(ax, 69, 3.5, 29, 8, "Trust Score 0–100, verdict band,\nexplanation, report", bold=True)
for x in (17, 50, 83):
    ax.plot([x, x], [17, 14], color="black", lw=0.9)
ax.plot([17, 83], [14, 14], color="black", lw=0.9)
arrow(ax, (17, 14), (17, 11.6))
arrow(ax, (32, 7.5), (38, 7.5))
arrow(ax, (63, 7.5), (69, 7.5))
ax.text(50, 0.4, "Solid boxes: implemented and evaluated.   Dashed boxes: designed, evaluation planned.", ha="center", fontsize=5.7)
fig.savefig(OUT / "fig_arch_bw.png", dpi=330, bbox_inches="tight", pad_inches=0.04)
print("written fig_arch_bw.png")
