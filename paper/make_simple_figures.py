"""Two plain black-and-white block diagrams for the 4-page paper (architecture and image pipeline).

    python paper/make_simple_figures.py   ->  paper/figures/fig_s1.png, fig_s2.png

Deliberately plain (white boxes, black lines, no colour) so that they look like quick draw.io sketches.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent / "figures"
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 6.5})


def box(ax, x, y, w, h, text, dashed=False, fs=6.3, bold=False):
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec="black", lw=0.9, ls=(0, (4, 2)) if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.9, color="black", shrinkA=0, shrinkB=0))


def arch():
    fig, ax = plt.subplots(figsize=(3.3, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 140)
    ax.axis("off")
    box(ax, 12, 124, 76, 12, "User uploads a file\n(image, video or document)")
    arrow(ax, (50, 124), (50, 116))
    box(ax, 12, 104, 76, 12, "Validation and routing\n(type from the file bytes)")
    for x in (16.5, 50, 83.5):
        arrow(ax, (x, 104), (x, 94))
    box(ax, 1, 72, 31, 22, "Image\nanalyser")
    box(ax, 34.5, 72, 31, 22, "Video\nanalyser", dashed=True)
    box(ax, 68, 72, 31, 22, "Document\nanalyser", dashed=True)
    for x in (16.5, 50, 83.5):
        arrow(ax, (x, 72), (x, 62))
    box(ax, 8, 50, 84, 12, "Evidence record\n(score, map, cues, flags)")
    arrow(ax, (50, 50), (50, 42))
    box(ax, 8, 30, 84, 12, "Calibration and Trust Score\n(0–100, three bands)")
    arrow(ax, (50, 30), (50, 22))
    box(ax, 8, 8, 84, 14, "Verdict, heatmap,\nexplanation, report", bold=True)
    ax.text(50, 1.5, "dashed = designed, not yet built", ha="center", fontsize=5.6, color="#444444")
    fig.savefig(OUT / "fig_s1.png", dpi=330, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def pipeline():
    steps = ["Uploaded image file", "Decode to 8-bit RGB\n(metadata is shown, not scored)", "Cut 10 patches, 224×224\n(centre + 3×3 grid, native size)",
             "Re-save each patch\nas JPEG, quality 95", "Frozen CLIP ViT-L/14\n(768 numbers per patch)", "Linear head (trained by us)\nlogit per patch",
             "Mean of the 10 logits\n= image score", "Platt calibration\n→ Trust Score and verdict"]
    fig, ax = plt.subplots(figsize=(3.3, 4.6))
    n = len(steps)
    h, gap = 11.2, 5.6
    total = n * h + (n - 1) * gap
    ax.set_xlim(0, 100)
    ax.set_ylim(0, total + 2)
    ax.axis("off")
    y = total - h + 1
    for i, t in enumerate(steps):
        box(ax, 10, y, 80, h, t, bold=(i == n - 1))
        if i < n - 1:
            arrow(ax, (50, y), (50, y - gap))
        y -= h + gap
    fig.savefig(OUT / "fig_s2.png", dpi=330, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    arch(), pipeline()
    print("written fig_s1.png, fig_s2.png")
