"""Diagrams for the full-project paper (image + video + document).

    python paper/make_full_figures.py   ->  paper/figures/fig_arch.png, fig_vd.png

The training pipeline (fig2.png) and the robustness chart (fig3.png) come from make_figures.py and are reused.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_figures import AMBER, GREEN, GREY, NAVY, TEAL, arrow, box

OUT = Path(__file__).resolve().parent / "figures"
DONE, TODO = "#E6F2F1", "#FFFFFF"


def fig_arch():
    fig, ax = plt.subplots(figsize=(7.1, 3.25))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 47)
    ax.axis("off")
    box(ax, 2, 39, 28, 6.5, "User upload\nimage, video or document", fs=6)
    box(ax, 37, 39, 40, 6.5, "Validation and routing\n(type read from the bytes, size limit)", fs=5.8)
    arrow(ax, (30.4, 42.2), (36.6, 42.2))
    # three analysers
    specs = [
        (2, "IMAGE ANALYSER (implemented)", "native-resolution patches\nfrozen CLIP + trained head\npatch heatmap", DONE, TEAL, False),
        (35, "VIDEO ANALYSER (designed)", "sampled frames, face crops\nframe scores, top-fraction\naggregation, timeline", TODO, GREY, True),
        (68, "DOCUMENT ANALYSER (designed)", "OCR, ELA / noise / DCT views\ndual-stream U-Net mask\narithmetic + PDF checks", TODO, GREY, True),
    ]
    for x, title, body, fc, ec, dashed in specs:
        box(ax, x, 17, 30, 17, "", fc=fc, ec=ec, dashed=dashed)
        ax.text(x + 15, 31.2, title, ha="center", va="center", fontsize=5.6, fontweight="bold", color="#111827")
        ax.text(x + 15, 24, body, ha="center", va="center", fontsize=5.7, color="#111827", linespacing=1.4)
        arrow(ax, (57, 38.6), (x + 15, 34.6), dashed=dashed)
    # evidence and scoring
    box(ax, 2, 3.5, 30, 8, "Evidence records\nscore, map, cues, flags", fs=5.8)
    box(ax, 38, 3.5, 25, 8, "Calibration and fusion\np = sigmoid(a*s + c)", fc="#FFF4DC", fs=5.8)
    box(ax, 69, 3.5, 29, 8, "Trust Score 0-100, verdict band,\nexplanation, report", fc="#E4F3E8", ec=GREEN, fs=5.8, bold=True)
    for x, dashed in ((17, False), (50, True), (83, True)):
        arrow(ax, (x, 16.6), (17 + (x - 17) * 0.15, 12.1), dashed=dashed)
    arrow(ax, (32.4, 7.5), (37.6, 7.5))
    arrow(ax, (63.4, 7.5), (68.6, 7.5))
    ax.text(50, 0.6, "Solid boxes: implemented and evaluated.  Dashed boxes: designed, evaluation planned.", ha="center", fontsize=5.8, color=GREY)
    fig.savefig(OUT / "fig_arch.png", dpi=330, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def fig_vd():
    fig, ax = plt.subplots(figsize=(7.1, 2.55))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")
    ax.text(1, 38.5, "(a) Video analyser", fontsize=6.8, fontweight="bold", color=NAVY)
    ax.text(52, 38.5, "(b) Document analyser", fontsize=6.8, fontweight="bold", color=NAVY)
    ax.plot([50.5, 50.5], [1, 38], color="#D1D5DB", lw=0.8)
    w, h = 14.2, 9
    # video: two rows of three
    v1 = ["Video file\n(short clip)", "Sample about\n1 frame/s\n(at most 32)", "Detect and align\nfaces (MTCNN),\nadd a margin"]
    v2 = ["Frame score:\nCLIP + trained\nhead", "Aggregate:\nmean, max or\ntop-fraction", "Video score and\ntimeline of\nsuspect frames"]
    xs = [1, 17.2, 33.4]
    for i, t in enumerate(v1):
        box(ax, xs[i], 25, w, h, t, fs=5.2, dashed=True, fc="white", ec=GREY)
    for i, t in enumerate(v2):
        box(ax, xs[i], 9.5, w, h, t, fs=5.2, dashed=True, fc="white" if i < 2 else "#E4F3E8", ec=GREY if i < 2 else GREEN)
    for i in range(2):
        arrow(ax, (xs[i] + w + 0.3, 29.5), (xs[i + 1] - 0.3, 29.5), dashed=True)
        arrow(ax, (xs[i] + w + 0.3, 14), (xs[i + 1] - 0.3, 14), dashed=True)
    arrow(ax, (xs[2] + w / 2, 24.6), (xs[0] + w / 2, 19.1), dashed=True, rad=0.0)
    ax.text(1, 3.2, "Split by video, never by frame. No face: whole-frame image\nanalyser as fallback.", fontsize=5.3, color=GREY, va="center")
    # document
    box(ax, 52, 28, 11, 8.5, "Scan,\nphoto\nor PDF", fs=5.2, dashed=True, fc="white", ec=GREY)
    box(ax, 66, 28, 12.5, 8.5, "Render page\n+ OCR", fs=5.2, dashed=True, fc="white", ec=GREY)
    arrow(ax, (63.3, 32.2), (65.7, 32.2), dashed=True)
    for k, t in enumerate(["RGB", "ELA", "SRM", "DCT"]):
        box(ax, 52.3 + k * 6.8, 15.5, 6.0, 5.2, t, fs=5.0, dashed=True, fc="white", ec=GREY)
    arrow(ax, (72, 27.6), (62, 20.9), dashed=True)
    box(ax, 52, 5.5, 26.5, 7, "Dual-stream U-Net  ->  tamper mask", fs=5.2, dashed=True, fc="white", ec=GREY)
    arrow(ax, (65, 15.1), (65, 12.9), dashed=True)
    box(ax, 82, 24.5, 17, 11.5, "Rule checks\ntotals vs. line items,\ndates, font height,\nPDF history", fs=4.9, dashed=True, fc="white", ec=GREY)
    arrow(ax, (78.8, 31.5), (81.7, 31.5), dashed=True)
    box(ax, 82, 5.5, 17, 11.5, "Document score,\nhighlighted regions,\nexplanation", fs=5.0, dashed=True, fc="#E4F3E8", ec=GREEN)
    arrow(ax, (78.8, 9), (81.7, 9), dashed=True)
    arrow(ax, (90.5, 24.1), (90.5, 17.4), dashed=True)
    fig.savefig(OUT / "fig_vd.png", dpi=330, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    fig_arch(), fig_vd()
    print("written to", OUT)
