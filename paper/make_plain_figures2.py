"""Plain black-and-white figures: the system-design architecture and the video / document pipelines.

    python paper/make_plain_figures2.py   ->  paper/figures/fig_sd.png, fig_vd_plain.png

fig_sd.png redraws the main architecture figure of docs/System_Design.tex (Fig. "High-level system architecture").
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent / "figures"
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 6.5})


def box(ax, x, y, w, h, title, sub="", fs=6.4):
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec="black", lw=0.9))
    if sub:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=fs, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=fs - 0.8, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, fontweight="bold" if not sub else "normal", linespacing=1.25)


def arrow(ax, p, q, style="-|>", dashed=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7, lw=0.9, color="black", ls=(0, (3, 2)) if dashed else "-", shrinkA=0, shrinkB=0))


def system_design():
    fig, ax = plt.subplots(figsize=(6.3, 4.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 68)
    ax.axis("off")
    box(ax, 35, 61, 18, 6, "User")
    box(ax, 4, 51, 68, 7, "Presentation layer — Streamlit web interface", "upload a file, view the Trust Score report, download it")
    box(ax, 4, 41, 68, 7, "Input validation & content router", "checks file type, size and integrity; selects the pipeline")
    xs = (4, 27.7, 51.3)
    names = (("Image pipeline", "(Phase 1)", "preprocessing, frozen\nCLIP encoder, trained\nhead, calibrated\nscore, heatmap"),
             ("Video pipeline", "(Phase 2)", "frame sampling,\nface detection, frame\nclassifier, temporal\naggregation"),
             ("Document pipeline", "(Phase 3)", "OCR, forensic maps,\ndual-stream U-Net,\nlogic & PDF checks"))
    for x, (t, ph, d) in zip(xs, names):
        ax.add_patch(Rectangle((x, 21), 20.7, 14, fc="white", ec="black", lw=0.9))
        ax.text(x + 10.35, 33.0, t, ha="center", va="center", fontsize=6.3, fontweight="bold")
        ax.text(x + 10.35, 30.6, ph, ha="center", va="center", fontsize=5.6)
        ax.text(x + 10.35, 25.2, d, ha="center", va="center", fontsize=5.2, linespacing=1.25)
    box(ax, 4, 11.5, 68, 6.5, "Evidence fusion & calibration", "model scores + forensic cues + rule flags → one calibrated probability")
    box(ax, 4, 2, 33, 7.5, "Unified Trust Scoring Engine", "Trust Score 0–100 and a three-level verdict", fs=6.2)
    box(ax, 39, 2, 33, 7.5, "Report builder", "score, verdict, heatmaps,\nevidence, explanation", fs=6.2)
    box(ax, 80, 26, 17, 13, "Model store", "weights and\ncalibration\nparameters")
    box(ax, 80, 51, 17, 9, "Offline training", "Colab / Kaggle GPU")
    arrow(ax, (44, 61), (44, 58.1))
    arrow(ax, (38, 51), (38, 48.1))
    for x in xs:
        arrow(ax, (x + 10.35, 41), (x + 10.35, 35.1))
        arrow(ax, (x + 10.35, 21), (x + 10.35, 18.1))
    arrow(ax, (20.5, 11.5), (20.5, 9.6))
    arrow(ax, (37.5, 5.7), (39, 5.7))
    arrow(ax, (88.5, 51), (88.5, 39.1))
    arrow(ax, (80, 32.5), (72, 28), dashed=True)
    ax.text(76, 31.2, "load", fontsize=5.4)
    ax.text(88.5, 45, "export", fontsize=5.4, ha="center")
    ax.plot([72, 77, 77, 80], [54.5, 54.5, 54.5, 54.5], color="white", lw=0)
    ax.plot([4, 1.2, 1.2], [5.7, 5.7, 54.5], color="black", lw=0.9)
    arrow(ax, (1.2, 54.5), (3.9, 54.5))
    ax.text(0.4, 30, "report shown to user", rotation=90, fontsize=5.4, va="center", ha="center")
    fig.savefig(OUT / "fig_sd.png", dpi=330, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def vd_plain():
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")
    ax.text(1, 38, "(a) Video analyser", fontsize=7, fontweight="bold")
    ax.text(52, 38, "(b) Document analyser", fontsize=7, fontweight="bold")
    ax.plot([50.5, 50.5], [1, 37], color="black", lw=0.5)
    v = [("Video file", 1, 26), ("Sample frames\n(about 1 per s)", 17, 26), ("Detect and align\nfaces (MTCNN)", 33, 26),
         ("Frame score\n(CLIP + head)", 1, 8), ("Aggregate scores\n(top-fraction mean)", 17, 8), ("Video score and\ntimeline", 33, 8)]
    for t, x, y in v:
        box(ax, x, y, 14.2, 8.5, t, fs=5.4)
    for x in (15.2, 31.2):
        arrow(ax, (x, 30.2), (x + 1.8, 30.2))
    for x in (15.2, 31.2):
        arrow(ax, (x, 12.2), (x + 1.8, 12.2))
    ax.plot([40.1, 40.1, 8.1], [26, 21.5, 21.5], color="black", lw=0.9)
    arrow(ax, (8.1, 21.5), (8.1, 16.6))
    ax.text(1, 2.5, "split by video, not by frame", fontsize=5.2)
    box(ax, 52, 27, 11, 8.5, "Scan, photo\nor PDF", fs=5.3)
    box(ax, 66, 27, 13, 8.5, "Render page\nand OCR", fs=5.3)
    arrow(ax, (63, 31.2), (66, 31.2))
    for k, t in enumerate(("RGB", "ELA", "SRM", "DCT")):
        box(ax, 52 + k * 6.8, 15, 6.0, 5.5, t, fs=5.2)
    ax.plot([72.5, 72.5], [27, 23.5], color="black", lw=0.9)
    ax.plot([55, 75.4], [23.5, 23.5], color="black", lw=0.9)
    for cx in (55, 61.8, 68.6, 75.4):
        arrow(ax, (cx, 23.5), (cx, 20.6))
        ax.plot([cx, cx], [15, 13], color="black", lw=0.9)
    ax.plot([55, 75.4], [13, 13], color="black", lw=0.9)
    box(ax, 52, 3, 26.5, 8, "Dual-stream U-Net → tamper mask", fs=5.3)
    arrow(ax, (65, 13), (65, 11.1))
    box(ax, 82, 27, 17, 8.5, "Rule checks\n(totals, dates, fonts,\nPDF history)", fs=5.1)
    arrow(ax, (79, 31.2), (82, 31.2))
    box(ax, 82, 3, 17, 8, "Score, regions,\nexplanation", fs=5.3)
    arrow(ax, (78.5, 7), (82, 7))
    arrow(ax, (90.5, 27), (90.5, 11.1))
    fig.savefig(OUT / "fig_vd_plain.png", dpi=330, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    system_design(), vd_plain()
    print("written fig_sd.png, fig_vd_plain.png")
