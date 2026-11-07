"""Draw the four figures of the research paper (flow diagrams and result charts).

    python paper/make_figures.py      ->  paper/figures/fig1..fig4.png

Result charts read prototype/results/*.json, so the numbers always match the experiments.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "prototype" / "results"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

NAVY, TEAL, GREY, AMBER, RED, GREEN = "#1F3A5F", "#2A7F7B", "#6B7280", "#E0A030", "#C0392B", "#3A8F5C"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})


def box(ax, x, y, w, h, text, fc="#EEF3F9", ec=NAVY, fs=5.6, dashed=False, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=0.9,
                                ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#111827",
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p, q, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.9, color=GREY,
                                 ls="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}"))


# ---------------------------------------------------------------- Fig. 1: system overview (full width)
def fig1(planned=True):
    fig, ax = plt.subplots(figsize=(7.1, 3.05 if planned else 2.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")
    w, h, gap, y1 = 13.6, 11.5, 3.2, 30
    xs = [0.8 + i * (w + gap) for i in range(6)]
    labels = ["Upload\n(JPG, PNG,\nWEBP, HEIC...)", "Decode to\n8-bit RGB\n(metadata is\nnot scored)",
              "10 patches\n224x224, native\nresolution\n(centre + 3x3)", "Re-save each\npatch as JPEG\nquality 95",
              "Frozen CLIP\nViT-L/14\n768 numbers\nper patch", "Linear head\n(trained by us)\nlogit per patch"]
    fcs = ["#EEF3F9"] * 4 + ["#E6F2F1", "#FFF4DC"]
    for x, t, fc in zip(xs, labels, fcs):
        box(ax, x, y1, w, h, t, fc=fc)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + w + 0.4, y1 + h / 2), (b - 0.4, y1 + h / 2))
    y2 = 14.5
    box(ax, xs[5], y2, w, h, "Mean of the\n10 patch logits\n= image score", fc="#FFF4DC")
    box(ax, xs[4], y2, w, h, "Platt scaling\np = sigmoid\n(a*s + c), fitted\non validation", fc="#FFF4DC")
    box(ax, xs[3], y2, w, h, "Trust Score\n100 x (1-p)\n3 verdict bands", fc="#E4F3E8", ec=GREEN, bold=True)
    box(ax, xs[2], y2, w, h, "Heatmap from\nthe 3x3 patch\nprobabilities", fc="#EEF3F9")
    box(ax, xs[1], y2, w, h, "Metadata panel\n(EXIF, C2PA)\nshown, not\nscored", fc="#F3F4F6", ec=GREY)
    arrow(ax, (xs[5] + w / 2, y1 - 0.4), (xs[5] + w / 2, y2 + h + 0.4))
    arrow(ax, (xs[5] - 0.4, y2 + h / 2), (xs[4] + w + 0.4, y2 + h / 2))
    arrow(ax, (xs[4] - 0.4, y2 + h / 2), (xs[3] + w + 0.4, y2 + h / 2))
    arrow(ax, (xs[5] + 3, y1 - 0.4), (xs[2] + w / 2, y2 + h + 0.4), rad=0.0, dashed=True)
    if planned:                                           # planned modules of the larger framework
        y3 = 0.8
        box(ax, xs[4], y3, w, 8.2, "Video analyser\n(planned)", fc="white", ec=GREY, dashed=True, fs=5.4)
        box(ax, xs[5], y3, w, 8.2, "Document\nanalyser\n(planned)", fc="white", ec=GREY, dashed=True, fs=5.4)
        arrow(ax, (xs[4] + w / 2, y3 + 8.2 + 0.4), (xs[3] + w - 2, y2 - 0.3), dashed=True)
        arrow(ax, (xs[5] + 3, y3 + 8.2 + 0.4), (xs[3] + w - 0.5, y2 - 0.3), dashed=True)
        ax.text(0.8, 9.0, "Implemented: image module (solid boxes).\nPlanned, not evaluated here: video and\ndocument modules (dashed boxes).", fontsize=6.3, color=GREY, va="top")
    else:
        ax.set_ylim(11, 44)
    fig.savefig(OUT / ("fig1.png" if planned else "fig1_image.png"), dpi=330, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 2: training pipeline (column width)
def fig2():
    fig, ax = plt.subplots(figsize=(3.45, 3.95))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 118)
    ax.axis("off")
    steps = [
        ("Public datasets: OpenFake (train), DiffSeg30k,\ntiny-audits (edited images with masks)", "#EEF3F9", NAVY),
        ("Two views per image:\nclean copy + degraded copy\n(resize, blur, JPEG quality 50-95)", "#EEF3F9", NAVY),
        ("3 random 224x224 patches per view,\nre-saved as JPEG quality 95", "#EEF3F9", NAVY),
        ("Frozen CLIP ViT-L/14, half precision\n(Kaggle T4 GPU, about 112 patches/s)", "#E6F2F1", TEAL),
        ("Feature files: 414,433 patches x 768 numbers\n+ labels, generator name, view type", "#E6F2F1", TEAL),
        ("Logistic regression, L2 penalty\n(laptop CPU, minutes)", "#FFF4DC", AMBER),
        ("Head = 768 weights + 1 bias (a few KB)\n+ Platt calibration + verdict bands", "#E4F3E8", GREEN),
    ]
    h, gap, y = 12.2, 4.6, 118 - 12.2 - 0.5
    for i, (t, fc, ec) in enumerate(steps):
        box(ax, 3, y, 94, h, t, fc=fc, ec=ec, fs=5.9)
        if i < len(steps) - 1:
            arrow(ax, (50, y - 0.3), (50, y - gap + 0.5))
        y -= h + gap
    fig.savefig(OUT / "fig2.png", dpi=330, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 3: share of AI images caught per condition
def fig3():
    r = json.loads((RES / "v3_results.json").read_text(encoding="utf-8"))
    models = [("v2_reference", "Earlier head (v2)", "#9CA3AF"), ("v3_noaug", "v3, no compression training", "#7FB3D5"),
              ("v3_proof", "v3 proof (no OpenAI/Google)", "#2A7F7B"), ("v3", "v3", NAVY)]
    conds = [("clean", "Clean"), ("jpeg70", "JPEG 70"), ("half_q75", "Half size\n+ JPEG 75")]
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    n = len(models)
    bw = 0.2
    for j, (key, label, col) in enumerate(models):
        vals = [r[key]["test_openfake"][c]["fake_caught"] * 100 for c, _ in conds]
        vals.append(r[key]["test_reddit"]["clean"]["fake_caught"] * 100)
        ax.bar([i + (j - (n - 1) / 2) * bw for i in range(4)], vals, bw, label=label, color=col)
    ax.set_xticks(range(4))
    ax.set_xticklabels([c for _, c in conds] + ["In the wild\n(Reddit)"], fontsize=6.2)
    ax.set_ylabel("AI images caught (%)", fontsize=6.5)
    ax.set_ylim(0, 105)
    ax.tick_params(axis="y", labelsize=6)
    ax.legend(fontsize=5.4, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0), columnspacing=1.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.savefig(OUT / "fig3.png", dpi=330, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 4: what the verdict bands do
def fig4():
    c = json.loads((RES / "v3_calibration.json").read_text(encoding="utf-8"))
    rows = [("Real photos, clean", c["test_openfake"]["clean"]["real"]), ("Real photos, JPEG 70", c["test_openfake"]["jpeg70"]["real"]),
            ("Real photos, in the wild", c["test_reddit"]["clean"]["real"]), ("AI images, clean", c["test_openfake"]["clean"]["fake"]),
            ("AI images, half size + JPEG 75", c["test_openfake"]["half_q75"]["fake"]), ("AI images, JPEG 70", c["test_openfake"]["jpeg70"]["fake"]),
            ("AI images, in the wild", c["test_reddit"]["clean"]["fake"])]
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    labels = [r[0] for r in rows][::-1]
    data = [r[1] for r in rows][::-1]
    left = [0.0] * len(rows)
    for key, name, col in (("authentic", "Likely Authentic", GREEN), ("review", "Needs Human Review", AMBER), ("ai", "Likely AI-generated", RED)):
        vals = [d[key] * 100 for d in data]
        ax.barh(labels, vals, left=left, color=col, label=name, height=0.62)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v >= 7:
                ax.text(l + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=5.6, color="white", fontweight="bold")
        left = [a + b for a, b in zip(left, vals)]
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of images (%)", fontsize=6.5)
    ax.tick_params(axis="both", labelsize=5.9)
    ax.legend(fontsize=5.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.42, 1.2), handlelength=1.0, columnspacing=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(OUT / "fig4.png", dpi=330, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    fig1(), fig1(planned=False), fig2(), fig3(), fig4()
    print("figures written to", OUT)
