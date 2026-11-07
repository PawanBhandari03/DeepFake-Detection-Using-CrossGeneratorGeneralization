"""Robustness results: v3, v3 without augmentation and the earlier v2 head under seven image conditions.

    python paper/robust_analysis.py    ->  paper/robust_results.json, paper/figures/fig_robust.png

Needs data/features/v3/test_robust_*.npz (from the Kaggle run kaggle/robust/robust.py) plus the existing
test_openfake_*.npz (conditions clean and JPEG 70).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "prototype"))
from train_v3 import WEIGHTS, image_level, load_set  # noqa: E402

ORDER = [("clean", "Clean"), ("jpeg90", "JPEG 90"), ("jpeg70", "JPEG 70"), ("jpeg50", "JPEG 50"),
         ("screenshot", "Screenshot-\nlike"), ("half", "Half size"), ("blur", "Blur")]
MODELS = [("v2", "Earlier head (v2)", "#9CA3AF"), ("v3_noaug", "v3, no augmentation", "#7FB3D5"), ("v3", "v3", "#1F3A5F")]


def head(key):
    if key == "v2":
        h = np.load(WEIGHTS / "our_head_v2.npz")
        return h["w"][:768], float(h["b"])
    h = np.load(WEIGHTS / f"our_head_{key}.npz")
    return h["w"], float(h["b"])


X1, d1 = load_set("test_openfake")
X2, d2 = load_set("test_robust")
keep1 = d1.variant.isin(["clean", "jpeg70"]).to_numpy()
X = np.concatenate([X1[keep1], X2])
D = pd.concat([d1[keep1], d2], ignore_index=True)

res = {}
for key, _, _ in MODELS:
    w, b = head(key)
    img = image_level(D, X @ w + b)
    res[key] = {}
    for var, _ in ORDER:
        g = img[img.variant == var]
        real, fake = g[g.label == 0], g[g.label == 1]
        res[key][var] = {"auc": round(float(roc_auc_score(g.label, g.logit)), 4), "caught": round(float((fake.logit > 0).mean()), 4),
                         "real_correct": round(float((real.logit < 0).mean()), 4), "n_real": int(len(real)), "n_fake": int(len(fake))}
(HERE / "robust_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.35))
for ax, field, title in ((axes[0], "auc", "ROC-AUC"), (axes[1], "caught", "AI images caught (%)")):
    for key, label, col in MODELS:
        vals = [res[key][v][field] * (100 if field == "caught" else 1) for v, _ in ORDER]
        ax.plot(range(len(ORDER)), vals, "o-", color=col, ms=3, lw=1.2, label=label)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([n for _, n in ORDER], fontsize=5.6)
    ax.set_title(title, fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylim(0.5, 1.0)
axes[1].set_ylim(0, 100)
axes[0].legend(fontsize=5.8, frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(HERE / "figures" / "fig_robust.png", dpi=330, bbox_inches="tight", pad_inches=0.03)
for key, label, _ in MODELS:
    print(label, {v: (res[key][v]["auc"], round(res[key][v]["caught"] * 100)) for v, _ in ORDER})
