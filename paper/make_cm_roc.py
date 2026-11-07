"""Confusion matrix and ROC curve of the v3 image analyser on the held-out test split (clean files).

    python paper/make_cm_roc.py     ->  paper/figures/fig_cm_roc.png, paper/cm_roc.json

Image-level scores as in the app (mean of the 10 patch logits); decision at probability 0.5 (logit > 0).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "prototype"))
from train_v3 import WEIGHTS, image_level, load_set  # noqa: E402

h = np.load(WEIGHTS / "our_head_v3.npz")
w, b = h["w"], float(h["b"])
X, df = load_set("test_openfake")
img = image_level(df, X @ w + b)
g = img[img.variant == "clean"]
y, s = g.label.to_numpy(), g.logit.to_numpy()
pred = (s > 0).astype(int)
tn, fp = int(((y == 0) & (pred == 0)).sum()), int(((y == 0) & (pred == 1)).sum())
fn, tp = int(((y == 1) & (pred == 0)).sum()), int(((y == 1) & (pred == 1)).sum())
auc = float(roc_auc_score(y, s))
acc = (tn + tp) / len(y)
json.dump({"tn": tn, "fp": fp, "fn": fn, "tp": tp, "auc": auc, "accuracy": acc, "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum())},
          open(HERE / "cm_roc.json", "w"), indent=1)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 2.55), gridspec_kw={"width_ratios": [1.05, 1]})
m = np.array([[tn, fp], [fn, tp]])
a1.imshow(m, cmap="Blues")
for i in range(2):
    for j in range(2):
        a1.text(j, i, f"{m[i, j]:,}", ha="center", va="center", fontsize=9, fontweight="bold", color="white" if m[i, j] > m.max() / 2 else "#1F3A5F")
a1.set_xticks([0, 1])
a1.set_xticklabels(["real", "AI-generated"], fontsize=7)
a1.set_yticks([0, 1])
a1.set_yticklabels(["real", "AI-generated"], fontsize=7, rotation=90, va="center")
a1.set_xlabel("Predicted label", fontsize=7)
a1.set_ylabel("True label", fontsize=7)
a1.set_title(f"Confusion matrix (accuracy {acc * 100:.1f}%)", fontsize=7.5)
fpr, tpr, _ = roc_curve(y, s)
a2.plot(fpr, tpr, color="#1F3A5F", lw=1.5, label=f"v3 (AUC = {auc:.3f})")
a2.plot([0, 1], [0, 1], "--", color="#9CA3AF", lw=0.9, label="chance (AUC = 0.5)")
a2.set_xlabel("False positive rate", fontsize=7)
a2.set_ylabel("True positive rate", fontsize=7)
a2.set_title("ROC curve", fontsize=7.5)
a2.tick_params(labelsize=6.5)
a2.legend(fontsize=6.5, frameon=False, loc="lower right")
a2.grid(alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig(HERE / "figures" / "fig_cm_roc.png", dpi=330, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

# column-width version: confusion matrix above the ROC curve
fig, (b1, b2) = plt.subplots(2, 1, figsize=(3.45, 4.7))
b1.imshow(m, cmap="Blues")
for i in range(2):
    for j in range(2):
        b1.text(j, i, f"{m[i, j]:,}", ha="center", va="center", fontsize=8.5, fontweight="bold", color="white" if m[i, j] > m.max() / 2 else "#1F3A5F")
b1.set_xticks([0, 1])
b1.set_xticklabels(["real", "AI-generated"], fontsize=6.5)
b1.set_yticks([0, 1])
b1.set_yticklabels(["real", "AI-generated"], fontsize=6.5, rotation=90, va="center")
b1.set_xlabel("Predicted label", fontsize=6.5)
b1.set_ylabel("True label", fontsize=6.5)
b1.set_title(f"Confusion matrix (accuracy {acc * 100:.1f}%)", fontsize=7)
b2.plot(fpr, tpr, color="#1F3A5F", lw=1.4, label=f"v3 (AUC = {auc:.3f})")
b2.plot([0, 1], [0, 1], "--", color="#9CA3AF", lw=0.9, label="chance (AUC = 0.5)")
b2.set_xlabel("False positive rate", fontsize=6.5)
b2.set_ylabel("True positive rate", fontsize=6.5)
b2.set_title("ROC curve", fontsize=7)
b2.tick_params(labelsize=6)
b2.legend(fontsize=6, frameon=False, loc="lower right")
b2.grid(alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig(HERE / "figures" / "fig_cm_roc_col.png", dpi=330, bbox_inches="tight", pad_inches=0.03)
print(json.dumps({"tn": tn, "fp": fp, "fn": fn, "tp": tp, "auc": round(auc, 4), "accuracy": round(acc, 4)}))
