"""Reliability diagram of the v3 image analyser on the held-out test split (before and after Platt scaling).

    python paper/make_calibration_plot.py     ->  paper/figures/fig_calib.png, paper/calibration_bins.json
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "prototype"))
from train_v3 import WEIGHTS, image_level, load_set  # noqa: E402

h = np.load(WEIGHTS / "our_head_v3.npz")
w, b = h["w"], float(h["b"])
cal = json.loads((WEIGHTS / "our_head_v3.json").read_text(encoding="utf-8"))["calibration"]

X, df = load_set("test_openfake")
img = image_level(df, X @ w + b)
y, s = img.label.to_numpy(), img.logit.to_numpy()
p_raw = 1 / (1 + np.exp(-s))
p_cal = 1 / (1 + np.exp(-(cal["a"] * s + cal["b"])))


def bins(p, n=10):
    edges = np.linspace(0, 1, n + 1)
    xs, ys, cnt = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.sum() >= 10:
            xs.append(float(p[m].mean()))
            ys.append(float(y[m].mean()))
            cnt.append(int(m.sum()))
    return xs, ys, cnt


rx, ry, rc = bins(p_raw)
cx, cy, cc = bins(p_cal)
fig, ax = plt.subplots(figsize=(3.45, 2.7))
ax.plot([0, 1], [0, 1], "--", color="#9CA3AF", lw=0.9, label="perfect calibration")
ax.plot(rx, ry, "o-", color="#C0392B", ms=3, lw=1.1, label="uncalibrated sigmoid")
ax.plot(cx, cy, "s-", color="#1F3A5F", ms=3, lw=1.1, label="after Platt scaling")
ax.set_xlabel("Predicted probability of AI", fontsize=6.5)
ax.set_ylabel("Observed share of AI images", fontsize=6.5)
ax.tick_params(labelsize=6)
ax.legend(fontsize=5.8, frameon=False, loc="upper left")
ax.grid(alpha=0.25, lw=0.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.savefig(HERE / "figures" / "fig_calib.png", dpi=330, bbox_inches="tight", pad_inches=0.03)
(HERE / "calibration_bins.json").write_text(json.dumps({"raw": {"p": rx, "obs": ry, "n": rc}, "platt": {"p": cx, "obs": cy, "n": cc}, "n_images": int(len(y))}, indent=1), encoding="utf-8")
print("bins raw:", len(rx), "platt:", len(cx), "images:", len(y))
