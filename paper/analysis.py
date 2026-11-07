"""Extra statistics for the research paper: bootstrap confidence intervals and calibration error (ECE).

    python paper/analysis.py        ->  paper/analysis.json

Uses the same v3 head, features and image-level scoring as prototype/train_v3.py (mean of the patch logits).
Bootstrap = resample test images with replacement 1000 times; the interval is the 2.5 - 97.5 percentile.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "prototype"))
from train_v3 import WEIGHTS, image_level, load_set  # noqa: E402

rng = np.random.default_rng(0)
h = np.load(WEIGHTS / "our_head_v3.npz")
w, b = h["w"], float(h["b"])
meta = json.loads((WEIGHTS / "our_head_v3.json").read_text(encoding="utf-8"))
cal = meta["calibration"]


def boot(y, s, n=1000):
    """95% bootstrap intervals for AUC and for the share of positives with score > 0."""
    aucs, caught = [], []
    idx = np.arange(len(y))
    for _ in range(n):
        i = rng.choice(idx, len(idx), replace=True)
        if y[i].min() == y[i].max():
            continue
        aucs.append(roc_auc_score(y[i], s[i]))
        pos = i[y[i] == 1]
        caught.append(float((s[pos] > 0).mean()))
    f = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    return f(aucs), f(caught)


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    tot, e = len(y), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum():
            e += m.sum() / tot * abs(y[m].mean() - p[m].mean())
    return float(e)


out = {}
Xt, test = load_set("test_openfake")
Xr, red = load_set("test_reddit")
Xv, val = load_set("val_openfake")
it, ir, iv = image_level(test, Xt @ w + b), image_level(red, Xr @ w + b), image_level(val, Xv @ w + b)

for name, df in (("test_clean", it[it.variant == "clean"]), ("test_jpeg70", it[it.variant == "jpeg70"]),
                 ("test_half_q75", it[it.variant == "half_q75"]), ("reddit", ir)):
    y, s = df.label.to_numpy(), df.logit.to_numpy()
    a, c = boot(y, s)
    out[name] = {"auc": round(float(roc_auc_score(y, s)), 4), "auc_ci": a, "caught": round(float((s[y == 1] > 0).mean()), 4), "caught_ci": c,
                 "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum())}

clean = it[it.variant == "clean"]
for src in ("gpt-image-1.5", "gpt-image-2", "nano-banana-pro", "veo-3"):
    sub = pd.concat([clean[clean.label == 0], clean[clean.src == src]])
    y, s = sub.label.to_numpy(), sub.logit.to_numpy()
    a, c = boot(y, s)
    out[src] = {"auc": round(float(roc_auc_score(y, s)), 4), "auc_ci": a, "caught": round(float((s[y == 1] > 0).mean()), 4), "caught_ci": c,
                "n_fake": int((y == 1).sum())}

# calibration error on the held-out test split (all three conditions pooled), before and after Platt scaling
y, s = it.label.to_numpy(), it.logit.to_numpy()
p_raw = 1 / (1 + np.exp(-s))
p_cal = 1 / (1 + np.exp(-(cal["a"] * s + cal["b"])))
out["ece_test"] = {"raw_sigmoid": round(ece(y, p_raw), 4), "platt": round(ece(y, p_cal), 4), "n_images": int(len(y))}
yv, sv = iv.label.to_numpy(), iv.logit.to_numpy()
out["ece_validation"] = {"raw_sigmoid": round(ece(yv, 1 / (1 + np.exp(-sv))), 4),
                         "platt": round(ece(yv, 1 / (1 + np.exp(-(cal["a"] * sv + cal["b"])))), 4)}
out["fake_share"] = {"validation": round(float(yv.mean()), 3), "test": round(float(y.mean()), 3)}
Path(__file__).with_name("analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
