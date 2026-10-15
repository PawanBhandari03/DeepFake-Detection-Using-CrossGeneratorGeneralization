"""Calibrate the v3 head and choose the verdict bands on VALIDATION data (never on test).

    python prototype/calibrate_v3.py

1. Image score = mean of the 10 patch logits (as in the app), for every validation image under three
   conditions (clean, JPEG 70, half size + JPEG 75).
2. Platt scaling: fit  p = sigmoid(a * logit + b)  on validation so that p behaves like a probability.
3. Verdict bands on the calibrated probability p:
     Likely AI-generated  when p >= P_AI   (P_AI chosen so that at most 3% of real validation images are called AI)
     Likely Authentic     when p <= P_REAL (P_REAL chosen so that at most 10% of AI validation images are called real)
     in between           -> Needs Human Review
4. The same bands are then checked on the held-out TEST split (honest numbers).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from train_v3 import FEATS, WEIGHTS, RESULTS, image_level, load_set  # noqa: E402

TARGET_REAL_AS_AI = 0.03
TARGET_AI_AS_REAL = 0.10


def main():
    h = np.load(WEIGHTS / "our_head_v3.npz")
    w, b = h["w"], float(h["b"])
    Xv, val = load_set("val_openfake")
    Xt, test = load_set("test_openfake")
    Xr, red = load_set("test_reddit")
    iv, it, ir = (image_level(d, X @ w + b) for d, X in ((val, Xv), (test, Xt), (red, Xr)))

    platt = LogisticRegression(C=1e6).fit(iv[["logit"]], iv.label)
    a, c = float(platt.coef_[0][0]), float(platt.intercept_[0])
    prob = lambda df: 1 / (1 + np.exp(-(a * df.logit + c)))
    iv["p"], it["p"], ir["p"] = prob(iv), prob(it), prob(ir)

    real, fake = iv[iv.label == 0], iv[iv.label == 1]
    p_ai = float(np.quantile(real.p, 1 - TARGET_REAL_AS_AI))
    p_real = float(np.quantile(fake.p, TARGET_AI_AS_REAL))
    p_ai, p_real = max(p_ai, 0.5), min(p_real, 0.5)       # keep the bands sensible around 0.5
    print(f"Platt: p = sigmoid({a:.3f} * logit + {c:.3f})  | P_AI = {p_ai:.3f}  P_REAL = {p_real:.3f}")
    trust_authentic, trust_ai = round(100 * (1 - p_real)), round(100 * (1 - p_ai))
    print(f"-> Trust Score >= {trust_authentic}: Likely Authentic | <= {trust_ai}: Likely AI-generated | between: Needs Human Review")

    def report(df, name):
        out = {}
        for var, g in df.groupby("variant"):
            r, f = g[g.label == 0], g[g.label == 1]
            res = {
                "real": {"authentic": float((r.p <= p_real).mean()), "review": float(((r.p > p_real) & (r.p < p_ai)).mean()), "ai": float((r.p >= p_ai).mean())},
                "fake": {"authentic": float((f.p <= p_real).mean()), "review": float(((f.p > p_real) & (f.p < p_ai)).mean()), "ai": float((f.p >= p_ai).mean())},
                "n_real": int(len(r)), "n_fake": int(len(f)),
            }
            out[var] = res
            print(f"[{name}/{var}] REAL photos: authentic {res['real']['authentic']:.1%} review {res['real']['review']:.1%} AI {res['real']['ai']:.1%}"
                  f" | AI images: authentic {res['fake']['authentic']:.1%} review {res['fake']['review']:.1%} AI {res['fake']['ai']:.1%}")
        return out

    results = {"platt_a": a, "platt_b": c, "p_ai": p_ai, "p_real": p_real, "trust_authentic_at_or_above": trust_authentic,
               "trust_ai_at_or_below": trust_ai, "targets": {"real_called_ai": TARGET_REAL_AS_AI, "ai_called_real": TARGET_AI_AS_REAL},
               "validation": report(iv, "validation"), "test_openfake": report(it, "test"), "test_reddit": report(ir, "reddit")}
    (RESULTS / "v3_calibration.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    # store the calibration next to the head so the app can use it
    meta_path = WEIGHTS / "our_head_v3.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["calibration"] = {"a": a, "b": c, "p_ai": p_ai, "p_real": p_real}
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
