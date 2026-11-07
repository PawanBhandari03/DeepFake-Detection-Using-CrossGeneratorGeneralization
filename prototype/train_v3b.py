"""Train v3b: v3 plus training views with strong resize-only, blur-only and mixed degradations.

    python prototype/train_v3b.py

Inputs (features from Kaggle, in data/features/v3/):
  train_openfake, train_hard (new), edit_diffseg, edit_tinyaudits    -> training
  val_openfake, val_hard                                              -> penalty strength, calibration, verdict bands
  test_openfake, test_robust, test_unseen                             -> evaluation only (nothing is tuned on it)
The test split has three conditions (scale35, blur25, mix) whose parameters differ from the training degradations.
Outputs: prototype/weights/our_head_v3b.(npz|json), prototype/results/v3b_results.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from train_v3 import RESULTS, WEIGHTS, fit, image_level, load_set, summarise  # noqa: E402


def stack(names):
    parts = [load_set(n) for n in names]
    return np.concatenate([p[0] for p in parts]), pd.concat([p[1] for p in parts], ignore_index=True)


def head(key):
    if key == "v2":
        h = np.load(WEIGHTS / "our_head_v2.npz")
        return h["w"][:768], float(h["b"])
    h = np.load(WEIGHTS / f"our_head_{key}.npz")
    return h["w"], float(h["b"])


def main():
    Xtr, tr = load_set("train_openfake")
    Xh, th = load_set("train_hard")
    edits = [load_set(n) for n in ("edit_diffseg", "edit_tinyaudits")]
    Xe = np.concatenate([x for x, _ in edits])
    ed = pd.concat([d for _, d in edits], ignore_index=True)
    rng = np.random.default_rng(0)                                   # same held-out edited images as train_v3.py
    ids = np.array(sorted(ed.img_id.unique()))
    held = set(ids[rng.permutation(len(ids))[: int(0.15 * len(ids))]])
    keep = (~ed.img_id.isin(held) & ed.label.isin([0, 1])).to_numpy()
    X = np.concatenate([Xtr, Xh, Xe[keep]])
    y = np.concatenate([tr.label.to_numpy(), th.label.to_numpy(), ed.label[keep].to_numpy()])
    print(f"training patches: {len(y)} ({len(Xtr)} OpenFake + {len(Xh)} hard + {int(keep.sum())} edited); fake share {np.mean(y == 1):.3f}", flush=True)

    Xv, val = stack(["val_openfake", "val_hard"])
    Xt, test = stack(["test_openfake", "test_robust", "test_unseen"])

    best = None
    for C in (0.3, 1.0, 3.0, 10.0):
        w, b = fit(X, y, C)
        s = summarise(image_level(val, Xv @ w + b))
        score = float(np.mean([v["auc"] for v in s.values()]))
        # also reward correct real photographs across conditions, so that false alarms are not traded for catches
        real_ok = float(np.mean([v["real_correct"] for v in s.values()]))
        print(f"  C={C}: validation mean AUC {score:.4f}, mean real-correct {real_ok:.3f}", flush=True)
        crit = score + 0.5 * real_ok
        if best is None or crit > best[0]:
            best = (crit, C, w, b)
    _, C, w, b = best
    print("chosen C =", C, flush=True)

    # calibration and verdict bands on ALL validation conditions pooled
    iv = image_level(val, Xv @ w + b)
    platt = LogisticRegression(C=1e6).fit(iv[["logit"]], iv.label)
    a, c = float(platt.coef_[0][0]), float(platt.intercept_[0])
    prob = lambda df: 1 / (1 + np.exp(-(a * df.logit + c)))
    iv["p"] = prob(iv)
    p_ai = max(float(np.quantile(iv[iv.label == 0].p, 1 - 0.03)), 0.5)
    p_real = min(float(np.quantile(iv[iv.label == 1].p, 0.10)), 0.5)
    t_auth, t_ai = round(100 * (1 - p_real)), round(100 * (1 - p_ai))
    print(f"Platt a={a:.3f} c={c:.3f}; Trust >= {t_auth} authentic, <= {t_ai} AI-generated", flush=True)

    names = {"v2": "earlier v2 head", "v3": "v3", "v3b": "v3b"}
    heads = {"v2": head("v2"), "v3": head("v3"), "v3b": (w, b)}
    res = {"C": C, "platt": {"a": a, "b": c}, "p_ai": p_ai, "p_real": p_real, "trust_authentic_at_or_above": t_auth,
           "trust_ai_at_or_below": t_ai, "n_train_patches": int(len(y)), "test": {}, "bands_on_test": {}}
    it = None
    for key, (wk, bk) in heads.items():
        img = image_level(test, Xt @ wk + bk)
        res["test"][key] = summarise(img)
        if key == "v3b":
            it = img.copy()
    it["p"] = prob(it)
    for var, g in it.groupby("variant"):
        r, f = g[g.label == 0], g[g.label == 1]
        res["bands_on_test"][var] = {
            "real": {"authentic": float((r.p <= p_real).mean()), "review": float(((r.p > p_real) & (r.p < p_ai)).mean()), "ai": float((r.p >= p_ai).mean())},
            "fake": {"authentic": float((f.p <= p_real).mean()), "review": float(((f.p > p_real) & (f.p < p_ai)).mean()), "ai": float((f.p >= p_ai).mean())}}
    # real photographs labelled real, by source and condition (v3 vs v3b)
    by_src = {}
    for key in ("v3", "v3b"):
        wk, bk = heads[key]
        im = image_level(test, Xt @ wk + bk)
        real = im[im.label == 0].assign(ok=lambda d: d.logit < 0)
        by_src[key] = {s: {v: round(float(x), 3) for v, x in g.groupby("variant").ok.mean().items()} for s, g in real.groupby("src")}
    res["real_ok_by_source"] = by_src

    np.savez(WEIGHTS / "our_head_v3b.npz", w=w, b=np.float32(b), freq_mean=np.zeros(64, np.float32), freq_std=np.ones(64, np.float32))
    meta = {"name": "Our model v3b (v3 + strong resize/blur training)",
            "trained_on": "the v3 training data plus about 12,000 further OpenFake images with strong resize-only, blur-only and mixed degradations",
            "normalize_features": True, "use_frequency": False, "jpeg_quality": 95, "regularization_C": C, "variant": "v3b",
            "calibration": {"a": a, "b": c, "p_ai": p_ai, "p_real": p_real}, "held_out": []}
    (WEIGHTS / "our_head_v3b.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (RESULTS / "v3b_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("saved weights and results")


if __name__ == "__main__":
    main()
