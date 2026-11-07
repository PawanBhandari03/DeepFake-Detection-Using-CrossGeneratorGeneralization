"""Does judging an image by its most suspicious patches (instead of the average) help? A held-out test.

    python prototype/region_eval.py

Image score options, all from the same 10 patch logits of model v3 (centre + 3x3 grid):
  mean      the current rule (mean of the 10 patch logits)
  max       the single most suspicious patch
  top2/top3 mean of the 2 / 3 most suspicious patches
  mean+max  average of the mean rule and the max rule (a compromise)
For every option the "called AI" threshold is chosen on the VALIDATION real images so that 3 % of them are
wrongly called AI (the same rule as the v3 bands, Entry 11). Nothing is tuned on the test images.
Reported on the held-out test images: % of real photos called AI (should stay near 3 %) and % of AI images
called AI, per variant; and, if data/features/v3/editgrid.npz exists (kaggle/editgrid), the same for edited
images held out of training, where the edit may cover only part of the picture.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
FEATS = ROOT / "data" / "features" / "v3"
HEAD = np.load(ROOT / "prototype" / "weights" / "our_head_v3.npz")
W, B = HEAD["w"].astype(np.float32), float(HEAD["b"])
FA = 0.03


def load(name):
    X, rows = [], []
    for f in sorted(glob.glob(str(FEATS / f"{name}_[0-9][0-9][0-9].npz"))):
        d = np.load(f)
        X.append(d["feats"])
        rows.append(pd.DataFrame({k: d[k] for k in d.files if k != "feats"}))
    X = np.concatenate(X).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    df = pd.concat(rows, ignore_index=True)
    df["logit"] = X @ W + B
    return df


AGG = {
    "mean": lambda v: v.mean(),
    "max": lambda v: v.max(),
    "top2": lambda v: np.sort(v)[-2:].mean(),
    "top3": lambda v: np.sort(v)[-3:].mean(),
    "mean+max": lambda v: 0.5 * (v.mean() + v.max()),
}


def image_scores(df):
    """One row per (image, variant) and each aggregation option."""
    out = []
    for (iid, var, kind), g in df[df.slot >= 0].groupby(["img_id", "variant", "kind"]):
        v = g.sort_values("slot").logit.to_numpy()
        row = {"img_id": iid, "variant": var, "src": g.src.iloc[0], "label": int(g.label.iloc[0]),
               "kind": kind, "n": len(v)}
        row.update({k: float(f(v)) for k, f in AGG.items()})
        out.append(row)
    return pd.DataFrame(out)


def main():
    val, test, red = image_scores(load("val_openfake")), image_scores(load("test_openfake")), image_scores(load("test_reddit"))
    thr = {}
    for k in AGG:                               # threshold from validation real images (clean variant)
        r = val[(val.variant == "clean") & (val.label == 0)][k]
        thr[k] = float(np.quantile(r, 1 - FA))
    res = {"threshold_from_validation": thr, "false_alarm_target": FA, "test": {}, "reddit": {}}
    for title, df in (("test", test), ("reddit", red)):
        for var, g in df.groupby("variant"):
            real, fake = g[g.label == 0], g[g.label == 1]
            res[title][var] = {k: {"real_called_AI": round(float((real[k] > thr[k]).mean()), 4),
                                   "AI_called_AI": round(float((fake[k] > thr[k]).mean()), 4),
                                   "auc": round(float(roc_auc_score(g.label, g[k])), 4)} for k in AGG}
    eg = FEATS / "editgrid.npz"
    if eg.exists():
        d = np.load(eg)
        df = pd.DataFrame({k: d[k] for k in d.files if k != "feats"})
        X = d["feats"].astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        df["logit"] = X @ W + B
        df["label"] = (df.kind == "edited").astype(int)
        e = image_scores(df)
        # share of an image's patches that overlap the edit (>= 50 % of the patch edited)
        share = df[df.kind == "edited"].assign(hit=lambda t: t.frac >= 0.5).groupby(["img_id", "variant"]).hit.mean().rename("edit_share")
        e = e.merge(share, left_on=["img_id", "variant"], right_index=True, how="left")
        res["edited"] = {}
        for var, g in e.groupby("variant"):
            ed = g[g.kind == "edited"]
            entry = {"n_edited": int(len(ed)), "n_original": int((g.kind == "orig").sum())}
            real_ref = test[(test.variant == var) & (test.label == 0)] if var in set(test.variant) else None
            for k in AGG:
                entry[k] = {"edited_called_AI": round(float((ed[k] > thr[k]).mean()), 4)}
                if (g.kind == "orig").sum() >= 5:
                    entry[k]["originals_called_AI"] = round(float((g[g.kind == "orig"][k] > thr[k]).mean()), 4)
                if real_ref is not None and len(real_ref):
                    entry[k]["auc_edited_vs_test_real"] = round(float(roc_auc_score(
                        np.r_[np.zeros(len(real_ref)), np.ones(len(ed))], np.r_[real_ref[k].to_numpy(), ed[k].to_numpy()])), 4)
                small = ed[ed.edit_share <= 0.3]            # edits covering at most 30 % of the grid
                if len(small) >= 10:
                    entry[k]["small_edit_called_AI"] = round(float((small[k] > thr[k]).mean()), 4)
                    entry["n_small_edit"] = int(len(small))
            res["edited"][var] = entry
    (ROOT / "prototype" / "results" / "region_eval.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for title in ("test", "reddit"):
        for var, tab in res[title].items():
            print(f"\n{title} / {var}")
            for k, v in tab.items():
                print(f"  {k:9s} real called AI {v['real_called_AI']*100:5.1f}%   AI called AI {v['AI_called_AI']*100:5.1f}%   AUC {v['auc']:.3f}")
    if "edited" in res:
        print("\nedited images held out of training")
        for var, e in res["edited"].items():
            print(f" {var}: {e['n_edited']} edited, {e['n_original']} originals")
            for k in AGG:
                print("  ", k, e[k])


if __name__ == "__main__":
    main()
