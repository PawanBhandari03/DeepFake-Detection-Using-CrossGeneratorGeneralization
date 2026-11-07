"""Compare the EfficientNet-B0 baseline with our model v3 on the same held-out OpenFake test images.

    python prototype/compare_effnet.py

Needs kaggle/effnet/out/preds.csv (from the Kaggle job kaggle/effnet). Our scores come from the saved v3 head applied to the
test features, image score = mean of the ten patch logits (exactly what the app does). Writes prototype/results/effnet_comparison.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import region_eval as R

PREDS = R.ROOT / "kaggle" / "effnet" / "out" / "preds.csv"


def summarise(df, col):
    out = {}
    for var, g in df.groupby("variant"):
        real, fake = g[g.label == 0], g[g.label == 1]
        e = {"n_real": int(len(real)), "n_fake": int(len(fake)), "auc": round(float(roc_auc_score(g.label, g[col])), 4),
             "real_correct": round(float((real[col] < 0).mean()), 4), "fake_caught": round(float((fake[col] > 0).mean()), 4),
             "per_generator": {}}
        for src, s in fake.groupby("src"):
            if len(s) >= 20:
                both = pd.concat([real, s])
                e["per_generator"][src] = {"n": int(len(s)), "auc": round(float(roc_auc_score(both.label, both[col])), 4),
                                           "caught": round(float((s[col] > 0).mean()), 4)}
        out[var] = e
    return out


def main():
    eff = pd.read_csv(PREDS).rename(columns={"logit": "eff"})
    ours = R.image_scores(R.load("test_openfake"))[["img_id", "variant", "mean"]].rename(columns={"mean": "ours"})
    df = eff.merge(ours, on=["img_id", "variant"], how="inner")
    print(f"{df.img_id.nunique()} images in common, variants {sorted(df.variant.unique())}")
    res = {"n_images": int(df.img_id.nunique()), "efficientnet_b0": summarise(df, "eff"), "ours_v3": summarise(df, "ours")}
    unseen = ["gpt-image-1.5", "nano-banana-pro", "midjourney-7", "flux.2-klein-9b", "z-image-turbo", "illustrious", "veo-3"]
    for key in ("efficientnet_b0", "ours_v3"):
        gens = res[key]["clean"]["per_generator"]
        vals = [gens[g]["auc"] for g in unseen if g in gens]
        res[key]["clean_average_auc_over_named_generators"] = round(float(np.mean(vals)), 4) if vals else None
    (R.ROOT / "prototype" / "results" / "effnet_comparison.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for var in ("clean", "jpeg70", "half_q75"):
        a, b = res["efficientnet_b0"][var], res["ours_v3"][var]
        print(f"{var:9s} EfficientNet AUC {a['auc']:.3f} caught {a['fake_caught']*100:5.1f}% real ok {a['real_correct']*100:5.1f}%   |   "
              f"ours AUC {b['auc']:.3f} caught {b['fake_caught']*100:5.1f}% real ok {b['real_correct']*100:5.1f}%")
    print("\nper generator (clean AUC): generator, EfficientNet, ours")
    for g, v in sorted(res["ours_v3"]["clean"]["per_generator"].items(), key=lambda kv: -kv[1]["n"])[:14]:
        print(f"  {g:22s} {res['efficientnet_b0']['clean']['per_generator'].get(g, {}).get('auc', float('nan')):.3f}  {v['auc']:.3f}  (n={v['n']})")
    print("named-generator average:", res["efficientnet_b0"]["clean_average_auc_over_named_generators"], res["ours_v3"]["clean_average_auc_over_named_generators"])


if __name__ == "__main__":
    main()
