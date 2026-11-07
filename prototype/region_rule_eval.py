"""Cost and benefit of the 'region second look': an image the mean rule calls Authentic is sent to Needs Human Review
when its mean+max score passes the threshold chosen on validation real images (3 % false alarms).

    python prototype/region_rule_eval.py
The rule can only move a verdict from Authentic to Needs Human Review, never the other way.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import region_eval as R

meta = json.loads((R.ROOT / "prototype" / "weights" / "our_head_v3.json").read_text(encoding="utf-8"))
cal = meta["calibration"]
authentic_trust = round(100 * (1 - cal["p_real"]))


def trust(logit_mean):
    p = 1 / (1 + np.exp(-(cal["a"] * logit_mean + cal["b"])))
    return np.round(100 * (1 - p))


def table(df, thr):
    out = {}
    for var, g in df.groupby("variant"):
        for label, name in ((0, "real"), (1, "ai"), ("edited", "edited"), ("orig", "orig")):
            sel = g[g.label == label] if label in (0, 1) else g[g.kind == label]
            if len(sel) < 5:
                continue
            auth = trust(sel["mean"]) >= authentic_trust
            flip = auth & (sel["mean+max"] > thr)
            out[f"{var}/{name}"] = {"n": int(len(sel)), "authentic_before": round(float(auth.mean()), 3),
                                     "sent_to_review": round(float(flip.mean()), 3),
                                     "authentic_after": round(float((auth & ~flip).mean()), 3)}
    return out


val = R.image_scores(R.load("val_openfake"))
import sys as _s
FLAG = float(_s.argv[1]) if len(_s.argv) > 1 else 0.03          # share of validation real photos (called Authentic) to flag
vr = val[(val.variant == "clean") & (val.label == 0)]
vr = vr[trust(vr["mean"]) >= authentic_trust]
thr = float(np.quantile(vr["mean+max"], 1 - FLAG))
res = {"flag_share_of_authentic_validation_real": FLAG, "threshold_mean_plus_max_logit": thr, "authentic_at_or_above": authentic_trust}
res["test"] = table(R.image_scores(R.load("test_openfake")), thr)
res["reddit"] = table(R.image_scores(R.load("test_reddit")), thr)
d = np.load(R.FEATS / "editgrid.npz")
df = pd.DataFrame({k: d[k] for k in d.files if k != "feats"})
X = d["feats"].astype(np.float32)
X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
df["logit"] = X @ R.W + R.B
df["label"] = (df.kind == "edited").astype(int)
res["edited_heldout"] = table(R.image_scores(df), thr)
(R.ROOT / "prototype" / "results" / "region_rule.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print("threshold", round(thr, 3), "authentic at or above", authentic_trust)
for k in ("test", "reddit", "edited_heldout"):
    print("\n" + k)
    for name, v in res[k].items():
        print(f"  {name:18s} n={v['n']:5d} authentic {v['authentic_before']*100:5.1f}% -> {v['authentic_after']*100:5.1f}%  (sent to review {v['sent_to_review']*100:4.1f}%)")
