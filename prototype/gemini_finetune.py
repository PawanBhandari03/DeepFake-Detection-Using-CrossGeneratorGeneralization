"""Experiment: does adding matched (real photo, Gemini/ChatGPT edit) pairs improve the v3 head on edits?

    python prototype/gemini_finetune.py --pairs <train_pairs_dir> --test <test_dir> [--out results.json]

Folder layout (same file stem = same pair):
    <pairs>/real/NN.jpg        <pairs>/gemini/NN.png        (optional: <pairs>/chatgpt/NN.png)
    <test>/real, <test>/chatgpt, <test>/gemini               (final test, never trained on)

What it does (no CLIP retraining, only the small logistic head):
  1. Encode every image like the app does (centre + 3x3 patches, JPEG q95 re-save, L2-normalised CLIP features).
  2. Leave-one-pair-out: for each pair, train v3 recipe + all OTHER pairs (weight W), score the held-out pair.
     So every pair is judged by a model that never saw it. Compared with the current v3 head on the same pairs.
  3. Final head (all pairs) is checked on the untouched test set and on the OpenFake test split.
  4. ACCEPT only if every rule below holds (fixed in advance, printed at the end):
       - leave-one-out AUC (real vs edit) is at least 0.05 above the v3 head,
       - OpenFake test AUC (clean) drops by at most 0.005,
       - real photos in the test set called AI (Trust <= 17) do not increase,
       - test-set AUC real vs edits does not drop.
Nothing in the app is changed by this script; weights are written to prototype/weights/our_head_v3g.* only if ACCEPT.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import detector as D  # noqa: E402
from train_v3 import WEIGHTS, RESULTS, load_set, image_level  # noqa: E402

KINDS = ("real", "gemini", "chatgpt")
WEIGHT_GRID = (10.0, 30.0)
LOO_BASE_PATCHES = 150_000          # base-data subsample used only inside leave-one-out (speed)


def encode_folder(det, root: Path, cache: Path, max_pairs: int | None = None) -> dict:
    """{kind: {stem: (10, 768) L2-normalised features}} with a disk cache."""
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["data"].item()
    head = det.heads["ours_v3"]
    out = {}
    for kind in KINDS:
        files = sorted(glob.glob(str(root / kind / "*.*")))
        if max_pairs:
            files = files[:max_pairs]
        if not files:
            continue
        out[kind] = {}
        for f in files:
            img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            img = D.ensure_min_size(img)
            boxes = D.grid_boxes(img.width, img.height, 3)
            patches = [head.prepare(p) for p in [D.center_crop(img)] + [img.crop(b) for b in boxes]]
            x = det.encoder.encode(patches).astype(np.float32)
            x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
            out[kind][Path(f).stem] = x
            print("  encoded", kind, Path(f).name, flush=True)
    np.savez(cache, data=np.array(out, dtype=object))
    return out


def fit(X, y, C, sw=None):
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    clf.fit(X, y, sample_weight=sw)
    return clf.coef_[0].astype(np.float32), float(clf.intercept_[0])


def img_logit(feats, w, b):
    return float((feats @ w + b).mean())


def pair_table(data):
    """list of (stem, kind, feats, label) for all images."""
    rows = []
    for kind, m in data.items():
        for stem, f in m.items():
            rows.append((stem, kind, f, 0 if kind == "real" else 1))
    return rows


def eval_set(data, w, b):
    real = [img_logit(f, w, b) for f in data["real"].values()]
    edits = [img_logit(f, w, b) for k in ("gemini", "chatgpt") if k in data for f in data[k].values()]
    auc = roc_auc_score([0] * len(real) + [1] * len(edits), real + edits)
    return {"auc": round(float(auc), 4), "real_called_ai": int(sum(l > 0 for l in real)), "n_real": len(real),
            "edit_called_ai": int(sum(l > 0 for l in edits)), "n_edit": len(edits)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default=str(RESULTS / "gemini_finetune.json"))
    ap.add_argument("--cache", default=str(Path(__file__).parent / "_cache"))
    ap.add_argument("--max-pairs", type=int, default=None, help="use only the first N pairs of --pairs")
    ap.add_argument("--no-save", action="store_true", help="dry run: never write weights")
    a = ap.parse_args()
    cache = Path(a.cache)
    cache.mkdir(exist_ok=True)

    det = D.Detector()
    train = encode_folder(det, Path(a.pairs), cache / (Path(a.pairs).name + f"_{a.max_pairs}_feats.npz"), a.max_pairs)
    test = encode_folder(det, Path(a.test), cache / (Path(a.test).name + "_feats.npz"))
    stems = sorted(set(train["real"]) & set(train.get("gemini", {})))
    print(f"{len(stems)} matched Gemini pairs for training/LOO", flush=True)
    if len(stems) < 5:
        sys.exit("need at least 5 matched pairs")

    # base data = exactly the v3 recipe (OpenFake train + edit patches)
    Xtr, tr = load_set("train_openfake")
    edits = [load_set(n) for n in ("edit_diffseg", "edit_tinyaudits")]
    import pandas as pd
    ed = pd.concat([d for _, d in edits], ignore_index=True)
    Xe = np.concatenate([x for x, _ in edits])
    rng = np.random.default_rng(0)
    ids = np.array(sorted(ed.img_id.unique()))
    held = set(ids[rng.permutation(len(ids))[: int(0.15 * len(ids))]])      # same hold-out as train_v3.py
    ed_train = ed[~ed.img_id.isin(held) & ed.label.isin([0, 1])]
    Xb = np.concatenate([Xtr, Xe[ed_train.index.to_numpy()]])
    yb = np.concatenate([tr.label.to_numpy(), ed_train.label.to_numpy()])
    C = json.loads((RESULTS / "v3_results.json").read_text())["v3"]["C"]
    print(f"base patches {len(yb)}, C={C}", flush=True)

    h = np.load(WEIGHTS / "our_head_v3.npz")
    w0, b0 = h["w"], float(h["b"])

    def new_rows(keep):
        X, y = [], []
        for s in keep:
            for kind in ("real", "gemini", "chatgpt"):
                if s in train.get(kind, {}):
                    X.append(train[kind][s]); y += [0 if kind == "real" else 1] * 10
        return np.concatenate(X), np.array(y)

    # ---- leave-one-pair-out on a base subsample (speed) ----
    sub = rng.permutation(len(yb))[:LOO_BASE_PATCHES]
    Xs, ys = Xb[sub], yb[sub]
    loo = {}
    # leave-one-pair-out for small sets, 5-fold grouped by pair for larger sets (every pair is still held out once)
    folds = [[s] for s in stems] if len(stems) <= 30 else [list(f) for f in np.array_split(np.array(stems), 5)]
    for W in WEIGHT_GRID:
        lg = {}
        for fold in folds:
            Xn, yn = new_rows([t for t in stems if t not in fold])
            w, b = fit(np.concatenate([Xs, Xn]), np.concatenate([ys, yn]), C,
                       np.concatenate([np.ones(len(ys)), np.full(len(yn), W)]))
            for s in fold:
                lg[s] = (img_logit(train["real"][s], w, b), img_logit(train["gemini"][s], w, b))
        real_l = [lg[s][0] for s in stems]; edit_l = [lg[s][1] for s in stems]
        auc = roc_auc_score([0] * len(stems) + [1] * len(stems), real_l + edit_l)
        wins = int(sum(e > r for r, e in zip(real_l, edit_l)))
        loo[W] = {"auc": round(float(auc), 4), "pairs_edit_more_ai": wins}
        print(f"LOO W={W}: AUC {auc:.3f}, edit more AI-like in {wins}/{len(stems)} pairs", flush=True)
    base_real = [img_logit(train["real"][s], w0, b0) for s in stems]
    base_edit = [img_logit(train["gemini"][s], w0, b0) for s in stems]
    base_auc = roc_auc_score([0] * len(stems) + [1] * len(stems), base_real + base_edit)
    base_wins = int(sum(e > r for r, e in zip(base_real, base_edit)))
    print(f"v3 (no new pairs) on the same pairs: AUC {base_auc:.3f}, {base_wins}/{len(stems)}", flush=True)
    bestW = max(loo, key=lambda k: loo[k]["auc"])

    # ---- final head on all pairs, full base ----
    Xn, yn = new_rows(stems)
    w1, b1 = fit(np.concatenate([Xb, Xn]), np.concatenate([yb, yn]), C,
                 np.concatenate([np.ones(len(yb)), np.full(len(yn), bestW)]))
    Xt, te = load_set("test_openfake")
    clean = (te.variant == "clean").to_numpy()
    def of_auc(w, b):
        g = image_level(te[clean].reset_index(drop=True), (Xt[clean] @ w + b))
        return float(roc_auc_score(g.label, g.logit))
    res = {
        "n_pairs": len(stems), "weight": bestW, "loo": loo,
        "v3_on_same_pairs": {"auc": round(float(base_auc), 4), "pairs_edit_more_ai": base_wins},
        "openfake_clean_auc": {"v3": round(of_auc(w0, b0), 4), "new": round(of_auc(w1, b1), 4)},
        "test_set": {"v3": eval_set(test, w0, b0), "new": eval_set(test, w1, b1)},
    }
    rules = {
        "loo_auc_gain>=0.05": loo[bestW]["auc"] - base_auc >= 0.05,
        "openfake_drop<=0.005": res["openfake_clean_auc"]["v3"] - res["openfake_clean_auc"]["new"] <= 0.005,
        "test_real_called_ai_not_higher": res["test_set"]["new"]["real_called_ai"] <= res["test_set"]["v3"]["real_called_ai"],
        "test_auc_not_lower": res["test_set"]["new"]["auc"] >= res["test_set"]["v3"]["auc"],
    }
    res["rules"] = rules
    res["decision"] = "ACCEPT" if all(rules.values()) else "REJECT (keep v3)"
    print(json.dumps(res, indent=1))
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    if res["decision"] == "ACCEPT" and not a.no_save:
        np.savez(WEIGHTS / "our_head_v3g.npz", w=w1, b=np.float32(b1), freq_mean=np.zeros(64, np.float32), freq_std=np.ones(64, np.float32))
        print("saved candidate weights our_head_v3g.npz (app still uses v3; switching needs recalibration)")


if __name__ == "__main__":
    main()
