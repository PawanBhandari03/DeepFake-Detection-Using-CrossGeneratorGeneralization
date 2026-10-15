"""Train and evaluate the v3 heads from the features extracted on Kaggle (kaggle/extract/extract.py).

    python prototype/train_v3.py            # train 3 heads, evaluate, save weights + results

Heads (all: L2-normalised frozen-CLIP features -> logistic regression, no frequency features):
  v3         trained on everything: OpenFake train (clean + JPEG/resize/blur-augmented views, ChatGPT and
             Gemini images included) + edited-image patches (DiffSeg30k, tiny-audits)
  v3_proof   the same, but every OpenAI and Google image removed from training -> honest cross-generator test
  v3_noaug   the same as v3 but trained on clean views only -> shows what the JPEG augmentation buys

Evaluation is image-level, exactly like the app: mean of the 10 patch logits (centre + 3x3 grid).
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
FEATS = ROOT / "data" / "features" / "v3"
WEIGHTS = Path(__file__).parent / "weights"
RESULTS = Path(__file__).parent / "results"
OPENAI_KEYS = ["gpt-image", "dalle", "dall-e", "sora"]
GOOGLE_KEYS = ["nano-banana", "imagen", "gemini", "veo"]


def family(model: str) -> str:
    m = str(model).lower()
    if any(k in m for k in OPENAI_KEYS):
        return "openai"
    if any(k in m for k in GOOGLE_KEYS):
        return "google"
    return "other"


def load_set(name: str) -> tuple[np.ndarray, pd.DataFrame]:
    files = sorted(glob.glob(str(FEATS / f"{name}_[0-9][0-9][0-9].npz")))
    if not files:
        raise FileNotFoundError(f"no feature files for {name} in {FEATS}")
    X, rows = [], []
    for f in files:
        d = np.load(f)
        X.append(d["feats"])
        rows.append(pd.DataFrame({k: d[k] for k in d.files if k != "feats"}))
    X = np.concatenate(X).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X, pd.concat(rows, ignore_index=True)


def fit(X, y, C):
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    clf.fit(X, y)
    return clf.coef_[0].astype(np.float32), float(clf.intercept_[0])


def image_level(df: pd.DataFrame, logit: np.ndarray) -> pd.DataFrame:
    t = df[["img_id", "variant", "src", "label"]].copy()
    t["logit"] = logit
    return t.groupby(["img_id", "variant"], as_index=False).agg(src=("src", "first"), label=("label", "first"), logit=("logit", "mean"))


def summarise(img: pd.DataFrame) -> dict:
    """AUC / % caught / % real correct, overall and per generator, per image variant."""
    out = {}
    for var, g in img.groupby("variant"):
        real, fake = g[g.label == 0], g[g.label == 1]
        if len(real) < 5 or len(fake) < 5:
            continue
        entry = {
            "n_real": int(len(real)), "n_fake": int(len(fake)),
            "auc": round(float(roc_auc_score(g.label, g.logit)), 4),
            "real_correct": round(float((real.logit < 0).mean()), 4),
            "fake_caught": round(float((fake.logit > 0).mean()), 4),
            "per_generator": {},
        }
        for src, s in fake.groupby("src"):
            if len(s) < 8:
                continue
            both = pd.concat([real, s])
            entry["per_generator"][src] = {
                "n": int(len(s)), "auc": round(float(roc_auc_score(both.label, both.logit)), 4),
                "caught": round(float((s.logit > 0).mean()), 4)}
        out[var] = entry
    return out


def edit_eval(X, df, w, b, held_ids) -> dict:
    """Patch-level results on held-out edited images (never used in training)."""
    m = df.img_id.isin(held_ids).to_numpy()
    d, logit = df[m], X[m] @ w + b
    res = {}
    for var_name, vm in (("clean", d.variant == "clean"), ("augmented", d.variant != "clean")):
        s = d[vm.to_numpy()].assign(logit=logit[vm.to_numpy()])
        pos, orig = s[s.kind == "edit"], s[s.kind == "orig"]
        entry = {"edited_patches": int(len(pos)), "edited_caught": round(float((pos.logit > 0).mean()), 4) if len(pos) else None}
        if len(orig) >= 5 and len(pos) >= 5:
            both = pd.concat([pos, orig])
            entry["orig_patches"] = int(len(orig))
            entry["orig_correct"] = round(float((orig.logit < 0).mean()), 4)
            entry["auc_edit_vs_original"] = round(float(roc_auc_score(both.kind == "edit", both.logit)), 4)
        bg = s[s.kind == "bg"]
        if len(bg):
            entry["unedited_area_flagged_ai"] = round(float((bg.logit > 0).mean()), 4)   # how AI-like is the area outside the mask
        res[var_name] = entry
    return res


def main():
    RESULTS.mkdir(exist_ok=True)
    print("loading features ...", flush=True)
    Xtr, tr = load_set("train_openfake")
    Xv, val = load_set("val_openfake")
    Xt, test = load_set("test_openfake")
    Xr, red = load_set("test_reddit")
    edits = []
    for name in ("edit_diffseg", "edit_tinyaudits"):
        try:
            edits.append(load_set(name))
        except FileNotFoundError as e:
            print("WARNING:", e)
    Xe = np.concatenate([x for x, _ in edits]) if edits else np.zeros((0, 768), np.float32)
    ed = pd.concat([d for _, d in edits], ignore_index=True) if edits else pd.DataFrame(columns=["img_id", "variant", "src", "label", "kind"])
    print(f"train {len(tr)} patches ({tr.img_id.nunique()} images) | val {len(val)} | test {len(test)} | reddit {len(red)} | edits {len(ed)}", flush=True)
    tr["family"] = tr.src.map(family)
    print("train label share fake:", round(float((tr.label == 1).mean()), 3), "| families:", tr[tr.label == 1].family.value_counts().to_dict(), flush=True)

    # hold out 15% of edited images (never trained on) to test region-edit detection
    rng = np.random.default_rng(0)
    ids = np.array(sorted(ed.img_id.unique())) if len(ed) else np.array([])
    held = set(ids[rng.permutation(len(ids))[: int(0.15 * len(ids))]]) if len(ids) else set()
    ed_train = ed[~ed.img_id.isin(held) & ed.label.isin([0, 1])] if len(ed) else ed
    Xe_train = Xe[ed_train.index.to_numpy()] if len(ed) else Xe

    def training_data(mask_openfake):
        X = np.concatenate([Xtr[mask_openfake.to_numpy()], Xe_train])
        y = np.concatenate([tr.label[mask_openfake].to_numpy(), ed_train.label.to_numpy() if len(ed) else np.zeros(0)])
        return X, y

    variants = {
        "v3": tr.label.notna(),
        "v3_proof": tr.family == "other",
        "v3_noaug": tr.variant == "clean",
    }
    descriptions = {
        "v3": "Our model v3 (all generators incl. ChatGPT + Gemini, JPEG/resize/blur augmentation, edited-image patches)",
        "v3_proof": "Proof model: v3 recipe with every OpenAI and Google image removed from training",
        "v3_noaug": "Ablation: v3 recipe trained on clean images only (no compression augmentation)",
    }
    NAMES = {"v3": "Our model v3 (strongest)", "v3_proof": "Our model v3 - proof (no OpenAI/Google)",
             "v3_noaug": "Our model v3 - no compression training"}
    all_results = {}
    for key, mask in variants.items():
        X, y = training_data(mask)
        # choose C on the validation split (image-level AUC over all three image variants)
        best = None
        for C in (0.3, 1.0, 3.0, 10.0):
            w, b = fit(X, y, C)
            auc = summarise(image_level(val, Xv @ w + b))
            score = np.mean([v["auc"] for v in auc.values()])
            print(f"  [{key}] C={C}: validation mean AUC {score:.4f}", flush=True)
            if best is None or score > best[0]:
                best = (score, C, w, b)
        _, C, w, b = best
        print(f"[{key}] trained on {len(y)} patches ({int((y == 1).sum())} fake / {int((y == 0).sum())} real), C={C}", flush=True)
        res = {
            "description": descriptions[key], "C": C, "n_train_patches": int(len(y)),
            "validation": summarise(image_level(val, Xv @ w + b)),
            "test_openfake": summarise(image_level(test, Xt @ w + b)),
            "test_reddit": summarise(image_level(red, Xr @ w + b)),
        }
        if len(ed):
            res["edited_heldout"] = edit_eval(Xe, ed, w, b, held)
        all_results[key] = res
        np.savez(WEIGHTS / f"our_head_{key}.npz", w=w, b=np.float32(b), freq_mean=np.zeros(64, np.float32), freq_std=np.ones(64, np.float32))
        (WEIGHTS / f"our_head_{key}.json").write_text(json.dumps({
            "name": NAMES[key], "trained_on": descriptions[key], "normalize_features": True, "use_frequency": False,
            "jpeg_quality": 95, "regularization_C": C, "variant": key,
            "held_out": ["all OpenAI and Google models"] if key == "v3_proof" else []}, indent=1), encoding="utf-8")

    # reference: the old v2 head (its frequency part left out; frequency features had no measurable effect)
    h = np.load(WEIGHTS / "our_head_v2.npz")
    w2, b2 = h["w"][:768], float(h["b"])
    all_results["v2_reference"] = {
        "description": "Old v2 head on the same features (frequency part omitted)",
        "validation": summarise(image_level(val, Xv @ w2 + b2)),
        "test_openfake": summarise(image_level(test, Xt @ w2 + b2)),
        "test_reddit": summarise(image_level(red, Xr @ w2 + b2)),
    }
    (RESULTS / "v3_results.json").write_text(json.dumps(all_results, indent=1), encoding="utf-8")
    print("saved", RESULTS / "v3_results.json")


if __name__ == "__main__":
    sys.exit(main())
