"""Writes the image-id lists that the editgrid and effnet Kaggle jobs embed (same splits as prototype/train_v3.py)."""
import base64, glob, json, zlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
F = ROOT / "data" / "features" / "v3"


def ids_labels(prefix):
    out = {}
    for f in sorted(glob.glob(str(F / f"{prefix}_[0-9][0-9][0-9].npz"))):
        d = np.load(f)
        for i, l, s in zip(d["img_id"].tolist(), d["label"].tolist(), d["src"].tolist()):
            out.setdefault(i, (int(l), s))
    return out


def pack(obj):
    return base64.b64encode(zlib.compress(json.dumps(obj, separators=(",", ":")).encode(), 9)).decode()


# edited images held out of v3 training: the same 15% rule as train_v3.py (seed 0 over sorted image ids)
ed_ids = set()
for name in ("edit_diffseg", "edit_tinyaudits"):
    for f in sorted(glob.glob(str(F / f"{name}_[0-9][0-9][0-9].npz"))):
        ed_ids |= set(np.load(f)["img_id"].tolist())
ids = np.array(sorted(ed_ids))
rng = np.random.default_rng(0)
held = sorted(ids[rng.permutation(len(ids))[: int(0.15 * len(ids))]].tolist())
print("edited images:", len(ids), "held out:", len(held), "diffseg:", sum("/" not in h for h in held))
(ROOT / "kaggle" / "editgrid" / "held_ids.txt").write_text(pack(held))

train = ids_labels("train_openfake")
test = ids_labels("test_openfake")
print("train ids", len(train), "test ids", len(test))
(ROOT / "kaggle" / "effnet" / "ids.txt").write_text(pack({"train": sorted(train), "test": sorted(test)}))
