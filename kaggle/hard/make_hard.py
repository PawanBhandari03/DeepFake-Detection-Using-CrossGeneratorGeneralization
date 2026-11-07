"""Generate the two Kaggle scripts of the "hard degradation" round from kaggle/extract/extract.py.

    python kaggle/hard/make_hard.py     ->  kaggle/hard_train/hard_train.py, kaggle/hard_eval/hard_eval.py

hard_train.py : about 12,000 NEW OpenFake training images, each with three strongly degraded views
                (resize only, blur only, mixed), 2 patches per view. Used to retrain the head (v3b).
hard_eval.py  : (a) the validation images under half size / blur / screenshot-like / JPEG 50, so that the verdict bands
                can be tuned on every kind of degradation, and (b) the test images under three NEW conditions
                whose parameters differ from the training ones (scale 0.35, blur 2.5, scale 0.6 + blur + JPEG 60).
Nothing is trained on the test split.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = (HERE.parent / "extract" / "extract.py").read_text(encoding="utf-8")


def swap(src, old, new):
    assert src.count(old) == 1, old[:70]
    return src.replace(old, new)


DEGRADE_HARD = '''def degrade_hard(img, rng, kind):
    """Stronger and more varied damage than degrade(): resize only, blur only, or a mixture."""
    filters = [Image.BILINEAR, Image.BICUBIC, Image.LANCZOS, Image.BOX]
    desc = []
    if kind == "resize" or (kind == "mixed" and rng.random() < 0.7):
        s = rng.uniform(0.25, 0.9) if kind == "resize" else rng.uniform(0.4, 1.0)
        img = img.resize((max(32, int(img.width * s)), max(32, int(img.height * s))), filters[rng.integers(len(filters))])
        desc.append(f"s{s:.2f}")
    if kind == "blur" or (kind == "mixed" and rng.random() < 0.5):
        sigma = rng.uniform(0.8, 3.0) if kind == "blur" else rng.uniform(0.5, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(sigma))
        desc.append(f"b{sigma:.1f}")
    if kind == "mixed" and rng.random() < 0.7:
        q = int(rng.integers(40, 96))
        img = jpeg_rt(img, q)
        desc.append(f"q{q}")
    return img, kind + ":" + ("_".join(desc) or "none")


'''

# ------------------------------------------------------------------ hard_train.py
t = SRC
t = swap(t, "def eval_variant(img, name):", DEGRADE_HARD + "def eval_variant(img, name):")
t = swap(t, '                if mode == "train":', '''                if mode == "hard":
                    for kind in ("resize", "blur", "mixed"):
                        v, desc = degrade_hard(img, rng, kind)
                        v = ensure_min(v)
                        for box in rand_boxes(v, rng, 2):
                            patches.append(to_patch(v, box))
                            meta.append(dict(img_id=iid, src=mod, label=lab, kind="full", variant="hard:" + desc,
                                             x=box[0], y=box[1], frac=np.nan, slot=-1))
                elif mode == "train":''')
start, end = t.index("EVAL_VARIANTS = "), t.index('if __name__ == "__main__":')
t = t[:start] + '''def stage_train_hard():
    rng = np.random.default_rng(SEED + 21)
    rows = scan_split("datasets/ComplexDataLab/OpenFake/core/train-*.parquet", n_of(60), rng)
    by, rand, _ = pick_rgs(rows, n_of(170), None, 0, rng)
    tasks = [(k, by[k], int(rng.integers(1 << 30))) for k in rand]
    cnt = Counter(("real" if lab == "real" else mod) for _, rowsk, _ in tasks for _, lab, mod in rowsk)
    print(f"[train_hard] {len(tasks)} row groups, {sum(cnt.values())} images; top {cnt.most_common(8)}", flush=True)
    json.dump(cnt, open(f"{OUT}/train_hard_plan.json", "w"))
    run(tasks, openfake_worker_factory("hard"), Sink("train_hard", 12000), label="train_hard")


STAGES = [("train_hard", stage_train_hard)]
BUDGET = {"train_hard": 14400}

''' + t[end:]
(HERE.parent / "hard_train").mkdir(exist_ok=True)
(HERE.parent / "hard_train" / "hard_train.py").write_text(t, encoding="utf-8")

# ------------------------------------------------------------------ hard_eval.py
e = SRC
start, end = e.index("def eval_variant(img, name):"), e.index("def app_boxes(img):")
e = e[:start] + '''def eval_variant(img, name):
    if name == "clean":
        return img
    if name.startswith("jpeg"):
        return jpeg_rt(img, int(name[4:]))
    if name == "half":
        return img.resize((max(32, img.width // 2), max(32, img.height // 2)), Image.BICUBIC)
    if name == "blur":
        return img.filter(ImageFilter.GaussianBlur(1.5))
    if name == "screenshot":
        return jpeg_rt(img.resize((max(32, int(img.width * 0.75)), max(32, int(img.height * 0.75))), Image.BICUBIC), 85)
    # three conditions whose parameters differ from the training degradations and from the conditions above
    if name == "scale35":
        return img.resize((max(32, int(img.width * 0.35)), max(32, int(img.height * 0.35))), Image.LANCZOS)
    if name == "blur25":
        return img.filter(ImageFilter.GaussianBlur(2.5))
    if name == "mix":
        x = img.resize((max(32, int(img.width * 0.6)), max(32, int(img.height * 0.6))), Image.BICUBIC).filter(ImageFilter.GaussianBlur(1.0))
        return jpeg_rt(x, 60)
    raise ValueError(name)


''' + e[end:]
start, end = e.index("EVAL_VARIANTS = "), e.index('if __name__ == "__main__":')
e = e[:start] + '''STAGES = [
    ("val_hard", lambda: stage_openfake_eval(
        "val_hard", "datasets/ComplexDataLab/OpenFake/core/validation-*.parquet", 15, 20, None, 0, 150, 600,
        ["half", "blur", "screenshot", "jpeg50"], 11)),
    ("test_unseen", lambda: stage_openfake_eval(
        "test_unseen", "datasets/ComplexDataLab/OpenFake/core/test-*.parquet", 13, 40,
        lambda m: str(m) in ("nano-banana-pro", "nano-banana", "veo-3", "sora-2"), 20, 250, 1200,
        ["scale35", "blur25", "mix"], 12)),
]
BUDGET = {"val_hard": 5400, "test_unseen": 7200}

''' + e[end:]
(HERE.parent / "hard_eval").mkdir(exist_ok=True)
(HERE.parent / "hard_eval" / "hard_eval.py").write_text(e, encoding="utf-8")

for name, code in (("hard_train", "hard_train.py"), ("hard_eval", "hard_eval.py")):
    meta = {"id": f"pawan0033/deepfake-v3-{name.replace('_', '-')}", "title": f"deepfake-v3-{name.replace('_', '-')}", "code_file": code,
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
    (HERE.parent / name / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("written: kaggle/hard_train, kaggle/hard_eval")
