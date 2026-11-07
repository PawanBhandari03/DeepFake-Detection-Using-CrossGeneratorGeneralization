"""Generate kaggle/robust/robust.py from kaggle/extract/extract.py.

The robustness run scores the SAME held-out test images as before (same seeds, same files) under more degradations:
JPEG 90 / 50, Gaussian blur, half size, and a screenshot-like re-save. Nothing is trained.

    python kaggle/robust/make_robust.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE.parent / "extract" / "extract.py").read_text(encoding="utf-8")


def swap(old, new):
    global src
    assert src.count(old) == 1, old[:60]
    src = src.replace(old, new)


# 1. new degradation conditions
start = src.index("def eval_variant(img, name):")
end = src.index("def app_boxes(img):")
src = src[:start] + '''def eval_variant(img, name):
    if name == "clean":
        return img
    if name.startswith("jpeg"):
        return jpeg_rt(img, int(name[4:]))
    if name == "half":
        return img.resize((max(32, img.width // 2), max(32, img.height // 2)), Image.BICUBIC)
    if name == "blur":
        return img.filter(ImageFilter.GaussianBlur(1.5))
    if name == "screenshot":      # screenshot-like re-save: scale to 75% then JPEG quality 85
        return jpeg_rt(img.resize((max(32, int(img.width * 0.75)), max(32, int(img.height * 0.75))), Image.BICUBIC), 85)
    raise ValueError(name)


''' + src[end:]

# 2. only the held-out test split, with the new conditions
start = src.index("EVAL_VARIANTS = ")
end = src.index('if __name__ == "__main__":')
src = src[:start] + '''ROBUST_VARIANTS = ["jpeg90", "jpeg50", "blur", "half", "screenshot"]
STAGES = [
    ("test_robust", lambda: stage_openfake_eval(
        "test_robust", "datasets/ComplexDataLab/OpenFake/core/test-*.parquet", 13, 40,
        lambda m: str(m) in ("nano-banana-pro", "nano-banana", "veo-3", "sora-2"), 20, 250, 1200, ROBUST_VARIANTS, 12)),
]

BUDGET = {"test_robust": 9000}

''' + src[end:]
(HERE / "robust.py").write_text(src, encoding="utf-8")
print("robust.py written,", len(src.splitlines()), "lines")
