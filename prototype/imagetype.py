"""Edge-case notices: tell the user when an image is outside what the detector was built for.

The detector was trained on camera-style photos. For drawings/graphics, tiny images and files that were
recompressed by a messaging app its score is less reliable, so we say so instead of sounding confident.
These checks never change the AI probability; only `graphic` makes the app show 'Needs Human Review'.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pathlib import Path

from PIL import Image

CROP = 224
FLAT_BLOCK_STD = 1.5          # an 8x8 block this smooth (grey-level std) counts as 'flat'
GRAPHIC_FLAT_FRACTION = 0.85  # share of flat blocks above which the picture looks like a graphic
GRAPHIC_FEW_COLOURS = 4000    # distinct colours on a small copy; photos have far more


SPARKLE_FILE = Path(__file__).parent / "weights" / "gemini_sparkle_template.npy"
SPARKLE_OFFSET = 113        # the sparkle's centre is about 113 px from the right and bottom edges
SPARKLE_HALF = 70           # template is 140 x 140 px
SPARKLE_THRESHOLD = 0.25    # correlation; real photos and non-Gemini AI images stay below 0.12
_sparkle_template = None


def gemini_sparkle(img: Image.Image) -> float:
    """Best correlation (about -1..1) between the image's bottom-right corner and Gemini's visible sparkle."""
    global _sparkle_template
    from scipy.ndimage import gaussian_filter      # scipy comes with scikit-learn
    if _sparkle_template is None:
        _sparkle_template = np.load(SPARKLE_FILE).astype(np.float32)
    g = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = g.shape
    if h < 400 or w < 400:
        return 0.0
    cy, cx = h - SPARKLE_OFFSET, w - SPARKLE_OFFSET
    c = g[max(cy - SPARKLE_HALF - 8, 0):cy + SPARKLE_HALF + 8, max(cx - SPARKLE_HALF - 8, 0):cx + SPARKLE_HALF + 8]
    c = c - gaussian_filter(c, 6)
    k = 2 * SPARKLE_HALF
    tm = _sparkle_template - _sparkle_template.mean()
    best = -1.0
    for dy in range(0, c.shape[0] - k + 1, 2):
        for dx in range(0, c.shape[1] - k + 1, 2):
            p = c[dy:dy + k, dx:dx + k]
            a = p - p.mean()
            best = max(best, float((a * tm).sum() / (np.sqrt((a * a).sum() * (tm * tm).sum()) + 1e-6)))
    return best


@dataclass
class Notice:
    kind: str          # graphic | small | messaging | watermark
    text: str


def flat_fraction(img: Image.Image, limit: int = 1024) -> float:
    g = img.convert("L")
    if max(g.size) > limit:
        s = limit / max(g.size)
        g = g.resize((max(1, int(g.width * s)), max(1, int(g.height * s))), Image.BILINEAR)
    a = np.asarray(g, dtype=np.float32)
    h, w = (a.shape[0] // 8) * 8, (a.shape[1] // 8) * 8
    if h < 8 or w < 8:
        return 0.0
    blocks = a[:h, :w].reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 64)
    return float((blocks.std(axis=1) < FLAT_BLOCK_STD).mean())


def distinct_colours(img: Image.Image, size: int = 256) -> int:
    s = img.convert("RGB")
    s.thumbnail((size, size), Image.BILINEAR)
    return int(len(np.unique(np.asarray(s).reshape(-1, 3), axis=0)))


def measure(img: Image.Image) -> dict:
    return {"flat_fraction": round(flat_fraction(img), 3), "colours": distinct_colours(img)}


def notices(img: Image.Image, meta: dict) -> list[Notice]:
    out: list[Notice] = []
    if gemini_sparkle(img) >= SPARKLE_THRESHOLD:
        out.append(Notice("watermark",
                          "A visible Gemini sparkle watermark was found in the bottom-right corner, so this image was made or "
                          "edited with Google's Gemini tools. (The mark can be cropped away, so its absence proves nothing.)"))
    m = measure(img)
    if m["flat_fraction"] >= GRAPHIC_FLAT_FRACTION and m["colours"] <= GRAPHIC_FEW_COLOURS:
        out.append(Notice("graphic",
                          "This looks like a drawing, graphic or screenshot, not a camera photo. The detector was "
                          "built for photographs, so its score is less reliable here and the verdict is set to human review."))
    if min(img.size) < 2 * CROP:
        out.append(Notice("small", "This image is small. Fewer patches can be checked, so the score is less reliable."))
    exif = meta.get("exif") or {}
    long_side = max(img.size)
    if meta.get("format") == "JPEG" and not exif and 1200 <= long_side <= 1300:
        out.append(Notice("messaging",
                          "This file has no camera data and is about 1280 px wide, which is typical of images "
                          "that may have been re-saved by a messaging app. Compression can hide the traces the detector looks for, "
                          "so the score is less reliable. Please use the original file if you have it."))
    return out
