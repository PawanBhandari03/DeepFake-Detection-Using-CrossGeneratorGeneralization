"""Edge-case notices: tell the user when an image is outside what the detector was built for.

The detector was trained on camera-style photos. For drawings/graphics, tiny images and files that were
recompressed by a messaging app its score is less reliable, so we say so instead of sounding confident.
These checks never change the AI probability; only `graphic` makes the app show 'Needs Human Review'.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

CROP = 224
FLAT_BLOCK_STD = 1.5          # an 8x8 block this smooth (grey-level std) counts as 'flat'
GRAPHIC_FLAT_FRACTION = 0.85  # share of flat blocks above which the picture looks like a graphic
GRAPHIC_FEW_COLOURS = 4000    # distinct colours on a small copy; photos have far more


@dataclass
class Notice:
    kind: str          # graphic | small | messaging
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
