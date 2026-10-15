"""Trust Score, verdict, heatmap picture and plain-language explanation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image

THRESHOLDS_FILE = Path(__file__).parent / "weights" / "thresholds.json"

# Provisional verdict bands (Trust Score). In Phase 1 they will be chosen on validation data
# so that automatic decisions stay below a target error rate (System Design, Algorithm 6).
AUTHENTIC_AT_OR_ABOVE = 65
MANIPULATED_AT_OR_BELOW = 35


@dataclass
class Verdict:
    label: str
    colour: str
    advice: str


def trust_score(p_fake: float) -> int:
    """Trust Score = 100 x (1 - probability of being AI-generated / manipulated)."""
    return int(round(100 * (1 - p_fake)))


def verdict(score: int, bands: tuple[int, int] | None = None) -> Verdict:
    """bands = (authentic_at_or_above, ai_at_or_below); v3 bands are chosen on validation data."""
    authentic_at, ai_at = bands or (AUTHENTIC_AT_OR_ABOVE, MANIPULATED_AT_OR_BELOW)
    if score >= authentic_at:
        return Verdict("Likely Authentic", "#3A9D5D", "No strong signs of AI generation were found.")
    if score <= ai_at:
        return Verdict("Likely AI-generated / Manipulated", "#D84343", "Strong signs of AI generation were found.")
    return Verdict("Needs Human Review", "#F2A33A", "The evidence is mixed; a person should check this image.")


def peak_threshold() -> float:
    """Spectral-peak level above which we mention periodic artifacts (set from real photos)."""
    try:
        return float(json.loads(THRESHOLDS_FILE.read_text())["spectral_peak_real_p95"])
    except (OSError, KeyError, ValueError):
        return 6.0


# ----------------------------------------------------------------------------
# Heatmap
# ----------------------------------------------------------------------------
def heatmap_overlay(img: Image.Image, grid: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Colour the image by patch score: green = looks real, red = looks AI-generated."""
    small = Image.fromarray((np.clip(grid, 0, 1) * 255).astype(np.uint8))
    smooth = np.asarray(small.resize(img.size, Image.BICUBIC), dtype=np.float32) / 255.0
    colours = (colormaps["RdYlGn_r"](smooth)[..., :3] * 255).astype(np.uint8)
    return Image.blend(img.convert("RGB"), Image.fromarray(colours), alpha)


def strongest_region(grid: np.ndarray) -> str:
    n = grid.shape[0]
    row, col = np.unravel_index(int(np.argmax(grid)), grid.shape)
    if n == 1:
        return "the image"
    rows = ["top", "middle", "bottom"] if n == 3 else ["top"] + ["middle"] * (n - 2) + ["bottom"]
    cols = ["left", "centre", "right"] if n == 3 else ["left"] + ["centre"] * (n - 2) + ["right"]
    name = f"{rows[row]}-{cols[col]}".replace("middle-centre", "centre")
    return f"the {name} area"


# ----------------------------------------------------------------------------
# Explanation
# ----------------------------------------------------------------------------
def explain(p_fake: float, grid: np.ndarray, peak: float, metadata: dict) -> list[str]:
    """Evidence-based sentences. Every sentence comes from something we measured."""
    lines = []
    if p_fake >= 0.8:
        lines.append(f"The AI detector found strong traces of AI generation (score {p_fake:.2f} out of 1).")
    elif p_fake >= 0.5:
        lines.append(f"The AI detector found some traces of AI generation (score {p_fake:.2f} out of 1).")
    elif p_fake >= 0.2:
        lines.append(f"The AI detector found only weak traces of AI generation (score {p_fake:.2f} out of 1).")
    else:
        lines.append(f"The AI detector found no clear traces of AI generation (score {p_fake:.2f} out of 1).")

    spread = float(grid.max() - grid.min())
    if grid.max() >= 0.5 and spread >= 0.3:
        lines.append(f"The traces are strongest in {strongest_region(grid)} (red on the heatmap).")
    elif grid.min() >= 0.5:
        lines.append("The traces are spread over the whole image, as is typical for fully AI-generated pictures.")

    # The spectral-peak number is shown in its own tab but not used here: in our experiment it did not
    # separate real from AI images (median 2.21 vs 2.19), so a sentence based on it could mislead.

    markers = metadata.get("ai markers", [])
    if markers:
        lines.append("The file itself contains provenance/AI markers: " + "; ".join(markers) + ".")
    camera = " ".join(v for k, v in metadata.get("exif", {}).items() if k in ("Camera make", "Camera model"))
    if camera:
        lines.append(f"The file carries camera information ({camera}). This is supporting evidence only; it can be faked.")
    elif not markers:
        lines.append("The file has no camera or AI metadata (common for screenshots and social-media images; not proof either way).")
    return lines
