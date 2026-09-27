"""Deep-learning detectors: a frozen CLIP ViT-L/14 image encoder + small classifier heads.

How an image is scored
  1. Cut 224x224 patches at the image's own resolution (no resizing, so fine traces survive):
     one centre patch + an n x n grid spread over the image.
  2. The frozen CLIP encoder turns each patch into 768 numbers ("features").
  3. A head (a single linear layer) turns the features into a logit; sigmoid(logit) is the
     probability that the patch is AI-generated.
  4. Image score = sigmoid(mean of patch logits). The grid of patch scores is the heatmap.

Three heads are available (all share the same frozen CLIP encoder):
  - "ours_v2": our head trained on many generators, but never on ChatGPT's or Gemini's (default in the app)
  - "ours":    our head trained on ONE generator only (SD1.5) - the pure cross-generator experiment
  - "ufd":     UniversalFakeDetect (Ojha et al., CVPR 2023), trained only on ProGAN images
Our heads are trained by prototype/experiment.py (see prototype/results/ for what each one saw).
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from forensics import frequency_features

ROOT = Path(__file__).parent
WEIGHTS = ROOT / "weights"
CROP = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# ----------------------------------------------------------------------------
# Patches
# ----------------------------------------------------------------------------
def ensure_min_size(img: Image.Image, size: int = CROP) -> Image.Image:
    """Upscale images smaller than one patch (rare: icons, thumbnails)."""
    w, h = img.size
    if min(w, h) >= size:
        return img
    scale = size / min(w, h)
    return img.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.BICUBIC)


def center_crop(img: Image.Image, size: int = CROP) -> Image.Image:
    img = ensure_min_size(img, size)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def grid_boxes(w: int, h: int, n: int, size: int = CROP) -> list[tuple[int, int, int, int]]:
    """n x n patch boxes spread evenly over the image, row by row."""
    xs = np.linspace(0, w - size, n).round().astype(int)
    ys = np.linspace(0, h - size, n).round().astype(int)
    return [(int(x), int(y), int(x) + size, int(y) + size) for y in ys for x in xs]


def jpeg_normalize(img: Image.Image, quality: int = 95) -> Image.Image:
    """Re-save as JPEG so every input has the same compression (removes format shortcuts)."""
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


# ----------------------------------------------------------------------------
# CLIP encoder (shared by all heads)
# ----------------------------------------------------------------------------
class ClipEncoder:
    """OpenAI CLIP ViT-L/14 image encoder, frozen (never trained by us)."""

    def __init__(self, device: str = "cpu"):
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai")
        self.model = model.to(device).eval()
        self.device = device
        self.to_input = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)])

    @torch.no_grad()
    def encode(self, patches: list[Image.Image], batch_size: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(patches), batch_size):
            batch = torch.stack([self.to_input(p) for p in patches[i:i + batch_size]]).to(self.device)
            out.append(self.model.encode_image(batch).float().cpu().numpy())
        return np.concatenate(out, axis=0)


# ----------------------------------------------------------------------------
# Heads
# ----------------------------------------------------------------------------
@dataclass
class Head:
    key: str
    name: str
    trained_on: str
    w: np.ndarray
    b: float
    normalize_features: bool = False      # L2-normalise CLIP features first
    use_frequency: bool = False           # append standardised radial-spectrum features
    jpeg_quality: int | None = None       # re-save patches as JPEG before scoring
    freq_mean: np.ndarray | None = None
    freq_std: np.ndarray | None = None
    info: dict = field(default_factory=dict)

    def prepare(self, patch: Image.Image) -> Image.Image:
        return jpeg_normalize(patch, self.jpeg_quality) if self.jpeg_quality else patch

    def logits(self, clip_features: np.ndarray, patches: list[Image.Image]) -> np.ndarray:
        x = clip_features
        if self.normalize_features:
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
        if self.use_frequency:
            freq = np.stack([frequency_features(p) for p in patches])
            x = np.concatenate([x, (freq - self.freq_mean) / self.freq_std], axis=1)
        return x @ self.w + self.b


def load_ufd_head() -> Head:
    state = torch.load(WEIGHTS / "ufd_fc_weights.pth", map_location="cpu")
    return Head(
        key="ufd",
        name="UniversalFakeDetect (published model)",
        trained_on="ProGAN only (a 2018 GAN), 20 object categories",
        w=state["weight"].numpy()[0].astype(np.float32),
        b=float(state["bias"].numpy()[0]),
        info={"source": "Ojha et al., CVPR 2023 - github.com/WisconsinAIVision/UniversalFakeDetect (MIT)"},
    )


def load_our_head(stem: str = "our_head", key: str = "ours") -> Head | None:
    path = WEIGHTS / f"{stem}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    meta = json.loads((WEIGHTS / f"{stem}.json").read_text(encoding="utf-8"))
    return Head(
        key=key,
        name=meta["name"],
        trained_on=meta["trained_on"],
        w=data["w"].astype(np.float32),
        b=float(data["b"]),
        normalize_features=meta["normalize_features"],
        use_frequency=meta["use_frequency"],
        jpeg_quality=meta.get("jpeg_quality"),
        freq_mean=data["freq_mean"] if meta["use_frequency"] else None,
        freq_std=data["freq_std"] if meta["use_frequency"] else None,
        info=meta,
    )


# ----------------------------------------------------------------------------
# Detector
# ----------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


@dataclass
class DetectionResult:
    head: Head
    probability: float                 # probability the image is AI-generated / manipulated
    logit: float
    grid: np.ndarray                   # n x n patch probabilities (heatmap)
    boxes: list[tuple[int, int, int, int]]
    center_probability: float


class Detector:
    def __init__(self, device: str = "cpu"):
        self.encoder = ClipEncoder(device)
        self.heads: dict[str, Head] = {}
        for stem, key in (("our_head_v2", "ours_v2"), ("our_head", "ours")):
            head = load_our_head(stem, key)
            if head is not None:
                self.heads[key] = head
        self.heads["ufd"] = load_ufd_head()

    def analyze(self, img: Image.Image, head_key: str, grid: int = 3) -> DetectionResult:
        head = self.heads[head_key]
        img = ensure_min_size(img)
        boxes = grid_boxes(img.width, img.height, grid)
        patches = [center_crop(img)] + [img.crop(b) for b in boxes]
        patches = [head.prepare(p) for p in patches]
        logits = head.logits(self.encoder.encode(patches), patches)
        probs = sigmoid(logits)
        mean_logit = float(np.mean(logits))
        return DetectionResult(
            head=head,
            probability=float(sigmoid(mean_logit)),
            logit=mean_logit,
            grid=probs[1:].reshape(grid, grid),
            boxes=boxes,
            center_probability=float(probs[0]),
        )
