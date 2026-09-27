"""Image loading and training-free forensic evidence.

Everything here works without any trained model:
  - load_image:        open any common format and convert it to a standard RGB image
  - frequency_features / spectrum_image / spectral_peak_score:
                       FFT of the noise residual (AI up-sampling leaves regular peaks)
  - ela_image:         Error Level Analysis (how parts of a JPEG react to re-saving)
  - read_metadata:     camera EXIF, editing software, Content Credentials (C2PA) hints
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageOps, UnidentifiedImageError

try:  # iPhone HEIC/HEIF support
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # the prototype still works without it, just no HEIC
    pillow_heif = None

Image.MAX_IMAGE_PIXELS = 60_000_000  # refuse "decompression bomb" images

SUPPORTED_TYPES = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "gif", "heic", "heif"]
N_RADIAL_BINS = 64


class ImageLoadError(ValueError):
    """Raised when an upload is not a readable image."""


@dataclass
class LoadedImage:
    rgb: Image.Image                  # standard 8-bit RGB image the models see
    file_format: str                  # real format detected from the file bytes
    original_mode: str                # e.g. RGBA, P, CMYK, I;16
    raw_bytes: bytes = field(repr=False)
    notes: list[str] = field(default_factory=list)


def load_image(data: bytes) -> LoadedImage:
    """Open image bytes in any supported format and convert to 8-bit RGB."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageLoadError(f"Could not read this file as an image ({exc}).") from exc

    fmt = img.format or "unknown"
    mode = img.mode
    notes: list[str] = []

    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)
        notes.append("Animated image: only the first frame was analysed.")

    img = ImageOps.exif_transpose(img)  # apply the phone's rotation flag

    if mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(img, dtype=np.float64)
        arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1) * 255
        img = Image.fromarray(arr.astype(np.uint8))
        notes.append("16-bit image converted to 8-bit.")
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, rgba)
        notes.append("Transparent areas were placed on a white background.")
    rgb = img.convert("RGB")
    return LoadedImage(rgb=rgb, file_format=fmt, original_mode=mode, raw_bytes=data, notes=notes)


# ----------------------------------------------------------------------------
# Frequency (FFT) evidence
# ----------------------------------------------------------------------------
def _gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32)


def noise_residual(img: Image.Image) -> np.ndarray:
    """Image minus a 3x3 median-filtered copy: removes content, keeps fine noise."""
    gray_img = img.convert("L")
    smooth = np.asarray(gray_img.filter(ImageFilter.MedianFilter(3)), dtype=np.float32)
    return _gray(gray_img) - smooth


def log_spectrum(residual: np.ndarray) -> np.ndarray:
    """Centred log-magnitude 2-D Fourier spectrum."""
    spec = np.fft.fftshift(np.fft.fft2(residual))
    return np.log1p(np.abs(spec)).astype(np.float32)


def _radius_grid(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    return r / r.max()


def frequency_features(img: Image.Image, bins: int = N_RADIAL_BINS) -> np.ndarray:
    """Radial (azimuthally averaged) spectrum of the noise residual: `bins` numbers."""
    spec = log_spectrum(noise_residual(img))
    ring = np.minimum((_radius_grid(spec.shape) * bins).astype(int), bins - 1)
    sums = np.bincount(ring.ravel(), weights=spec.ravel(), minlength=bins)
    counts = np.bincount(ring.ravel(), minlength=bins)
    return (sums / np.maximum(counts, 1)).astype(np.float32)


def _limit(img: Image.Image, size: int = 1024) -> Image.Image:
    """Centre crop large images (no resizing, so fine traces are kept)."""
    w, h = img.size
    if w <= size and h <= size:
        return img
    left, top = max((w - size) // 2, 0), max((h - size) // 2, 0)
    return img.crop((left, top, left + min(size, w), top + min(size, h)))


def spectral_peak_score(img: Image.Image) -> float:
    """How strongly isolated peaks stand out in the high-frequency spectrum.

    Each spectrum value is compared with the average of its ring; the score is the
    99.95th percentile of those z-values outside the central (low-frequency) area.
    Regular up-sampling grids produce sharp peaks, so higher = more AI-like.
    """
    spec = log_spectrum(noise_residual(_limit(img)))
    r = _radius_grid(spec.shape)
    ring = np.minimum((r * 128).astype(int), 127)
    mean = np.bincount(ring.ravel(), spec.ravel(), 128) / np.maximum(np.bincount(ring.ravel(), minlength=128), 1)
    sq = np.bincount(ring.ravel(), (spec ** 2).ravel(), 128) / np.maximum(np.bincount(ring.ravel(), minlength=128), 1)
    std = np.sqrt(np.maximum(sq - mean ** 2, 1e-6))
    z = (spec - mean[ring]) / std[ring]
    outer = z[r > 0.25]
    return float(np.percentile(outer, 99.95))


def spectrum_image(img: Image.Image) -> Image.Image:
    """Grayscale picture of the log spectrum, for display."""
    spec = log_spectrum(noise_residual(_limit(img)))
    lo, hi = np.percentile(spec, [1, 99.8])
    norm = np.clip((spec - lo) / max(hi - lo, 1e-6), 0, 1)
    return Image.fromarray((norm * 255).astype(np.uint8))


# ----------------------------------------------------------------------------
# Error Level Analysis
# ----------------------------------------------------------------------------
def ela_image(img: Image.Image, quality: int = 90) -> Image.Image:
    """Re-save as JPEG and amplify the difference. Edited JPEG regions often stand out."""
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    resaved = Image.open(io.BytesIO(buf.getvalue()))
    diff = np.asarray(ImageChops.difference(img, resaved), dtype=np.float32)
    scale = 255.0 / max(np.percentile(diff, 99.5), 1.0)
    return Image.fromarray(np.clip(diff * scale, 0, 255).astype(np.uint8))


# ----------------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------------
EXIF_TAGS = {271: "Camera make", 272: "Camera model", 305: "Software", 306: "Date/time"}

# Text markers that some tools write into files. They are hints, not proof.
AI_MARKERS = {
    b"c2pa": "Content Credentials (C2PA) manifest",
    b"trainedAlgorithmicMedia": "IPTC tag 'trained algorithmic media' (AI-generated)",
    b"compositeWithTrainedAlgorithmicMedia": "IPTC tag 'composite with AI-generated media'",
    b"OpenAI": "mentions OpenAI",
    b"DALL": "mentions DALL-E",
    b"Midjourney": "mentions Midjourney",
    b"Stable Diffusion": "mentions Stable Diffusion",
    b"Adobe Firefly": "mentions Adobe Firefly",
    b"Google AI": "mentions Google AI",
}


def read_metadata(loaded: LoadedImage) -> dict:
    """Camera EXIF, editing software and AI/provenance markers found in the file."""
    img = Image.open(io.BytesIO(loaded.raw_bytes))
    exif = {}
    try:
        raw = img.getexif()
        for tag, name in EXIF_TAGS.items():
            if tag in raw and str(raw[tag]).strip():
                exif[name] = str(raw[tag]).strip("\x00 ")
    except Exception:  # broken EXIF should never stop the analysis
        pass

    text_info = {k: str(v)[:200] for k, v in img.info.items() if isinstance(v, (str, bytes)) and k.lower() in
                 ("parameters", "prompt", "workflow", "software", "comment", "description")}

    markers = sorted({label for key, label in AI_MARKERS.items() if key in loaded.raw_bytes})
    return {
        "format": loaded.file_format,
        "size": f"{loaded.rgb.width} x {loaded.rgb.height}",
        "colour mode": loaded.original_mode,
        "exif": exif,
        "text chunks": text_info,
        "ai markers": markers,
    }
