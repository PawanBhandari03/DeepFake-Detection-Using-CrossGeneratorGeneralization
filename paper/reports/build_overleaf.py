"""Prepare the three LaTeX reports for Overleaf.

    python paper/reports/build_overleaf.py [folder-with-raw-screenshots]

What it does
  1. Crops the college logo and the raw app screenshots (if a folder is given) into shared/.
  2. Copies the shared files (preamble, team details, references, figures, logo, screenshots)
     into abstract/, synopsis/ and stage1_report/, so that every folder is a complete project.
  3. Writes one zip per report to paper/reports/overleaf_zips/ (Overleaf: New Project -> Upload Project).

Edit names, PRN numbers and the principal's name only in shared/team.tex and run this script again.
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageChops

HERE = Path(__file__).resolve().parent
SHARED = HERE / "shared"
FIGS = SHARED / "figs"
ZIPS = HERE / "overleaf_zips"
PAPER_FIGS = HERE.parent / "figures"

SHOT_NAMES = {            # raw name -> (final name, crop height in pixels or None for 'content only')
    "landing": ("app_landing.png", None),
    "result_ai": ("app_result_ai.png", 1250),
    "result_real": ("app_result_real.png", 1250),
    "result_watermark": ("app_result_watermark.png", 1250),
    "result_drawing": ("app_result_drawing.png", 1250),
    "crossgen": ("app_crossgen.png", 1500),
}


def trim(im: Image.Image, margin: int = 12) -> Image.Image:
    bg = Image.new(im.mode, im.size, (255, 255, 255) if im.mode == "RGB" else 255)
    box = ImageChops.difference(im, bg).getbbox()
    if not box:
        return im
    l, t, r, b = box
    return im.crop((max(l - margin, 0), max(t - margin, 0), min(r + margin, im.width), min(b + margin, im.height)))


def prepare_assets(raw_shots: Path | None) -> None:
    logo = HERE / "_assets" / "bsiotr_logo.png"
    if logo.exists():
        trim(Image.open(logo).convert("RGB"), 6).save(SHARED / "bsiotr_logo.png")
    if (PAPER_FIGS / "fig_cm_roc.png").exists():
        shutil.copy(PAPER_FIGS / "fig_cm_roc.png", FIGS / "fig_cm_roc.png")
    if raw_shots and raw_shots.exists():
        for raw, (final, height) in SHOT_NAMES.items():
            src = raw_shots / f"{raw}.png"
            if not src.exists():
                print("missing screenshot:", src)
                continue
            im = Image.open(src).convert("RGB")
            if height:
                im = im.crop((0, 0, im.width, min(height, im.height)))
            else:
                im = trim(im)
            im.save(FIGS / final, optimize=True)


def copy_project(name: str, figs: list[str], need_refs: bool) -> Path:
    dst = HERE / name
    (dst / "figs").mkdir(parents=True, exist_ok=True)
    for f in ("preamble.tex", "team.tex") + (("refs.tex",) if need_refs else ()):
        shutil.copy(SHARED / f, dst / f)
    if (SHARED / "bsiotr_logo.png").exists():
        shutil.copy(SHARED / "bsiotr_logo.png", dst / "bsiotr_logo.png")
    for f in figs:
        if (FIGS / f).exists():
            shutil.copy(FIGS / f, dst / "figs" / f)
    return dst


def make_zip(folder: Path) -> Path:
    ZIPS.mkdir(exist_ok=True)
    out = ZIPS / f"{folder.name}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix.lower() not in {".pdf", ".aux", ".log", ".out", ".toc", ".lof", ".lot", ".synctex.gz"}:
                z.write(p, p.relative_to(folder).as_posix())
    return out


def main() -> None:
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    prepare_assets(raw)
    synopsis_figs = ["fig_existing.tex", "fig_architecture.tex", "fig_training.tex", "fig_image_pipeline.tex"]
    all_figs = sorted(p.name for p in FIGS.iterdir() if p.is_file())
    for name, figs, refs in (("abstract", [], False), ("synopsis", synopsis_figs, True), ("stage1_report", all_figs, True)):
        folder = copy_project(name, figs, refs)
        print("zip:", make_zip(folder))


if __name__ == "__main__":
    main()
