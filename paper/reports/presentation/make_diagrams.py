"""Render the TikZ diagrams of shared/figs as PNG pictures for the slides.

    python paper/reports/presentation/make_diagrams.py <path-to-tectonic.exe>

Needs the Tectonic LaTeX engine (or any LaTeX that has tikz, pgfplots, pgfgantt and standalone).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pymupdf
from PIL import Image, ImageChops

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "shared"
OUT = HERE / "diagrams"
BS = chr(92)

PREAMBLE = BS + r"""documentclass[border=10pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
\usepackage{xcolor}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,shapes.misc,shapes.multipart,calc,fit,backgrounds,decorations.pathreplacing}
\usepackage{pgfplots}
\pgfplotsset{compat=1.16}
\usepackage{pgfgantt}
\input{tikzstyles}
\newlength{\figtextheight}
\setlength{\figtextheight}{26cm}
\let\oldtextheight\textheight
\renewcommand{\textheight}{\figtextheight}
\begin{document}
\begin{minipage}{18cm}
\input{FIG}
\end{minipage}
\end{document}
"""

FIGS = ["fig_architecture", "fig_training", "fig_image_pipeline", "fig_dfd0", "fig_dfd1", "fig_usecase", "fig_flow",
        "fig_sequence", "fig_class", "fig_gantt", "fig_existing"]
PLAIN = ["fig_auc", "fig_baseline"]          # pgfplots charts have their own size


def trim(im: Image.Image, margin: int = 14) -> Image.Image:
    bg = Image.new("RGB", im.size, (255, 255, 255))
    box = ImageChops.difference(im.convert("RGB"), bg).getbbox()
    if not box:
        return im
    l, t, r, b = box
    return im.crop((max(l - margin, 0), max(t - margin, 0), min(r + margin, im.width), min(b + margin, im.height)))


def main() -> None:
    tectonic = sys.argv[1] if len(sys.argv) > 1 else "tectonic"
    OUT.mkdir(exist_ok=True)
    for name in FIGS + PLAIN:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            td = Path(td)
            shutil.copy(SHARED / "tikzstyles.tex", td / "tikzstyles.tex") if (SHARED / "tikzstyles.tex").exists() else None
            (td / "figs").mkdir()
            for f in (SHARED / "figs").glob("*.tex"):
                shutil.copy(f, td / "figs" / f.name)
            body = PREAMBLE.replace("FIG", f"figs/{name}")
            if name in PLAIN:
                body = body.replace(BS + "begin{minipage}{18cm}", "").replace(BS + "end{minipage}", "")
            (td / "main.tex").write_text(body, encoding="utf-8")
            r = subprocess.run([tectonic, "main.tex"], cwd=td, capture_output=True, text=True)
            pdf = td / "main.pdf"
            if not pdf.exists():
                print("FAILED", name)
                print((r.stdout + r.stderr)[-1500:])
                continue
            doc = pymupdf.open(pdf)
            pix = doc[0].get_pixmap(dpi=220)
            doc.close()
            im = trim(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
            im.save(OUT / f"{name}.png", optimize=True)
            print("ok", name, im.size)


if __name__ == "__main__":
    main()
