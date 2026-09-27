"""Convert a Markdown file to a styled A4 PDF.

Usage:  python docs/tools/md2pdf.py PROJECT_ROADMAP.md docs/Review3_QA.md
Writes a .pdf next to each .md file.
Needs:  pip install markdown pymupdf   (Windows fonts are used for text)
"""
import re
import shutil
import sys
from pathlib import Path

import markdown
import pymupdf

FONT_DIR = r"C:\Windows\Fonts"

CSS = """
@font-face { font-family: body; src: url(segoeui.ttf); }
@font-face { font-family: body; src: url(segoeuib.ttf); font-weight: bold; }
@font-face { font-family: body; src: url(segoeuii.ttf); font-style: italic; }
@font-face { font-family: body; src: url(segoeuiz.ttf); font-weight: bold; font-style: italic; }
@font-face { font-family: mono; src: url(consola.ttf); }
@font-face { font-family: sym; src: url(seguisym.ttf); }
body { font-family: body; font-size: 10pt; line-height: 1.35; color: #1F2937; }
h1 { font-size: 18pt; color: #3949AB; margin: 0 0 6pt 0; }
h2 { font-size: 14pt; color: #3949AB; margin: 14pt 0 4pt 0; }
h3 { font-size: 11.5pt; color: #00897B; margin: 10pt 0 3pt 0; }
h4 { font-size: 10.5pt; margin: 8pt 0 2pt 0; }
p { margin: 3pt 0 5pt 0; }
ul, ol { margin: 2pt 0 5pt 0; }
li { margin: 1pt 0; }
table { border-collapse: collapse; margin: 4pt 0 8pt 0; width: 100%; }
th { font-weight: bold; text-align: left; color: #283593; border-bottom: 1.2pt solid #3949AB; }
th, td { border: 0.6pt solid #9FA8DA; padding: 3pt 4pt; font-size: 9pt; vertical-align: top; }
code { font-family: mono; font-size: 8.8pt; color: #283593; }
pre { font-family: mono; font-size: 8.5pt; padding: 2pt 0 2pt 8pt; border-left: 2pt solid #C5CAE9; }
blockquote { margin: 5pt 0; padding: 2pt 8pt; border-left: 2.5pt solid #00897B; }
hr { border: 0; border-top: 0.6pt solid #C5CAE9; margin: 8pt 0; }
a { color: #00897B; text-decoration: none; }
.box { font-family: sym; }
img { width: 100%; }
"""

MARGIN = 48          # points (about 17 mm)
PAGE = pymupdf.paper_rect("a4")


EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-☏☑-⛿✀-➿️ℹ]+ ?")  # keeps ☐ (U+2610)
SYMBOLS = {"❌": "✗", "✅": "✓", "⭐": "★", "☑": "☑"}


def md_to_html(text: str) -> str:
    # PDF fonts cannot draw colour emoji: map the meaningful ones to plain symbols, drop the rest
    for i, emoji in enumerate(SYMBOLS):
        text = text.replace(emoji, f"@@SYM{i}@@")
    text = EMOJI.sub("", text)
    for i, symbol in enumerate(SYMBOLS.values()):
        text = text.replace(f"@@SYM{i}@@", f'<span class="box">{symbol}</span>')
    html = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    # GitHub-style task boxes -> ballot box symbols
    html = re.sub(r"\[ \]", '<span class="box">&#9744;</span>', html)
    html = re.sub(r"\[[xX]\]", '<span class="box">&#9745;</span>', html)
    html = html.replace("\u2610", '<span class="box">&#9744;</span>')
    return html


def convert(md_path: Path) -> Path:
    pdf_path = md_path.with_suffix(".pdf")
    html = md_to_html(md_path.read_text(encoding="utf-8"))
    archive = pymupdf.Archive(FONT_DIR)
    archive.add(str(md_path.parent))          # lets <img src="img/..."> find pictures next to the .md
    story = pymupdf.Story(html=html, user_css=CSS, archive=archive)
    where = PAGE + (MARGIN, MARGIN, -MARGIN, -MARGIN - 10)
    writer = pymupdf.DocumentWriter(str(pdf_path))
    more = True
    while more:
        device = writer.begin_page(PAGE)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()

    # page numbers, then keep only the font glyphs actually used (small file to share)
    doc = pymupdf.open(str(pdf_path))
    for i, page in enumerate(doc, start=1):
        page.insert_text((PAGE.width / 2 - 20, PAGE.height - 24), f"{i} / {doc.page_count}",
                         fontsize=8, color=(0.42, 0.45, 0.5))
    doc.subset_fonts()
    tmp = pdf_path.with_suffix(".tmp.pdf")
    doc.ez_save(str(tmp))
    doc.close()
    shutil.copyfile(tmp, pdf_path)   # copy, not rename: works even if a viewer has the PDF open
    tmp.unlink()
    return pdf_path


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print("wrote", convert(Path(arg)))
