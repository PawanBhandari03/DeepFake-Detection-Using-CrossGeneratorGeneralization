"""Small two-column A4 page layout engine on top of PyMuPDF's Story (used by build_full_paper.py).

Features: full-width title block, two columns, whole blocks (tables, figures) that float to the next column
when they do not fit, headings that never end up alone at the bottom of a column, full-width figures at the
top of chosen pages, and citation numbers that follow the order of first use ([[key1,key2]] in the text).
"""
import re
import shutil
from pathlib import Path

import pymupdf

PAGE = pymupdf.paper_rect("a4")
MARGIN_X, TOP, BOTTOM, GAP = 46, 42, 50, 16
COL_W = (PAGE.width - 2 * MARGIN_X - GAP) / 2
FONT_DIR = r"C:\Windows\Fonts"


def number_citations(html: str, refs: dict) -> tuple[str, list[str]]:
    """Replace [[a,b]] by [1], [2] in order of first use; return the new html and the ordered keys."""
    order: list[str] = []

    def num(key):
        key = key.strip()
        if key not in refs:
            raise KeyError(f"unknown reference key: {key}")
        if key not in order:
            order.append(key)
        return order.index(key) + 1

    def repl(m):
        nums = [num(k) for k in m.group(1).split(",")]
        return ", ".join(f"[{n}]" for n in nums)

    return re.sub(r"\[\[([^\]]+)\]\]", repl, html), order


def strip_citations(html: str) -> str:
    """Version without references: remove every [[key]] marker together with the space before it."""
    html = re.sub(r"\s*\[\[[^\]]+\]\]", "", html)
    html = re.sub(r"\(\s*\)", "", html)
    return html


def build_pdf(out: Path, head_html: str, body_html: str, blocks: dict, css: str, fig_dir: Path, full_width: dict,
              footer: str, meta: dict, small_words=(), refs_html=None) -> int:
    archive = pymupdf.Archive(FONT_DIR)
    archive.add(str(fig_dir))
    writer = pymupdf.DocumentWriter(str(out))
    bottom = PAGE.height - BOTTOM

    head = pymupdf.Story(html=head_html, user_css=css, archive=archive)
    dev = writer.begin_page(PAGE)
    _, filled = head.place(pymupdf.Rect(MARGIN_X, TOP, PAGE.width - MARGIN_X, bottom))
    head.draw(dev)
    st = {"page": 1, "col": 0, "top": filled[3] + 6, "dev": dev}
    st["y"] = st["top"]
    deferred = []

    def col_rect():
        x = MARGIN_X + st["col"] * (COL_W + GAP)
        return pymupdf.Rect(x, st["y"], x + COL_W, bottom)

    def height_of(html):
        probe = pymupdf.Story(html=html, user_css=css, archive=archive)
        _, used = probe.place(pymupdf.Rect(0, 0, COL_W, 3000))
        return used[3] + 2

    def draw_block(html, height):
        story = pymupdf.Story(html=html, user_css=css, archive=archive)
        r = col_rect()
        story.place(pymupdf.Rect(r.x0, st["y"], r.x1, st["y"] + height + 4))
        story.draw(st["dev"])
        st["y"] += height + 3

    def place_deferred():
        while deferred and st["y"] + deferred[0][1] <= bottom:
            html, h = deferred.pop(0)
            draw_block(html, h)

    def new_column():
        st["col"] += 1
        if st["col"] == 2:
            writer.end_page()
            st["dev"] = writer.begin_page(PAGE)
            st["page"] += 1
            st["col"], st["top"] = 0, TOP
            if st["page"] in full_width:                 # full-width figure at the top of this page
                fw = pymupdf.Story(html=full_width[st["page"]], user_css=css, archive=archive)
                _, fr = fw.place(pymupdf.Rect(MARGIN_X, TOP, PAGE.width - MARGIN_X, bottom))
                fw.draw(st["dev"])
                st["top"] = fr[3] + 6
        st["y"] = st["top"]
        place_deferred()

    items = []
    for i, piece in enumerate(re.split(r"@@(\w+)@@", body_html)):
        if i % 2 == 0:
            items += [("text", c) for c in re.split(r"(?=<h[23]>)", piece) if c.strip()]
        else:
            items.append(("block", piece))

    for kind, piece in items:
        if kind == "text":
            if piece.lstrip().startswith("<h") and bottom - st["y"] < 62:
                new_column()                             # not even the heading and two lines fit here
            if bottom - st["y"] < 26:                    # less than about two lines left: start the next column
                new_column()
            story = pymupdf.Story(html=piece.strip(), user_css=css, archive=archive)
            while True:
                more, filled = story.place(col_rect())
                story.draw(st["dev"])
                if not more:
                    st["y"] = filled[3] - 8              # Story reports about 11 pt more than the text uses
                    break
                new_column()
                if bottom - st["y"] < 26:
                    new_column()
        else:
            html = blocks[piece]
            h = height_of(html)
            if not deferred and st["y"] + h <= bottom:
                draw_block(html, h)
            else:
                deferred.append((html, h))
    while deferred:
        if deferred[0][1] > bottom - st["y"]:
            new_column()
            continue
        html, h = deferred.pop(0)
        draw_block(html, h)
    if refs_html:
        # all references go into the RIGHT column of the last page; the body ends in the left column
        if st["col"] == 0:
            st["col"], st["y"] = 1, st["top"]
        else:
            print("WARNING: the body already reaches the right column; the references follow it")
        rstory = pymupdf.Story(html=refs_html, user_css=css, archive=archive)
        while True:
            more, filled = rstory.place(col_rect())
            rstory.draw(st["dev"])
            if not more:
                print(f"references end at y={filled[3]:.0f} of {bottom:.0f} (right column, page {st['page']})")
                break
            print("WARNING: the references do not fit into one column")
            new_column()
    writer.end_page()
    writer.close()

    doc = pymupdf.open(str(out))
    for i, page in enumerate(doc, start=1):
        page.insert_text((PAGE.width / 2 - 60, PAGE.height - 26), f"{footer} | {i}", fontsize=8, color=(0.4, 0.42, 0.45))
    doc.set_metadata(meta)
    doc.subset_fonts()
    tmp = out.with_suffix(".tmp.pdf")
    doc.ez_save(str(tmp))
    n = doc.page_count
    doc.close()
    shutil.copyfile(tmp, out)
    tmp.unlink()
    return n
