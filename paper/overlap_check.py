"""Rough text-overlap check of a paper against other documents (our own reassurance; not a plagiarism checker).

    python paper/overlap_check.py <new.pdf> <other.pdf|.md|.tex> [more files ...]

Counts how many word 5-grams and 8-grams of the new paper (title block and references removed) also occur in each
other document. PDFs are read with PyMuPDF; .md / .tex / .txt are read as plain text.
"""
import re
import sys
from pathlib import Path

import pymupdf


def text_of(path):
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return " ".join(page.get_text() for page in pymupdf.open(str(p)))
    return p.read_text(encoding="utf-8", errors="ignore")


def words(path, drop_refs=True):
    text = text_of(path)
    if drop_refs and "REFERENCES" in text:
        text = text[: text.rindex("REFERENCES")]
    elif drop_refs and "References" in text:
        text = text[: text.rindex("References")]
    return re.findall(r"[a-z]+", text.lower())


def grams(w, n):
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


if __name__ == "__main__":
    new_words = words(sys.argv[1])
    new5, new8 = grams(new_words, 5), grams(new_words, 8)
    print(f"checked text: {len(new_words)} words, {len(new5)} five-word and {len(new8)} eight-word sequences")
    for other in sys.argv[2:]:
        ow = words(other, drop_refs=False)
        s5, s8 = new5 & grams(ow, 5), new8 & grams(ow, 8)
        print(f"{Path(other).name}: {len(s5)} five-word ({100 * len(s5) / len(new5):.1f}%) | {len(s8)} eight-word ({100 * len(s8) / len(new8):.1f}%)")
        for s in sorted(s8)[:6]:
            print("     ", s)
