import re, glob
BS = chr(92)
refs = set(re.findall(re.escape(BS) + r"bibitem\{([^}]+)\}", open("shared/refs.tex", encoding="utf-8").read()))
for doc in ("synopsis", "stage1_report"):
    files = [f"{doc}/main.tex"] + glob.glob(f"{doc}/chapters/*.tex")
    text = "".join(open(f, encoding="utf-8").read() for f in files)
    cites = set(k.strip() for m in re.findall(re.escape(BS) + r"cite\{([^}]+)\}", text) for k in m.split(","))
    labels = set(re.findall(re.escape(BS) + r"label\{([^}]+)\}", text))
    rf = set(re.findall(re.escape(BS) + r"ref\{([^}]+)\}", text))
    print(doc, "| missing cite keys:", sorted(cites - refs), "| uncited refs:", len(refs - cites), "| missing labels:", sorted(rf - labels))
