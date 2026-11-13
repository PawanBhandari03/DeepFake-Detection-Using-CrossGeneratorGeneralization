"""Quick static check of the .tex sources (no LaTeX needed): balanced braces and begin/end pairs."""
import glob
import re

BS = chr(92)
for f in sorted(glob.glob("**/*.tex", recursive=True)):
    norm = f.replace(BS, "/")
    if "/figs/" in norm and norm.split("/")[0] in ("abstract", "synopsis", "stage1_report"):
        continue
    t = open(f, encoding="utf-8").read()
    t2 = re.sub("(?<!" + re.escape(BS) + ")%.*", "", t)
    env = {}
    for m in re.finditer(re.escape(BS) + r"(begin|end)\{([^}]+)\}", t2):
        env[m.group(2)] = env.get(m.group(2), 0) + (1 if m.group(1) == "begin" else -1)
    bad = {k: v for k, v in env.items() if v}
    t3 = t2.replace(BS + "{", "").replace(BS + "}", "")
    print(f"{f:48s} braces {t3.count('{') - t3.count('}'):+d}  env-imbalance {bad or 'none'}")
