"""LAYOUT TEST ONLY. Builds a temporary folder with STAND-IN numbers so that the page layout and the citation
placement of the image paper can be checked before the real Kaggle robustness results exist.

    python paper/make_layout_test_data.py <temp_dir>

The stand-in numbers are copied from conditions we really measured (clean, JPEG 70, half size) and reused for the
others, so they are NOT real results for those conditions. The temp folder is never inside paper/ and its PDFs must not
be shown to anyone as results.
"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
out = Path(sys.argv[1])
(out / "figures").mkdir(parents=True, exist_ok=True)
for f in (HERE / "figures").glob("*.png"):
    shutil.copy(f, out / "figures" / f.name)
shutil.copy(HERE / "figures" / "fig3.png", out / "figures" / "fig_robust.png")      # stand-in picture

R = json.loads((HERE.parent / "prototype" / "results" / "v3_results.json").read_text(encoding="utf-8"))
src = {"clean": "clean", "jpeg70": "jpeg70", "jpeg90": "clean", "jpeg50": "jpeg70", "screenshot": "half_q75", "half": "half_q75", "blur": "jpeg70"}
res = {}
for key, rk in (("v2", "v2_reference"), ("v3_noaug", "v3_noaug"), ("v3", "v3")):
    res[key] = {}
    for cond, s in src.items():
        e = R[rk]["test_openfake"][s]
        res[key][cond] = {"auc": e["auc"], "caught": e["fake_caught"], "real_correct": e["real_correct"], "n_real": e["n_real"], "n_fake": e["n_fake"]}
(out / "robust_results.json").write_text(json.dumps(res), encoding="utf-8")
print("stand-in layout-test data written to", out)
