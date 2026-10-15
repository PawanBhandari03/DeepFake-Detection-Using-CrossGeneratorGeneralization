"""Probe run on Kaggle: GPU + CLIP speed, dataset access, and what is inside each OpenFake split.
Prints one JSON report; nothing is trained or saved."""
import json, subprocess, sys, time, zipfile
from collections import Counter

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch", "huggingface_hub", "pyarrow"], check=False)
report = {}

import torch
report["torch"] = torch.__version__
report["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout)
print(subprocess.run(["df", "-h", "/kaggle/working", "/tmp"], capture_output=True, text=True).stdout)

from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq
fs = HfFileSystem()

def sample_meta(pattern, picks, rg=0):
    files = sorted(fs.glob(pattern))
    out = {"n_files": len(files)}
    for idx in picks:
        if idx >= len(files):
            continue
        t0 = time.time()
        with fs.open(files[idx], "rb") as f:
            pf = pq.ParquetFile(f)
            t = pf.read_row_groups([rg], columns=["label", "model", "type"])
            c = Counter(zip(t.column("label").to_pylist(), t.column("model").to_pylist()))
            out[str(idx)] = {"rows_total": pf.metadata.num_rows, "row_groups": pf.num_row_groups,
                             "rg_rows": t.num_rows, "secs": round(time.time() - t0, 1), "top": c.most_common(40)}
    return out

for name, pat, picks in [
    ("train", "datasets/ComplexDataLab/OpenFake/core/train-*.parquet", [0, 150, 300, 450, 607]),
    ("validation", "datasets/ComplexDataLab/OpenFake/core/validation-*.parquet", [0, 7]),
    ("test", "datasets/ComplexDataLab/OpenFake/core/test-*.parquet", [0, 6]),
    ("reddit", "datasets/ComplexDataLab/OpenFake/reddit/test-*.parquet", [0]),
]:
    try:
        report[name] = sample_meta(pat, picks)
    except Exception as e:
        report[name] = {"error": repr(e)}
    print(name, json.dumps(report[name])[:1500], flush=True)

# DiffSeg30k: can we read the zip without downloading all of it?
try:
    t0 = time.time()
    with fs.open("datasets/Chaos2629/Diffseg30k/validation.zip", "rb") as f:
        z = zipfile.ZipFile(f)
        names = z.namelist()
        report["diffseg"] = {"n_files": len(names), "first": names[:6], "secs": round(time.time() - t0, 1)}
except Exception as e:
    report["diffseg"] = {"error": repr(e)}
print("diffseg", report["diffseg"], flush=True)

# tiny-audits layout
try:
    c = Counter()
    for p in fs.find("datasets/DivyaApp/tiny-audits/COCO"):
        parts = p.split("/")
        c[(parts[-3], parts[-2])] += 1
    report["tiny_audits_coco"] = {f"{a}/{b}": n for (a, b), n in sorted(c.items())}
except Exception as e:
    report["tiny_audits_coco"] = {"error": repr(e)}

# CLIP speed on GPU
try:
    import open_clip
    from open_clip.model import _build_vision_tower
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    cfg = open_clip.get_model_config("ViT-L-14-quickgelu")
    with torch.device("meta"):
        visual = _build_vision_tower(cfg["embed_dim"], cfg["vision_cfg"], quick_gelu=True)
    path = hf_hub_download("timm/vit_large_patch14_clip_224.openai", "open_clip_model.safetensors")
    with safe_open(path, framework="pt") as f:
        w = {k[len("visual."):]: f.get_tensor(k) for k in f.keys() if k.startswith("visual.")}
    visual.load_state_dict(w, assign=True)
    visual = visual.cuda().eval()
    x = torch.randn(128, 3, 224, 224, device="cuda")
    with torch.no_grad():
        for dt in (torch.float32, torch.float16):
            m = visual if dt == torch.float32 else visual.half()
            xi = x.to(dt)
            m(xi[:16])
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(3):
                m(xi)
            torch.cuda.synchronize()
            report[f"clip_patches_per_sec_{str(dt)[6:]}"] = round(3 * 128 / (time.time() - t0))
except Exception as e:
    report["clip"] = repr(e)

print("REPORT", json.dumps(report))
json.dump(report, open("/kaggle/working/probe_report.json", "w"))
