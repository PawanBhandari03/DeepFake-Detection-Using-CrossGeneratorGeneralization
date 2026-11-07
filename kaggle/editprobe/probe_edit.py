"""Probe of UCSC-VLAA/GPT-Image-Edit-1.5M on Kaggle (CPU only, nothing is trained).

Streams the first archive part of ultraedit, omniedit and hqedit, records what the tar members look like
(task folders, input/output order, counts, image sizes) and saves a few example input/output pairs as JPEG
thumbnails so that they can be inspected by eye.
"""
import gzip, io, json, os, subprocess, sys, tarfile, time
from collections import Counter, defaultdict

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=False)
from PIL import Image
from huggingface_hub import HfFileSystem

try:
    from kaggle_secrets import UserSecretsClient
    for lab in ("HF_TOKEN", "HF_Token"):
        try:
            os.environ["HF_TOKEN"] = UserSecretsClient().get_secret(lab)
            break
        except Exception:
            pass
except Exception:
    pass

fs = HfFileSystem()
REPO = "datasets/UCSC-VLAA/GPT-Image-Edit-1.5M/gpt-edit/"
OUT = "/kaggle/working/examples"
os.makedirs(OUT, exist_ok=True)
report = {}


def thumb(data, path, side=448):
    im = Image.open(io.BytesIO(data)).convert("RGB")
    size = im.size
    im.thumbnail((side, side))
    im.save(path, "JPEG", quality=90)
    return size


def probe(part, budget_s=700, want=6):
    t0, seen, counts, first = time.time(), 0, Counter(), {}
    pending = {}                 # (task_dir, file_id) -> input saved
    sizes = defaultdict(list)
    saved_pairs = []
    with fs.open(REPO + part, "rb") as f:
        tf = tarfile.open(fileobj=gzip.GzipFile(fileobj=f), mode="r|")
        for m in tf:
            if time.time() - t0 > budget_s:
                break
            if not m.isfile():
                continue
            seen += 1
            parts = m.name.split("/")           # dataset/task/(input|output|metadata)/file
            if len(parts) < 4:
                continue
            ds, task, kind, fname = parts[0], parts[1], parts[2], parts[-1]
            counts[(task, kind)] += 1
            first.setdefault((task, kind), (m.name, seen))
            if kind == "input" and len(pending) < want and (task, fname) not in pending:
                data = tf.extractfile(m).read()
                p = f"{OUT}/{ds}_{task}_{fname.split('.')[0]}_in.jpg"
                osz = thumb(data, p)
                pending[(task, fname)] = {"in": p, "in_size": osz}
            elif kind == "output" and (task, fname) in pending and "out" not in pending[(task, fname)]:
                data = tf.extractfile(m).read()
                p = f"{OUT}/{ds}_{task}_{fname.split('.')[0]}_out.jpg"
                pending[(task, fname)]["out_size"] = thumb(data, p)
                pending[(task, fname)]["out"] = p
                saved_pairs.append(pending[(task, fname)])
            if kind in ("input", "output") and len(sizes[(task, kind)]) < 15 and seen % 50 == 0:
                try:
                    sizes[(task, kind)].append(Image.open(io.BytesIO(tf.extractfile(m).read())).size)
                except Exception:
                    pass
            if seen % 2000 == 0:
                print(part, "members", seen, "secs", round(time.time() - t0), flush=True)
            if len(saved_pairs) >= want:
                break
    return {"members_seen": seen, "secs": round(time.time() - t0),
            "counts": {f"{k[0]}/{k[1]}": v for k, v in counts.items()},
            "first_member": {f"{k[0]}/{k[1]}": v for k, v in first.items()},
            "sample_sizes": {f"{k[0]}/{k[1]}": v for k, v in sizes.items()},
            "pairs_saved": saved_pairs}


for part in ("ultraedit.tar.gz.part001", "omniedit.tar.gz.part001", "hqedit.tar.gz.part001"):
    try:
        report[part] = probe(part)
    except Exception as e:
        report[part] = {"error": repr(e)}
    print(part, json.dumps(report[part])[:1500], flush=True)
    json.dump(report, open("/kaggle/working/edit_probe_report.json", "w"), default=str, indent=1)
print("DONE")
