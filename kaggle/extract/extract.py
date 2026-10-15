"""v3 feature extraction on a Kaggle GPU.

Streams images from Hugging Face, builds 224x224 patches at native resolution (no resizing of the
patch itself), applies the same JPEG-q95 patch normalisation as the app, runs the frozen CLIP
ViT-L/14 image encoder in half precision and saves features + labels. Nothing is trained here.

Sets produced (each saved in chunks under /kaggle/working/feats/):
  train_openfake   OpenFake core/train: random row groups + extra OpenAI/Google images.
                   Per image: 3 patches from the clean image + 3 from a randomly degraded copy
                   (resize / blur / JPEG 50-95)  -> this is the "JPEG augmentation".
  val_openfake     OpenFake core/validation, app layout (centre + 3x3 grid), 3 variants.
  test_openfake    OpenFake core/test (held-out generators: GPT-Image-1.5, Nano Banana Pro...), 3 variants.
  test_reddit      OpenFake reddit/test (in-the-wild, already shared/compressed), clean variant.
  edit_diffseg     DiffSeg30k validation: diffusion-edited images with masks -> patches inside the edit,
                   patches outside it (kept separately because their label is uncertain).
  edit_tinyaudits  tiny-audits COCO: original / manipulated / mask triplets.
Variants for eval sets: clean, jpeg70 (whole image JPEG q70), half_q75 (half size then JPEG q75).
"""
import copy, io, json, os, random, subprocess, sys, time, traceback, zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch", "huggingface_hub", "pyarrow"], check=False)

import numpy as np
import torch
from PIL import Image, ImageFilter

# ------------------------------------------------------------------ config
SCALE = 1.0             # 1.0 = full run; 0.02 = quick smoke test
SEED = 0
K_TRAIN = 3             # patches per variant (train images)
OUT = "/kaggle/working/feats"
os.makedirs(OUT, exist_ok=True)
CROP = 224
WORKERS = 6
DEADLINE = float('inf')    # set per stage in main(): no new tasks are started after it, so a stage always ends cleanly
SCAN_WORKERS = 24       # scanning only reads metadata columns; it is network-latency bound
OPENAI_KEYS = ["gpt-image", "dalle", "dall-e", "sora"]
GOOGLE_KEYS = ["nano-banana", "imagen", "gemini", "veo"]

try:
    from kaggle_secrets import UserSecretsClient
    _client, _tok = UserSecretsClient(), None
    for _label in ("HF_TOKEN", "HF_Token", "hf_token"):      # Kaggle secret labels are case-sensitive
        try:
            _tok = _client.get_secret(_label)
            break
        except Exception:
            continue
    if _tok:
        os.environ["HF_TOKEN"] = _tok
        print("HF token found (anonymous limits avoided)")
    else:
        print("no HF token secret attached; using anonymous access")
except Exception:
    print("no HF token; using anonymous access")

from huggingface_hub import HfFileSystem, hf_hub_download
import pyarrow.parquet as pq
fs = HfFileSystem()


def retry(fn, tries=5):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1) + random.random())


def n_of(x):
    return max(1, int(round(x * SCALE)))


def family(model):
    m = str(model).lower()
    if any(k in m for k in OPENAI_KEYS):
        return "openai"
    if any(k in m for k in GOOGLE_KEYS):
        return "google"
    return "other"


# ------------------------------------------------------------------ CLIP (frozen, image part only)
import open_clip
from open_clip.model import _build_vision_tower
from safetensors import safe_open

cfg = open_clip.get_model_config("ViT-L-14-quickgelu")
with torch.device("meta"):
    visual = _build_vision_tower(cfg["embed_dim"], cfg["vision_cfg"], quick_gelu=True)
path = hf_hub_download("timm/vit_large_patch14_clip_224.openai", "open_clip_model.safetensors")
with safe_open(path, framework="pt") as f:
    w = {k[len("visual."):]: f.get_tensor(k) for k in f.keys() if k.startswith("visual.")}
visual.load_state_dict(w, assign=True)
visual32 = visual.cuda().eval()
visual16 = copy.deepcopy(visual32).half().eval()
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device="cuda").view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device="cuda").view(1, 3, 1, 1)


def prep(arr, half=True):
    x = torch.from_numpy(arr).cuda().permute(0, 3, 1, 2).float() / 255.0
    x = (x - MEAN) / STD
    return x.half() if half else x


@torch.no_grad()
def encode(arr, bs=128):
    out = []
    for i in range(0, len(arr), bs):
        out.append(visual16(prep(arr[i:i + bs])).float().cpu().numpy().astype(np.float16))
    return np.concatenate(out) if out else np.zeros((0, 768), np.float16)


# ------------------------------------------------------------------ image helpers (same as the app)
def ensure_min(img, size=CROP):
    w, h = img.size
    if min(w, h) >= size:
        return img
    s = size / min(w, h)
    return img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)


def jpeg_rt(img, q):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=int(q))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def grid_boxes(w, h, n, size=CROP):
    xs = np.linspace(0, w - size, n).round().astype(int)
    ys = np.linspace(0, h - size, n).round().astype(int)
    return [(int(x), int(y), int(x) + size, int(y) + size) for y in ys for x in xs]


def to_patch(img, box):
    return np.asarray(jpeg_rt(img.crop(box), 95))          # q95 patch normalisation, as in the app


def load_rgb(data):
    img = Image.open(io.BytesIO(data))
    img.load()
    return img.convert("RGB")


def degrade(img, rng, mask=None):
    """Random real-world damage: resize, blur, JPEG. Returns (image, mask, description)."""
    desc = []
    if rng.random() < 0.5:
        s = rng.uniform(0.5, 1.0)
        size = (max(32, int(img.width * s)), max(32, int(img.height * s)))
        rs = [Image.BILINEAR, Image.BICUBIC, Image.LANCZOS][rng.integers(3)]
        img = img.resize(size, rs)
        if mask is not None:
            mask = mask.resize(size, Image.NEAREST)
        desc.append(f"s{s:.2f}")
    if rng.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
        desc.append("blur")
    if rng.random() < 0.9:
        q = int(rng.integers(50, 96))
        img = jpeg_rt(img, q)
        desc.append(f"q{q}")
    return img, mask, "_".join(desc) or "none"


def eval_variant(img, name):
    if name == "clean":
        return img
    if name == "jpeg70":
        return jpeg_rt(img, 70)
    if name == "half_q75":
        return jpeg_rt(img.resize((max(32, img.width // 2), max(32, img.height // 2)), Image.BICUBIC), 75)
    raise ValueError(name)


def app_boxes(img):
    img = ensure_min(img)
    w, h = img.size
    cx, cy = (w - CROP) // 2, (h - CROP) // 2
    return img, [(cx, cy, cx + CROP, cy + CROP)] + grid_boxes(w, h, 3)      # centre first, then 3x3


def rand_boxes(img, rng, k):
    w, h = img.size
    return [(int(x), int(y), int(x) + CROP, int(y) + CROP)
            for x, y in zip(rng.integers(0, w - CROP + 1, k), rng.integers(0, h - CROP + 1, k))]


# ------------------------------------------------------------------ output buffer
class Sink:
    def __init__(self, name, flush_at=20000):
        self.name, self.flush_at, self.n_saved, self.chunk = name, flush_at, 0, 0
        self.patches, self.meta = [], defaultdict(list)

    def add(self, patches, meta_rows):
        self.patches.extend(patches)
        for row in meta_rows:
            for k, v in row.items():
                self.meta[k].append(v)
        if len(self.patches) >= self.flush_at:
            self.flush()

    def flush(self):
        if not self.patches:
            return
        t0 = time.time()
        arr = np.stack(self.patches)
        feats = encode(arr)
        np.savez_compressed(f"{OUT}/{self.name}_{self.chunk:03d}.npz", feats=feats,
                            **{k: np.array(v) for k, v in self.meta.items()})
        self.n_saved += len(feats)
        print(f"[{self.name}] chunk {self.chunk}: {len(feats)} patches encoded+saved in {time.time()-t0:.0f}s "
              f"(total {self.n_saved})", flush=True)
        self.chunk += 1
        self.patches, self.meta = [], defaultdict(list)


def run(tasks, worker, sink, window=WORKERS + 2, label=""):
    """Run worker(task) in threads (bounded window), push results into sink from the main thread."""
    t0, done, fails = time.time(), 0, 0
    with ThreadPoolExecutor(WORKERS) as ex:
        pending, it = set(), iter(tasks)
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < window:
                if time.time() > DEADLINE:
                    print(f"[{label}] stage time budget reached; stopping with what we have", flush=True)
                    exhausted = True
                    break
                try:
                    pending.add(ex.submit(worker, next(it)))
                except StopIteration:
                    exhausted = True
            if not pending:
                break
            fin, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fu in fin:
                try:
                    p, m = fu.result()
                    sink.add(p, m)
                except Exception as e:
                    fails += 1
                    print(f"[{label}] task failed: {e!r}", flush=True)
                done += 1
                if done % 20 == 0:
                    print(f"[{label}] {done} tasks, {time.time()-t0:.0f}s, failures {fails}", flush=True)
    sink.flush()
    print(f"[{label}] finished: {done} tasks, {fails} failures, {time.time()-t0:.0f}s", flush=True)


# ------------------------------------------------------------------ OpenFake planning
def scan_split(pattern, n_files, rng):
    files = sorted(fs.glob(pattern))
    if n_files < len(files):
        files = [files[i] for i in sorted(rng.choice(len(files), n_files, replace=False))]
    out = []                                              # (file, rg, row_in_rg, label, model)

    def one(fp):
        def go():
            with fs.open(fp, "rb") as f:
                pf = pq.ParquetFile(f)
                t = pf.read(columns=["label", "model"])
                offs = np.cumsum([0] + [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])
            return fp, offs, t.column("label").to_pylist(), t.column("model").to_pylist()
        return retry(go)
    with ThreadPoolExecutor(SCAN_WORKERS) as ex:
        for fp, offs, labels, models in ex.map(one, files):
            for rg in range(len(offs) - 1):
                for r in range(offs[rg + 1] - offs[rg]):
                    out.append((fp, rg, r, labels[offs[rg] + r], models[offs[rg] + r]))
    print(f"scanned {len(files)} files of {pattern.split('/')[-1]}: {len(out)} rows", flush=True)
    return out


def group_rows(rows):
    by = defaultdict(list)
    for fp, rg, r, lab, mod in rows:
        by[(fp, rg)].append((r, lab, mod))
    return by


def pick_rgs(rows, n_random, targets, n_targeted, rng):
    by = group_rows(rows)
    keys = sorted(by)
    rand = [keys[i] for i in rng.choice(len(keys), min(n_random, len(keys)), replace=False)]
    chosen = set(rand)
    targeted = []
    if targets is not None and n_targeted:
        scored = sorted(((sum(1 for _, _, m in by[k] if targets(m)), k) for k in keys if k not in chosen), reverse=True)
        targeted = [k for s, k in scored[:n_targeted] if s > 0]
    return by, rand, targeted


def openfake_worker_factory(mode, variants=None):
    """mode 'train': K random patches from clean + degraded copy. mode 'eval': app layout per variant."""
    def worker(task):
        (fp, rg), keep_rows, seed = task
        rng = np.random.default_rng(seed)

        def go():
            with fs.open(fp, "rb") as f:
                pf = pq.ParquetFile(f)
                return pf.read_row_group(rg, columns=["image", "label", "model"])
        t = retry(go)
        imgs, labels, models = t.column("image").to_pylist(), t.column("label").to_pylist(), t.column("model").to_pylist()
        patches, meta = [], []
        for r, _, _ in keep_rows:
            try:
                img = ensure_min(load_rgb(imgs[r]["bytes"]))
                lab, mod = int(labels[r] != "real"), models[r]
                iid = f"{os.path.basename(fp)}:{rg}:{r}"
                if mode == "train":
                    views = [("clean", img)]
                    d, _, desc = degrade(img, rng)
                    views.append((desc, ensure_min(d)))
                    for vname, v in views:
                        for box in rand_boxes(v, rng, K_TRAIN):
                            patches.append(to_patch(v, box))
                            meta.append(dict(img_id=iid, src=mod, label=lab, kind="full",
                                             variant="clean" if vname == "clean" else "aug:" + vname,
                                             x=box[0], y=box[1], frac=np.nan, slot=-1))
                else:
                    for vname in variants:
                        v, boxes = app_boxes(eval_variant(img, vname))
                        for slot, box in enumerate(boxes):
                            patches.append(to_patch(v, box))
                            meta.append(dict(img_id=iid, src=mod, label=lab, kind="full", variant=vname,
                                             x=box[0], y=box[1], frac=np.nan, slot=slot))
            except Exception as e:
                print("skip image:", repr(e)[:100], flush=True)
        return patches, meta
    return worker


def cap_rows(by, rgs, cap_per_model, real_cap, rng):
    """Pick rows inside the chosen row groups, capped per generator and for real images."""
    pool = defaultdict(list)
    for k in rgs:
        for r, lab, mod in by[k]:
            pool["real" if lab == "real" else mod].append((k, r, lab, mod))
    chosen = defaultdict(list)
    for src, items in pool.items():
        cap = real_cap if src == "real" else cap_per_model
        for i in rng.permutation(len(items))[:cap]:
            k, r, lab, mod = items[i]
            chosen[k].append((r, lab, mod))
    return chosen


def stage_openfake_eval(name, pattern, n_files, n_rgs, targets, n_targeted, per_model, real_cap, variants, seed):
    rng = np.random.default_rng(seed)
    rows = scan_split(pattern, n_files, rng)
    by, rand, targeted = pick_rgs(rows, n_of(n_rgs), targets, n_of(n_targeted) if n_targeted else 0, rng)
    chosen = cap_rows(by, rand + targeted, per_model, real_cap, rng)
    tasks = [(k, v, int(rng.integers(1 << 30))) for k, v in chosen.items()]
    cnt = Counter(("real" if lab == "real" else mod) for v in chosen.values() for _, lab, mod in v)
    print(f"[{name}] {len(tasks)} row groups, {sum(cnt.values())} images; top sources: {cnt.most_common(15)}", flush=True)
    json.dump(cnt, open(f"{OUT}/{name}_plan.json", "w"))
    run(tasks, openfake_worker_factory("eval", variants), Sink(name, 12000), label=name)


def stage_train():
    rng = np.random.default_rng(SEED)
    rows = scan_split("datasets/ComplexDataLab/OpenFake/core/train-*.parquet", n_of(150), rng)
    tg = lambda m: family(m) != "other"
    by, rand, targeted = pick_rgs(rows, n_of(420), tg, n_of(220), rng)
    tasks = [(k, by[k], int(rng.integers(1 << 30))) for k in rand]
    tasks += [(k, [x for x in by[k] if tg(x[2])], int(rng.integers(1 << 30))) for k in targeted]
    cnt = Counter(("real" if lab == "real" else mod) for _, rowsk, _ in tasks for _, lab, mod in rowsk)
    fam = Counter(family(m) for m in cnt.elements() if m != "real")
    print(f"[train] {len(tasks)} row groups, {sum(cnt.values())} images; families {dict(fam)}; top {cnt.most_common(12)}", flush=True)
    json.dump(cnt, open(f"{OUT}/train_openfake_plan.json", "w"))
    run(tasks, openfake_worker_factory("train"), Sink("train_openfake", 12000), label="train")


# ------------------------------------------------------------------ edited-image sets
def patches_with_mask(img, mask, rng, n_edit=4, n_bg=2, n_mixed=2, tries=300):
    """Random windows classified by the share of masked (edited) pixels inside them."""
    w, h = img.size
    if w < CROP or h < CROP:
        return []
    m = (np.asarray(mask) > 0).astype(np.float32)
    ii = np.pad(m, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    xs, ys = rng.integers(0, w - CROP + 1, tries), rng.integers(0, h - CROP + 1, tries)
    out, got = [], Counter()
    limits = {"edit": n_edit, "bg": n_bg, "mixed": n_mixed}
    for x, y in zip(xs, ys):
        frac = (ii[y + CROP, x + CROP] - ii[y, x + CROP] - ii[y + CROP, x] + ii[y, x]) / (CROP * CROP)
        kind = "edit" if frac >= 0.5 else "bg" if frac == 0 else "mixed" if frac > 0.05 else None
        if kind and got[kind] < limits[kind]:
            got[kind] += 1
            out.append((kind, (int(x), int(y), int(x) + CROP, int(y) + CROP), float(frac)))
    return out


DIFFSEG_NAMES = {1: "sd2-inpaint", 2: "kolors", 3: "sd3.5", 4: "flux", 5: "sdxl-inpaint", 6: "glide", 7: "hunyuandit", 8: "kandinsky"}
KIND_LABEL = {"edit": 1, "bg": -1, "mixed": -2}


def diffseg_worker(task):
    names, seed = task
    rng = np.random.default_rng(seed)
    patches, meta = [], []
    f = retry(lambda: fs.open("datasets/Chaos2629/Diffseg30k/validation.zip", "rb"))
    z = zipfile.ZipFile(f)
    for stem in names:
        try:
            img = load_rgb(z.read(stem + ".image.png"))
            mask = Image.open(io.BytesIO(z.read(stem + ".mask.png"))).convert("L")
            if mask.size != img.size:
                mask = mask.resize(img.size, Image.NEAREST)
            marr = np.asarray(mask)
            vals = marr[marr > 0]
            editor = DIFFSEG_NAMES.get(int(np.bincount(vals).argmax()), "unknown") if vals.size else "none"
            for vname in ("clean", "aug"):
                v, m, desc = (img, mask, "clean") if vname == "clean" else degrade(img, rng, mask)
                for kind, box, frac in patches_with_mask(v, m, rng):
                    patches.append(to_patch(v, box))
                    meta.append(dict(img_id=stem, src=editor, label=KIND_LABEL[kind], kind=kind,
                                     variant="clean" if vname == "clean" else "aug:" + desc,
                                     x=box[0], y=box[1], frac=frac, slot=-1))
        except Exception as e:
            print("skip diffseg", stem, repr(e)[:100], flush=True)
    f.close()
    return patches, meta


def stage_diffseg():
    rng = np.random.default_rng(SEED + 1)
    f = retry(lambda: fs.open("datasets/Chaos2629/Diffseg30k/validation.zip", "rb"))
    names = sorted({n[:-len(".image.png")] for n in zipfile.ZipFile(f).namelist() if n.endswith(".image.png")})
    f.close()
    pick = [names[i] for i in rng.permutation(len(names))[:n_of(3000)]]
    tasks = [(pick[i:i + 40], int(rng.integers(1 << 30))) for i in range(0, len(pick), 40)]
    print(f"[edit_diffseg] {len(pick)} images in {len(tasks)} tasks", flush=True)
    run(tasks, diffseg_worker, Sink("edit_diffseg", 15000), label="edit_diffseg")


def tinyaudits_worker(task):
    items, seed = task                      # items: (gen, stem, manip_path, mask_path, orig_path or None)
    rng = np.random.default_rng(seed)
    patches, meta = [], []
    for gen, stem, mp, kp, op in items:
        try:
            img = load_rgb(retry(lambda: fs.cat_file(mp)))
            mask = Image.open(io.BytesIO(retry(lambda: fs.cat_file(kp)))).convert("L")
            mask = mask.point(lambda p: 255 if p > 127 else 0)
            if mask.size != img.size:
                mask = mask.resize(img.size, Image.NEAREST)
            for vname in ("clean", "aug"):
                v, m, desc = (img, mask, "clean") if vname == "clean" else degrade(img, rng, mask)
                vl = "clean" if vname == "clean" else "aug:" + desc
                for kind, box, frac in patches_with_mask(v, m, rng, n_edit=3, n_bg=1, n_mixed=1):
                    patches.append(to_patch(v, box))
                    meta.append(dict(img_id=f"{gen}/{stem}", src=gen, label=KIND_LABEL[kind], kind=kind,
                                     variant=vl, x=box[0], y=box[1], frac=frac, slot=-1))
            if op:
                orig = load_rgb(retry(lambda: fs.cat_file(op)))
                for vname in ("clean", "aug"):
                    v, _, desc = (orig, None, "clean") if vname == "clean" else degrade(orig, rng)
                    v = ensure_min(v)
                    for box in rand_boxes(v, rng, 3):
                        patches.append(to_patch(v, box))
                        meta.append(dict(img_id=f"{gen}/{stem}", src=gen, label=0, kind="orig",
                                         variant="clean" if vname == "clean" else "aug:" + desc,
                                         x=box[0], y=box[1], frac=0.0, slot=-1))
        except Exception as e:
            print("skip tinyaudits", gen, stem, repr(e)[:100], flush=True)
    return patches, meta


def stage_tinyaudits():
    rng = np.random.default_rng(SEED + 2)
    files = retry(lambda: fs.find("datasets/DivyaApp/tiny-audits/COCO"))
    table = defaultdict(dict)
    for p in files:
        parts = p.split("/")
        gen, typ, fname = parts[-3], parts[-2], parts[-1]
        if typ in ("manipulated", "mask", "original"):
            table[(gen, fname.rsplit(".", 1)[0])][typ] = p
    items = defaultdict(list)
    for (gen, stem), d in table.items():
        if "manipulated" in d and "mask" in d:
            items[gen].append((gen, stem, d["manipulated"], d["mask"], d.get("original")))
    allitems = []
    for gen, lst in sorted(items.items()):
        allitems += [lst[i] for i in rng.permutation(len(lst))[:n_of(250)]]
    tasks = [(allitems[i:i + 20], int(rng.integers(1 << 30))) for i in range(0, len(allitems), 20)]
    print(f"[edit_tinyaudits] {len(allitems)} triplets, generators {sorted(items)}", flush=True)
    run(tasks, tinyaudits_worker, Sink("edit_tinyaudits", 15000), label="edit_tinyaudits")


# ------------------------------------------------------------------ main
def fp16_check():
    """Do half-precision features agree with full precision? (the app runs full precision)"""
    rng = np.random.default_rng(0)
    arr = (rng.random((32, 224, 224, 3)) * 255).astype(np.uint8)
    with torch.no_grad():
        a = visual32(prep(arr, half=False)).float().cpu().numpy()
    b = encode(arr).astype(np.float32)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    print(f"fp16 vs fp32 feature cosine: min {cos.min():.5f} mean {cos.mean():.5f}", flush=True)
    return float(cos.min())


EVAL_VARIANTS = ["clean", "jpeg70", "half_q75"]
STAGES = [
    ("fp16_check", fp16_check),
    ("edit_tinyaudits", stage_tinyaudits),
    ("edit_diffseg", stage_diffseg),
    ("val_openfake", lambda: stage_openfake_eval(
        "val_openfake", "datasets/ComplexDataLab/OpenFake/core/validation-*.parquet", 15, 20, None, 0, 150, 600, EVAL_VARIANTS, 11)),
    ("test_openfake", lambda: stage_openfake_eval(
        "test_openfake", "datasets/ComplexDataLab/OpenFake/core/test-*.parquet", 13, 40,
        lambda m: str(m) in ("nano-banana-pro", "nano-banana", "veo-3", "sora-2"), 20, 250, 1200, EVAL_VARIANTS, 12)),
    ("test_reddit", lambda: stage_openfake_eval(
        "test_reddit", "datasets/ComplexDataLab/OpenFake/reddit/test-*.parquet", 5, 25, None, 0, 100000, 100000, ["clean"], 13)),
    ("train_openfake", stage_train),
]

BUDGET = {"fp16_check": 300, "edit_tinyaudits": 1500, "edit_diffseg": 1500, "val_openfake": 2400,
          "test_openfake": 3000, "test_reddit": 1500, "train_openfake": 10800}      # seconds per stage

if __name__ == "__main__":
    T0, status = time.time(), {}
    for name, fn in STAGES:
        t0 = time.time()
        DEADLINE = t0 + BUDGET[name]
        try:
            status[name] = {"ok": True, "result": fn(), "secs": round(time.time() - t0)}
        except Exception:
            traceback.print_exc()
            status[name] = {"ok": False, "secs": round(time.time() - t0)}
        json.dump(status, open(f"{OUT}/status.json", "w"), default=str)
        print("STAGE", name, status[name], flush=True)
    print("ALL DONE in", round(time.time() - T0), "s", flush=True)
