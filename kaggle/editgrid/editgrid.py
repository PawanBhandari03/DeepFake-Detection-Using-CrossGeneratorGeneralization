"""Edited-image grid test on a Kaggle GPU.

The v3 edit features (kaggle/extract) are patches sampled inside / outside the edit mask, so they cannot say how
the app would judge a whole edited image. This job takes the edited images that were held out of v3 training
(the same 15 % rule as prototype/train_v3.py, ids in held_ids.txt) and extracts the app layout for each:
one centre patch + a 3x3 grid, JPEG-q95 patches, frozen CLIP ViT-L/14 features, in two variants (clean, jpeg70).
For every patch it also stores the share of edited pixels inside it (from the mask), and for the
tiny-audits triplets the same layout of the ORIGINAL (unedited) photo. Nothing is trained here.
"""
import base64, copy, io, json, os, subprocess, sys, time, traceback, zipfile, zlib
from collections import defaultdict

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch", "huggingface_hub"], check=False)

import numpy as np
import torch
from PIL import Image

OUT = "/kaggle/working/feats"
os.makedirs(OUT, exist_ok=True)
CROP = 224

try:
    from kaggle_secrets import UserSecretsClient
    _c, _t = UserSecretsClient(), None
    for _l in ("HF_TOKEN", "HF_Token", "hf_token"):
        try:
            _t = _c.get_secret(_l)
            break
        except Exception:
            continue
    if _t:
        os.environ["HF_TOKEN"] = _t
except Exception:
    pass

from huggingface_hub import HfFileSystem, hf_hub_download
fs = HfFileSystem()


def retry(fn, tries=5):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


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
visual16 = visual.cuda().half().eval()
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device="cuda").view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device="cuda").view(1, 3, 1, 1)


@torch.no_grad()
def encode(arr, bs=128):
    out = []
    for i in range(0, len(arr), bs):
        x = torch.from_numpy(arr[i:i + bs]).cuda().permute(0, 3, 1, 2).float() / 255.0
        out.append(visual16(((x - MEAN) / STD).half()).float().cpu().numpy().astype(np.float16))
    return np.concatenate(out) if out else np.zeros((0, 768), np.float16)


def ensure_min(img, size=CROP):
    w, h = img.size
    if min(w, h) >= size:
        return img, 1.0
    s = size / min(w, h)
    return img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC), s


def jpeg_rt(img, q):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=int(q))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def grid_boxes(w, h, n, size=CROP):
    xs = np.linspace(0, w - size, n).round().astype(int)
    ys = np.linspace(0, h - size, n).round().astype(int)
    return [(int(x), int(y), int(x) + size, int(y) + size) for y in ys for x in xs]


def layout(img, mask):
    """centre patch first, then the 3x3 grid (as the app); mask resized with the image if it was upscaled."""
    img, _ = ensure_min(img)
    if mask is not None and mask.size != img.size:
        mask = mask.resize(img.size, Image.NEAREST)
    w, h = img.size
    cx, cy = (w - CROP) // 2, (h - CROP) // 2
    boxes = [(cx, cy, cx + CROP, cy + CROP)] + grid_boxes(w, h, 3)
    m = (np.asarray(mask) > 0).astype(np.float32) if mask is not None else None
    return img, boxes, m


def load_rgb(data):
    img = Image.open(io.BytesIO(data))
    img.load()
    return img.convert("RGB")


HELD_B64 = "eNqNm12rJLkNQP/LPgdiy5YsvYbdLIElLOQlEEKYZGdgYJgJm0lC/n1UXd13u8s+VXu5TzpVLlmW9eGq/ss3v/v0/vNP73/69uOHD//+18cvn39b9j9vtcg3v2HeXesJj9a8M68t8v+EaxlxxkdY2AmPaD6Yi4xu7YRbVz3RX8K6nfDWeul2xq2Pk/k1bRrlhIfU0U9515P597zC6xn3XONT3svp+K7tjEtPBzrhvbVoZzwNdHq/eT3Tf1gtZ+N79HFif60u7cT+KtsOOOMR9Wx8G2087b8f3n19//nrtMtKG89WhqvS12q/virM4ldc5eVaL03n08urQrRfPrFWi+dIA1dJj1Yur8ptFZfa1/StuH6i6miXc6zuVa71ihwrfs1Vdq1XOla/vErEZVz6hKi35ygLV1nGossntmpp/curWu/X2jcTv7ZXGypyqX2mqRaX65hu49d69WLjetf2KrVfX2UjR7u8alQTvb5qDL++yiXqtV6esfn6qsj0cun3Kl710gszkbRyPVYu5Li+KuNla9dXZXTfVvv3H39+/+HT/97CrZabGQ/iuqXTWZyOXMcs1mK3iHAU17DFIFvAilk86h74j+LedCW2KPN0am0uvhD36rMmVTLwz5rkGLqYfNVWV2KvmVkX4kz2C018aJsHkWJ9Je5qi0eKbvOZxeZ6K9AO4owZoyzE0WPWW9z2rPMqbrLty4VYXGdNshQsK3GWKH0b+/sf/vDtd7+k8r2sfBVGuUXWF2G6uYyjMOeschBmNGo3N3gRti5xvL1GVrV6EGahs5v9RShWbLpSopTjjKRbnR4kZtqO08x1sd4PwlZ8mE3C0FvJ8CKsW0l6FPZQma5UL3pUPpNJ9+n2DF/zg3If29HIvW4zPQrTxPVokL5l0zYJQ27twrNQS5ZGxyvfCs8XYcsFPaqk6i6T8FF3pvD77/74Fvgy37ZZmgFXZ+mQh0mfpHLvJ1+lrarN13apNo+bV7ZphFzUUXySbjtwvjZLfiuTNPvMPkklk2GZZpz1allJrbUxSR8d5EEa+2Z/lT659rPUxOanjXCJSbol0jpLs/Sf7CCRre80QpM0j8/S0eukQ4asvft8lT5tpiepZoEyFlJ1W0htnkU6ZLrJJPUYOs0iU27MVs+sWOpkh57u69PK95H1zUI6bPb1t7rqVRrmZRohS4fikyXTYNoW0tFmq2uXUccs9bG4VnN6fZZGnT1KcxePmKXZnG1Po0ZQ45bjsE/ccxLgx/ESdXMtC+kTHO7OOFvBxqq9HS0Rbvs6IQ4/UW1kgdMRy+0axjVd8uRu9X5zLOhRsnRTP8GZp0/uzgCg/Oy3/Edd1PDORm2R7sCD9xrF+e4uIifO1LP6rue487zT4kXYapqwn+Ds7k+erfXeblD/kzX4Ce7py3aC7/EAsGU7d3L3I9P/+OW/73/+8d3Hz1/fcrXqsCXJRsyWpNatPFyS3Iu33TgTr3tPPZO49zELor0tydvWmsnIbVHW5J5hZ5Lt6+18dyK5T/YieEHuZxcLcj/tnYlmUexLYt3L0tYtc1FbrtzbqeqCWNQ1yfXRpdY9E1ldWie7G13Pp4+m1dbknldn4rlLlkRLiCxXW7fjdV2T+8nDTDJ82no0y322XAXNeB6xJq3e9sKfvv3zD79UxHE793yWZYHZj9dlgXs7kX2R3Xfds2xrG2SW3Wz/LPPMHMfxPBe8HGSRXfxBv7rVJX6Q1SzvYpKFTddl3VmPsr6H22dZBnA76LfFhX7U5dF1vsjGfgb5Iru/PXqSZRyucXiu9Cwk9Ci7N5zPMu3djveOvr/BeJZlw1T8KMtIcbCBZEt8tMGWfvthvNx5+8nRs+yRh19kfa9UX2Xux3v7ljKOsoh6sFXT+1HJi8zHOD7jUXk/yfrWExxssMW9dtC5a7rgYb5vufZFtnWsR5mWdli33NZdD9dp7aKH+WqG56Ofqlgps8yP+mUwbtMzMhOVgw00m/Tdfl/f/f3T+/m1jPfb8QXgRy9MOGPICX68WyE8zE5Ue7y3XeNfNtsa57TsBFvbqzzAo+0nn4AzbN/K1zWW7VCH5y3ZFIx+gu+VNWDxvScArPcaBnCWEXGiWtwLsTXONHIPXIDjFKdNOz+7Zat/st7ZF/seDNfYRpSTwYepstUyRQ+vJ/j+ihjwo2IB3B/hZY2zTSts87RKPdlEPQuszltQN3eqJ7hnA8s4G4qT/b21WYMn9vYC4z/vPn386d3XJH/7x5d/Pk6Nb283l6jeXhOtkfJdg9FtbZdoT5JrdNsJS9RvG3SJ9NbnrFHDZ5nhXd47I77r9oJ8iYKVj0FTzjKLUdCz0rmEEQ64h6glamiNRKhGGw2Roxq9kdukG6KGyvPae7slMiXfqF5QeccpC+8UYZ+XvYdZI9yw0itZQxTNKxqIrKOGZgWRk6HEG97l/KwQ1DBwXllYFEZ4l6CzZZUtiAY+SzDoZTtDbpO5RBGhA9wb6TUSvItXOVshVGNg+GqjkwM0x0C0fS8BKPvXigh9o1cXRrS/uqDlE6GGEjTl3gYq3/guLahG9iuI0EXvLcAaOSKOh90GIyfH7iPwrv2Efo0wtW1NHiAtFZGgY2eXR8/KChEH7LgoqrUiQmdTxZCSCJ9l6ABqQT6vA+NGPgo1HJjadLDyrqi847ysdLKh8SZK5yUNzTAF2MCIff8oZoVGwXg4CtZsQ6Qhwvw1Gkbs+7vXNULHHoqLMhSz+eC4ke2tIDK0oeFeHsMQeSPHHlweeK0FES6Kc/ntFatlb9hxuKEN3fsJQjUczRsFk29wsRT716hr1AKRGSJclBCsiEKwh42GvV407AKiYdcWit1NKFo+eBOFYg0QirkyTPGuUVD5gaktHB07OMNG0NarZf/qZomEKoe6HRIR6lTO1aIU57dvdBANCUQdNXQKX7V4w2dhaksUaKiQiohib80/R0St2faSBpGQz9ft4wlCGNkqnwNsX5His7TgXYOSb9qi4Lyw46jbl/aAZP/4Z40cUaUaIJEGImuIKDtUEUXlBXelYNDbfg6B81Kel7IaAzeRBEWb2jBJJaLYW5tQ2bMhUr71igNiv5yIatHa8Oijcr+cCPdyG4F3Oa5ycyr1M2Hj1uuF+pRE1DzWLoURnQNsL3xRjR60l7tR1kuEq5xlLzlbH6yhYxRNhM8KDEQ9qF9OhNtcC66y4tFH3X4ZQKijs6lijFIzRLzNlc2rjjtFY9AqW8ENaxUj9v1l2RKJ4YCc9bIuY0TldxbEuL8MO99qIYzQsUelo6pEhgjfp2yI/HBwYTaMB8RTx0Q8r0H9VyIMlYPNO4LawESN7nLs6KtXrA8dO/oNkW9kvYlq8KJsP7lCxMoP6qRuv0pEhEvpPnhAXOWo9GYkETX7ibD8Di44o2HZE2zeRIYII0AoK4/nojWC5xVYi2ZsAGeT7eCZURCqVH1JETqpkNLodbBsH3wT6pToE1VUozsqjxFbts91ECkaCjuOrT9gZCcIlfeBU46Ghgp6x5FtCoUU4Xe+WWEXQVRxwDYQ4fvK7dsbYxSIgp8VaA0TvGuYMELlnaKNbFsFETU+iZQRNY8bCkS4KMLbXCp1AYnQ8oJvORNRSbx1tzgvM7zLKN1IdqM44Oi0KyVQjfuHa2tE9XwaA+NhNqoVESrf8H2ltIaBqDXFZynu5WYYRbNRxQG94V14AiMt0BpZpSoi9PmOZ1/S5QQ5omaoYaciMBEVS5lEcafwG9Xt2AY15OzQscnKIpWO4BIpoooBVgXjhrJ5tdGXGIno9CARq8HZQbWiGoZZT41eZIgOnpdTpScauM0Vv4FJhNYwfAMuJugbhn2KGJbEiVB563RilggzrHV0UVOM2NmA8130gcH2NRMOaIoDDkyjFvRqLGMNdVKJcCnvPzxdooqGGoKOPQQXZXBkG1zODV6voVizDcNgPrAnksGJY4TgszgeDt5EXjBu8DtfcWmM6DPIRNQUb5/OMeIBsb3dvik2RNg7OH5Ym0hxyoq9nhvmFB9YErsXXBTnAQMLGA/cDh5YY/NJxfbRIikfDdWIjt6bHmqIlBHW2GH05lFiYJyPgZsoHGu22L6R/uv/AXl3eOM="   # from kaggle/make_payload.py
held = json.loads(zlib.decompress(base64.b64decode(HELD_B64)))
print("held-out edited images:", len(held), flush=True)

patches, meta = [], defaultdict(list)


def add_image(img, mask, iid, src, kind):
    for variant in ("clean", "jpeg70"):
        v = img if variant == "clean" else jpeg_rt(img, 70)
        v, boxes, m = layout(v, mask)
        for slot, box in enumerate(boxes):
            patches.append(np.asarray(jpeg_rt(v.crop(box), 95)))
            frac = float(m[box[1]:box[3], box[0]:box[2]].mean()) if m is not None else 0.0
            for k, val in (("img_id", iid), ("src", src), ("kind", kind), ("variant", variant),
                           ("slot", slot), ("x", box[0]), ("y", box[1]), ("frac", frac)):
                meta[k].append(val)


t0 = time.time()
# ---- DiffSeg30k (validation.zip): edited image + mask
diff = [h for h in held if h.startswith("validation_copy/")]
if diff:
    DIFF = {1: "sd2-inpaint", 2: "kolors", 3: "sd3.5", 4: "flux", 5: "sdxl-inpaint", 6: "glide", 7: "hunyuandit", 8: "kandinsky"}
    z = zipfile.ZipFile(retry(lambda: fs.open("datasets/Chaos2629/Diffseg30k/validation.zip", "rb")))
    for i, stem in enumerate(diff):
        try:
            img = load_rgb(z.read(stem + ".image.png"))
            mask = Image.open(io.BytesIO(z.read(stem + ".mask.png"))).convert("L")
            if mask.size != img.size:
                mask = mask.resize(img.size, Image.NEAREST)
            arr = np.asarray(mask)
            vals = arr[arr > 0]
            editor = DIFF.get(int(np.bincount(vals).argmax()), "unknown") if vals.size else "none"
            add_image(img, mask, stem, editor, "edited")
        except Exception as e:
            print("skip diffseg", stem, repr(e)[:100], flush=True)
        if i % 50 == 0:
            print("diffseg", i, len(diff), f"{time.time()-t0:.0f}s", flush=True)

# ---- tiny-audits COCO: manipulated + mask + original
tiny = [h for h in held if not h.startswith("validation_copy/")]
if tiny:
    table = defaultdict(dict)
    for p in retry(lambda: fs.find("datasets/DivyaApp/tiny-audits/COCO")):
        parts = p.split("/")
        gen, typ, fname = parts[-3], parts[-2], parts[-1]
        if typ in ("manipulated", "mask", "original"):
            table[f"{gen}/{fname.rsplit('.', 1)[0]}"][typ] = p
    for i, iid in enumerate(tiny):
        d = table.get(iid)
        if not d or "manipulated" not in d or "mask" not in d:
            print("missing", iid, flush=True)
            continue
        try:
            img = load_rgb(retry(lambda: fs.cat_file(d["manipulated"])))
            mask = Image.open(io.BytesIO(retry(lambda: fs.cat_file(d["mask"])))).convert("L").point(lambda p: 255 if p > 127 else 0)
            if mask.size != img.size:
                mask = mask.resize(img.size, Image.NEAREST)
            gen = iid.split("/")[0]
            add_image(img, mask, iid, gen, "edited")
            if "original" in d:
                add_image(load_rgb(retry(lambda: fs.cat_file(d["original"]))), None, iid, gen, "orig")
        except Exception as e:
            print("skip tiny", iid, repr(e)[:100], flush=True)
        if i % 50 == 0:
            print("tiny", i, len(tiny), f"{time.time()-t0:.0f}s", flush=True)

print("patches to encode:", len(patches), flush=True)
feats = encode(np.stack(patches))
np.savez_compressed(f"{OUT}/editgrid.npz", feats=feats, **{k: np.array(v) for k, v in meta.items()})
print("DONE", len(feats), "patches in", round(time.time() - t0), "s", flush=True)
