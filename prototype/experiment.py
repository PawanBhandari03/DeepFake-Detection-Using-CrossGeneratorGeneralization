"""Cross-generator experiment: extract features, train our head on ONE generator, test on all.

    python prototype/experiment.py extract   # images -> CLIP + frequency features (slow, CPU)
    python prototype/experiment.py train     # train heads, evaluate per generator, save results

Data (downloaded from Hugging Face into data/raw/):
  - Tiny-GenImage  (subset of GenImage, NeurIPS 2023): real ImageNet photos + 8 generators
  - OpenFakeTiny   (subset of OpenFake, 2025): real Pexels/LAION photos + modern generators
"""
from __future__ import annotations

import io
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from detector import ClipEncoder, center_crop, jpeg_normalize, load_ufd_head, sigmoid  # noqa: E402
from forensics import frequency_features, load_image, spectral_peak_score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
FEATURES = ROOT / "data" / "features"
RESULTS = Path(__file__).parent / "results"
WEIGHTS = Path(__file__).parent / "weights"

GENIMAGE_NAMES = ["Real", "ADM", "BigGAN", "GLIDE", "Midjourney", "SD14", "SD15", "VQDM", "Wukong"]
GENIMAGE_YEARS = {"ADM": 2021, "BigGAN": 2018, "GLIDE": 2021, "Midjourney": 2022, "SD14": 2022,
                  "SD15": 2022, "VQDM": 2021, "Wukong": 2022}
SEED = 0

# How many images to use (kept small so it runs on a laptop CPU overnight)
N_TRAIN_PER_CLASS = 800
N_TEST_PER_GENERATOR = 150
N_TEST_REAL = 300
N_OPENFAKE_FAKE = 10_000   # i.e. all fake images in the OpenFake test file
N_OPENFAKE_REAL = 300

# Friendly names for well-known generators in OpenFake (others keep their dataset name)
FRIENDLY = {
    "gpt-image-1.5": "GPT-Image-1.5 (ChatGPT, OpenAI)",
    "nano-banana-pro": "Nano Banana Pro (Gemini, Google)",
    "veo-3": "Veo 3 video frames (Google)",
    "midjourney-7": "Midjourney v7",
    "flux.2-klein-9b": "FLUX.2 klein 9B",
    "z-image-turbo": "Z-Image Turbo",
    "illustrious": "Illustrious (SDXL fine-tune)",
    "recraft-v3": "Recraft v3",
    "wan-video-2.5": "Wan 2.5 video frames",
}


# ----------------------------------------------------------------------------
# Reading the parquet files
# ----------------------------------------------------------------------------
def _label_names(pf: pq.ParquetFile, column: str) -> list[str] | None:
    meta = json.loads(pf.schema_arrow.metadata.get(b"huggingface", b"{}") or b"{}")
    feat = meta.get("info", {}).get("features", {}).get(column, {})
    return feat.get("names")


def _year(value) -> int:
    text = str(value or "")[:4]
    return int(text) if text.isdigit() else 0


# Each sample is (file, row_index, source_name, label, year, dataset). Image bytes are read
# later, one file at a time, so the laptop's memory is never filled with all images at once.
def genimage_rows(split: str):
    for path in sorted((RAW / "Tiny-GenImage" / "data").glob(f"{split}-*.parquet")):
        t = pq.read_table(path, columns=["label", "generator"])
        for i, (label, gen) in enumerate(zip(t.column("label").to_pylist(), t.column("generator").to_pylist())):
            name = GENIMAGE_NAMES[gen]
            yield str(path), i, name, int(label), GENIMAGE_YEARS.get(name, 0), "GenImage"


def openfake_rows(split: str = "test"):
    for path in sorted((RAW / "OpenFakeTiny" / "core").glob(f"{split}-*.parquet")):
        label_names = _label_names(pq.ParquetFile(path), "label")
        t = pq.read_table(path, columns=["label", "model", "release_date"]).to_pylist()
        for i, row in enumerate(t):
            lab = row["label"]
            if isinstance(lab, int) and label_names:
                lab = label_names[lab]
            is_fake = int(str(lab).lower() not in ("real", "0"))
            year = _year(row.get("release_date")) if is_fake else 0
            source = FRIENDLY.get(str(row["model"]), str(row["model"])) if is_fake else "Real"
            yield str(path), i, source, is_fake, year, "OpenFake"


def stream_images(items):
    """Yield (image_bytes, source, label, year, dataset) for samples, reading each file in small batches."""
    by_file = defaultdict(list)
    for it in items:
        by_file[it[0]].append(it)
    for file, file_items in by_file.items():
        wanted = {it[1]: it for it in file_items}
        row = 0
        for batch in pq.ParquetFile(file).iter_batches(batch_size=32, columns=["image"]):
            for img in batch.column(0).to_pylist():
                if row in wanted:
                    yield (img["bytes"],) + tuple(wanted[row][2:])
                row += 1


def pick_samples() -> dict[str, list]:
    """Choose which images go into train / seen-test / unseen-test groups."""
    rng = np.random.default_rng(SEED)
    groups: dict[str, list] = defaultdict(list)

    train_rows = list(genimage_rows("train"))
    train_counts = Counter(r[2] for r in train_rows)
    fake_counts = {k: v for k, v in train_counts.items() if k != "Real"}
    train_gen = "SD14" if fake_counts.get("SD14", 0) >= 200 else max(
        ("SD15", "SD14"), key=lambda k: fake_counts.get(k, 0))
    print("train-split counts:", dict(train_counts), "-> training generator:", train_gen, flush=True)

    groups["train"] = [r for r in train_rows if r[2] in ("Real", train_gen)]
    groups["test_genimage"] = list(genimage_rows("validation"))
    groups["openfake"] = list(openfake_rows())

    def cap(items, per_source: dict[str, int], default: int):
        by = defaultdict(list)
        for it in items:
            by[it[2]].append(it)
        out = []
        for src, lst in by.items():
            k = per_source.get(src, default)
            idx = rng.permutation(len(lst))[:k]
            out.extend(lst[i] for i in idx)
        return sorted(out, key=lambda r: (r[0], r[1]))

    groups["train"] = cap(groups["train"], {}, N_TRAIN_PER_CLASS)
    groups["test_genimage"] = cap(groups["test_genimage"], {"Real": N_TEST_REAL}, N_TEST_PER_GENERATOR)
    of = groups["openfake"]
    of_real = [r for r in of if r[3] == 0]
    of_fake = [r for r in of if r[3] == 1]
    of_real = [of_real[i] for i in rng.permutation(len(of_real))[:N_OPENFAKE_REAL]]
    of_fake = [of_fake[i] for i in rng.permutation(len(of_fake))[:N_OPENFAKE_FAKE]]
    groups["openfake"] = sorted(of_real + of_fake, key=lambda r: r[1])
    for g, items in groups.items():
        print(f"{g}: {len(items)} images", dict(Counter(i[2] for i in items).most_common(12)), flush=True)
    (RESULTS).mkdir(exist_ok=True, parents=True)
    (RESULTS / "training_generator.txt").write_text(train_gen)
    return groups


# ----------------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------------
def extract():
    FEATURES.mkdir(parents=True, exist_ok=True)
    groups = pick_samples()
    encoder = ClipEncoder()
    for group, items in groups.items():
        out_path = FEATURES / f"{group}.npz"
        if out_path.exists():
            print("skip (exists):", out_path, flush=True)
            continue
        cols = defaultdict(list)
        t0 = time.time()
        done = 0
        by_file = defaultdict(list)
        for it in items:
            by_file[it[0]].append(it)
        for file, file_items in by_file.items():
            wanted = {it[1]: it for it in file_items}
            pending, row = [], 0
            # stream the file in small batches so memory stays low (the laptop has 7.8 GB RAM)
            for batch in pq.ParquetFile(file).iter_batches(batch_size=32, columns=["image"]):
                for img in batch.column(0).to_pylist():
                    if row in wanted:
                        pending.append((img["bytes"],) + tuple(wanted[row][2:]))
                    row += 1
                if len(pending) >= 16:
                    done = _extract_chunks(pending, encoder, cols, group, done, len(items), t0)
                    pending = []
            if pending:
                done = _extract_chunks(pending, encoder, cols, group, done, len(items), t0)
        arrays = {k: (np.concatenate(v) if k.startswith("clip") else np.array(v)) for k, v in cols.items()}
        np.savez_compressed(out_path, **arrays)
        print("saved", out_path, flush=True)


def _extract_chunks(items, encoder, cols, group, done, total, t0) -> int:
    """Centre patch -> CLIP features (original and JPEG-normalised), frequency features, peak score."""
    for start in range(0, len(items), 16):
        chunk = items[start:start + 16]
        raw_crops, jpg_crops = [], []
        for img_bytes, source, label, year, ds in chunk:
            loaded = load_image(img_bytes)
            crop = center_crop(loaded.rgb)
            raw_crops.append(crop)
            jpg_crops.append(jpeg_normalize(crop, 95))
            cols["peak"].append(spectral_peak_score(loaded.rgb))
            cols["freq_raw"].append(frequency_features(crop))
            cols["freq_jpeg"].append(frequency_features(jpg_crops[-1]))
            cols["label"].append(label)
            cols["source"].append(source)
            cols["year"].append(year)
            cols["dataset"].append(ds)
            cols["format"].append(Image.open(io.BytesIO(img_bytes)).format or "?")
            cols["width"].append(loaded.rgb.width)
        cols["clip_raw"].append(encoder.encode(raw_crops))
        cols["clip_jpeg"].append(encoder.encode(jpg_crops))
        done += len(chunk)
        rate = (time.time() - t0) / done
        print(f"{group}: {done}/{total}  {rate:.2f}s/img  eta {rate * (total - done) / 60:.1f} min", flush=True)
    return done


# ----------------------------------------------------------------------------
# Training and evaluation
# ----------------------------------------------------------------------------
def load_group(name: str) -> dict:
    with np.load(FEATURES / f"{name}.npz", allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def build_x(d: dict, variant: str, freq_stats=None):
    if variant == "raw_clip":
        return l2(d["clip_raw"])
    if variant == "jpeg_clip":
        return l2(d["clip_jpeg"])
    if variant == "jpeg_clip_freq":
        mean, std = freq_stats
        return np.concatenate([l2(d["clip_jpeg"]), (d["freq_jpeg"] - mean) / std], axis=1)
    raise ValueError(variant)


def train():
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score

    train_gen = (RESULTS / "training_generator.txt").read_text().strip()
    tr = load_group("train")
    tests = {"GenImage": load_group("test_genimage"), "OpenFake": load_group("openfake")}
    freq_stats = (tr["freq_jpeg"].mean(0), tr["freq_jpeg"].std(0) + 1e-6)

    models = {}
    for variant in ("raw_clip", "jpeg_clip", "jpeg_clip_freq"):
        clf = LogisticRegressionCV(Cs=[0.1, 0.3, 1, 3, 10, 30], cv=5, max_iter=5000, scoring="roc_auc")
        clf.fit(build_x(tr, variant, freq_stats), tr["label"])
        models[variant] = clf
        print(variant, "chosen C =", clf.C_[0], flush=True)

    ufd = load_ufd_head()

    def scores(variant, d):
        if variant == "ufd":
            return sigmoid(d["clip_raw"] @ ufd.w + ufd.b)
        return models[variant].predict_proba(build_x(d, variant, freq_stats))[:, 1]

    names = {
        "ufd": "UniversalFakeDetect (trained on ProGAN)",
        "raw_clip": f"Ours: CLIP, no format fix (trained on {train_gen})",
        "jpeg_clip": f"Ours: CLIP + format fix (trained on {train_gen})",
        "jpeg_clip_freq": f"Ours: CLIP + frequency + format fix (trained on {train_gen})",
    }
    rows = []
    for variant in names:
        for ds, d in tests.items():
            p = scores(variant, d)
            real = d["label"] == 0
            real_acc = float(np.mean(p[real] < 0.5)) if real.any() else float("nan")
            rows.append(dict(model=variant, model_name=names[variant], dataset=ds, source="Real",
                             seen=ds == "GenImage", year=0, n=int(real.sum()),
                             accuracy=real_acc, auc=None, mean_p=float(np.mean(p[real]))))
            for src in sorted(set(d["source"][~real])):
                m = (d["source"] == src) & ~real
                y = np.r_[np.zeros(real.sum()), np.ones(m.sum())]
                s = np.r_[p[real], p[m]]
                rows.append(dict(model=variant, model_name=names[variant], dataset=ds, source=str(src),
                                 seen=variant != "ufd" and ds == "GenImage" and src == train_gen,
                                 year=int(d["year"][m][0]), n=int(m.sum()),
                                 accuracy=float(np.mean(p[m] >= 0.5)),
                                 auc=float(roc_auc_score(y, s)), mean_p=float(np.mean(p[m]))))
            # all fakes of this dataset together
            y_all = d["label"]
            rows.append(dict(model=variant, model_name=names[variant], dataset=ds, source="ALL (whole dataset)",
                             seen=False, year=0, n=int(len(y_all)),
                             accuracy=float(np.mean((p >= 0.5) == (y_all == 1))),
                             auc=float(roc_auc_score(y_all, p)), mean_p=float(np.mean(p))))

    RESULTS.mkdir(exist_ok=True, parents=True)
    summary = {
        "training_generator": train_gen,
        "n_train": {"real": int((tr["label"] == 0).sum()), "fake": int((tr["label"] == 1).sum())},
        "test_sizes": {ds: int(len(d["label"])) for ds, d in tests.items()},
        "rows": rows,
    }
    (RESULTS / "crossgen_results.json").write_text(json.dumps(summary, indent=1))

    # save our final head (design choice: CLIP + frequency + format fix)
    final = "jpeg_clip_freq"
    clf = models[final]
    np.savez(WEIGHTS / "our_head.npz", w=clf.coef_[0].astype(np.float32), b=np.float32(clf.intercept_[0]),
             freq_mean=freq_stats[0], freq_std=freq_stats[1])
    (WEIGHTS / "our_head.json").write_text(json.dumps({
        "name": f"Our model (trained on {train_gen} only)",
        "trained_on": f"GenImage: {summary['n_train']['real']} real ImageNet photos vs "
                      f"{summary['n_train']['fake']} {train_gen} images",
        "normalize_features": True, "use_frequency": True, "jpeg_quality": 95,
        "regularization_C": float(clf.C_[0]), "variant": final,
    }, indent=1))
    peaks_real = tr["peak"][tr["label"] == 0]
    (WEIGHTS / "thresholds.json").write_text(json.dumps({
        "spectral_peak_real_p95": float(np.percentile(peaks_real, 95)),
        "spectral_peak_real_median": float(np.median(peaks_real)),
        "spectral_peak_fake_median": float(np.median(tr["peak"][tr["label"] == 1])),
    }, indent=1))

    # readable table
    print(f"\nTrained on: real vs {train_gen}.  AUC per source (accuracy at threshold 0.5)")
    for variant in names:
        print("\n" + names[variant])
        for r in rows:
            if r["model"] == variant:
                auc = f"{r['auc']:.3f}" if r["auc"] is not None else "  -  "
                print(f"  {r['dataset']:9s} {r['source'][:34]:34s} n={r['n']:4d}  AUC={auc}  acc={r['accuracy']:.3f}")


# ----------------------------------------------------------------------------
# Model v2: multi-source training, ChatGPT and Gemini kept out (still unseen)
# ----------------------------------------------------------------------------
# Every OpenAI model (ChatGPT's GPT-Image, DALL-E) and every Google model (Gemini's Nano Banana, Imagen, Veo)
# is kept OUT of v2's training data, so ChatGPT and Gemini images stay truly unseen.
HELD_OUT_KEYWORDS = ("gpt", "dall", "openai", "imagen", "nano-banana", "nano banana", "gemini", "veo", "google")
HELD_OUT = {"all OpenAI models (GPT-Image-1, GPT-Image-1.5, DALL-E 3)",
            "all Google models (Nano Banana, Nano Banana Pro, Imagen 3/4, Veo 3)"}
N_V2_PER_GENIMAGE_GENERATOR = 200


def is_held_out(source: str) -> bool:
    return any(k in source.lower() for k in HELD_OUT_KEYWORDS)


def random_crop(img: Image.Image, rng, size: int = 224) -> Image.Image:
    from detector import ensure_min_size
    img = ensure_min_size(img, size)
    x = int(rng.integers(0, img.width - size + 1))
    y = int(rng.integers(0, img.height - size + 1))
    return img.crop((x, y, x + size, y + size))


def extract_v2():
    """Extra training data for v2: the other GenImage generators + the OpenFake TRAIN split
    (never the OpenFake test images, and never ChatGPT's or Gemini's generator)."""
    rng = np.random.default_rng(SEED + 1)
    train_gen = (RESULTS / "training_generator.txt").read_text().strip()
    gi = [r for r in genimage_rows("train") if r[2] not in ("Real", train_gen)]
    by = defaultdict(list)
    for r in gi:
        by[r[2]].append(r)
    gi = sorted((x for lst in by.values() for x in (lst[i] for i in rng.permutation(len(lst))[:N_V2_PER_GENIMAGE_GENERATOR])),
                key=lambda r: (r[0], r[1]))
    of = [r for r in openfake_rows("train") if not is_held_out(r[2])]
    print("held out of v2 training:", dict(Counter(r[2] for r in openfake_rows("train") if is_held_out(r[2]))), flush=True)
    groups = {"v2_genimage": gi, "v2_openfake_train": of}
    for g, items in groups.items():
        print(f"{g}: {len(items)} images", dict(Counter(i[2] for i in items).most_common(12)), flush=True)

    encoder = ClipEncoder()
    for group, items in groups.items():
        out_path = FEATURES / f"{group}.npz"
        if out_path.exists():
            print("skip (exists):", out_path, flush=True)
            continue
        cols, t0, done, batch = defaultdict(list), time.time(), 0, []

        def flush():
            cols["clip_jpeg"].append(encoder.encode([p for p, _ in batch]))
            batch.clear()

        for img_bytes, source, label, year, ds in stream_images(items):
            rgb = load_image(img_bytes).rgb
            for kind, patch in (("centre", center_crop(rgb)), ("random", random_crop(rgb, rng))):
                patch = jpeg_normalize(patch, 95)
                batch.append((patch, kind))
                cols["freq_jpeg"].append(frequency_features(patch))
                cols["label"].append(label)
                cols["source"].append(source)
                cols["dataset"].append(ds)
                cols["patch"].append(kind)
            if len(batch) >= 16:
                flush()
            done += 1
            if done % 16 == 0:
                rate = (time.time() - t0) / done
                print(f"{group}: {done}/{len(items)}  {rate:.2f}s/img  eta {rate * (len(items) - done) / 60:.1f} min",
                      flush=True)
        if batch:
            flush()
        arrays = {k: (np.concatenate(v) if k == "clip_jpeg" else np.array(v)) for k, v in cols.items()}
        np.savez_compressed(out_path, **arrays)
        print("saved", out_path, flush=True)


def train_v2():
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score

    parts = [load_group("train"), load_group("v2_genimage"), load_group("v2_openfake_train")]
    clip = np.concatenate([p["clip_jpeg"] for p in parts])
    freq = np.concatenate([p["freq_jpeg"] for p in parts])
    y = np.concatenate([p["label"] for p in parts])
    seen_sources = set(np.concatenate([p["source"] for p in parts])) - {"Real"}
    freq_stats = (freq.mean(0), freq.std(0) + 1e-6)
    x = np.concatenate([l2(clip), (freq - freq_stats[0]) / freq_stats[1]], axis=1)
    clf = LogisticRegressionCV(Cs=[0.1, 0.3, 1, 3, 10, 30], cv=5, max_iter=5000, scoring="roc_auc",
                               class_weight="balanced")
    clf.fit(x, y)
    print("v2 trained on", len(y), "patches; real", int((y == 0).sum()), "fake", int((y == 1).sum()),
          "; C =", clf.C_[0], flush=True)

    name = "Ours v2: many generators, never OpenAI or Google"
    assert not any(is_held_out(s) for s in seen_sources), "an OpenAI/Google image leaked into v2 training"
    rows = []
    for ds, d in {"GenImage": load_group("test_genimage"), "OpenFake": load_group("openfake")}.items():
        xt = np.concatenate([l2(d["clip_jpeg"]), (d["freq_jpeg"] - freq_stats[0]) / freq_stats[1]], axis=1)
        p = clf.predict_proba(xt)[:, 1]
        real = d["label"] == 0
        rows.append(dict(model="v2", model_name=name, dataset=ds, source="Real", seen=True, year=0,
                         n=int(real.sum()), accuracy=float(np.mean(p[real] < 0.5)), auc=None,
                         mean_p=float(np.mean(p[real]))))
        for src in sorted(set(d["source"][~real])):
            m = (d["source"] == src) & ~real
            yt = np.r_[np.zeros(real.sum()), np.ones(m.sum())]
            # Names differ between OpenFake splits (e.g. midjourney-6 in train vs Midjourney v7 in test), so only the
            # OpenAI/Google models count as unseen for v2; for the rest, related versions were in training.
            rows.append(dict(model="v2", model_name=name, dataset=ds, source=str(src), seen=not is_held_out(str(src)),
                             year=int(d["year"][m][0]), n=int(m.sum()), accuracy=float(np.mean(p[m] >= 0.5)),
                             auc=float(roc_auc_score(yt, np.r_[p[real], p[m]])), mean_p=float(np.mean(p[m]))))
        rows.append(dict(model="v2", model_name=name, dataset=ds, source="ALL (whole dataset)", seen=False, year=0,
                         n=int(len(p)), accuracy=float(np.mean((p >= 0.5) == (d["label"] == 1))),
                         auc=float(roc_auc_score(d["label"], p)), mean_p=float(np.mean(p))))

    res_path = RESULTS / "crossgen_results.json"
    res = json.loads(res_path.read_text())
    res["rows"] = [r for r in res["rows"] if r["model"] != "v2"] + rows
    res["v2"] = {"n_patches": int(len(y)), "held_out": sorted(HELD_OUT), "seen_generators": sorted(seen_sources)}
    res_path.write_text(json.dumps(res, indent=1))

    np.savez(WEIGHTS / "our_head_v2.npz", w=clf.coef_[0].astype(np.float32), b=np.float32(clf.intercept_[0]),
             freq_mean=freq_stats[0], freq_std=freq_stats[1])
    (WEIGHTS / "our_head_v2.json").write_text(json.dumps({
        "name": "Our model v2 (many generators, never ChatGPT or Gemini)",
        "trained_on": "real photos vs 7 GenImage generators + about 60 modern OpenFake generators; every OpenAI "
                      "(ChatGPT, DALL-E) and Google (Gemini, Imagen, Veo) model was kept out",
        "normalize_features": True, "use_frequency": True, "jpeg_quality": 95,
        "regularization_C": float(clf.C_[0]), "variant": "v2", "held_out": sorted(HELD_OUT),
    }, indent=1))
    for r in rows:
        auc = f"{r['auc']:.3f}" if r["auc"] is not None else "  -  "
        print(f"  {r['dataset']:9s} {r['source'][:34]:34s} {'seen' if r['seen'] else 'UNSEEN':6s} n={r['n']:4d}  "
              f"AUC={auc}  acc={r['accuracy']:.3f}")


if __name__ == "__main__":
    {"extract": extract, "train": train, "extract_v2": extract_v2, "train_v2": train_v2}[sys.argv[1]]()
