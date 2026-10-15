"""Streamlit web app — image prototype of the Unified Content Authenticity Verification System.

Run:  .venv\\Scripts\\streamlit run prototype\\app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from detector import Detector  # noqa: E402
from forensics import (SUPPORTED_TYPES, ImageLoadError, ela_image, load_image,  # noqa: E402
                       read_metadata, spectral_peak_score, spectrum_image)
from trust import explain, heatmap_overlay, peak_threshold, trust_score, verdict  # noqa: E402

HERE = Path(__file__).parent
RESULTS_FILE = HERE / "results" / "crossgen_results.json"
TEST_IMAGES = HERE / "test_images"

st.set_page_config(page_title="Content Authenticity Checker", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Loading the AI detector (first time only, about 30 seconds)...")
def get_detector(models_version: str = "2026-10-15-v3") -> Detector:
    # Changing this version string (it is part of the cache key) makes a running app reload the model heads,
    # for example after new weights are pushed. Without it a hosted app keeps serving the old heads.
    return Detector()


def model_picker() -> str:
    det = get_detector()
    keys = list(det.heads)
    default = next((keys.index(k) for k in ("ours_v3", "ours_v2", "ours") if k in keys), 0)
    key = st.sidebar.selectbox("Detector model", keys, index=default,
                               format_func=lambda k: det.heads[k].name)
    st.sidebar.caption(f"Trained on: {det.heads[key].trained_on}")
    return key


def score_card(score: int, p_fake: float, bands=None) -> None:
    v = verdict(score, bands)
    st.markdown(
        f"""
<div style="border:2px solid {v.colour}; border-radius:14px; padding:18px 22px; background:{v.colour}14;">
  <div style="font-size:14px; opacity:.75;">Trust Score</div>
  <div style="font-size:60px; font-weight:800; line-height:1.05; color:{v.colour};">{score}<span style="font-size:24px;">/100</span></div>
  <div style="font-size:22px; font-weight:700; color:{v.colour}; margin-top:4px;">{v.label}</div>
  <div style="margin-top:6px;">{v.advice}</div>
  <div style="font-size:13px; opacity:.7; margin-top:8px;">Probability of AI generation / manipulation: {p_fake:.2f}</div>
</div>""",
        unsafe_allow_html=True,
    )


# ============================================================================
# Page 1 - Check an image
# ============================================================================
def page_check() -> None:
    st.title("🛡️ Is this image real or AI-generated?")
    st.caption("Prototype · Phase 1 (images) · Unified Content Authenticity Verification System · The Perceptrons")
    head_key = model_picker()

    upload = st.file_uploader("Drag and drop an image (JPG, PNG, WEBP, BMP, TIFF, GIF, HEIC)", type=SUPPORTED_TYPES)
    samples = sorted(p for p in TEST_IMAGES.glob("*/*") if p.suffix.lower().lstrip(".") in SUPPORTED_TYPES) \
        if TEST_IMAGES.exists() else []
    sample = None
    if upload is None and samples:
        sample = st.selectbox("...or try a sample image", [None] + samples,
                              format_func=lambda p: "(choose a sample)" if p is None else f"{p.parent.name} / {p.name}")
    if upload is None and sample is None:
        st.info("Upload an image to get its Trust Score, verdict, heatmap and explanation.")
        return
    data, name = (upload.getvalue(), upload.name) if upload is not None else (sample.read_bytes(), sample.name)

    try:
        loaded = load_image(data)
    except ImageLoadError as exc:
        st.error(str(exc))
        return

    det = get_detector()
    with st.spinner("Analysing the image (about 10-20 seconds on a laptop CPU)..."):
        result = det.analyze(loaded.rgb, head_key, grid=3)
        peak = spectral_peak_score(loaded.rgb)
        meta = read_metadata(loaded)
    score = trust_score(result.probability)

    left, right = st.columns([1.15, 1])
    with left:
        st.image(loaded.rgb, caption=f"{name} · {meta['format']} · {meta['size']} px", width="stretch")
    with right:
        score_card(score, result.probability, result.head.bands)
        st.markdown("#### Why?")
        for line in explain(result.probability, result.grid, peak, meta):
            st.markdown(f"- {line}")
        st.caption(f"Model: {result.head.name} — trained on {result.head.trained_on}.")

    tab_heat, tab_freq, tab_ela, tab_meta, tab_tech = st.tabs(
        ["🔥 Heatmap", "📈 Frequency spectrum", "🔍 Error Level Analysis", "📄 File & metadata", "⚙️ Technical details"])
    with tab_heat:
        st.image(heatmap_overlay(loaded.rgb, result.grid), width="stretch",
                 caption="Red = patches that look AI-generated, green = patches that look real.")
        st.caption("The image is checked as a 3 x 3 grid of 224 x 224 patches at full resolution, plus the centre. "
                   "This prototype heatmap is coarse; precise pixel-level maps come later (U-Net, Phase 3).")
    with tab_freq:
        c1, c2 = st.columns(2)
        c1.image(spectrum_image(loaded.rgb), width="stretch",
                 caption="Fourier spectrum of the image's fine noise (centre = low frequencies).")
        with c2:
            st.metric("Spectral peak strength", f"{peak:.1f}", help="How much the strongest isolated peaks stand out.")
            st.write("Older AI generators (especially GANs) build images by repeated up-sampling, which can leave "
                     "bright dots or grid lines in this picture. Real camera photos usually give a smooth, cloudy spectrum.")
            st.caption(f"Honest note: in our experiment this single number did not separate real from modern AI images "
                       f"(95% of real photos stayed below {peak_threshold():.1f}, and AI images looked the same), so it "
                       "is shown for learning and is not used in the Trust Score or the explanation.")
    with tab_ela:
        c1, c2 = st.columns(2)
        c1.image(ela_image(loaded.rgb), width="stretch", caption="Error Level Analysis (brighter = changes more on re-saving)")
        c2.write("Error Level Analysis re-saves the image as JPEG and shows how much each area changes. "
                 "In an edited JPEG photo, pasted or retouched regions often look brighter or darker than the rest.")
        if meta["format"] != "JPEG":
            c2.warning(f"This file is {meta['format']}, not JPEG, so ELA is less informative here.")
    with tab_meta:
        info = {"File format (from file bytes)": meta["format"], "Size": meta["size"], "Colour mode": meta["colour mode"]}
        info.update(meta["exif"] or {"Camera EXIF": "none found"})
        info["AI / provenance markers"] = "; ".join(meta["ai markers"]) or "none found"
        for k, v in meta["text chunks"].items():
            info[f"Embedded text '{k}'"] = v
        st.table(pd.DataFrame(info.items(), columns=["Field", "Value"]))
        for note in loaded.notes:
            st.caption("ℹ️ " + note)
        st.caption("Metadata is shown separately from the model score: it helps when present, but it is easy to remove "
                   "(for example by taking a screenshot).")
    with tab_tech:
        st.write(f"**Model:** {result.head.name}  \n**Trained on:** {result.head.trained_on}")
        st.write(f"**Mean logit:** {result.logit:.3f} → probability = sigmoid(logit) = {result.probability:.3f}  \n"
                 f"**Centre patch probability:** {result.center_probability:.3f}  \n"
                 f"**Trust Score** = round(100 × (1 − {result.probability:.3f})) = {score}")
        st.write("Patch probabilities (3 x 3 grid):")
        st.dataframe(pd.DataFrame(np.round(result.grid, 3)), width="content")


# ============================================================================
# Page 2 - Cross-Generator Test
# ============================================================================
V3_RESULTS = HERE / "results" / "v3_results.json"
V3_CALIBRATION = HERE / "results" / "v3_calibration.json"


def v3_section() -> None:
    """Results of the v3 experiment (held-out OpenFake test split: generators never used in training)."""
    if not V3_RESULTS.exists():
        return
    r = json.loads(V3_RESULTS.read_text(encoding="utf-8"))
    names = {"v2_reference": "Old v2", "v3_noaug": "v3 without compression training",
             "v3_proof": "v3 proof (no OpenAI / Google in training)", "v3": "Our model v3"}
    st.header("Model v3 - results on held-out data")
    st.write("v3 was trained on many more generators (including ChatGPT and Gemini images) and with random JPEG compression, "
             "resizing and blur. To keep the claim honest, **the \"proof\" model has every OpenAI and Google image removed from "
             "training**, and all numbers below come from images **never used in training** (OpenFake test split: "
             "1,200 real photos and 2,192 AI images from 20 generators that are not in the training set).")
    rows = []
    for key, label in names.items():
        row = {"Model": label}
        for var, col in (("clean", "Clean"), ("jpeg70", "JPEG 70"), ("half_q75", "Half size + JPEG 75")):
            e = r[key]["test_openfake"].get(var)
            row[col] = f"AUC {e['auc']:.3f} · {e['fake_caught'] * 100:.0f}% caught" if e else "-"
        e = r[key]["test_reddit"].get("clean")
        row["In the wild (Reddit)"] = f"AUC {e['auc']:.3f} · {e['fake_caught'] * 100:.0f}% caught" if e else "-"
        row["Real photos called real"] = f"{r[key]['test_openfake']['clean']['real_correct'] * 100:.0f}%"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("AUC = how well the model separates AI images from real photos (0.5 = guessing, 1.0 = perfect). "
               "% caught = AI images labelled AI at probability 0.5. JPEG 70 / half size = the same images after compression.")

    st.write("**ChatGPT, Gemini and other modern generators (clean images):**")
    gens = [("gpt-image-1.5", "GPT-Image-1.5 (ChatGPT)"), ("gpt-image-2", "GPT-Image-2 (ChatGPT)"),
            ("nano-banana-pro", "Nano Banana Pro (Gemini)"), ("veo-3", "Veo 3 (Google video)"), ("sora-2", "Sora 2 (OpenAI video)"),
            ("midjourney-7", "Midjourney v7"), ("flux.2-klein-9b", "FLUX.2 klein 9B"), ("z-image-turbo", "Z-Image Turbo")]
    rows = []
    for key, label in gens:
        g = r["v3"]["test_openfake"]["clean"]["per_generator"].get(key)
        if not g:
            continue
        row = {"Generator": label, "Images": g["n"]}
        for mk, col in (("v2_reference", "Old v2"), ("v3_proof", "v3 proof"), ("v3", "v3")):
            x = r[mk]["test_openfake"]["clean"]["per_generator"].get(key)
            row[col] = f"AUC {x['auc']:.3f} · {x['caught'] * 100:.0f}%" if x else "-"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("In the test split these generators are not in the training set (v3 saw only older versions, for example "
               "GPT-Image-1 and Imagen; the proof model saw no OpenAI or Google image at all).")

    if V3_CALIBRATION.exists():
        c = json.loads(V3_CALIBRATION.read_text(encoding="utf-8"))
        t = c["test_openfake"]["clean"]
        st.info(f"**Verdict bands (chosen on validation data, not guessed):** Trust Score {c['trust_authentic_at_or_above']} or more = "
                f"Likely Authentic · {c['trust_ai_at_or_below']} or less = Likely AI-generated · in between = Needs Human Review. "
                f"On the held-out test split, {t['real']['ai'] * 100:.1f}% of real photos are called AI and "
                f"{t['fake']['authentic'] * 100:.1f}% of clean AI images are called authentic.")
    with st.expander("Honest limitations of v3"):
        st.markdown(chr(10).join([
            "- Compressed AI images are still the hardest case: at JPEG quality 70 about a third are still missed "
            "(the model more often says *Needs Human Review* than makes a wrong call).",
            "- Real photos can land in *Needs Human Review* (roughly 12% on the clean test photos).",
            "- Edited images: on held-out edits from research datasets the model catches 89% (clean) / 80% (compressed) of "
            "edited patches, but we have **not** yet measured edits made by ChatGPT or Gemini on real photos.",
            "- The heatmap is still coarse (3 x 3 patches).",
        ]))
    st.divider()
    st.header("Original prototype experiment (models v1 and v2)")


def page_crossgen() -> None:
    st.title("🧪 Cross-Generator Test")
    st.write("**Cross-generator generalization** means the detector still works on images from AI tools it "
             "**never saw during training**. We train on some generators and test on others it has never seen.")

    v3_section()
    if RESULTS_FILE.exists():
        res = json.loads(RESULTS_FILE.read_text())
        df = pd.DataFrame(res["rows"])
        main = [m for m in ("v2", "jpeg_clip_freq", "ufd") if m in set(df.model)]
        label = {"v2": "Our model v2", "jpeg_clip_freq": "Our model v1", "ufd": "UniversalFakeDetect",
                 "jpeg_clip": "v1 without frequency", "raw_clip": "v1 without format fix"}
        trained = {"v2": "Many generators, no OpenAI / Google",
                   "jpeg_clip_freq": "Only SD1.5",
                   "ufd": "Only ProGAN (2018)"}

        def cell(model, ds, src, field="auc"):
            r = df[(df.model == model) & (df.dataset == ds) & (df.source == src)]
            if r.empty or r.iloc[0][field] is None or pd.isna(r.iloc[0][field]):
                return None
            return float(r.iloc[0][field])

        def unseen_mean(model, ds):
            r = df[(df.model == model) & (df.dataset == ds) & (df.source != "Real") & ~df.source.str.startswith("ALL")
                   & (df.n >= 10) & ~df.seen.astype(bool)]
            return float(r.auc.mean()) if len(r) else None

        st.subheader("Summary")
        chat = "GPT-Image-1.5 (ChatGPT, OpenAI)"
        gem = "Nano Banana Pro (Gemini, Google)"
        def fmt(auc, acc=None):
            if auc is None:
                return "—"
            return f"{auc:.2f}" + (f" · {acc * 100:.0f}% caught" if acc is not None else "")

        summary = []
        for m in main:
            summary.append({
                "Model": label[m], "Trained on": trained[m],
                "ChatGPT images": fmt(cell(m, "OpenFake", chat), cell(m, "OpenFake", chat, "accuracy")),
                "Gemini images (4)": fmt(cell(m, "OpenFake", gem), cell(m, "OpenFake", gem, "accuracy")),
                "Modern generators it never saw": fmt(unseen_mean(m, "OpenFake")),
                "Real photos correct": f"{cell(m, 'OpenFake', 'Real', 'accuracy') * 100:.0f}%",
            })
        st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch")
        st.caption("None of these three models ever saw a ChatGPT or Gemini image. Numbers are AUC (how well the model "
                   "separates that generator's images from real photos: 0.5 = guessing, 1.0 = perfect) and the share of "
                   "images labelled AI at probability 0.5. \"Modern generators it never saw\" = average AUC over OpenFake "
                   "generators that model never saw (at least 10 test images each). v2 = our improved model; "
                   "v1 = the pure experiment trained on one generator; UniversalFakeDetect = published model.")

        st.subheader("Every generator, side by side")
        ds_choice = st.radio("Test set", ["OpenFake (modern, 2024-2026)", "GenImage (2018-2022)"], horizontal=True)
        ds = ds_choice.split()[0]
        sub = df[(df.dataset == ds) & df.model.isin(main) & ~df.source.str.startswith("ALL")]
        table = []
        for src in sub[sub.model == main[-1]].source:
            row = {"Source": src, "Year": "", "Images": None}
            for m in main:
                r = sub[(sub.model == m) & (sub.source == src)]
                if r.empty:
                    continue
                r = r.iloc[0]
                row["Year"] = str(r.year) if r.year else ""
                row["Images"] = int(r.n)
                if src == "Real":
                    row[label[m]] = f"{r.accuracy * 100:.0f}% correct"
                    continue
                if not r.seen:
                    mark = "❌ unseen"
                elif m == "v2" and ds == "OpenFake":
                    mark = "≈ similar seen"
                else:
                    mark = "✅ seen"
                row[label[m]] = f"{mark} · {r.auc:.2f} · {r.accuracy * 100:.0f}%"
            table.append(row)
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
        st.caption("Each cell: whether that model saw this generator in training (≈ similar seen = a related version "
                   "was in training) · AUC · % of that generator's images labelled AI at probability 0.5. "
                   "For real photos: % labelled real. Sources with fewer than 10 images are too small to trust.")
        chart = sub[sub.source != "Real"].assign(model_name=sub.model.map(label)).pivot_table(
            index="source", columns="model_name", values="auc")
        if not chart.empty:
            st.write("**AUC per generator** (higher is better; 0.5 = guessing)")
            st.bar_chart(chart, horizontal=True, stack=False, height=520)
    else:
        st.warning("Experiment results not found yet (prototype/results/crossgen_results.json). "
                   "Run `python prototype/experiment.py extract` then `train`.")

    st.divider()
    st.subheader("Test your own images")
    st.write("Put images into folders inside `prototype/test_images/` — one folder per source, for example "
             "`real/`, `chatgpt/`, `gemini/`. The folder named **real** holds real photos; every other folder is "
             "treated as AI-generated. Fast mode checks the centre patch of each image (about 1 second per image).")
    head_key = model_picker()
    folders = sorted(p for p in TEST_IMAGES.iterdir() if p.is_dir()) if TEST_IMAGES.exists() else []
    counts = {f.name: len([p for p in f.iterdir() if p.suffix.lower().lstrip('.') in SUPPORTED_TYPES]) for f in folders}
    st.write("Folders found: " + (", ".join(f"`{k}` ({v} images)" for k, v in counts.items()) or "none"))
    if st.button("Run the test on these folders", type="primary", disabled=not any(counts.values())):
        det = get_detector()
        rows, progress = [], st.progress(0.0, text="Starting...")
        files = [(f.name, p) for f in folders for p in sorted(f.iterdir())
                 if p.suffix.lower().lstrip(".") in SUPPORTED_TYPES]
        per_source: dict[str, list[float]] = {}
        for i, (source, path) in enumerate(files, 1):
            try:
                img = load_image(path.read_bytes()).rgb
                p = det.analyze(img, head_key, grid=1).center_probability
                per_source.setdefault(source, []).append(p)
            except ImageLoadError:
                pass
            progress.progress(i / len(files), text=f"{i}/{len(files)}: {source}/{path.name}")
        for source, ps in per_source.items():
            ps = np.array(ps)
            is_real = source.lower() == "real"
            correct = np.mean(ps < 0.5) if is_real else np.mean(ps >= 0.5)
            rows.append({"Source": source, "Type": "real photos" if is_real else "AI-generated",
                         "Images": len(ps), "Detected correctly": f"{correct * 100:.1f} %",
                         "Average AI probability": round(float(ps.mean()), 3)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"Model: {det.heads[head_key].name}. A small hand-made test set is a demonstration, not a scientific result.")


# ============================================================================
# Page 3 - How it works
# ============================================================================
def page_how() -> None:
    st.title("ℹ️ How it works")
    st.graphviz_chart("""
digraph G {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#E8EAF6", color="#3949AB", fontname="Helvetica"];
  up [label="Upload\\n(any image format)"]; std [label="Convert to\\nstandard RGB"];
  crops [label="Cut 224x224 patches\\n(no resizing)"]; clip [label="CLIP ViT-L/14\\n(frozen, 768 features)", fillcolor="#E0F2F1", color="#00897B"];
  head [label="Classifier head\\n(logistic regression)", fillcolor="#E0F2F1", color="#00897B"];
  score [label="Trust Score\\n= 100 x (1 - p)"]; out [label="Verdict + heatmap\\n+ explanation"];
  fft [label="FFT spectrum, ELA,\\nmetadata (no training)", fillcolor="#FFF3E0", color="#F2A33A"];
  up -> std -> crops -> clip -> head -> score -> out; std -> fft -> out;
}""", width="stretch")
    st.markdown("""
**1. Standardise the file.** Any format (JPG, PNG, WEBP, HEIC...) is converted to plain RGB pixels, so the model never
sees the file format — only the picture.

**2. Cut patches.** The image is cut into 224 x 224 pixel patches at its own resolution (resizing would blur away the
tiny traces AI generators leave).

**3. CLIP features.** Each patch goes through CLIP ViT-L/14, a model pre-trained on about 400 million images. We keep it
**frozen** (we never change it), so it cannot memorise the quirks of one generator. It outputs 768 numbers per patch.

**4. Classifier head.** A single linear layer (logistic regression) turns those numbers into a probability that the
patch is AI-generated. You can choose the head in the sidebar:
- **Our model v3** (default): our strongest model. Trained on real photos vs many generators **including ChatGPT and Gemini
  images**, with random compression / resizing / blur during training, and with edited-image patches. Probabilities are
  calibrated and the verdict bands are chosen on validation data.
- **v3 proof** and **v3 no-augmentation**: the same recipe with all OpenAI/Google images removed (cross-generator proof),
  and without compression training (shows what that training buys).
- **Our model v2**: trained on real photos vs many generators — but **never** on any OpenAI (ChatGPT, DALL-E)
  or Google (Gemini, Imagen, Veo) image, so those stay unseen.
- **Our model v1**: trained on real photos vs **one** generator only (Stable Diffusion 1.5) — the pure experiment.
- **UniversalFakeDetect**: a published model (CVPR 2023) trained on ProGAN only.

**5. Trust Score.** The patch scores are averaged: Trust Score = 100 x (1 - probability). for model v3 the bands come from validation data
(see the Technical details tab: Likely Authentic from 54, Likely AI-generated up to 17, in between = Needs Human Review).
Older models use provisional bands (65 / 35).

**6. Evidence.** The heatmap shows which patches look most AI-like; the frequency spectrum, Error Level Analysis and
metadata are extra evidence that need no training.

**Why can a model trained on one generator catch another?** It does not learn "what Stable Diffusion images look like";
it learns traces that many generators share — they all build pixels with neural up-sampling and none has a real camera's
lens and sensor noise — and what real photos look like, which does not change when a new AI tool appears.
The **Cross-Generator Test** page measures how well this works, generator by generator.
""")


pg = st.navigation([
    st.Page(page_check, title="Check an image", icon="🔍", default=True),
    st.Page(page_crossgen, title="Cross-Generator Test", icon="🧪"),
    st.Page(page_how, title="How it works", icon="ℹ️"),
], position="top")
pg.run()
