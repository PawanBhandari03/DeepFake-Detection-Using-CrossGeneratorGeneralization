"""Streamlit web app — DeepFake Detection · Upload an image, get a result.

Run:  .venv\\Scripts\\python.exe -m streamlit run prototype\\app.py
"""
from __future__ import annotations

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

st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🛡️",
    layout="wide",
)

# ── minimal style ────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, header, footer {visibility: hidden;}
.block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading AI detector (first run ~30 s)…")
def get_detector() -> Detector:
    return Detector()


def score_card(score: int, p_fake: float) -> None:
    v = verdict(score)
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


# ── main ──────────────────────────────────────────────────────────────────────
st.title("🛡️ DeepFake Image Detector")
st.caption("Upload any image — get an instant AI authenticity verdict.")

upload = st.file_uploader(
    "Drag & drop or click to upload an image (JPG, PNG, WEBP, BMP, TIFF, GIF, HEIC)",
    type=SUPPORTED_TYPES,
)

if upload is None:
    st.info("⬆️ Upload an image above to check whether it is real or AI-generated.")
    st.stop()

# ── analyse ───────────────────────────────────────────────────────────────────
try:
    loaded = load_image(upload.getvalue())
except ImageLoadError as exc:
    st.error(str(exc))
    st.stop()

det = get_detector()
# pick the best available head automatically
keys = list(det.heads)
head_key = next((k for k in ("ours_v2", "ours") if k in keys), keys[0])

with st.spinner("Analysing… (10-20 s on CPU)"):
    result = det.analyze(loaded.rgb, head_key, grid=3)
    peak   = spectral_peak_score(loaded.rgb)
    meta   = read_metadata(loaded)

score = trust_score(result.probability)

# ── results ───────────────────────────────────────────────────────────────────
left, right = st.columns([1.15, 1])
with left:
    st.image(loaded.rgb,
             caption=f"{upload.name} · {meta['format']} · {meta['size']} px",
             width="stretch")
with right:
    score_card(score, result.probability)
    st.markdown("#### Why?")
    for line in explain(result.probability, result.grid, peak, meta):
        st.markdown(f"- {line}")
    st.caption(f"Model: {result.head.name} — trained on {result.head.trained_on}.")

# ── detail tabs ───────────────────────────────────────────────────────────────
tab_heat, tab_freq, tab_ela, tab_meta, tab_tech = st.tabs(
    ["🔥 Heatmap", "📈 Frequency", "🔍 ELA", "📄 Metadata", "⚙️ Technical"])

with tab_heat:
    st.image(heatmap_overlay(loaded.rgb, result.grid), width="stretch",
             caption="Red = AI-like patches · Green = real-looking patches.")

with tab_freq:
    c1, c2 = st.columns(2)
    c1.image(spectrum_image(loaded.rgb), width="stretch",
             caption="Fourier spectrum of image noise.")
    with c2:
        st.metric("Spectral peak strength", f"{peak:.1f}")
        st.write("GAN images can leave bright dots in this spectrum. "
                 "Real photos give a smooth, cloudy pattern.")

with tab_ela:
    c1, c2 = st.columns(2)
    c1.image(ela_image(loaded.rgb), width="stretch",
             caption="Error Level Analysis — brighter = more re-save change.")
    c2.write("ELA re-saves as JPEG and highlights regions that changed most. "
             "Edited or pasted areas often stand out.")
    if meta["format"] != "JPEG":
        c2.warning(f"This file is {meta['format']}, not JPEG — ELA is less informative.")

with tab_meta:
    info = {"File format": meta["format"], "Size": meta["size"], "Colour mode": meta["colour mode"]}
    info.update(meta["exif"] or {"Camera EXIF": "none found"})
    info["AI / provenance markers"] = "; ".join(meta["ai markers"]) or "none found"
    for k, v in meta["text chunks"].items():
        info[f"Embedded text '{k}'"] = v
    st.table(pd.DataFrame(info.items(), columns=["Field", "Value"]))
    for note in loaded.notes:
        st.caption("ℹ️ " + note)

with tab_tech:
    st.write(f"**Model:** {result.head.name}  \n**Trained on:** {result.head.trained_on}")
    st.write(f"**Mean logit:** {result.logit:.3f} → probability = {result.probability:.3f}  \n"
             f"**Centre patch probability:** {result.center_probability:.3f}  \n"
             f"**Trust Score** = round(100 × (1 − {result.probability:.3f})) = {score}")
    st.write("Patch probabilities (3 × 3 grid):")
    st.dataframe(pd.DataFrame(np.round(result.grid, 3)), width="content")
