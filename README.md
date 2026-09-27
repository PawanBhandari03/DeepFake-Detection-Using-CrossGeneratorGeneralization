# DeepFake Detection Using Cross-Generator Generalization

**Unified Content Authenticity Verification System** — a final-year BE Computer Engineering project by **The Perceptrons** (Pawan Singh Jagdish Bhandari, Rahul H. Bramhankar, Bhoomi Kushwah), JSPM's BSIOTR, Wagholi, Pune. Guide: Dr. G. M. Bhandari (HOD).

Upload an image and get a **Trust Score (0–100)**, a verdict (*Likely Authentic / Needs Human Review / Likely AI-generated*), a heatmap and a plain-language explanation. The core idea is **cross-generator generalization**: the detector must catch images from AI tools it **never saw during training**.

> Status: **Phase 1 (images) — working prototype.** Video (Phase 2) and document tamper localization (Phase 3) follow. See the [roadmap](PROJECT_ROADMAP.md).

---

## Results (prototype)

Our model **v2** was trained on many generators, but **every OpenAI and Google image was removed** from its training data — so ChatGPT and Gemini images are completely unseen.

| Model | Trained on | ChatGPT (GPT-Image-1.5) | Gemini (Nano Banana Pro) | Google Veo 3 | Real photos correct |
|---|---|---|---|---|---|
| **Ours v2** | many generators, no OpenAI / Google | AUC **0.84** · 68% caught | 0.90 · 3 of 4 | **0.92** · 71% | 89% |
| **Ours v1** | one generator only (Stable Diffusion 1.5) | AUC **0.85** · 60% caught | 0.83 · 3 of 4 | 0.86 · 57% | 86% |
| UniversalFakeDetect (published) | one generator only (ProGAN, 2018) | AUC 0.48 · 6% caught | 0.53 · 1 of 4 | 0.49 · 7% | 89% |

AUC 0.5 = guessing, 1.0 = perfect. A detector trained on one old generator collapses on modern ones; ours still separates them from real photos. Full tables, honest limitations (notably JPEG compression) and screenshots: [Walkthrough 01](docs/walkthroughs/01_image_prototype.md).

---

## Quick start

Requires Python 3.10+ (no GPU needed; about 12–15 s per image on a laptop CPU).

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r prototype\requirements.txt    # Windows
.venv\Scripts\streamlit run prototype\app.py
```

On macOS/Linux use `.venv/bin/python` and `.venv/bin/streamlit`. The app opens at `http://localhost:8501`; the first run downloads the CLIP model (~1.7 GB).

---

## How it works

```
image (any format) → standard RGB → 10 patches of 224×224 at full resolution (no resizing)
      → JPEG re-save (removes file-format shortcuts)
      → frozen CLIP ViT-L/14 (768 features) + FFT noise-spectrum features (64)
      → logistic-regression head → probability p
      → Trust Score = 100 × (1 − p) → verdict + heatmap + explanation
```

Extra evidence that needs no training: frequency spectrum, Error Level Analysis, and file metadata (camera EXIF, C2PA Content Credentials).

---

## Repository structure

```
prototype/            Streamlit web app and model code
  app.py              web interface (Check an image · Cross-Generator Test · How it works)
  detector.py         CLIP encoder, patching, classifier heads
  forensics.py        image loading, FFT spectrum, ELA, metadata
  trust.py            Trust Score, verdict, heatmap, explanations
  experiment.py       feature extraction, training and cross-generator evaluation
  weights/            trained heads (ours v1, v2) and the published UniversalFakeDetect head
  results/            evaluation results shown in the app
  test_images/        44 demo images (real, ChatGPT, Gemini, Midjourney v7, FLUX.2)
docs/
  walkthroughs/       step-by-step explanation of each build (start here)
  System_Design.tex   full system design: architecture, algorithms, UML/DFD diagrams
  Overview_and_Algorithms.tex   4-page overview
  Review3_QA.md, Quick_Answers.md   review preparation
PROJECT_ROADMAP.md    26-week plan and weekly log
```

To reproduce the experiment, download the datasets as described in the walkthrough, then run `python prototype/experiment.py extract`, `train`, `extract_v2`, `train_v2`.

---

## Datasets and third-party components

| Component | Source | Licence |
|---|---|---|
| CLIP ViT-L/14 image encoder | OpenAI, loaded with [open_clip](https://github.com/mlfoundations/open_clip) | MIT |
| UniversalFakeDetect head (`prototype/weights/ufd_fc_weights.pth`) | Ojha, Li & Lee, CVPR 2023 — [github.com/WisconsinAIVision/UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) | MIT |
| Tiny-GenImage (training / testing) | [TheKernel01/Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage), subset of GenImage (NeurIPS 2023) | CC BY-NC-SA 4.0 |
| OpenFakeTiny (training / testing, demo images) | [ComplexDataLab/OpenFakeTiny](https://huggingface.co/datasets/ComplexDataLab/OpenFakeTiny), subset of OpenFake | CC BY-NC 4.0 |

Datasets are not included in this repository (only 44 small demo images, used for non-commercial academic demonstration). Our own code is released under the [MIT License](LICENSE).
