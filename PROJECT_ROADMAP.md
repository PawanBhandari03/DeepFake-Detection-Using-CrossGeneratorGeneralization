# Project Roadmap — Unified Content Authenticity Verification System

> **Team:** The Perceptrons — Pawan Singh Jagdish Bhandari · Rahul H. Bramhankar · Bhoomi Kushwah
>
> **Guide:** Dr. Gayatri Bhandari (HOD) · **College:** JSPM's BSIOTR, Wagholi, Pune · BE Computer Engineering
>
> **Last updated:** 2026-09-27

This file is our single source of truth. Every week we tick the boxes, fill the weekly log, and use it to report to our supervisor.

**Project documents:**

| Document | Source |
|---|---|
| This roadmap | `PROJECT_ROADMAP.md` |
| Project overview and algorithms (4 pages) | [docs/Overview_and_Algorithms.tex](docs/Overview_and_Algorithms.tex) |
| Full system design: architecture, algorithms, UML/DFD diagrams (31 pages) | [docs/System_Design.tex](docs/System_Design.tex) |
| Review-III questions and answers | [docs/Review3_QA.md](docs/Review3_QA.md) |
| Quick answers cheat sheet (1–3 line answers) | [docs/Quick_Answers.md](docs/Quick_Answers.md) |
| **Walkthrough 01 — Image prototype** (what we built, how it works, datasets, results) | [docs/walkthroughs/01_image_prototype.md](docs/walkthroughs/01_image_prototype.md) |

**Code:** [prototype/](prototype/) — the image prototype web app (run: `.venv\Scripts\streamlit run prototype\app.py`).

To make a PDF for sharing: `python docs/tools/md2pdf.py <file>.md` for Markdown files; the `.tex` files compile on Overleaf.

---

## Current status

| | |
|---|---|
| **Current phase** | Phase 1 — Image (started early: first prototype built) |
| **Current week** | Week 1 · Sep 28 – Oct 4, 2026 · Setup and learning |
| **Next milestone** | Prototype submission — Mon 28 Sept 2026 (teacher's request); then Phase 1 demo — Week 11 |
| **Blocked on** | Nothing |

> **Change to the plan (27 Sept):** the teacher asked for a working prototype by Mon 28 Sept, so a first version of the Week 11 app was built early — see [Walkthrough 01](docs/walkthroughs/01_image_prototype.md). It already covers parts of Weeks 1–5 (environment, data, evaluation script, CLIP model, cross-generator test). The weekly plan below stays; those weeks now **improve** the prototype instead of starting from zero.

---

## 1. What we are building (in plain words)

A web app where a user uploads an **image, a video, or a document**, and gets back:

1. A **Trust Score from 0 to 100** (100 = looks authentic).
2. A **verdict**: *Likely Authentic*, *Needs Human Review*, or *Likely Manipulated*.
3. **Evidence**: a heatmap of suspicious regions and a short plain-language explanation.

Our main research idea is **cross-generator generalization**: the model must catch fakes made by AI tools it **never saw during training** (for example, trained on Stable Diffusion, tested on Midjourney or FLUX). Every result we report must include these "unseen" tests.

We build it in three phases, one at a time. Each phase ends with a working demo.

---

## 2. How we work

- **One phase at a time.** We finish and demo a phase before starting the next.
- **One small goal per week.** Each week has a "Show supervisor" line: that is what we report.
- **Every week:** tick the boxes below, add an entry to the [Weekly log](#11-weekly-log), push code to GitHub.
- **If exams or holidays take a week:** write "Buffer — exams" in the log and push the remaining weeks down. Don't skip steps.
- **Roles rotate each phase** so everyone can answer any viva question.

| Role | What they do | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|---|
| **Model lead** | Trains models, runs experiments | _(name)_ | _(name)_ | _(name)_ |
| **Data & Evaluation lead** | Datasets, train/test splits, evaluation scripts, results tables | _(name)_ | _(name)_ | _(name)_ |
| **App & Report lead** | Streamlit app, Trust Score, explanations, weekly report, this file | _(name)_ | _(name)_ | _(name)_ |

### Key dates (fill in)

| Event | Date |
|---|---|
| Semester exams | _(fill in)_ |
| Project review dates | _(fill in)_ |
| Final submission | _(fill in)_ |

---

## 3. The three phases at a glance

| Phase | Weeks | Dates | Goal | Ends with |
|---|---|---|---|---|
| **1 — Image** | 1–11 | Sep 28 – Dec 13, 2026 | Detect fully AI-generated images, including from unseen generators | Streamlit demo: upload image → Trust Score + heatmap |
| **2 — Video** | 12–16 | Dec 14, 2026 – Jan 17, 2027 | Detect face-swapped / manipulated videos | Video tab: upload video → score + timeline of suspicious frames |
| **3 — Document** | 17–26 | Jan 18 – Mar 28, 2027 | Find tampered regions in documents | Full app (image + video + document) + final report |

Phase 1 is the longest because it includes setup, learning, and building the Trust Score and app that Phases 2 and 3 reuse.

---

## 4. Decisions we have made (and why)

These came from our research-gap study. Status "Proposed" means we still need to confirm with the team and guide.

| # | Decision | Why | Status |
|---|---|---|---|
| D1 | "Deepfake" in our project means three things, one per phase: (1) fully AI-generated images, (2) face-manipulated videos, (3) tampered documents | These are different problems with different datasets; mixing them caused confusion in our first plan | Proposed |
| D2 | Main image model = **frozen CLIP encoder + small classifier head**. **EfficientNet** is our baseline to compare against | Recent papers show CLIP-style encoders generalize much better to unseen generators, and it is cheap to train on free Colab | Proposed |
| D3 | Always test on **generators / methods the model never saw** | This is our core claim; a model that only works on its training generator proves nothing | Proposed |
| D4 | Make real and fake images the **same format and size** before training | Known trap: models learn "JPEG = real, PNG = fake" instead of real artifacts (paper: *Fake or JPEG?*) | Proposed |
| D5 | Trust Score = **calibrated** 0–100 with a **Needs Human Review** band | Detectors are overconfident; a calibrated score with a review band is honest and is one of our contributions | Proposed |
| D6 | Do **not** claim "first unified system" | ForensicHub (NeurIPS 2025) already unifies these domains as a benchmark. We claim: explainable, calibrated, end-user system with generalization tested in every phase | Proposed |
| D7 | Do **not** use the CIFAKE dataset | 32×32 images; gives 98% accuracy that means nothing for real-world generalization | Proposed |
| D8 | For documents, test on **real human-made forgeries**, not only our own synthetic ones | Models trained on synthetic forgeries often fail on real ones | Proposed |

---

## 5. Setup: tools and datasets

### Tools

| Purpose | Tool |
|---|---|
| Code | Python 3.10+, PyTorch, GitHub (shared repo) |
| GPU | Google Colab / Kaggle (free tier) |
| Image models | `open_clip` or Hugging Face `transformers` (CLIP), `timm` (EfficientNet) |
| Vision & forensics | OpenCV, NumPy, scikit-image |
| Evaluation | scikit-learn (accuracy, AUC, F1, calibration) |
| Explainability | `pytorch-grad-cam` |
| App | Streamlit |
| Phase 2 | Face detector (MTCNN from `facenet-pytorch`, or MediaPipe) |
| Phase 3 | Tesseract / EasyOCR, `segmentation_models_pytorch` (U-Net), PyMuPDF / pikepdf (PDF checks) |

**Colab tip:** keep datasets as `.zip` files in Google Drive and unzip them to Colab's local disk each session. Reading thousands of small files directly from Drive is very slow.

### Datasets — send access requests in Week 1

Some datasets need a form or email and can take weeks to approve. Request everything now, even the ones for Phase 2 and 3.

| Dataset | Used in | How to get it | Status |
|---|---|---|---|
| GenImage (subsets) | Phase 1 | Public download — very large, download only the subsets we need | ☐ Not started |
| FaceForensics++ | Phase 2 | Fill the request form on their GitHub | ☐ Not requested |
| Celeb-DF v2 | Phase 2 | Request form | ☐ Not requested |
| DF40 (optional) | Phase 2 | Check their GitHub for access | ☐ Not checked |
| DocTamper | Phase 3 | Email the authors with college ID | ☐ Not requested |
| RTM (real manual text forgeries) | Phase 3 | Check their GitHub for access | ☐ Not checked |
| OSTF (optional, AI text edits) | Phase 3 | Check their GitHub for access | ☐ Not checked |

---

## 6. Phase 1 — Image (Weeks 1–11)

**Goal:** upload an image → the app says whether it is AI-generated, with a Trust Score and evidence, and it still works on generators the model never saw.

**Standard test setup (from the GenImage benchmark):** train on **Stable Diffusion v1.4** only, then test on all 8 generators: SD v1.4, SD v1.5, Midjourney, ADM, GLIDE, Wukong, VQDM, BigGAN. Anything other than SD v1.4 counts as "unseen".

**What we measure:** accuracy, AUC, precision, recall, F1 — for **each generator separately**.

### Week 1 · Sep 28 – Oct 4 · Setup and learning
- [ ] Create the shared GitHub repo and a shared Google Drive folder
- [ ] Everyone runs one PyTorch beginner tutorial (train a small image classifier) on Colab
- [ ] Send all dataset access requests from [Section 5](#datasets--send-access-requests-in-week-1)
- [ ] Everyone reads two papers: *UniversalFakeDetect* (CLIP for fake detection) and *Fake or JPEG?* (dataset bias)
- [ ] Confirm decisions D1–D8 with the team and guide

**Show supervisor:** repo set up, dataset requests sent, 1-page notes on the two papers.

### Week 2 · Oct 5 – Oct 11 · Get the image data
- [ ] Download GenImage subsets: start small (about 10k real + 10k fake from SD v1.4 for training; 1–2k per other generator for testing)
- [ ] Split into train / validation / test folders
- [ ] Write a **data card**: image counts, sizes, and file formats (JPEG vs PNG) for real and fake
- [ ] Make a grid of sample images (real vs each generator)

**Show supervisor:** data card table + sample image grid.

### Week 3 · Oct 12 – Oct 18 · Build the evaluation script first
- [ ] Write `evaluate.py`: takes predictions → prints accuracy, AUC, precision, recall, F1, confusion matrix, and a **per-generator results table**
- [ ] Test it with random predictions (should give ~50% accuracy, ~0.5 AUC)
- [ ] Create the results table in the [Results board](#9-results-board)

**Show supervisor:** working evaluation script and empty results table layout.

### Week 4 · Oct 19 – Oct 25 · Baseline: EfficientNet
- [ ] Fine-tune EfficientNet-B0 (pretrained on ImageNet) on SD v1.4 real vs fake
- [ ] Test on all 8 generators
- [ ] Expected: high accuracy on SD v1.4, lower on others. **This shows the generalization problem our project solves.**

**Show supervisor:** first per-generator results table — seen vs unseen gap.

### Week 5 · Oct 26 – Nov 1 · Main model v1: CLIP features
- [ ] Extract CLIP image features once and save them to disk (`.npy`)
- [ ] Train a simple classifier (logistic regression) on the saved features
- [ ] Tip: **center-crop** images instead of resizing them — resizing blurs away the tiny artifacts we want to detect
- [ ] Test on all 8 generators and compare with EfficientNet

**Show supervisor:** EfficientNet vs CLIP comparison table.

### Week 6 · Nov 2 – Nov 8 · Remove dataset shortcuts
- [ ] Re-save all real and fake images the same way (same format and JPEG quality, same crop size)
- [ ] Add random JPEG compression during training
- [ ] Retrain both models and compare before vs after

**Show supervisor:** before/after table — proof the model learns real artifacts, not file-format tricks.

### Week 7 · Nov 9 – Nov 15 · Forensic evidence: frequency and noise
- [ ] Compute noise residuals (image minus a smoothed copy) and their **FFT spectrum**
- [ ] Plot the average spectrum for real images and each generator — look for grid-like peaks
- [ ] Try adding these frequency features to the CLIP classifier; does it help on unseen generators?

**Show supervisor:** spectrum figures (real vs each generator) + whether fusion helped.

### Week 8 · Nov 16 – Nov 22 · Robustness tests
- [ ] Test the best model on images after: JPEG quality 90 / 70 / 50, resize to half, blur, a screenshot-like re-save
- [ ] Plot accuracy vs each change (robustness curve)

**Show supervisor:** robustness curves — how the model behaves on "real-world" shared images.

### Week 9 · Nov 23 – Nov 29 · Our own 2026 unseen test set
- [ ] Generate 500–1000 images on Colab with newer open models (for example SDXL / SD 3.5 class) using COCO captions as prompts
- [ ] Use the matching real COCO images as the "real" side
- [ ] Test our model on this set — generators newer than anything in training

**Show supervisor:** results on our own 2026 test set (this is an original contribution).

### Week 10 · Nov 30 – Dec 6 · Explanations and calibrated Trust Score
- [ ] Heatmaps: Grad-CAM on EfficientNet; a patch-level score map for the CLIP model (score small crops of the image)
- [ ] Calibrate the model's probability on the validation set (temperature / Platt scaling); plot a reliability diagram
- [ ] Turn it into a 0–100 Trust Score and choose the three verdict bands using validation data (not guessed numbers)
- [ ] Write template sentences for the plain-language explanation

**Show supervisor:** heatmaps, calibration plot, verdict bands.

### Week 11 · Dec 7 – Dec 13 · Phase 1 demo ⭐ Milestone
- [ ] Streamlit app v1: upload image → Trust Score, verdict, heatmap, frequency plot, explanation
- [ ] Write a 2–3 page Phase 1 summary (method, results tables, limitations)
- [ ] Fill in the Phase 1 section of the [Results board](#9-results-board)

**Show supervisor:** live demo + Phase 1 summary.

**Phase 1 done when:** ☐ demo works ☐ results table filled (seen + unseen + own 2026 set) ☐ summary written ☐ supervisor has seen it

---

## 7. Phase 2 — Video (Weeks 12–16)

**Goal:** upload a video → per-frame analysis of faces → one Trust Score for the whole video + a timeline showing which moments look fake.

**Standard test setup:** train on **FaceForensics++**, test on **Celeb-DF v2** (a different dataset = "unseen" methods).

**What we measure:** frame-level and video-level AUC and accuracy, on the training dataset and the unseen dataset.

**Important:** split train/test **by video**, never by frame. Frames from the same video in both train and test = cheating (data leakage).

### Week 12 · Dec 14 – Dec 20 · Video basics
- [ ] Read a video with OpenCV and sample frames (for example, 1–2 per second)
- [ ] Detect and crop faces (MTCNN or MediaPipe)
- [ ] Script: `video → folder of face crops`

**Show supervisor:** face crops extracted from a sample video.

### Week 13 · Dec 21 – Dec 27 · Prepare video data
- [ ] Run the script on a FaceForensics++ subset (use the c23 compression version)
- [ ] Split by video ID into train / validation / test; write a data card
- [ ] Do the same for a Celeb-DF v2 test subset

**Show supervisor:** data card for both video datasets.

### Week 14 · Dec 28 – Jan 3 · Frame-level model
- [ ] First, test the Phase 1 image model directly on face frames (does it work without retraining?)
- [ ] Then train the CLIP + classifier model on FaceForensics++ face crops
- [ ] Compare both

**Show supervisor:** frame-level results table.

### Week 15 · Jan 4 – Jan 10 · From frames to one video score
- [ ] Combine frame scores into a video score: average, maximum, average of the top 20%
- [ ] Test on Celeb-DF v2 (unseen) and compare the methods

**Show supervisor:** video-level results, trained on FF++, tested on Celeb-DF.

### Week 16 · Jan 11 – Jan 17 · Phase 2 demo ⭐ Milestone
- [ ] Heatmaps on the most suspicious face frames
- [ ] Timeline chart of fake-score over the video
- [ ] Calibrate the video score and reuse the Trust Score bands
- [ ] Add a Video tab to the Streamlit app; write the Phase 2 summary

**Show supervisor:** live video demo + Phase 2 summary.

**Phase 2 done when:** ☐ video tab works ☐ cross-dataset results filled ☐ summary written ☐ supervisor has seen it

---

## 8. Phase 3 — Document (Weeks 17–26)

**Goal:** upload a scanned document, photo of a document, or PDF → the app highlights edited regions and runs logic checks, then gives a Trust Score.

**Types of tampering we detect:** copy-move (copying a word or number elsewhere), splicing (pasting text from another document), text replacement.

**What we measure:** pixel-level F1 and IoU (how well the highlighted area matches the true edited area), plus image-level accuracy/AUC. Tested on **our synthetic forgeries** and on **real human-made forgeries**.

### Week 17 · Jan 18 – Jan 24 · Clean documents and OCR
- [ ] Collect clean documents: public receipt/invoice datasets and synthetic invoices generated with Python (no personal documents)
- [ ] Run Tesseract or EasyOCR and draw word boxes on the documents

**Show supervisor:** set of clean documents + OCR output with word boxes.

### Week 18 · Jan 25 – Jan 31 · Forgery generator
- [ ] Python script that creates forged documents **plus a mask** of the edited area:
    - copy-move a word or number
    - paste text from another document
    - erase a number and re-type it in a similar font
- [ ] Add realistic after-effects: JPEG re-save, slight blur, scan-like noise
- [ ] Generate 2k–5k forged documents

**Show supervisor:** forged documents next to their masks.

### Week 19 · Feb 1 – Feb 7 · Forensic features for documents
- [ ] Compute and visualize: Error Level Analysis (ELA), SRM noise residuals, DCT-based maps
- [ ] Find which feature makes each type of edit visible

**Show supervisor:** figure comparing features on the same forged document.

### Week 20 · Feb 8 – Feb 14 · Localization model v1
- [ ] Train a U-Net (EfficientNet encoder, `segmentation_models_pytorch`) on RGB images to predict the edit mask
- [ ] Measure pixel F1 and IoU

**Show supervisor:** predicted masks vs true masks + scores.

### Week 21 · Feb 15 – Feb 21 · Dual-stream model
- [ ] Add a second input stream with the noise / ELA / DCT features (start simple: stack them as extra input channels)
- [ ] Compare with v1

**Show supervisor:** v1 vs dual-stream comparison.

### Week 22 · Feb 22 – Feb 28 · Real-forgery test
- [ ] Test on real human-made forgeries (RTM; DocTamper if access was granted)
- [ ] Measure the gap between synthetic and real results; try stronger after-effect augmentation to shrink it
- [ ] Optional: test on AI-edited text (OSTF) — the "unseen generators" of documents

**Show supervisor:** synthetic vs real results — honest about the gap.

### Week 23 · Mar 1 – Mar 7 · Logic and file checks
- [ ] OCR logic checks on invoices: do line items add up to the total? Are dates valid?
- [ ] Font consistency: flag words whose height or baseline differs from their neighbours
- [ ] PDF checks: edited-after-creation history, mismatched creation/modification dates, unexpected extra fonts
- [ ] Image metadata (EXIF) check

**Show supervisor:** examples of forgeries caught by each check.

### Week 24 · Mar 8 – Mar 14 · Document Trust Score and tab
- [ ] Combine the localization score and check results into one calibrated document Trust Score
- [ ] Explanation text: what was found and where
- [ ] Add a Document tab to the Streamlit app

**Show supervisor:** document demo.

### Week 25 · Mar 15 – Mar 21 · Full system integration
- [ ] One app for all three: suggest the right mode from the file type, same Trust Score report style
- [ ] Test end-to-end on a mixed set of images, videos, and documents; fix bugs

**Show supervisor:** full system demo.

### Week 26 · Mar 22 – Mar 28 · Final report ⭐ Milestone
- [ ] Final report with all results tables and limitations
- [ ] Final presentation slides
- [ ] Viva preparation: every member explains every phase

**Show supervisor:** final report, slides, and demo.

**Phase 3 done when:** ☐ document tab works ☐ synthetic + real results filled ☐ full app integrated ☐ final report submitted

---

## 9. Results board

Fill these in as experiments finish. Use `—` for "not run yet".

### Phase 1 — Image (accuracy / AUC, trained on SD v1.4)

| Model | SD v1.4 (seen) | SD v1.5 | Midjourney | ADM | GLIDE | Wukong | VQDM | BigGAN | Own 2026 set | Unseen average |
|---|---|---|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | — | — | — | — | — | — | — | — | — | — |
| CLIP + classifier | — | — | — | — | — | — | — | — | — | — |
| CLIP + classifier (debiased) | — | — | — | — | — | — | — | — | — | — |
| CLIP + frequency features | — | — | — | — | — | — | — | — | — | — |

### Phase 2 — Video (video-level AUC)

| Model | FaceForensics++ (seen) | Celeb-DF v2 (unseen) |
|---|---|---|
| Phase 1 image model, no retraining | — | — |
| CLIP + classifier on face frames | — | — |

### Phase 3 — Document (pixel F1 / IoU)

| Model | Our synthetic forgeries | Real forgeries (RTM) | DocTamper (if available) |
|---|---|---|---|
| U-Net (RGB only) | — | — | — |
| Dual-stream U-Net | — | — | — |

---

## 10. Rules to avoid fooling ourselves

1. **Always report unseen results next to seen results.** Never only the good number.
2. **Same processing for real and fake** (format, size, compression).
3. **Split by source**: by video for Phase 2, by original document for Phase 3.
4. **Never tune on the test set.** Choose thresholds and settings on the validation set.
5. **Write every experiment down** in the Results board and the weekly log, including failures.
6. **Cite everything.** No made-up statistics in slides or reports.

---

## 11. Weekly log

Copy this block for each week.

```
### Week N · <dates> · <status: Done / Partly done / Buffer>
- Done:
- Results:
- Problems / blockers:
- Told supervisor:
- Next week:
```

### Week 0 · Sep 21 – Sep 27, 2026 · Done
- Done: Finalized project idea (Projects 2 + 3 merged); research-gap study; created this roadmap; wrote the System Design document (algorithms, flowcharts, UML/DFD diagrams) in LaTeX; Review-III Q&A and cheat sheet.
- Done (26–27 Sept, prototype): built the first working **image prototype** (`prototype/`): drag-and-drop upload → frozen CLIP ViT-L/14 + classifier → Trust Score, verdict, heatmap, frequency spectrum, ELA, metadata. Trained our own head on SD1.5 only (Tiny-GenImage) and tested on unseen generators, including ChatGPT's and Gemini's image models (OpenFakeTiny). See [Walkthrough 01](docs/walkthroughs/01_image_prototype.md).
- Results: Decided to build in three phases (Image → Video → Document) and to test on unseen generators in every phase. Prototype results (AUC, never-seen generators): ChatGPT's GPT-Image-1.5 — our v2 **0.84**, our v1 (trained on SD1.5 only) **0.85**, published UniversalFakeDetect **0.48** (guessing); Google Veo 3 — v2 **0.92**. Biggest weakness: JPEG-compressed AI images. Details: Walkthrough 01, Section 7.
- Problems / blockers: DocTamper and FaceForensics++ need access requests. The Tiny-GenImage copy has no SD1.4, so we train on SD1.5. Laptop has 7.8 GB RAM and no GPU, so data is read in small batches and experiments stay small.
- Told supervisor: _(fill in after Monday)_
- Next week: Week 1 — show the prototype; setup, dataset requests, first two papers.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **Generator** | An AI tool that makes images or videos (Stable Diffusion, Midjourney, DALL·E, FLUX, GANs) |
| **Cross-generator generalization** | Detecting fakes from generators the model never saw in training |
| **Seen / unseen** | Seen = generator used in training; unseen = not used in training |
| **CLIP** | A large pretrained image-text model; we use its image features as input to a small classifier |
| **EfficientNet** | A standard efficient CNN; our baseline |
| **AUC** | How well the model separates real from fake across all thresholds (0.5 = random, 1.0 = perfect) |
| **Calibration** | Making "80% confident" actually be right about 80% of the time |
| **Grad-CAM** | A heatmap showing which image regions drove the model's decision |
| **FFT / DCT** | Ways to look at an image as frequencies; AI and edits leave patterns there |
| **SRM filters / noise residual** | Filters that remove image content and keep the fine noise, where editing traces hide |
| **ELA (Error Level Analysis)** | Re-saving a JPEG and looking at differences; edited regions often react differently |
| **U-Net** | A model that outputs a mask (which pixels are edited) instead of one label |
| **Pixel F1 / IoU** | How well the predicted edit mask overlaps the true edit mask |
| **Data leakage** | When test data sneaks into training (for example, frames from the same video), making results look falsely good |

---

## 13. Key references

- GenImage benchmark (NeurIPS 2023) — the standard train-on-SD-v1.4, test-on-others protocol
- Ojha et al., *Towards Universal Fake Image Detectors* (UniversalFakeDetect), CVPR 2023 — CLIP features for fake detection
- Grommelt et al., *Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets* — https://arxiv.org/abs/2403.17608
- Yermakov et al., *Deepfake Detection that Generalizes Across Benchmarks* (2025) — https://arxiv.org/abs/2508.06248
- *How well are open sourced AI-generated image detection models out-of-the-box* (2026) — https://arxiv.org/abs/2602.07814
- Chandra et al., *Deepfake-Eval-2024* — https://arxiv.org/abs/2503.02857
- Rössler et al., *FaceForensics++*, ICCV 2019
- DF40 (NeurIPS 2024) — https://github.com/YZY-stack/DF40
- ForensicHub (NeurIPS 2025) — https://arxiv.org/abs/2505.11003
- Qu et al., *DocTamper*, CVPR 2023 — https://github.com/qcf-568/DocTamper
- Luo et al., *Toward Real Text Manipulation Detection* (RTM) — https://arxiv.org/abs/2312.06934
- Qu et al., *Revisiting Tampered Scene Text Detection in the Era of Generative AI* (OSTF), AAAI 2025 — https://github.com/qcf-568/OSTF
- Dhouib et al., *Efficient Document Tampering Localization with Multi-Level Discrepancy Features and Unified DCT-Quantization Embedding* (June 2026) — https://arxiv.org/abs/2606.22285
