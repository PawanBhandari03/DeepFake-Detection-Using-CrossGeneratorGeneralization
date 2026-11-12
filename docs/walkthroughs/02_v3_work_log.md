# Walkthrough 02 — v3 Work Log (stronger model, more data)

> **Started:** 13–14 Oct 2026 · **Phase:** 1 (Images) · **Status:** v3 trained and in the app; edited-image and team-photo tests still to do
>
> This is the log we use to explain to the teachers and external examiners **what we did, why, and what we found**. Every entry has: what we did, why, how, and the result. Entries are added as the work happens, newest at the bottom. Nothing here is written before it is done.
>
> Related: [Walkthrough 01](01_image_prototype.md) (first prototype) · [Team Briefing](../Team_Briefing_v3_Plan.md) (plan for the team) · [Roadmap](../../PROJECT_ROADMAP.md)

---

## 0. Why we are doing this

**Problem.** The first prototype (models v1 and v2) was shown to the teacher. Feedback: it was trained on older generators and does not work well on new images from ChatGPT and Gemini; she will test with real photos (faces, clothes, backgrounds) and with images that are only **partly** edited by AI.

**Measured weaknesses of v1 / v2** (see Walkthrough 01, Section 7): ChatGPT images only 68% caught (AUC 0.84); ChatGPT JPEG copies 1 of 6 caught; only 4 Gemini test images; about 6,000 training images in total; only a coarse 3 × 3 heatmap, so small edited regions are averaged away.

**Decision.** Keep v1 and v2 unchanged as the *experiment* that proves cross-generator generalization. Build **v3** as the strong demo model, and a **proof model** (same recipe with one generator family held out) so the claim stays honest.

---

## 1. Log

### Entry 1 — 13 Oct · Recovered the project context
- **What:** Lost chat history (laptop cleaned, folder path changed). Rebuilt the full picture from the repository: `PROJECT_ROADMAP.md`, Walkthrough 01, the prototype code, results and weights.
- **Why:** The new work must continue the same plan and use the same experiment protocol.
- **Result:** Estimated progress: planning and documents about 90%; Phase 1 about 45–50%; whole project about 20%. All roadmap boxes were still unticked even though parts of Weeks 1–7 were already done by the prototype. (We tick them as we verify each item, see Section 2.)

### Entry 2 — 13–14 Oct · Planning decisions made with the team lead
| Decision | Reason |
|---|---|
| Keep v1 and v2 frozen | They are our evidence that the method generalizes to unseen generators |
| Build v3 with ChatGPT and Gemini images in training | The demo must work on them |
| Build a proof model with one family held out | Keeps the cross-generator claim honest |
| Train with **JPEG augmentation** (random compression, resize, blur) | The biggest weakness: compression erased the traces (1 of 6 ChatGPT JPEGs caught) |
| Finer heatmap and a verdict based on the most suspicious region | Handles partly edited images |
| Add masked-edit datasets | To learn region edits (inpainting, generative fill) |
| Use Kaggle GPU for feature extraction | The laptop CPU manages only about 6,000 images overnight |
| Keep Streamlit as fallback; new frontend talks to a FastAPI wrapper | Streamlit cannot serve as an API for a separate frontend |

### Entry 3 — 14 Oct · Looked for edited-image datasets
- **How:** Web search for datasets with edit masks, then queried the Hugging Face API for what is actually downloadable (public or gated, size, licence, file layout).
- **Found and confirmed on Hugging Face (public, no access request):**
  - **DiffSeg30k** (`Chaos2629/Diffseg30k`, Apache-2.0): about 30k images edited by 8 diffusion editors (SD2 inpainting, Kolors, SD3.5, FLUX, SDXL inpainting, GLIDE, HunyuanDiT, Kandinsky), each with a pixel **mask** of the edited region; images can have 1–3 edits. Train 21.5 GB, validation 4.9 GB.
  - **tiny-audits** (`DivyaApp/tiny-audits`): COCO and NEWS images with `original`, `manipulated` and `mask` folders for about 10 inpainting methods (GLIDE, SDXL, PowerPaint, Firefly…); about 2.7 GB; licence not stated.
  - **OpenFake** (`ComplexDataLab/OpenFake`, CC-BY-NC-4.0): see Entry 4.
- **Not confirmed (papers only, no downloadable dataset found):** COCO-Inpaint, PromptForge-350k, MLLM-Edit. We do **not** plan around them.

### Entry 4 — 14 Oct · Read the full OpenFake layout (important)
The full OpenFake dataset (v2 release) has three parts that fit our needs exactly:

| Part | Size | What it is | Our use |
|---|---|---|---|
| `core/train` | 2.3 M images, 608 files, 3.3 TB | Real photos + many generators, including some OpenAI and Google models (GPT-Image-1, DALL-E 3, Nano Banana, Imagen) | v3 **training** (we only read slices) |
| `core/validation` | 58 k images, 72 GB | Held-out split for choosing thresholds | Calibration and thresholds (roadmap Week 10) |
| `core/test` | 91 k images, 68 GB | **Held-out generators**, no overlap with training generators, real images from different sources | The honest **cross-generator test**; includes the newer **GPT-Image-1.5** and **Nano Banana Pro** |
| `reddit/test` | 36 k images, 25 GB | **In-the-wild** images scraped from AI-art and photography communities (compressed, re-shared) | Realistic robustness test |

**Why this matters.** If v3 trains on older ChatGPT/Gemini versions (GPT-Image-1, Nano Banana) and is tested on the *newer* versions (GPT-Image-1.5, Nano Banana Pro) in the held-out test split, that is a true, honest test on unseen generators that the examiners can follow.

*Verified in Entry 5:* which generators appear in which split.

### Entry 5 — 14 Oct · Kaggle probe run (can we do this on a free GPU?)
- **What:** Wrote `kaggle/probe/probe.py` and ran it on Kaggle through the Kaggle command-line tool (the account is already logged in on the laptop). It prints a report and trains nothing.
- **Why:** Before building a big pipeline, check three things: the GPU speed, whether Hugging Face data can be read from Kaggle, and what is inside each OpenFake split.
- **Results:**
  - GPU: Tesla T4. The frozen CLIP ViT-L/14 encoder processes **112 patches per second in half precision** (25 in full precision). That is about 20 times the laptop CPU (about 1 patch per second).
  - OpenFake rows are **shuffled inside each file**, so reading a few row groups from many files gives a good mix of real images and about 60 generators. About 45% are real (LAION and Pexels photos), the rest fake.
  - `gpt-image-1` (older ChatGPT) and `dalle-3` appear in training; the held-out test split has `gpt-image-1.5`, `gpt-image-2` (ChatGPT) and `nano-banana-pro` (Gemini), `veo-3`, `sora-2`.
  - DiffSeg30k (zip) and tiny-audits can be read remotely without downloading everything.

### Entry 6 — 14 Oct · Wrote the feature-extraction pipeline (`kaggle/extract/extract.py`)
What it does, step by step (this is the "how v3 is made" for the examiners):

| Step | What happens | Why |
|---|---|---|
| 1 | Pick images: random row groups of OpenFake train, plus extra row groups that contain OpenAI or Google generators | A wide mix of generators, and enough ChatGPT/Gemini images (they are rare in the dataset, about 3% and under 1%) |
| 2 | Each training image gives **two views**: the clean image, and a **randomly degraded copy** (50% resize to 0.5–1.0×, 20% blur, 90% JPEG quality 50–95) | **JPEG augmentation.** Real shared images are compressed; this teaches the model to find traces in damaged images |
| 3 | 3 random 224 × 224 patches per view, taken at native resolution (no resizing of the patch) | Same as the app: resizing would blur away the tiny traces |
| 4 | Every patch is re-saved as JPEG quality 95 | Same as the app, so training and use see the same thing, and the model cannot learn a file-format shortcut |
| 5 | Frozen CLIP encoder (half precision) gives 768 numbers per patch; saved with the label, the generator name, the variant and the patch position | Training later takes minutes |
| 6 | **Evaluation sets** use the app layout (centre patch + 3 × 3 grid) and three variants: clean, JPEG 70, half-size + JPEG 75 | Lets us measure robustness the same way the app scores an image |
| 7 | **Edited-image sets:** DiffSeg30k (8 diffusion editors) and tiny-audits (10 inpainting tools), each with a **mask**. Patches that are at least 50% inside the mask count as AI-edited. Patches fully outside the mask are saved with label -1 (uncertain) and patches in between with -2 | In an edited image the unedited area may still carry AI traces (for example if the whole image is passed through the model), so we do not assume it is "real"; we measure it first |

Splits used: OpenFake **train** for training, **validation** for thresholds and calibration, **test** (held-out generators) for the honest cross-generator result, **reddit** (in-the-wild, already shared and compressed) for realism.

### Entry 7 — 14 Oct · Smoke test of the pipeline, and a correctness check
- **What:** Ran the whole pipeline at 2% size on Kaggle (all stages finished). Then checked the new features against the **old v2 model** (trained on the laptop): if the new pipeline is correct, v2 should reproduce its known results on the new features.
- **Result of the parity check:** v2 on the new features, modern generators (held-out test split sample, 296 images): **AUC 0.957**; our earlier laptop result was 0.95. Half-precision features agree with full precision (cosine similarity 0.99999). So the Kaggle pipeline matches the app.
- **Robustness finding (new, same 296 images, v2 head):**

| Image condition | AUC | AI images caught |
|---|---|---|
| Clean | 0.957 | 86% |
| JPEG quality 70 | 0.829 | **14%** |
| Half size + JPEG 75 | 0.741 | 45% |

  This is the quantified version of the weakness the teacher saw. v3's JPEG augmentation is designed to fix exactly this, and we will repeat this table for v3.
- **Problem found:** the "scan which rows to read" step was slow (about 3 minutes per file). Fix: run the scans in 24 parallel threads.
- **Data sanity:** edit sets gave edited / unedited / mixed patches as expected; `gpt-image-2`, `gpt-image-1.5` and Veo 3 are present in the test sample.

### Entry 8 — 14 Oct · Full extraction launched on Kaggle
- Kernel `pawan0033/deepfake-v3-extract`, version 2: about 30k training images (about 180k patches), validation 1.5k images, test about 3.5k images, reddit about 3.7k images, edit sets about 5k images. Expected duration 2–3 hours. *(Result recorded in the next entry.)*

### Entry 9 — 14 Oct · Full extraction result
- The full Kaggle run (about 5 hours) produced **414,433 patches**: 30,420 training images (182k patches, 52.6% AI), 600 validation images, 3,392 held-out test images (1,200 real + 2,192 AI, 20 generators), 3,859 Reddit images, and 50k edited-image patches. Training contains **2,143 OpenAI images** (GPT-Image-1, DALL-E 3) and **590 Google images** (Imagen 3/4, Nano Banana 2).
- Problems and fixes: (1) the first attempt to wait for the run exited early because the laptop briefly lost internet; (2) the file download from Kaggle was cut off once, so the validation files arrived only on a retry. All 414,433 patches were checked and none was corrupted. (3) The Hugging Face token secret was spelled `HF_Token` while the script looked for `HF_TOKEN`; the script now accepts both (this run used anonymous access and still worked).

### Entry 10 — 14 Oct · v3 results (honest numbers, held-out data)
Training: `prototype/train_v3.py`. Features are L2-normalised and a logistic regression is fitted on top of the frozen CLIP features; the regularisation strength is chosen on the validation split. Evaluation is image-level exactly like the app (mean of 10 patch scores). All numbers below are on images **never used in training**.

**Held-out test split (OpenFake `core/test`: generators that do not appear in training).** AUC, % of AI images caught at probability 0.5, and "real correct" = real photos labelled real.

| Model | Condition | AUC | AI caught | Real correct |
|---|---|---|---|---|
| v2 (old) | clean | 0.947 | 81% | 94% |
| v2 (old) | JPEG quality 70 | 0.839 | **14%** | 100% |
| v2 (old) | half size + JPEG 75 | 0.705 | 50% | 79% |
| v3 without compression training | clean | 0.985 | 93% | 95% |
| v3 without compression training | JPEG 70 | 0.949 | 53% | 99% |
| v3 without compression training | half size + JPEG 75 | 0.929 | 76% | 93% |
| v3 proof (no OpenAI / Google images in training) | clean | 0.977 | 92% | 93% |
| v3 proof | JPEG 70 | 0.936 | 62% | 97% |
| v3 proof | half size + JPEG 75 | 0.935 | 85% | 87% |
| **v3** | clean | **0.982** | **94%** | 93% |
| **v3** | JPEG 70 | **0.949** | **66%** | 97% |
| **v3** | half size + JPEG 75 | **0.950** | **86%** | 90% |

**In-the-wild test (Reddit images, already shared and compressed):** v2 AUC 0.876 / 36% caught → **v3 AUC 0.937 / 73% caught**, real photos called real 92%.

**ChatGPT and Gemini (newer versions than any training image):**

| Generator | Images | v2 (old) | v3 proof (never saw OpenAI/Google) | v3 |
|---|---|---|---|---|
| GPT-Image-1.5 (ChatGPT) | 250 | AUC 0.918 · 70% | 0.957 · 84% | 0.972 · 89% |
| GPT-Image-2 (ChatGPT) | 53 | 0.823 · 43% | 0.952 · 85% | 0.972 · 92% |
| Nano Banana Pro (Gemini) | 47 | 0.923 · 66% | 0.984 · 98% | 0.985 · 94% |
| Veo 3 (Google video) | 250 | 0.939 · 72% | 0.979 · 92% | 0.981 · 93% |
| Sora 2 (OpenAI video) | 76 | 0.782 · 26% | 0.924 · 66% | 0.950 · 76% |

**What we conclude**
1. **v3 fixes the main weakness.** Compressed images: the share of AI images caught at JPEG 70 rose from 14% (v2) to 66% (v3); in-the-wild images from 36% to 73%.
2. **Cross-generator generalization is confirmed on a fair test.** The proof model never saw any OpenAI or Google image and still reaches AUC 0.957 on GPT-Image-1.5 (v2: 0.918) and 0.984 on Nano Banana Pro. Training on more generator families helped even for unseen ones. Adding ChatGPT/Gemini images to training improves it a little more (v3 vs proof: 84% → 89% on GPT-Image-1.5), as expected.
3. **Compression training works, with a small cost.** Without it, the clean-image score is slightly higher (AUC 0.985 vs 0.982) but the model degrades under compression (JPEG 70: 53% vs 66% caught; half size: 76% vs 86%). We chose robustness.
4. **Compressed images are still the hardest case.** At JPEG 70 about a third of AI images are still missed (AUC 0.949 means the model separates them well, but at the 0.5 threshold it is cautious). Entry 11 shows how the verdict bands treat this.
5. **At probability 0.5, about 7% of real photos in the test split are wrongly flagged**, more than we would like. The calibrated bands in Entry 11 address this.
6. **Weak spots by generator:** FLUX.2 klein under compression (JPEG 70: 26% caught) and Sora 2 (76%).

**Edited images (patch level, held-out images from DiffSeg30k and tiny-audits; v3):**

| | Clean | Compressed / resized |
|---|---|---|
| Patches inside the edit caught | 89% | 80% |
| Patches from original (unedited) photos called real | 84% | 86% |
| AUC edited vs original patch | 0.942 | 0.904 |
| Unedited area inside an edited image flagged as AI | 21% | 21% |

Reading: the model can tell AI-edited regions from real photos at the patch level, and the area outside the edit mostly looks real (79%), which supports localising region edits with a finer heatmap. **Important limits:** these are held-out *images*, but they come from the same datasets and editors as the training edits, and they are not ChatGPT or Gemini edits. We still need the team's own edited photos for that test.

### Entry 11 — 14 Oct · Calibration and verdict bands chosen on validation data
- **Method** (`prototype/calibrate_v3.py`): fit Platt scaling `p = sigmoid(1.513 × logit + 0.722)` on the validation split so the score behaves like a probability. Then pick the bands from the validation data: *Likely AI-generated* when at most 3% of real validation images would be called AI; *Likely Authentic* when at most 10% of AI validation images would be called real.
- **Result:** Trust Score **≥ 54 → Likely Authentic**, **≤ 17 → Likely AI-generated**, in between → **Needs Human Review**.
- **Checked on the held-out test split** (bands fixed beforehand):

| Images | Called Authentic | Needs Review | Called AI |
|---|---|---|---|
| Real photos, clean | 85.5% | 11.7% | 2.8% |
| Real photos, JPEG 70 | 92.7% | 6.2% | 1.1% |
| AI images, clean | 2.5% | 10.4% | 87.2% |
| AI images, half size + JPEG 75 | 6.9% | 15.8% | 77.3% |
| AI images, JPEG 70 | 18.9% | 28.7% | 52.3% |
| Reddit real photos | 84.1% | 12.9% | 3.0% |
| Reddit AI images | 11.5% | 32.8% | 55.7% |

  So a real photo is wrongly called AI only about 1–4% of the time, and a clean AI image is wrongly called authentic only 2.5% of the time. For compressed AI images the model more often says "needs review" than makes a wrong call, although 7–19% are still called authentic.

### Entry 12 — 14 Oct · v3 added to the app
- `prototype/detector.py`: new heads `ours_v3` (default), `ours_v3_proof`, `ours_v3_noaug`; heads can carry calibration and their own verdict bands.
- `prototype/trust.py`: `verdict()` accepts per-model bands. `prototype/app.py`: v3 is the default model and the "How it works" page describes it.
- Quick check on the demo images (model v3): ChatGPT images Trust 0–25, Gemini images 0–8, real photos 34 / 78 / 97 / 98. One real demo photo (`real_01`) lands in Needs Human Review at Trust 34.
- **Not yet done:** finer heatmap and region-based verdict, update of the Cross-Generator Test page with v3 numbers, tests on the team's own ChatGPT/Gemini edited photos.

### Entry 13 — 15 Oct · App check in a real browser, v3 results page, deployment
- **Browser test (automated Chromium, then the three pages):** uploaded a ChatGPT image → Trust Score 1, *Likely AI-generated*, with the C2PA "made by OpenAI" marker shown; uploaded a real photo → Trust Score 97, *Likely Authentic*. The Cross-Generator Test and How-it-works pages opened without errors.
- **Fixes found while testing:** (1) the sidebar text for v3 repeated the model name as its description; model names and descriptions are now separate. (2) The Cross-Generator Test page only showed the old v1/v2 experiment; it now starts with a **v3 section** (held-out test table, ChatGPT / Gemini / other generators, verdict bands, honest limitations) and keeps the old experiment below it.
- **Deployment (15 Oct):** the hosted app (Streamlit Community Cloud, https://deepfake-detector-perceptrons.streamlit.app) updates from the GitHub `main` branch. v3 weights are tiny (a few KB each) and live in `prototype/weights/`; the large Kaggle outputs and features (over 500 MB) are excluded from Git through `.gitignore`. Commits: `ec5db71` (v3), `fa97fde` (cache key), `9f97f5f` (rebuild trigger).
- **Problem found on the live site:** after the first push the hosted app showed the new Cross-Generator page but still listed only the old v1/v2 models, because the running server kept the old model code and a cached detector in memory. Fix: the detector cache key now includes a models version string, and a change to `requirements.txt` made Streamlit Cloud rebuild and restart the app.
- **Live check (automated browser against the hosted site):** default model shows "Our model v3 (strongest)"; ChatGPT test image → Trust Score 1 (*Likely AI-generated*), Gemini test image → 0 (*Likely AI-generated*), real photo → 97 (*Likely Authentic*); each analysis takes about 10 seconds. *Lesson for later pushes:* if new weights do not appear online, change the models version string in `app.py`.

### Entry 14 — 15 Oct · Assessment and v4 plan (discussion with the team lead, nothing built yet)
- **Assessment:** v3 is strong on clean and held-out generators (AUC 0.98, ChatGPT 89%, Gemini 94%), but compressed images and ChatGPT/Gemini edits of real photos remain the open problems; the next real test is the team's own photos.
- **Clarified for the viva:** the model is a frozen pretrained CLIP encoder plus our own trained classifier head (transfer learning), not a network trained from scratch; datasets are existing public ones; the same CLIP is used for v1, v2, v3 and only the head differs. See `docs/Work_Summary_Oct_2026.md` Section 9.
- **Why only about 30,000 training images:** v3 was a first pass (laptop plus one Kaggle run) and used about 1% of OpenFake's 2.3 million training images. With a linear head, more data eventually stops helping, so the plan tests this first (train on 25 / 50 / 100% of the data).
- **v4 plan (to start when the team lead says go):** CLIP middle-layer features; more data; phone-style real photos and harsher degradations (WhatsApp-style double compression, screenshots); a stronger classifier (small network or light fine-tuning, using long Kaggle GPU runs); finer heatmap; the missing roadmap items. Details: Work Summary Section 11.
- **Waiting for:** the team's images (Task A and the extra phone-photo task), the Review 4 document and the teacher's requested changes.

### Entry 15 — 25 Oct · Research paper for Review 4, and Review 3 cleanup
- **Research paper** (`paper/`): a five-page, two-column paper on the implemented image module, in two versions that differ only in the author block, acknowledgement and code link: `Image_Research_Paper_With_Authors.pdf` and `Image_Research_Paper_Without_Authors.pdf` (renamed on 25 Oct; see Entry 16). It reports the v3 method and results; video and document modules are described only as planned work, because they are not implemented.
- **Every number comes from the experiment files** (`prototype/results/v3_results.json`, `v3_calibration.json`, `paper/analysis.json`); the figures and PDFs are rebuilt by `paper/make_figures.py` and `paper/build_paper.py`. Tables show held-out results with 95% bootstrap confidence intervals (`paper/analysis.py`, 1,000 resamples of the test images) and the calibration error (ECE: 0.106 before and 0.034 after Platt scaling on the test split).
- **References were checked** before use: the Yermakov et al. paper is cited as the arXiv preprint (2508.06248) because no conference acceptance could be confirmed; OpenFake (2509.09495) and DiffSeg30k (2511.19111) were confirmed with their authors. The WACV 2026 DOI in the earlier draft paper could not be verified and is not used.
- **Originality check:** the new text shares 0 of 2,490 five-word sequences with the earlier draft paper (`paper/overlap_check.py`). This is our own check, not a replacement for the institute's plagiarism software.
- **Cleanup:** the Review 3 documents (`docs/Review3_QA.md`, `docs/Quick_Answers.md`, the Review 3 slides) were removed from the working folder; they remain in the Git history. The roadmap and README no longer list them.

### Entry 16 — 25 Oct · Full-project research paper (image + video + document)
- **Decision:** the Review 4 paper should cover the whole project, not only the image part. The image-only papers were kept and renamed `Image_Research_Paper_With_Authors.pdf` / `Image_Research_Paper_Without_Authors.pdf`; the new full papers are `Full_Research_Paper_With_Authors.pdf` / `Full_Research_Paper_Without_Authors.pdf` (6 pages, two columns, 4 figures, 4 tables, 42 numbered references).
- **What is in it:** the unified architecture (upload, routing, three analysers, calibration and fusion, Trust Score with a review band); the **implemented and evaluated** image analyser with all results; the **designed** video analyser (frame sampling, MTCNN face crops, frame scorer, top-fraction aggregation, timeline, split by video, Celeb-DF transfer test) and document analyser (OCR, ELA, SRM noise and DCT views, dual-stream U-Net with BCE + Dice, synthetic forgery generator, real-forgery test, arithmetic and PDF checks); the evaluation protocol for all three modules.
- **Honesty rule:** results are reported only for the image analyser. Video and document modules are labelled as designs with planned evaluation, so the paper does not claim anything we have not measured; it will be updated when those modules are built.
- **Citations on every page:** citations are numbered automatically in order of first use; 14 additional references (video, document, calibration, GAN and diffusion background) were added and checked; ForensicHub (NeurIPS 2025, arXiv:2505.11003) was verified and is cited as the existing unified benchmark, so the paper does not claim to be the first unified system.
- **Originality check** (`paper/overlap_check.py`, `paper/Plagiarism_Report.md`): 0.4% shared five-word sequences and no shared eight-word sequence with the earlier draft papers; below 1% with our project documents; 19.5% with the image-only paper (our own text).

### Entry 17 — 25 Oct · Image paper completed to the full image scope; citations on every page; robustness run
- **Citation fix:** the first image-only paper had no citations on page 3 (Sections V and VI) and the full paper only 4 and 1 on pages 4 and 5. Both papers now cite real prior work throughout the setup, results and discussion (ROC-AUC, bootstrap, the GenImage protocol, findings that agree with ours); the full paper has 44 references, the image paper 27.
- **Kaggle robustness run** (`kaggle/robust/robust.py`, 39 minutes, same 3,392 test images as before): the models were scored under five more conditions — JPEG 90, JPEG 50, Gaussian blur, half size, and a screenshot-like re-save. Results for v3 (AUC / share caught / real photos labelled real): clean 0.982 / 94% / 92%; JPEG 90 0.977 / 91% / 94%; JPEG 70 0.949 / 66% / 97%; **JPEG 50 0.916 / 38% / 99%**; screenshot-like 0.965 / 88% / 92%; half size 0.960 / 96% / **72%**; blur 0.866 / 91% / **58%**. The earlier v2 head falls to 1% caught at JPEG 50 and to an AUC of 0.552 under blur.
- **Important finding:** under blur and half size the high share caught comes together with many false alarms on real photographs (58% and 72% labelled real), so the share caught alone would be misleading; the paper says so and states that the verdict bands were not tuned on these two conditions.
- **Other new measurements in the paper:** reliability diagram of the calibration (`paper/make_calibration_plot.py`), a data card of the evaluation splits, and the prototype-scale ablations of the format fix and the frequency features (AUC 0.974 → 0.883 on GenImage with the format fix, unchanged on modern generators; frequency features no gain).
- **Final files** in `paper/`: `Image_Research_Paper_With_Authors.pdf` and `Image_Research_Paper_Without_Authors.pdf` (5 pages; scope = complete image module; unfinished items — EfficientNet baseline, finer heatmap, phone-photo and ChatGPT/Gemini-edit tests — are listed as "in progress" in Table VII and have no results), plus the two `Full_Research_Paper_*` files. The first image-only version is kept in `paper/archive/`.
- **Rule followed:** no dummy or estimated numbers appear in any paper; unfinished experiments are marked as unfinished. A temporary layout test with stand-in numbers was built only in a scratch folder and discarded.

### Entry 18 — 25 Oct · Guide's instruction, 4-page submission paper, and the real use case
- **Guide's instruction:** the paper may be at most 4 pages, covers the whole project, and should concentrate on the process (requirements, system design, algorithms, workflow) with little data. Built `Final_Paper_4pages_With_Authors.pdf` / `Final_Paper_4pages_Without_Authors.pdf` (`paper/build_short_paper.py`): 4 pages, 32 verified references, citations on every page; sections: requirements (SRS-based table), layered architecture, screening algorithm, the three analysers, data and training workflow, evaluation protocol, current status, limitations. Results shown: clean test images only (AUC 0.982, ChatGPT 89%, Gemini 94%, proof model 84% / 98%). The longer 6-page and image-only papers are kept.
- **Priority set by the team lead:** the demonstration case is clean images and **real photographs edited by ChatGPT or Gemini**; heavy compression and WhatsApp-style recompression are postponed. The two Kaggle runs started earlier for resize/blur robustness (`kaggle/hard_train`, `kaggle/hard_eval`) are therefore optional.
- **Key finding for that case:** the public dataset `UCSC-VLAA/GPT-Image-Edit-1.5M` (CC-BY-4.0, arXiv:2507.21033) contains about 1.5 million editing examples whose output images were regenerated with GPT-Image-1, including background and environment changes, object swaps, additions and removals (OmniEdit, UltraEdit and HQ-Edit sources). It is stored as very large compressed archives (about 4.5 TB), with input images before output images inside each archive. A CPU probe (`kaggle/editprobe`) was started to inspect real example pairs before any training, because some sources (HQ-Edit) have AI-generated inputs and must not be used as "real".
- **Originality check of the 4-page paper:** 0.1% five-word overlap with the earlier draft and no eight-word overlap with any earlier document (see `paper/Plagiarism_Report.md`).

### Entry 19 — 25 Oct · 4-page paper finalised (figures added, page 4 used)
- **Guide's message re-read:** either discuss image detection in detail (any new algorithm) or add the entire process; "entire process" = the whole project, so the 4-page whole-project paper is the one to submit. The 6-page and image-only papers are backups and are not submitted.
- **Changes:** added the image-analyser pipeline diagram (Fig. 2) in the image section, and a confusion matrix and ROC curve drawn from **our own** test results (Fig. 4: 1,110 of 1,200 real photographs accepted, 2,050 of 2,192 AI images caught, accuracy 93.2%, AUC 0.982; `paper/make_cm_roc.py`). Figures were renumbered (1 architecture, 2 image pipeline, 3 video and document pipelines, 4 confusion matrix and ROC); text now refers to them; the text reflows so that page 4 is filled and the references use both columns. Still exactly 4 pages, 32 references, citations on every page.
- **Images from a teammate** (a confusion matrix with 98% accuracy on about 8,000 images and an ROC curve with AUC 0.88) were **not** used: their model, data and test split are unknown, they do not agree with each other, and they do not match our measured results.
- Originality check repeated: see `paper/Plagiarism_Report.md`.
- **Image-only paper** (`Image_Research_Paper_With_Authors.pdf` / `Image_Research_Paper_Without_Authors.pdf`) re-styled to exactly 4 pages (compact type, same content: 27 references, citations on every page) so it can be shown if the examiner asks about the image module alone.
- **One-page explainer** of the 4-page paper for the viva: `docs/Paper_Explained_One_Page.pdf` (section summary, numbers to remember, key terms, likely questions).

### Entry 20 — 25 Oct · "With / Without Authors" clarified; references moved to one column
- **Clarification from the team lead:** "authors" means the authors of the **cited works**. So `..._With_Authors.pdf` has citation numbers and the reference list, and `..._Without_Authors.pdf` has **no citation numbers and no reference list**, while *our* names, the college heading and the acknowledgment stay in both. (The earlier anonymous versions, which had removed our names, were a misunderstanding; they were moved to `paper/archive/` together with the older 6-page papers.)
- **Layout:** in all four 4-page PDFs (whole project and image-only, with and without references) the text and figures now fill the left side and the whole reference list sits in the **right column of page 4** (32 references for the whole-project paper, 27 for the image paper). For this the confusion matrix and ROC curve became one column-width figure (confusion matrix above ROC), and in the image paper the robustness chart moved to page 3 and the verdict-outcome chart (Fig. 5, redundant with the text) was removed.
- **Builders:** `paper/build_short_paper.py` (whole project) and `paper/build_image_paper.py` (image); the layout engine (`paper/layout.py`) can now reserve the right column of the last page for the references and can strip citation markers.
- **Checks:** both versions have exactly 4 pages; the without-references versions contain no [n] marker and no broken sentence; the only 8-word overlap with the earlier draft is our own author and college line (see `paper/Plagiarism_Report.md`).

### Entry 21 — 25 Oct · v3b (resize/blur training): measured, not adopted
- Two optional Kaggle runs finished (`kaggle/hard_train`: 12,000 new OpenFake training images with strong resize-only, blur-only and mixed degradations; `kaggle/hard_eval`: validation images under the same degradations and the test images under three new conditions: scale 0.35, blur 2.5, scale 0.6 + blur + JPEG 60). `prototype/train_v3b.py` trained v3b on the v3 data plus the 69,630 new patches (C = 10, bands Trust >= 60 authentic, <= 16 manipulated).
- **Result:** v3b reduces false alarms on blurred real photographs (real photos labelled real under blur 58% -> 80%, under blur 2.5: 45% -> 81%) and improves the AUC under the three new conditions (scale 0.35: 0.915 -> 0.933; blur 2.5: 0.806 -> 0.862; mix: 0.953 -> 0.961). Halving or shrinking high-resolution camera photos is still a problem (DOCCI photos labelled real after halving: 42% -> 54%).
- **Cost:** clean images get slightly worse: AUC 0.982 -> 0.976, AI caught 94% -> 92%, GPT-Image-1.5 89% -> 86%, Nano Banana Pro 94% -> 89%.
- **Decision:** v3b is **not** used. The team lead's priority is clean images and ChatGPT/Gemini images (and their edits of real photos), where v3 is better; the app and all papers keep v3. v3b stays available (`prototype/weights/our_head_v3b.*`, `prototype/results/v3b_results.json`) as an option if blurred or resized real photographs must be handled later.

### Entry 22 — 25 Oct · Project Review-IV (design): 17 answers, diagrams, slides
- **Source of the questions:** the PDF with the 17 Review-IV design questions (the photo sent with it was the Review-III sheet; its remark about hardware for about one lakh images is answered in Q3).
- **Report:** `docs/Review4_Design_Answers.pdf` (16 pages; `.md` next to it) answers all 17 questions for our own project, with the same tone as the example (answer, table where useful, figure). Every class, function and file name was checked against `prototype/` (for example `Detector.analyze`, `DetectionResult`, `load_image`, `trust_score`, `verdict`, `explain`). Parts that exist only as designs (video analyser, document analyser, report builder, the abstract `BaseAnalyzer`, the download button) are labelled as designed.
- **Diagrams** (`docs/diagrams/*.png`, drawn by `docs/diagrams/make_review4_diagrams.py`): architecture, use case, class, object, sequence, activity, state chart, DFD level 0, DFD level 1, control flow, transaction and transformation flow. The new ones (object, activity, state, control flow, transaction/transformation flow) were also added to `docs/System_Design.tex` (section "Additional diagrams for design review IV"); the existing architecture, use-case, DFD, sequence, class and deployment diagrams of that file were kept.
- **Presentation:** `docs/presentations/Review4_Design.pptx`, 18 slides (title + one slide per question), with the full answer and a speaking hint in the speaker notes of every slide. Content for the report and the slides comes from one file (`docs/tools/review4_content.py`) and is built by `docs/tools/build_review4.py`, so they cannot disagree.

### Entry 23 — 25 Oct · HOD corrections to the paper
- HOD's four points: (1) guide's name first among the authors; (2) AI plagiarism must be checked; (3) the content seems to be taken from AI; (4) result discussion is missing. Meeting at 10:00 next morning.
- **Done:** guide listed first in the author line of all four 4-page PDFs (the separate thank-you line was removed); the whole-project paper's section VIII is now "Results and Discussion" with a discussion of the three main findings and a short, clearly labelled preliminary test on the team's 25 ChatGPT/Gemini images; the image paper's section VI is "Results and Discussion". All four PDFs are still exactly 4 pages.
- **Not done on purpose:** no attempt was made to disguise AI assistance from detectors. The team should rewrite the text in their own words, be able to explain every sentence, and disclose AI assistance if the institute requires it (see the chat notes).

### Entry 24 — 25 Oct · Plainer paper layout after the HOD's remark on the diagrams
- The HOD's concern was that the diagrams look AI-made. In the 4-page whole-project paper: the colourful generated diagrams were replaced by plain black-and-white block diagrams (`paper/make_plain_figures2.py`); page 2 now shows the architecture from the system design document (user, presentation layer, router, image / video / document pipelines, evidence fusion, trust scoring engine, report builder, model store, offline training) with Algorithm 1 in a small box below it; the image-pipeline figure was removed; the video and document pipelines are drawn as normal solid boxes; the requirements are a short boxed list instead of a table; the results table is a small plain table. Figures were renumbered (1 architecture, 2 video and document, 3 confusion matrix and ROC).
- In the architecture figure the image pipeline now says "frozen CLIP encoder, trained head" instead of "CLIP + frequency features" as in the older system-design text, because the final model does not use frequency features (they gave no gain). `docs/System_Design.tex` still carries the older wording and should be aligned before it is submitted.
- Still 4 pages for both versions. The image-only paper and `docs/Review4_Design_Answers.pdf` / slides still use the earlier generated diagrams.

### Entry 25 — 25 Oct · Layout bug in the paper builder (lost Discussion text) fixed
- **Symptom:** page 4 of the whole-project paper had a large gap and the "Discussion" paragraphs were missing. **Cause:** when a figure or table left almost no room at the bottom of a column, the next text block was given an invalid space by the layout engine and was silently dropped. **Fix:** `paper/layout.py` now starts a new column when fewer than about two lines remain. After the fix every paragraph of both builders was checked against the PDF text (none missing), and all four PDFs were rebuilt (4 pages each).
- Page 2 of the whole-project paper now has the layered architecture (black and white; solid = implemented, dashed = designed) at the top, and Algorithm 1 sits again at the end of the System Design section, before Analysis Modules. The requirement list and Algorithm 1 are plain black text.

### Entry 26 — 25 Oct · Second, simplified Review-IV report and presentation
- Because the first diagrams looked machine-made and some were large, a **second version** was made; the first report (`docs/Review4_Design_Answers.pdf`) and slides (`docs/presentations/Review4_Design.pptx`) were left unchanged.
- **New files:** `docs/Review4_Design_Answers_2.pdf` (+ `.md`, 10 pages instead of 16) and `docs/presentations/Review4_Design_2.pptx` (18 slides, speaker notes included). The 11 diagrams are redrawn smaller and in black and white (grey fill = implemented, dashed = designed) in `docs/diagrams/v2/` by `docs/diagrams/make_review4_diagrams_v2.py`: class diagram 10 classes (ReportBuilder left out), sequence diagram 5 lifelines and 8 messages, state chart 5 states, 8 use cases, one merged activity flow, one-strip transaction/transformation flow.
- The answers were adjusted only where they counted diagram elements (Q5, Q6, Q10, Q12, Q14, Q16); everything else is identical. Built by `docs/tools/make_review4_v2.py` from `review4_content_v2.py`.

### Entry 27 — 26 Oct · Third Review-IV pair: simple diagrams, bigger text
- Same simple v2 diagrams and the same answers, but larger bullet and table text on the slides and larger figures in the report (`docs/Review4_Design_Answers_3.pdf`, 16 pages; `docs/presentations/Review4_Design_3.pptx`, 18 slides). Built by `docs/tools/make_review4_v3.py`. Versions 1 and 2 are unchanged.

### Entry 28 — 26 Oct · Fourth Review-IV pair: every answer starts with Yes and is explained in sentences
- `docs/Review4_Design_Answers_4.pdf` and `docs/presentations/Review4_Design_4.pptx`: all 17 answers begin with "Yes" (Q3 reads "Yes, in the design", because the video and document tests are not run yet); the slide points are now full explanatory sentences, with the Yes sentence in bold, instead of short data fragments. Same simple diagrams as version 2. Built by `docs/tools/make_review4_v4.py` from `review4_content_v4.py`. Versions 1–3 are unchanged. The slides were not render-checked because PowerPoint was already open on the laptop (automation blocked); the report was checked.

### Entry 29 — 7 Nov · Review-4 files removed; remaining image work started; region-verdict test; finer heatmap
- **Removed (7 Nov, at the team lead's request):** the Review-IV report (four versions), the four Review-IV slide decks, the diagram folder and their build scripts. They were never committed, so they are not in Git history; Entries 22 and 26-28 describe files that no longer exist. Kept: the research papers in `paper/` (4-page submission paper, image paper, full paper) and `docs/tools/md2pdf.py`.
- **Design documents aligned with the final model:** `docs/System_Design.tex` and `docs/Overview_and_Algorithms.tex` now say the frequency branch was tested, gave no gain and is not used; a new subsection "Status: what the final image model changed" lists the differences (patch-wise scoring, OpenFake training data with compression and edit patches, calibration and bands, like-for-like baseline). Script: `docs/tools/align_image_docs.py`. The PDFs of the two .tex files were not rebuilt (no LaTeX on this laptop).
- **Region-based verdict, tested honestly** (`prototype/region_eval.py`, `prototype/region_rule_eval.py`, results in `prototype/results/region_eval.json` and `region_rule.json`). Kaggle job `kaggle/editgrid` extracted the app layout (centre + 3x3 grid) for 764 edited images that were held out of v3 training (DiffSeg30k and tiny-audits; 323 of them with their original photo), in two variants (clean, JPEG 70). Image score options (mean, max, top-2, top-3, mean+max) were compared with the threshold chosen on validation real photos only (3% false alarms).
  - Whole images (held-out test): "mean+max" gives AUC 0.983 (mean: 0.982) and catches 86.8% of clean AI images at 2.8% false alarms (mean: 83.1% at 2.1%); on Reddit 54.1% vs 47.9%.
  - Held-out edited images (clean): mean 48.7% called AI, mean+max 53.7%; originals called AI 2.2% vs 3.4%; edits covering at most 30% of the grid 18% vs 25%.
  - As a "second look" that can only move Authentic to Needs Human Review (3% of authentic validation photos flagged): clean real photos sent to review 2.4%, held-out edited images 4.6% (JPEG 70: 5.5%). The gain over the cost is under 2:1, so it **was not added to the app**. The verdict rule stays the mean of the ten patches.
  - **Caveat:** these edits are diffusion inpainting from research datasets, not ChatGPT/Gemini edits of real photos; the result says region scoring is a weak lever for this model, not that Gemini edits are impossible. The decisive data is still the team's own edited photos with their originals.
- **Rahul's images:** the WhatsApp files in Downloads (a custard-apple photo and a royal portrait) cannot be matched to the labels in his report (the custard image scores Trust 24 here, which matches none of his Gemini scores), so they were not used as labelled tests. Original files and labels are needed.
- **Finer heatmap (done):** `Detector.fine_grid()` scores a 6x6 grid for display only (about 40 s on a laptop CPU); the app's Heatmap tab has the option "Finer heatmap (6 x 6 patches)". The verdict still uses the ten-patch rule. Smoke-tested with Streamlit's AppTest on a sample image, no exceptions. Models version string bumped so the hosted app reloads.
- **EfficientNet baseline (SRS 4.2):** Kaggle job `kaggle/effnet` trains EfficientNet-B0 on the same 30,420 OpenFake training images and scores the same 3,392 test images in three variants. Result to be added below when it finishes.

### Entry 30 — 7 Nov · EfficientNet-B0 baseline measured (SRS 4.2)
- **Run:** Kaggle job `kaggle/effnet/effnet.py` (34 minutes, T4). ImageNet-pretrained EfficientNet-B0 (timm), whole image resized to 224 x 224, random flip, AdamW lr 2e-4, 3 epochs, one-cycle schedule, mixed precision. Recipe fixed beforehand, nothing tuned on test images. Trained on the same 30,420 OpenFake training images as v3 (all loaded, 52.6% fake) and scored on the same 3,392 held-out test images (generators never seen in training) in the same three variants. Comparison script `prototype/compare_effnet.py`; results `prototype/results/effnet_comparison.json`; predictions `kaggle/effnet/out/preds.csv`.
- **Result (image level, test images):**

| Variant | EfficientNet-B0: AUC / AI caught / real OK | Our v3: AUC / AI caught / real OK |
|---|---|---|
| Clean | 0.831 / 65.1% / 85.6% | **0.982** / 93.5% / 92.5% |
| JPEG 70 | 0.830 / 59.9% / 89.2% | **0.949** / 66.0% / 97.2% |
| Half size + JPEG 75 | 0.869 / 54.5% / 94.7% | **0.950** / 86.0% / 89.5% |

  Per generator (clean AUC, EfficientNet vs ours): FLUX.2 klein 0.703 vs 0.982; GPT-Image-1.5 0.903 vs 0.972; Midjourney v7 0.834 vs 0.985; Veo 3 0.805 vs 0.981; Z-Image Turbo 0.825 vs 0.984; Recraft v3 0.579 vs 0.988; Nano Banana Pro 0.856 vs 0.985. Average over the seven largest held-out generators: **0.841 vs 0.984**.
- **SRS 4.2 image target met:** seen AUC >= 0.95 and unseen-generator AUC >= 0.80 and above the EfficientNet baseline (0.984 vs 0.841 average over the named held-out generators).
- **Honest reading:** the baseline's training loss fell to 0.109 while its test AUC stayed at 0.83, the usual picture of a fully fine-tuned CNN fitting its training generators. The two models differ in more than the encoder (ours scores native-resolution patches; the baseline sees the whole image resized, the standard way), so the table compares two complete recipes, not CLIP against EfficientNet alone. The original roadmap baseline (trained on Stable Diffusion v1.4 only) was not run; this baseline uses the same modern training data as v3, which is the fairer test.

### Entry 31 — 10 Nov · Team photos: real phone photos and their ChatGPT / Gemini versions

- **Data (from Bhoomi's friend, Drive folder "Project Images"):** 10 real phone photos (`real`), and the same 10 scenes re-made in ChatGPT (`chatgpt`) and Gemini (`gemini`), paired by number 01-10. Everything is a JPEG of about 1 megapixel and 60-350 KB with no camera metadata; the sender confirmed the files were sent through WhatsApp before the Drive link was made, so they are WhatsApp-compressed copies, not camera originals (a phone photo is normally 2-6 MB). Labels come from the folder names.
- **Result with v3 (Trust bands: Authentic >= 54, AI <= 17):**

| Folder | Authentic | Needs Human Review | AI |
|---|---|---|---|
| real (10) | 2 | 5 | 3 |
| ChatGPT (10) | 1 | 5 | 4 |
| Gemini (10) | 3 | 3 | 4 |

  AUC real vs ChatGPT 0.66, real vs Gemini 0.54, real vs both 0.60. The edit scored more AI-like than its own original in 7 of 10 ChatGPT pairs and 5 of 10 Gemini pairs.
- **Honest reading:** on these compressed phone-size files v3 does not separate real from ChatGPT/Gemini images, and 3 of 10 real photos were called AI. This is far below the held-out test (AUC 0.982). The WhatsApp compression is the likely cause (the robustness tests already showed compression is the weak point: JPEG 50 catches 38%), but it is not proven until the full-size originals are tested. Ten images per group is also a small sample.
- **Other checks done the same day:** 20 full-size PNG ChatGPT downloads (Rahul's zip): 17 called AI, 2 Review, 1 Authentic. One Gemini image (with the visible Gemini mark): Trust 84, Authentic. A zip of 20 "real" photos (mostly WhatsApp copies, labels unverified): 12 Authentic, 5 Review, 3 AI, so not used as a result.
- **Known limitation, stated in the project:** v3 works best on full-size, lightly compressed images. Small or heavily compressed photos, and photos re-made by Gemini or ChatGPT, are not reliable and should go to human review. No retraining was done. Scores are in `prototype/results/team_photos_check.json`.

---

### Entry 32 — 10 Nov · Team photos re-tested with full-size originals

The friend re-sent the files without WhatsApp: 10 real phone photos (about 12 MP, EXIF present, JPEG), 10 ChatGPT edits (about 1 MP PNG) and 10 Gemini edits (about 4 MP PNG). Same v3 analyser, same scoring as Entry 31. Per-image scores: `prototype/results/team_photos_fullsize_check.json`.

| Set | Authentic | Needs Human Review | AI |
|---|---|---|---|
| Real photos | 9 | 0 | 1 |
| ChatGPT edits | 0 | 2 | 8 |
| Gemini edits | 2 | 7 | 1 |

- AUC, real vs ChatGPT: 0.95. Real vs Gemini: 0.83. Real vs both: 0.89 (10 + 20 images, so only a rough guide).
- In all 10 pairs, for both ChatGPT and Gemini, the edit scored more AI-like than its own original.
- One real photo (02) was called AI (Trust 13): a false alarm. Two Gemini edits (05, 09) were called Authentic: misses.
- Compared with Entry 31 (WhatsApp copies, AUC 0.66 / 0.54), full-size files work much better. This supports compression as the cause of the earlier weak result, but the two sets also differ in format and size, so it is not a controlled proof.
- Limitation to state: ChatGPT edits are caught well; Gemini edits are mostly sent to human review rather than flagged AI; heavily compressed copies (for example WhatsApp) are unreliable. No retraining was done.

### Entry 33 — 12 Nov · Edge-case notices in the app (drawings, small images, messaging-app files)

The HOD tried a hand-made Paint drawing: Trust 45, Needs Human Review. Drawings are outside what the detector was built for, so the app now says so. New file `prototype/imagetype.py`; the AI probability is never changed.

- **Drawing / graphic / screenshot:** at least 85% flat 8x8 blocks and at most 4000 distinct colours (on a 256 px copy). Shows a notice and sets the verdict to Needs Human Review.
- **Small image:** shorter side under 448 px. Notice only.
- **Messaging-app file:** JPEG, no EXIF, long side 1200-1300 px. Notice only (heuristic, worded "may have been").
- **Checked for false alarms:** none of the 30 full-size team photos (real, ChatGPT, Gemini; flat share up to 0.78, at least 6849 colours) and none of the 44 sample test images were flagged as graphic. Two synthetic doodles (flat share 0.94 and 0.98) were flagged. The thresholds were set from these numbers, so this is a sanity check, not an accuracy measurement; real HOD / Bhumi drawings will be the real test.

### Entry 34 — 12 Nov · Gemini fine-tune experiment (rejected) and Rahul's phone-photo pairs

**Experiment.** Goal: improve Gemini-edit detection by adding matched (real photo, Gemini edit) pairs to the v3 head (frozen CLIP, logistic head only). Data: 150 pairs from Apple's Pico-Banana-400K (Nano-Banana edits of Open Images photos; CC BY-NC-ND 4.0, research use; originals fetched from the Open Images copy because Flickr blocked downloads). Script: `prototype/gemini_finetune.py`; accept rules were fixed before the run. Result: `prototype/results/gemini_pico_finetune.json`.

| Check | v3 | v3 + 150 pairs |
|---|---|---|
| Held-out Pico pairs (5-fold by pair), AUC | 0.748 | 0.941 |
| OpenFake clean AUC | 0.9821 | 0.9902 |
| Team test (10 real + 10 ChatGPT + 10 Gemini), AUC | 0.890 | 0.875 |
| Real photos called AI (team test) | 1 | 1 |

Decision: **REJECT, keep v3** (rule "AUC on the untouched team test must not drop" failed; the drop is about one or two images out of 30, so it is within noise, but the rule was set in advance and is not relaxed). Reading: the gain is large on Pico-style pairs but did not carry over to phone photos, likely because the Pico real images are web photos. No weights were changed.

**Rahul's pairs (matched by eye).** 22 Gemini images paired with their source photos. 12 pairs have a full-size camera original (EXIF, 3072x4096 or 4000x3000) on the real side; the other 10 have WhatsApp/Snapchat copies, screenshots or an AI image as the "real" side and are not used for the accuracy figure. Per-image scores: `prototype/results/rahul_pairs_check.json`.

- 12 clean pairs, v3: AUC real vs Gemini **0.986**; the edit scored more AI-like than its original in 12 of 12 pairs.
- Real photos: 11 Authentic, 1 Needs Human Review, 0 called AI.
- Gemini edits: 6 AI, 5 Needs Human Review, 1 Authentic.
- Together with the team test: 22 Gemini edits of full-size phone photos vs their 22 originals, AUC **0.913**. Gemini is therefore detected better than the first team test (AUC 0.83 alone) suggested, but not as well as ChatGPT (0.95); about half of the edits still land in Needs Human Review.
- The 10 other pairs are shown for information only: for example one real-side WhatsApp copy (960x1280) was called AI, and screenshots/posters fall in the new "not a camera photo" notice.

Limits: small samples (12 and 10 pairs), single-person photo sets, one Gemini version. The numbers are indicative, not exact.

## 2. Roadmap items ticked so far

See `PROJECT_ROADMAP.md`. Items are ticked only when the work exists and was checked. Items completed in earlier work by the prototype (setup, first data, CLIP model, cross-generator test) were ticked on 14 Oct with a pointer to Walkthrough 01.

---

## 3. What is next

1. Team ChatGPT / Gemini photos tested on 10 Nov (Entry 31): weak on compressed phone-size files. Re-test with full-size originals if they can be sent as documents.
2. Finer heatmap (for example 6 × 6) and a verdict based on the most suspicious region, for partly edited images.
3. Update the Cross-Generator Test page with the v3 numbers.
4. FastAPI wrapper so the separate frontend can call the model.
5. Decide the hosting route; update slides and the Team Briefing.
