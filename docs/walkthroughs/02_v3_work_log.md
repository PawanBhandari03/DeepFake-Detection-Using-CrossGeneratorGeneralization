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
- **Deployment:** the hosted app (Streamlit Community Cloud) updates from the GitHub `main` branch. v3 weights are small (about 6 KB each) and live in `prototype/weights/`; the large Kaggle outputs and features (over 500 MB) are excluded from Git through `.gitignore`.

---

## 2. Roadmap items ticked so far

See `PROJECT_ROADMAP.md`. Items are ticked only when the work exists and was checked. Items completed in earlier work by the prototype (setup, first data, CLIP model, cross-generator test) were ticked on 14 Oct with a pointer to Walkthrough 01.

---

## 3. What is next

1. Test v3 on the team's own ChatGPT / Gemini photos, including partly edited ones (waiting for their images).
2. Finer heatmap (for example 6 × 6) and a verdict based on the most suspicious region, for partly edited images.
3. Update the Cross-Generator Test page with the v3 numbers.
4. FastAPI wrapper so the separate frontend can call the model.
5. Decide the hosting route; update slides and the Team Briefing.
