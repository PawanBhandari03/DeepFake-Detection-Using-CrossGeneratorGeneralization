# Team Briefing — Where We Are and What Happens Next (v3 Plan)

> **For:** The Perceptrons — Pawan, Rahul, Bhoomi · **Written:** 14 Oct 2026 · **Next review:** about 1–2 weeks from now
>
> Read this once, top to bottom. It replaces the chat history Pawan had with Claude (that history was lost when the laptop was cleaned). Everything decided so far is here, plus what each of us does next.

---

## 1. The project in 60 seconds

A web app where a user uploads an **image** and gets a **Trust Score (0–100)**, a **verdict** (*Likely Authentic* / *Needs Human Review* / *Likely AI-generated*), a **heatmap** of the most suspicious regions, and a short **explanation**.

Our research idea is **cross-generator generalization**: the model should catch fakes from AI tools it **never saw during training**. We do not just memorise one dataset. Full details are in `PROJECT_ROADMAP.md` and `docs/System_Design.tex`.

Three phases: **Image** (now) → Video → Document. We are working on Phase 1 only. The final system is a deployed full-stack application for the external examiners.

---

## 2. What exists today (the prototype)

A working Streamlit app in `prototype/`. Run it with:

```
.venv\Scripts\streamlit run prototype\app.py
```

How it works, in one paragraph: the uploaded image is converted to plain RGB pixels (file format and metadata do **not** affect the score), cut into 224 × 224 patches **without resizing**, each patch is re-saved as JPEG quality 95 (so the model cannot cheat by learning "PNG = fake"), and each patch goes through **CLIP ViT-L/14** — a big pretrained model from OpenAI that we keep **frozen**. CLIP turns each patch into 768 numbers. A tiny classifier (logistic regression, about 4 KB) that **we trained** turns those numbers into "probability this patch is AI". The patch results are averaged into the Trust Score and shown as a 3 × 3 heatmap. Metadata (camera EXIF, C2PA "made by OpenAI" marker) is shown on a separate tab and is never fed to the model.

Full write-up: `docs/walkthroughs/01_image_prototype.md`.

### The three model versions

| Version | Trained on | Why it exists |
|---|---|---|
| **v1** | Real photos + one old generator (Stable Diffusion 1.5), 1,600 images | Shows that training on one generator still catches others |
| **v2** | About 60 generators, but **no OpenAI and no Google images** | Shows ChatGPT / Gemini images are caught even though the model never saw them |
| **v3 (to build)** | Many more generators **including** ChatGPT and Gemini, compressed copies, varied real photos, edited images | The strong model for the demo and the examiners |

**v1 and v2 stay unchanged.** They are our *evidence* for the cross-generator claim. v3 is the *product*.

---

## 3. The problem we must fix

Pawan showed the prototype to the teacher. Her feedback: it is trained on older generators and does not work well on new images from ChatGPT, Gemini and similar tools. She will test with **real photos, including faces, clothes and backgrounds**, and with images **partly edited** by AI tools.

Measured weaknesses (from the walkthrough):

| Weakness | Evidence |
|---|---|
| ChatGPT images only partly caught | v2: AUC 0.84, only 68% caught at the default threshold |
| **JPEG-compressed images fail** | On 12 ChatGPT demo images, v2 caught 6 of 6 PNGs but only **1 of 6 JPEG copies** |
| Gemini test set far too small | Only 4 Nano Banana Pro test images |
| Training data too small and too old | About 6,000 images in total; real photos are mostly old, small ImageNet pictures |
| Partly edited images not handled | Only a coarse 3 × 3 heatmap; the average hides a small edited region |

---

## 4. The plan: v3 and the "proof model"

### 4.1 Two models, two jobs

- **Demo model (v3):** trained on everything we can get. Best accuracy. This is what the examiners use.
- **Proof model:** the same recipe, but **one generator family is held out** (for example all Google images) during training, then tested on exactly that family. This gives the honest cross-generator number. We report both.

If an examiner asks "does it only work because it saw ChatGPT?", we point to v1, v2 and the proof model.

### 4.2 What changes in v3

1. **Much more data**, from the sources in Section 5.
2. **JPEG augmentation.** While training, we randomly compress, resize and blur copies of each image (as WhatsApp, Instagram and screenshots do). Compression erases the faint traces we detect, so the model must learn to find them in damaged images. This is the biggest expected gain on real-world images.
3. **More varied real photos** (phones, cameras, social media), not only ImageNet.
4. **Finer heatmap** (for example 6 × 6 or 8 × 8 patches) and a verdict based on the **most suspicious region**, not the average. This is how we handle partly edited images.
5. **Thresholds chosen from validation data** and probabilities calibrated (roadmap Week 10), not guessed numbers.

### 4.3 Whole-image regeneration vs. region edits

| Case | What happens | Our expectation |
|---|---|---|
| **A. The tool redraws the whole image** (ChatGPT and Gemini editing usually do this, even for parts that look untouched) | AI traces spread across the whole picture | Should already work; v3 should be stronger. We will test it directly with the team's edited photos |
| **B. Only a region is changed** (inpainting, generative fill) | One part AI, the rest real camera data | Needs the finer grid plus training on masked-edit data. Possible, but harder |

### 4.4 Honest expectations (do not over-promise to the examiners)

| Situation | Realistic expectation |
|---|---|
| Fully generated image, clean file | Likely above 85% after v3 |
| Fully generated, JPEG-compressed | About 75–90% once JPEG augmentation is in |
| Region edits (face, clothes, background) | Roughly 60–80%, depending on edit size |
| Small edit **and** heavy compression | Hardest case in the whole research field; we report it honestly |

**Our target:** 80–85% on a test set that looks like the teacher's (real photos + ChatGPT/Gemini images, including edited ones). The most important single number is the **real-photo false-positive rate**: calling a teacher's own photo "fake" would fail the demo, so we tune to keep it very low and send unsure cases to *Needs Human Review*.

Note: garbled letters on a sign are a clue a *human* sees by reading. The model sees only pixel statistics, so we cannot promise it uses that clue. A separate text-reading check is a possible extra later.

---

## 5. Data sources (what Claude found on 14 Oct)

All are on Hugging Face and **not gated** (no access request needed), unless noted. Licences must be respected; all of this is for academic use.

| Dataset | What it is | Size | Use |
|---|---|---|---|
| **OpenFake** (`ComplexDataLab/OpenFake`) | Real photos + about 60 modern generators (GPT-Image, Nano Banana, Imagen, Veo, Midjourney v7, FLUX…). CC-BY-NC-4.0 | Huge (many TB); we download **only slices** | Main source of ChatGPT / Gemini / modern fakes |
| **DiffSeg30k** (`Chaos2629/Diffseg30k`) | About 30k images edited by diffusion models, **with pixel masks** of the edited regions; images may have 1–3 edits; 8 editing models. Apache-2.0 | About 26 GB (train 21.5 GB, validation 4.9 GB) | Region-edit training (Case B) |
| **tiny-audits** (`DivyaApp/tiny-audits`) | COCO and NEWS images, each with `original`, `manipulated` and `mask` folders, for about 10 inpainting methods (GLIDE, SDXL, PowerPaint, Firefly, …) | About 2.7 GB | Smaller, paired original + edit + mask set; licence not stated, so check before reuse beyond coursework |
| **GenImage** (already used) | Real ImageNet + 8 older generators | already downloaded | Older generators, v1/v2 comparison |
| **COCO-Inpaint** (paper) | About 238k inpainted training images with masks | Not yet found on Hugging Face under that exact name | Candidate only; verify access first |
| **PromptForge-350k, MLLM-Edit** (papers) | Prompt-based editor datasets (closest to ChatGPT / Gemini editing) | Unknown | Candidates only; we have **not** confirmed they are downloadable or which editors they include |

Our own team-collected images (Section 7) are the most important test set, because they match what the teacher will do.

---

## 6. How the work is actually done (so it is not a mystery)

Think of four stations.

1. **Get data** — download datasets (Kaggle notebook or laptop). Just collecting and organising.
2. **Turn images into numbers** — every image (patches) goes through CLIP, and the 768 numbers per patch are saved to a file (`features.npz`). This is the slow step: it needs a GPU, so it runs on **Kaggle**. Done once.
3. **Train the classifier** — a tiny model learns "real vs fake" from that table of numbers. Takes minutes, runs on a laptop. The result is a ~4 KB weights file in `prototype/weights/`.
4. **The app** — Streamlit loads CLIP + our weights, so an uploaded image goes image → CLIP → our weights → Trust Score. It is deployed on Streamlit Community Cloud and updates when we push to GitHub.

One work cycle: Claude writes the scripts and a Kaggle notebook → Pawan uploads it to Kaggle and presses Run → Pawan brings back the output file → Claude trains v3 and tests → the app gets the new weights → push to GitHub.

Only the Hugging Face token (read-only) is needed for downloads. It goes into Kaggle as a **secret**. **Never paste tokens or passwords into chat, GitHub or documents.**

---

## 7. Tasks for the team

Rule of thumb: Pawan + Claude do model training; the other two do the tasks below. Everything here is doable manually or with a free coding assistant.

### Task A — Collect test images (most urgent, ~2–3 hours)

**Ethics:** use only your own photos or photos of people who agree. Do not upload other people's faces to online AI tools without their consent.

1. **Real photos:** at least **100** original photos from your phones (faces, clothes, rooms, outdoors, signs/boards with text). Keep the **original files**, not WhatsApp copies.
2. **Edited by ChatGPT and Gemini:** take **50 of those photos for each tool**. Upload the photo and ask for an edit: change clothes, change the background, add or remove an object, change a face detail, change text on a board. Download the result.
3. **Fully generated images:** **50 per tool** from ChatGPT and Gemini (people, objects, scenes, indoor/outdoor). Optional extras: Meta AI, Grok, Copilot, Midjourney (any free tool).
4. For every image also save a **compressed copy**: a screenshot, or send it via WhatsApp and save the received file.
5. **Name files so pairs stay together.** Suggested layout:

```
team_images/
  real/            real_001.jpg, real_002.jpg ...
  edited/          pair_001_original.jpg
                   pair_001_chatgpt.png
                   pair_001_chatgpt_screenshot.jpg
                   pair_001_gemini.png ...
  generated/       gen_chatgpt_001.png, gen_gemini_001.png ...
  pairs.csv        columns: pair_id, tool, edit_type, region (face/clothes/background/object/text), notes
```

Do not rename, crop or resize the original downloads. Put everything in the shared Google Drive and tell Pawan.

### Task B — Frontend (intermediate; Antigravity or similar is fine)

Build the web page against a **fixed JSON contract**. Claude will provide a small API (FastAPI) around the existing detector; until then, use mock data. **Proposed** response for `POST /analyze` (image file in):

```
{
  "trust_score": 12,
  "verdict": "Likely AI-generated",
  "probability_ai": 0.88,
  "heatmap_grid": [[0.9, 0.8, ...], ...],
  "heatmap_png_base64": "...",
  "explanation": ["sentence 1", "sentence 2"],
  "metadata": {"c2pa": "...", "camera": "...", "software": "..."},
  "model_version": "v3"
}
```

Needed screens: upload (drag and drop), result card with score + verdict colour, heatmap overlay, explanation, a "details" panel for metadata, and a history list. Do **not** change anything inside `prototype/detector.py`, `forensics.py` or `trust.py`.

### Task C — Documents and slides (low effort, high value)

Keep `PROJECT_ROADMAP.md` weekly log up to date, prepare Review slides with the new numbers once v3 is trained, and write the "limitations and honest results" section.

### Pawan + Claude

Kaggle notebooks, feature extraction, training v3 and the proof model, API wrapper, deployment decision, results tables.

---

## 8. Deployment: why the model is "big", and what we will do

- The file size is **not** our fault. It comes from the pretrained **CLIP ViT-L/14** image encoder (about 300 million parameters, about 1.2 GB). Our own trained part is about 4 KB.
- Peak memory is about **1.5 GB**. Render's free plan gives 512 MB, and Vercel cannot run PyTorch at all, so neither can host the model.
- **Today it runs** on Streamlit Community Cloud (about 2.7 GB limit); an image takes 30–60 seconds online.

Options for the final full-stack demo (to decide after accuracy is settled):

1. Test a **smaller CLIP** (ViT-B/16, about a quarter of the size) — costs some accuracy; we will measure it.
2. **Compress the model** (half precision / int8) and measure the change.
3. Host the backend where memory is enough: Streamlit Cloud (works now), a small paid plan on Render or similar, or a cloud free tier (needs a card; limits must be checked).
4. **Frontend on Vercel** calling that backend through the API.
5. For a review: run the API on Pawan's laptop and expose it with a free tunnel (for example Cloudflare Tunnel). Fine for a demo.

---

## 9. Timeline (about 1–2 weeks to the review)

| When | Who | What |
|---|---|---|
| Days 1–2 | Claude | Confirm dataset access; write the Kaggle notebook |
| Days 1–3 | Teammates | Task A image collection; Task B mock frontend |
| Days 2–4 | Pawan | Run notebook on Kaggle; bring back the features file |
| Days 4–6 | Pawan + Claude | Train v3 and the proof model; test on the team's images |
| Days 6–8 | Claude / teammates | API wrapper; connect the frontend; update results and slides |
| Day 9+ | All | Rehearsal; check on real and edited photos like the teacher's |

---

## 10. Open questions

1. Will all test photos include people? (affects which datasets we prioritise)
2. Which hosting route for the final deployment (Section 8)?
3. Keep Streamlit as the fallback while the new frontend is built? (Recommended: yes.)
4. Confirm roles for this phase (the roadmap's role table is still blank).

---

## 11. Glossary

- **AUC** — how well a model separates fakes from real images at any threshold (0.5 = guessing, 1.0 = perfect).
- **Cross-generator generalization** — still working on AI tools never seen in training.
- **CLIP** — a big pretrained image model from OpenAI; we use its image encoder, frozen.
- **Frozen** — its weights are never changed, so it cannot memorise one generator's quirks.
- **Head / classifier** — the tiny layer we train on top of CLIP.
- **JPEG augmentation** — randomly compressing/resizing/blurring training images so the model survives real-world damage.
- **Inpainting** — an AI redraws only a masked region of a photo.
- **Mask** — an image marking which pixels were edited.
- **False positive** — a real photo wrongly called fake.
- **Patch** — a 224 × 224 piece of the image scored separately.

---

## 12. Sources

**Our own project files**
- `PROJECT_ROADMAP.md`, `docs/walkthroughs/01_image_prototype.md`, `docs/System_Design.tex`, `docs/SRS.md`, `docs/Review3_QA.md` (in the GitHub repository).

**Research papers**
- Ojha et al., *Towards Universal Fake Image Detectors that Generalize Across Generative Models* (UniversalFakeDetect), CVPR 2023 — <https://arxiv.org/abs/2302.10174>
- Zhu et al., *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image*, NeurIPS 2023 — <https://arxiv.org/abs/2306.08571>
- *COCO-Inpaint: A Benchmark for Detecting and Localizing Inpainting-Based Image Manipulations* — <https://arxiv.org/pdf/2504.18361>
- *DiffSeg30k: A Multi-Turn Diffusion Editing Benchmark for Localized AIGC Detection* — <https://arxiv.org/pdf/2511.19111v2>
- *PromptForge-350k* — <https://arxiv.org/pdf/2603.29386>
- *MLLM-Edit: Benchmarking Image Forgery Detection and Localization under MLLM-based Editing* (NeurIPS 2026 poster) — <https://neurips.cc/virtual/2026/poster/139518>
- *Can VLMs Detect and Localize Fine-Grained AI-Edited Images?* (FragFake) — <https://arxiv.org/pdf/2505.15644>

**Datasets (Hugging Face)**
- OpenFake — <https://huggingface.co/datasets/ComplexDataLab/OpenFake>
- OpenFakeTiny (used in the prototype) — <https://huggingface.co/datasets/ComplexDataLab/OpenFakeTiny>
- DiffSeg30k — <https://huggingface.co/datasets/Chaos2629/Diffseg30k>
- tiny-audits — <https://huggingface.co/datasets/DivyaApp/tiny-audits>
- Tiny-GenImage (used in the prototype) — <https://huggingface.co/datasets/TheKernel01/Tiny-GenImage>

**Note on reliability:** the dataset facts in Section 5 were read from Hugging Face on 14 Oct 2026. PromptForge-350k and MLLM-Edit were found through a web search only; their availability and editor lists are **unverified**. The accuracy expectations in Section 4.4 are estimates, not measurements, until v3 is trained and tested.
