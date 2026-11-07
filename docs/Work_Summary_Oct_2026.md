# Work Summary — What We Did, How Accurate It Is, What Is Left

> **For:** The Perceptrons — Pawan, Rahul, Bhoomi · **Period:** 13–15 Oct 2026 · **Written:** 15 Oct 2026
>
> A short, complete summary of the v3 work so everyone knows what changed. Detailed step-by-step log: [Walkthrough 02](walkthroughs/02_v3_work_log.md). Plan and tasks for the team: [Team Briefing](Team_Briefing_v3_Plan.md) (`Team_Briefing_v3_Plan.pdf`).

**Live app:** <https://deepfake-detector-perceptrons.streamlit.app> · **Code:** GitHub repository `DeepFake-Detection-Using-CrossGeneratorGeneralization`

---

## 1. Why we did this

After the first prototype was shown to the teacher, the feedback was: *it was trained on old generators, and does not work well on new images from ChatGPT and Gemini*. We measured the problem on the old model (v2):

- ChatGPT images: only 70% caught.
- Images after JPEG compression (how people really share them): only **14%** caught.
- Only 4 Gemini test images, so the result meant almost nothing.
- Training data: about 6,000 images, mostly old generators.

## 2. What we did (in order)

1. **Recovered the project context** from the repository (the old chat history was lost) and agreed the plan: keep v1 and v2 as the experiment, build a stronger **v3** model for the demo and a **proof model** to keep the cross-generator claim honest.
2. **Found better data.** The full **OpenFake** dataset (millions of images, about 60 modern generators, with held-out test generators including GPT-Image-1.5, GPT-Image-2 and Nano Banana Pro) plus two datasets of **AI-edited images with masks** (DiffSeg30k, tiny-audits).
3. **Used a free Kaggle GPU** (run from the laptop through the Kaggle command line) to turn about 40,000 images into 414,433 patch features with the frozen CLIP model. Every training image also got a **randomly damaged copy** (resized, blurred, JPEG 50–95) — this is "compression augmentation".
4. **Checked the pipeline** before trusting it: the old v2 model gave the same accuracy on the new features (AUC 0.957 vs 0.95 before), so the new pipeline matches the app.
5. **Trained three models:** **v3** (everything, including ChatGPT and Gemini images), **v3 proof** (the same, but with every OpenAI and Google image removed) and **v3 without compression training** (to show what that training adds).
6. **Chose the verdict thresholds from validation data** (not guessed): Trust Score 54 or more = *Likely Authentic*, 17 or less = *Likely AI-generated*, in between = *Needs Human Review*.
7. **Put v3 into the app**, tested it in a real browser, added a **v3 results section** to the Cross-Generator Test page, and **deployed** it. Found and fixed a hosting problem where the live site kept the old model.
8. **Wrote the documents:** work log, team briefing, SRS, updated roadmap with ticks and a results board.

## 3. How accurate is it? (held-out images the model never trained on)

Test set: 1,200 real photos + 2,192 AI images from 20 generators that are **not** in the training set. "Caught" = share of AI images labelled AI.

| | Old v2 | **v3** |
|---|---|---|
| AI images caught, clean files | 81% | **94%** |
| After JPEG compression (quality 70) | 14% | **66%** |
| Half size + JPEG | 50% | **86%** |
| In-the-wild images (Reddit) | 36% | **73%** |
| **ChatGPT** (GPT-Image-1.5) | 70% | **89%** |
| **ChatGPT** (GPT-Image-2) | 43% | **92%** |
| **Gemini** (Nano Banana Pro, 47 images) | 66% | **94%** |
| Real photos called real | 94% | 93% |

**The cross-generator proof:** the proof model never saw any OpenAI or Google image in training, yet catches **84% of ChatGPT** and **98% of Gemini** images (old v2: 70% and 66%). So the model learns what AI images have in common, not one company's style.

**What the verdicts mean in practice** (held-out test split):

| Image | Called Authentic | Needs Review | Called AI |
|---|---|---|---|
| Real photo, clean | 85% | 12% | 3% |
| AI image, clean | 3% | 10% | 87% |
| AI image, JPEG 70 | 19% | 29% | 52% |
| In-the-wild AI image (Reddit) | 12% | 33% | 56% |

**Edited images** (patch level, held-out images from research datasets): 89% of edited patches caught on clean images, 80% after compression; the unedited area of an edited image looks real 79% of the time.

## 4. Honest limits (say these to the examiners before they find them)

- **Compressed or screenshotted AI images are still the hardest case** — the model usually asks for human review instead of being wrong, but 7–19% are still called authentic.
- **A few real photos land in *Needs Human Review*** (about 12% of clean real photos) and about 1–4% are wrongly called AI.
- **Not yet measured:** edits made by **ChatGPT or Gemini on real photos**. The edited-image numbers above come from research datasets. This is exactly what the team's image collection (Task A) will test.
- The heatmap is still coarse (3 × 3 patches); the finer version is not built yet.
- The old test numbers use one dataset family; our own team-collected images are the fair real-world check.

## 5. What changed in the project (files)

| File or folder | Change |
|---|---|
| `kaggle/extract/extract.py`, `kaggle/probe/probe.py` | New: Kaggle GPU scripts (feature extraction, speed check) |
| `prototype/train_v3.py` | New: trains and evaluates v3, v3 proof, v3 without compression training |
| `prototype/calibrate_v3.py` | New: probability calibration and verdict bands |
| `prototype/weights/our_head_v3*.npz/json` | New: the trained v3 models (a few KB each) |
| `prototype/results/v3_results.json`, `v3_calibration.json` | New: all result numbers |
| `prototype/detector.py`, `trust.py`, `app.py` | v3 is the default model; per-model verdict bands; v3 results on the Cross-Generator page; cache fix for hosting |
| `docs/walkthroughs/02_v3_work_log.md` | New: running log of everything done (13 entries) |
| `docs/Team_Briefing_v3_Plan.md` and PDF | New: plan and tasks for the team |
| `docs/SRS.md`, `docs/presentations/` | Added to the repository |
| `PROJECT_ROADMAP.md` | Ticks, status, Week 3 log, v3 results board |

## 6. Where are we? (progress)

The project has three phases: **Image** (Weeks 1–11), **Video** (12–16), **Document** (17–26).

| Phase | Checklist items done | Our honest estimate |
|---|---|---|
| **1 — Image** | 14 of 37 done, 4 partly | **about 65–70%** |
| 2 — Video | 0 of 15 | 0% |
| 3 — Document | 0 of 26 | 0% |
| **Whole project** | | **about 28%** (Phase 1 is 11 of the 26 weeks) |

Why two numbers for Phase 1: the roadmap checklist counts only items ticked (about 43%), but several unticked items were replaced by better work (for example OpenFake gave us proper train / validation / test splits and far more data than the planned small GenImage download). Judged by what the phase is for — *detect AI images including unseen generators, with a Trust Score, heatmap and a working app* — the core is built and deployed. The remaining Phase 1 work is validation and write-up.

**Phase 1 still open:**
- test on the team's own ChatGPT and Gemini photos, including edited ones (Task A)
- finer heatmap and a verdict based on the most suspicious region (for partly edited images)
- EfficientNet baseline for the comparison table
- robustness curves at more compression levels and a reliability (calibration) plot
- a data card, sample-image grid and the 2–3 page Phase 1 summary
- the API and separate frontend, and the final hosting decision

## 7. What each person does next

| Who | Task |
|---|---|
| **Teammate (images)** | **Task A** in the Team Briefing: 100+ real photos, 50 edits each in ChatGPT and Gemini, 50 fully generated images from each, plus compressed copies, with the naming and `pairs.csv` described there. Put everything in the shared Drive folder. |
| **Teammate (frontend)** | **Task B**: upload page, result card with score and verdict colour, heatmap overlay, explanation, details panel, history. Build against the JSON format in the Team Briefing with fake data; the API comes from Pawan and Claude. |
| **Pawan + Claude** | Measure the team's images, finer heatmap, API wrapper, hosting decision, slides. |

## 8. Which file is which

| Question | File |
|---|---|
| "The file we made a day ago for the team" | `docs/Team_Briefing_v3_Plan.md` (PDF: `docs/Team_Briefing_v3_Plan.pdf`) |
| This summary | `docs/Work_Summary_Oct_2026.md` (PDF: `docs/Work_Summary_Oct_2026.pdf`) |
| Every step with numbers | `docs/walkthroughs/02_v3_work_log.md` |
| First prototype explained | `docs/walkthroughs/01_image_prototype.md` |
| Plan, ticks, results board | `PROJECT_ROADMAP.md` |

PDFs are not stored on GitHub (they are generated locally from the `.md` files); ask Pawan for the latest PDF or create one with `python docs/tools/md2pdf.py <file>.md`.

---

## 9. How the system is built — "who learns what?" (read this before the viva)

### 9.1 The two parts of the model

| Part | Who made it | Do we train it? |
|---|---|---|
| **CLIP ViT-L/14** — a big neural network that looks at an image and turns it into 768 numbers describing it | OpenAI, already trained on about 400 million images | **No.** It is *frozen*. We only use it as the "eyes". |
| **Our classifier ("head")** — a small model (logistic regression) that takes those 768 numbers and outputs "probability this is AI" | **Us** | **Yes.** We train it on labelled real and fake examples. |

So we **train a classifier on top of a pretrained model**. This is standard in research (*transfer learning* / *linear probing*) and it is the method of the paper we followed (UniversalFakeDetect, CVPR 2023). It is also the reason the model works on generators it never saw: CLIP is too general to memorise one generator's style, and the small head can only learn what AI images have in common.

**Who learns how generators make images?** Nobody. The head does not learn how a generator works. It learns which patterns in CLIP's numbers appear in AI images but not in real photos. It is a statistical learner on labelled examples.

**What to say to the teacher:** "We use a frozen pretrained CLIP encoder and train our own classifier on many generators. Our work is the data design, the training recipe (including compression augmentation), the cross-generator tests, the calibration, and the application." **Do not say** we trained a neural network from scratch — we did not.

### 9.2 The datasets
They are **existing public datasets, already labelled real or fake** (OpenFake, GenImage, DiffSeg30k, tiny-audits, all from Hugging Face). We did not create them and nothing is trained "from scratch". Our own ChatGPT/Gemini photos from the team will be an extra *test* set.

### 9.3 What was actually done, step by step (practical)
1. **Checked the data** with a Python script that asks Hugging Face what datasets exist, their size, licence and file layout.
2. **Wrote a Python script and ran it on Kaggle's GPU** (through the Kaggle command line, using Pawan's login). It does *inference only, no training*: it reads images from Hugging Face, cuts 224 × 224 patches, applies random compression to a copy of every training image, runs the frozen CLIP on every patch (768 numbers per patch) and saves the numbers with their labels. Result: **414,433 patches**.
3. **Downloaded those files to the laptop and trained the head** with scikit-learn (logistic regression): a few minutes of computing. The result is a tiny file with 768 weights and one bias.
4. **Tested on images never used in training** and chose the verdict thresholds on a separate validation split.
5. **Plugged the new head into the app** and deployed it.

### 9.4 Libraries and files
FastAPI is **not** used yet (it is planned for the separate frontend).

| Tool | Used for |
|---|---|
| PyTorch + open_clip | Run the CLIP model |
| Pillow, NumPy | Open images, cut patches, JPEG compression |
| pyarrow, huggingface_hub | Read datasets from Hugging Face |
| scikit-learn, pandas | Train the head, compute the results |
| Streamlit | The web app |
| Kaggle command line | Start the GPU job from the laptop |

| File | What it does |
|---|---|
| `kaggle/extract/extract.py` | (Kaggle) images → patches → CLIP features |
| `prototype/train_v3.py` | Trains the heads and computes all results |
| `prototype/calibrate_v3.py` | Probability calibration and verdict bands |
| `prototype/detector.py` | Scoring: image → patches → CLIP → head → probability and heatmap |
| `prototype/trust.py` | Trust Score, verdict, explanation sentences |
| `prototype/forensics.py` | File loading, frequency plot, error-level analysis, metadata |
| `prototype/app.py` | The Streamlit pages |

### 9.5 How the parts are connected
There is **no separate backend and frontend**: Streamlit is both, in one Python program. When an image is uploaded, `app.py` calls `detector.py`, which loads CLIP (downloaded from Hugging Face) and our small weights file and returns a probability; `trust.py` turns it into a Trust Score and verdict, and Streamlit draws the page. GitHub holds the code and Streamlit Community Cloud runs it.

### 9.6 What are v1, v2, v3 in the sidebar?
All use the **same frozen CLIP**; only the head (= the training data) differs. Choosing one in the sidebar loads a different weights file.

| Version | Trained on | Why it exists |
|---|---|---|
| v1 | 1,600 images, Stable Diffusion 1.5 only | Pure experiment: one generator |
| v2 | About 6,000 images, many generators, no ChatGPT/Gemini | The first prototype |
| UniversalFakeDetect | A published model trained on ProGAN only (not ours) | Comparison with other people's work |
| **v3** | About 30,000 images incl. ChatGPT and Gemini, compression training, edited-image patches | The strong model for the demo |
| v3 proof | v3 without any OpenAI or Google image | Proves the cross-generator claim |
| v3 no compression training | v3 on clean images only | Shows what compression training adds |

---

## 10. How good is it? Our honest assessment

- **Strong for a final-year project, and honest.** On held-out generators it catches about 94% of clean AI images and about 90% of ChatGPT and Gemini images, with real photos called real about 93% of the time; and the cross-generator claim has real evidence (the proof model, 84% ChatGPT / 98% Gemini, never saw either).
- **Weak spots:** compressed or screenshotted AI images (about a third still missed), and edits of real photos by ChatGPT/Gemini have **not been measured yet**.
- **The remaining Phase 1 items are not all "easy":** write-ups, data card, extra plots and the finer heatmap are easy; the EfficientNet baseline, API, frontend and hosting are medium; **verifying on the team's real images is the uncertain one** — if the results are poor it means another round of training.

---

## 11. Plan to improve the image section (v4) — *to be done when Pawan says go*

The image section is the focus of the project, so it should be as strong as possible. Ordered by the gain we expect (expectations are guesses until tested):

1. **Use CLIP's middle layers, not only its last output.** Research on CLIP-based detectors suggests the middle layers keep the low-level traces that compression erases. One Kaggle run, no new data. Best chance to improve compressed images.
2. **More data.** v3 used about 30,000 of OpenFake's 2.3 million training images (about 1%). First a cheap test: train on 25%, 50% and 100% of what we have and see whether the score still rises. If it does, extract 100,000–300,000 images on Kaggle (several runs; each Kaggle GPU session can last about 9–12 hours, with a weekly GPU allowance).
3. **Closer-to-real data.** (a) Phone-style real photos from the team (OpenFake's real photos come from stock-photo sites, but the teacher will upload phone photos). (b) Harsher, more realistic damage: WhatsApp-style double compression, screenshots, upscaling, different JPEG qualities. Part of the team's photos must be kept aside as a *test set*, not used for training.
4. **A stronger classifier.** A small neural network on top of CLIP instead of a linear layer, or lightly fine-tuning CLIP itself (LoRA). This is where a long Kaggle training run (several hours) makes sense. Risk: the model can start memorising the training generators, so the proof test must be repeated.
5. **More patches at test time** (6 × 6 grid instead of 3 × 3): no retraining, better for partly edited images and the heatmap, slower analysis.
6. **Finer heatmap and a verdict based on the most suspicious region**, for partly edited images.
7. **The missing roadmap items:** EfficientNet baseline (a network trained on our own data, a good comparison), robustness curves at more compression levels, calibration plot, data card, the 2–3 page Phase 1 summary.

**Order we will follow:** (1) measure the team's real images first; (2) one Kaggle run with middle-layer features plus the "does more data still help" test; (3) decide whether to go bigger (more data, small network, fine-tuning).

### Why not 1 million images already? (comparison with a plant-disease project)

| | Plant-disease classifier | Our AI-image detector |
|---|---|---|
| Training images | About 1 million | About 30,000 (182,520 patches, plus about 32,000 edited-image patches) |
| What is trained | Usually a whole neural network, for hours | Only a small head on top of frozen CLIP, in minutes |
| The task | Fixed classes (known plants and diseases) | Open-ended: new AI generators appear every few months |
| Why accuracy is high / lower | Test images look like the training images | The test **deliberately** uses generators and conditions not in training |

A very high accuracy on a test that resembles the training data is not comparable with our numbers: our test is built to be *different* from training. v3 is a first pass built on a laptop and a short Kaggle run; with a linear head, more data eventually stops helping, so a bigger run only pays off together with more model capacity (point 4).

### Extra tasks for the teammates (in addition to Task A and Task B in the Team Briefing)

- **Phone-style real photos:** at least 200 real photos from your own phones (different phones if possible; indoor, outdoor, people, text on boards), original files. Also send some through WhatsApp and take screenshots of some.
- **Do not edit or filter them.** Keep a clear folder structure and write which phone took which photo.
- **Test vs training:** the photos are split: about 30% will be kept only for testing, the rest may be used for training v4.

### Still to come (not decided yet)
Review 4 took place on 26 October 2026 (the review documents were removed from the working folder on 7 November). The remaining image work is listed in Entry 29 of the work log (Walkthrough 02); this summary and the Team Briefing will be updated when it is finished.
