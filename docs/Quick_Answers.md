# Quick Answers — Review-III Cheat Sheet

> **Unified Content Authenticity Verification System** · Team The Perceptrons · Review-III: 28 Sept (Batch A) / 29 Sept 2026 (Batch B)
>
> **Tip:** say the answer, then stop and let them ask the follow-up. Details are in `Review3_QA.pdf`.

---

## 1. The project in 5 lines

1. A web app: upload an **image, video or document** → get a **Trust Score (0–100)**, a **verdict** and **evidence**.
2. Verdicts: **Likely Authentic / Needs Human Review / Likely Manipulated**.
3. Main idea — **cross-generator generalization**: trained on one AI generator, tested on generators it has never seen.
4. Three phases: **Image** (Oct–Dec 2026) → **Video** (Dec–Jan) → **Document** (Jan–Mar 2027).
5. All free: Python, PyTorch, Colab/Kaggle GPUs, Streamlit.

---

## 2. How it works

**Image:** crop to 224×224 → **CLIP** gives 768 "meaning" features + **FFT** gives 64 "frequency" features → **logistic regression** → calibrated probability → Trust Score + heatmap.

**Video:** take 1 frame per second → **MTCNN** finds faces → each face is scored by the same kind of model → average of the top 20% scores = video score + timeline of suspicious moments.

**Document:** **OCR** reads the words; **ELA, SRM, DCT** show compression and noise differences; a **dual-stream U-Net** marks edited pixels; logic checks catch wrong totals and dates; PDF checks find edit history.

**Trust Score:** all evidence → logistic model → probability *p* → **T = 100 × (1 − p)**. Two thresholds chosen on validation data give the three verdicts.

**Accuracy:** measured on test data never used in training, always split into **seen vs unseen** generators, using accuracy, precision, recall, F1 and **AUC** (plus pixel F1 / IoU for documents).

### Key terms (one line each)

| Term | Meaning |
|---|---|
| CLIP | Big pretrained image model; we use its features, frozen, so it can't memorize one generator |
| FFT | Turns an image into frequencies; AI up-sampling leaves regular peaks there |
| Grad-CAM | Heatmap of the image regions that most affected the decision |
| MTCNN | Face detector made of 3 small CNNs (propose, refine, output face + landmarks) |
| ELA / SRM / DCT | Compression and noise tests: edited regions react differently from the rest |
| U-Net | Network that outputs a mask of which pixels were edited |
| Calibration | Adjusting scores so "80% fake" is right about 80% of the time |
| AUC | Chance a random fake scores higher than a random real (0.5 = guessing, 1.0 = perfect) |

---

## 3. Guide's questions

**Q. New AI generators will come in 4–5 years. How will you handle them?**
We learn cues shared by all generators (CLIP features + up-sampling traces), not one generator's fingerprint, and we always test on unseen generators. If a new tool confuses the model it says "Needs Human Review" instead of guessing, and the small classifier retrains in minutes with a few hundred new images.

**Q. How will you get and keep the accuracy?**
Targets: AUC ≥ 0.95 on the training generator and ≥ 0.80 average on unseen ones, above our EfficientNet baseline. Measured per generator on test data never used in training, and re-measured whenever new generators appear.

**Q. How does detection work, and how is accuracy measured?**
Validate → analyze (CLIP + frequency / faces / U-Net) → fuse evidence → calibrate → Trust Score and report. Accuracy is measured on separate test sets, seen vs unseen, with accuracy, precision, recall, F1 and AUC.

**Q. Which formats do you support?**
Images: JPG, PNG, WEBP, BMP, TIFF, HEIC. Videos: MP4, MOV, AVI, MKV, WEBM. Documents: PDF and scanned images. Everything is converted to plain RGB pixels (or frames / pages) first, so the model never sees the format itself.

**Q. What if the format changes (PNG → JPEG, WhatsApp compression)?**
Compression removes fine traces, so we train with random JPEG compression and test at quality 90/70/50. Real and fake training images are saved in the same format, so the model can't learn "PNG = fake". If evidence gets weak, the score moves to "Needs Human Review".

---

## 4. Review-III questions (from the notice)

**1. Is information domain analysis complete, consistent and accurate?**
Yes — content (input files, features, outputs, stored models), flow (DFD Level 0/1, sequence diagram) and structure (one shared `Evidence` object). Every data item has a defined producer and user.

**2. Is the problem statement categorized and targeted?**
Area: AI → Deep Learning → Computer Vision → Media Forensics. Target: AI-generated images, face-swapped videos and tampered documents, for banking, journalism, insurance and legal use.

**3. Are external and internal interfaces defined?**
External: Streamlit web UI, laptop CPU / cloud GPU, Python libraries, HTTPS. Internal: every analyzer uses `analyze(file) → Evidence`; the Trust Engine and Report Builder read that `Evidence`.

**4. Are requirements consistent with schedule, resources and budget?**
Yes — 26 weeks in 3 phases, PERT gives ~96% chance of finishing on time, 3 members, free GPUs and open-source tools, zero cash cost (COCOMO ≈ 25 person-months of effort).

**5. Are all requirements traceable to system level?**
Yes — a Requirements Traceability Matrix links each requirement (FR1–FR7 and the non-functional ones) to a module, a design section and a test.

**6. What is the product?**
A web app where anyone uploads an image, video or document and gets a Trust Score (0–100), a verdict and evidence (heatmap + plain-language explanation).

**7. Is there a demand for the product?**
Deloitte projects US gen-AI fraud losses up to $40B by 2027; Arup lost $25M to a deepfake video call (2024); India's IT Rules 2026 require labelling synthetic content; current detectors fail on new generators.

**8. Are stakeholders identified properly?**
Users: fact-checkers, bank/KYC teams, insurers, legal investigators, social media moderators, public. Others: people shown in media, guide/HOD/examiners, dataset authors, regulators (MeitY).

**9. Are all requirements captured in line with scope?**
Yes — FR1–FR7 plus non-functional (speed, formats, privacy, reliability). In scope: image, video, document. Out of scope: audio, live streams, legal certification.

**10. Are all types of analysis classes identified?**
Boundary: UploadPage, ReportView. Control: Router, Image/Video/Document Analyzer, TrustEngine, ReportBuilder. Entity: UploadedFile, Evidence, TrustResult, Report, ModelArtifact, Dataset.

**11. Are acceptance criteria decided?**
Targets: image unseen AUC ≥ 0.80, video Celeb-DF AUC ≥ 0.75, document pixel F1 ≥ 0.70 (synthetic), calibration error ≤ 0.05, image result ≤ 5 s on a laptop CPU.

**12. Is the SRS as per IEEE format complete and correct?**
It follows IEEE 830: Introduction, Overall Description, Specific Requirements (interfaces, functional, performance, constraints, attributes) and appendices (diagrams, traceability matrix).

**13. Progress for Review-III?**
Topic finalized (projects 2 + 3 merged), literature survey and research gaps, requirement analysis, full system design with algorithms and UML/DFD, 26-week plan starting 28 Sept.

---

## 5. Likely follow-up questions

**Why merge projects 2 and 3?** Both detect hidden manipulation traces (frequency, noise) that the eye can't see; one system with one Trust Score is deeper than two small tools.

**What is new in your project?** Tested on unseen generators in every phase; a calibrated Trust Score with a human-review band; explanations built only from measured evidence.

**Why CLIP and not only EfficientNet?** A fully trained CNN memorizes its training generator; frozen CLIP features can't, so they generalize better. EfficientNet is our baseline for comparison.

**Why logistic regression?** Simple, fast, doesn't over-fit, and its weights show which evidence mattered.

**Mathematical model?** S = {s, e, X, Y, F, DD, NDD, Sc, Fc}: X = uploaded files, Y = {score, verdict, heatmap, explanation}, F = routing, analysis, fusion and scoring functions, DD = model weights, NDD = user uploads.

**P or NP?** Analyzing a file is linear in pixels / frames → P. The classifier head (logistic regression) is convex, so training it is also polynomial.

**What if accuracy on unseen generators is low?** We report it honestly per generator; the review band prevents confident mistakes; we can tune the encoder's LayerNorm layers or add more generators.

**Do you have the datasets?** GenImage is public; FaceForensics++, Celeb-DF and DocTamper need access requests (sent in Week 1). Backup: our own synthetic forgeries and generated test set.

**What hardware?** Training on free Colab / Kaggle GPUs; the app runs on a normal laptop CPU.

**Why Streamlit?** Quick Python web interface, free hosting, ideal for demos.

---

## 6. Never say

- Accuracy **results** — nothing is trained yet. Say "our **target** is…".
- "We have FaceForensics++ / DocTamper" — say "**requested**".
- "First unified system" — ForensicHub (2025) exists. Say "explainable, calibrated, tested on unseen generators".
