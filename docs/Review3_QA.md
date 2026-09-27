# Review-III Preparation — Questions and Answers

> **Project:** Unified Content Authenticity Verification System (Deepfake Detection with Cross-Generator Generalization and Document Tamper Localization)
>
> **Team:** The Perceptrons — Pawan Singh Jagdish Bhandari · Rahul H. Bramhankar · Bhoomi Kushwah · **Guide:** Dr. G. M. Bhandari (HOD)
>
> **Review-III (Requirement Analysis):** 28 Sept 2026 (Batch A) / 29 Sept 2026 (Batch B)

**How to use this file:** every answer starts with a **short answer** you can say in 20–30 seconds. The points under it are for when the examiner asks a follow-up. Section numbers like "System Design §3" refer to `docs/System_Design.pdf`.

---

## Part A — Questions the guide asked us

### A1. New AI generators will keep coming (in 4–5 years there will be tools we have never seen). How will the system handle them, and how will we keep the accuracy?

**Short answer:** Our model does not memorize the fingerprint of one generator. It learns what all generators have in common and what real camera images look like, and we test it the way the future will test it: on generators it has never seen. When something completely new appears, the system is designed to say **"Needs Human Review"** instead of guessing, and the model can be updated in minutes with a few hundred new examples.

**If they ask for details:**

1. **We learn shared cues, not one fingerprint.**
    - *Semantic branch:* we use a frozen CLIP encoder, trained on about 400 million image–text pairs. Ojha et al. (CVPR 2023) showed that a simple classifier on frozen CLIP features detects images from unseen generators far better than a fully trained CNN, because it cannot over-fit to one generator's quirks.
    - *Forensic branch:* most image generators produce the final picture through a learned **up-sampling decoder**. Up-sampling leaves periodic traces in the frequency spectrum (Durall et al., CVPR 2020) that are shared across many generators. We measure these with an FFT of the noise residual.
2. **The "real" side does not change.** Cameras still work the same way (lens, sensor noise, demosaicing, JPEG). Half of what the model learns — what a real photo looks like — stays valid no matter which new generator comes out.
3. **We simulate the future in our testing.** We train only on Stable Diffusion v1.4 (a 2022 model) and test on 7 other generators plus **our own 2026 test set** made with newer models. This measures exactly what happens when a new generator appears. A 2026 benchmark found that open-source detectors drop from about 79% accuracy on 2020–21 generators to about 38% on 2024 generators — we report our own drop honestly, per generator.
4. **Safe failure instead of wrong answers.** The Trust Score is **calibrated**, and we have a review band. An unfamiliar image usually gets a middle score, so it goes to *Needs Human Review* rather than a confident wrong verdict. Thresholds are chosen so that automatic decisions stay below a target error (5%) on validation data.
5. **Cheap, fast updates.** The big encoder is frozen; only a small classifier head is trained. When a new generator appears, we collect a few hundred of its images, retrain the head in minutes, and replace the file in the model store — the app does not change. It works like antivirus updates.
6. **Several independent kinds of evidence.** A new generator has to fool the semantic features, the frequency features **and** the metadata checks at the same time. We also read **provenance metadata** (such as C2PA Content Credentials) when present — India's IT Amendment Rules 2026 now require platforms to label synthetic content and attach provenance metadata where feasible, so this signal will grow over time. For documents, logic checks (does the invoice total add up?) do not depend on the editing tool at all.
7. **Honest line to end with:** "No detector stays accurate forever — it is an arms race, like antivirus. Our contribution is a design that fails safely, can be updated cheaply, and whose drop on new generators is measured, not hidden."

**How we will get and prove the accuracy (targets we set, not results yet):** seen-generator AUC ≥ 0.95; average AUC on unseen generators ≥ 0.80 and clearly above our EfficientNet baseline; results reported per generator. See A2 for how it is measured.

---

### A2. How will the detection process work, and how will the accuracy be measured?

**Short answer:** Every file goes through the same five steps: **validate → analyze → fuse the evidence → calibrate → report.** Accuracy is measured on test data the model has never seen, and always split into *seen* and *unseen* generators, using accuracy, precision, recall, F1 and AUC.

**Detection process, step by step:**

| Content | What happens |
|---|---|
| **Image** | Re-encode and centre-crop to 224×224 → **CLIP features** (768 numbers) + **frequency features** (64 numbers from the FFT of the noise residual) → logistic-regression classifier → calibrated probability → Trust Score + patch heatmap + spectrum plot |
| **Video** | Sample 1 frame per second (max 32) → detect faces with **MTCNN** → crop faces → frame classifier → combine frame scores (mean of the top 20%) → Trust Score + timeline of suspicious moments. Videos without faces use the image model on full frames. |
| **Document** | PDF → structure checks + render pages; image → **OCR** (words and boxes) + forensic maps (**ELA, SRM noise, DCT**) → **dual-stream U-Net** outputs a mask of edited pixels → logic checks (totals, dates, font size) → Trust Score + highlighted regions |
| **All** | Evidence is fused by a logistic model → calibrated → **Trust Score = 100 × (1 − probability of manipulation)** → verdict: Likely Authentic / Needs Human Review / Likely Manipulated |

**How accuracy is measured:**

1. **Data splits:** training, validation and test sets are kept separate (for example 70/15/15). Videos are split **by video** and documents **by source document**, so nothing from the test set leaks into training.
2. **Seen vs unseen:** we always report two numbers side by side — on the generator we trained on, and on generators we never trained on (GenImage generators, our own 2026 set, Celeb-DF for video, real human-made forgeries (RTM) for documents).
3. **Metrics:**
    - Accuracy = (TP + TN) / all; Precision = TP / (TP + FP); Recall = TP / (TP + FN); F1 = 2PR / (P + R)
    - **AUC** — how well real and fake are separated at *every* threshold (0.5 = random, 1.0 = perfect). We use it because the best threshold often shifts on unseen generators.
    - **Documents:** pixel F1 and **IoU** between the predicted edit mask and the true mask.
    - **Calibration:** Expected Calibration Error (ECE) — when the model says 80%, is it right about 80% of the time?
4. **Robustness tests:** the same test sets after JPEG compression (quality 90/70/50), resizing, blur and screenshots.
5. **Baseline comparison:** every result is compared with a fine-tuned EfficientNet-B0, so we can show our method actually improves things.

---

### A3. Images, videos and documents come in many formats. Which formats does the platform accept, and will a change of format cause problems?

**Short answer:** We accept all common formats and convert every file into one standard internal form — images to RGB pixel arrays, videos to sampled frames, PDFs to page images — so the model never sees the file format, only the content. Format changes *can* weaken evidence (compression destroys fine traces), so we train with them, test with them, and when evidence is weak the Trust Score falls into *Needs Human Review* instead of giving a wrong verdict.

**Supported formats:**

| Type | Accepted formats | How we read it | Converted to |
|---|---|---|---|
| Image | JPG/JPEG, PNG, WEBP, BMP, TIFF, HEIC (iPhone) | Pillow (+ pillow-heif), OpenCV | 8-bit RGB array; rotation from EXIF applied; transparency removed; 16-bit and CMYK converted |
| Video | MP4, MOV, AVI, MKV, WEBM (codecs H.264, H.265, VP9, AV1) | FFmpeg (through OpenCV / PyAV) | frames sampled by timestamp (1 fps), RGB |
| Animated | GIF, animated WEBP | Pillow | treated as a short video |
| Document | PDF, scanned or photographed pages (JPG, PNG, TIFF) | PyMuPDF (PDF), Pillow | 300-dpi page images + PDF structure information |

**Limits:** images up to 20 MB, videos up to 100 MB / 2 minutes, PDFs up to 10 pages.

**Safety checks on upload:**

- The real type is detected from the file's **signature bytes**, not its extension, so a renamed file still works (and a mismatch is noted).
- Corrupt or unsupported files get a clear error message.
- A pixel-count limit blocks "decompression bomb" images; PDFs are parsed, never executed (no JavaScript or macros).

**Does a format change cause problems? Yes, in three ways — and each is handled:**

1. **Lossy re-compression** (PNG → JPEG, WhatsApp, Instagram) removes fine high-frequency traces, which weakens the frequency evidence.
   *Handling:* we train with random JPEG compression; the CLIP branch is less sensitive to compression; we test at JPEG 90/70/50, resize and screenshots; and the calibrated score lowers its confidence instead of flipping the verdict.
2. **Format shortcut in datasets:** public datasets often store real photos as JPEG and fakes as PNG, so a naive model learns "PNG = fake" — then simply converting a fake to JPEG fools it. The paper *Fake or JPEG?* showed that removing this bias improves cross-generator accuracy by more than 11 points.
   *Handling:* we re-save real and fake training images with exactly the same format and quality, so the format carries no information.
3. **Format-aware forensics for documents:** ELA and DCT analysis only make sense for JPEG files. For lossless PNG we rely on SRM noise, the U-Net and OCR logic checks; for PDFs we add structure checks (edit history, dates, fonts). The pipeline chooses the evidence based on the actual format.

**Video-specific:** MP4/MOV/MKV are only containers; FFmpeg decodes the codec inside. Compression strength is handled by training and testing on FaceForensics++ at two compression levels (c23 and c40), and variable frame rate is handled by sampling by timestamp.

---

## Part B — Project Review-III: Requirement Analysis (13 questions from the notice)

### 1. Is information domain analysis complete, consistent and accurate?

**Short answer:** Yes. We analysed the information domain in its three parts — **information content, information flow and information structure** — and documented them with DFDs, a data dictionary (the `Evidence` object) and dataset data cards.

- **Information content (data objects):**
    - *Inputs:* image, video and document files, with metadata (EXIF, PDF properties).
    - *Intermediate data:* sampled frames, face crops, CLIP embeddings, frequency spectra, OCR words and boxes, ELA/SRM/DCT maps, predicted masks.
    - *Outputs:* Evidence object, Trust Score, verdict, heatmaps, explanation, downloadable report.
    - *Stored data:* datasets, trained model weights, calibration parameters, result log.
- **Information flow:** DFD Level 0 and Level 1 and the sequence diagram (System Design §7.2–7.3).
- **Information structure:** the `Evidence` data structure shared by all pipelines (System Design §8.2) and the datasets table (§8.1).
- **Consistent:** one common `Evidence` interface for all three content types; every data object has a defined producer and consumer.
- **Accurate:** checked against the real formats of the datasets we will use (GenImage, FaceForensics++, RTM).

### 2. Is the problem statement categorized in an identified area and targeted towards a specific area in it?

**Short answer:** Yes. Area: **Artificial Intelligence → Deep Learning → Computer Vision → Digital Media Forensics.** Within it we target three specific problems: (1) detecting AI-generated images that generalizes to unseen generators, (2) detecting face-manipulated videos, and (3) localizing tampered regions in documents.

- **Application sectors:** journalism and fact-checking, banking and KYC, insurance claims, legal evidence, social media moderation.
- **Problem statement (one line):** build an explainable system that verifies whether an image, video or document has been AI-generated or tampered with, including content from AI tools not seen during training, and shows *where* and *why*.

### 3. Is external and internal interfacing properly defined?

**Short answer:** Yes. External interfaces (user, hardware, software, communication) and internal module interfaces are defined in the class and sequence diagrams, with one common function signature: `analyze(file) → Evidence`.

| Interface | Definition |
|---|---|
| **User interface** | Streamlit web page: upload file, view Trust Score, verdict, heatmap and explanation, download report |
| **Hardware** | Inference on a normal laptop CPU (8 GB RAM); training on free cloud GPUs (Colab / Kaggle) |
| **Software** | Python, PyTorch, OpenCV, FFmpeg, Tesseract / EasyOCR, PyMuPDF, scikit-learn |
| **Communication** | Browser ↔ app over HTTPS; Google Drive for datasets; GitHub for code |
| **Router → Analyzer** | `analyze(file, content_type) → Evidence` |
| **Analyzer → Trust Engine** | `Evidence.to_vector()` → Trust Score and verdict |
| **Trust Engine → Report Builder** | `build(T, verdict, Evidence) → Report` |
| **Model store** | weights (`.pt`), calibration parameters and thresholds (`.json`), loaded at start-up |

### 4. Are the requirements consistent with the schedule, resources and budget?

**Short answer:** Yes. The schedule is 26 weeks in three phases, PERT gives about a 96% chance of finishing on time, all tools and GPUs are free, and anything that did not fit (audio deepfakes, real-time streams) was moved out of scope.

- **Schedule:** 28 Sept 2026 – 28 Mar 2027. Image 11 weeks, Video 5 weeks, Document 10 weeks (roadmap). PERT: expected 25.9 weeks, σ ≈ 1.19 weeks, P(finish within 28 weeks) ≈ 96%.
- **Resources:** 3 members with rotating roles; free GPUs (Colab / Kaggle); frozen backbone means features are computed once, so GPU needs are small.
- **Budget:** no cash spending — open-source software, research datasets, free hosting. The COCOMO estimate (≈ 25 person-months, semi-detached) is an effort estimate, not money spent.

### 5. Are all requirements traceable to system level?

**Short answer:** Yes. We have a Requirements Traceability Matrix — every requirement maps to a module, a design section and a test.

| ID | Requirement | Module | Design reference | Test |
|---|---|---|---|---|
| FR1 | Upload and validate a file | ContentRouter | System Design Fig. 3 | Valid, invalid, renamed and corrupt files |
| FR2 | Detect AI-generated images | ImageAnalyzer | §3, Algorithms 1–2 | Per-generator AUC, seen and unseen |
| FR3 | Detect manipulated videos | VideoAnalyzer | §4, Algorithm 3 | FaceForensics++ → Celeb-DF AUC |
| FR4 | Localize document tampering | DocumentAnalyzer | §5, Algorithms 4–5 | Pixel F1 / IoU on synthetic and real forgeries |
| FR5 | Trust Score and verdict | TrustEngine | §6, Algorithm 6 | ECE; error of automatic decisions |
| FR6 | Explanation (heatmap + text) | ReportBuilder | §3.7, §6.4 | Every report lists its evidence |
| FR7 | Download report | App / ReportBuilder | Fig. 13 | PDF export works |
| NFR1 | Response time | All pipelines | §10 | Timing on a laptop CPU |
| NFR2 | Privacy | App | §10 | Uploads deleted after the session |

### 6. What is the product?

**Short answer:** A web application where anyone can upload an image, a video or a document and get a **Trust Score from 0 to 100**, a **verdict** (Likely Authentic / Needs Human Review / Likely Manipulated) and the **evidence** behind it — heatmaps, highlighted edited regions and a plain-language explanation.

**Deliverables:** the web app, trained models, our own 2026 test set of new generators, a synthetic document-forgery dataset with masks, source code, and the SRS, design and final reports.

### 7. Is there a demand for the product?

**Short answer:** Yes — deepfake fraud is already causing real losses, new regulation requires platforms to deal with synthetic content, and current detectors fail on new generators.

- **Financial fraud:** Deloitte's Center for Financial Services projects that generative-AI-enabled fraud losses in the US could reach **$40 billion by 2027**, up from $12.3 billion in 2023.
- **Real incident:** in January 2024 an employee of the engineering firm **Arup** in Hong Kong transferred about **$25 million** after a video call in which the "CFO" and other colleagues were all deepfakes.
- **Regulation in India:** MeitY notified the **IT (Intermediary Guidelines) Amendment Rules, 2026** on 10 Feb 2026 (in force 20 Feb 2026). They require synthetically generated content to be clearly labelled and platforms to verify it — which needs detection tools.
- **Technology gap:** open-source detectors lose most of their accuracy on newer generators and on deepfakes collected from the internet (best AUC only about 0.58 on Deepfake-Eval-2024).
- **Documents:** banks, insurers and KYC teams must verify documents that are easy to edit as PDFs or images.

### 8. Is identification of stakeholders done properly?

**Short answer:** Yes. We identified stakeholders by three questions — who uses the system, who is affected by it, and who decides about it.

| Stakeholder | Interest | How the project addresses it |
|---|---|---|
| Journalists and fact-checkers | Verify viral images and videos quickly | Fast web app, explanation they can quote |
| Bank / NBFC KYC and fraud teams | Catch forged documents and fake faces | Document tamper localization, Trust Score |
| Insurance claim verifiers | Detect edited photos and bills | Image + document pipelines, logic checks |
| Legal and forensic investigators | Evidence they can explain | Heatmaps, evidence list, downloadable report |
| Social media moderators | Label synthetic content (IT Rules 2026) | Calibrated score, review band |
| General public | Check suspicious content | Simple interface, plain language |
| People shown in media | Privacy and dignity | No storage of uploads, no training on user data |
| Guide, HOD, project coordinators, examiners | Academic quality, timely progress | Weekly roadmap, reviews, documentation |
| Dataset authors | Licence terms respected | Research-only use, citations |
| Regulators (MeitY) | Compliance with synthetic-content rules | Provenance metadata check, labelling support |

### 9. Are all requirements captured and documented in line with the scope?

**Short answer:** Yes. Functional requirements FR1–FR7, non-functional requirements (performance, formats, reliability, privacy, usability, maintainability) and constraints are documented, and each maps to one of the three phases. Anything outside scope is listed explicitly.

- **In scope:** images, short videos (faces, plus full-frame analysis), scanned/photographed documents and PDFs, calibrated Trust Score, explanations, Streamlit demo.
- **Out of scope:** audio deepfakes, real-time streaming, legal certification, training large language models.
- **Where documented:** SRS (§3 Specific Requirements), System Design §1.5 and §10, and the project roadmap.

### 10. Are all types of analysis classes identified?

**Short answer:** Yes. Using the standard three types of analysis classes — **boundary, control and entity** — we identified:

| Type | Classes |
|---|---|
| **Boundary** (interact with the outside) | UploadPage (Streamlit UI), ReportView, ModelStore loader |
| **Control** (logic and coordination) | ContentRouter, ImageAnalyzer, VideoAnalyzer, DocumentAnalyzer, TrustEngine, ReportBuilder, Trainer / Evaluator |
| **Entity** (data that is stored or passed) | UploadedFile, Evidence, TrustResult (score + verdict), Report, ModelArtifact (weights + calibration), Dataset / DataCard, ForgeryMask |

The class diagram is in System Design §7.4 (Fig. 14).

### 11. Are the acceptance criteria decided?

**Short answer:** Yes. We set measurable targets for every phase and for the whole system. These are **targets to be confirmed with the guide**, not results.

| Area | Acceptance criterion (target) |
|---|---|
| Functional | Accepts all supported formats; rejects invalid or corrupt files with a message; every result has a score, a verdict and at least one piece of evidence; report downloads |
| Image | Seen-generator AUC ≥ 0.95; average unseen-generator AUC ≥ 0.80 and above the EfficientNet baseline |
| Video | FaceForensics++ video-level AUC ≥ 0.90; Celeb-DF (unseen) AUC ≥ 0.75 |
| Document | Pixel F1 ≥ 0.70 on our synthetic test set; on real forgeries, better than the RGB-only U-Net baseline |
| Trust Score | ECE ≤ 0.05; error of automatic decisions ≤ 5% on validation data |
| Robustness | AUC drop ≤ 0.10 at JPEG quality 70 |
| Performance | Image ≤ 5 s, 30-second video ≤ 60 s, document page ≤ 15 s on a laptop CPU |

### 12. Is the SRS document as per IEEE format complete and correct?

> **Action needed:** this question needs the actual SRS document. The notice lists it for 26 Sept 2026.

**Short answer:** Our SRS follows the **IEEE 830** structure (now continued as ISO/IEC/IEEE 29148): introduction, overall description and specific requirements, with every requirement numbered and traceable.

- **1. Introduction:** purpose, scope, definitions and abbreviations, references, overview.
- **2. Overall description:** product perspective, product functions, user classes, operating environment, design constraints, assumptions and dependencies.
- **3. Specific requirements:** external interfaces (user, hardware, software, communication); functional requirements FR1–FR7; performance requirements; design constraints; software attributes (reliability, security, maintainability, portability).
- **Appendices:** analysis models (DFDs, use case, class and sequence diagrams), traceability matrix, acceptance criteria.

### 13. Review-III meeting / presentation for progress of project work (28 Sept – Batch A, 29 Sept – Batch B)

**Short answer:** Since Review-II we have finalized the merged topic, completed the literature survey and research-gap study, completed requirement analysis, written the full system design with algorithms and UML/DFD diagrams, and planned a 26-week, three-phase implementation that starts this week.

**Suggested slide order (8–10 minutes):**

1. Title and team.
2. Problem and product in one slide (what goes in, what comes out).
3. Literature survey and research gaps: detectors fail on new generators; dataset shortcuts (JPEG bias); synthetic-to-real gap for documents; no calibrated, explainable trust score.
4. Requirement analysis: FR/NFR, stakeholders, acceptance criteria (this document).
5. System architecture and the image pipeline figure.
6. Video and document pipelines.
7. Trust Score engine and verdict bands.
8. Plan: three phases, Gantt chart, COCOMO/PERT, risks.
9. Next 3 weeks: repository and environment setup, dataset access requests, evaluation script, EfficientNet baseline.

**Opening line:** "Our system verifies images, videos and documents and returns a calibrated Trust Score with evidence. Its key property is that it is tested on AI generators it has never seen."

---

## Part C — Likely follow-up questions

### C1. What is the mathematical model of your system?

Using set theory, the system is **S = {s, e, X, Y, F, DD, NDD, Sc, Fc}**:

- **s** (start): the user uploads a file. **e** (end): the report is displayed.
- **X** (input) = { x | x is an image, video or document }.
- **Y** (output) = { T, V, H, E } — Trust Score, verdict, heatmap/mask, explanation.
- **F** (functions) = { f_route, f_img, f_vid, f_doc, f_fuse, f_trust, f_report }, where
    - f_img(x) = σ(a·(wᵀ[z; φ] + b) + c), with z = CLIP features, φ = radial frequency features
    - f_vid(V) = mean of the top-k frame probabilities
    - f_doc(D) = mean of the top 1% of U-Net mask probabilities, plus rule flags
    - f_trust(e) = round(100 · (1 − σ(βᵀe + β₀)))
    - verdict: Authentic if T ≥ τ_high; Manipulated if T ≤ τ_low; otherwise Needs Review
- **DD** (deterministic data): model weights, calibration parameters, thresholds.
- **NDD** (non-deterministic data): user uploads — the generator or editing tool is unknown.
- **Sc** (success): a report with a calibrated score and evidence is produced. **Fc** (failure): invalid file, unsupported format or processing timeout.

### C2. Is the problem P, NP-complete or NP-hard?

Inference (analyzing one file) runs in **polynomial time** — linear in the number of pixels, frames or pages — so it is in **P**. Training deep networks is a non-convex optimization problem, but our main classifier head is **logistic regression, which is convex** and solved in polynomial time; the large encoder is pretrained and frozen.

### C3. Why CLIP and not only EfficientNet?

A fully fine-tuned CNN like EfficientNet learns the artifacts of the one generator it is trained on and generalizes poorly. Frozen CLIP features are general and cannot over-fit to one generator, which is why they transfer to unseen generators (Ojha et al., CVPR 2023). We keep EfficientNet as a **baseline** to show the improvement.

### C4. What if the accuracy on unseen generators turns out to be low?

That is itself a research finding, and we will report it per generator. Our fallbacks: the review band prevents confident wrong answers; we can fine-tune only the LayerNorm layers of the encoder (reported to generalize across 14 benchmarks); and we can add more generators to training.

### C5. Do you have the datasets?

GenImage is publicly downloadable. FaceForensics++, Celeb-DF and DocTamper need access requests, which we are sending in Week 1. If access is delayed, our fallback is our own synthetic forgeries and our own generated 2026 test set.

---

## Part D — Do not claim these

- Do not quote accuracy numbers as results — nothing is trained yet. Say "our target is…".
- Do not say we already have FaceForensics++ or DocTamper — say "requested".
- Do not say "first system to unify deepfake and document detection" — ForensicHub (NeurIPS 2025) already unifies them as a benchmark. Our claim is an explainable, calibrated end-user system tested on unseen generators.
- The COCOMO cost (≈ $15,000) is a notional effort cost, not money spent.

---

## Sources

- Deloitte, generative-AI fraud projection: <https://www.deloitte.com/us/en/insights/industry/financial-services/deepfake-banking-fraud-risk-on-the-rise.html>
- Arup deepfake case (CNN): <https://www.cnn.com/2024/05/16/tech/arup-deepfake-scam-loss-hong-kong-intl-hnk>
- IT Amendment Rules 2026 (Khaitan & Co summary): <https://www.khaitanco.com/thought-leadership/MeitY-notifies-the-IT-Amendment-Rules-2026>
- Detector accuracy vs generator year: <https://arxiv.org/abs/2602.07814>
- Deepfake-Eval-2024: <https://arxiv.org/abs/2503.02857>
- Fake or JPEG?: <https://arxiv.org/abs/2403.17608>
- ForensicHub: <https://arxiv.org/abs/2505.11003>
- Full reference list: `docs/System_Design.pdf`, References section
