# Software Requirements Specification (SRS)

> **Project:** Unified Content Authenticity Verification System — Deepfake Detection with Cross-Generator Generalization and Document Tamper Localization
>
> **Team:** The Perceptrons — Pawan Singh Jagdish Bhandari · Rahul H. Bramhankar · Bhoomi Kushwah · **Guide:** Dr. G. M. Bhandari (HOD)
>
> **Institute:** Department of Computer Engineering, JSPM's BSIOTR, Wagholi, Pune · **Version:** 1.0 · 27 September 2026
>
> **Standard:** IEEE Std 830-1998 structure (continued as ISO/IEC/IEEE 29148:2018)

---

## 1. Introduction

### 1.1 Purpose
This document specifies the requirements of a web-based system that verifies whether an image, a video or a document is authentic, AI-generated or tampered with. It is written for the project team, the guide, the project coordinators and the examiners.

### 1.2 Scope
The system accepts an image, video or document, analyses it with deep-learning and forensic methods, and returns a **Trust Score (0–100)**, a **verdict** (Likely Authentic / Needs Human Review / Likely Manipulated) and **evidence** (heatmap or highlighted regions and a plain-language explanation). Its key property is **cross-generator generalization**: it must detect content from AI generators it never saw during training.

**In scope:** still images; short videos (face-manipulated and fully generated); scanned/photographed documents and PDFs; calibrated Trust Score; explanations; web interface.
**Out of scope:** audio deepfakes, real-time streaming, legal-grade certification, training large language models.

### 1.3 Definitions and abbreviations

| Term | Meaning |
|---|---|
| Generator | An AI tool that creates images or videos (e.g. Stable Diffusion, Midjourney, ChatGPT image model, Gemini image model) |
| Cross-generator generalization | Detecting content from generators not used in training |
| CLIP | Pretrained vision-language model (OpenAI); its frozen image encoder provides features |
| Trust Score | 100 × (1 − probability of manipulation), from 0 to 100 |
| AUC | Area under the ROC curve; 0.5 = guessing, 1.0 = perfect |
| ELA / SRM / DCT | Error Level Analysis / Spatial Rich Model noise filters / Discrete Cosine Transform |
| FR / NFR | Functional / Non-functional requirement |

### 1.4 References
1. IEEE Std 830-1998, *Recommended Practice for Software Requirements Specifications*.
2. Project System Design document (`docs/System_Design.tex`) and Project Roadmap (`PROJECT_ROADMAP.md`).
3. Ojha et al., "Towards Universal Fake Image Detectors…", CVPR 2023. · Zhu et al., "GenImage", NeurIPS 2023. · Rössler et al., "FaceForensics++", ICCV 2019. · Qu et al., "DocTamper", CVPR 2023.

### 1.5 Overview
Section 2 describes the product in general; Section 3 lists the specific requirements; Section 4 gives traceability and acceptance criteria.

---

## 2. Overall Description

### 2.1 Product perspective
A new, self-contained system with three analysis pipelines (image, video, document) that share one Trust Scoring Engine and one web interface. Models are trained offline on free cloud GPUs; the web application only loads the trained files (see System Design §2).

### 2.2 Product functions
1. Accept and validate uploaded files of common formats.
2. Detect AI-generated images, including from unseen generators.
3. Detect face-manipulated videos.
4. Localize tampered regions in documents.
5. Fuse evidence into a calibrated Trust Score and verdict.
6. Explain the result (heatmap, highlighted regions, text).
7. Provide a downloadable report.

### 2.3 User classes

| User class | Characteristics |
|---|---|
| General public, students | Non-technical; need a simple verdict and explanation |
| Journalists, fact-checkers, moderators | Frequent use; need speed and clear evidence |
| Bank / KYC / insurance / legal staff | Verify documents and faces; need traceable evidence |
| Project team (administrators) | Train, evaluate and update models |

### 2.4 Operating environment
Web browser (Chrome, Edge, Firefox) on desktop or mobile. Server: Python 3.10+ on Windows or Linux, CPU only (free hosting: Streamlit Community Cloud). Training: Google Colab / Kaggle GPU or a laptop CPU.

### 2.5 Design and implementation constraints
Free and open-source tools only; zero budget; datasets used under their research licences; laptop with 8 GB RAM and no GPU; free hosting memory limit of 2.7 GB; academic timeline of 26 weeks (Sept 2026 – Mar 2027).

### 2.6 Assumptions and dependencies
Public datasets remain available (GenImage, OpenFake; FaceForensics++, Celeb-DF and DocTamper need access requests). Pretrained CLIP weights remain downloadable. Uploaded files are not deliberately adversarial beyond normal editing and compression.

---

## 3. Specific Requirements

### 3.1 External interface requirements
- **User interface:** web page with drag-and-drop upload; result card (Trust Score, colour-coded verdict); heatmap; explanation; tabs for technical evidence; pages for cross-generator results and "How it works".
- **Hardware interface:** runs on a standard laptop CPU (inference); GPU optional for training.
- **Software interface:** Python, PyTorch, open_clip, OpenCV/FFmpeg, Pillow (+HEIC), Tesseract/EasyOCR, PyMuPDF, scikit-learn, Streamlit.
- **Communication interface:** HTTPS between browser and server; GitHub for code; automatic redeploy on push.

### 3.2 Functional requirements

| ID | Requirement | Input → Output |
|---|---|---|
| **FR1** | The system shall validate uploads by file signature, accept JPG, PNG, WEBP, BMP, TIFF, GIF, HEIC images, MP4/MOV/AVI/MKV/WEBM videos and PDF/scanned documents, and reject corrupt or unsupported files with a clear message. | File → standard internal form, or error |
| **FR2** | The system shall estimate the probability that an image is AI-generated using a frozen CLIP encoder and a trained classifier, scoring multiple 224 × 224 patches at native resolution. | Image → probability, patch heatmap |
| **FR3** | The system shall sample video frames, detect faces, score each frame and aggregate scores into one video score with a timeline. | Video → probability, timeline |
| **FR4** | The system shall localize tampered document regions using OCR, forensic maps (ELA, SRM, DCT) and a dual-stream U-Net, plus logic and PDF-structure checks. | Document → mask, region score, flags |
| **FR5** | The system shall compute Trust Score = 100 × (1 − p) and assign a verdict: Authentic (≥ τ_high), Manipulated (≤ τ_low), otherwise Needs Human Review. | Evidence → score, verdict |
| **FR6** | The system shall explain every result with a heatmap/highlighted regions and sentences generated only from measured evidence, including file metadata (EXIF, C2PA). | Evidence → explanation |
| **FR7** | The system shall let the user view and download a report of the analysis. | Result → report |

### 3.3 Performance requirements
- Image result within 5 s (target) on a laptop CPU; prototype currently 12–15 s. 30-second video within 60 s; document page within 15 s.
- Server memory below 2.7 GB (free hosting limit); prototype peak about 1.5 GB.

### 3.4 Software system attributes
- **Reliability:** calibrated probabilities; uncertain cases go to *Needs Human Review*; no verdict without evidence.
- **Security and privacy:** uploads processed in temporary memory and never stored or used for training; files parsed, never executed; image-size limit against decompression attacks.
- **Usability:** colour-coded verdicts, plain-language explanations, results in one screen.
- **Maintainability:** independent analyzers behind one `Evidence` interface; models replaceable without changing the app.
- **Portability:** runs on Windows and Linux; CPU-only.

### 3.5 Other requirements
Datasets and pretrained models are credited with their licences; no claims of legal certification; results reported honestly for seen and unseen generators.

---

## 4. Appendices

### 4.1 Requirements traceability matrix

| ID | Module | Design reference | Test |
|---|---|---|---|
| FR1 | ContentRouter | System Design Fig. 3 | Valid, invalid, renamed and corrupt files |
| FR2 | ImageAnalyzer | §3, Algorithms 1–2 | Per-generator AUC, seen vs unseen |
| FR3 | VideoAnalyzer | §4, Algorithm 3 | FaceForensics++ → Celeb-DF AUC |
| FR4 | DocumentAnalyzer | §5, Algorithms 4–5 | Pixel F1 / IoU on synthetic and real forgeries |
| FR5 | TrustEngine | §6, Algorithm 6 | Calibration error; error of automatic decisions |
| FR6 | ReportBuilder / trust module | §3.7, §6.4 | Every result lists its evidence |
| FR7 | ReportBuilder | Fig. 13 | Report download works |

### 4.2 Acceptance criteria (targets)

| Area | Target |
|---|---|
| Image | Seen AUC ≥ 0.95; average unseen-generator AUC ≥ 0.80 and above the EfficientNet baseline |
| Video | FaceForensics++ AUC ≥ 0.90; Celeb-DF (unseen) AUC ≥ 0.75 |
| Document | Pixel F1 ≥ 0.70 on synthetic test; better than RGB-only baseline on real forgeries |
| Trust Score | Calibration error ≤ 0.05; automatic-decision error ≤ 5% |
| Functional | All listed formats accepted; invalid files rejected with a message; every result has score, verdict and evidence |

### 4.3 Analysis models
Use case diagram, DFD Level 0 and 1, sequence diagram and class diagram: see System Design §7.
