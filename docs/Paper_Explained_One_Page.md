# Our 4-page paper in one page (for your viva)

**One sentence:** we built a screening system that takes an image, video or document, sends it to a specialised checker, and gives back a **Trust Score (0–100)**, a verdict (*Authentic / Needs Human Review / Manipulated*) and an explanation. The **image checker is built and tested**; video and document checkers are **designed** (no results yet).

| Section | What it says, in plain words |
|---|---|
| I–II Introduction, Related work | Fakes are easy to make; detectors trained on one generator fail on new ones; compression hides traces. Others study one medium each; we combine them in a tool a normal person can use. |
| III Requirements | From our SRS: validate files, check images / videos / documents, give a Trust Score, explain it, make a report. Must run on a CPU, under 2.7 GB, and never store uploads. |
| IV System design | Layers: upload → validation → router → analyser (image / video / document) → score and calibration → explanation → report. All analysers return the same "evidence" format, so one can be swapped without touching the rest. Training is done once on a free GPU; the app only runs the trained model. |
| V Analysis modules | **Image:** cut the picture into 224×224 patches without resizing, turn each into 768 numbers with the frozen **CLIP** model, a small classifier **we trained** gives the AI probability, then average. **Video:** sample frames, find faces, score them, average the most suspicious ones, draw a timeline. **Document:** OCR plus three forensic maps feed a two-stream U-Net that marks edited regions, plus checks on totals, dates, fonts, PDF history. **Trust Score:** *T* = 100 × (1 − *p*), with limits chosen on validation data. |
| VI–VII Data, workflow, evaluation | Public datasets only. Extract features once on Kaggle, train in minutes, tune on a validation split, test once. Always test on generators **not seen in training**. |
| VIII–X Status, limits, conclusion | Image checker deployed and measured; next: real photos edited by ChatGPT/Gemini. Not forensic proof; uncertain cases go to a human. |

**Numbers to remember (clean test images, 1,200 real + 2,192 AI, 20 unseen generators):** AUC **0.982**; AI caught **94%**; real photos accepted **92%** (accuracy 93.2%); ChatGPT GPT-Image-1.5 **89%**, Gemini Nano Banana Pro **94%**; the **proof model** (never saw OpenAI or Google images) still catches **84%** and **98%**. Trained on about **30,000** images (214,440 patches). Trust bands: **≥ 54 authentic, ≤ 17 manipulated**. Calibration error **0.106 → 0.034**. About **10 seconds** per image.

**Terms:** *frozen CLIP* = OpenAI's pretrained image model, not changed by us. *Head* = our small trained classifier on top. *Patch* = a 224×224 crop. *Unseen generator* = an AI tool absent from training. *Proof model* = same recipe with all OpenAI/Google images removed. *Platt scaling* = turns a raw score into an honest probability. *Review band* = the middle zone where the system asks a human.

**Likely questions**

- **Did you train CLIP?** No. We use it frozen and train only the small classifier (transfer learning).
- **What is new?** One evidence interface for three media, a calibrated score with a review band, and a training recipe with degraded copies plus a proof model for the unseen-generator claim.
- **Video and documents?** Fully designed and specified; implementation and tests are the next phases, so we claim no results.
- **Real photo edited by ChatGPT?** Not measured yet; we are collecting real pairs. Do not quote a number.
- **Weakness?** Heavy compression, blur and downscaling reduce accuracy; the verdict band sends uncertain cases to a person.
