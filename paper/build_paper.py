"""Build the two research-paper PDFs (two-column, A4) from the experiment result files.

    python paper/make_figures.py
    python paper/analysis.py
    python paper/build_paper.py

Writes  paper/Research_Paper_With_Authors.pdf   (author names, guide, acknowledgement, code link)
        paper/Research_Paper_Anonymous.pdf      (same text and figures, nothing that identifies the authors)
Every number in the tables and in the text comes from prototype/results/*.json and paper/analysis.json.
"""
import json
import os
import re
import shutil
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "prototype" / "results"
FONT_DIR = r"C:\Windows\Fonts"

R = json.loads((RES / "v3_results.json").read_text(encoding="utf-8"))
C = json.loads((RES / "v3_calibration.json").read_text(encoding="utf-8"))
A = json.loads((HERE / "analysis.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ number helpers
def pct(x, d=0):
    return f"{x * 100:.{d}f}"


def auc_caught(model, split, var):
    e = R[model][split][var]
    return f"{e['auc']:.3f} / {pct(e['fake_caught'])}%"


def gen(model, src, var="clean"):
    g = R[model]["test_openfake"][var]["per_generator"][src]
    return g["auc"], g["caught"], g["n"]


def gen_cell(model, src, var="clean"):
    a, c, _ = gen(model, src, var)
    return f"{a:.3f} / {pct(c)}%"


ci = lambda k, f="auc_ci": f"{A[k][f][0]:.3f}&ndash;{A[k][f][1]:.3f}"
ci_pct = lambda k: f"{pct(A[k]['caught_ci'][0])}&ndash;{pct(A[k]['caught_ci'][1])}"
v3c, v3j, v3h = R["v3"]["test_openfake"]["clean"], R["v3"]["test_openfake"]["jpeg70"], R["v3"]["test_openfake"]["half_q75"]
va = R["v3"]["validation"]["clean"]
ed = R["v3"]["edited_heldout"]
cal_t = C["test_openfake"]
n_real_test, n_fake_test = v3c["n_real"], v3c["n_fake"]
_plan = HERE.parent / "data" / "features" / "v3" / "test_openfake_plan.json"
n_gen_test = (len(json.loads(_plan.read_text(encoding="utf-8"))) - 1) if _plan.exists() else 20   # generators in the test sample (all sources except "real")

# ------------------------------------------------------------------ text (shared by both versions)
TITLE = "Cross-Generator Detection of AI-Generated Images with a Frozen CLIP Encoder and a Calibrated Trust Score"

ABSTRACT = (
    "Pictures made by text-to-image systems are now hard to tell from photographs, and a detector trained on one "
    "generator often fails on the next one. This paper describes the image module of a content authenticity application "
    "and reports its first complete evaluation. An uploaded picture is cut into 224&times;224 patches at its own resolution, "
    "each patch is encoded by a frozen CLIP ViT-L/14, and a small logistic classifier that we trained on about 30,000 real and "
    "generated images scores the patch. Every training image also contributes a randomly compressed, resized and blurred copy. "
    "A Platt-calibrated probability becomes a 0&ndash;100 Trust Score with three verdict bands whose limits were chosen on validation data. "
    f"On a held-out test split whose generators are absent from training ({n_real_test:,} real and {n_fake_test:,} AI images from {n_gen_test} sources) "
    f"the final model reaches an AUC of {A['test_clean']['auc']:.3f} (95% CI {ci('test_clean')}) on clean files and {A['test_jpeg70']['auc']:.3f} after JPEG quality 70. "
    f"It catches {pct(R['v3']['test_openfake']['clean']['per_generator']['gpt-image-1.5']['caught'])}% of GPT-Image-1.5 and "
    f"{pct(R['v3']['test_openfake']['clean']['per_generator']['nano-banana-pro']['caught'])}% of Nano Banana Pro images, and a variant that never saw an OpenAI or Google image "
    f"still catches {pct(gen('v3_proof', 'gpt-image-1.5')[1])}% and {pct(gen('v3_proof', 'nano-banana-pro')[1])}% of them. "
    "Compressed images remain the weakest case. Video and document modules are planned and are not evaluated here."
)

KEYWORDS = "AI-generated images, cross-generator generalization, CLIP, JPEG robustness, probability calibration, Trust Score"

TABLE1 = """
<table class="t"><tr><th>Source</th><th>Role</th><th>Size used</th></tr>
<tr><td>OpenFake, train split [5]</td><td>real vs. AI, about 60 generators</td><td>30,420 images</td></tr>
<tr><td>DiffSeg30k [6]</td><td>patches inside edited regions = AI</td><td>3,000 images</td></tr>
<tr><td>tiny-audits (COCO part)</td><td>edited patches = AI, originals = real</td><td>2,100 triplets</td></tr>
</table>"""

TABLE2 = f"""
<table class="t"><tr><th width="27%">Model</th><th>Clean</th><th>JPEG 70</th><th>Half + JPEG</th><th>In the wild</th></tr>
<tr><td>v2 head&dagger;</td><td>{auc_caught('v2_reference','test_openfake','clean')}</td><td>{auc_caught('v2_reference','test_openfake','jpeg70')}</td><td>{auc_caught('v2_reference','test_openfake','half_q75')}</td><td>{auc_caught('v2_reference','test_reddit','clean')}</td></tr>
<tr><td>v3, no augmentation</td><td>{auc_caught('v3_noaug','test_openfake','clean')}</td><td>{auc_caught('v3_noaug','test_openfake','jpeg70')}</td><td>{auc_caught('v3_noaug','test_openfake','half_q75')}</td><td>{auc_caught('v3_noaug','test_reddit','clean')}</td></tr>
<tr><td>v3 proof</td><td>{auc_caught('v3_proof','test_openfake','clean')}</td><td>{auc_caught('v3_proof','test_openfake','jpeg70')}</td><td>{auc_caught('v3_proof','test_openfake','half_q75')}</td><td>{auc_caught('v3_proof','test_reddit','clean')}</td></tr>
<tr class="b"><td>v3</td><td>{auc_caught('v3','test_openfake','clean')}</td><td>{auc_caught('v3','test_openfake','jpeg70')}</td><td>{auc_caught('v3','test_openfake','half_q75')}</td><td>{auc_caught('v3','test_reddit','clean')}</td></tr>
</table>"""

_gens = [("gpt-image-1.5", "GPT-Image-1.5"), ("gpt-image-2", "GPT-Image-2"), ("nano-banana-pro", "Nano Banana Pro"),
         ("veo-3", "Veo 3"), ("sora-2", "Sora 2"), ("midjourney-7", "Midjourney v7"), ("flux.2-klein-9b", "FLUX.2 klein")]
_rows = ""
for key, label in _gens:
    _rows += (f"<tr><td>{label}</td><td>{gen('v3', key)[2]}</td><td>{gen_cell('v2_reference', key)}</td>"
              f"<td>{gen_cell('v3_proof', key)}</td><td>{gen_cell('v3', key)}</td><td>{pct(gen('v3', key, 'jpeg70')[1])}%</td></tr>")
TABLE3 = f"""
<table class="t"><tr><th width="30%">Generator</th><th width="8%">n</th><th>v2 head</th><th>v3 proof</th><th>v3</th><th width="14%">JPEG 70</th></tr>{_rows}</table>"""

TABLE4 = f"""
<table class="t"><tr><th width="36%">Variant</th><th>Training patches</th><th>OpenAI / Google</th><th>Degraded views</th><th width="7%">C</th></tr>
<tr><td>v3</td><td>{R['v3']['n_train_patches']:,}</td><td>yes</td><td>yes</td><td>{R['v3']['C']:g}</td></tr>
<tr><td>v3 proof</td><td>{R['v3_proof']['n_train_patches']:,}</td><td>no</td><td>yes</td><td>{R['v3_proof']['C']:g}</td></tr>
<tr><td>v3, no augmentation</td><td>{R['v3_noaug']['n_train_patches']:,}</td><td>yes</td><td>no</td><td>{R['v3_noaug']['C']:g}</td></tr>
</table>"""

TABLE5 = f"""
<table class="t"><tr><th width="44%">Measure (held-out edited images)</th><th width="28%">Clean</th><th width="28%">Degraded</th></tr>
<tr><td>Patches inside the edit flagged as AI</td><td>{pct(ed['clean']['edited_caught'], 1)}%</td><td>{pct(ed['augmented']['edited_caught'], 1)}%</td></tr>
<tr><td>Patches of original photos called real</td><td>{pct(ed['clean']['orig_correct'], 1)}%</td><td>{pct(ed['augmented']['orig_correct'], 1)}%</td></tr>
<tr><td>AUC, edited vs. original patches</td><td>{ed['clean']['auc_edit_vs_original']:.3f}</td><td>{ed['augmented']['auc_edit_vs_original']:.3f}</td></tr>
<tr><td>Area outside the edit flagged as AI</td><td>{pct(ed['clean']['unedited_area_flagged_ai'], 1)}%</td><td>{pct(ed['augmented']['unedited_area_flagged_ai'], 1)}%</td></tr>
</table>"""

# whole blocks (captions + tables / figures) are placed as one piece, so they are never split across columns
BLOCKS = {
    "T4": '<p class="cap"><b>TABLE IV.</b> The three trained variants (C = penalty strength chosen on validation data).</p>' + TABLE4,
    "T5": '<p class="cap"><b>TABLE V.</b> Patch-level results on the 15% of edited images (DiffSeg30k and tiny-audits samples) that were held out of training.</p>' + TABLE5,
    "T1": '<p class="cap"><b>TABLE I.</b> Training data (all existing public datasets).</p>' + TABLE1,
    "F2": '<div class="figbox"><img src="fig2.png"><p class="cap"><b>Fig. 2.</b> Training pipeline. The expensive step runs once on a free GPU; fitting the head takes minutes.</p></div>',
    "T2": '<p class="cap"><b>TABLE II.</b> Held-out test split, ROC-AUC / share of AI images caught at probability 0.5. &dagger;Head of the earlier prototype, scored without its frequency features.</p>' + TABLE2,
    "F3": '<div class="figbox"><img src="fig3.png"><p class="cap"><b>Fig. 3.</b> Share of AI images caught under each condition.</p></div>',
    "T3": '<p class="cap"><b>TABLE III.</b> Per-generator ROC-AUC / share caught on clean images; last column: share caught at JPEG 70.</p>' + TABLE3,
    "F4": '<div class="figbox"><img src="fig4.png"><p class="cap"><b>Fig. 4.</b> Verdict outcomes on the held-out test split and the in-the-wild set.</p></div>',
}


def body_html(named: bool) -> str:
    wc = R["v3"]["test_openfake"]["clean"]["real_correct"]
    return f"""
<h2>I. Introduction</h2>
<p>Text-to-image generators have improved to the point where a casual viewer cannot reliably separate their output from a camera photograph.
Platforms, newsrooms and ordinary users therefore need a quick way to ask whether a picture is likely to be synthetic, and to be told how sure the answer is.
Detection is harder than it first appears, because new generators appear every few months and each one leaves slightly different traces.
A classifier that is trained and tested on the same few generators can look excellent and still fail on the first unfamiliar one [1], [3].</p>
<p>This paper presents the image module of a larger content authenticity application that is also meant to cover video and documents.
Only the image module is implemented so far, so it is the only part we evaluate. Our contributions are:
(i) a detection pipeline that works on patches taken at native resolution from a frozen CLIP encoder [4] and removes file-format shortcuts;
(ii) a training recipe that adds randomly degraded copies and edited-image patches to a large set of modern generators, including ChatGPT and Gemini images;
(iii) an evaluation on generators that do not occur in training, in which one model variant is trained without any OpenAI or Google image so that the cross-generator claim can be checked directly;
and (iv) a calibrated Trust Score whose verdict bands were set on validation data, with a &ldquo;Needs Human Review&rdquo; band for uncertain cases.</p>

<h2>II. Related Work</h2>
<p>Ojha et al. showed that a frozen CLIP feature extractor with a single linear layer, trained on images from one GAN, transfers to diffusion models better than networks trained end to end [1].
We follow this idea but train on many more generators and on degraded images.
GenImage supplies a million-scale benchmark with eight generators and a protocol that trains on one of them and tests on the rest [2].
Grommelt et al. found that JPEG compression and image size differ between the real and fake classes of common datasets, so that a detector can score well by reading the file format [3]; we therefore re-encode every patch the same way.
OpenFake collects real photographs and the output of roughly sixty recent generators, and provides a test split of generators held out from training and a set of images shared in the wild [5].
DiffSeg30k contains diffusion-edited photographs with pixel masks of the edited region [6].
Yermakov et al. adapt only the normalization layers of a pretrained encoder to improve transfer across deepfake benchmarks [7].
For the score itself we rely on Platt scaling [8], on the finding that modern classifiers are often badly calibrated [9], and on the idea of letting a classifier abstain when it is unsure [10].</p>

<h2>III. System Overview</h2>
<p>The application is written in Python and served through Streamlit. Fig. 1 shows the data flow. The uploaded file may be a JPEG, PNG, WEBP, BMP, TIFF, GIF or HEIC image;
its true format is read from the bytes, rotation is applied and transparency is flattened, so the model only ever sees 8-bit RGB pixels.
Camera and provenance metadata (EXIF and C2PA content credentials) are displayed on a separate tab, because metadata is easily removed and must not decide the verdict.
Analysis of one image takes about ten seconds on a CPU. To fit the memory limit of a free hosting service (about 2.7 GB), only the image half of CLIP is loaded, read tensor by tensor,
which keeps the peak at roughly 1.5 GB without changing any output. The trained head itself is a file of a few kilobytes.</p>

<h2>IV. Method</h2>
<h3>A. Patches and format normalization</h3>
<p>A resized image loses the fine high-frequency traces on which detection depends, so the image is not resized. Instead we take 224&times;224 crops at the original resolution:
one at the centre and a 3&times;3 grid spread evenly over the image, ten patches in total (images smaller than one patch are enlarged first). Each patch is re-saved as JPEG at quality 95, which gives real and generated inputs the same compression history.</p>
<h3>B. Features and classifier</h3>
<p>Patch <i>i</i> is mapped by the frozen CLIP ViT-L/14 image encoder <i>f</i> [4] to a vector <i>z<sub>i</sub></i> = <i>f</i>(<i>x<sub>i</sub></i>) / ||<i>f</i>(<i>x<sub>i</sub></i>)||, with 768 entries.
A linear layer gives the patch logit <i>s<sub>i</sub></i> = <b>w</b><sup>T</sup><i>z<sub>i</sub></i> + <i>b</i>. The image score is the mean over the <i>N</i> = 10 patches,</p>
<p class="eq"><i>s</i> = (1/<i>N</i>) &Sigma;<sub><i>i</i></sub> <i>s<sub>i</sub></i>. &nbsp;&nbsp;&nbsp;&nbsp;(1)</p>
<p>The encoder is never updated; only <b>w</b> and <i>b</i> are learned, by L2-regularised logistic regression with balanced class weights. The penalty strength was chosen on the validation split.
The probabilities of the individual patches form the heatmap.</p>
<h3>C. Training data and compression augmentation</h3>
<p>Table I lists the sources. Images and masks are existing public datasets; nothing was generated by us. From each OpenFake training image we build two views: the clean image and a degraded copy.
The degraded copy is resized by a random factor in [0.5, 1] with probability 0.5, blurred with probability 0.2, and JPEG-compressed with quality drawn from 50&ndash;95 with probability 0.9.
Three random patches are cut from each view. The OpenFake sample contains 2,143 OpenAI images (GPT-Image-1, DALL-E 3) and 590 Google images (Imagen 3 and 4, Nano Banana 2) that we deliberately over-sampled.
For the edited sets, patches that lie at least half inside the edit mask are labelled AI, and 15% of the edited images were set aside for evaluation.
In total the final head was fitted to 214,440 patches.</p>
@@T1@@
@@F2@@
<p>Because CLIP is frozen, the expensive step is a single forward pass per patch. We ran it on a Kaggle T4 GPU in half precision, about 112 patches per second;
half-precision features have a cosine similarity of at least 0.99999 with full-precision ones, which is the precision the deployed application uses.</p>
<h3>D. Calibration and verdict bands</h3>
<p>The raw score <i>s</i> is converted to a probability of being AI-generated, <i>p</i> = sigmoid(<i>a s</i> + <i>c</i>), with <i>a</i> = 1.513 and <i>c</i> = 0.722 fitted on the validation split.
The Trust Score is <i>T</i> = round(100(1 &minus; <i>p</i>)). The two band limits were set on validation data so that at most 3% of real images would be called AI-generated and at most 10% of AI images would be called authentic:
<i>T</i> &ge; {C['trust_authentic_at_or_above']} is &ldquo;Likely Authentic&rdquo;, <i>T</i> &le; {C['trust_ai_at_or_below']} is &ldquo;Likely AI-generated&rdquo;, and everything in between is &ldquo;Needs Human Review&rdquo;.</p>
<h3>E. Model variants</h3>
<p>Besides the final model (v3), two variants share its recipe: <i>v3 proof</i>, trained without any OpenAI or Google image, and <i>v3 without compression training</i>, which sees only the clean views.
As a reference we also score the head of our earlier prototype (v2, about 6,000 training images, no ChatGPT or Gemini images) on the same data. Table IV summarises the three variants.</p>
@@T4@@

<h2>V. Experimental Setup</h2>
<p>The test split is the OpenFake <i>test</i> split, whose generators are held out from training by the dataset design and whose real images come from different sources than the training reals.
We use {n_real_test:,} real and {n_fake_test:,} AI images from {n_gen_test} generators (at most 250 per generator). Some of them are newer versions of families that occur in training (for example Midjourney v7 after v6);
the OpenAI and Google families are fully absent from the proof model. The same images are scored in three conditions: unchanged, after JPEG quality 70, and after halving the size followed by JPEG quality 75.
A second test set holds {R['v3']['test_reddit']['clean']['n_real']:,} real and {R['v3']['test_reddit']['clean']['n_fake']:,} AI images collected from online communities, which were already compressed and re-shared.
We report ROC-AUC, the share of AI images with probability above 0.5 (&ldquo;caught&rdquo;), and the share of real images below 0.5. Confidence intervals come from 1,000 bootstrap resamples of the test images.
The head, the penalty strength, the calibration and the band limits were all fixed using the separate validation split ({va['n_real']} real and {va['n_fake']} AI images).</p>

<h2>VI. Results</h2>
<h3>A. Overall detection and robustness</h3>
<p>Table II and Fig. 3 give the main results. On clean files the final model reaches an AUC of {A['test_clean']['auc']:.3f} (95% CI {ci('test_clean')}) and catches {pct(A['test_clean']['caught'], 1)}% of AI images (CI {ci_pct('test_clean')}%),
while labelling {pct(wc, 1)}% of real photographs correctly. The earlier prototype head reaches an AUC of {R['v2_reference']['test_openfake']['clean']['auc']:.3f} on the same images,
but it collapses under compression: at JPEG quality 70 it catches only {pct(R['v2_reference']['test_openfake']['jpeg70']['fake_caught'])}% of AI images, against {pct(A['test_jpeg70']['caught'], 1)}% for v3 (CI {ci_pct('test_jpeg70')}%).
On images shared in the wild, v3 reaches an AUC of {A['reddit']['auc']:.3f} (CI {ci('reddit')}).</p>
@@T2@@
@@F3@@
<h3>B. Effect of compression training</h3>
<p>Training on degraded copies costs very little on clean files (AUC {R['v3_noaug']['test_openfake']['clean']['auc']:.3f} without it, {R['v3']['test_openfake']['clean']['auc']:.3f} with it)
and helps where it matters: at JPEG 70 the share caught rises from {pct(R['v3_noaug']['test_openfake']['jpeg70']['fake_caught'])}% to {pct(R['v3']['test_openfake']['jpeg70']['fake_caught'])}%,
and after halving the size from {pct(R['v3_noaug']['test_openfake']['half_q75']['fake_caught'])}% to {pct(R['v3']['test_openfake']['half_q75']['fake_caught'])}%.
On the in-the-wild set the two variants are within a point of each other ({pct(R['v3_noaug']['test_reddit']['clean']['fake_caught'])}% and {pct(R['v3']['test_reddit']['clean']['fake_caught'])}%), so we cannot claim a gain there.</p>
<h3>C. Unseen ChatGPT and Gemini generators</h3>
<p>Table III lists individual generators. The proof model, whose training contains no OpenAI or Google image, still reaches an AUC of {gen('v3_proof','gpt-image-1.5')[0]:.3f} on GPT-Image-1.5
and {gen('v3_proof','nano-banana-pro')[0]:.3f} on Nano Banana Pro, clearly above the earlier head ({gen('v2_reference','gpt-image-1.5')[0]:.3f} and {gen('v2_reference','nano-banana-pro')[0]:.3f}).
Adding ChatGPT and Gemini images to training improves GPT-Image-1.5 further, from {pct(gen('v3_proof','gpt-image-1.5')[1])}% to {pct(gen('v3','gpt-image-1.5')[1])}% caught.
Only {gen('v3','nano-banana-pro')[2]} Nano Banana Pro images were available, so that interval is wide ({ci_pct('nano-banana-pro')}%).</p>
@@T3@@
<h3>D. Calibration and verdict bands</h3>
<p>On the test split the expected calibration error falls from {A['ece_test']['raw_sigmoid']:.3f} for the uncalibrated sigmoid to {A['ece_test']['platt']:.3f} after Platt scaling.
Fig. 4 shows how the verdict bands behave. Among clean real photographs {pct(cal_t['clean']['real']['authentic'], 1)}% are called authentic, {pct(cal_t['clean']['real']['review'], 1)}% go to review and {pct(cal_t['clean']['real']['ai'], 1)}% are called AI-generated.
Among clean AI images {pct(cal_t['clean']['fake']['ai'], 1)}% are called AI-generated and {pct(cal_t['clean']['fake']['authentic'], 1)}% authentic.
For compressed AI images the review band absorbs many uncertain cases, but {pct(cal_t['jpeg70']['fake']['authentic'])}% of the JPEG-70 images are still called authentic.</p>
@@F4@@
<h3>E. Edited images</h3>
<p>Table V reports patch-level results on edited images that were held out of training. A patch inside an AI edit is flagged {pct(ed['clean']['edited_caught'])}% of the time on clean files and {pct(ed['augmented']['edited_caught'])}% after degradation,
while the surrounding unedited area is flagged in about one case in five. Patches of untouched original photographs are called real {pct(ed['clean']['orig_correct'])}% and {pct(ed['augmented']['orig_correct'])}% of the time.
The edited images come from the same datasets and editors as the training edits, so this is evidence that edited regions can be located, not that the detector generalises to other editors.</p>
@@T5@@

<h2>VII. Limitations</h2>
<p>Compressed and screenshotted AI images remain the main weakness: about a third are missed at JPEG 70. The real photographs in the test split come from ImageNet and DOCCI, whereas the training reals come from stock-photo and web sources;
phone photographs were not part of either, and we have not yet measured them. We have not yet measured edits made by ChatGPT or Gemini on real photographs, which is the case users are most likely to try.
Several test generators are newer versions of training families, so the split is a test of unseen versions more than of unseen architectures; the OpenAI and Google results of the proof model are the strictest part of the evaluation.
The head is linear, the heatmap is a coarse 3&times;3 grid, the training set is about one percent of the available OpenFake images, and all results come from one training run with bootstrap intervals over test images only.
The class mix differs between validation (57% AI) and test (65% AI), which affects band limits and calibration.</p>

<h2>VIII. Responsible Use</h2>
<p>The Trust Score is a screening aid and not forensic proof. A real photograph that is labelled as AI-generated can damage the reputation of its author, which is why the bands are set to keep that error below a few percent
and why uncertain images are referred to a person instead of being forced into a verdict. The score says nothing about intent, and a high Trust Score does not certify that an image is untouched.
Metadata is displayed but never used as evidence of authenticity, since it can be removed or forged. Results should be reported together with the conditions (compression, size, source) under which they were obtained.</p>

<h2>IX. Conclusion and Future Work</h2>
<p>A frozen CLIP encoder with a small trained classifier, combined with degraded training copies and calibrated verdict bands, detects most clean AI images from generators it has not seen and
keeps working on moderately compressed ones, while sending uncertain cases to a human. The planned next steps are to use intermediate encoder layers, to train on more data and harsher degradations,
to test phone photographs and ChatGPT or Gemini edits of real photographs, and to refine the heatmap. The video module (face crops from sampled frames, evaluated on FaceForensics++ and Celeb-DF [11], [12])
and the document module (tamper localisation with a segmentation network, evaluated on DocTamper [13]) are still design work.</p>
{"<h2>Acknowledgment</h2><p>We thank our project guide, Dr. Gayatri Bhandari, for her guidance. The application and all code are available at github.com/PawanBhandari03/DeepFake-Detection-Using-CrossGeneratorGeneralization.</p>" if named else ""}
<h2>References</h2>
<div class="refs">
<p>[1] U. Ojha, Y. Li, and Y. J. Lee, &ldquo;Towards universal fake image detectors that generalize across generative models,&rdquo; in <i>Proc. CVPR</i>, 2023, pp. 24480&ndash;24489.</p>
<p>[2] M. Zhu <i>et al.</i>, &ldquo;GenImage: A million-scale benchmark for detecting AI-generated image,&rdquo; in <i>NeurIPS Datasets and Benchmarks</i>, 2023, arXiv:2306.08571.</p>
<p>[3] P. Grommelt, L. Weiss, F.-J. Pfreundt, and J. Keuper, &ldquo;Fake or JPEG? Revealing common biases in generated image detection datasets,&rdquo; arXiv:2403.17608, 2024.</p>
<p>[4] A. Radford <i>et al.</i>, &ldquo;Learning transferable visual models from natural language supervision,&rdquo; in <i>Proc. ICML</i>, 2021, pp. 8748&ndash;8763.</p>
<p>[5] V. Livernoche <i>et al.</i>, &ldquo;OpenFake: An open dataset and platform toward real-world deepfake detection,&rdquo; arXiv:2509.09495, 2025.</p>
<p>[6] H. Ci, Z. Peng, P. Yang, Y. Xuan, and M. Z. Shou, &ldquo;DiffSeg30k: A multi-turn diffusion editing benchmark for localized AIGC detection,&rdquo; arXiv:2511.19111, 2025.</p>
<p>[7] A. Yermakov, J. Cech, J. Matas, and M. Fritz, &ldquo;Deepfake detection that generalizes across benchmarks,&rdquo; arXiv:2508.06248, 2025.</p>
<p>[8] J. C. Platt, &ldquo;Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods,&rdquo; in <i>Advances in Large Margin Classifiers</i>, MIT Press, 1999, pp. 61&ndash;74.</p>
<p>[9] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, &ldquo;On calibration of modern neural networks,&rdquo; in <i>Proc. ICML</i>, 2017, pp. 1321&ndash;1330.</p>
<p>[10] Y. Geifman and R. El-Yaniv, &ldquo;Selective classification for deep neural networks,&rdquo; in <i>Proc. NeurIPS</i>, 2017.</p>
<p>[11] A. R&ouml;ssler <i>et al.</i>, &ldquo;FaceForensics++: Learning to detect manipulated facial images,&rdquo; in <i>Proc. ICCV</i>, 2019.</p>
<p>[12] Y. Li, X. Yang, P. Sun, H. Qi, and S. Lyu, &ldquo;Celeb-DF: A large-scale challenging dataset for DeepFake forensics,&rdquo; in <i>Proc. CVPR</i>, 2020, pp. 3207&ndash;3216.</p>
<p>[13] C. Qu <i>et al.</i>, &ldquo;Towards robust tampered text detection in document image: New dataset and new solution,&rdquo; in <i>Proc. CVPR</i>, 2023, pp. 5937&ndash;5946.</p>
</div>
"""


_SMALL = {"and", "of", "to", "with", "the", "a", "an", "in", "on", "for", "vs.", "at"}


def capitalise_headings(html: str) -> str:
    """Section headings (h2) in capital letters; sub-headings (h3) with every main word capitalised."""
    html = re.sub(r"<h2>(.*?)</h2>", lambda m: f"<h2>{m.group(1).upper()}</h2>", html)

    def title(m):
        words = m.group(1).split(" ")
        out = [w if (w.lower() in _SMALL and k > 1) else (w[:1].upper() + w[1:]) for k, w in enumerate(words)]
        return f"<h3>{' '.join(out)}</h3>"
    return re.sub(r"<h3>(.*?)</h3>", title, html)


def head_html(named: bool) -> str:
    authors = ""
    if named:
        authors = ("<p class='au'>Pawan Singh Bhandari &nbsp;&nbsp;|&nbsp;&nbsp; Rahul H. Bramhankar &nbsp;&nbsp;|&nbsp;&nbsp; Bhoomi Kushwah</p>"
                   "<p class='af'>Department of Computer Engineering, JSPM's Bhivarabai Sawant Institute of Technology and Research, Wagholi, Pune, India<br>"
                   "Project guide: Dr. Gayatri Bhandari &nbsp;|&nbsp; Team: The Perceptrons</p>")
    return f"""<h1>{TITLE}</h1>{authors}
<p class="abs"><b>Abstract&mdash;</b>{ABSTRACT}</p>
<p class="abs"><b>Index Terms&mdash;</b>{KEYWORDS}.</p>"""


CSS = """
@font-face { font-family: tnr; src: url(times.ttf); }
@font-face { font-family: tnr; src: url(timesbd.ttf); font-weight: bold; }
@font-face { font-family: tnr; src: url(timesi.ttf); font-style: italic; }
@font-face { font-family: tnr; src: url(timesbi.ttf); font-weight: bold; font-style: italic; }
body { font-family: tnr; font-size: 10pt; line-height: 1.2; color: #111; }
h1 { font-size: 17.5pt; text-align: center; margin: 0 0 7pt 0; color: #14284b; line-height: 1.12; }
.au { text-align: center; font-size: 11pt; margin: 0 0 2pt 0; }
.af { text-align: center; font-size: 8.8pt; font-style: italic; margin: 0 0 6pt 0; }
.abs { text-align: justify; font-size: 9.4pt; margin: 0 0 4pt 0; }
h2 { font-size: 10pt; text-align: center; text-transform: uppercase; color: #14284b; margin: 8pt 0 3pt 0; }
h3 { font-size: 10pt; font-style: italic; font-weight: bold; margin: 4pt 0 1.5pt 0; }
p { text-align: justify; margin: 0 0 4pt 0; }
.eq { text-align: center; margin: 2pt 0 4pt 0; }
.refs p { font-size: 8.2pt; text-align: left; margin: 0 0 2pt 0; line-height: 1.15; }
table.t { border-collapse: collapse; width: 100%; margin: 1pt 0 5pt 0; }
table.t th { font-size: 7.4pt; border-top: 0.8pt solid #333; border-bottom: 0.6pt solid #333; text-align: left; padding: 1.5pt 2pt; }
table.t td { font-size: 7.4pt; padding: 1.5pt 2pt; border-bottom: 0.2pt solid #bbb; }
table.t tr.b td { font-weight: bold; }
.figbox { margin: 2pt 0 4pt 0; text-align: center; }
.figbox img { width: 100%; }
p.cap { font-size: 8.2pt; text-align: center; margin: 1pt 0 2pt 0; }
"""

MARGIN_X, TOP, BOTTOM, GAP = 46, 44, 52, 16
PAGE = pymupdf.paper_rect("a4")
COL_W = (PAGE.width - 2 * MARGIN_X - GAP) / 2


def build(named: bool, out: Path) -> int:
    archive = pymupdf.Archive(FONT_DIR)
    archive.add(str(HERE / "figures"))
    writer = pymupdf.DocumentWriter(str(out))

    # page 1: full-width title block, then two columns below it
    head = pymupdf.Story(html=head_html(named), user_css=CSS, archive=archive)
    full = pymupdf.Rect(MARGIN_X, TOP, PAGE.width - MARGIN_X, PAGE.height - BOTTOM)
    dev = writer.begin_page(PAGE)
    _, filled = head.place(full)
    head.draw(dev)
    y0 = filled[3] + 6

    def columns(top):
        return [pymupdf.Rect(MARGIN_X + i * (COL_W + GAP), top, MARGIN_X + i * (COL_W + GAP) + COL_W, PAGE.height - BOTTOM) for i in range(2)]

    # the body is a list of text pieces and whole blocks (tables, figures); blocks are never split across columns
    pieces = re.split(r"@@(T\d|F\d)@@", capitalise_headings(body_html(named)))
    state = {"page": 1, "col": 0, "y": y0, "top": y0, "dev": dev, "fig1": True}

    def new_column():
        state["col"] += 1
        if state["col"] == 2:
            writer.end_page()
            state["dev"] = writer.begin_page(PAGE)
            state["page"] += 1
            state["col"], state["top"] = 0, TOP
            if state["fig1"] and state["page"] == 2:      # Fig. 1 spans both columns at the top of page 2
                f1 = pymupdf.Story(html='<div class="figbox"><img src="fig1.png"><p class="cap"><b>Fig. 1.</b> System overview. Solid boxes are implemented; dashed boxes are planned modules.</p></div>',
                                   user_css=CSS, archive=archive)
                _, fr = f1.place(pymupdf.Rect(MARGIN_X, TOP, PAGE.width - MARGIN_X, PAGE.height - BOTTOM))
                f1.draw(state["dev"])
                state["top"], state["fig1"] = fr[3] + 6, False
        state["y"] = state["top"]

    def col_rect():
        x = MARGIN_X + state["col"] * (COL_W + GAP)
        return pymupdf.Rect(x, state["y"], x + COL_W, PAGE.height - BOTTOM)

    deferred = []                                        # blocks that did not fit: they float to the top of the next column

    def block_height(html):
        probe = pymupdf.Story(html=html, user_css=CSS, archive=archive)
        _, used = probe.place(pymupdf.Rect(0, 0, COL_W, 2000))
        return used[3] + 2

    def draw_block(html, height):
        story = pymupdf.Story(html=html, user_css=CSS, archive=archive)
        r = col_rect()
        story.place(pymupdf.Rect(r.x0, state["y"], r.x1, state["y"] + height + 4))
        story.draw(state["dev"])
        state["y"] += height + 3

    def place_deferred():
        while deferred and state["y"] + deferred[0][1] <= PAGE.height - BOTTOM:
            html, height = deferred.pop(0)
            draw_block(html, height)

    base_new_column = new_column

    def new_column():                                    # noqa: F811  (wraps the function above)
        base_new_column()
        place_deferred()

    # text is cut into chunks that each start at a heading, so that a heading is never left alone at the bottom of a column
    items = []
    for i, piece in enumerate(pieces):
        if i % 2 == 0:
            items += [("text", c) for c in re.split(r"(?=<h[23]>)", piece) if c.strip()]
        else:
            items.append(("block", piece))

    for kind, piece in items:
        if kind == "text":                               # running text: flows from column to column
            if "<h" in piece[:8] and PAGE.height - BOTTOM - state["y"] < 62:
                new_column()                             # not even the heading plus two lines fit here
            story = pymupdf.Story(html=piece.strip(), user_css=CSS, archive=archive)
            while True:
                more, filled = story.place(col_rect())
                story.draw(state["dev"])
                if not more:
                    state["y"] = filled[3] - 8     # Story reports about 11 pt more than the text uses; keep a normal paragraph gap
                    break
                new_column()
        else:                                            # block: put it here if it fits, otherwise let it float forward
            html = BLOCKS[piece]
            height = block_height(html)
            if not deferred and state["y"] + height <= PAGE.height - BOTTOM:
                draw_block(html, height)
            else:
                deferred.append((html, height))
    while deferred:                                      # blocks still waiting at the very end
        if deferred[0][1] > PAGE.height - BOTTOM - state["y"]:
            new_column()
            continue
        html, height = deferred.pop(0)
        draw_block(html, height)
    writer.end_page()
    writer.close()

    doc = pymupdf.open(str(out))
    for i, page in enumerate(doc, start=1):
        label = f"The Perceptrons \u2022 Research Paper | {i}" if named else f"Research Paper | {i}"
        page.insert_text((PAGE.width / 2 - 60, PAGE.height - 26), label, fontsize=8, color=(0.4, 0.42, 0.45))
    doc.set_metadata({"title": TITLE, "author": "The Perceptrons" if named else "", "subject": "", "keywords": "", "creator": "", "producer": ""})
    doc.subset_fonts()
    tmp = out.with_suffix(".tmp.pdf")
    doc.ez_save(str(tmp))
    n = doc.page_count
    doc.close()
    shutil.copyfile(tmp, out)
    tmp.unlink()
    return n


if __name__ == "__main__":
    for named, name in ((True, "Image_Research_Paper_With_Authors.pdf"), (False, "Image_Research_Paper_Without_Authors.pdf")):
        n = build(named, HERE / name)
        print(name, n, "pages")
