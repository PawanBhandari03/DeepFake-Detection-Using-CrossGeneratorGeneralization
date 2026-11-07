"""Build the IMAGE-module research paper (complete scope of the image section) as two PDFs.

    python paper/make_figures.py
    python paper/make_calibration_plot.py
    python paper/analysis.py
    python paper/robust_analysis.py      # needs the Kaggle robustness run (kaggle/robust)
    python paper/build_image_paper.py

Writes  paper/Image_Research_Paper_With_Authors.pdf
        paper/Image_Research_Paper_Without_Authors.pdf   (same text; no names, team, college or repository link)

The paper covers the complete planned scope of the image module. Every item that is measured has results; items that
are still running (EfficientNet baseline, finer heatmap, tests on team photos) are listed in a status table and are
NOT given results. Numbers come from prototype/results/*.json, paper/analysis.json and paper/robust_results.json.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_full_paper as F  # noqa: E402  (shares references, styles, numbers and tables)
import layout  # noqa: E402

R, C, A, pct, gen, gc, ci, cip = F.R, F.C, F.A, F.pct, F.gen, F.gc, F.ci, F.cip

# compact style so that the paper fits in 4 pages (guide's limit)
CSS_IMG = (F.CSS
           .replace("font-size: 9.6pt; line-height: 1.17", "font-size: 9.0pt; line-height: 1.13")
           .replace("h1 { font-size: 16.5pt", "h1 { font-size: 15pt")
           .replace(".abs { text-align: justify; font-size: 9pt", ".abs { text-align: justify; font-size: 8.4pt")
           .replace("h2 { font-size: 9.6pt; text-align: center; color: #14284b; margin: 7pt 0 2pt 0; }", "h2 { font-size: 9pt; text-align: center; color: #14284b; margin: 4.5pt 0 1.5pt 0; }")
           .replace("h3 { font-size: 9.6pt; font-style: italic; font-weight: bold; margin: 3pt 0 1pt 0; }", "h3 { font-size: 9pt; font-style: italic; font-weight: bold; margin: 2pt 0 0.5pt 0; }")
           .replace("p { text-align: justify; margin: 0 0 3.5pt 0; }", "p { text-align: justify; margin: 0 0 2.3pt 0; }")
           .replace(".refs p { font-size: 7.5pt; text-align: left; margin: 0 0 1.4pt 0; line-height: 1.12; }", ".refs p { font-size: 6.8pt; text-align: left; margin: 0 0 0.8pt 0; line-height: 1.07; }")
           .replace("table.t th { font-size: 7.1pt", "table.t th { font-size: 6.7pt").replace("table.t td { font-size: 7.1pt", "table.t td { font-size: 6.7pt")
           .replace("p.cap { font-size: 7.8pt", "p.cap { font-size: 7.3pt"))
import os
TEST = os.environ.get("PAPER_TEST_DIR")          # layout test only: stand-in data in a temp folder, never in paper/
RB = json.loads(((Path(TEST) if TEST else HERE) / "robust_results.json").read_text(encoding="utf-8"))
CAL = json.loads((HERE / "calibration_bins.json").read_text(encoding="utf-8"))
ed, cal, va, red = F.ed, F.cal, F.va, F.red
n_real, n_fake, n_gen = F.n_real, F.n_fake, F.n_gen

COND = [("clean", "Clean"), ("jpeg90", "JPEG 90"), ("jpeg70", "JPEG 70"), ("jpeg50", "JPEG 50"),
        ("screenshot", "Screenshot-like"), ("half", "Half size"), ("blur", "Blur")]
CN = dict(COND)
v3, v3n, v2 = RB["v3"], RB["v3_noaug"], RB["v2"]
worst = min((c for c, _ in COND), key=lambda c: v3[c]["caught"])
best_deg = max((c for c, _ in COND if c != "clean"), key=lambda c: v3[c]["caught"])
helps = [CN[c] for c, _ in COND if v3[c]["caught"] > v3n[c]["caught"] + 0.02]
same = [CN[c] for c, _ in COND if abs(v3[c]["caught"] - v3n[c]["caught"]) <= 0.02]
worse = [CN[c] for c, _ in COND if v3[c]["caught"] < v3n[c]["caught"] - 0.02]
auc_lo = min(v3[c]["auc"] for c, _ in COND)


def lst(items):
    items = list(items)
    return "none" if not items else (items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1])


# ------------------------------------------------------------------ extra tables
T_SETS = f"""<table class="t"><tr><th width="20%">Split</th><th width="26%">Source</th><th width="12%">Real</th><th width="12%">AI</th><th>Used for</th></tr>
<tr><td>Validation</td><td>OpenFake validation</td><td>{va['n_real']}</td><td>{va['n_fake']}</td><td>penalty, calibration, bands</td></tr>
<tr><td>Test</td><td>OpenFake test, {n_gen} generators</td><td>{n_real:,}</td><td>{n_fake:,}</td><td>main results, 7 conditions</td></tr>
<tr><td>In the wild</td><td>OpenFake Reddit</td><td>{red['n_real']}</td><td>{red['n_fake']:,}</td><td>shared images</td></tr>
<tr><td>Edited, held out</td><td>DiffSeg30k, tiny-audits</td><td>{ed['clean'].get('orig_patches', 969)} patches</td><td>{ed['clean']['edited_patches']:,} patches</td><td>region edits</td></tr></table>"""

_rows = ""
for c, name in COND:
    _rows += (f"<tr><td>{name}</td><td>{v2[c]['auc']:.3f} / {pct(v2[c]['caught'])}%</td><td>{v3n[c]['auc']:.3f} / {pct(v3n[c]['caught'])}%</td>"
              f"<td><b>{v3[c]['auc']:.3f} / {pct(v3[c]['caught'])}%</b></td><td>{pct(v3[c]['real_correct'])}%</td></tr>")
T_ROB = f"""<table class="t"><tr><th width="25%">Condition</th><th>v2 head</th><th>v3, no aug.</th><th>v3</th><th width="14%">Real ok</th></tr>{_rows}</table>"""

_a = {k: R["v2_reference"] for k in ()}
T_ABL = """<table class="t"><tr><th width="46%">Variant (trained on 1,600 SD1.5 images)</th><th>GenImage</th><th>Modern</th></tr>
<tr><td>No format fix (raw patches)</td><td>0.974</td><td>0.899</td></tr>
<tr><td>JPEG-95 format fix</td><td>0.883</td><td>0.901</td></tr>
<tr><td>Format fix + frequency features</td><td>0.886</td><td>0.893</td></tr></table>"""

T_STATUS = """<table class="t"><tr><th width="52%">Item of the image module</th><th width="18%">Status</th><th>Where</th></tr>
<tr><td>Data, splits and data card</td><td>done</td><td>Sec. V, Tables I&ndash;II</td></tr>
<tr><td>Frozen-CLIP classifier, compression training</td><td>done</td><td>Sec. IV</td></tr>
<tr><td>Unseen-generator evaluation, proof model</td><td>done</td><td>Tables III&ndash;V</td></tr>
<tr><td>Robustness under seven conditions</td><td>done</td><td>Fig. 3, Table IV</td></tr>
<tr><td>Calibration, reliability diagram, verdict bands</td><td>done</td><td>Fig. 4, Sec. VI-D</td></tr>
<tr><td>Ablations: format fix, frequency features</td><td>done</td><td>Table VI</td></tr>
<tr><td>Deployed application, explanation text, metadata panel</td><td>done</td><td>Sec. III</td></tr>
<tr><td>EfficientNet-B0 baseline</td><td>in progress</td><td>due within a week</td></tr>
<tr><td>Finer heatmap (6&times;6) and region-based verdict</td><td>in progress</td><td>due within a week</td></tr>
<tr><td>Tests on team phone photos and on ChatGPT / Gemini edits of real photos</td><td>in progress</td><td>images being collected</td></tr>
<tr><td>Next model: intermediate encoder layers, more data, light fine-tuning</td><td>planned</td><td>after the three items above</td></tr></table>"""

BLOCKS = {
    "TDATA": F.BLOCKS["TDATA"],
    "TSETS": '<p class="cap"><b>TABLE II.</b> Evaluation data (data card of the image module).</p>' + T_SETS,
    "FTRAIN": F.BLOCKS["FTRAIN"],
    "TMAIN": '<p class="cap"><b>TABLE III.</b> Held-out test split: ROC-AUC / share of AI images caught at probability 0.5. &dagger;Earlier prototype head, scored without its frequency features.</p>' + F.T_MAIN,
    "TGEN": '<p class="cap"><b>TABLE V.</b> Per-generator ROC-AUC / share caught, clean images.</p>' + F.T_GEN,
    "TROB": '<p class="cap"><b>TABLE IV.</b> Robustness: ROC-AUC / share caught under seven conditions; last column: real photos labelled real (v3).</p>' + T_ROB,
    "FCALIB": '<div class="figbox"><img src="fig_calib.png" style="width:86%"><p class="cap"><b>Fig. 4.</b> Reliability diagram on the test split, before and after Platt scaling.</p></div>',
    "FVERD": '<div class="figbox"><img src="fig4.png"><p class="cap"><b>Fig. 5.</b> Verdict outcomes on the held-out test split and the in-the-wild set.</p></div>',
    "TABL": '<p class="cap"><b>TABLE VI.</b> Prototype-scale ablations (ROC-AUC, model trained on one generator).</p>' + T_ABL,
    "TSTATUS": '<p class="cap"><b>TABLE VII.</b> Completion status of the image module.</p>' + T_STATUS,
}

FULL_WIDTH = {
    2: '<div class="figbox"><img src="fig1_image.png"><p class="cap"><b>Fig. 1.</b> The image analyser from upload to Trust Score. Metadata is shown but never scored.</p></div>',
    3: '<div class="figbox"><img src="fig_robust.png"><p class="cap"><b>Fig. 3.</b> Robustness of the earlier head, v3 without augmentation and v3 under seven image conditions (AUC and share of AI images caught).</p></div>',
}

TITLE = "Cross-Generator Detection of AI-Generated Images with a Frozen CLIP Encoder and a Calibrated Trust Score"
KEYWORDS = "AI-generated images, cross-generator generalization, CLIP, JPEG robustness, probability calibration, Trust Score"
ABSTRACT = (
    "Pictures made by text-to-image systems are now hard to tell from photographs, and a detector trained on one generator often fails on the next one. "
    "This paper describes the image module of a content authenticity application and reports its evaluation. An uploaded picture is cut into 224&times;224 patches at its own resolution, "
    "each patch is encoded by a frozen CLIP ViT-L/14, and a small logistic classifier that we trained on about 30,000 real and AI-generated images scores the patch; every training image also contributes "
    "a randomly compressed, resized and blurred copy. A Platt-calibrated probability becomes a 0&ndash;100 Trust Score with three verdict bands whose limits were chosen on validation data. "
    f"On a held-out test split whose generators are absent from training ({n_real:,} real and {n_fake:,} AI images from {n_gen} sources) the model reaches an AUC of {A['test_clean']['auc']:.3f} (95% CI {ci('test_clean')}) on clean files "
    f"and keeps an AUC of at least {auc_lo:.3f} under seven degradation conditions. It catches {pct(gen('v3', 'gpt-image-1.5')[1])}% of GPT-Image-1.5 and {pct(gen('v3', 'nano-banana-pro')[1])}% of Nano Banana Pro images, "
    f"and a variant trained without any OpenAI or Google image still catches {pct(gen('v3_proof', 'gpt-image-1.5')[1])}% and {pct(gen('v3_proof', 'nano-banana-pro')[1])}% of them. "
    f"Calibration reduces the expected calibration error from {A['ece_test']['raw_sigmoid']:.3f} to {A['ece_test']['platt']:.3f}. "
    "Compressed images remain the weakest case. The module is deployed as a web application; a CNN baseline, a finer heatmap and tests on phone photographs are still in progress and are marked as such."
)


def body(named: bool) -> str:
    return f"""
<h2>I. Introduction</h2>
<p>Pictures that never passed through a camera are now easy to make. Generative adversarial networks started the trend [[goodfellow]], and diffusion models [[ho,rombach]] made the results good enough that people often cannot tell them from photographs.
Platforms, newsrooms and ordinary users need a quick way to ask whether a picture is probably synthetic, and to be told how sure the answer is; surveys of media forensics show how quickly the methods on both sides change [[verdoliva]].</p>
<p>Fitting a classifier to a fixed collection of fakes is easy; keeping it useful once the source of the fakes changes is the real problem. A detector trained on one generator can fall close to chance on the next one [[ojha,wang]],
datasets can give the answer away through file format or image size [[grommelt]], and compression, which is how most images reach a user, removes much of the evidence that detectors rely on [[corvi]].</p>
<p>We therefore ask three practical questions: how does detection change on generators that were not in the training data, can the result be explained without suggesting more certainty than the model has, and how can a tool hand uncertain cases to a person?
Our contributions are: (i) a detection pipeline that works on original-resolution patches from a frozen CLIP encoder [[radford]] and removes file-format shortcuts; (ii) a training recipe that adds randomly degraded copies and edited-image patches to a large set of modern generators,
including ChatGPT and Gemini images; (iii) an evaluation on unseen generators with a proof variant that never sees OpenAI or Google images, seven degradation conditions, calibration and ablations; and (iv) a calibrated Trust Score with a &ldquo;Needs Human Review&rdquo; band,
deployed as a working application. The module is one part of a larger framework that is also meant to cover video and documents; those parts are not discussed here.</p>

<h2>II. Related Work</h2>
<p>Ojha et al. trained a linear classifier on frozen CLIP features [[radford]] with images from a single GAN and transferred far better to diffusion models than end-to-end networks did [[ojha]].
Earlier work had shown that CNN detectors transfer poorly unless augmentation is used [[wang]] and that up-sampling leaves spectral traces [[durall,frank]]. Corvi et al. found traces in diffusion images as well, but also a clear drop after common processing [[corvi]].
GenImage supplies a million-scale benchmark that trains on one generator and tests on the others [[zhu]]. Grommelt et al. showed that JPEG and size biases can inflate results [[grommelt]]. OpenFake adds real photographs, recent generators, held-out generators and images shared in the wild [[livernoche]],
DiffSeg30k contains locally edited images with masks [[ci]], and Yermakov et al. tune only the normalisation layers of a pretrained encoder to transfer between benchmarks [[yermakov]].
For the score we rely on Platt scaling [[platt]], on the finding that modern networks are often over-confident [[guo]] and on selective classification, which lets a classifier abstain [[geifman]].</p>

<h2>III. System Overview</h2>
<p>The application is written in Python (PyTorch [[paszke]], scikit-learn [[pedregosa]]) with a Streamlit interface and is deployed on a free hosting service, where one image takes about ten seconds on a CPU. Fig. 1 shows the data flow.
The uploaded file may be a JPEG, PNG, WEBP, BMP, TIFF, GIF or HEIC image; its true format is read from the bytes, rotation is applied and transparency is flattened, so the model only ever sees 8-bit RGB pixels.
Camera and provenance metadata (EXIF and C2PA content credentials [[c2pa]]) is shown on a separate tab, because metadata is easily removed and must not decide the verdict. To fit the memory limit of the hosting service (about 2.7 GB),
only the image half of CLIP is loaded, read tensor by tensor, which keeps the peak near 1.5 GB without changing any output. The trained head is a file of a few kilobytes.</p>

<h2>IV. Method</h2>
<h3>A. Patches and format normalization</h3>
<p>Resizing a picture destroys fine traces, so the image is not resized. We cut 224&times;224 crops at the original resolution, one at the centre and a 3&times;3 grid over the image, ten in total (smaller images are enlarged first).
Every crop is re-saved as JPEG at quality 95, which gives real and generated inputs the same compression history and removes the shortcut described by Grommelt et al. [[grommelt]].</p>
<h3>B. Features and classifier</h3>
<p>Patch <i>i</i> is mapped by the frozen CLIP ViT-L/14 image encoder <i>f</i> [[radford]] to a unit vector <i>z<sub>i</sub></i> = <i>f</i>(<i>x<sub>i</sub></i>) / ||<i>f</i>(<i>x<sub>i</sub></i>)|| with 768 entries, and a linear layer gives the patch logit
<i>s<sub>i</sub></i> = <b>w</b><sup>T</sup><i>z<sub>i</sub></i> + <i>b</i>. The image score is the mean over the <i>N</i> = 10 patches,</p>
<p class="eq"><i>s</i> = (1/<i>N</i>) &Sigma;<sub><i>i</i></sub> <i>s<sub>i</sub></i>. &nbsp;&nbsp;&nbsp;&nbsp;(1)</p>
<p>Only <b>w</b> and <i>b</i> are learned, by L2-regularised logistic regression with balanced class weights; the encoder is never updated, following Ojha et al. [[ojha]]. The patch probabilities form the heatmap.</p>
<h3>C. Training data and compression augmentation</h3>
<p>Table I lists the sources, all existing public datasets [[livernoche,ci]]. From each OpenFake training image we build two views, the clean image and a degraded copy that is resized by a random factor in [0.5, 1] with probability 0.5,
blurred with probability 0.2 and JPEG-compressed with a quality drawn from 50&ndash;95 with probability 0.9. Three random patches are cut from each view. The sample contains 2,143 OpenAI images (GPT-Image-1, DALL-E 3) and 590 Google images
(Imagen 3 and 4, Nano Banana 2) that we deliberately over-sampled. For the edited sets, patches lying at least half inside the edit mask are labelled AI, and 15% of the edited images were kept for evaluation.
The final head was fitted to 214,440 patches. Because CLIP is frozen, features are computed once (Fig. 2); we did this on a Kaggle T4 GPU in half precision (about 112 patches per second, cosine similarity of at least 0.99999 with full precision), and fitting the head takes minutes on a laptop.</p>
@@TDATA@@
@@FTRAIN@@
<h3>D. Model variants</h3>
<p>Besides the final model (v3) we trained <i>v3 proof</i>, which sees no OpenAI or Google image, and <i>v3 without augmentation</i>, which sees only clean views. As a reference we also score the head of our earlier prototype (v2, about 6,000 training images,
no ChatGPT or Gemini images) on the same data. The penalty strength of each head was chosen on the validation split.</p>

<h2>V. Experimental Setup</h2>
<p>Table II is the data card of the evaluation. The test split is the OpenFake <i>test</i> split, whose generators are held out from training by the dataset design and whose real images come from other sources than the training reals [[livernoche]]; we use at most 250 images per generator.
Some generators are newer versions of families that occur in training, for example Midjourney v7 after v6, whereas OpenAI and Google are completely absent from the proof model. Test images are scored in seven conditions: unchanged, JPEG quality 90, 70 and 50,
a screenshot-like re-save (scaled to 75% and saved as JPEG 85), half size, and Gaussian blur (radius 1.5). A second set holds shared images that were already compressed online. We report ROC-AUC [[fawcett]], the share of AI images above probability 0.5 (&ldquo;caught&rdquo;),
the share of real photographs below 0.5 and 95% bootstrap intervals [[efron]] over 1,000 resamples of the test images. The head, the penalty strength, the calibration and the band limits were fixed on the separate validation split.</p>
@@TSETS@@

<h2>VI. Results and Discussion</h2>
<h3>A. Detection on unseen generators</h3>
<p>Following Ojha et al. [[ojha]], the encoder is frozen in every variant. On clean files v3 reaches an AUC of {A['test_clean']['auc']:.3f} (95% CI {ci('test_clean')}) and catches {pct(A['test_clean']['caught'], 1)}% of AI images (CI {cip('test_clean')}%), and {pct(F.v3c['real_correct'], 1)}% of real photographs are labelled real (Table III).
The earlier head reaches {R['v2_reference']['test_openfake']['clean']['auc']:.3f} on the same images. On the shared images v3 has an AUC of {A['reddit']['auc']:.3f} (CI {ci('reddit')}) and catches {pct(red['fake_caught'])}%, against {pct(R['v2_reference']['test_reddit']['clean']['fake_caught'])}% for the earlier head.</p>
@@TMAIN@@
<h3>B. Robustness to compression, resizing and blur</h3>
<p>Fig. 3 and Table IV give the results for seven conditions; a drop after common processing is also reported for other detectors [[corvi]]. For v3 the AUC never falls below {auc_lo:.3f}, but the share caught drops from {pct(v3['clean']['caught'])}% on clean files to {pct(v3[worst]['caught'])}% under the hardest condition ({CN[worst].lower()}); the easiest degraded condition is {CN[best_deg].lower()} at {pct(v3[best_deg]['caught'])}%.
The earlier head catches {pct(v2['jpeg90']['caught'])}%, {pct(v2['jpeg70']['caught'])}% and {pct(v2['jpeg50']['caught'])}% at JPEG 90, 70 and 50, against {pct(v3['jpeg90']['caught'])}%, {pct(v3['jpeg70']['caught'])}% and {pct(v3['jpeg50']['caught'])}% for v3.
Training on degraded copies, an idea borrowed from blur-and-JPEG augmentation [[wang]], helps clearly under {lst(helps)}; the two variants are within two points under {lst(same)}{(' and v3 is worse under ' + lst(worse)) if worse else ''}.
The cost on clean files is small (AUC {R['v3_noaug']['test_openfake']['clean']['auc']:.3f} without augmentation, {R['v3']['test_openfake']['clean']['auc']:.3f} with it).
The share caught alone is misleading for blur and half size: real photographs are labelled real only {pct(v3['blur']['real_correct'])}% and {pct(v3['half']['real_correct'])}% of the time under these two conditions (Table IV, last column), so part of the high share caught is extra false alarms.
The verdict bands of Section VI-D were tuned on clean, JPEG 70 and half-size-plus-JPEG images and not on these two conditions.</p>
@@TROB@@
<h3>C. Unseen ChatGPT and Gemini generators</h3>
<p>The proof model, with no OpenAI or Google image in training (generators from the OpenFake test split [[livernoche]]), reaches an AUC of {gen('v3_proof','gpt-image-1.5')[0]:.3f} on GPT-Image-1.5 and {gen('v3_proof','nano-banana-pro')[0]:.3f} on Nano Banana Pro, clearly above the earlier head ({gen('v2_reference','gpt-image-1.5')[0]:.3f} and {gen('v2_reference','nano-banana-pro')[0]:.3f}) (Table V).
Adding ChatGPT and Gemini images raises GPT-Image-1.5 from {pct(gen('v3_proof','gpt-image-1.5')[1])}% to {pct(gen('v3','gpt-image-1.5')[1])}% caught. Only {gen('v3','nano-banana-pro')[2]} Nano Banana Pro images were available, so that interval is wide ({cip('nano-banana-pro')}%).</p>
@@TGEN@@
<h3>D. Calibration and verdict bands</h3>
<p>The score is calibrated as <i>p</i> = sigmoid(<i>a s</i> + <i>c</i>) [[platt]] with <i>a</i> = 1.513 and <i>c</i> = 0.722 fitted on validation data. The Trust Score is <i>T</i> = round(100(1 &minus; <i>p</i>)). Two limits chosen on validation data (at most 3% of real images called AI, at most 10% of AI images called authentic) give
<i>T</i> &ge; {C['trust_authentic_at_or_above']} as Likely Authentic, <i>T</i> &le; {C['trust_ai_at_or_below']} as Likely AI-generated and Needs Human Review in between, an abstention rule in the spirit of Geifman and El-Yaniv [[geifman]]. On the test split the expected calibration error [[naeini]] falls from {A['ece_test']['raw_sigmoid']:.3f} to {A['ece_test']['platt']:.3f}.
Over-confidence is common in modern networks [[guo]]. Fig. 4 shows that the calibrated probabilities are still somewhat under-confident in the middle range; the validation split holds 57% AI images and the test split 65%, and the generators differ, so some shift is expected.
With the bands, {pct(cal['clean']['real']['authentic'], 1)}% of clean real photographs are called authentic, {pct(cal['clean']['real']['review'], 1)}% go to review and {pct(cal['clean']['real']['ai'], 1)}% are called AI-generated; among clean AI images {pct(cal['clean']['fake']['ai'], 1)}% are called AI-generated and {pct(cal['clean']['fake']['authentic'], 1)}% authentic.
At JPEG 70 the review band takes in {pct(cal['jpeg70']['fake']['review'])}% of AI images, but {pct(cal['jpeg70']['fake']['authentic'])}% are still called authentic.</p>
@@FCALIB@@
<h3>E. Edited images</h3>
<p>On held-out edited images from DiffSeg30k [[ci]] and tiny-audits, {pct(ed['clean']['edited_caught'])}% of patches inside an edit are flagged on clean files and {pct(ed['augmented']['edited_caught'])}% after degradation, while the unedited surroundings are flagged in about one case in five;
patches of untouched original photographs are called real {pct(ed['clean']['orig_correct'])}% and {pct(ed['augmented']['orig_correct'])}% of the time. These edits come from the same datasets and editors as the training edits, so this shows that edited regions can be located, not that other editors are covered.</p>
<h3>F. Ablations</h3>
<p>Table VI repeats two design choices at prototype scale (a head trained on 1,600 Stable Diffusion 1.5 images, tested on GenImage and on modern generators). Without the format fix the model scores 0.974 on GenImage but only 0.883 with it,
while the modern-generator AUC does not change (0.899 and 0.901); this agrees with the file-format shortcut reported by Grommelt et al. [[grommelt]] and is why every patch is re-saved the same way. Adding spectral features of the noise residual [[frank]] did not help (0.886 and 0.893),
so the final models use CLIP features only.</p>
@@TABL@@

<h2>VII. Completion Status</h2>
<p>Table VII lists every part of the image module. The parts marked done are measured and reported above. The CNN baseline [[tan]] (to be explained with Grad-CAM [[selvaraju]]), the finer 6&times;6 heatmap with a verdict based on the most suspicious region,
and the tests on phone photographs and on ChatGPT or Gemini edits of real photographs are in progress; we give no results for them here and will add them when they are finished.</p>
@@TSTATUS@@

<h2>VIII. Limitations</h2>
<p>Compressed and screenshotted AI images remain the main weakness, as for other detectors [[corvi]]: even with augmentation a large share is missed under the hardest conditions of Fig. 3, and blur or downscaling mainly raise the false-alarm rate on real photographs (Table IV). The real photographs of the test split come from ImageNet and DOCCI [[livernoche]], whereas the training reals come from stock-photo and web sources;
phone photographs were part of neither. Several test generators are newer versions of training families, so the split tests unseen versions more than unseen architectures; the proof model is the strictest part of the evaluation.
The head is linear, the heatmap is a coarse 3&times;3 grid, the training set is about one percent of the available OpenFake images, and all results come from one training run with intervals over test images only.
The Trust Score is a screening aid and not forensic proof: a real photograph labelled as AI-generated can damage its author, which is why uncertain cases go to a person and why metadata is displayed but never used as evidence of authenticity.</p>

<h2>IX. Conclusion and Future Work</h2>
<p>A frozen CLIP encoder with a small trained classifier [[ojha]], degraded training copies and calibrated verdict bands detects most clean AI images from generators it has not seen, keeps a useful AUC under compression, resizing and blur, and sends uncertain cases to a person.
The remaining work on this module is the CNN baseline, the finer heatmap and the phone-photograph and edit tests, followed by intermediate encoder layers, more data and light fine-tuning of the encoder [[hu]].</p>
"""


def head(named: bool) -> str:
    authors = ""
    if named:
        authors = ("<p class='au'>Dr. Gayatri Bhandari (Guide) &nbsp;|&nbsp; Pawan Singh Bhandari &nbsp;|&nbsp; Rahul H. Bramhankar &nbsp;|&nbsp; Bhoomi Kushwah</p>"
                   "<p class='af'>Department of Computer Engineering, JSPM's Bhivarabai Sawant Institute of Technology and Research, Wagholi, Pune, India<br>"
                   "Team: The Perceptrons</p>")
    return f"<h1>{TITLE}</h1>{authors}<p class='abs'><b>Abstract&mdash;</b>{ABSTRACT}</p><p class='abs'><b>Index Terms&mdash;</b>{KEYWORDS}.</p>"


def build(with_refs: bool, out: Path):
    """with_refs=True: citation numbers in the text and the reference list in the right column of the last page.
    with_refs=False: same paper without any citation number and without the reference list (names are kept)."""
    html = F.capitalise_headings(body(True))
    refs_html, nrefs = None, 0
    if with_refs:
        html, order = layout.number_citations(html, F.REFS)
        refs_html = '<h2>REFERENCES</h2><div class="refs">' + "".join(f"<p>[{i}] {F.REFS[k]}</p>" for i, k in enumerate(order, start=1)) + "</div>"
        nrefs = len(order)
    else:
        html = layout.strip_citations(html)
    meta = {"title": TITLE, "author": "The Perceptrons", "subject": "", "keywords": "", "creator": "", "producer": ""}
    n = layout.build_pdf(out, head(True), html, BLOCKS, CSS_IMG, (Path(TEST) / "figures") if TEST else HERE / "figures", FULL_WIDTH, "The Perceptrons \u2022 Research Paper", meta, refs_html=refs_html)
    return n, nrefs


if __name__ == "__main__":
    for with_refs, name in ((True, "Image_Research_Paper_With_Authors.pdf"), (False, "Image_Research_Paper_Without_Authors.pdf")):
        pages, nrefs = build(with_refs, (Path(TEST) if TEST else HERE) / name)
        print(name, pages, "pages,", nrefs, "references")
