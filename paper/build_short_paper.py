"""Build the 4-page whole-project paper (requirements, design, algorithms, workflow; short results).

    python paper/make_figures.py
    python paper/make_full_figures.py
    python paper/analysis.py
    python paper/robust_analysis.py
    python paper/build_short_paper.py

Writes  paper/Final_Paper_4pages_With_Authors.pdf
        paper/Final_Paper_4pages_Without_Authors.pdf   (same text; no names, team, college or repository link)

Guide's instruction: at most 4 pages, the whole project, focus on the process. Results are only for the image analyser
(the only implemented module); video and document modules are described as designs.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_full_paper as F  # noqa: E402
import layout  # noqa: E402

R, C, A, pct, gen, ci = F.R, F.C, F.A, F.pct, F.gen, F.ci
RB = json.loads((HERE / "robust_results.json").read_text(encoding="utf-8"))
v3 = RB["v3"]
n_real, n_fake, n_gen = F.n_real, F.n_fake, F.n_gen

TITLE = "An Explainable System for Screening Images, Videos and Documents: Requirements, Design and Cross-Generator Detection"
KEYWORDS = "system design, requirements, deepfake detection, document tamper localization, CLIP, calibration, cross-generator generalization"

ABSTRACT = (
    "Realistic synthetic images, manipulated face videos and edited documents come from different tools, yet the person who receives such a file needs one answer that can be understood. "
    "This paper describes the requirements, the design and the development process of a screening system that accepts any of the three file types, routes it to a specialised analyser and converts the evidence "
    "into a calibrated Trust Score, a verdict with a human-review band and a plain-language explanation. We present the requirements taken from the software specification, a layered architecture with one common evidence interface, "
    "and the algorithms of the three analysers. The image analyser is implemented: a frozen CLIP encoder with a classifier trained on degraded copies of images from modern generators. "
    f"On held-out generators it reaches an AUC of {A['test_clean']['auc']:.3f} and catches {pct(gen('v3', 'gpt-image-1.5')[1])}% of GPT-Image-1.5 images, and a variant that never saw an OpenAI or Google image still catches {pct(gen('v3_proof', 'gpt-image-1.5')[1])}%. "
    "The video and document analysers are fully specified and under construction; no results are claimed for them."
)

T_REQ = """<div class="reqbox">
<p><span class="id">FR1</span> Check every upload by its file signature; accept the usual image, video and PDF or scan formats and refuse damaged files with a clear message.</p>
<p><span class="id">FR2</span> Images: estimate the chance of AI generation from full-resolution patches (frozen CLIP encoder plus a trained classifier) and draw a heatmap.</p>
<p><span class="id">FR3</span> Videos: sample frames, find faces, score every frame and merge the scores into one result with a timeline.</p>
<p><span class="id">FR4</span> Documents: run OCR and forensic maps (ELA, SRM, DCT), segment edits with a dual-stream U-Net, and add logic and PDF-structure checks.</p>
<p><span class="id">FR5</span> Turn the probability <i>p</i> into a Trust Score 100(1 &minus; <i>p</i>) and a verdict: Authentic, Manipulated, or Needs Human Review.</p>
<p><span class="id">FR6</span> Show a visual and sentences built only from measured evidence; display file metadata apart from the score.</p>
<p><span class="id">FR7</span> Offer the user a report to view and download.</p>
<p><span class="id">NFR</span> CPU only, under 2.7 GB of memory, about 5 s per image, no uploads kept, calibration error at most 0.05, and an AUC target on unseen sources for every module.</p>
</div>"""

_rows = [
    ("All 20 unseen generators", f"{v3['clean']['auc']:.3f}", f"{pct(v3['clean']['caught'])}%"),
    ("ChatGPT, GPT-Image-1.5", f"{gen('v3', 'gpt-image-1.5')[0]:.3f}", f"{pct(gen('v3', 'gpt-image-1.5')[1])}%"),
    ("Gemini, Nano Banana Pro", f"{gen('v3', 'nano-banana-pro')[0]:.3f}", f"{pct(gen('v3', 'nano-banana-pro')[1])}%"),
    ("Proof model, ChatGPT", f"{gen('v3_proof', 'gpt-image-1.5')[0]:.3f}", f"{pct(gen('v3_proof', 'gpt-image-1.5')[1])}%"),
    ("Proof model, Gemini", f"{gen('v3_proof', 'nano-banana-pro')[0]:.3f}", f"{pct(gen('v3_proof', 'nano-banana-pro')[1])}%"),
]
T_RES = ('<table class="plain"><tr><th width="60%">Test images</th><th>AUC</th><th>Caught</th></tr>'
         + "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in _rows) + "</table>")

T_ALG = """<div class="alg"><p class="algh">Algorithm 1 &nbsp; Screening procedure</p>
<p><span class="n">1</span> <b>validate</b>(file) &mdash; read the type from the bytes, check size, decode; reject if corrupt</p>
<p><span class="n">2</span> analyser &larr; <b>route</b>(type) &mdash; image, video or document</p>
<p><span class="n">3</span> evidence &larr; analyser.<b>analyse</b>(file) &mdash; score, visual, cues, flags</p>
<p><span class="n">4</span> <i>p</i> &larr; <b>calibrate</b>(<b>fuse</b>(evidence)); &nbsp;<i>T</i> &larr; round(100(1 &minus; <i>p</i>))</p>
<p><span class="n">5</span> verdict &larr; <b>band</b>(<i>T</i>) &mdash; Authentic, Needs Human Review or Manipulated</p>
<p><span class="n">6</span> report &larr; <b>explain</b>(evidence, <i>T</i>, verdict); show metadata separately</p>
</div>"""

BLOCKS = {
    "TREQ": '<p class="cap"><b>Requirements</b> (summarised from the SRS)</p>' + T_REQ,
    "TALG": T_ALG,
    "TRES": f'<p class="cap"><b>TABLE I.</b> Image analyser on {n_real:,} real and {n_fake:,} AI test images from {n_gen} unseen generators (95% CI of the clean AUC {ci("test_clean")}).</p>' + T_RES,
}
CM = json.loads((HERE / "cm_roc.json").read_text(encoding="utf-8"))
FULL_WIDTH = {
    2: ('<div class="figbox"><img src="fig_arch_bw.png" style="width:90%"><p class="cap"><b>Fig. 1.</b> Layered architecture. Solid boxes are implemented; dashed boxes are designed.</p></div>'),
    3: '<div class="figbox"><img src="fig_vd_plain.png" style="width:88%"><p class="cap"><b>Fig. 2.</b> Designed pipelines: (a) video, (b) document (ELA = error-level analysis, SRM = noise residual, DCT = frequency features).</p></div>',
}
BLOCKS["FCM"] = ('<div class="figbox"><img src="fig_cm_roc_col.png" style="width:74%"><p class="cap"><b>Fig. 3.</b> Image analyser on the held-out test split (clean files, threshold 0.5): '
                 f'confusion matrix and ROC curve for {CM["n_real"]:,} real and {CM["n_fake"]:,} AI images from unseen generators.</p></div>')


def body(named: bool) -> str:
    return f"""
<h2>I. Introduction</h2>
<p>Synthetic images are now easy to produce [[rombach]], face-swapped video is common [[rossler]], and a figure on an invoice can be changed in minutes. Reviewing such files by eye is unreliable, and automatic detectors tend to be tuned to the sources they were built on:
a detector trained on one generator can fall close to chance on the next [[ojha,wang]], datasets can leak the answer through file format [[grommelt]], and compression removes much of the evidence [[corvi]].
Research is organised by medium; ForensicHub brings several of them into one benchmark [[forensichub]], but a benchmark is not a tool for a non-specialist.</p>
<p>We describe the requirements, design and development process of such a tool. Our contributions are: (i) a requirements-driven design in which images, videos and documents are handled by independent analysers behind one evidence interface;
(ii) a calibrated scoring layer with a human-review band shared by all analysers; (iii) the algorithms of the three analysers, with the image analyser, which uses degraded training copies and a model variant that never sees OpenAI or Google images, implemented and evaluated on unseen generators;
and (iv) an evaluation method built around unseen sources. We do not claim to be the first unified system.</p>

<h2>II. Related Work</h2>
<p>Ojha et al. trained a linear classifier on frozen CLIP features [[radford]] with images from one GAN and transferred well to diffusion models [[ojha]]; GenImage [[zhu]] and OpenFake [[livernoche]] provide cross-generator benchmarks, the latter with recent generators and images shared online.
For face video, FaceForensics++ [[rossler]] and Celeb-DF [[li2020]] are the standard training and transfer sets, and faces are located with MTCNN [[zhang]].
Document tampering is studied with DocTamper [[qu]]; two-stream networks combine RGB with noise features [[zhou,fridrich]], error-level analysis [[krawetz]] is a classical cue, and U-Net [[ronneberger]] with a Dice term [[milletari]] is the usual segmentation setup.
Probabilities of neural networks are often over-confident [[guo]]; Platt scaling [[platt]] and selective classification [[geifman]] motivate our calibrated score and review band.</p>

<h2>III. Requirements Analysis</h2>
<p>The software requirements specification defines seven functional requirements (listed below) and acceptance targets: an unseen-generator AUC of at least 0.80 for images above an EfficientNet baseline [[tan]], FaceForensics++ and Celeb-DF AUC targets for video, pixel F1 for documents,
and a calibration error of at most 0.05. Non-functional requirements decide the design: inference must run on a CPU inside the 2.7 GB memory limit of free hosting, uploads are processed in memory and never stored, and every verdict must come with evidence.
Provenance metadata such as EXIF fields and C2PA credentials [[c2pa]] is displayed but never scored, because it is easy to remove or forge.</p>
@@TREQ@@

<h2>IV. System Design</h2>
<p>Fig. 1 shows the architecture. A Streamlit interface receives the upload; the validation layer reads the true file type from the bytes, applies size limits and decodes the file; a router calls the image, video or document analyser. Each analyser implements the same interface and returns an evidence record:
a score, a visual (heatmap, timeline or mask), numeric cues and flags raised by rules. A scoring layer fuses and calibrates the record and an explanation layer writes the sentences. Because analysers share only this interface, a model can be replaced without touching the application,
and the three phases of the project (image, video, document) can be built one after the other.</p>
<p>Training and inference are separated. Heavy work, such as extracting features from hundreds of thousands of patches, runs once on a free cloud GPU; the application loads the frozen encoder and a trained head of a few kilobytes and runs on a CPU (PyTorch [[paszke]], scikit-learn [[pedregosa]]).
Loading only the image half of CLIP, tensor by tensor, keeps the peak memory near 1.5 GB. The processing flow is summarised in Algorithm 1; a typical use case is a user who uploads a photograph and receives a Trust Score, a heatmap of the most suspicious regions and the sentences that explain them.</p>

@@TALG@@

<h2>V. Analysis Modules</h2>
<h3>A. Image analyser (implemented)</h3>
<p>The image is not resized, because resizing removes fine traces. Ten 224&times;224 patches (centre and a 3&times;3 grid) are cut at native resolution and re-saved as JPEG quality 95, so real and generated inputs share one compression history [[grommelt]].
Each patch is mapped by the frozen CLIP ViT-L/14 encoder <i>f</i> to <i>z</i> = <i>f</i>(<i>x</i>)/||<i>f</i>(<i>x</i>)|| and scored by a linear head <i>s</i> = <b>w</b><sup>T</sup><i>z</i> + <i>b</i>; the image score is the mean of the patch scores, and the patch scores form the heatmap.
Only <b>w</b> and <i>b</i> are trained, by L2-regularised logistic regression, on about 30,000 images of real photographs and roughly sixty generators, including ChatGPT and Gemini images [[livernoche]], and edited-image patches [[ci]].
Every training image also contributes a randomly resized, blurred and JPEG-compressed copy, an idea taken from blur-and-JPEG augmentation [[wang]]. A <i>proof</i> variant is trained without any OpenAI or Google image, so that detection of ChatGPT and Gemini images is a genuinely unseen-source test.</p>
<h3>B. Video analyser (designed)</h3>
<p>About one frame per second is sampled (at most 32; Fig. 2a); MTCNN [[zhang]] finds and aligns the faces, each crop gets a margin so that the blending boundary stays visible [[lilyu]], and every crop is scored, first by the image analyser unchanged and then by a head trained on FaceForensics++ [[rossler]].
Frame scores are summarised by averaging only the top-scoring share of frames, which keeps a short manipulated segment from being diluted by clean ones, and they also give a timeline. Splits are made by video, never by frame, and transfer is tested on Celeb-DF [[li2020]].</p>
<h3>C. Document analyser (designed)</h3>
<p>Pages are rendered and read by OCR [[smith]] (Fig. 2b). Error-level analysis [[krawetz]], spatial-rich-model noise residuals [[fridrich]] and DCT features are stacked as a second input stream next to the RGB page of a dual-stream U-Net [[ronneberger,zhou]] with an EfficientNet encoder [[tan]],
trained with binary cross-entropy plus a Dice term [[milletari]] on synthetic forgeries (text replacement, copy-move, splicing) made from clean public documents with exact masks and scanner-like noise, and tested on human-made forgeries and DocTamper [[qu]].
Rule checks (line items against the total, dates, font heights, PDF modification history) add flags that are auxiliary evidence, never proof.</p>
<h3>D. Trust Score and review band</h3>
<p>A raw score <i>s</i> becomes a probability <i>p</i> = sigmoid(<i>a s</i> + <i>c</i>) by Platt scaling [[platt]] fitted on validation data; several signals are fused by a small logistic regression. The Trust Score is <i>T</i> = round(100(1 &minus; <i>p</i>)).
Two limits chosen on validation data (at most 3% of real items called manipulated, at most 10% of manipulated items called authentic) divide <i>T</i> into Likely Authentic, Needs Human Review and Likely Manipulated. For the image analyser they are <i>T</i> &ge; {C['trust_authentic_at_or_above']} and <i>T</i> &le; {C['trust_ai_at_or_below']};
the calibration error [[naeini]] fell from {A['ece_test']['raw_sigmoid']:.3f} to {A['ece_test']['platt']:.3f} on the test split.</p>

<h2>VI. Data Design and Training Workflow</h2>
<p>Every analyser is built in the same way. Public datasets with documented splits are collected; each item is cut into patches or frames at native resolution; features of the frozen encoder are computed once on a free GPU and stored with their label, source and view type;
the small head is fitted on a laptop in minutes; the penalty strength, the calibration and the verdict limits are chosen on a validation split; and a held-out test split is scored once. For images the sources are OpenFake [[livernoche]] for real photographs and about sixty generators and DiffSeg30k [[ci]] for locally edited photographs with masks.
For the case in which only part of a picture is machine-made, public data of real photographs re-edited with GPT-Image-1 exists [[gptedit]] and is planned as the next training and test source. Splits follow the unit of the problem: by generator for images, by video for video and by document for documents.
Because features are stored, a new head or a new calibration costs minutes instead of hours.</p>

<h2>VII. Evaluation Protocol</h2>
<p>Each module is judged on sources unseen in training: the held-out generators of the OpenFake test split for images (at most 250 images per generator), Celeb-DF after training on FaceForensics++ for video, and human-made forgeries for documents, as in the GenImage protocol [[zhu]].
We report ROC-AUC [[fawcett]], the share of manipulated items caught, the share of real items accepted, 95% bootstrap intervals [[efron]] and the expected calibration error [[naeini]]. A variant trained without OpenAI and Google images makes ChatGPT and Gemini detection a true unseen-source test;
the baselines are a fine-tuned EfficientNet [[tan]] and an earlier prototype head. Image analysis takes about ten seconds on a CPU.</p>

<h2>VIII. Results and Discussion</h2>
<p>The project follows a 26-week plan in three phases, each ending with a working demonstration. Every phase is evaluated on sources that were not used in training: another generator for images, another dataset for video and real forgeries for documents.
The image analyser is implemented, deployed as a web application and evaluated; Table I gives its results on clean images from {n_gen} unseen generators, with {pct(v3['clean']['real_correct'])}% of real photographs correctly accepted. Fig. 3 shows the confusion matrix and the ROC curve of the same test: {CM['tn']:,} of {CM['n_real']:,} real photographs are accepted and {CM['tp']:,} of {CM['n_fake']:,} AI images are caught (accuracy {CM['accuracy'] * 100:.1f}%).
An EfficientNet-B0 baseline trained on the same images reaches an AUC of 0.831 on this test, against 0.982 for our analyser. For real photographs edited by ChatGPT or Gemini, the team test below gives AUCs of 0.95 and 0.92; compressed or resized images remain a harder case left for later work. The video and document analysers are specified in this paper and will be built next.</p>
@@FCM@@
@@TRES@@
<p><b>Discussion.</b> Three findings stand out. First, the proof variant, which never saw an OpenAI or Google image, still separates ChatGPT and Gemini images from real photographs (Table I), so what the head learns is shared by many generators and is not a memory of one company's output; adding such images to training helps a little more. Second, the confusion matrix of Fig. 3 shows that errors are not symmetric: {CM['fn']} AI images are missed against {CM['fp']} real photographs flagged, and the calibrated review band is meant to catch many of both. Third, robustness is the open weakness: with compression training the AUC stays at {min(v3[c]['auc'] for c in v3):.2f} or higher under seven degradations, but at JPEG quality 50 only {pct(v3['jpeg50']['caught'])}% of AI images are caught, and blur or downscaling make real photographs look synthetic ({pct(v3['blur']['real_correct'])}% and {pct(v3['half']['real_correct'])}% are still accepted).</p>
<p>A test on full-size photographs taken by our team supports this: the edit scored as more AI-like than its original in all 10 ChatGPT pairs and all 47 Gemini pairs (AUC 0.95 and 0.92), and 38 of 47 originals were accepted, but 28 of 47 Gemini edits went to the review band. The sets are small. When only a region of a photograph is machine-made, the average over patches dilutes the signal; scoring only the most suspicious patches did not help enough to adopt.</p>

<h2>IX. Limitations and Responsible Use</h2>
<p>Results exist for the image analyser only, from one training run, and the linear head and the 3&times;3 heatmap are coarse. Several test generators are newer versions of training families [[livernoche]], so the proof variant is the strictest test.
Compression and resizing weaken all detectors [[corvi]] and are left for later work, and the file-format shortcut [[grommelt]] is removed by re-saving every patch in the same way. The Trust Score is a screening aid and not forensic proof;
a real photograph labelled as AI-generated can harm its author, which is why uncertain cases go to a person.</p>

<h2>X. Conclusion</h2>
<p>Requirements taken from a specification led to a layered design with independent analysers, a shared evidence interface and a calibrated, explainable scoring layer. The image analyser shows that a frozen CLIP encoder [[ojha]] with a small head trained on degraded copies detects most clean AI images from unseen generators,
while the video and document analysers are specified for the next phases.</p>
"""


CSS = (F.CSS
       .replace("font-size: 9.6pt; line-height: 1.17", "font-size: 9.8pt; line-height: 1.17")
       .replace("h1 { font-size: 16.5pt", "h1 { font-size: 15pt")
       .replace(".abs { text-align: justify; font-size: 9pt", ".abs { text-align: justify; font-size: 8.6pt")
       .replace("h2 { font-size: 9.6pt; text-align: center; color: #14284b; margin: 7pt 0 2pt 0; }", "h2 { font-size: 9.8pt; text-align: center; color: #14284b; margin: 6pt 0 2pt 0; }")
       .replace("h3 { font-size: 9.6pt; font-style: italic; font-weight: bold; margin: 3pt 0 1pt 0; }", "h3 { font-size: 9.8pt; font-style: italic; font-weight: bold; margin: 2.5pt 0 0.5pt 0; }")
       .replace("p { text-align: justify; margin: 0 0 3.5pt 0; }", "p { text-align: justify; margin: 0 0 3pt 0; }")
       .replace(".refs p { font-size: 7.5pt; text-align: left; margin: 0 0 1.4pt 0; line-height: 1.12; }", ".refs p { font-size: 7.3pt; text-align: left; margin: 0 0 1.1pt 0; line-height: 1.1; }")
       .replace("table.t th { font-size: 7.1pt", "table.t th { font-size: 6.9pt").replace("table.t td { font-size: 7.1pt", "table.t td { font-size: 6.9pt"))


CSS += """
.reqlist p { text-align: left; font-size: 8.4pt; margin: 0 0 1.4pt 0; line-height: 1.12; }
table.plain { border-collapse: collapse; width: 100%; margin: 1pt 0 4pt 0; }
table.plain th { font-size: 8pt; border-top: 0.8pt solid #222; border-bottom: 0.6pt solid #222; text-align: left; padding: 1.5pt 3pt; }
table.plain td { font-size: 8pt; padding: 1.2pt 3pt; }
table.plain tr:last-child td { border-bottom: 0.8pt solid #222; }
.reqbox p { text-align: left; font-size: 8.8pt; margin: 0 0 2pt 0; line-height: 1.14; }
.reqbox .id { font-weight: bold; }
.alg { border-top: 0.8pt solid #000; border-bottom: 0.8pt solid #000; margin: 3pt 0 5pt 0; padding: 0 2pt 2pt 2pt; }
.alg p { text-align: left; font-size: 8.6pt; margin: 0 0 1.5pt 0; }
.alg .algh { font-weight: bold; border-bottom: 0.5pt solid #000; margin-bottom: 2.5pt; padding: 2pt 0 1.5pt 0; }
.alg .n { font-weight: bold; }
table.side { width: 100%; }
table.side td { text-align: center; vertical-align: top; }
"""


def head(named: bool) -> str:
    authors = ""
    if named:
        authors = ("<p class='au'>Dr. Gayatri Bhandari (Guide) &nbsp;|&nbsp; Pawan Singh Bhandari &nbsp;|&nbsp; Rahul H. Bramhankar &nbsp;|&nbsp; Bhoomi Kushwah</p>"
                   "<p class='af'>Department of Computer Engineering, JSPM's Bhivarabai Sawant Institute of Technology and Research, Wagholi, Pune, India<br>"
                   "Team: The Perceptrons</p>")
    return f"<h1>{TITLE}</h1>{authors}<p class='abs'><b>Abstract&mdash;</b>{ABSTRACT}</p><p class='abs'><b>Index Terms&mdash;</b>{KEYWORDS}.</p>"


def add_reference_links(pdf_path: Path, order: list) -> int:
    """Make every reference in the list a clickable link. Text, colours and layout are not changed."""
    import shutil
    import pymupdf
    from ref_links import REF_URLS
    import re
    doc = pymupdf.open(str(pdf_path))
    page = doc[-1]
    lines = []                                    # (rect, text) of every text line of the reference column
    for blk in page.get_text("dict")["blocks"]:
        for ln in blk.get("lines", []):
            r = pymupdf.Rect(ln["bbox"])
            if r.x0 > page.rect.width / 2 - 5:
                lines.append((r, "".join(s["text"] for s in ln["spans"]).strip()))
    lines.sort(key=lambda x: (round(x[0].y0), x[0].x0))
    start = next((k for k, (r, s) in enumerate(lines) if s.upper().startswith("REFERENCES")), -1)
    groups, cur = [], None
    for r, s in lines[start + 1:]:
        m = re.match(r"\[(\d+)\]", s)
        if m:
            cur = [int(m.group(1)), pymupdf.Rect(r)]
            groups.append(cur)
        elif cur is not None and r.y0 > page.rect.height - 60:
            break                                  # footer line
        elif cur is not None:
            cur[1] |= r
    done = 0
    for num, rect in groups:
        if 1 <= num <= len(order):
            page.insert_link({"kind": pymupdf.LINK_URI, "from": rect, "uri": REF_URLS[order[num - 1]]})
            done += 1
    tmp = pdf_path.with_suffix(".tmp.pdf")
    doc.ez_save(str(tmp))
    doc.close()
    shutil.copyfile(tmp, pdf_path)
    tmp.unlink()
    return done


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
    n = layout.build_pdf(out, head(True), html, BLOCKS, CSS, HERE / "figures", FULL_WIDTH, "The Perceptrons \u2022 Research Paper", meta, refs_html=refs_html)
    if with_refs:
        linked = add_reference_links(out, order)
        print(f"{linked} of {nrefs} references are clickable links")
    return n, nrefs


if __name__ == "__main__":
    for with_refs, name in ((True, "Final_Paper_4pages_With_Authors.pdf"), (False, "Final_Paper_4pages_Without_Authors.pdf")):
        pages, nrefs = build(with_refs, HERE / name)
        print(name, pages, "pages,", nrefs, "references")
