"""Build the final project presentation in the department's format.

    python paper/reports/presentation/build_pptx.py <template.pptx> [output.pptx]

The template is the department's sample deck (brain-tumour project): the script keeps its theme, layouts, logos,
fonts and slide order, and fills it with this project's content. Diagrams come from make_diagrams.py (diagrams/),
screenshots from shared/figs/app_*.png. Every slide carries speaker notes that explain it in plain words.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from lxml import etree

HERE = Path(__file__).resolve().parent
DIAG = HERE / "diagrams"
FIGS = HERE.parent / "shared" / "figs"
PAPER_FIGS = HERE.parent.parent / "figures"

TEMPLATE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\Users\bhand\Downloads\BE\Final Project Presentation.pptx")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "Final_Project_Presentation.pptx"

prs = Presentation(str(TEMPLATE))
SW, SH = Emu(prs.slide_width).inches, Emu(prs.slide_height).inches

# ------------------------------------------------------------------ logos from the template
_logo = {}
for sh in prs.slides[1].shapes:
    if sh.shape_type == 13:
        _logo["left" if sh.left < Inches(2) else "right"] = (sh.image.blob, sh.left, sh.top, sh.width, sh.height)


def add_logos(slide):
    for key in ("left", "right"):
        blob, l, t, w, h = _logo[key]
        slide.shapes.add_picture(io.BytesIO(blob), l, t, w, h)


# ------------------------------------------------------------------ remove the sample's slides except the first
sld_id_lst = prs.slides._sldIdLst
for sld in list(sld_id_lst)[1:]:
    prs.part.drop_rel(sld.rId)
    sld_id_lst.remove(sld)


# ------------------------------------------------------------------ helpers
def notes(slide, text: str) -> None:
    slide.notes_slide.notes_text_frame.text = text


def set_title(slide, text: str, size: int = 36) -> None:
    t = slide.shapes.title
    t.text_frame.text = text
    for p in t.text_frame.paragraphs:
        for r in p.runs:
            r.font.name = "Times New Roman"
            r.font.size = Pt(size)
            r.font.bold = True


def new_slide(title: str, size: int = 36):
    if size == 36:
        size = 36 if len(title) <= 20 else (32 if len(title) <= 25 else 28)
    s = prs.slides.add_slide(prs.slide_layouts[1])
    set_title(s, title, size)
    add_logos(s)
    return s


def fill_body(slide, items, size: int = 20, left=0.5, top=1.85, width=9.0, height=5.3, bold_first: bool = False):
    body = slide.placeholders[1]
    body.left, body.top, body.width, body.height = Inches(left), Inches(top), Inches(width), Inches(height)
    tf = body.text_frame
    tf.word_wrap = True
    first = True
    for it in items:
        level, text = (it if isinstance(it, tuple) else (0, it))
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = level
        r = p.add_run()
        r.text = text
        r.font.size = Pt(size - 2 * level)
    return body


def content_slide(title, items, size=20, note="", **kw):
    s = new_slide(title)
    fill_body(s, items, size, **kw)
    if note:
        notes(s, note)
    return s


def picture_slide(title, image: Path, note="", caption="", box=(0.4, 1.7, 9.2, 5.4), title_size=36):
    s = new_slide(title, title_size)
    ph = s.placeholders[1]
    ph._element.getparent().remove(ph._element)          # no empty content box behind the picture
    im = Image.open(image)
    bl, bt, bw, bh = box
    if caption:
        bh -= 0.45
    scale = min(bw / im.width, bh / im.height)
    w, h = im.width * scale, im.height * scale
    s.shapes.add_picture(str(image), Inches(bl + (bw - w) / 2), Inches(bt + (bh - h) / 2), Inches(w), Inches(h))
    if caption:
        tb = s.shapes.add_textbox(Inches(0.5), Inches(bt + bh + 0.05), Inches(9.0), Inches(0.4))
        p = tb.text_frame.paragraphs[0]
        p.alignment = 2
        r = p.add_run()
        r.text = caption
        r.font.size = Pt(14)
        r.font.name = "Times New Roman"
    if note:
        notes(s, note)
    return s


def set_cell(cell, text, size=14, bold=False, align=None):
    cell.text = ""
    tf = cell.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.name = "Times New Roman"
    if align is not None:
        p.alignment = align
    cell.margin_left = cell.margin_right = Inches(0.06)
    cell.margin_top = cell.margin_bottom = Inches(0.03)


def table_slide(title, header, rows, widths, size=14, note="", top=1.7, row_h=0.5, title_size=36):
    s = new_slide(title, title_size)
    ph = s.placeholders[1]
    ph._element.getparent().remove(ph._element)
    shape = s.shapes.add_table(len(rows) + 1, len(header), Inches(0.42), Inches(top), Inches(sum(widths)), Inches(row_h * (len(rows) + 1)))
    tbl = shape.table
    # plain grid style (black lines, no coloured fill) as in the sample
    tblPr = tbl._tbl.tblPr
    for el in tblPr.findall(qn("a:tableStyleId")):
        el.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"
    for el in ("firstRow", "bandRow"):
        tblPr.set(el, "0")
    for j, w in enumerate(widths):
        tbl.columns[j].width = Inches(w)
    for j, h in enumerate(header):
        set_cell(tbl.cell(0, j), h, size, True, 2)
    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            set_cell(tbl.cell(i, j), v, size, False, 2 if len(v) <= 10 else None)
    if note:
        notes(s, note)
    return s


def slide_crop(src: Path, box, name: str) -> Path:
    """Crop an app screenshot to the part that matters on a slide (keeps the original for the reports)."""
    im = Image.open(src).convert("RGB")
    l, t_, r, b = box
    out = HERE / "diagrams" / f"slide_{name}.png"
    im.crop((l, t_, min(r, im.width), min(b, im.height))).save(out, optimize=True)
    return out


# ------------------------------------------------------------------ preprocessing picture (made from a real photograph)
def make_prep_image(src: Path, dst: Path) -> None:
    try:
        font = ImageFont.truetype("times.ttf", 34)
        small = ImageFont.truetype("times.ttf", 28)
    except OSError:
        font = small = ImageFont.load_default()
    im = Image.open(src).convert("RGB")
    W, H = im.size
    canvas = Image.new("RGB", (1600, 900), "white")
    d = ImageDraw.Draw(canvas)
    d.text((20, 8), "Ten patches per image: centre + 3 x 3 grid", fill="black", font=font)
    view_h = 780
    sc = view_h / H
    photo = im.resize((int(W * sc), view_h))
    canvas.paste(photo, (20, 70))
    P = 224
    xs = [round(i * (W - P) / 2) for i in range(3)]
    ys = [round(i * (H - P) / 2) for i in range(3)]
    for yy in ys:
        for xx in xs:
            d.rectangle([20 + xx * sc, 70 + yy * sc, 20 + (xx + P) * sc, 70 + (yy + P) * sc], outline=(30, 90, 200), width=3)
    cx, cy = (W - P) // 2, (H - P) // 2
    d.rectangle([20 + cx * sc, 70 + cy * sc, 20 + (cx + P) * sc, 70 + (cy + P) * sc], outline=(230, 120, 0), width=5)
    d.text((820, 8), "The same 224 x 224 patch in three versions", fill="black", font=font)
    patch = im.crop((cx, cy, cx + P, cy + P))

    def jpeg(img, q):
        b = io.BytesIO()
        img.save(b, "JPEG", quality=q)
        b.seek(0)
        return Image.open(b).convert("RGB")

    from PIL import ImageFilter
    v1 = jpeg(patch, 95)
    v2 = jpeg(patch.resize((P // 2, P // 2), Image.BILINEAR).resize((P, P), Image.BILINEAR), 70)
    v3 = jpeg(patch.filter(ImageFilter.GaussianBlur(1.6)), 50)
    labels = ["Training copy 1:\nclean, JPEG q95", "Training copy 2:\nhalf size, JPEG q70", "Training copy 3:\nblurred, JPEG q50"]
    for k, (v, lab) in enumerate(zip((v1, v2, v3), labels)):
        x0 = 820 + k * 260
        canvas.paste(v.resize((240, 240), Image.NEAREST), (x0, 80))
        d.multiline_text((x0, 335), lab, fill="black", font=small, spacing=4)
    d.text((820, 520), "Each training image gives a clean and a randomly damaged\ncopy, so the head also works on shared, compressed files.", fill="black", font=small, spacing=6)
    canvas.save(dst, optimize=True)


# ================================================================== slide 1: title (edit the sample's own slide)
s1 = prs.slides[0]


def set_par_text(par, text):
    runs = par.runs
    runs[0].text = text
    for r in runs[1:]:
        r.text = ""


for sh in s1.shapes:
    if not sh.has_text_frame:
        continue
    pars = sh.text_frame.paragraphs
    if sh.name == "Title 1":
        set_par_text(pars[3], "“DEEPFAKE DETECTION WITH CROSS‑GENERATOR GENERALIZATION”")
        for r in pars[3].runs:
            r.font.size = Pt(28)
        for p in (pars[1], pars[2]):
            for r in p.runs:
                r.font.size = Pt(26)
        for p in (pars[4], pars[0]):                      # drop the empty first line and the old second title line
            p._p.getparent().remove(p._p)
    elif sh.name.startswith("Subtitle") and pars[0].runs[0].text.startswith("Guided"):
        pars[0].runs[0].text = "Guided By "
        pars[0].runs[1].text = ": Dr. G. M. Bhandari"
        for r in pars[0].runs[2:]:
            r.text = ""
    elif sh.name.startswith("Subtitle"):
        names = ["Pawan Singh Jagdish Bhandari (               )", "Rahul H. Bramhankar (               )", "Bhoomi Kushwah (               )"]
        for i, n in enumerate(names, start=1):
            set_par_text(pars[i], n)
        for p in pars[4:]:
            p._p.getparent().remove(p._p)
notes(s1, "Introduce the project: a web system that checks whether an image, video or document is real, AI-generated or tampered with. "
          "Phase 1 (images) is complete; video and document analysers are designed. Team: The Perceptrons. Guide: Dr. G. M. Bhandari.")

# ================================================================== 2 contents
content_slide("Introduction", [
    "Problem Statement", "Introduction", "Objectives", "Literature Survey", "System Architecture", "UML Diagrams", "Algorithm",
    "Data Preprocessing and Augmentation", "CLIP Encoder and Trained Head", "System Requirements", "Project Plan", "Results", "Test Cases",
    "Applications", "Conclusion", "Future Enhancement", "References"], size=15,
    note="Outline of the talk. The order follows the department format: problem, background, design diagrams, algorithm, results, tests, conclusion.",
    top=2.2, height=5.0, left=1.2, width=8.7)
# the sample shows a small 'Content :' label under the title on this slide
cb = prs.slides[1].shapes.add_textbox(Inches(1.2), Inches(1.55), Inches(3.4), Inches(0.7))
r = cb.text_frame.paragraphs[0].add_run()
r.text = "Content :"
r.font.name, r.font.size, r.font.bold = "Times New Roman", Pt(28), True

# ================================================================== 3 problem statement
content_slide("Problem Statement", [
    "To develop a web-based, explainable system that checks whether an image, a video or a document is authentic, AI-generated or tampered with, "
    "and that keeps working on generators that were not seen in training."], size=22, top=1.9, height=3.0,
    note="One sentence: one tool, three kinds of files, a score people can understand, and it must work on new AI tools, not only the ones it was trained on.")

# ================================================================== 4 introduction
content_slide("Introduction", [
    "AI tools such as Stable Diffusion, Midjourney, ChatGPT and Gemini now make pictures that look like photographs.",
    "Face-swapped videos and edited documents are also easy to make, and checking them by eye is unreliable.",
    "Many detectors only work for the generators they were trained on and fail on new ones.",
    "Thus one explainable screening system that works on unseen generators would be helpful."],
    size=20, note="Why the project matters. The key difficulty is that a detector trained on one AI tool can fail on the next one.")

# ================================================================== 5 objectives
content_slide("Objectives", [
    "To detect AI-generated images, including images from generators that were not seen in training.",
    "To give every result as a Trust Score (0 to 100), a verdict and a Needs Human Review band.",
    "To explain every verdict with a heatmap and plain-language sentences.",
    "To run on a CPU within 2.7 GB of memory and never store uploaded files.",
    "Next phases: face-manipulated videos and tampered documents."],
    size=20, top=1.95, note="Five objectives. The first four are achieved for images in Phase 1; video and documents are the next two phases.")

# ================================================================== 6-8 literature survey
LIT = [
    ("01", "CNN-generated images are surprisingly easy to spot... for now", "S.-Y. Wang", "2020",
     "A CNN trained on one GAN, with blur and JPEG augmentation, detects other GANs; weaker on other generator families."),
    ("02", "Towards Universal Fake Image Detectors that Generalize Across Generative Models", "U. Ojha", "2023",
     "A linear classifier on frozen CLIP features transfers better to unseen generators; whole-image score only."),
    ("03", "On the detection of synthetic images generated by diffusion models", "R. Corvi", "2023",
     "Diffusion-model images can be detected, but detectors weaken after resizing and compression."),
    ("04", "Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets", "P. Grommelt", "2024",
     "Some datasets can be solved from file format and compression instead of image content."),
    ("05", "GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image", "M. Zhu", "2023",
     "Cross-generator benchmark built from eight generators."),
    ("06", "OpenFake: An Open Dataset and Platform Toward Real-World Deepfake Detection", "V. Livernoche", "2025",
     "Real photographs and about sixty generators, with held-out test generators; used in our training."),
]
for k in range(3):
    table_slide("Literature Survey", ["Sr. No.", "Paper Name", "Authors", "Publication Year", "Conclusion"],
                [list(r) for r in LIT[2 * k:2 * k + 2]], [0.55, 2.45, 1.35, 1.2, 3.2], size=14, row_h=0.9, title_size=36,
                note="Related work. The common gap: results on the training generators are good, results on new generators and on compressed files are not. "
                     "Our design answers this with CLIP features, degraded training copies and tests on unseen generators.")

# ================================================================== 9 system architecture
s = picture_slide("System Architecture", DIAG / "fig_architecture.png", box=(1.2, 1.55, 7.6, 4.55),
                  note="Layered architecture. The upload is validated and routed to the image, video or document analyser. All analysers return the same evidence record, "
                       "which is calibrated into a Trust Score and explained. Solid boxes are built (image); dashed boxes are designed (video, document).")
tb = s.shapes.add_textbox(Inches(0.6), Inches(6.2), Inches(8.8), Inches(0.9))
tf = tb.text_frame
tf.word_wrap = True
r = tf.paragraphs[0].add_run()
r.text = "Dataset link: "
r.font.size, r.font.name, r.font.bold = Pt(16), "Times New Roman", True
r = tf.paragraphs[0].add_run()
r.text = "https://huggingface.co/datasets/ComplexDataLab/OpenFake"
r.font.size = Pt(16)
r.hyperlink.address = "https://huggingface.co/datasets/ComplexDataLab/OpenFake"

# ================================================================== 10 UML list + diagram slides
content_slide("UML Diagrams", [
    "Use Case Diagram", "DFD Level 0", "DFD Level 1", "Control Flow Diagram", "Sequence Diagram", "Class Diagram"],
    size=22, top=2.1, note="The next six slides show the diagrams. Each one is explained in the speaker notes.")
picture_slide("Use Case Diagram", DIAG / "fig_usecase.png", note=(
    "Two kinds of users. The User uploads a file, sees the Trust Score and verdict, the heatmap, the explanation, and can download a report. "
    "The project team trains and updates the models, evaluates them on unseen generators and updates the calibration."))
picture_slide("DFD Level 0", DIAG / "fig_dfd0.png", note=(
    "Context diagram: the whole system is one process. The user sends a file and receives score, verdict, heatmap and text. "
    "The team supplies trained heads and calibration and receives evaluation results. Public datasets and CLIP weights feed the training."))
picture_slide("DFD Level 1", DIAG / "fig_dfd1.png", note=(
    "Four processes: 1.0 validate the upload (bad files get an error message), 2.0 analyse with the routed analyser using the stored heads, "
    "3.0 calibrate and score using the stored calibration limits, 4.0 explain and report back to the user."))
picture_slide("Control Flow Diagram", DIAG / "fig_flow.png", note=(
    "Step by step: read the true file type from the bytes, check size, decode; if invalid show a clear error. Otherwise route to the analyser, "
    "get the evidence record, calibrate to a Trust Score, and choose the band: 54 or more Likely Authentic, 17 or less Likely AI-generated, otherwise Needs Human Review."))
picture_slide("Sequence Diagram", DIAG / "fig_sequence.png", note=(
    "One image check over time. The web interface passes the bytes to the validator and router, which asks the image analyser. "
    "The analyser cuts ten patches, re-saves them as JPEG, encodes them with CLIP and scores them with the linear head. "
    "The scoring layer applies Platt scaling, makes the Trust Score, verdict and sentences, and the card, heatmap and report come back to the user."))
picture_slide("Class Diagram", DIAG / "fig_class.png", note=(
    "Main software components. Every analyser implements the same Analyser interface and returns an Evidence record; the image analyser owns the CLIP encoder and the linear head; "
    "the trust engine holds the calibration and the band limits; the report builder writes the explanation. Dashed classes (video, document) are designed, not built yet."))

# ================================================================== 17 algorithm
s = picture_slide("Algorithm", DIAG / "fig_image_pipeline.png", box=(0.4, 1.7, 9.2, 3.2), note=(
    "The image algorithm: do not resize; cut ten 224 by 224 patches; re-save each as JPEG quality 95 so real and fake share one compression history; "
    "encode with frozen CLIP ViT-L/14; score with a linear head; average the ten scores; calibrate with Platt scaling; Trust Score = 100 x (1 - p)."))
tb = s.shapes.add_textbox(Inches(0.6), Inches(5.0), Inches(8.8), Inches(2.0))
tf = tb.text_frame
tf.word_wrap = True
lines = ["Patch score:  s_i = wT z_i + b,   image score:  s = mean of the ten patch scores",
         "Probability:  p = sigmoid(a s + c) (Platt scaling),   Trust Score:  T = round(100 (1 - p))",
         "Verdict:  T >= 54 Likely Authentic,   T <= 17 Likely AI-generated,   otherwise Needs Human Review"]
for i, ln in enumerate(lines):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    r = p.add_run()
    r.text = ln
    r.font.size = Pt(16)

# ================================================================== 18-20 preprocessing
prep = HERE / "diagrams" / "prep_example.png"
sample_photo = Path(r"C:\Users\bhand\AppData\Local\Temp\shotsrc\real04.jpg")
if sample_photo.exists():
    make_prep_image(sample_photo, prep)
content_slide("Data Preprocessing", [
    "The image is not resized, because resizing removes the fine traces that give AI images away.",
    "Ten patches of 224 x 224 pixels (the centre and a 3 x 3 grid) are cut at the image's own resolution.",
    "Every patch is re-saved as JPEG with quality 95, so real and generated images share one compression history and the file format cannot leak the answer."],
    size=20, note="Preprocessing keeps the evidence and removes shortcuts. Native-resolution patches keep fine noise; the common JPEG re-save stops a detector from learning 'PNG means AI'.")
content_slide("Data Augmentation", [
    "Every training image also gives a randomly damaged copy: resized, blurred and JPEG-compressed (quality 50 to 95).",
    "Reason: pictures are resized and compressed every time they are shared, and this removes much of the evidence.",
    "Effect: AI images caught after JPEG quality 70 rose from 14% (first prototype) to 66%.",
    "Training data: real photographs and about sixty generators from OpenFake, plus patches of edited photographs."],
    size=20, note="Augmentation is the main reason the final model survives sharing. Without it, only 14% of compressed AI images were caught; with it, 66%.")
if prep.exists():
    picture_slide("Preprocessing Results", prep, note="Left: a real photograph with its ten patches (orange = centre patch). Right: the same patch in the three training versions.",
                  title_size=36)

# ================================================================== 21 model + 22 deployment
content_slide("CLIP Encoder and Trained Head", [
    "CLIP is a vision-language model of OpenAI; we use its ViT-L/14 image encoder as it is (frozen).",
    "Only a small linear head (768 weights and a bias) is trained, by L2-regularised logistic regression.",
    "The head is a few kilobytes, so training takes minutes on a laptop and the app runs on a CPU.",
    "",
    "ROC-AUC: 0.982 on 20 unseen generators (EfficientNet-B0 baseline: 0.831)"],
    size=20, note="Why this design: the heavy model is not trained; only a tiny head is. That keeps the app small and lets us retrain in minutes. "
                  "AUC of 0.982 means a randomly chosen AI image gets a higher score than a randomly chosen real photo 98% of the time.")
content_slide("Deployment: Streamlit", [
    "Using Streamlit we built the web interface of our system.",
    "Streamlit is a Python based web framework.",
    "The application Content Authenticity Checker is hosted on Streamlit Community Cloud and redeploys from GitHub.",
    "Here the user can upload an image and receive the Trust Score, verdict, heatmap and explanation.",
    "The app also warns for drawings, screenshots, small images and compressed files, and recognises the visible Gemini watermark."],
    size=20, note="Live link: https://deepfake-detector-perceptrons.streamlit.app . Uploads are processed in memory and never stored.")

# ================================================================== 23 requirements
s = new_slide("Hardware/Software Requirements", 32)
fill_body(s, ["Hardware Requirements :"], size=24, top=1.7, height=0.5)
body = s.placeholders[1]
for p in body.text_frame.paragraphs:
    for r in p.runs:
        r.font.bold = True
        r.font.underline = True
for t in ["1. Operating System: Windows 10 or 11, or Linux (64-bit)", "2. Processor: x86 64-bit CPU, no GPU needed for use", "3. Memory: 2 GB for the app (peak about 1.5 GB); 8 GB for development"]:
    p = body.text_frame.add_paragraph()
    r = p.add_run()
    r.text = t
    r.font.size = Pt(20)
p = body.text_frame.add_paragraph()
r = p.add_run()
r.text = "Software Requirements :"
r.font.size, r.font.bold, r.font.underline = Pt(24), True, True
for t in ["Python 3.11, PyTorch (CPU), open_clip, scikit-learn, Pillow, NumPy, Matplotlib", "Streamlit (web interface); Git and GitHub (code and hosting)", "Training: free Kaggle GPU for feature extraction"]:
    p = body.text_frame.add_paragraph()
    r = p.add_run()
    r.text = t
    r.font.size = Pt(20)
notes(s, "Everything is free and open source. No GPU is needed to use the system; a GPU was only used once to compute stored features.")

# ================================================================== 24-25 project plan
table_slide("Project Plan", ["Work Task", "Description", "Duration"], [
    ["Phase 1: Image", "Setup, data, baseline, CLIP model, robustness, calibration, web app", "Weeks 1-11 (28 Sep - 13 Dec 2026) - done"],
    ["Phase 2: Video", "Face crops, frame model, video score and timeline", "Weeks 12-16 (14 Dec 2026 - 17 Jan 2027)"],
    ["Phase 3: Document", "OCR, forensic maps, dual-stream U-Net, rule checks, integration", "Weeks 17-26 (18 Jan - 28 Mar 2027)"],
    ["System testing", "Unseen-source tests, team photographs, notices and checks", "In every phase"],
    ["Initial Report", "Preliminary (Stage 1) report, synopsis, abstract", "Done"],
    ["Final Report", "Final report and research paper", "Week 26"]],
    [2.2, 3.7, 2.9], size=14, row_h=0.8, note="Three phases, one medium at a time. Each phase ends with a working demonstration. Phase 1 (image) is finished.")
picture_slide("Project Schedule", DIAG / "fig_gantt.png", note="The same plan as a chart: green bars are finished work, blue bars are planned.")

# ================================================================== 26-31 results (screenshots)
picture_slide("Results", slide_crop(FIGS / "app_landing.png", (290, 0, 1262, 560), "landing"), caption="Web Interface", box=(0.5, 1.7, 9.0, 5.4),
              note="Start page of the application: drag and drop an image. A sample image can also be chosen.")
picture_slide("Result: Real Photograph", slide_crop(FIGS / "app_result_real.png", (310, 270, 1250, 1050), "real"), caption="Trust Score 90 / 100 - Likely Authentic", box=(0.5, 1.6, 9.0, 5.6),
              note="A real full-size phone photograph. The score is high, the explanation says no clear traces of AI were found, and the camera information is shown as supporting evidence only.")
picture_slide("Result: AI-generated Image", slide_crop(FIGS / "app_result_ai.png", (310, 270, 1250, 1050), "ai"), caption="Likely AI-generated / Manipulated", box=(0.5, 1.6, 9.0, 5.6),
              note="A ChatGPT image. The score is very low, the traces are spread over the whole image, and the file's own provenance markers are listed.")
picture_slide("Result: Gemini Watermark", slide_crop(FIGS / "app_result_watermark.png", (310, 270, 1250, 1050), "watermark"), caption="Visible Gemini watermark found - verdict AI-generated", box=(0.5, 1.6, 9.0, 5.6),
              note="Images downloaded from the Gemini app carry a small star at the bottom-right corner. The app finds it and reports it separately from the model score. "
                   "The mark can be cropped away, so its absence proves nothing.")
picture_slide("Result: Drawing", slide_crop(FIGS / "app_result_drawing.png", (310, 270, 1250, 1050), "drawing"), caption="Not a camera photo - verdict set to Needs Human Review", box=(0.5, 1.6, 9.0, 5.6),
              note="Drawings, graphics and screenshots are outside what the detector was built for. The app says so and sends them to a person instead of guessing.")
picture_slide("Cross-Generator Test Page", slide_crop(FIGS / "app_crossgen.png", (300, 120, 1280, 1150), "crossgen"), box=(0.5, 1.6, 9.0, 5.6),
              note="A page of the application that shows the held-out results per generator.")

# ================================================================== graphs
picture_slide("Graph of ROC-AUC of Model", DIAG / "fig_auc.png", box=(0.5, 1.8, 9.0, 4.6),
              note="ROC-AUC on 20 generators not seen in training (1,200 real and 2,192 AI test images). The proof model never saw an OpenAI or Google image and still reaches 0.957 and 0.984.")
picture_slide("Comparison with Baseline", DIAG / "fig_baseline.png", box=(1.2, 1.8, 7.6, 4.6),
              note="An EfficientNet-B0 trained on the same images reaches 0.831; our CLIP-based analyser reaches 0.982 on the same clean test files.")
cm = PAPER_FIGS / "fig_cm_roc.png"
picture_slide("Confusion Matrix and ROC Curve", cm, box=(0.4, 1.8, 9.2, 4.2), caption="1,110 of 1,200 real photographs accepted; 2,050 of 2,192 AI images caught (accuracy 93.2%)",
              note="Left: confusion matrix at threshold 0.5. Right: ROC curve. Errors are not symmetric: 142 AI images missed against 90 real photographs flagged; the review band catches many of both.")
table_slide("Tests on Team Photographs", ["Measure", "ChatGPT edits", "Gemini edits"], [
    ["Pairs (photograph and its edit)", "10", "47"],
    ["ROC-AUC, edit against original", "0.95", "0.92"],
    ["Edit scores more AI-like than its original", "10 of 10", "47 of 47"],
    ["Edits called AI-generated", "8 of 10", "14 of 47"],
    ["Edits sent to human review", "2 of 10", "28 of 47"],
    ["Original photographs called authentic", "9 of 10", "38 of 47"]],
    [4.6, 2.1, 2.1], size=16, row_h=0.6, note=(
        "Full-size phone photographs taken by the team and their edits made with the ChatGPT and Gemini apps; used for testing only. "
        "The sets are small, so the numbers are indicative. Gemini edits are ranked well but more often sent to human review; the watermark check, reported separately, covers Gemini-app downloads."))

# ================================================================== test cases
TC = [
    ["001", "Real phone photo", "4000 x 3000 photo", "Authentic (87)", "Authentic", "P"],
    ["002", "ChatGPT edit", "Edited photo", "AI-generated (2)", "AI-generated", "P"],
    ["003", "Gemini-app image", "Image with star mark", "AI + watermark alert", "AI-generated", "P"],
    ["004", "Drawing", "Black image, white strokes", "Review + notice", "Human review", "P"],
    ["005", "Very small image", "416 x 752 image", "Small-image notice", "Notice", "P"],
    ["006", "Corrupt JPEG", "Cut-off JPEG file", "Clear error message", "Error", "P"],
    ["007", "Text file as .jpg", "Renamed text file", "Clear error message", "Error", "P"],
    ["008", "Empty file", "0-byte file", "Clear error message", "Error", "P"],
    ["009", "PDF document", "PDF file (Phase 3)", "Clear error message", "Error", "P"],
]
table_slide("Test Cases", ["Test case ID", "Test case", "Test case i/p", "Actual result", "Expected result", "P/F"], TC,
            [0.75, 1.75, 2.05, 2.0, 1.55, 0.6], size=13, row_h=0.47, top=1.6,
            note="Nine test cases. 001 to 005 were run on the application with images from the team test sets; 006 to 009 were run on the file validation code with real bad files. "
                 "Test 009 shows the present behaviour: the image analyser refuses a PDF with a clear message; the document analyser is planned for Phase 3.")

# ================================================================== applications, conclusion, future scope
content_slide("Applications", [
    "Fact-checking and newsrooms: first screening of pictures that arrive from the public.",
    "Social media and content moderation: flagging likely AI images for a human reviewer.",
    "Banks, insurers and KYC teams: first-level screening of submitted documents (planned).",
    "Education and awareness: showing why a picture looks synthetic through the heatmap and explanation.",
    "Personal use: checking a picture received in a messaging group."], size=20,
    note="Applications are screening aids: the Trust Score is not forensic proof.")
content_slide("Conclusion", [
    "A frozen CLIP encoder with a small trained head, a degraded-copy training set and a calibrated, explainable scoring layer detects most AI images from unseen generators.",
    "The system reaches ROC-AUC 0.982 on 20 unseen generators, against 0.831 for an EfficientNet baseline, and accepts 92% of real photographs.",
    "The proof model, which never saw an OpenAI or Google image, still catches 84% of ChatGPT images.",
    "The layered design lets the same Trust Score serve images, videos and documents."], size=20,
    note="Summary of Phase 1. Video and document analysers are designed for the next phases.")
content_slide("Future Scope", [
    "Build and evaluate the video analyser on FaceForensics++ and Celeb-DF.",
    "Build the document analyser (dual-stream U-Net, OCR, rule checks) and test it on DocTamper and human-made forgeries.",
    "Pixel-level maps to find a small machine-made region inside a real photograph.",
    "Better robustness to compression and resizing; add new generators to the unseen-source tests.",
    "A separate front end that calls the analysers through an API."], size=20,
    note="What comes next, in order of the project plan.")

# ================================================================== references (clickable)
REFS = [
    ("A. Radford et al., “Learning transferable visual models from natural language supervision,” in Proc. ICML, 2021.", "https://arxiv.org/abs/2103.00020"),
    ("U. Ojha, Y. Li, and Y. J. Lee, “Towards universal fake image detectors that generalize across generative models,” in Proc. CVPR, 2023.", "https://arxiv.org/abs/2302.10174"),
    ("S.-Y. Wang et al., “CNN-generated images are surprisingly easy to spot... for now,” in Proc. CVPR, 2020.", "https://arxiv.org/abs/1912.11035"),
    ("R. Corvi et al., “On the detection of synthetic images generated by diffusion models,” in Proc. ICASSP, 2023.", "https://arxiv.org/abs/2211.00680"),
    ("P. Grommelt et al., “Fake or JPEG? Revealing common biases in generated image detection datasets,” arXiv:2403.17608, 2024.", "https://arxiv.org/abs/2403.17608"),
    ("M. Zhu et al., “GenImage: A million-scale benchmark for detecting AI-generated image,” NeurIPS Datasets and Benchmarks, 2023.", "https://arxiv.org/abs/2306.08571"),
    ("V. Livernoche et al., “OpenFake: An open dataset and platform toward real-world deepfake detection,” arXiv:2509.09495, 2025.", "https://arxiv.org/abs/2509.09495"),
    ("H. Ci et al., “DiffSeg30k: A multi-turn diffusion editing benchmark for localized AIGC detection,” arXiv:2511.19111, 2025.", "https://arxiv.org/abs/2511.19111"),
    ("M. Tan and Q. V. Le, “EfficientNet: Rethinking model scaling for convolutional neural networks,” in Proc. ICML, 2019.", "https://arxiv.org/abs/1905.11946"),
    ("C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proc. ICML, 2017.", "https://arxiv.org/abs/1706.04599"),
    ("A. Rössler et al., “FaceForensics++: Learning to detect manipulated facial images,” in Proc. ICCV, 2019.", "https://arxiv.org/abs/1901.08971"),
    ("O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in Proc. MICCAI, 2015.", "https://arxiv.org/abs/1505.04597"),
]
for part in (0, 1):
    s = new_slide("References")
    body = s.placeholders[1]
    body.left, body.top, body.width, body.height = Inches(0.5), Inches(1.75), Inches(9.0), Inches(5.4)
    tf = body.text_frame
    tf.word_wrap = True
    for k, (txt, url) in enumerate(REFS[part * 6:part * 6 + 6]):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = f"{part * 6 + k + 1}. {txt}"
        r.font.size = Pt(15)
        r.hyperlink.address = url
    notes(s, "Each reference is a clickable link to its arXiv page. The full list of 32 references is in the research paper.")

# ================================================================== thank you
s = prs.slides.add_slide(prs.slide_layouts[6])
add_logos(s)
tb = s.shapes.add_textbox(Inches(2.0), Inches(5.6), Inches(7.6), Inches(1.2))
p = tb.text_frame.paragraphs[0]
p.alignment = 3
r = p.add_run()
r.text = "…Thank You !!!"
r.font.name, r.font.size = "Times New Roman", Pt(54)
notes(s, "Invite questions. Live demo: https://deepfake-detector-perceptrons.streamlit.app")

prs.save(str(OUT))
print("saved", OUT, "with", len(prs.slides), "slides")
