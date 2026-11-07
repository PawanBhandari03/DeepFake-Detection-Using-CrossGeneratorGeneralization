"""One-off: align the design documents with the final image model (v3): frequency features were evaluated and not used.

    python docs/tools/align_image_docs.py

Edits docs/System_Design.tex and docs/Overview_and_Algorithms.tex. Each replacement is asserted to match exactly once.
"""
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent


def edit(path, pairs):
    raw = (DOCS / path).read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    s = raw.replace("\r\n", "\n")
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:60], s.count(old))
        s = s.replace(old, new)
    (DOCS / path).write_bytes((s.replace("\n", "\r\n") if crlf else s).encode("utf-8"))
    print("edited", path, len(pairs), "replacements")


STATUS = r"""\subsection{Status: what the final image model changed}
\label{sec:image-status}
This section was written as the design. The implemented model (version 3, October 2026) follows it with these
changes, each decided from measurements:
\begin{itemize}
  \item \textbf{Frequency branch evaluated and not used.} In the prototype ablation the radial-spectrum features gave no
        measurable gain over the CLIP features alone, and the spectral peak score did not separate real from AI
        images (median 2.21 against 2.19). The final model uses the CLIP features only; the branch, the spectrum plot and
        the peak-score sentence below remain as the design that was tested.
  \item \textbf{Patch-wise scoring.} An image is scored on ten $224{\times}224$ patches at native resolution (one centre patch and a
        $3{\times}3$ grid); the image score is the mean patch logit and the patch scores form the heatmap.
  \item \textbf{Training data.} OpenFake (real photographs and more than 60 generators, including ChatGPT and Gemini images),
        each image also as a randomly compressed, resized or blurred copy, plus patches from AI-edited images
        (DiffSeg30k, tiny-audits). GenImage with Stable Diffusion v1.4 only was the first prototype.
  \item \textbf{Calibration and bands.} Platt scaling on validation data; the Trust Score bands (Authentic from 54, Manipulated
        up to 17) are chosen on validation data. Test: held-out generators, an in-the-wild Reddit set and four degradation
        conditions, with 95\% bootstrap intervals.
  \item \textbf{Baseline.} The EfficientNet-B0 baseline is retrained on the same OpenFake training images and tested on the same
        held-out images, so the comparison is like for like (see the work log, Walkthrough 02).
\end{itemize}

"""

edit("System_Design.tex", [
    ("(CLIP), and forensic traces such as\nfrequency-domain artifacts and noise inconsistencies. The same principle",
     "(CLIP) read patch by patch at the image's own resolution. Forensic frequency features were also tested and\n"
     "gave no extra gain (Section~\\ref{sec:image-status}). The same principle"),
    ("Frozen CLIP encoder + frequency features + logistic head & GenImage: real + SD v1.4 & 7 other GenImage generators + own 2026 test set",
     "Frozen CLIP encoder (patch-wise) + calibrated logistic head & OpenFake: real + 60+ generators, compression-augmented & OpenFake held-out generators (e.g. GPT-Image-1.5, Nano Banana Pro) + Reddit in-the-wild set"),
    ("heatmaps, frequency evidence, OCR logic checks", "heatmaps, OCR logic checks"),
    ("preprocessing, CLIP + frequency features, calibrated classifier, heatmap", "preprocessing, patch-wise CLIP features, calibrated classifier, heatmap"),
    ("The image pipeline (Figure~\\ref{fig:image-pipeline}) has two feature branches: a \\emph{semantic}\n"
     "branch that uses a frozen CLIP encoder, and a \\emph{forensic} branch that measures\n"
     "frequency-domain artifacts. Their features are joined and passed to a small classifier whose\n"
     "output is calibrated into a probability.",
     "The image pipeline (Figure~\\ref{fig:image-pipeline}) was designed with two feature branches: a \\emph{semantic}\n"
     "branch that uses a frozen CLIP encoder, and a \\emph{forensic} branch that measures\n"
     "frequency-domain artifacts. Their features are joined and passed to a small classifier whose\n"
     "output is calibrated into a probability. The forensic branch was tested and is not part of the final model\n"
     "(Section~\\ref{sec:image-status})."),
    ("\\textbf{Frequency branch}\\\\ {\\footnotesize noise residual",
     "\\textbf{Frequency branch (tested, not used in v3)}\\\\ {\\footnotesize noise residual"),
    ("\\subsection{Pipeline}\nThe image pipeline", STATUS + "\\subsection{Pipeline}\nThe image pipeline"),
    ("\\subsection{Forensic branch: frequency features}\n",
     "\\subsection{Forensic branch: frequency features (tested, not used in the final model)}\n"),
    ("\\textbf{Spectrum plot.} The user also sees the frequency spectrum", "\\textbf{Spectrum plot (design; shown in a separate tab, not used in the verdict).} The user also sees the frequency spectrum"),
    ("High spectrum peak score & ``The frequency spectrum shows periodic peaks, a typical trace of AI up-sampling.'' \\\\\n",
     "High spectrum peak score (design only; not used, it did not separate real from AI) & ``The frequency spectrum shows periodic peaks, a typical trace of AI up-sampling.'' \\\\\n"),
    ("CLIP and frequency features, training", "patch-wise CLIP features, training"),
    ("Debiasing, frequency, robustness", "Debiasing, robustness (frequency tested, no gain)"),
])

edit("Overview_and_Algorithms.tex", [
    ("(general visual features and frequency\nartifacts)", "(general visual features from a frozen CLIP encoder,\nread patch by patch)"),
    ("Frozen CLIP features + FFT frequency features + logistic regression", "Frozen CLIP features (patch-wise) + calibrated logistic regression"),
    ("Forensic features & \\textbf{FFT} of the noise residual, radial spectrum (64 values) & Generators' up-sampling leaves periodic frequency traces \\\\",
     "Forensic features (tested, not used) & \\textbf{FFT} of the noise residual, radial spectrum (64 values) & Designed to catch up-sampling traces; gave no gain in our test, so the final model leaves it out \\\\"),
    ("\\subsection*{Frequency features (Phase 1)}\n", "\\subsection*{Frequency features (Phase 1; tested, not used in the final model)}\n"),
    ("The two feature vectors are joined and scored by logistic regression,\n$\\ell=\\mathbf{w}^{\\top}[\\mathbf{z};\\boldsymbol{\\phi}]+b$, trained",
     "In the final model only the CLIP vector is used (the frequency vector was tested and dropped), scored by logistic regression,\n$\\ell=\\mathbf{w}^{\\top}\\mathbf{z}+b$, trained"),
    ("(model score, frequency score, localization score, rule flags)", "(model score, localization score, rule flags)"),
])
