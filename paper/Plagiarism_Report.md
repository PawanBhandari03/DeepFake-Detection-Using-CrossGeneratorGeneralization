# Originality Check — Research Papers (25 Oct 2026)

> **For:** The Perceptrons · **What was checked:** the "Without Authors" PDF of each paper (the "With Authors" PDF has the same text apart from the author block and the acknowledgement)
>
> **Important:** this is **our own text-overlap check**, run with `paper/overlap_check.py`. It is **not** Turnitin or the institute's software and cannot predict their percentage. Please run the institute's checker on the final PDF as well.

## 0. Which file is which (changed 25 Oct)

Both versions of every paper show **our names and the college heading**. They differ only in the references:

| File | Our names | Citation numbers [1], [2], ... in the text | Reference list |
|---|---|---|---|
| `..._With_Authors.pdf` | yes | yes | yes, all together in the right column of the last page |
| `..._Without_Authors.pdf` | yes | **no** | **no** |

"Authors" here means the authors of the cited works. Use the **Without_Authors** file for a plagiarism checker, because then the reference list cannot match other documents.

Overlap of the **Without_Authors** versions (checked text: 2,314 words for the 4-page whole-project paper, 2,887 words for the image paper):

| Compared with | 4-page whole-project paper: 5-word / 8-word | Image paper: 5-word / 8-word |
|---|---|---|
| Earlier draft paper | 1.0% / 0.7% | 0.9% / 0.6% |
| SRS | 0.7% / 0 | 0.2% / 0 |
| System Design | 0.8% / 0.3% | 0.7% / 0.2% |
| Overview and Algorithms | 0.2% / 0 | 0.0% / 0 |
| Project roadmap | 0.2% / 0 | 0.2% / 0 |
| Walkthrough 01 | 0.7% / 0 | 0.3% / 0 |
| Walkthrough 02 (work log) | 0.8% / 0.0% | 1.7% / 0.4% |

The 17 eight-word matches with the earlier draft are **only the author and college lines** at the top (our names, the department, "JSPM's Bhivarabai Sawant Institute of Technology and Research, Wagholi, Pune, India"), which are our own details and appear in every document of the project. No sentence of the text is shared.

## 1. How the text was written

- Both papers were written from scratch for this project, from our own code, experiments and design documents. No sentences were copied from other papers, from the earlier draft paper or from a friend's report.
- Statements about other work are written in our own words and followed by numbered citations. The numbers follow the order of first use, and citations appear on every page that contains running text. All references were checked for existence; where a conference acceptance could not be confirmed (Yermakov et al.) the paper cites the arXiv preprint.
- Every number in the results comes from our own experiment files (`prototype/results/*.json`, `paper/analysis.json`, `paper/robust_results.json`). The full-project paper gives **no results for the video and document modules**, and the image paper marks its unfinished items as "in progress", because nothing was run for them yet.

## 2. Method of the check

For the paper body (title block and reference list removed) we count how many **5-word** and **8-word sequences** also occur in each other document. Short shared sequences are normal (technical terms, dataset names, the names of our verdict bands); long shared sequences (8 words) would indicate copied sentences.

## 3a. The 4-page paper for submission (`Final_Paper_4pages_Without_Authors.pdf`, 2,183 words)

Written after the guide's instruction (at most 4 pages, whole project, focus on the process). The requirements table was re-worded in our own words (it summarises the SRS instead of copying it), and three sentences that resembled the earlier draft were rewritten. Final result:

| Compared with | Shared 5-word | Shared 8-word |
|---|---|---|
| Earlier draft paper (`Research_Paper_Final-1.pdf`) | 2 (0.1%) | **0** |
| Our SRS | 11 (0.5%) | **0** |
| Our System Design (LaTeX) | 6 (0.3%) | 0 |
| Our Overview and Algorithms (LaTeX) | 4 (0.2%) | 0 |
| Project roadmap | 3 (0.1%) | 0 |
| Walkthrough 01 (prototype) | 16 (0.7%) | 0 |
| Walkthrough 02 (v3 work log) | 13 (0.6%) | 1 (0.0%) |
| Our 6-page full paper / our image paper | about 9% / 6% | about 3% / 2% (same authors, same project) |

The remaining shared 5-word sequences are technical terms ("frozen CLIP encoder", "dual-stream U-Net", "error-level analysis") and the names of our verdict bands. It has 32 references, all checked, and citations appear on every page.

## 3. Result for the earlier, longer papers

| Compared with | Full-project paper (3,274 words): 5-word / 8-word | Image paper (2,823 words): 5-word / 8-word |
|---|---|---|
| Earlier draft paper (`Research_Paper_Final-1.pdf`, same text as `Research_Paper_Final_.pdf`) | 13 (0.4%) / **0** | 5 (0.2%) / **0** |
| Our SRS | 6 (0.2%) / 0 | 3 (0.1%) / 0 |
| Our System Design (LaTeX) | 5 (0.2%) / 0 | 8 (0.3%) / 0 |
| Our Overview and Algorithms (LaTeX) | 3 (0.1%) / 0 | 1 (0.0%) / 0 |
| Project roadmap | 5 (0.2%) / 0 | 5 (0.2%) / 0 |
| Walkthrough 01 (prototype) | 6 (0.2%) / 0 | 8 (0.3%) / 0 |
| Walkthrough 02 (v3 work log) | 38 (1.2%) / 9 (0.3%) | 44 (1.6%) / 9 (0.3%) |
| **The other one of our two papers** | about 41% / 35% | about 40% / 35% |

**Reading the table**

- Against other people's work and the earlier draft the overlap is below 1%, with no 8-word sequence in common. Three phrases of the earlier draft that had slipped in by coincidence were reworded.
- The 9 shared 8-word sequences with the v3 work log are our own measured facts (for example the list of OpenAI and Google generators in training); the work log is in our public GitHub repository, so a checker that crawls GitHub may show these small matches.
- The **large overlap between the two papers is our own text**: the full paper contains the image section of the image paper. **Submit only one of them to a checker**, or tell the examiner that the image paper is the image part of the full paper.
- Reference lists, dataset names (OpenFake, DiffSeg30k, FaceForensics++, ...), standard terms ("frozen CLIP encoder", "ROC-AUC") and the verdict names will always match other documents. Most checkers can exclude references; use that option.

## 4. What this check cannot tell

- It does not search the internet or a paper database. Another paper could share wording we have never seen; with text written independently this is unlikely beyond common technical phrases.
- It does not predict any AI-text detector. Such detectors are unreliable in both directions, so do not use them as proof either way. What helps is that the authors know and can explain every sentence, number and figure, and keep the scripts and result files (`paper/`, `prototype/results/`, `kaggle/`) as evidence of the work.
- Check your institute's rules on writing support and disclose if they require it.

## 5. If a checker reports a high percentage

1. Open the report and look at which passages match. Matches on references, dataset names and the verdict labels are normal.
2. If a real passage matches an outside source, tell us the passage and we will rewrite it.
3. If it matches our own earlier documents or the other paper, that is self-overlap (see above).
