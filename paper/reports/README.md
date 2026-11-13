# Project reports in LaTeX (Overleaf)

Three documents and the final presentation for **Deepfake Detection with Cross-Generator Generalization** (The Perceptrons), written in the format of the department's earlier project reports (Abstract, Synopsis, Stage 1 / preliminary report).

| Folder | Document | Pages (approx.) |
|---|---|---|
| `abstract/` | Project Abstract | 1 |
| `synopsis/` | Project Synopsis | 12–15 |
| `stage1_report/` | Project Stage 1 (preliminary) Report | 45–60 |

## How to use on Overleaf

1. Overleaf → **New Project → Upload Project** → choose one zip from `overleaf_zips/` (for example `stage1_report.zip`).
2. Menu → **Compiler: pdfLaTeX**. Main file: `main.tex`.
3. Click *Recompile* two or three times so that the contents, list of figures and list of tables fill in.

Every folder is a complete project (preamble, team details, references, figures, logo). Nothing needs to be installed.

## What you must fill in (left blank on purpose)

Edit **`shared/team.tex`** only, then run `python paper/reports/build_overleaf.py` again (or edit `team.tex` inside the Overleaf project).

* `\PRNOne`, `\PRNTwo`, `\PRNThree` — PRN numbers (printed as a blank line until filled).
* `\PrincipalName` — the current principal's name.
* `\StudentTwo`, `\StudentThree` — full names with father's name, if the university format needs them.
* Certificate date and external examiner are left blank for signature.
* The guide and the Head of Department are the same person in the project files (Dr. G. M. Bhandari), so the certificate shows her in both places; change `\GuideName` if the guide is somebody else.

## Diagrams

All diagrams are drawn in TikZ (Overleaf renders them; Mermaid is not available on Overleaf). They are in `shared/figs/`:
architecture, training and inference workflow, image pipeline, DFD level 0 and 1, use case, flow chart, Gantt chart, result charts (pgfplots).
The app screenshots (`app_*.png`) were captured from the running application on 13 Nov 2026.

## Sources of the numbers

Every result comes from the project files: `prototype/results/*.json`, `paper/analysis.json`, `docs/walkthroughs/02_v3_work_log.md` (Entries 30–36) and the research paper `paper/build_short_paper.py`. The references are the same 32 as in the research paper.

## Compiled PDFs and the presentation

* `pdf/` holds the compiled PDFs (abstract 1 page, synopsis 12 pages, Stage 1 report 47 pages) and a PDF copy of the presentation. PDFs are not stored in git; rebuild them with any LaTeX (Overleaf, or Tectonic: `tectonic main.tex` inside each folder).
* `presentation/Final_Project_Presentation.pptx` follows the department's sample deck exactly (same theme, logos, fonts, slide order) and has 40 slides with speaker notes on every slide. Rebuild: `python paper/reports/presentation/make_diagrams.py <tectonic.exe>` then `python paper/reports/presentation/build_pptx.py "<sample deck .pptx>"`.
* The PRN / exam numbers on the title slide are blank brackets to fill in.

All three LaTeX projects compile without errors (checked with Tectonic 0.17, 13 Nov 2026); only line-spacing warnings remain.
