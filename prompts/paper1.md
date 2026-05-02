# Paper 1 session prompt — Donut-web-app repo

You are working in the Donut-web-app repository. This repo holds the
production OCR-and-KIE web service deployed at https://image-to-text.fit/
plus the legacy research artefacts (run logs, training PDFs, evaluation
outputs) from a multi-dataset DONUT training programme that was
conducted earlier in the project lifetime.

**GOAL:** produce Paper 1 — a complete, IEEE-Access-format research
paper — by re-rendering the existing legacy PDF artefacts and run-log
evidence into a modern LaTeX manuscript.

**THESIS OF PAPER 1:** "Multi-dataset training of DONUT for receipt
key-information extraction: demonstrating that a curated mix of
CORD-v2 + WildReceipt + SROIE training data lifts SROIE Task-3 F1 above
0.88 without architectural modification."

## Constraints

- Use ONLY real numbers from the legacy PDF artefacts and run logs
  already in this repository. Do not fabricate any F1, NED, EM, or
  ablation cell.
- The paper must read as a complete, standalone contribution on the
  data-curation axis. Do NOT underplay it as "just the data axis" or
  hint that it is one of a series. Use full research-paper language.
- Format: IEEE Access conference template. Single column.
- Bibliography: include all foundational citations (DONUT, SROIE,
  CORD-v2, WildReceipt) plus relevant data-centric AI references.
- Include a "Reproducibility" appendix listing the run-IDs in this
  repository whose artefacts back every cited number.

## Deliverables

1. `paper1/` directory with LaTeX template + sections + `references.bib`.
2. A make-target (`make paper1-pdf`) that compiles the paper end-to-end
   from the legacy artefacts via the inject pipeline (mirror the
   kaggle2 pattern: `combined_metrics.json` -> `\VAR{}` -> tectonic).
3. A `README.md` describing how to reproduce the paper from the legacy
   run-IDs.

## Workflow notes

- Inventory the legacy PDFs first. List which papers exist, which
  numbers they cite, and which run-IDs back them.
- Cross-check 3–5 numbers against the run-log JSON to confirm
  provenance before writing prose.
- The web service code at `/app/` is operational; do NOT modify it.
  Only the paper artefacts are in scope.
- The user has previously confirmed F1 > 0.88 on SROIE under
  multi-dataset training; verify the exact number from the run log
  and use it.

**START WITH AN INVENTORY REPORT** before writing any LaTeX. Show what
legacy artefacts exist and which numbers each cites. Wait for
confirmation before proceeding to the paper draft.
