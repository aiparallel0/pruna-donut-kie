# pruna-donut-kie

![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![PyTorch 2.4](https://img.shields.io/badge/pytorch-2.4-ee4c2c.svg)
![Pruna OSS](https://img.shields.io/badge/pruna-OSS-green.svg)
![Status](https://img.shields.io/badge/status-in%20progress-yellow.svg)

Compression study of fine-tuned DONUT for receipt key-information extraction on the canonical ICDAR-2019 SROIE Task-3 test set, extending the architectural-decomposition result from [aiparallel0/kaggle2](https://github.com/aiparallel0/kaggle2).

## Motivation

A companion study (Bekteş & Keskinöz, manuscript in preparation) shows that on the 347-image SROIE Task-3 test set, a YOLOv8 + TrOCR + 1.16M-parameter attention-assigner pipeline (≈66.93M parameters total) matches a fine-tuned DONUT (≈260.78M parameters) at one-quarter the parameter budget. The pipeline reaches global token-F1 = 85.8% versus DONUT's 82.7%; paired-bootstrap 95% CI on ΔF1 ∈ [−0.0529, −0.0087], McNemar p = 0.0248. This is an architectural lower bound — it shows that decomposing the problem into detect → read → assign is parameter-efficient relative to end-to-end vision-to-text decoding at this data scale.

The natural next question is whether compression of the end-to-end model closes the gap. If post-hoc quantization, pruning, and distillation can shrink DONUT below the pipeline's parameter floor while preserving F1, the architectural argument weakens. If compression hits an F1 cliff before that floor is reached, the architectural argument strengthens — and the pipeline is justified not as a novelty but as a structurally cheaper substrate.

## Research question

At what compression budget does compressed-DONUT cross below the pipeline's 66.93M-parameter floor on SROIE Task-3, and does the pipeline's F1 advantage survive aggressive compression of its DONUT competitor?

Two sub-questions flow from this:

1. Is the F1-vs-parameters Pareto frontier dominated by compressed-DONUT, by the pipeline, or by neither (i.e. they trade)?
2. Does the pipeline's per-field interpretability advantage (cross-attention maps over detected text lines) survive when compressed-DONUT becomes parameter-comparable, or is interpretability a separable axis?

## Methodology

The compression sweep uses Pruna AI's open-source library (`pip install pruna`) applied to the fine-tuned DONUT checkpoint produced by the companion repo. Each compression configuration is evaluated on the same canonical 347-image SROIE Task-3 test set, with the same evaluation harness (paired-bootstrap CIs, McNemar test, energy/CO₂eq accounting) used in the source paper, to keep numbers directly comparable.

Planned configurations:

- Int8 quantization (bitsandbytes, row-wise scaling).
- Int4 quantization (bitsandbytes NF4 with double quantization).
- Structured pruning at 25% and 50% sparsity, applied to the BART decoder's feed-forward blocks.
- Teacher-student distillation into a half-depth student (≈130M parameters), KL on decoder logits at τ=2.0.
- Combinations (int8 + 25% prune, distill + int8) — combinations are exposed via Pruna's `smash_config` builders so each cell of the sweep is a single API call.

Each configuration reports: global token-F1, per-field F1, parameter count, peak VRAM, mean inference latency, p95 latency, USD/run, and CO₂eq. All numbers are persisted to `results/compression_grid.csv` and rendered into a Pareto plot (`results/pareto_frontier.png`) that overlays compressed-DONUT against the pipeline and published Task-3 baselines (LayoutLMv3, BROS, TILT, PICK).

Single-seed by default for the initial sweep; the harness is seed-parametric and a multi-seed re-run is queued once the cell layout is finalised.

## Expected deliverables

1. **Pareto plot** (`results/pareto_frontier.png`, `.pdf`) — F1-vs-parameters frontier comparing compressed-DONUT, the pipeline, uncompressed DONUT, and published baselines on SROIE Task-3.
2. **Compression grid table** (`results/compression_grid.csv`) — one row per (compression config, seed) tuple with all reported metrics.
3. **Two-page technical note** (`paper/note.md`, `paper/note.pdf`) — methods, results, and a discussion of which compression operator dominates the Pareto frontier; written so the result stands alone whether positive or negative.
4. **Latency table** (`results/latency.json`) — mean / p50 / p95 / p99 inference latency per configuration on RTX 4090.
5. **Reproducibility bundle** — `make all` regenerates every figure, table, and PDF from a fresh checkout; environment snapshot pinned via `requirements.txt` and `uv.lock`.

## Repo layout

```
pruna-donut-kie/
├── README.md                       # this file
├── Makefile                        # `make all`, `make sweep`, `make plot`, `make paper`
├── requirements.txt
├── notebooks/
│   ├── 01_baseline.ipynb           # uncompressed DONUT on SROIE-347
│   ├── 02_quantize_int8.ipynb      # bitsandbytes int8
│   ├── 03_quantize_int4.ipynb      # bitsandbytes NF4
│   ├── 04_prune_structured.ipynb   # 25% / 50% structured pruning
│   ├── 05_distill.ipynb            # half-depth student
│   └── 06_combinations.ipynb       # Pruna smash combos
├── src/
│   ├── eval.py                     # delegates to kaggle2's eval harness
│   ├── compress.py                 # smash_config builders
│   ├── bench_latency.py            # torch.cuda.Event timing
│   └── plot_pareto.py              # results → Pareto plot
├── scripts/
│   ├── fetch_checkpoint.sh         # downloads fine-tuned DONUT checkpoint
│   └── fetch_sroie.sh              # downloads and sha256-verifies SROIE Task-3 test set
├── data/                           # symlinked SROIE Task-3 (sha256-pinned download in companion repo)
├── checkpoints/                    # fine-tuned DONUT (loaded from companion repo)
├── results/
│   ├── compression_grid.csv
│   ├── pareto_frontier.{png,pdf}
│   ├── latency.json
│   └── per_config/                 # one JSON per compression cell
└── paper/
    ├── note.md
    ├── note.tex
    └── figs/
```

## Reproduction recipe

```bash
git clone https://github.com/aiparallel0/pruna-donut-kie
cd pruna-donut-kie
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
# Download fine-tuned DONUT checkpoint from companion repo
bash scripts/fetch_checkpoint.sh
# Download SROIE Task-3 test set (sha256-verified)
bash scripts/fetch_sroie.sh
# Run the full compression sweep + paper
make all
```

Hardware target: a single RTX 4090 (24 GB VRAM). Approximate cost per full sweep on vast.ai: under USD 15.

## Companion work

- **Source paper and pipeline code**: [aiparallel0/kaggle2](https://github.com/aiparallel0/kaggle2). The fine-tuned DONUT checkpoint and the YOLOv8 + TrOCR + attention-assigner pipeline live there. This repo treats `kaggle2` as upstream; the eval harness is imported, not duplicated.
- **Production deployment**: [image-to-text.fit](https://image-to-text.fit) — the fine-tuned DONUT pipeline serving real users for receipt → structured-JSON inference. Latency and energy questions surfaced in production motivated this compression study.
- **Pruna AI**: [PrunaAI/pruna](https://github.com/PrunaAI/pruna) is the upstream compression framework used here.

## Status

**Done**

- Repo skeleton and README.
- `src/eval.py` wired to the companion repo's evaluation harness.
- `notebooks/01_baseline.ipynb` reproducing uncompressed DONUT F1 on SROIE-347.

**In progress**

- `notebooks/02_quantize_int8.ipynb` — Pruna int8 sweep.
- `src/bench_latency.py` — torch.cuda.Event-based latency benchmarking on RTX 4090.

**Planned**

- Notebooks 03–06 (int4, pruning, distillation, combinations).
- `src/plot_pareto.py` and the headline Pareto plot.
- Two-page technical note (`paper/note.{md,pdf}`).
- Multi-seed re-run (n=5) once cell layout is final.

## Citation

```bibtex
@misc{bektes2026prunadonutkie,
  author       = {Bekte\c{s}, Efe},
  title        = {pruna-donut-kie: Compression study of fine-tuned {DONUT} for receipt {KIE} on {SROIE} {Task-3}},
  year         = {2026},
  howpublished = {\url{https://github.com/aiparallel0/pruna-donut-kie}},
  note         = {Extends \url{https://github.com/aiparallel0/kaggle2}}
}
```

## License

MIT — see [LICENSE](LICENSE).
