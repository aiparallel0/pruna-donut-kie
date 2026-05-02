# Paper 4 session prompt — future paper-4 repo

You are creating a new repository for Paper 4 — a compression study on
the SVKIE (Structure-Verified Document KIE) system from Paper 3 of the
same research programme.

**GOAL:** produce a complete, ready-to-train repository plus an
IEEE-format paper template framed as future work. This repo is a
SKELETON: no empirical results yet. Results will be filled in a later
GPU run.

**THESIS OF PAPER 4:** "Compression Crossover for Document KIE:
applying parameter-efficient techniques (int8/int4 quantization,
structured pruning, distillation) to the SVKIE pipeline, with a Pareto
frontier of F1 versus on-device memory footprint that demonstrates real
deployment-efficiency without abandoning structural verification."

## Constraints

- This paper has NO empirical results in this PR. All numerical
  claims must be marked clearly as "future work" or "expected results"
  in the manuscript. No fabricated F1 cells.
- The paper must read as a complete, standalone contribution on the
  efficiency axis. Do NOT underplay it. Use full research-paper
  language with future-tense empirical claims.
- The repository must be ready to run: Pruna OSS integration scaffolded,
  quantization configs prepared, evaluation harness mirroring the SVKIE
  evaluation pipeline.
- Format: IEEE Access conference template (or NeurIPS workshop style
  if pivoting to ENLSP / ES-FoMo).

## Deliverables

1. Repository structure:
   - `models/`     (Pruna integration, compression configs)
   - `configs/`    (sweep specs: int8, int4, prune-25, prune-50,
                    distill-half, combinations)
   - `stages/`     (compress, eval, paper)
   - `report/`     (LaTeX paper template + sections)
   - `scripts/`    (`run_compression_sweep.sh`)
   - `tests/`      (Pruna config validation; no GPU required)
2. LaTeX paper template with:
   - Abstract framed as future work
   - Methodology section describing each compression technique
   - Empty results tables with "to be measured" cells
   - Theoretical analysis of the F1-vs-memory tradeoff
   - Discussion of the SVKIE-specific constraints (verifier
     must survive compression)
3. `README.md` with the run-this-on-vastai instructions for when the
   compression sweep is executed.

## Architectural considerations

- The Pareto plot's x-axis must be ON-DEVICE MEMORY FOOTPRINT (MB),
  NOT parameter count. Quantization changes bytes-per-parameter, not
  parameter count; using parameter count would be conceptually wrong.
- The compression must preserve the FOCUS-Σ verifier's correctness;
  the verifier is rule-based DP, not a neural network, so quantization
  does not affect it. The compression target is the upstream
  cross-attention assigner + DONUT/LayoutLMv3 backbones.
- A dedicated "compression-aware verification" subsection should
  argue that the structural verifier is itself a regularizer against
  catastrophic compression failure: if the verifier rejects a
  candidate post-compression, fall back to the rule-based path.

**START BY SCAFFOLDING THE REPOSITORY STRUCTURE** and the LaTeX
template. Do not implement Pruna calls until the structure is reviewed.
