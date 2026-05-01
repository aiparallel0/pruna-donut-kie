"""Pareto plot generator for the pruna-donut-kie compression sweep.

Reads results/compression_grid.csv and results/competitors.json, then
produces results/pareto_frontier.png and results/pareto_frontier.pdf.

X-axis: parameter count in millions (log scale).
Y-axis: global token-F1 on SROIE Task-3 (347 images).

Four sets of points are plotted:
  (a) Compressed-DONUT configurations from compression_grid.csv.
  (b) Baseline uncompressed DONUT — a single point (260.78M, 0.827).
  (c) YOLOv8 + TrOCR + attention-assigner pipeline — single point
      (66.93M, 0.858) with paired-bootstrap 95% CI whisker.
  (d) Published competitors loaded from results/competitors.json
      (LayoutLMv3, BROS, TILT, PICK).

The Pareto frontier (maximise F1, minimise parameter count) is annotated
as a step-line.  A colour-blind-safe palette (Wong 2011) is used throughout.

Returns: nothing (side-effects: writes PNG and PDF to results/).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("results")
GRID_CSV = RESULTS_DIR / "compression_grid.csv"
COMPETITORS_JSON = RESULTS_DIR / "competitors.json"
OUT_PNG = RESULTS_DIR / "pareto_frontier.png"
OUT_PDF = RESULTS_DIR / "pareto_frontier.pdf"

# ---------------------------------------------------------------------------
# Wong (2011) colour-blind-safe palette
# ---------------------------------------------------------------------------
WONG = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
}

# Known baseline figures (from the companion study; do not alter).
DONUT_PARAMS_M: float = 260.78
DONUT_F1: float = 0.827

PIPELINE_PARAMS_M: float = 66.93
PIPELINE_F1: float = 0.858
# Paired-bootstrap 95% CI on ΔF1 (DONUT − pipeline); converted to absolute
# pipeline whisker: ±half-width of CI projected onto pipeline F1.
PIPELINE_CI_LOW: float = 0.858 + 0.0087   # upper bound (less negative delta)
PIPELINE_CI_HIGH: float = 0.858 + 0.0529  # would be above DONUT if positive


def _pareto_frontier(
    params: np.ndarray, f1: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (params, f1) of the Pareto-optimal points (min params, max F1).

    A point is Pareto-optimal if no other point has strictly fewer parameters
    *and* strictly higher F1.
    """
    order = np.argsort(params)
    params_s = params[order]
    f1_s = f1[order]
    frontier_mask = np.zeros(len(params_s), dtype=bool)
    best_f1 = -np.inf
    for i, (p, f) in enumerate(zip(params_s, f1_s)):
        if f >= best_f1:
            frontier_mask[i] = True
            best_f1 = f
    return params_s[frontier_mask], f1_s[frontier_mask]


def load_grid(path: Path) -> pd.DataFrame:
    """Load compression_grid.csv; raise if missing or malformed.

    Expects columns: config_name, param_count_m, global_f1.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'make sweep' before 'make plot'."
        )
    df = pd.read_csv(path)
    required = {"config_name", "param_count_m", "global_f1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}. "
            "Check that all sweep notebooks have completed successfully."
        )
    if df.empty:
        raise ValueError(f"{path} contains no rows. Run 'make sweep' first.")
    return df


def load_competitors(path: Path) -> list[dict[str, object]]:
    """Load published competitor results; raise if file is missing.

    Each entry must have keys: name, param_count_m, global_f1.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Create it with published competitor figures "
            "before running 'make plot'."
        )
    with path.open() as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a JSON array.")
    return data  # type: ignore[return-value]


def plot(
    grid: pd.DataFrame,
    competitors: list[dict[str, object]],
    out_png: Path,
    out_pdf: Path,
) -> None:
    """Render the Pareto plot and save PNG + PDF.

    Parameters
    ----------
    grid:
        DataFrame with one row per compression config (from compression_grid.csv).
    competitors:
        List of published-baseline dicts loaded from competitors.json.
    out_png:
        Output path for the PNG figure.
    out_pdf:
        Output path for the PDF figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # --- (a) Compressed-DONUT configurations ---
    for _, row in grid.iterrows():
        ax.scatter(
            row["param_count_m"],
            row["global_f1"],
            color=WONG["orange"],
            zorder=5,
            s=60,
            label="Compressed DONUT" if _ == grid.index[0] else "",
        )
        ax.annotate(
            str(row["config_name"]),
            (row["param_count_m"], row["global_f1"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )

    # --- Pareto frontier over compressed configs ---
    if len(grid) > 0:
        pf_params, pf_f1 = _pareto_frontier(
            grid["param_count_m"].to_numpy(), grid["global_f1"].to_numpy()
        )
        # Extend the step-line to the right for visual clarity
        step_params = np.concatenate([pf_params, [pf_params[-1] * 1.1]])
        step_f1 = np.concatenate([pf_f1, [pf_f1[-1]]])
        ax.step(
            step_params,
            step_f1,
            where="post",
            color=WONG["orange"],
            linewidth=1.2,
            linestyle="--",
            alpha=0.7,
            label="Pareto frontier (compressed DONUT)",
        )

    # --- (b) Baseline uncompressed DONUT ---
    ax.scatter(
        DONUT_PARAMS_M,
        DONUT_F1,
        color=WONG["blue"],
        marker="D",
        zorder=6,
        s=80,
        label=f"DONUT (uncompressed, {DONUT_PARAMS_M:.0f}M)",
    )

    # --- (c) Pipeline with 95% CI whisker ---
    # The CI is on ΔF1 = DONUT_F1 − pipeline_F1, so pipeline whisker is
    # the CI bounds reflected: pipeline + (−CI_bound on delta).
    ci_low_pipeline = PIPELINE_F1 - 0.0529   # lower bound on pipeline F1
    ci_high_pipeline = PIPELINE_F1 + 0.0087  # upper bound on pipeline F1
    ax.errorbar(
        PIPELINE_PARAMS_M,
        PIPELINE_F1,
        yerr=[[PIPELINE_F1 - ci_low_pipeline], [ci_high_pipeline - PIPELINE_F1]],
        fmt="s",
        color=WONG["bluish_green"],
        zorder=6,
        markersize=8,
        capsize=4,
        label=f"YOLOv8+TrOCR+assigner pipeline ({PIPELINE_PARAMS_M:.2f}M)",
    )

    # Vertical reference line at pipeline parameter floor
    ax.axvline(
        PIPELINE_PARAMS_M,
        color=WONG["bluish_green"],
        linewidth=0.8,
        linestyle=":",
        alpha=0.6,
    )
    ax.annotate(
        f"Pipeline floor\n{PIPELINE_PARAMS_M:.1f}M",
        (PIPELINE_PARAMS_M, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.70),
        textcoords="offset points",
        xytext=(4, 6),
        fontsize=7,
        color=WONG["bluish_green"],
    )

    # --- (d) Published competitors ---
    comp_colors = [
        WONG["vermillion"],
        WONG["reddish_purple"],
        WONG["sky_blue"],
        WONG["black"],
    ]
    for i, comp in enumerate(competitors):
        if comp.get("global_f1") is None:
            # Skip competitors whose F1 has not yet been verified from source paper.
            print(
                f"WARNING: skipping competitor '{comp['name']}' — "
                "global_f1 is null in results/competitors.json. "
                "Fill in the verified value from the source paper before plotting."
            )
            continue
        ax.scatter(
            comp["param_count_m"],
            comp["global_f1"],
            color=comp_colors[i % len(comp_colors)],
            marker="^",
            zorder=4,
            s=60,
            label=str(comp["name"]),
        )

    # --- Axes ---
    ax.set_xscale("log")
    ax.set_xlabel("Parameter count (M, log scale)", fontsize=11)
    ax.set_ylabel("Global token-F1 — SROIE Task-3 (347 images)", fontsize=11)
    ax.set_title(
        "F1 vs. parameter count: compressed DONUT, pipeline, and published baselines",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


def main() -> None:
    """Entry point: load data, validate, render plot."""
    grid = load_grid(GRID_CSV)
    competitors = load_competitors(COMPETITORS_JSON)
    plot(grid, competitors, OUT_PNG, OUT_PDF)


if __name__ == "__main__":
    main()
