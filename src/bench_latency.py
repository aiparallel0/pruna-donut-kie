"""Latency benchmarking for compression sweep configurations.

Uses torch.cuda.Event for microsecond-accurate GPU timing on a single
RTX 4090 (24 GB VRAM). Records mean, p50, p95, and p99 latency per
compression configuration and writes results/latency.json.

Returns: nothing (side-effect: writes results/latency.json).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import torch

RESULTS_DIR = Path("results")
LATENCY_JSON = RESULTS_DIR / "latency.json"

# Number of warm-up and measurement passes.
N_WARMUP: int = 5
N_RUNS: int = 50


def measure_latency(
    model: Any,
    inputs: dict[str, torch.Tensor],
    *,
    n_warmup: int = N_WARMUP,
    n_runs: int = N_RUNS,
    device: str = "cuda",
) -> dict[str, float]:
    """Measure inference latency using torch.cuda.Event timing.

    Parameters
    ----------
    model:
        A HuggingFace model in eval mode, already on ``device``.
    inputs:
        Tokenised/processed inputs as a dict of tensors, already on ``device``.
    n_warmup:
        Number of discarded warm-up passes before timing starts.
    n_runs:
        Number of timed measurement passes.
    device:
        CUDA device string.  CPU timing is not supported.

    Returns
    -------
    dict with keys: mean_ms, p50_ms, p95_ms, p99_ms.
    Raises RuntimeError if CUDA is not available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for latency benchmarking. "
            "Run on a machine with a GPU (target: RTX 4090)."
        )

    model.eval()
    latencies_ms: list[float] = []

    with torch.no_grad():
        # Warm-up
        for _ in range(n_warmup):
            model(**inputs)
        torch.cuda.synchronize()

        # Timed runs
        for _ in range(n_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()  # type: ignore[no-untyped-call]
            model(**inputs)
            end.record()  # type: ignore[no-untyped-call]
            torch.cuda.synchronize()
            latencies_ms.append(start.elapsed_time(end))

    sorted_ms = sorted(latencies_ms)
    n = len(sorted_ms)
    return {
        "mean_ms": statistics.mean(sorted_ms),
        "p50_ms": sorted_ms[int(0.50 * n)],
        "p95_ms": sorted_ms[int(0.95 * n)],
        "p99_ms": sorted_ms[int(0.99 * n)],
    }


def save_latency(results: dict[str, dict[str, float]], path: Path = LATENCY_JSON) -> None:
    """Persist per-configuration latency dict to JSON.

    Parameters
    ----------
    results:
        Mapping from config_name to the dict returned by measure_latency.
    path:
        Output path (created if parent directories exist).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Latency results written to {path}")
