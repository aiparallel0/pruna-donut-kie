"""Evaluation harness for SROIE Task-3.

Delegates to the companion repo (aiparallel0/kaggle2) evaluation harness.
Exports compute_f1, load_donut, and load_test_set so that each notebook
cell can call them without duplicating logic.

Returns: see individual function docstrings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# The companion repo (pip install -e ../kaggle2 or PYTHONPATH=../kaggle2)
# publishes exactly these names.
from kaggle2.eval import compute_f1, load_donut, load_test_set  # type: ignore[import-untyped]

__all__ = ["compute_f1", "load_donut", "load_test_set", "evaluate_model"]


def evaluate_model(
    model: Any,
    processor: Any,
    *,
    test_dir: Path = Path("data/test"),
    device: str = "cuda",
) -> dict[str, Any]:
    """Run the full eval loop and return a metrics dict.

    Parameters
    ----------
    model:
        A HuggingFace model compatible with the DONUT inference API.
    processor:
        Corresponding DonutProcessor.
    test_dir:
        Path to the 347-image SROIE Task-3 test split (sha256-pinned).
    device:
        Torch device string passed through to kaggle2's harness.

    Returns
    -------
    dict with keys: global_f1, per_field_f1, param_count, config_name.
    Raises FileNotFoundError if test_dir does not exist.
    """
    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test directory not found: {test_dir}. "
            "Run 'bash scripts/fetch_sroie.sh' first."
        )

    test_set = load_test_set(test_dir)
    metrics: dict[str, Any] = compute_f1(
        model=model,
        processor=processor,
        dataset=test_set,
        device=device,
    )
    return metrics
