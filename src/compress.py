"""Pruna compression configuration builders for the DONUT sweep.

Each public function returns a configured pruna.SmashConfig ready to be
passed to pruna.smash(). No compression is applied here; this module only
builds configuration objects.

Returns: pruna.SmashConfig instances (one per compression strategy).
"""

from __future__ import annotations

from typing import Any

import pruna  # type: ignore[import-untyped]


def int8() -> Any:
    """Return a SmashConfig for bitsandbytes int8 row-wise quantization."""
    cfg = pruna.SmashConfig()
    cfg["quantizer"] = "bitsandbytes_int8"
    return cfg


def int4() -> Any:
    """Return a SmashConfig for bitsandbytes NF4 with double quantization."""
    cfg = pruna.SmashConfig()
    cfg["quantizer"] = "bitsandbytes_nf4"
    cfg["double_quant"] = True
    return cfg


def prune25() -> Any:
    """Return a SmashConfig for 25% structured pruning on BART FF blocks."""
    cfg = pruna.SmashConfig()
    cfg["pruner"] = "structured"
    cfg["sparsity"] = 0.25
    cfg["target_modules"] = ["encoder.layers.*.fc1", "decoder.layers.*.fc1"]
    return cfg


def prune50() -> Any:
    """Return a SmashConfig for 50% structured pruning on BART FF blocks."""
    cfg = pruna.SmashConfig()
    cfg["pruner"] = "structured"
    cfg["sparsity"] = 0.50
    cfg["target_modules"] = ["encoder.layers.*.fc1", "decoder.layers.*.fc1"]
    return cfg


def distill_half() -> Any:
    """Return a SmashConfig for half-depth student distillation at τ=2.0."""
    cfg = pruna.SmashConfig()
    cfg["distiller"] = "kl_divergence"
    cfg["student_depth_ratio"] = 0.5
    cfg["temperature"] = 2.0
    return cfg


def int8_prune25() -> Any:
    """Return a SmashConfig combining int8 quantization and 25% structured pruning."""
    cfg = pruna.SmashConfig()
    cfg["quantizer"] = "bitsandbytes_int8"
    cfg["pruner"] = "structured"
    cfg["sparsity"] = 0.25
    cfg["target_modules"] = ["encoder.layers.*.fc1", "decoder.layers.*.fc1"]
    return cfg


def distill_int8() -> Any:
    """Return a SmashConfig combining half-depth distillation and int8 quantization."""
    cfg = pruna.SmashConfig()
    cfg["distiller"] = "kl_divergence"
    cfg["student_depth_ratio"] = 0.5
    cfg["temperature"] = 2.0
    cfg["quantizer"] = "bitsandbytes_int8"
    return cfg
