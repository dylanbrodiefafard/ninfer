"""Mechanical artifact-writing helpers shared by Qwen3.6-family converters.

This module deliberately knows neither target identity nor checkpoint geometry.
Target packages own their config validation, object inventory, source recipe,
and conversion entry point.
"""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Mapping
from pathlib import Path

import torch

from tools.artifact.layouts import encode_direct
from tools.convert.common.quantize import quantize_and_encode

from .inventory import DIRECT_FORMATS, TensorSpec


def load_json(path: str | Path) -> dict[str, object]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def check_members(
    scope: str,
    actual: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    """Require the target-selected members while allowing unrelated metadata."""

    mismatches = [
        f"{scope}.{name}: expected {value!r}, got {actual.get(name)!r}"
        for name, value in expected.items()
        if actual.get(name) != value
    ]
    if mismatches:
        raise ValueError("checkpoint config mismatch:\n  " + "\n  ".join(mismatches))


def encode_tensor_payload(
    tensor: torch.Tensor,
    spec: TensorSpec,
    device: str | torch.device,
) -> bytes:
    if spec.format in DIRECT_FORMATS:
        return encode_direct(tensor, spec.format)
    return quantize_and_encode(tensor, spec.format, device=device)


def converter_revision(repo_root: str | Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def environment(device: torch.device) -> dict[str, object]:
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "resolved_device": str(device),
        "gpu": gpu,
    }


__all__ = [
    "check_members",
    "converter_revision",
    "encode_tensor_payload",
    "environment",
    "load_json",
]
