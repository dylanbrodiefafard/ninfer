"""Convert the pinned UD-IQ1_S GGUF shards into one verifier `.ninfer`.

The operation is byte-preserving: it parses and validates the GGUF directories,
then copies each encoded tensor span into the native artifact container.  GGUF
metadata and parsing do not enter the C++ runtime.

Canonical invocation::

    python3 -m tools.convert.qwen4.convert \
      --model /path/to/qwen4-preview-gguf/UD-IQ1_S \
      --out /path/to/qwen4_ud_iq1_s_verify.ninfer
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Iterable, Sequence

from tools.artifact.container import ArtifactIdentity, ArtifactWriter, TensorSpec

from .gguf import GGUFFile, TensorInfo, read_gguf
from . import inventory


RECIPE_ID = "qwen4_ud_iq1_s_verify-v1"
SHARDS = (
    (
        "Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf",
        10_946_624,
        "88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd",
        0,
    ),
    (
        "Qwen3.8-Flash-Next-UD-IQ1_S-00002-of-00003.gguf",
        49_990_818_368,
        "3a62e35bbf9add4733bd1438ebd3a67649d5edd6cb0e72bb78e33c913992b2b6",
        595,
    ),
    (
        "Qwen3.8-Flash-Next-UD-IQ1_S-00003-of-00003.gguf",
        22_544_696_352,
        "0e25ceaeb89b8a80aa973c6c0c7448943682f7408c2855b2ebd016b7643a861a",
        629,
    ),
)
EXPECTED_FORMAT_COUNTS = {
    "BF16": 24,
    "FP32": 557,
    "IQ1_S": 68,
    "IQ2_XXS": 28,
    "IQ4_NL": 49,
    "Q4_K": 2,
    "Q5_K": 212,
    "Q6_K": 40,
    "Q8_0": 244,
}
EXPECTED_FORMAT_BYTES = {
    "BF16": 39_321_600,
    "FP32": 313_495_040,
    "IQ1_S": 11_141_120_000,
    "IQ2_XXS": 6_055_526_400,
    "IQ4_NL": 51_449_379_840,
    "Q4_K": 715_161_600,
    "Q5_K": 1_526_046_720,
    "Q6_K": 501_580_800,
    "Q8_0": 793_804_800,
}
EXPECTED_TENSOR_BYTES = 72_535_436_800
EXPECTED_MAPPED_BYTES = 45_996_784_640
EXPECTED_DEVICE_BYTES = 26_538_652_160
_COPY_CHUNK = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class SourceTensor:
    shard: GGUFFile
    tensor: TensorInfo


@dataclass(frozen=True, slots=True)
class Preflight:
    model_dir: Path
    shards: tuple[GGUFFile, ...]
    source_by_name: dict[str, SourceTensor]
    shard_hashes: dict[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(_COPY_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _require_metadata(shards: Sequence[GGUFFile]) -> None:
    for index, shard in enumerate(shards):
        metadata = shard.metadata
        expected = {
            "split.no": index,
            "split.count": 3,
            "split.tensors.count": 1224,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"{shard.path.name}: {key} must be {value!r}, got {metadata.get(key)!r}"
                )
    root = shards[0].metadata
    if root.get("general.architecture") != "qwen4exp":
        raise ValueError("GGUF architecture must be qwen4exp")
    if root.get("qwen4exp.block_count") != 48:
        raise ValueError("GGUF qwen4exp.block_count must be 48")
    if root.get("general.quantization_version") != 2:
        raise ValueError("GGUF quantization version must be 2")


def preflight(model_dir: str | Path, *, verify_hashes: bool = True) -> Preflight:
    model = Path(model_dir)
    parsed: list[GGUFFile] = []
    hashes: dict[str, str] = {}
    for name, expected_bytes, expected_hash, expected_tensors in SHARDS:
        path = model / name
        if not path.is_file():
            raise ValueError(f"missing pinned GGUF shard: {path}")
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"{name}: expected {expected_bytes} bytes, got {actual_bytes}"
            )
        shard = read_gguf(path)
        if len(shard.tensors) != expected_tensors:
            raise ValueError(
                f"{name}: expected {expected_tensors} tensors, got {len(shard.tensors)}"
            )
        parsed.append(shard)
        if verify_hashes:
            actual_hash = _sha256(path)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"{name}: SHA-256 mismatch; expected {expected_hash}, got {actual_hash}"
                )
            hashes[name] = actual_hash

    _require_metadata(parsed)
    sources: dict[str, SourceTensor] = {}
    for shard in parsed:
        for tensor in shard.tensors:
            if tensor.name in sources:
                raise ValueError(f"duplicate tensor across GGUF shards: {tensor.name}")
            sources[tensor.name] = SourceTensor(shard, tensor)

    expected_names = set(inventory.TENSORS_BY_NAME)
    actual_names = set(sources)
    if expected_names != actual_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(
            "GGUF tensor inventory mismatch: "
            f"missing={missing[:1]!r}, extra={extra[:1]!r}"
        )
    for name, spec in inventory.TENSORS_BY_NAME.items():
        actual = sources[name].tensor
        if actual.shape != spec.shape or actual.format != spec.format:
            raise ValueError(
                f"{name}: expected {spec.format} {spec.shape}, "
                f"got {actual.format} {actual.shape}"
            )

    counts = Counter(source.tensor.format for source in sources.values())
    format_bytes: Counter[str] = Counter()
    for source in sources.values():
        format_bytes[source.tensor.format] += source.tensor.bytes
    tensor_bytes = sum(format_bytes.values())
    mapped_bytes = sum(
        sources[spec.name].tensor.bytes
        for spec in inventory.TENSOR_SPECS
        if spec.mapped_host
    )
    if dict(counts) != EXPECTED_FORMAT_COUNTS:
        raise ValueError(f"GGUF format counts changed: {dict(counts)!r}")
    if dict(format_bytes) != EXPECTED_FORMAT_BYTES:
        raise ValueError(f"GGUF format byte totals changed: {dict(format_bytes)!r}")
    if tensor_bytes != EXPECTED_TENSOR_BYTES or mapped_bytes != EXPECTED_MAPPED_BYTES:
        raise ValueError("GGUF tensor placement byte totals changed")
    if tensor_bytes - mapped_bytes != EXPECTED_DEVICE_BYTES:
        raise ValueError("GGUF device-resident byte total changed")
    return Preflight(model.resolve(), tuple(parsed), sources, hashes)


def _artifact_specs() -> tuple[TensorSpec, ...]:
    return tuple(
        TensorSpec(spec.name, spec.shape, spec.format, spec.layout)
        for spec in inventory.TENSOR_SPECS
    )


def _payload_chunks(source: SourceTensor) -> Iterable[bytes]:
    begin, end = source.shard.absolute_span(source.tensor)
    with source.shard.path.open("rb") as file:
        file.seek(begin)
        remaining = end - begin
        while remaining:
            chunk = file.read(min(remaining, _COPY_CHUNK))
            if not chunk:
                raise IOError(f"short read from {source.shard.path}")
            remaining -= len(chunk)
            yield chunk


def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    verify_hashes: bool = True,
) -> Path:
    started = time.perf_counter()
    checked = preflight(model_dir, verify_hashes=verify_hashes)
    output = Path(out_path)
    if output.exists():
        raise ValueError(f"refusing to overwrite existing artifact: {output}")
    report_path = Path(str(output) + ".conversion.json")
    if report_path.exists():
        raise ValueError(f"refusing to overwrite existing report: {report_path}")
    output.parent.mkdir(parents=True, exist_ok=True)

    specs = _artifact_specs()
    with ArtifactWriter(
        output,
        ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        specs,
    ) as writer:
        for index, spec in enumerate(inventory.TENSOR_SPECS, 1):
            writer.write(spec.name, _payload_chunks(checked.source_by_name[spec.name]))
            print(f"[{index}/{len(specs)}] {spec.name}", flush=True)

    final_bytes = output.stat().st_size
    report = {
        "recipe_id": RECIPE_ID,
        "identity": {
            "model_id": inventory.MODEL_ID,
            "weights_id": inventory.WEIGHTS_ID,
        },
        "source": {
            "model_dir": str(checked.model_dir),
            "shards": [
                {
                    "name": shard.path.name,
                    "bytes": shard.file_bytes,
                    "sha256": checked.shard_hashes.get(shard.path.name),
                    "tensors": len(shard.tensors),
                }
                for shard in checked.shards
            ],
        },
        "artifact": {
            "path": str(output.resolve()),
            "bytes": final_bytes,
            "tensors": len(specs),
            "tensor_bytes": EXPECTED_TENSOR_BYTES,
            "mapped_host_tensor_bytes": EXPECTED_MAPPED_BYTES,
            "device_tensor_bytes": EXPECTED_DEVICE_BYTES,
            "format_counts": EXPECTED_FORMAT_COUNTS,
            "format_bytes": EXPECTED_FORMAT_BYTES,
        },
        "profile": {
            "registered": False,
            "batch": 1,
            "max_context": 4096,
            "qsa_kv": "NVFP4-G16",
            "mapped_host": ["per_layer_token_embd", "routed_gate", "routed_up"],
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
        file.write("\n")
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="skip the expensive shard hash pass (structural checks still run)",
    )
    args = parser.parse_args(argv)
    convert(args.model, args.out, verify_hashes=not args.skip_hash)


if __name__ == "__main__":
    main()


__all__ = ["Preflight", "SourceTensor", "convert", "main", "preflight"]
