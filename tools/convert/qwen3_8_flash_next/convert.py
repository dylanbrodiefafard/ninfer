"""Convert Qwen3.8-Flash-Next into the `qwen3.8-flash-next/f3-fp8-nvfp4` `.ninfer` artifact.

Inputs are explicit paths: the BF16 Hugging Face checkpoint (dense tensors, config, frontend
files; its routed-expert tensors are never read), the expert-format study's packed
`mixed_mse_gptq` store, and the dense-format study's record and results directories. Routed
experts are repacked into `moe-expert-tiles-v1` records without changing a code, scale, book, or
multiplier word. The dense-format study's recipe `fallback_mid` (with `ATTN_PARENT.md` and
`HC.md`) is ingested word for word: FP8 GDN projections, attention output, head and
hyper-connection down/up, NVFP4 attention-input parents and NVFP4 shared-expert records. Every
other tensor, including the BF16 token embedding, is an exact copy of its source words.

    python3 -m tools.convert.qwen3_8_flash_next.convert \\
      --model /home/battlefront/ai/qwen3.8-flash-next-bf16 \\
      --experts /home/battlefront/ai/study/quant/mixed_mse_gptq/experts.bin \\
      --experts-manifest /home/battlefront/ai/study/quant/mixed_mse_gptq/manifest.json \\
      --dense-records /home/battlefront/ai/study/quant/dense \\
      --dense-results /home/battlefront/ai/study/results/dense \\
      --out models/qwen3.8-flash-next-f3-fp8-nvfp4.ninfer

The output must not exist. After writing, the converter re-reads the artifact and validates every
object against its sources (`validate.py`), then writes `<out>.conversion.json`.
`--validate-only` runs that validation on an existing artifact and writes
`<out>.validation.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tools.artifact.container import ArtifactIdentity, ArtifactWriter, TensorSpec
from tools.artifact.layouts import encode_direct, encode_expert_tiles_record
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion

from .inventory import (
    E2M1,
    E2M1_RECORDS,
    EXPERT_CODECS,
    F3,
    F3_CODEBOOKS,
    F3_RECORDS,
    FRONTEND_RESOURCES,
    MATRIX_SCALES,
    MODEL_ID,
    STUDY_RECIPE,
    STUDY_TRANSFORMS,
    WEIGHTS_ID,
    ObjectPlan,
    Transform,
    build_inventory,
)
from .materialize import encode_payload, encode_study_payload, materialize
from .study_dense import DenseStudy, check_record
from .study_store import EXPERTS, HIDDEN, INTERMEDIATE, LAYERS, ExpertCodec, StudyExpertStore
from .validate import validate_artifact

RECIPE_ID = "qwen3_8_flash_next_f3_fp8_nvfp4-v1"
_THREADS = 8

_TEXT_CONFIG = {
    "hidden_size": HIDDEN,
    "num_hidden_layers": LAYERS,
    "full_attention_interval": 4,
    "num_experts": EXPERTS,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": INTERMEDIATE,
    "shared_expert_intermediate_size": INTERMEDIATE,
    "hc_count": 4,
    "hc_lowrank": 320,
    "head_dim": 256,
    "num_attention_heads": 24,
    "num_key_value_heads": 2,
    "indexer_head_dim": 128,
    "indexer_n_heads": 4,
    "indexer_kv_heads": 1,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "ple_conv_kernel_size": 4,
    "ple_embed_dim": HIDDEN,
    "ple_layer_ids": [2],
    "ngram_size": 3,
    "vocab_size": 248320,
    "tie_word_embeddings": False,
    "output_gate_type": "sigmoid",
}


def _log(message: str) -> None:
    print(message, flush=True)


def validate_config(model_dir: Path) -> dict[str, object]:
    config = family_conversion.load_json(model_dir / "config.json")
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise ValueError("config.json has no text_config")
    family_conversion.check_members("text_config", text, _TEXT_CONFIG)
    expected_types = [
        "full_attention" if layer % 4 == 3 else "linear_attention" for layer in range(LAYERS)
    ]
    if text.get("layer_types") != expected_types:
        raise ValueError("text_config.layer_types is not three GDN layers per QSA layer")
    return {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        **{name: text[name] for name in _TEXT_CONFIG},
    }


def _frontend_bytes(model_dir: Path) -> dict[str, int]:
    return {name: (model_dir / name).stat().st_size for name in FRONTEND_RESOURCES}


@dataclass(frozen=True, slots=True)
class Sources:
    """Explicit input paths."""

    model: Path
    experts: Path
    experts_manifest: Path
    dense_records: Path
    dense_results: Path

    def dense_study(self) -> DenseStudy:
        return DenseStudy(self.dense_records, self.dense_results)


_CHECKPOINT_TRANSFORMS = (
    frozenset(Transform) - STUDY_TRANSFORMS - {Transform.RESOURCE, Transform.ROUTED_EXPERTS}
)


def preflight_sources(
    plans: tuple[ObjectPlan, ...], reader: ShardReader, study: DenseStudy
) -> None:
    """Require every checkpoint tensor (with its dtype) and study record before any output."""

    names = sorted(
        {
            name
            for plan in plans
            if plan.transform in _CHECKPOINT_TRANSFORMS
            for name in plan.sources
        }
        | {name for plan in plans for name in plan.reference}
    )
    missing = [name for name in names if not reader.has(name)]
    if missing:
        raise ValueError("source checkpoint lacks: " + ", ".join(missing[:8]))
    metadata = reader.metadata(names)
    wrong = [
        name
        for name, item in metadata.items()
        if item.dtype
        != ("I64" if name.endswith(("multipliers", "offsets", "vocab_sizes")) else "BF16")
    ]
    if wrong:
        raise ValueError("unexpected source dtypes: " + ", ".join(wrong[:8]))
    records = sorted(
        {name for plan in plans if plan.transform in STUDY_TRANSFORMS for name in plan.sources}
    )
    for record in records:
        check_record(study, record)


def _expert_records(
    store: StudyExpertStore,
    codec: ExpertCodec,
    numeric_format: str,
    scale_table: np.ndarray,
    filled: np.ndarray,
) -> Iterator[memoryview]:
    """Yield the records of one codec in (layer, expert) order and fill their scale entries."""

    for layer in range(LAYERS):
        for expert in range(EXPERTS):
            if store.codec(layer, expert) is not codec:
                continue
            source = store.read(layer, expert)
            scale_table[layer, expert] = source.matrix_scale_words
            filled[layer, expert] = True
            record = encode_expert_tiles_record(source.words, numeric_format, HIDDEN, INTERMEDIATE)
            yield memoryview(record.numpy())
        _log(f"  {numeric_format} records: layer {layer} written")


def _routed_tables(
    store: StudyExpertStore, scale_table: np.ndarray, filled: np.ndarray
) -> dict[str, bytes]:
    if not filled.all():
        raise ValueError("matrix-scale table is incomplete")
    words = scale_table.view(np.float32)
    if not np.isfinite(words).all() or (words < 0).any():
        raise ValueError("study matrix scales must be finite and nonnegative")
    scales = torch.from_numpy(words.copy())
    return {
        EXPERT_CODECS: encode_direct(torch.from_numpy(store.codecs.astype(np.int32)), "I32"),
        F3_CODEBOOKS: encode_direct(
            torch.from_numpy(store.book_words.astype(np.uint32).view(np.int32)), "I32"
        ),
        MATRIX_SCALES: encode_direct(scales, "FP32"),
    }


def _inventory_summary(plans: tuple[ObjectPlan, ...], objects) -> dict[str, object]:
    by_family: Counter[str] = Counter()
    by_format: Counter[str] = Counter()
    for plan, obj in zip(plans, objects, strict=True):
        by_family[plan.family.value] += obj.bytes
        key = plan.spec.format if isinstance(plan.spec, TensorSpec) else plan.spec.encoding
        by_format[key] += obj.bytes
    return {
        "objects": len(plans),
        "tensors": sum(isinstance(plan.spec, TensorSpec) for plan in plans),
        "resources": sum(not isinstance(plan.spec, TensorSpec) for plan in plans),
        "payload_bytes_by_family": dict(sorted(by_family.items())),
        "payload_bytes_by_format": dict(sorted(by_format.items())),
        "payload_bytes": sum(obj.bytes for obj in objects),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _peak_rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _source_record(sources: Sources, store: StudyExpertStore) -> dict[str, object]:
    record: dict[str, object] = {
        "model": str(sources.model.resolve()),
        "safetensors_index_sha256": _sha256(sources.model / "model.safetensors.index.json"),
        "routed_experts": {
            "store": str(store.path.resolve()),
            "manifest": str(store.manifest_path.resolve()),
            "manifest_sha256": _sha256(store.manifest_path),
            "store_sha256_recorded_by_manifest": store.manifest.get("sha256"),
            "format": store.manifest.get("format"),
            "nvfp4_experts": store.nvfp4_experts,
            "f3_experts": store.f3_experts,
        },
    }
    record["dense_study"] = {
        "recipe": STUDY_RECIPE,
        "records": str(sources.dense_records.resolve()),
        "results": str(sources.dense_results.resolve()),
        "fingerprint_manifests_sha256": {
            name: _sha256(sources.dense_results / name)
            for name in (
                "artifact_hashes.json",
                "summary.json",
                "attn_parent_validation.json",
                "hc_validation.json",
                "hc_calibration.json",
            )
        },
    }
    return record


def convert(sources: Sources, output: Path) -> Path:
    started = time.perf_counter()
    config = validate_config(sources.model)
    frontend = _frontend_bytes(sources.model)
    study = sources.dense_study()
    with (
        StudyExpertStore(sources.experts, sources.experts_manifest) as store,
        ShardReader(sources.model) as reader,
    ):
        plans = build_inventory(store.nvfp4_experts, store.f3_experts, frontend)
        preflight_sources(plans, reader, study)
        # Create-only: claim the path before the writer truncates it.
        os.close(os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644))
        _log(f"preflight complete: {len(plans)} objects")
        scale_table = np.zeros((LAYERS, EXPERTS, 3), dtype=np.uint32)
        filled = np.zeros((LAYERS, EXPERTS), dtype=bool)
        identity = ArtifactIdentity(MODEL_ID, WEIGHTS_ID)
        with ArtifactWriter(output, identity, [plan.spec for plan in plans]) as writer:
            tables: dict[str, bytes] = {}
            for index, plan in enumerate(plans, start=1):
                name = plan.spec.name
                if plan.transform is Transform.RESOURCE:
                    payload: object = (sources.model / plan.sources[0]).read_bytes()
                elif name == E2M1_RECORDS:
                    payload = _expert_records(store, ExpertCodec.NVFP4, E2M1, scale_table, filled)
                elif name == F3_RECORDS:
                    payload = _expert_records(store, ExpertCodec.F3, F3, scale_table, filled)
                elif plan.transform is Transform.ROUTED_EXPERTS:
                    tables = tables or _routed_tables(store, scale_table, filled)
                    payload = tables[name]
                elif plan.transform in STUDY_TRANSFORMS:
                    payload = encode_study_payload(plan, study)
                else:
                    assert isinstance(plan.spec, TensorSpec)
                    payload = encode_payload(plan.spec, materialize(plan, reader))
                writer.write(name, payload)
                del payload
                _log(f"[{index}/{len(plans)}] {name}")
        objects = writer.objects
    write_seconds = time.perf_counter() - started
    _log(f"wrote {output} ({output.stat().st_size} bytes) in {write_seconds:.0f}s; validating")
    with StudyExpertStore(sources.experts, sources.experts_manifest) as store:
        validation = validate_artifact(output, sources.model, store, study, frontend, _log)
        source_record = _source_record(sources, store)
    report = {
        "identity": {"model_id": MODEL_ID, "weights_id": WEIGHTS_ID},
        "target_key": "qwen3_8_flash_next",
        "recipe_id": RECIPE_ID,
        "encoders": {
            "dense": f"exact ingestion of the dense-format study recipe {STUDY_RECIPE} "
            "with ATTN_PARENT.md attention-input parents and HC.md hyper-connection projections",
            "routed_experts": "exact repack of the study store words",
        },
        "source": source_record,
        "source_config": config,
        "converter": {
            "revision": family_conversion.converter_revision(Path(__file__).resolve().parents[3]),
            "environment": family_conversion.environment(torch.device("cpu")),
            "torch_threads": torch.get_num_threads(),
        },
        "inventory": _inventory_summary(plans, objects),
        "write_seconds": round(write_seconds, 1),
        "validation": validation.to_json(),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "peak_rss_bytes": _peak_rss_bytes(),
        "artifact": {"path": str(output.resolve()), "bytes": output.stat().st_size},
    }
    report_path = Path(str(output) + ".conversion.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    _log(f"validated; receipt {report_path}")
    return report_path


def validate_only(sources: Sources, artifact: Path) -> Path:
    validate_config(sources.model)
    with StudyExpertStore(sources.experts, sources.experts_manifest) as store:
        validation = validate_artifact(
            artifact,
            sources.model,
            store,
            sources.dense_study(),
            _frontend_bytes(sources.model),
            _log,
        )
    report = {
        "artifact": str(artifact.resolve()),
        "validation": validation.to_json(),
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    report_path = Path(str(artifact) + ".validation.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    _log(f"validated; report {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, required=True, help="BF16 checkpoint directory")
    parser.add_argument("--experts", type=Path, required=True, help="study experts.bin")
    parser.add_argument("--experts-manifest", type=Path, required=True, help="study manifest.json")
    parser.add_argument("--dense-records", type=Path, required=True, help="dense study quant/dense")
    parser.add_argument(
        "--dense-results", type=Path, required=True, help="dense study results/dense"
    )
    parser.add_argument("--out", type=Path, required=True, help="artifact path (create-only)")
    parser.add_argument(
        "--validate-only", action="store_true", help="validate an existing --out artifact"
    )
    args = parser.parse_args()
    torch.set_num_threads(_THREADS)
    sources = Sources(
        args.model, args.experts, args.experts_manifest, args.dense_records, args.dense_results
    )
    if args.validate_only:
        validate_only(sources, args.out)
    else:
        convert(sources, args.out)


if __name__ == "__main__":
    main()
