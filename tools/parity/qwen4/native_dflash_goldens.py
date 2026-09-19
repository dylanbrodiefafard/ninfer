"""Generate tool-only FP64 DFlash oracle planes from represented public inputs.

The binary is deliberately NOT a product artifact: a JSON plane directory gives
byte offsets, shapes, and F64/I64/BF16 dtypes. The companion weights remain .ninfer.
Context features are synthetic BF16 values, not a claimed full-target HC trace.
Anchor/mask embeddings and all five layers are authentic checkpoint weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tools.artifact.container import Artifact
from tools.reference.qwen4.dflash import block_inputs, decode_nvfp4_weight, forward


def read_weights(path: Path) -> dict[str, torch.Tensor]:
    weights = {}
    with Artifact(path) as artifact:
        for item in artifact.objects:
            payload = bytes(artifact.payload(item.name))
            if item.format == "BF16":
                weights[item.name] = torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).reshape(item.shape)
            elif item.format == "NVFP4":
                weights[item.name] = decode_nvfp4_weight(payload, item.shape)
            else:
                raise ValueError(f"unexpected DFlash weight format: {item.format}")
    return weights


def generate(weight_path: Path, input_path: Path, output: Path) -> None:
    for path in (output, output.with_suffix(".json")):
        if path.exists():
            raise FileExistsError(path)
    with Artifact(input_path) as artifact:
        row_shape = artifact.find("anchor.embeddings").shape
        anchors = torch.frombuffer(bytearray(artifact.payload("anchor.embeddings")), dtype=torch.bfloat16).reshape(row_shape)
        mask = torch.frombuffer(bytearray(artifact.payload("mask.embedding")), dtype=torch.bfloat16).clone()
        anchor_ids = torch.frombuffer(bytearray(artifact.payload("anchor.ids")), dtype=torch.int32).tolist()
    weights = read_weights(weight_path)
    records = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        def put(name: str, value: torch.Tensor) -> None:
            value = value.contiguous().cpu()
            dtype = {torch.bfloat16: "BF16", torch.float64: "F64", torch.int64: "I64"}[value.dtype]
            payload = value.view(torch.uint8).numpy().tobytes()
            records.append(dict(name=name, dtype=dtype, shape=list(value.shape),
                                offset=stream.tell(), bytes=len(payload)))
            stream.write(payload)

        cases = []
        for case_index, (context_count, query_count) in enumerate(((0, 1), (3, 4), (65, 7))):
            prefix = f"c{context_count}_k{query_count}"
            coordinate = torch.arange(context_count * 12800, dtype=torch.float64).reshape(context_count, 12800)
            features = (.5 * torch.sin(coordinate * .013 + .7)
                        + .25 * torch.cos(coordinate * .007 - .3)).bfloat16()
            # Include nonzero absolute positions; all accepted context precedes anchor.
            positions = torch.arange(17, 17 + context_count, dtype=torch.int64)
            anchor = 17 + context_count
            noise, queries = block_inputs(anchors[case_index], mask, anchor, query_count)
            noise = noise.bfloat16()
            result = forward(features, noise, positions, queries, weights)
            put(prefix + "/input/features", features)
            put(prefix + "/input/noise_embeddings", noise)
            put(prefix + "/input/context_positions", positions)
            put(prefix + "/input/query_positions", queries)
            materialized = forward(features, noise, positions, queries, weights, materialize_public_bf16=True)
            for suffix, trace_result in (("", result), ("/materialized", materialized)):
                put(prefix + suffix + "/context", trace_result.context)
                put(prefix + suffix + "/hidden", trace_result.hidden)
                for layer, (trace, cache) in enumerate(zip(trace_result.layers, trace_result.cache)):
                    for name, value in trace.items():
                        put(f"{prefix}{suffix}/layers/{layer}/{name}", value)
                    put(f"{prefix}{suffix}/cache/{layer}/keys", cache.keys)
                    put(f"{prefix}{suffix}/cache/{layer}/values", cache.values)
                    put(f"{prefix}{suffix}/cache/{layer}/positions", cache.positions)
            cases.append(dict(name=prefix, context=context_count, queries=query_count,
                              anchor=anchor, anchor_token_id=anchor_ids[case_index],
                              final_l2=float(torch.linalg.vector_norm(result.hidden)),
                              final_max_abs=float(result.hidden.abs().max())))
            print(f"DFlash FP64 golden {prefix}: final max={cases[-1]['final_max_abs']:.8g}", flush=True)
    with output.with_suffix(".json").open("x") as stream:
        json.dump(dict(weights=str(weight_path), inputs=str(input_path),
                       oracle="complete ideal FP64 formula from represented BF16 inputs and exact represented weights",
                       intermediate_casts="none in default planes; /materialized/ is supplementary explicit public-Op BF16 casts only",
                       feature_provenance="deterministic synthetic BF16, not captured full-target HC features",
                       encoding="little-endian concatenated planes; offsets measured from binary start",
                       cases=cases, planes=records), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    generate(args.weights, args.inputs, args.out)
