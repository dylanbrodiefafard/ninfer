"""Audit the pinned native 48-layer text inventory using HTTP headers, never weight payloads.

The expected grammar is declared independently of any sampled layer. This is metadata
qualification, not a native full-model artifact, runtime binding, or arithmetic test.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import urllib.request

from tools.parity.qwen4.native_source import BASE, REPOSITORY, REVISION, read_header

ITEM_BYTES = {"BF16": 2, "F32": 4, "U8": 1, "F8_E4M3": 1, "I64": 8}


def layer_grammar(layer: int) -> dict:
    if not 0 <= layer < 48:
        raise ValueError("native source has exactly 48 layers")
    result = {}
    def add(role, shape, dtype="BF16"):
        result[f"model.language_model.layers.{layer}.{role}"] = (dtype, shape)
    for connection in ("attn_hyper_connection", "mlp_hyper_connection"):
        for role, shape in (("hc_norm", [10240]), ("block_inject_weight", [4, 10240]),
                            ("input_mix_weight_down", [320, 10240]),
                            ("input_mix_weight_up", [10240, 320])):
            add(f"{connection}.{role}.weight", shape)
    add("mlp.gate.weight", [512, 2560])
    add("mlp.shared_expert_gate.weight", [1, 2560])
    for role in ("gate", "up", "down"):
        add(f"mlp.shared_expert.{role}_proj.weight", [2560, 640] if role == "down" else [640, 2560])
    if layer % 4 == 3:
        shapes = {"indexer.index_qk_proj.weight": [640, 2560],
                  "indexer.k_layernorm.weight": [128], "indexer.q_layernorm.weight": [128],
                  "k_norm.weight": [256], "q_norm.weight": [256],
                  "q_proj.weight": [12288, 2560], "k_proj.weight": [512, 2560],
                  "v_proj.weight": [512, 2560], "o_proj.weight": [2560, 6144]}
        mixer = "self_attn"
    else:
        shapes = {"A_log": [48], "dt_bias": [48], "norm.weight": [128],
                  "conv1d.weight": [10240, 1, 4], "in_proj_a.weight": [48, 2560],
                  "in_proj_b.weight": [48, 2560], "in_proj_qkv.weight": [10240, 2560],
                  "in_proj_z.weight": [6144, 2560], "out_proj.weight": [2560, 6144]}
        mixer = "linear_attn"
    for role, shape in shapes.items():
        add(f"{mixer}.{role}", shape)
    for expert in range(512):
        for role in ("gate", "up", "down"):
            prefix = f"mlp.experts.{expert}.{role}_proj."
            add(prefix + "weight", [2560, 320] if role == "down" else [640, 1280], "U8")
            add(prefix + "weight_scale", [2560, 40] if role == "down" else [640, 160], "F8_E4M3")
            add(prefix + "weight_scale_2", [], "F32")
            add(prefix + "input_scale", [], "F32")
    return result


def validate_header(header: dict) -> None:
    spans = []
    for name, item in header.items():
        if name == "__metadata__":
            continue
        shape = item["shape"]
        if any(type(x) is not int or x < 0 for x in shape):
            raise ValueError(f"invalid shape: {name}")
        begin, end = item["data_offsets"]
        if type(begin) is not int or type(end) is not int or begin < 0 or end - begin != math.prod(shape) * ITEM_BYTES[item["dtype"]]:
            raise ValueError(f"invalid tensor span: {name}")
        spans.append((begin, end, name))
    stop = 0
    for begin, end, name in sorted(spans):
        if begin < stop:
            raise ValueError(f"overlapping tensor span: {name}")
        stop = end


def validate_layer(layer: int, actual: dict) -> dict:
    expected = layer_grammar(layer)
    if actual.keys() != expected.keys():
        raise ValueError(f"layer {layer} names differ: missing={sorted(expected.keys()-actual.keys())[:3]} extra={sorted(actual.keys()-expected.keys())[:3]}")
    sizes = Counter()
    for name, (dtype, shape) in expected.items():
        item = actual[name]
        if (item["dtype"], item["shape"]) != (dtype, shape):
            raise ValueError(f"native source dtype/shape differs: {name}")
        sizes[dtype] += math.prod(shape) * ITEM_BYTES[dtype]
    return {"layer": layer, "mixer": "QSA" if layer % 4 == 3 else "GDN",
            "tensors": len(actual), "payload_bytes": sum(sizes.values()), "bytes_by_dtype": dict(sizes)}


def audit(index: dict, headers: dict) -> dict:
    for header in headers.values():
        validate_header(header)
    weight_map = index["weight_map"]
    for shard, header in headers.items():
        for name in header:
            if name.startswith("model.language_model.layers.") and ".ple." not in name:
                if weight_map.get(name) != shard:
                    raise ValueError(f"source header/index mapping mismatch: {name}")
    layers = []
    selected = set()
    for layer in range(48):
        prefix = f"model.language_model.layers.{layer}."
        names = {name: shard for name, shard in weight_map.items()
                 if name.startswith(prefix) and not name.startswith(prefix + "ple.")}
        actual = {name: headers[shard][name] for name, shard in names.items()}
        report = validate_layer(layer, actual)
        report["shards"] = sorted(set(names.values()))
        layers.append(report)
        selected.update(names)
    # Reject unrecognized main-layer indices rather than silently excluding extra layers.
    for name in weight_map:
        if name.startswith("model.language_model.layers.") and ".ple." not in name and name not in selected:
            raise ValueError(f"unexpected text compute tensor: {name}")
    sizes = Counter()
    for layer in layers:
        sizes.update(layer["bytes_by_dtype"])
    compute = sum(sizes.values())
    return {"repository": REPOSITORY, "revision": REVISION, "layers": layers,
            "compute_tensors": sum(x["tensors"] for x in layers),
            "compute_payload_bytes": compute, "compute_bytes_by_dtype": dict(sizes),
            "vram_bytes": 32 * 1024**3, "compute_fits_32gib": compute <= 32 * 1024**3,
            "boundary": "All 48 source compute inventories validated independently; no tensor payload downloaded. Excludes token embedding, final GR, output head, PLE, vision, MTP, controls preparation overhead, and runtime state/workspace. Metadata cannot qualify scale values or numerical execution."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.out_dir / "model.safetensors.index.json"
    if not index_path.exists():
        with urllib.request.urlopen(BASE + index_path.name, timeout=120) as response:
            data = response.read(128 * 1024 * 1024 + 1)
        if len(data) > 128 * 1024 * 1024:
            raise ValueError("unexpected index size")
        index_path.write_bytes(data)
    index = json.loads(index_path.read_text())
    shards = sorted({shard for name, shard in index["weight_map"].items()
                     if name.startswith("model.language_model.layers.") and ".ple." not in name})
    headers, ranges = {}, []
    for shard in shards:
        cached = args.out_dir / (shard + ".header.json")
        if cached.exists():
            item = json.loads(cached.read_text())
            base, header = item["data_base"], item["header"]
        else:
            base, header = read_header(shard)
            cached.write_text(json.dumps({"data_base": base, "header": header}))
        print(f"metadata only: {shard}: {base} bytes", flush=True)
        headers[shard] = header
        ranges.append({"shard": shard, "inclusive_ranges": [[0, 7], [8, base - 1]]})
    report = audit(index, headers)
    report["header_requests"] = ranges
    report["header_bytes"] = sum(x["inclusive_ranges"][1][1] + 1 for x in ranges)
    (args.out_dir / "qwen4-native-inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("layers", "header_requests")}, indent=2))


if __name__ == "__main__":
    main()
