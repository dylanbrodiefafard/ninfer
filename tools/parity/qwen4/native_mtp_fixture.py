"""Acquire bounded private MTP weights and convert the exact NVFP4 W4A16 fixture.

Never downloads target layers or PLE. The candidate's input scales are published
placeholders: this fixture is explicitly A16-only, not an A4 calibration source.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import struct
import urllib.request

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter, TensorSpec
from tools.artifact.layouts import encode_direct, encode_nvfp4_experts
from tools.parity.qwen4.native_source import ITEM_BYTES, read_range

SOURCES = {
    "nvidia": ("nvidia/Qwen3.8-Flash-Next-NVFP4",
               "fc694b54fb0174e0913e6adf86691ef85a4ead47", "", 3101, 2698026496),
    "nvfp4": ("limpincat/flashnext-drafters",
              "39d7d235eb4748cd90d3ae575a2a2e54b49018c9", "mtp-nvfp4/", 6173, 1596726784),
}
PREFIX = "mtp.layers.0.mlp.experts."
PROJECTIONS = {"gate_proj": (640, 2560), "up_proj": (640, 2560),
               "down_proj": (2560, 640)}
CHUNK = 16 * 1024 * 1024


def inventory(kind: str) -> tuple[str, dict, list, dict]:
    repository, revision, folder, count, payload = SOURCES[kind]
    base = f"https://huggingface.co/{repository}/resolve/{revision}/{folder}"
    with urllib.request.urlopen(base + "model.safetensors.index.json", timeout=120) as response:
        names = {n: s for n, s in json.load(response)["weight_map"].items()
                 if n.startswith("mtp.")}
    if len(names) != count:
        raise ValueError("pinned MTP tensor count mismatch")
    headers = {}
    for shard in sorted(set(names.values())):
        size = struct.unpack("<Q", read_range(base + shard, 0, 7))[0]
        if not 2 <= size <= 32 * 1024 * 1024:
            raise ValueError("unexpected source header size")
        headers[shard] = (8 + size, json.loads(read_range(base + shard, 8, 7 + size)))
    ordered = sorted(names, key=lambda n: (names[n], headers[names[n]][1][n]["data_offsets"][0]))
    output_header, spans, total = {}, [], 0
    for name in ordered:
        shard = names[name]
        offset, header = headers[shard]
        item = header[name]
        begin, end = item["data_offsets"]
        size = math.prod(item["shape"]) * ITEM_BYTES[item["dtype"]]
        if begin < 0 or end - begin != size:
            raise ValueError(f"invalid source span: {name}")
        if name.startswith(PREFIX):
            rest = name[len(PREFIX):].split(".")
            if len(rest) != 3 or not 0 <= int(rest[0]) < 512:
                raise ValueError("unexpected MTP expert name")
            n, k = PROJECTIONS[rest[1]]
            expected = ({"weight": ("U8", [n, k // 2]),
                         "weight_scale": ("F8_E4M3", [n, k // 16]),
                         "weight_scale_2": ("F32", []), "input_scale": ("F32", [])}
                        if kind == "nvfp4" else
                        {"weight": ("F8_E4M3", [n, k]),
                         "weight_scale_inv": ("BF16", [n // 128, k // 128])})
            if expected.get(rest[2]) != (item["dtype"], item["shape"]):
                raise ValueError(f"unexpected expert dtype/shape: {name}")
        elif item["dtype"] != "BF16":
            raise ValueError(f"protected role is not source BF16: {name}")
        output_header[name] = dict(item, data_offsets=[total, total + size])
        total += size
        a, b = offset + begin, offset + end
        if spans and spans[-1][0] == shard and spans[-1][2] == a:
            spans[-1][2] = b
        else:
            if spans and spans[-1][0] == shard and spans[-1][2] > a:
                raise ValueError("overlapping source spans")
            spans.append([shard, a, b])
    if total != payload:
        raise ValueError("pinned MTP payload size mismatch")
    report = dict(repository=repository, revision=revision, folder=folder,
                  tensors=len(names), payload_bytes=total, source_file_intervals=spans,
                  dtype_counts=dict(Counter(v["dtype"] for v in output_header.values())))
    return base, output_header, spans, report


def acquire(root: Path, kind: str) -> Path:
    output = root / f"qwen4-mtp-{kind}-source.safetensors"
    report_path = output.with_suffix(".json")
    if output.exists():
        if not report_path.exists():
            raise ValueError("existing MTP source lacks provenance")
        report = json.loads(report_path.read_text())
        if (report.get("repository"), report.get("revision")) != SOURCES[kind][:2]:
            raise ValueError("existing MTP source provenance mismatch")
        return output
    base, header, spans, report = inventory(kind)
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < report["payload_bytes"] + 128 * 1024 * 1024:
        raise ValueError("insufficient disk for bounded MTP source")
    partial = output.with_suffix(".partial")
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with partial.open("xb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        done = 0
        for shard, begin, end in spans:
            for start in range(begin, end, CHUNK):
                stop = min(start + CHUNK, end)
                stream.write(read_range(base + shard, start, stop - 1))
                done += stop - start
                print(f"{kind}: {done}/{report['payload_bytes']} bytes", flush=True)
    partial.rename(output)
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return output


def build(root: Path) -> None:
    import torch
    from safetensors import safe_open

    source = acquire(root, "nvfp4")
    reference = acquire(root, "nvidia")
    output = root / "qwen4-mtp-nvfp4.ninfer"
    if output.exists():
        raise FileExistsError(output)
    with safe_open(str(source), framework="pt", device="cpu") as candidate, \
            safe_open(str(reference), framework="pt", device="cpu") as native:
        ordinary = sorted(n for n in candidate.keys() if not n.startswith(PREFIX))
        native_ordinary = sorted(n for n in native.keys() if not n.startswith(PREFIX))
        if len(ordinary) != 29 or ordinary != native_ordinary:
            raise ValueError("protected MTP role inventory mismatch")
        controls = {}
        for name in ordinary:
            a, b = candidate.get_tensor(name), native.get_tensor(name)
            if a.shape != b.shape or a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
                raise ValueError(f"protected MTP role shape/dtype mismatch: {name}")
            if not torch.equal(a.view(torch.uint16), b.view(torch.uint16)):
                raise ValueError(f"candidate protected role differs from NVIDIA source: {name}")
            controls[name] = list(a.shape)
        print("All 29 protected BF16 MTP tensors exactly match pinned NVIDIA words", flush=True)
        specs = [TensorSpec(n, tuple(shape), "BF16", "contiguous-le-v1")
                 for n, shape in controls.items()]
        for projection, (n, k) in PROJECTIONS.items():
            specs.append(TensorSpec(PREFIX + projection + ".weight", (512, n, k),
                                    "NVFP4_EXPERT_F32M", "expert-blockscale-k16-m128x4-v1"))
        report = dict(candidate_repository=SOURCES["nvfp4"][0], candidate_revision=SOURCES["nvfp4"][1],
                      reference_repository=SOURCES["nvidia"][0], reference_revision=SOURCES["nvidia"][1],
                      protected_bf16_exact=controls, activation_policy="A16-only",
                      input_scale_source="publisher-placeholder-not-calibration",
                      source_weight_loss=[])
        # Fixed independent decode table: signed E2M1 low nibble is even K.
        e2m1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                             -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=torch.float64)
        partial = output.with_suffix(".partial")
        with ArtifactWriter(partial, ArtifactIdentity("qwen4/native-mtp-qualification",
                            "limpincat-nvfp4-w4a16-source"), specs) as writer:
            for name in ordinary:
                writer.write(name, encode_direct(candidate.get_tensor(name), "BF16"))
            for spec in specs[len(ordinary):]:
                projection = spec.name.split(".")[-2]
                codes, scales, multipliers, inputs = [], [], [], []
                for expert in range(512):
                    stem = PREFIX + f"{expert}.{projection}."
                    code = candidate.get_tensor(stem + "weight")
                    scale = candidate.get_tensor(stem + "weight_scale").view(torch.uint8)
                    mult = candidate.get_tensor(stem + "weight_scale_2").reshape(())
                    inp = candidate.get_tensor(stem + "input_scale").reshape(())
                    if inp.item() != 1.0 or not math.isfinite(mult.item()) or mult.item() <= 0:
                        raise ValueError("candidate scale differs from audited placeholder profile")
                    codes.append(code); scales.append(scale); multipliers.append(mult); inputs.append(inp)
                    if expert in (0, 511):
                        n, k = PROJECTIONS[projection]
                        unpacked = torch.empty((n, k), dtype=torch.long)
                        unpacked[:, 0::2] = code & 15
                        unpacked[:, 1::2] = code >> 4
                        decoded = e2m1[unpacked] * scale.view(torch.float8_e4m3fn).double().repeat_interleave(16, 1) * mult.double()
                        ref = native.get_tensor(stem + "weight").double()
                        ref_scale = native.get_tensor(stem + "weight_scale_inv").double()
                        ref *= ref_scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
                        delta = decoded - ref
                        result = dict(expert=expert, projection=projection,
                                      relative_l2=(delta.norm() / ref.norm()).item(),
                                      max_abs=delta.abs().max().item(),
                                      criterion="reported source-weight loss, not an Op implementation gate")
                        report["source_weight_loss"].append(result)
                        print(result, flush=True)
                writer.write(spec.name, encode_nvfp4_experts(torch.stack(codes), torch.stack(scales),
                             torch.stack(multipliers), torch.stack(inputs), spec.shape))
                print(f"Converted {spec.name}", flush=True)
        partial.rename(output)
        with output.with_suffix(".json").open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")


def verify(root: Path) -> None:
    """Independent exact address oracle, not encoder/decoder roundtrip parity."""
    import torch
    from safetensors import safe_open

    with safe_open(str(root / "qwen4-mtp-nvfp4-source.safetensors"), framework="pt", device="cpu") as source, \
            Artifact(root / "qwen4-mtp-nvfp4.ninfer") as artifact:
        if artifact.identity != ArtifactIdentity("qwen4/native-mtp-qualification",
                                                "limpincat-nvfp4-w4a16-source"):
            raise ValueError("MTP fixture identity mismatch")
        for obj in artifact.objects:
            payload = artifact.payload(obj)
            if not obj.name.startswith(PREFIX):
                expected = source.get_tensor(obj.name).view(torch.uint8).numpy().tobytes()
                if payload != expected:
                    raise ValueError(f"BF16 artifact words changed: {obj.name}")
            else:
                projection = obj.name.split(".")[-2]
                n, k = PROJECTIONS[projection]
                code_bytes = n * k // 2
                scale_bytes = n * k // 16
                scale_start = 512 * code_bytes  # naturally 256-byte aligned
                mult_start = scale_start + 512 * scale_bytes
                # Registered m128x4 scalar address formula, independently evaluated.
                row = torch.arange(n, dtype=torch.long)[:, None]
                group = torch.arange(k // 16, dtype=torch.long)[None, :]
                address = ((row // 128) * (k // 64) + group // 4) * 512 + \
                          (row % 32) * 16 + ((row % 128) // 32) * 4 + group % 4
                for expert in range(512):
                    stem = PREFIX + f"{expert}.{projection}."
                    a = expert * code_bytes
                    expected = source.get_tensor(stem + "weight").numpy().tobytes()
                    if payload[a:a + code_bytes] != expected:
                        raise ValueError("MTP expert code plane changed")
                    a = scale_start + expert * scale_bytes
                    stored = torch.frombuffer(bytearray(payload[a:a + scale_bytes]), dtype=torch.uint8)
                    expected_scale = source.get_tensor(stem + "weight_scale").view(torch.uint8)
                    if not torch.equal(stored[address], expected_scale):
                        raise ValueError("MTP expert scale address oracle mismatch")
                    for suffix, base in (("weight_scale_2", mult_start),
                                         ("input_scale", mult_start + 512 * 4)):
                        a = base + expert * 4
                        expected = source.get_tensor(stem + suffix).reshape(1).view(torch.uint8).numpy().tobytes()
                        if payload[a:a + 4] != expected:
                            raise ValueError("MTP expert scalar word changed")
                print(f"Exact all-512 expert code/scale/multiplier oracle passed: {projection}", flush=True)
            payload.release()
    print("Complete private MTP artifact exact-word verification passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        for source in SOURCES:
            print(json.dumps(inventory(source)[3], indent=2))
    else:
        import torch
        torch.set_num_threads(2)
        if not args.verify_only:
            build(args.out_dir)
        verify(args.out_dir)
