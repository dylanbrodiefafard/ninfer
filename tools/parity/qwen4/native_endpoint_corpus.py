"""Held-out endpoint/PLE storage loss from authentic represented component inputs.

Existing CUDA format/Op oracles remain the implementation gates. This tool evaluates
source-weight loss using independent FP64 formulas; it does not qualify a model recipe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tools.artifact.container import Artifact
from tools.convert.common.fp8_quantize import encode_source_fp8
from tools.parity.qwen4.native_weight_candidates import decode_rows, error, lut, read_bf16
from tools.reference.qwen4.common import linear
from tools.reference.qwen4.gated_residual import source_read
from tools.reference.qwen4.ple import source_inject


def metrics(actual, reference):
    actual, reference = actual.double(), reference.double()
    result = error(actual, reference)
    delta = (actual-reference).reshape(len(actual), -1)
    denominator = torch.linalg.vector_norm(reference.reshape(len(reference), -1), dim=1)
    relative = torch.linalg.vector_norm(delta, dim=1)/denominator.clamp_min(1e-30)
    result.update(max_token_relative_l2=float(relative.max()), worst_token=int(relative.argmax()))
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise ValueError("nonfinite source precision result")
    return result


def fp8(weight):
    return decode_rows(encode_source_fp8(weight), *weight.shape)


def ple_panels(manifest, source, captures):
    prefix = "model.language_model.layers.1.ple."
    with Artifact(source / "qwen4-ple-component.ninfer") as artifact:
        key = read_bf16(artifact, prefix+"key_proj.weight")
        value = read_bf16(artifact, prefix+"value_proj.weight")
        norms = [read_bf16(artifact, prefix+name) for name in
                 ("norm_key.weight", "norm_query.weight", "norm_conv.weight")]
        conv = read_bf16(artifact, prefix+"conv1d.weight").reshape(10240, 4)
    qkey, qvalue = fp8(key), fp8(value)
    results = {}
    for panel in manifest["panels"]:
        if panel["split"] != "heldout":
            continue
        count = panel["tokens"]
        folder = captures / panel["id"]
        metadata = json.loads((folder / "capture.json").read_text())
        if len(metadata["tokens"]) != count:
            raise ValueError("wrong native capture width")
        hidden = torch.frombuffer(bytearray((folder / "layer1.residual_input.actual.f32").read_bytes()),
                                  dtype=torch.float32).reshape(count, 4, 2560)
        if not torch.equal(hidden, hidden.bfloat16().float()):
            raise ValueError("PLE input is not represented BF16")
        with Artifact(panel["panel"]) as artifact:
            tokens = torch.frombuffer(bytearray(artifact.payload("token.ids")), dtype=torch.int32)
            if metadata["tokens"] != tokens.tolist():
                raise ValueError("native PLE capture/token misalignment")
            embedding = read_bf16(artifact, "token.embeddings")
            packed = bytearray(artifact.payload("ple.rows"))
            rows = artifact.find("ple.rows").shape[0]
            codes = torch.frombuffer(packed, dtype=torch.uint8, count=rows*160).reshape(rows, 160)
            scale = torch.frombuffer(packed, dtype=torch.bfloat16, count=1, offset=rows*160).double()
            indices = torch.frombuffer(bytearray(artifact.payload("ple.local_rows")), dtype=torch.int32).long()
            gathered = (lut()[codes[indices].long()]*scale).bfloat16().reshape(count, 16, 160)
        state = torch.zeros((10240, 9), dtype=torch.bfloat16)
        def inject(k, v):
            return source_inject(hidden, gathered, k, v, *norms, conv, state)
        baseline = inject(key, value)
        result = {"tokens": count, "embedding_source_loss": metrics(fp8(embedding).bfloat16(), embedding),
                  "prefix_failures": metadata["prefix_failures"], "ple": {}}
        for name, k, v in (("key", qkey, value), ("value", key, qvalue), ("both", qkey, qvalue)):
            candidate = inject(k, v)
            result["ple"][name] = {field: metrics(getattr(candidate, field), getattr(baseline, field))
                                  for field in ("output", "injection", "gate")}
            result["ple"][name]["next_conv_state"] = error(candidate.next_conv_state, baseline.next_conv_state)
        results[panel["id"]] = result
        print(panel["id"], json.dumps(result), flush=True)
    return results


def endpoints(manifest, source, captures):
    provenance = json.loads((captures / "qwen4-dflash-corpus-features.json").read_text())
    if (provenance["profile"] != "qwen4-ud-iq1-s-diagnostic-disjoint-corpus"
            or provenance["corpus_repository"] != manifest["repository"]
            or provenance["corpus_revision"] != manifest["revision"]):
        raise ValueError("wrong full-depth diagnostic capture source")
    captured = {panel["id"]: panel for panel in provenance["panels"]}
    prefix = "model.language_model.hyper_connection_mixer."
    with Artifact(source / "qwen4-endpoint.ninfer") as artifact:
        norm = read_bf16(artifact, prefix+"hc_norm.weight")
        down = read_bf16(artifact, prefix+"input_mix_weight_down.weight")
        up = read_bf16(artifact, prefix+"input_mix_weight_up.weight")
        head = read_bf16(artifact, "lm_head.weight")
    qdown, qup = fp8(down), fp8(up)
    inputs, results = [], {}
    for panel in manifest["panels"]:
        if panel["split"] != "heldout":
            continue
        folder = captures / panel["id"]
        record = captured[panel["id"]]
        with Artifact(panel["panel"]) as artifact:
            tokens = torch.frombuffer(bytearray(artifact.payload("token.ids")), dtype=torch.int32).tolist()
        if (record["split"] != "heldout" or record["accepted_prompt_tokens"] != 128
                or record["token_ids"] != tokens):
            raise ValueError("endpoint capture and heldout token alignment differs")
        # A diagnostic target provides these full-depth public boundaries; this is
        # not a claim that its quantized weights equal the native NVIDIA target.
        hidden = torch.frombuffer(bytearray((folder / "target_residual.bf16").read_bytes()),
                                  dtype=torch.bfloat16).reshape(128, 4, 2560)
        x = torch.frombuffer(bytearray((folder / "target_head_input.bf16").read_bytes()),
                             dtype=torch.bfloat16).reshape(128, 2560)[[31, 63, 127]]
        inputs.append(x)
        reference = source_read(hidden, norm, down, up, None).mixed
        results[panel["id"]] = {"final_gr": {
            name: metrics(source_read(hidden, norm, d, u, None).mixed, reference)
            for name, d, u in (("down", qdown, up), ("up", down, qup), ("both", qdown, qup))}}
    inputs = torch.cat(inputs)
    original, quantized = [], []
    # Each row has its own scale, so row chunks give the identical complete head recipe.
    for start in range(0, 248320, 1024):
        weight = head[start:start+1024]
        original.append(linear(inputs, weight))
        quantized.append(linear(inputs, fp8(weight)))
    original, quantized = torch.cat(original, dim=1), torch.cat(quantized, dim=1)
    # Logical license/mask domain, not physical padded rows. Softmax diagnostics
    # remain separate from the engine's p-less/epsilon sampling law.
    original, quantized = original[:, :248077], quantized[:, :248077]
    p, q = torch.log_softmax(original, dim=1), torch.log_softmax(quantized, dim=1)
    for i, name in enumerate(results):
        begin = 3*i
        results[name]["head"] = metrics(quantized[begin:begin+3], original[begin:begin+3])
        results[name]["head"].update(positions=[31,63,127],
            argmax_agreement=(original[begin:begin+3].argmax(1)==quantized[begin:begin+3].argmax(1)).tolist(),
            softmax_kl=(p[begin:begin+3].exp()*(p[begin:begin+3]-q[begin:begin+3])).sum(1).tolist())
        print(name, json.dumps(results[name]), flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--stage", choices=("ple", "endpoint"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    manifest = json.loads(args.manifest.read_text())
    result = (ple_panels if args.stage=="ple" else endpoints)(manifest, args.source, args.captures)
    args.out.write_text(json.dumps({"stage": args.stage, "manifest": str(args.manifest),
        "results": result, "oracle": "Independent decoded E4M3 times stored BF16 row scale; complete FP64 component formulas from represented source inputs.",
        "scope": "Heldout source-storage loss, not CUDA oracle admission, model PPL or sampling equivalence. PLE uses native prefix inputs; endpoints use full-depth diagnostic-target inputs. Head softmax uses logical 248077 rows and is not p-less/epsilon sampling."}, indent=2)+"\n")
