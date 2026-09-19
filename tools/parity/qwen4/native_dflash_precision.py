"""Bounded mathematical weight-precision ablation on captured diagnostic target features.

This is an offline reference experiment, not a new artifact profile, calibration,
kernel qualification, or an inference path. Role-group ablations distinguish a
small context-entry precision exception from broader weight-quantization loss.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tools.artifact.container import Artifact
from tools.parity.qwen4.native_dflash_goldens import read_weights
from tools.reference.qwen4.dflash import block_inputs, forward


def run(draft: Path, native: Path, captured: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    provenance = json.loads((captured / "qwen4-dflash-target-features.json").read_text())
    if (provenance["profile"] != "qwen4-ud-iq1-s-diagnostic-accepted-prompt"
            or provenance["accepted_prompt_tokens"] != 24
            or provenance["taps"] != [4, 16, 24, 36, 44]):
        raise ValueError("unexpected captured feature authority")
    features = torch.frombuffer(bytearray(
        (captured / "qwen4-dflash-target-features.bin").read_bytes()),
        dtype=torch.bfloat16).reshape(24, 12800)
    with Artifact(native / "qwen4-text-panel.ninfer") as artifact:
        tokens = torch.frombuffer(bytearray(artifact.payload("token.ids")), dtype=torch.int32)
        if tokens.tolist() != provenance["token_ids"]:
            raise ValueError("feature/embedding token alignment differs")
        anchors = torch.frombuffer(bytearray(artifact.payload("token.embeddings")),
                                   dtype=torch.bfloat16).reshape(33, 2560).clone()
    with Artifact(draft / "qwen4-dflash-inputs.ninfer") as artifact:
        mask = torch.frombuffer(bytearray(artifact.payload("mask.embedding")),
                                dtype=torch.bfloat16).clone()
    source = read_weights(draft / "qwen4-dflash-bf16.ninfer")
    quantized = read_weights(draft / "qwen4-dflash-nvfp4.ninfer")
    matrices = {name for name in source if source[name].ndim == 2}
    protected = {
        "bf16_reference": matrices,
        "nvfp4": set(),
        "nvfp4_protected_feature_fusion": {"fc.weight"},
        "nvfp4_protected_context_entry": {"fc.weight"} | {
            name for name in matrices if ".self_attn.k_proj." in name or ".self_attn.v_proj." in name},
        "nvfp4_protected_attention_and_fusion": {"fc.weight"} | {
            name for name in matrices if ".self_attn." in name},
        "nvfp4_protected_mlp": {name for name in matrices if ".mlp." in name},
    }
    results, profiles = {}, {}
    with Artifact(draft / "qwen4-dflash-bf16.ninfer") as bf, Artifact(draft / "qwen4-dflash-nvfp4.ninfer") as nv:
        for name, names in protected.items():
            profiles[name] = {
                "bf16_matrix_names": sorted(names),
                "bf16_matrix_parameters": sum(source[key].numel() for key in names),
                "additional_payload_bytes": sum(len(bf.payload(key)) - len(nv.payload(key)) for key in names),
            }
    for count in (8, 16, 24):
        embeddings, positions = block_inputs(anchors[count], mask, count, 7)
        reference = None
        results[str(count)] = {}
        for name, names in protected.items():
            weights = {key: source[key] if key in names else value for key, value in quantized.items()}
            result = forward(features[:count], embeddings.bfloat16(), torch.arange(count), positions,
                             weights, materialize_public_bf16=True)
            if reference is None:
                reference = result
            def relative(actual, ideal):
                return float(torch.linalg.vector_norm(actual - ideal) / torch.linalg.vector_norm(ideal))
            results[str(count)][name] = {
                "hidden_relative_l2": relative(result.hidden, reference.hidden),
                "context_relative_l2": relative(result.context, reference.context),
                "layer_hidden_relative_l2": [relative(a["hidden"], b["hidden"])
                                             for a, b in zip(result.layers, reference.layers)],
            }
            print(count, name, results[str(count)][name], flush=True)
    with output.open("x") as stream:
        json.dump({"feature_provenance": provenance, "contexts": [8, 16, 24], "queries": 7,
                   "profiles": profiles, "results": results,
                   "oracle": "Independent FP64 complete formula with explicit public BF16 Op outputs; exact represented weight decode",
                   "boundary": "Three overlapping prefixes of one diagnostic-target prompt, not independent holdout data; causal precision ablation only. No new storage profile, calibration, GPU qualification, PPL, acceptance or default admission."}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--captured", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    run(args.draft, args.native, args.captured, args.output)
