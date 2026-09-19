"""Bounded mathematical weight-precision ablation on captured diagnostic target features.

This is an offline reference experiment, not a new artifact profile, calibration,
kernel qualification, or an inference path. Protecting feature fusion is tested
because this single projection conditions every draft layer's accepted context.
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
        anchor = torch.frombuffer(bytearray(artifact.payload("token.embeddings")),
                                  dtype=torch.bfloat16).reshape(33, 2560)[24].clone()
    with Artifact(draft / "qwen4-dflash-inputs.ninfer") as artifact:
        mask = torch.frombuffer(bytearray(artifact.payload("mask.embedding")),
                                dtype=torch.bfloat16).clone()
    embeddings, positions = block_inputs(anchor, mask, 24, 7)
    source = read_weights(draft / "qwen4-dflash-bf16.ninfer")
    quantized = read_weights(draft / "qwen4-dflash-nvfp4.ninfer")
    profiles = {"bf16_reference": source, "nvfp4": quantized,
                "nvfp4_protected_feature_fusion": dict(quantized, **{"fc.weight": source["fc.weight"]})}
    results = {}
    reference = None
    for name, weights in profiles.items():
        result = forward(features, embeddings.bfloat16(), torch.arange(24), positions,
                         weights, materialize_public_bf16=True)
        if reference is None:
            reference = result.hidden
        error = torch.linalg.vector_norm(result.hidden - reference)
        norm = torch.linalg.vector_norm(reference)
        results[name] = {"hidden_relative_l2": float(error / norm)}
        print(name, results[name], flush=True)
    with output.open("x") as stream:
        json.dump({"feature_provenance": provenance, "context": 24, "queries": 7,
                   "results": results,
                   "oracle": "Independent FP64 complete formula with explicit public BF16 Op outputs; exact represented weight decode",
                   "boundary": "One diagnostic-target context; causal precision ablation only. No new storage profile, calibration, GPU qualification, PPL, acceptance or default admission."}, stream, indent=2)
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
