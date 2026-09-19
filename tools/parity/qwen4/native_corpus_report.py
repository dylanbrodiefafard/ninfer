"""Summarize native resident capture error without changing admission criteria."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from tools.artifact.container import Artifact
from tools.parity.qwen4.native_endpoint_corpus import metrics
from tools.reference.qwen4.common import linear


def read(path, columns):
    return torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.float32).reshape(-1, columns)


def report(manifest_path, captures, source):
    manifest = json.loads(manifest_path.read_text())
    results = {}
    routers = []
    for layer in range(4):
        with safe_open(str(source/f"qwen4-layer-{layer}.safetensors"), framework="pt", device="cpu") as weights:
            routers.append(weights.get_tensor(f"model.language_model.layers.{layer}.mlp.gate.weight").double())
    for panel in manifest["panels"]:
        directory = captures/panel["id"]
        metadata = json.loads((directory/"capture.json").read_text())
        with Artifact(panel["panel"]) as artifact:
            tokens = torch.frombuffer(bytearray(artifact.payload("token.ids")), dtype=torch.int32).tolist()
        if metadata["tokens"] != tokens or len(tokens) != panel["tokens"]:
            raise ValueError("capture/token identity differs")
        result = {"tokens": len(tokens), "split": panel["split"],
                  "prefix_failures": metadata["prefix_failures"], "layers": {}}
        for layer in range(4):
            prefix = f"layer{layer}."
            values = {}
            for name, columns in (("residual_input",10240), ("attn_input",2560),
                                  ("mixer_output",2560), ("moe_input",2560),
                                  ("moe_output",2560), ("residual_output",10240)):
                actual = read(directory/(prefix+name+".actual.f32"),columns)
                reference = read(directory/(prefix+name+".reference.f32"),columns)
                if len(actual) != len(tokens) or actual.shape != reference.shape:
                    raise ValueError("capture shape differs")
                values[name] = metrics(actual,reference)
            if layer < 3:
                # State rows are channels/heads, not time. Label that distinction.
                for name, columns in (("conv_state",3), ("recurrent_state",128*128)):
                    actual = read(directory/(prefix+name+".actual.f32"),columns)
                    reference = read(directory/(prefix+name+".reference.f32"),columns)
                    if name == "conv_state":
                        actual = actual.reshape(3,10240).T
                        reference = reference.reshape(3,10240).T
                    state = metrics(actual,reference)
                    state["max_row_relative_l2"] = state.pop("max_token_relative_l2")
                    state["worst_row"] = state.pop("worst_token")
                    values[name] = state
            actual_ids = torch.frombuffer(bytearray((directory/(prefix+"expert_ids.i32")).read_bytes()),
                                          dtype=torch.int32).reshape(len(tokens),10).long()
            actual_input = read(directory/(prefix+"moe_input.actual.f32"),2560)
            reference_input = read(directory/(prefix+"moe_input.reference.f32"),2560)
            actual_logits = linear(actual_input,routers[layer])
            reference_logits = linear(reference_input,routers[layer])
            local_ids = torch.argsort(actual_logits,dim=1,descending=True,stable=True)[:,:10]
            reference_ids = torch.argsort(reference_logits,dim=1,descending=True,stable=True)[:,:10]
            cutoff = torch.sort(actual_logits,dim=1,descending=True).values
            changed = (actual_ids != reference_ids).any(1)
            set_changed = (actual_ids.sort(1).values != reference_ids.sort(1).values).any(1)
            values["routing"] = {
                "same_input_exact_order_mismatches": int((actual_ids != local_ids).sum()),
                "propagated_order_changed_tokens": int(changed.sum()),
                "propagated_set_changed_tokens": int(set_changed.sum()),
                "changed_token_positions": changed.nonzero().flatten().tolist(),
                "changed_actual_input_cutoff_margins": (cutoff[:,9]-cutoff[:,10])[changed].tolist(),
            }
            result["layers"][str(layer)] = values
        # Below the 2048-token QSA selection budget all causal tokens are selected;
        # these panels cannot establish sparse top-k cutoff stability at long context.
        result["qsa_selection_scope"] = "All causal keys fit within 2048; no sparse cutoff claim."
        results[panel["id"]] = result
        print(panel["id"], "prefix_failures", metadata["prefix_failures"], flush=True)
    return {"manifest":str(manifest_path), "panels":results,
            "scope":"Independent accumulated component references and exact same-input routing. Metrics are evidence, not replacement thresholds or whole-model PPL."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","captures","source","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    args.out.write_text(json.dumps(report(args.manifest,args.captures,args.source),indent=2)+"\n")
