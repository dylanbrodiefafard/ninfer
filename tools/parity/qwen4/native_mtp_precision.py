"""Localize propagated MTP seed sensitivity with represented-boundary interventions.

This supplementary experiment does not change any Op oracle or acceptance gate.
Replacing a prefix by its recorded GPU public state attributes downstream drift;
it is not an alternative correctness oracle or a proposed arithmetic profile.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from tools.reference.qwen4.mtp import D, Weights, QsaOracle, bf16, gr, inject, moe, represented, stem
from tools.reference.qwen4.common import linear, source_grouped_rmsnorm


def relative(actual, reference):
    return float(torch.linalg.vector_norm(actual - reference) / torch.linalg.vector_norm(reference))


def run(source: Path, trace: Path, output: Path, study: str) -> None:
    if output.exists():
        raise FileExistsError(output)
    report = json.loads(trace.read_text())
    if report["profile"] != "qwen4-mtp-vllm-tokenspeed-frozen-domain-w4a16":
        raise ValueError("unexpected MTP trace policy")
    record = report["records"][0]
    if record["draft"] or record["frontier"] != 0:
        raise ValueError("experiment needs the first authentic seed from an empty private cache")
    embedding = represented(record["embedding"]).reshape(-1, D)
    count = len(embedding)
    hidden = represented(record["target_hidden"]).reshape(count, 4, D)
    positions = torch.tensor(record["positions"], dtype=torch.long).reshape(count, 3).T
    actual = {name: represented(value) for name, value in record["stages"].items()}
    for name in ("stem", "attention_injected", "moe_injected"):
        actual[name] = actual[name].reshape(count, 4, D)
    for name in ("attention_read", "attention", "moe_read", "moe", "final_read"):
        actual[name] = actual[name].reshape(count, D)
    for name in ("attention_write", "moe_write"):
        actual[name] = actual[name].reshape(count, 4)
    cases = (("independent_chain", "after_stem", "after_attention_read", "after_attention",
              "after_attention_inject", "after_moe_read", "after_moe") if study == "boundary" else
             ("independent_chain", "bf16_norm_bf16_projection", "bf16_norm_fp32_projection",
              "fp32_norm_bf16_projection"))
    results = {}
    ideal_carry = None
    with safe_open(str(source), framework="pt", device="cpu") as file:
        weights = Weights(file)
        for case in cases:
            boundary = cases.index(case) if study == "boundary" else 0
            residual = bf16(stem(weights, embedding, hidden)) if boundary == 0 else actual["stem"]
            if study == "stem" and case != "independent_chain":
                # Deliberately separate implementation-profile simulation: these
                # private casts are NOT copied into the independent stem oracle.
                en = source_grouped_rmsnorm(embedding, weights.get("pre_fc_norm_embedding.weight"), group_size=D)
                hn = source_grouped_rmsnorm(hidden.flatten(-2), weights.get("pre_fc_norm_hidden.weight"), group_size=4*D)
                # New hypothesis changes only the private normalization store;
                # retain original BF16 projection/add boundaries and complete suffix.
                if case == "fp32_norm_bf16_projection":
                    en, hn = en.float().double(), hn.float().double()
                else:
                    en, hn = bf16(en), bf16(hn)
                ep = linear(en, weights.get("fc_embedding.weight"))[:, None]
                hp = linear(hn.reshape(-1, 4, D), weights.get("fc_hidden.weight"))
                if case in ("bf16_norm_bf16_projection", "fp32_norm_bf16_projection"):
                    residual = bf16(bf16(ep) + bf16(hp))
                else:
                    residual = bf16((ep.float() + hp.float()).double())
            stem_error = relative(residual, bf16(stem(weights, embedding, hidden)))
            read = gr(weights, residual, "attn")
            mixed = bf16(read.mixed) if boundary < 2 else actual["attention_read"]
            write = bf16(read.injection_scales) if boundary < 2 else actual["attention_write"]
            attention = bf16(QsaOracle()(weights, mixed, positions, False)) if boundary < 3 else actual["attention"]
            residual = bf16(inject(residual, attention, write)) if boundary < 4 else actual["attention_injected"]
            read = gr(weights, residual, "mlp")
            mixed = bf16(read.mixed) if boundary < 5 else actual["moe_read"]
            write = bf16(read.injection_scales) if boundary < 5 else actual["moe_write"]
            experts = moe(weights, mixed)
            value = bf16(experts.output) if boundary < 6 else actual["moe"]
            residual = bf16(inject(residual, value, write))
            if ideal_carry is None:
                ideal_carry = residual.clone()
            final = gr(weights, residual, "final").mixed
            errors = [relative(actual["moe_injected"][i], residual[i]) for i in range(count)]
            changes = [len(set(actual["moe_ids"].reshape(count, 10)[i].tolist()) -
                           set(experts.expert_ids[i].tolist())) for i in range(count)]
            results[case] = {"carry_relative_l2_per_token": errors,
                             "carry_relative_l2_max": max(errors),
                             "worst_token": errors.index(max(errors)),
                             "final_read_relative_l2": relative(actual["final_read"], final),
                             "router_membership_changes_per_token": changes,
                             "stem_relative_l2_vs_ideal_store": stem_error,
                             "carry_relative_l2_vs_ideal_chain_per_token": [relative(residual[i], ideal_carry[i]) for i in range(count)]}
            print(case, "max carry", max(errors), "token", errors.index(max(errors)),
                  "final", results[case]["final_read_relative_l2"], "route changes", sum(changes),
                  "max vs ideal", max(results[case]["carry_relative_l2_vs_ideal_chain_per_token"]), flush=True)
    with output.open("x") as stream:
        json.dump({"source": str(source), "trace": str(trace), "seed_tokens": count,
                   "oracle": "Unchanged independent FP64 MTP formula, explicit public BF16 stores",
                   "study": study,
                   "intervention": "boundary: replace every public prefix boundary through named stage by GPU words; stem: simulate private BF16/FP32 norm and BF16/FP32 projection stores using ideal FP64 dot products; propagate remaining unchanged formulas",
                   "results": results,
                   "boundary": "Attribution only; replacing actual states is not an Op oracle, numerical admission, weight calibration, acceptance or PPL evidence"}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study", choices=("boundary", "stem"), default="boundary")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.source, args.trace, args.output, args.study)
