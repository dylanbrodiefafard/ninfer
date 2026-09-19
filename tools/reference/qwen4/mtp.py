"""Native private-MTP represented-input oracle and emitted GPU trace qualification.

Reuses the independent GR/QSA/MoE formulas. Only public BF16 Op outputs, index-key
and core-KV state are rounded; no CUDA staging/reduction algorithm is reproduced.
This evaluates a bounded component, never a CPU inference backend.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from .common import ideal_softmax, linear, partial_rope, sigmoid, source_grouped_rmsnorm
from .gated_residual import inject, source_read
from .moe import sparse_moe
from .qsa import source_select, source_sparse_attention

D, F = 2560, 10240
# Predeclared existing Op envelopes, not fitted to these newly acquired weights.
STEM = (.01, .005, .02)
GR = (.006, .004, .005)
WRITE = (.0035, .0015, .003)
INJECT = (.003, .002, .002)
QSA = (.02, .005, .02)
MOE = (2.5 / 255, 1 / 32768, 2 / 255)
COMPOSED = (.02, .005, .02)


def bf16(x):
    return x.to(torch.bfloat16).double()


def represented(record):
    if record["dtype"] == "BF16":
        return torch.tensor(record["words"], dtype=torch.uint16).view(torch.bfloat16).double()
    return torch.tensor(record["values"], dtype=torch.float64 if record["dtype"] == "F32" else torch.long)


class Weights:
    def __init__(self, source):
        self.source = source

    @lru_cache(maxsize=32)
    def get(self, name):
        return self.source.get_tensor("mtp." + name).double()

    @lru_cache(maxsize=32)
    def expert(self, role, expert):
        p = f"mtp.layers.0.mlp.experts.{expert}.{role}."
        packed = self.source.get_tensor(p + "weight").long()
        code = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
        table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                              -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=torch.float64)
        scale = self.source.get_tensor(p + "weight_scale").view(torch.uint8).long()
        exponent, mantissa = (scale >> 3) & 15, scale & 7
        if torch.any((scale & 128) != 0) or torch.any(scale == 127):
            raise ValueError("invalid source block-scale code")
        exact_scale = torch.where(exponent == 0, mantissa.double() * 2.**-9,
                                   (1 + mantissa.double()/8) * torch.pow(2., exponent.double()-7))
        multiplier = self.source.get_tensor(p + "weight_scale_2").double()
        return table[code] * exact_scale.repeat_interleave(16, 1) * multiplier


class ExpertBank:
    def __init__(self, weights, role):
        self.weights, self.role = weights, role
        self.shape = (512,)

    def __getitem__(self, expert):
        return self.weights.expert(self.role, expert)


def stem(w, embedding, hidden):
    e = source_grouped_rmsnorm(embedding, w.get("pre_fc_norm_embedding.weight"), group_size=D)
    h = source_grouped_rmsnorm(hidden.flatten(-2), w.get("pre_fc_norm_hidden.weight"), group_size=F)
    return linear(h.reshape(-1, 4, D), w.get("fc_hidden.weight")) + linear(e, w.get("fc_embedding.weight"))[:, None]


def gr(w, hidden, role):
    p = "hyper_connection_mixer." if role == "final" else f"layers.0.{role}_hyper_connection."
    return source_read(hidden, w.get(p + "hc_norm.weight"), w.get(p + "input_mix_weight_down.weight"),
                       w.get(p + "input_mix_weight_up.weight"),
                       None if role == "final" else w.get(p + "block_inject_weight.weight"))


def moe(w, hidden):
    p = "layers.0.mlp."
    return sparse_moe(hidden, w.get(p + "gate.weight"), ExpertBank(w, "gate_proj"),
        ExpertBank(w, "up_proj"), ExpertBank(w, "down_proj"),
        w.get(p + "shared_expert.gate_proj.weight"), w.get(p + "shared_expert.up_proj.weight"),
        w.get(p + "shared_expert.down_proj.weight"), w.get(p + "shared_expert_gate.weight"))


class QsaOracle:
    """Independently evolved cache for the propagated-chain diagnostic only."""
    def __init__(self):
        self.keys, self.values, self.index_keys, self.positions, self.seed = None, None, None, None, None

    def __call__(self, w, hidden, positions, draft):
        p = "layers.0.self_attn."
        qg = linear(hidden, w.get(p + "q_proj.weight")).reshape(-1, 24, 512)
        keys = linear(hidden, w.get(p + "k_proj.weight")).reshape(-1, 2, 256)
        values = linear(hidden, w.get(p + "v_proj.weight")).reshape(-1, 2, 256)
        index = bf16(linear(hidden, w.get(p + "indexer.index_qk_proj.weight")))
        old = 0 if self.keys is None else self.keys.shape[0]
        self.keys = keys if old == 0 else torch.cat((self.keys, keys))
        self.values = values if old == 0 else torch.cat((self.values, values))
        self.index_keys = index[:, 512:] if old == 0 else torch.cat((self.index_keys, index[:, 512:]))
        self.positions = positions if old == 0 else torch.cat((self.positions, positions), dim=1)
        if draft:
            selection = [self.seed]
        else:
            selection = source_select(index[:, :512].reshape(-1, 4, 128), self.index_keys.to(torch.bfloat16),
                [torch.arange(old+i+1) for i in range(hidden.shape[0])], positions, self.positions,
                w.get(p + "indexer.q_layernorm.weight"), w.get(p + "indexer.k_layernorm.weight"),
                mrope_section=(11, 11, 10))
            selection = [s.token_ids for s in selection]
            self.seed = selection[-1]
        result = source_sparse_attention(qg[:, :, :256], self.keys, self.values, selection,
            positions, self.positions, w.get(p + "q_norm.weight"), w.get(p + "k_norm.weight"),
            qg[:, :, 256:].flatten(-2), w.get(p + "o_proj.weight"),
            core_cache_dtype=torch.bfloat16, mrope_section=(11, 11, 10))
        return result.output

@dataclass
class RecordedQsaState:
    keys: torch.Tensor
    values: torch.Tensor
    raw_keys: torch.Tensor
    positions: torch.Tensor
    selected_ids: torch.Tensor
    selected_count: int


def recorded_qsa_state(record, live):
    cache = record["cache"]
    capacity = cache["k"]["shape"][1]
    def core(name):
        return represented(cache[name]).reshape(2, capacity, 256)[:, :live].permute(1, 0, 2)
    return RecordedQsaState(core("k"), core("v"),
        represented(cache["raw_keys"]).reshape(capacity, 128)[:live],
        represented(cache["positions"]).reshape(capacity, 3)[:live].T,
        represented(record["selected_ids"]), int(record["selected_count"]["values"][0]))


def local_qsa(w, hidden, positions, draft, previous):
    """Complete QSA formula from actual previous GPU state and new public inputs.

    Old K rows are already normalized/rotated BF16 state: never reconstruct or
    transform them again. Only new projections cross norm/RoPE/cache boundaries.
    Returns independently expected state; callers advance using the next GPU record,
    not this expectation. The separate QsaOracle retains its own evolving cache.
    """
    p = "layers.0.self_attn."
    qg = linear(hidden, w.get(p + "q_proj.weight")).reshape(-1, 24, 512)
    keys = linear(hidden, w.get(p + "k_proj.weight")).reshape(-1, 2, 256)
    values = bf16(linear(hidden, w.get(p + "v_proj.weight"))).reshape(-1, 2, 256)
    index = bf16(linear(hidden, w.get(p + "indexer.index_qk_proj.weight")))
    keys = source_grouped_rmsnorm(keys, w.get(p + "k_norm.weight"), group_size=256)
    keys = bf16(partial_rope(keys, positions, rotary_dim=64, theta=1e7, mrope_section=(11, 11, 10)))
    raw_keys = index[:, 512:]
    old = 0 if previous is None else previous.keys.shape[0]
    cache_positions = positions
    if previous is not None:
        keys = torch.cat((previous.keys, keys))
        values = torch.cat((previous.values, values))
        raw_keys = torch.cat((previous.raw_keys, raw_keys))
        cache_positions = torch.cat((previous.positions, positions), dim=1)
    if draft:
        if previous is None:
            raise ValueError("draft QSA requires represented prior state")
        selected_ids, count = previous.selected_ids, previous.selected_count
        selections = [selected_ids[:count]]
    else:
        selections = source_select(index[:, :512].reshape(-1, 4, 128), raw_keys.to(torch.bfloat16),
            [torch.arange(old+i+1) for i in range(hidden.shape[0])], positions, cache_positions,
            w.get(p + "indexer.q_layernorm.weight"), w.get(p + "indexer.k_layernorm.weight"),
            mrope_section=(11, 11, 10))
        selections = [s.token_ids for s in selections]
        count = selections[-1].numel()
        selected_ids = torch.full((2051,), -1, dtype=torch.long)
        selected_ids[:count] = selections[-1]
    query = source_grouped_rmsnorm(qg[:, :, :256], w.get(p + "q_norm.weight"), group_size=256)
    query = partial_rope(query, positions, rotary_dim=64, theta=1e7, mrope_section=(11, 11, 10))
    core = torch.empty_like(query)
    for token, ids in enumerate(selections):
        for head in range(24):
            scores = keys[ids, head // 12] @ query[token, head] / math.sqrt(256)
            core[token, head] = (ideal_softmax(scores)[:, None] * values[ids, head // 12]).sum(0)
    output = linear(core.flatten(-2) * sigmoid(qg[:, :, 256:].flatten(-2)), w.get(p + "o_proj.weight"))
    return output, RecordedQsaState(keys, values, raw_keys, cache_positions, selected_ids, count)


def check_exact(label, actual, expected):
    passed = torch.equal(actual, expected)
    print(f"{label}: exact {'PASS' if passed else 'FAIL'}", flush=True)
    return int(not passed)


def check_qsa_state(label, actual, expected, previous):
    old = 0 if previous is None else previous.keys.shape[0]
    failures = 0
    for name in ("keys", "values", "raw_keys"):
        actual_rows, expected_rows = getattr(actual, name), getattr(expected, name)
        if previous is not None:
            failures += check_exact(f"{label} historical {name}", actual_rows[:old], getattr(previous, name))
        failures += check(f"{label} newly appended {name}", actual_rows[old:], expected_rows[old:], (.004, .004, .008))
    failures += check_exact(f"{label} positions", actual.positions, expected.positions)
    failures += check_exact(f"{label} selected IDs", actual.selected_ids, expected.selected_ids)
    failures += check_exact(f"{label} selected count", torch.tensor(actual.selected_count), torch.tensor(expected.selected_count))
    return failures


def check(label, actual, expected, criterion):
    actual, expected = actual.double(), expected.double()
    delta = actual - expected
    rel = (delta.norm() / expected.norm().clamp_min(1e-300)).item()
    maximum = delta.abs().max().item()
    limit = criterion[1] + criterion[2] * expected.abs().max().item()
    passed = bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()) and rel <= criterion[0] and maximum <= limit
    print(f"{label}: rel={rel:.8g} max={maximum:.8g} gross_limit={limit:.8g} {'PASS' if passed else 'FAIL'}", flush=True)
    return int(not passed)


def check_bf16_store(label, actual, ideal):
    """Keep the oracle ideal; require the actual public store to be correctly rounded.

    All three inject operands are represented BF16. The exact product plus sum fits
    FP64; the acceptance set is the single nearest-even BF16 encoding of that value.
    This is an output-format criterion, not an oracle staging cast.
    """
    check(label + " legacy envelope (diagnostic)", actual, ideal, INJECT)
    passed = bool(torch.isfinite(ideal).all()) and torch.equal(actual, bf16(ideal))
    print(f"{label}: exact nearest-even BF16 output {'PASS' if passed else 'FAIL'}", flush=True)
    return int(not passed)


def qualify(source: Path, trace: Path):
    report = json.loads(trace.read_text())
    if report["profile"] != "qwen4-mtp-vllm-tokenspeed-frozen-domain-w4a16":
        raise ValueError("unexpected MTP trace policy")
    failures = 0
    composition_screens = 0
    with safe_open(str(source), framework="pt", device="cpu") as file:
        w = Weights(file)
        previous_qsa, propagated_qsa = None, QsaOracle()
        reference_carry = None
        for step, record in enumerate(report["records"]):
            e = represented(record["embedding"]).reshape(-1, D)
            t = e.shape[0]
            pos = torch.tensor(record["positions"], dtype=torch.long).reshape(t, 3).T
            actual = {name: represented(value) for name, value in record["stages"].items()}
            for name in ("stem", "attention_injected", "moe_injected"):
                actual[name] = actual[name].reshape(t, 4, D)
            for name in ("attention_read", "attention", "moe_read", "moe", "final_read"):
                actual[name] = actual[name].reshape(t, D)
            for name in ("attention_write", "moe_write"):
                actual[name] = actual[name].reshape(t, 4)
            h = previous_actual[-1:] if record["draft"] else represented(record["target_hidden"]).reshape(t, 4, D)
            failures += check(f"step{step} stem local", actual["stem"], stem(w, e, h), STEM)
            attention_read = gr(w, actual["stem"], "attn")
            failures += check(f"step{step} attention GR local", actual["attention_read"], attention_read.mixed, GR)
            failures += check(f"step{step} attention write local", actual["attention_write"], attention_read.injection_scales, WRITE)
            if record["frontier"] != (0 if previous_qsa is None else previous_qsa.keys.shape[0]):
                raise ValueError("MTP trace does not continue the represented QSA frontier")
            attention, expected_qsa = local_qsa(w, actual["attention_read"], pos, record["draft"], previous_qsa)
            failures += check(f"step{step} QSA local", actual["attention"], attention, QSA)
            failures += check_bf16_store(f"step{step} attention inject local", actual["attention_injected"],
                inject(actual["stem"], actual["attention"], actual["attention_write"]))
            mlp = gr(w, actual["attention_injected"], "mlp")
            failures += check(f"step{step} MLP GR local", actual["moe_read"], mlp.mixed, GR)
            failures += check(f"step{step} MLP write local", actual["moe_write"], mlp.injection_scales, WRITE)
            experts = moe(w, actual["moe_read"])
            for token in range(t):
                failures += check(f"step{step} MoE token{token} local", actual["moe"][token], experts.output[token], MOE)
            if not torch.equal(actual["moe_ids"].reshape(t, 10), experts.expert_ids):
                print(f"step{step} exact router IDs FAIL", flush=True); failures += 1
            failures += check(f"step{step} route weights", actual["moe_weights"].reshape(t, 10), experts.route_weights,
                               (2e-6, 2e-6, 2e-6))
            failures += check_bf16_store(f"step{step} MLP inject local", actual["moe_injected"],
                inject(actual["attention_injected"], actual["moe"], actual["moe_write"]))
            failures += check(f"step{step} final GR local", actual["final_read"], gr(w, actual["moe_injected"], "final").mixed, GR)
            live = record["frontier"] + t
            actual_qsa = recorded_qsa_state(record, live)
            failures += check_qsa_state(f"step{step} QSA state", actual_qsa, expected_qsa, previous_qsa)
            previous_qsa = actual_qsa
            # Independently propagate complete formulas across every public BF16 boundary.
            ref_h = reference_carry if record["draft"] else h
            r = bf16(stem(w, e, ref_h))
            g = gr(w, r, "attn")
            a = bf16(propagated_qsa(w, bf16(g.mixed), pos, record["draft"]))
            r = bf16(inject(r, a, bf16(g.injection_scales)))
            g = gr(w, r, "mlp")
            propagated_moe = moe(w, bf16(g.mixed))
            for token in range(t):
                local_ids = set(experts.expert_ids[token].tolist())
                propagated_ids = set(propagated_moe.expert_ids[token].tolist())
                print(f"step{step} token{token} propagated router membership changes="
                      f"{len(local_ids - propagated_ids)}/10 (diagnostic)", flush=True)
            m = bf16(propagated_moe.output)
            r = bf16(inject(r, m, bf16(g.injection_scales)))
            for token in range(t):
                composition_screens += check(f"step{step} composed carry token{token} (diagnostic)", actual["moe_injected"][token], r[token], COMPOSED)
            composition_screens += check(f"step{step} composed final read (diagnostic)", actual["final_read"], gr(w, r, "final").mixed, COMPOSED)
            reference_carry, previous_actual = r[-1:], actual["moe_injected"]
    print(f"MTP independent propagated-chain 2% diagnostic screen failures={composition_screens}; "
          "local represented-input Op gates and exact state transitions are admission evidence, "
          "not full-model acceptance/PPL proof", flush=True)
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    failures = qualify(args.source, args.trace)
    print(f"Native MTP independent oracle failures={failures}", flush=True)
    raise SystemExit(int(failures != 0))
