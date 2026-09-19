"""Offline native layer-3 cache counterfactual on an authentic qualified prefix.

This is a numerical candidate screen, not an FP8 GPU codec, runtime, quality or
PPL admission. The suffix reuses the independent GR/MoE source formulas. Its
QSA formula evaluates explicitly decoded candidate persistent state in FP64.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from tools.artifact.container import Artifact
from tools.reference.qwen4.common import ideal_softmax, linear, partial_rope, sigmoid, source_grouped_rmsnorm
from tools.reference.qwen4.mtp import Weights, bf16, gr, inject, moe
from tools.reference.qwen4.qsa import source_sparse_attention


class LayerSource:
    """Reuse the existing exact NVFP4 expert decoder with main-layer source names."""
    def __init__(self, source):
        self.source = source

    def get_tensor(self, name):
        if not name.startswith("mtp.layers.0."):
            raise ValueError("assessment requested a tensor outside layer 3")
        return self.source.get_tensor(name.replace("mtp.layers.0.", "model.language_model.layers.3.", 1))


def e4m3_table():
    codes = torch.arange(127)
    exponent, mantissa = codes >> 3, codes & 7
    return torch.where(exponent == 0, mantissa.double() * 2.**-9,
                       (1 + mantissa.double()/8) * 2.**(exponent.double()-7))


def nearest(value, table):
    """Independent nearest-code search; even encoded index wins exact ties."""
    hi = torch.searchsorted(table, value.abs().double()).clamp(max=table.numel()-1)
    lo = (hi-1).clamp(min=0)
    dl, dh = (value.abs()-table[lo]).abs(), (table[hi]-value.abs()).abs()
    code = torch.where((dl < dh) | ((dl == dh) & ((lo & 1) == 0)), lo, hi)
    return code, torch.copysign(table[code], value.double())


def codec(value, profile):
    """BF16 append input; exact specified FP32 scale/division/reconstruction boundaries."""
    value = bf16(value).float()
    if profile == "bf16":
        return value.double()
    table = e4m3_table()
    if profile == "fp8":
        scale = value.abs().amax(-1, keepdim=True) / 448.
        scale = torch.where(scale == 0, 1., scale)
        _, decoded = nearest(value / scale, table)
        return bf16(decoded.float() * scale)
    if profile != "nvfp4":
        raise ValueError(profile)
    groups = value.reshape(*value.shape[:-1], 16, 16)
    _, scale = nearest(groups.abs().amax(-1, keepdim=True) / 6., table)
    scale = scale.float()
    normalized = groups / torch.where(scale == 0, 1., scale)
    _, decoded = nearest(normalized, torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float64))
    return bf16(decoded.float() * scale).reshape_as(value)


def codec_checks():
    table = e4m3_table()
    codes, values = nearest(table, table)
    assert torch.equal(codes, torch.arange(127)) and torch.equal(values, table)
    mid = (table[:-1] + table[1:]) / 2
    expected = torch.arange(126) + (torch.arange(126) & 1)
    assert torch.equal(nearest(mid, table)[0], expected)
    assert nearest(torch.tensor([1e20]), table)[0].item() == 126
    assert torch.signbit(nearest(torch.tensor([-0.]), table)[1]).item()
    assert torch.equal(nearest(-table, table)[1], -table)
    fp4 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float64)
    assert torch.equal(nearest(fp4, fp4)[0], torch.arange(8))
    assert torch.equal(nearest((fp4[:-1]+fp4[1:])/2, fp4)[0],
                       torch.arange(7)+(torch.arange(7) & 1))
    for profile in ("fp8", "nvfp4"):
        assert torch.equal(codec(torch.zeros(2, 2, 256), profile), torch.zeros(2, 2, 256))


def qsa(w, hidden, profile):
    p = "layers.0.self_attn."
    count = len(hidden)
    qg = linear(hidden, w.get(p+"q_proj.weight")).reshape(count, 24, 512)
    raw_k = linear(hidden, w.get(p+"k_proj.weight")).reshape(count, 2, 256)
    raw_v = linear(hidden, w.get(p+"v_proj.weight")).reshape(count, 2, 256)
    positions = torch.arange(count)[None, :] + torch.arange(3)[:, None]
    def normalized(x, role):
        return partial_rope(source_grouped_rmsnorm(x, w.get(p+role+"_norm.weight"), group_size=256),
                            positions, rotary_dim=64, theta=1e7, mrope_section=(11, 11, 10))
    query = normalized(qg[:, :, :256], "q")
    keys = codec(normalized(raw_k, "k"), "fp8" if profile == "fp8_k" else
                 "nvfp4" if profile.startswith("nvfp4") else "bf16")
    values = codec(raw_v, "nvfp4" if profile == "nvfp4_both" else "bf16")
    # T33 has <512 complete blocks: selection must contain every causal ID,
    # regardless of block score ordering. The raw BF16 selector is unchanged.
    selections = [torch.arange(token+1) for token in range(count)]
    core = torch.empty_like(query)
    for token, ids in enumerate(selections):
        for head in range(24):
            scores = keys[ids, head//12] @ query[token, head] / 16.
            core[token, head] = (ideal_softmax(scores)[:, None] * values[ids, head//12]).sum(0)
    output = linear(core.flatten(-2) * sigmoid(qg[:, :, 256:].flatten(-2)), w.get(p+"o_proj.weight"))
    if profile == "bf16":
        independent = source_sparse_attention(qg[:, :, :256], raw_k, raw_v, selections,
            positions, positions, w.get(p+"q_norm.weight"), w.get(p+"k_norm.weight"),
            qg[:, :, 256:].flatten(-2), w.get(p+"o_proj.weight"),
            core_cache_dtype=torch.bfloat16, mrope_section=(11, 11, 10)).output
        if not torch.equal(output, independent):
            raise ValueError("explicit-state attention differs from the existing QSA formula")
    return bf16(output)


def suffix(w, residual, mixed, write, profile):
    attention = qsa(w, mixed, profile)
    carry = bf16(inject(residual, attention, write))
    read = gr(w, carry, "mlp")
    experts = moe(w, bf16(read.mixed))
    final = bf16(inject(carry, bf16(experts.output), bf16(read.injection_scales)))
    return attention, final, experts.expert_ids


def metrics(actual, expected, criterion=(.02, .005, .02)):
    # Supplementary whole-panel screen. In particular this is NOT the QSA Op's
    # per-token represented-input arithmetic gate or a cache-quality admission.
    delta = actual - expected
    rel = float(delta.norm() / expected.norm().clamp_min(1e-300))
    maximum = float(delta.abs().max())
    limit = criterion[1] + criterion[2] * float(expected.abs().max())
    rows = delta.flatten(1).norm(dim=1) / expected.flatten(1).norm(dim=1).clamp_min(1e-300)
    return dict(relative_l2=rel, maximum_absolute=maximum, gross_limit=limit,
                per_token_relative_l2=rows.tolist(),
                passed=bool(torch.isfinite(actual).all()) and rel <= criterion[0] and maximum <= limit)


def run(root, capture, output, profiles):
    if output.exists():
        raise FileExistsError(output)
    codec_checks()
    metadata = json.loads((capture / "capture.json").read_text())
    if (metadata["profile"] != "qwen4-native-text33-layer3-kv-assessment" or
            metadata["prefix_failures"] or metadata["positions"] != "t+axis" or
            metadata["dtype"] != "little-endian-f32-representing-bf16"):
        raise ValueError("capture is not the qualified authentic prefix")
    with Artifact(root / "qwen4-text-panel.ninfer") as panel:
        tokens = torch.frombuffer(bytearray(panel.payload("token.ids")), dtype=torch.int32).tolist()
        if tokens != metadata["tokens"] or len(tokens) != 33:
            raise ValueError("captured prefix token identity differs")
    def load(name, side, width):
        value = torch.frombuffer(bytearray((capture / f"{name}.{side}.f32").read_bytes()), dtype=torch.float32)
        if value.numel() != 33*width or not torch.equal(value, value.bfloat16().float()):
            raise ValueError("capture extent/represented BF16 boundary differs")
        return value.double().reshape(33, 4, 2560) if width == 10240 else value.double().reshape(33, width)
    results = {}
    with safe_open(str(root / "qwen4-layer-3.safetensors"), framework="pt", device="cpu") as source:
        w = Weights(LayerSource(source))
        reference = suffix(w, load("residual", "reference", 10240), load("mixed", "reference", 2560),
                           load("write", "reference", 4), "bf16")
        # Cross-language independent FP64 formulas should differ by no more than
        # occasional public BF16 boundary rounding, under the original screen.
        validation = {"qsa": metrics(reference[0], load("qsa", "reference", 2560)),
                      "final": metrics(reference[1], load("final", "reference", 10240))}
        if not all(item["passed"] for item in validation.values()):
            raise ValueError(f"existing accumulated reference mismatch: {validation}")
        baseline = None
        for profile in ["bf16", *profiles]:
            actual = suffix(w, load("residual", "actual", 10240), load("mixed", "actual", 2560),
                            load("write", "actual", 4), profile)
            if baseline is None:
                baseline = actual
                validation["gpu_qsa_same_input"] = metrics(load("qsa", "actual", 2560),
                    actual[0], (.02, .00025, .02))
                validation["gpu_final_same_prefix"] = metrics(load("final", "actual", 10240), actual[1])
                if not all(item["passed"] for item in validation.values()):
                    raise ValueError(f"captured GPU baseline mismatch: {validation}")
            item = {"qsa_accumulated": metrics(actual[0], reference[0], (.02, .00025, .02)),
                    "final_accumulated": metrics(actual[1], reference[1]),
                    "qsa_storage_only": metrics(actual[0], baseline[0], (.02, .00025, .02)),
                    "final_storage_only": metrics(actual[1], baseline[1]),
                    "router_membership_changes": sum(len(set(a.tolist())-set(b.tolist()))
                                                     for a,b in zip(actual[2], baseline[2])),
                    "core_kv_payload_fraction": {"bf16": 1., "fp8_k": .75390625,
                        "nvfp4_k": .640625, "nvfp4_both": .28125}[profile]}
            results[profile] = item
            print(profile, "QSA", item["qsa_accumulated"]["relative_l2"], "final",
                  item["final_accumulated"]["relative_l2"], "aggregate_screen_pass",
                  item["qsa_accumulated"]["passed"] and item["final_accumulated"]["passed"], flush=True)
        w.get.cache_clear(); w.expert.cache_clear()
    report = dict(capture=metadata, profiles=results, reference_validation=validation,
        criterion="Supplementary whole-panel QSA source-loss screen {.02,.00025,.02}, not the existing per-token same-input Op gate; whole-panel accumulated screen {.02,.005,.02}. The passed fields mean aggregate screen only, never runtime or quality admission.",
        boundary="One authentic33 prefix, CPU counterfactual suffix only; no GPU candidate codec, long-context selection, runtime admission, PPL or full-model quality. Per-token drift and routing changes remain visible even when aggregate screens pass. Payload fraction excludes unchanged selector/positions/page metadata. NVFP4 controls isolate V loss; no new production precision.")
    with output.open("x") as file:
        json.dump(report, file, indent=2); file.write("\n")
    return 0 if all(v["qsa_accumulated"]["passed"] and v["final_accumulated"]["passed"]
                    for v in results.values()) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=("fp8_k", "nvfp4_k", "nvfp4_both"),
                        default=["fp8_k", "nvfp4_k", "nvfp4_both"])
    args = parser.parse_args()
    torch.set_num_threads(8)
    raise SystemExit(run(args.root, args.capture, args.output, args.profiles))
