"""Check native DFlash GPU public-Op traces against independent same-input FP64 math.

Every local oracle consumes the ACTUAL preceding GPU public output, not a composed
reference's intermediate values. Packed weights are decoded exactly in FP64.
The independent whole-formula and public-BF16 composition screens remain separate.
Exit status qualifies local Op correctness only; failed quality screens are always
reported and never promoted to a passing model-quality claim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch

from tools.parity.qwen4.native_dflash_goldens import read_weights
from tools.reference.qwen4.common import linear, ordinary_rmsnorm, partial_rope, silu
from tools.reference.qwen4.dflash import attention

# Mirrors the public diagnostic boundary enum, not production arithmetic.
BOUNDARIES = (
    "FeatureProjection", "FusedContext", "ContextKeyProjection", "ContextValue", "ContextKeyNorm",
    "ContextKey", "InputNorm", "QueryProjection", "KeyProjection", "Value", "QueryNorm", "KeyNorm",
    "Query", "Key", "Attention", "AttentionProjection", "AttentionResidual", "PostNorm", "Gate", "Up",
    "Activated", "Down", "LayerOutput", "FinalOutput",
)

# Existing independent Op qualification criteria, fixed before inspecting GPU traces.
REDUCTION_CRITERIA = {
    "linear": (1. / 256., 1. / 256., 2. / 256.),
    "rmsnorm": (1.85e-3, 1e-5, 3.4e-3),
    "attention": (2.95e-3, 3e-4, 5.7e-3),
    "composition": (.02, .005, .02),
}
POINTWISE_CRITERIA = {"silu_mul": (2e-5, 4.05e-3), "residual_add": (0., 3.95e-3)}
ROPE_PAIR_RELATIVE = 6.9e-3
EPSILON = float(torch.tensor(1e-6, dtype=torch.float32))


def nearest_bf16(value: torch.Tensor) -> torch.Tensor:
    """Nearest finite BF16 from FP64 without a possible FP32 double-rounding tie.

    The ordinary cast supplies a nearby code only. Compare its two neighbors in
    FP64 and use the even code on exact ties; this is a representability proof,
    not an implementation arithmetic profile or a relaxed tolerance.
    """
    value = value.double()
    candidate = value.bfloat16().view(torch.uint16).to(torch.int64)
    sign = candidate & 0x8000
    magnitude = (candidate & 0x7fff).clamp(max=0x7f7f)
    best = torch.zeros_like(value)
    best_error = torch.full_like(value, math.inf)
    best_even = torch.zeros_like(value, dtype=torch.bool)
    for offset in (-1, 0, 1):
        word = (magnitude + offset).clamp(min=0, max=0x7f7f) | sign
        represented = word.to(torch.uint16).view(torch.bfloat16).double()
        distance = (represented - value).abs()
        even = (word & 1) == 0
        take = (distance < best_error) | ((distance == best_error) & even & ~best_even)
        best = torch.where(take, represented, best)
        best_error = torch.where(take, distance, best_error)
        best_even = torch.where(take, even, best_even)
    return best


def compare(actual: torch.Tensor, expected: torch.Tensor, kind: str,
            rope_input: torch.Tensor | None = None) -> dict:
    actual, expected = actual.double(), expected.double()
    if actual.shape != expected.shape:
        raise ValueError(f"trace/oracle shape mismatch: {actual.shape} vs {expected.shape}")
    if not actual.numel():
        raise ValueError("a traced Op output must not be empty")
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    if not finite:
        return dict(passed=False, finite=False, elements=actual.numel(), criterion=kind)
    error = (actual - expected).abs()
    expected_norm = float(torch.linalg.vector_norm(expected))
    error_norm = float(torch.linalg.vector_norm(error))
    relative = error_norm / max(expected_norm, 1e-30)
    result = dict(finite=True, elements=actual.numel(), criterion=kind,
                  relative_l2=relative, max_abs=float(error.max()))
    if kind in REDUCTION_CRITERIA:
        limit, absolute, gross = REDUCTION_CRITERIA[kind]
        envelope = absolute + gross * float(expected.abs().max())
        violating = error > envelope
        passed = relative <= limit and not bool(violating.any())
        result.update(relative_l2_limit=limit, gross_envelope=envelope)
    elif kind in POINTWISE_CRITERIA:
        absolute, rel = POINTWISE_CRITERIA[kind]
        violating = error > absolute + rel * expected.abs()
        passed = not bool(violating.any())
        result.update(absolute_limit=absolute, relative_limit=rel)
    elif kind == "rope":
        if rope_input is None or rope_input.shape != expected.shape or expected.shape[-1] % 2:
            raise ValueError("RoPE qualification needs represented input pairs")
        half = expected.shape[-1] // 2
        pairs = torch.sqrt(rope_input[..., :half].double().square()
                           + rope_input[..., half:].double().square())
        envelope = torch.cat((pairs, pairs), dim=-1) * ROPE_PAIR_RELATIVE
        violating = error > envelope
        passed = not bool(violating.any())
        result.update(pair_relative_limit=ROPE_PAIR_RELATIVE)
    else:
        raise ValueError(f"unknown numerical criterion {kind}")
    result.update(passed=passed, gross_violations=int(violating.sum()))
    if bool(violating.any()):
        index = torch.nonzero(violating, as_tuple=False)[0].tolist()
        result["first_violation"] = dict(index=index, actual=float(actual[tuple(index)]),
                                         reference=float(expected[tuple(index)]))
        floor = nearest_bf16(expected)
        floor_error = (floor - expected).abs()
        allowed = (envelope if kind in REDUCTION_CRITERIA or kind == "rope"
                   else absolute + rel * expected.abs())
        impossible = floor_error > allowed
        result["bf16_representability"] = dict(
            scope="diagnostic only; the original gate above remains unchanged",
            unavoidable_gross_violations=int(impossible.sum()),
            actual_is_nearest_at_all_gross_violations=bool(torch.equal(actual[violating], floor[violating])),
            actual_is_nearest_entire_output=bool(torch.equal(actual, floor)),
            nearest_at_first_violation=float(floor[tuple(index)]),
            minimum_error_at_first_violation=float(floor_error[tuple(index)]))
    return result


def read_trace(path: Path) -> dict[tuple[str, int], torch.Tensor]:
    metadata = json.loads(path.with_suffix(".json").read_text())
    payload = path.read_bytes()
    result = {}
    expected_offset = 0
    for plane in metadata:
        stage, layer = plane["stage"], plane["layer"]
        if not isinstance(stage, int) or not 0 <= stage < len(BOUNDARIES):
            raise ValueError("unknown native DFlash trace stage")
        key = BOUNDARIES[stage], layer
        shape, begin, count = plane["shape"], plane["offset"], plane["bytes"]
        if (key in result or len(shape) != 4 or any(d <= 0 for d in shape)
                or count != 2 * math.prod(shape) or begin != expected_offset
                or begin + count > len(payload)):
            raise ValueError("duplicate or malformed native DFlash trace plane")
        # C++ Tensor stores the first dimension contiguously. Reverse metadata
        # to logical [batch,token,head,dimension], then consumers name their views.
        result[key] = torch.frombuffer(bytearray(payload[begin:begin + count]),
                                      dtype=torch.bfloat16).double().reshape(tuple(reversed(shape)))
        expected_offset += count
    if expected_offset != len(payload):
        raise ValueError("unreferenced bytes in native DFlash trace")
    return result


def read_goldens(path: Path, case: str) -> tuple[dict, dict[str, torch.Tensor]]:
    metadata = json.loads(path.with_suffix(".json").read_text())
    spec = next(item for item in metadata["cases"] if item["name"] == case)
    result = {}
    dtypes = {"BF16": torch.bfloat16, "F64": torch.float64, "I64": torch.int64}
    with path.open("rb") as stream:
        for plane in metadata["planes"]:
            prefix = case + "/"
            if not plane["name"].startswith(prefix):
                continue
            stream.seek(plane["offset"])
            payload = stream.read(plane["bytes"])
            dtype = dtypes[plane["dtype"]]
            tensor = (torch.frombuffer(bytearray(payload), dtype=dtype)
                      if payload else torch.empty(0, dtype=dtype))
            result[plane["name"][len(prefix):]] = tensor.reshape(plane["shape"])
    return spec, result


@dataclass
class LocalOracle:
    """Every dependency lookup is an actual observed public GPU tensor."""
    trace: dict[tuple[str, int], torch.Tensor]
    weights: dict[str, torch.Tensor]
    inputs: dict[str, torch.Tensor]
    context_count: int
    query_count: int

    def value(self, stage: str, layer: int, rows: int) -> torch.Tensor:
        return self.trace[stage, layer].reshape(rows, -1)

    def expected(self, stage: str, layer: int) -> tuple[torch.Tensor, str, torch.Tensor | None]:
        w, t, c = self.weights, self.query_count, self.context_count
        p = f"layers.{layer}."
        norm = lambda x, name: ordinary_rmsnorm(x, w[name], eps=EPSILON)
        x = lambda name: self.value(name, layer, t)
        prior = lambda: (self.inputs["input/noise_embeddings"].double() if layer == 0
                         else self.value("LayerOutput", layer - 1, t))
        if stage == "FeatureProjection":
            return linear(self.inputs["input/features"], w["fc.weight"]), "linear", None
        if stage == "FusedContext":
            return norm(self.value("FeatureProjection", -1, c), "hidden_norm.weight"), "rmsnorm", None
        if stage in ("ContextKeyProjection", "ContextValue"):
            name = "k_proj" if stage == "ContextKeyProjection" else "v_proj"
            return linear(self.value("FusedContext", -1, c), w[p + f"self_attn.{name}.weight"]), "linear", None
        if stage == "ContextKeyNorm":
            width = w[p + "self_attn.k_norm.weight"].numel()
            value = self.value("ContextKeyProjection", layer, c).reshape(c, -1, width)
            return norm(value, p + "self_attn.k_norm.weight"), "rmsnorm", None
        if stage == "ContextKey":
            width = w[p + "self_attn.k_norm.weight"].numel()
            value = self.value("ContextKeyNorm", layer, c).reshape(c, -1, width)
            return partial_rope(value, self.inputs["input/context_positions"], rotary_dim=width, theta=1e7), "rope", value
        if stage == "InputNorm":
            return norm(prior(), p + "input_layernorm.weight"), "rmsnorm", None
        projections = {"QueryProjection": ("InputNorm", "self_attn.q_proj.weight"),
                       "KeyProjection": ("InputNorm", "self_attn.k_proj.weight"),
                       "Value": ("InputNorm", "self_attn.v_proj.weight"),
                       "AttentionProjection": ("Attention", "self_attn.o_proj.weight"),
                       "Gate": ("PostNorm", "mlp.gate_proj.weight"),
                       "Up": ("PostNorm", "mlp.up_proj.weight"),
                       "Down": ("Activated", "mlp.down_proj.weight")}
        if stage in projections:
            source, matrix = projections[stage]
            return linear(x(source), w[p + matrix]), "linear", None
        if stage in ("QueryNorm", "KeyNorm"):
            source, role = ("QueryProjection", "q_norm") if stage == "QueryNorm" else ("KeyProjection", "k_norm")
            width = w[p + f"self_attn.{role}.weight"].numel()
            return norm(x(source).reshape(t, -1, width), p + f"self_attn.{role}.weight"), "rmsnorm", None
        if stage in ("Query", "Key"):
            role = "q_norm" if stage == "Query" else "k_norm"
            width = w[p + f"self_attn.{role}.weight"].numel()
            value = x(stage + "Norm").reshape(t, -1, width)
            return partial_rope(value, self.inputs["input/query_positions"], rotary_dim=width, theta=1e7), "rope", value
        if stage == "Attention":
            width = w[p + "self_attn.k_norm.weight"].numel()
            query = x("Query").reshape(t, -1, width)
            key = x("Key").reshape(t, -1, width)
            value = x("Value").reshape(t, -1, width)
            if c:
                # These are the actual public BF16 cache append values, not
                # projected keys from an independently propagated reference.
                key = torch.cat((self.value("ContextKey", layer, c).reshape(c, -1, width), key))
                value = torch.cat((self.value("ContextValue", layer, c).reshape(c, -1, width), value))
            reference = attention(query, key, value, self.inputs["input/context_positions"],
                                  int(self.inputs["input/query_positions"][0]), t)
            return reference, "attention", None
        if stage == "AttentionResidual":
            return prior() + x("AttentionProjection"), "residual_add", None
        if stage == "PostNorm":
            return norm(x("AttentionResidual"), p + "post_attention_layernorm.weight"), "rmsnorm", None
        if stage == "Activated":
            return silu(x("Gate")) * x("Up"), "silu_mul", None
        if stage == "LayerOutput":
            return x("AttentionResidual") + x("Down"), "residual_add", None
        if stage == "FinalOutput":
            return norm(self.value("LayerOutput", 4, t), "norm.weight"), "rmsnorm", None
        raise ValueError(f"unsupported native DFlash boundary {stage}")


def check_case(weights: dict[str, torch.Tensor], golden_path: Path,
               trace_path: Path, case: str) -> dict:
    spec, inputs = read_goldens(golden_path, case)
    trace = read_trace(trace_path)
    c, t = spec["context"], spec["queries"]
    required = {("FinalOutput", -1)}
    if c:
        required |= {("FeatureProjection", -1), ("FusedContext", -1)}
        required |= {(name, layer) for name in BOUNDARIES[2:6] for layer in range(5)}
    required |= {(name, layer) for name in BOUNDARIES[6:23] for layer in range(5)}
    if trace.keys() != required:
        raise ValueError(f"incomplete/extra DFlash trace boundaries: missing={required - trace.keys()}, extra={trace.keys() - required}")
    oracle = LocalOracle(trace, weights, inputs, c, t)
    reports = []
    for (name, layer), actual in trace.items():
        expected, kind, rope_input = oracle.expected(name, layer)
        metrics = compare(actual.reshape(expected.shape), expected, kind, rope_input)
        reports.append(dict(stage=name, layer=layer, **metrics))
        if not metrics["passed"]:
            print(f"FAIL local {case} {name}[{layer}]: {json.dumps(metrics)}", flush=True)
    actual = trace["FinalOutput", -1].reshape(t, -1)
    screens = {"whole_unmaterialized": compare(actual, inputs["hidden"], "composition"),
               "public_bf16_composition": compare(actual, inputs["materialized/hidden"], "composition")}
    local_pass = all(item["passed"] for item in reports)
    print(f"{case}: local_ops={'PASS' if local_pass else 'FAIL'} ({len(reports)}), "
          f"whole_ideal={screens['whole_unmaterialized'].get('relative_l2', 'nonfinite')} "
          f"({'PASS' if screens['whole_unmaterialized']['passed'] else 'FAIL'}), "
          f"public_bf16_composition={screens['public_bf16_composition'].get('relative_l2', 'nonfinite')} "
          f"({'PASS' if screens['public_bf16_composition']['passed'] else 'FAIL'})", flush=True)
    return dict(case=case, local_ops_pass=local_pass, operators=reports, separate_screens=screens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile", choices=("bf16", "nvfp4"))
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    torch.set_num_threads(8)
    profiles = (args.profile,) if args.profile else ("bf16", "nvfp4")
    result = {}
    for profile in profiles:
        weights = read_weights(args.model_dir / f"qwen4-dflash-{profile}.ninfer")
        golden = args.model_dir / f"qwen4-dflash-{profile}-goldens.bin"
        cases = json.loads(golden.with_suffix(".json").read_text())["cases"]
        result[profile] = [check_case(weights, golden, args.trace_dir / f"{profile}-{case['name']}.bin", case["name"])
                           for case in cases]
        del weights
    passed = all(case["local_ops_pass"] for cases in result.values() for case in cases)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        json.dump(dict(local_ops_pass=passed, profiles=result,
                       scope="Same-input independent mathematical Op correctness; not full-target acceptance/PPL or NVFP4 draft quality",
                       separate_screens="unchanged 2% whole-unmaterialized and public-BF16 composition; not merged into local Op admission"), stream, indent=2)
        stream.write("\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
