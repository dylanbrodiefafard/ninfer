"""Qwen4 Gated DeltaNet mathematical oracles for both represented artifacts.

Each closed-Op helper evaluates its direct formula in float64 from represented
operands.  :func:`source_sublayer` and :func:`actual_gguf_sublayer` additionally
model the public representations that connect those Ops: BF16 projection and
convolution, recurrence, and gated-norm outputs; FP32 controls; BF16 convolution
history; and FP32 recurrent state.
Only the final output projection remains ideal, ready for comparison with its
represented production output under that Op's numerical criterion.

The pinned source checkpoint and converted GGUF do not expose the same operands:
the source stores ``A_log`` and grouped Q/K-to-V head association, while the GGUF
stores already-folded ``ssm_a=-exp(A_log)`` and tiled V-side order.  Their public
entry points are deliberately distinct so an actual-artifact comparison cannot
silently apply the source transform a second time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .common import as_f64, linear, ordinary_rmsnorm, sigmoid


@dataclass(frozen=True, slots=True)
class GDNResult:
    output: torch.Tensor
    final_state: torch.Tensor
    state_records: tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class GDNState:
    """Persistent state for one GDN layer and one request."""

    conv_history: torch.Tensor
    recurrent: torch.Tensor


@dataclass(frozen=True, slots=True)
class SourceGDNWeights:
    """Pinned source-checkpoint weights, including unfused ``A_log``."""

    qkv: torch.Tensor
    z: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    conv: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    norm: torch.Tensor
    output: torch.Tensor


@dataclass(frozen=True, slots=True)
class ActualGgufGDNWeights:
    """Converted GGUF weights, including direct folded ``ssm_a``."""

    qkv: torch.Tensor
    z: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    conv: torch.Tensor
    ssm_a: torch.Tensor
    dt_bias: torch.Tensor
    norm: torch.Tensor
    output: torch.Tensor


@dataclass(frozen=True, slots=True)
class GDNSublayerResult:
    """Final ideal output plus represented state and intermediate diagnostics."""

    output: torch.Tensor
    final_state: GDNState
    convolved_qkv: torch.Tensor
    repeated_query: torch.Tensor
    repeated_key: torch.Tensor
    log_decay: torch.Tensor
    beta: torch.Tensor
    core: torch.Tensor


def causal_depthwise_convolution(
    value: torch.Tensor,
    weight: torch.Tensor,
    initial_history: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a causal channel-wise convolution and SiLU to ``[T,C]``.

    ``initial_history`` is ``[C,K-1]`` in its persistent representation.  New
    raw projection rows remain ideal oracle values for the current call; only
    the returned persistent history crosses the declared representation
    boundary.  This matches the fused projection/convolution Op contract rather
    than inventing an intermediate projection cast.
    """

    if value.ndim != 2 or weight.ndim != 2 or initial_history.ndim != 2:
        raise ValueError("convolution operands must have shapes [T,C], [C,K], [C,K-1]")
    tokens, channels = value.shape
    if weight.shape[0] != channels or weight.shape[1] < 1:
        raise ValueError("depthwise weight must have shape [C,K] with K >= 1")
    width = weight.shape[1]
    if initial_history.shape != (channels, width - 1):
        raise ValueError("convolution history must have shape [C,K-1]")
    if not initial_history.dtype.is_floating_point:
        raise ValueError("convolution history must use a floating-point representation")

    projected = as_f64(value)
    sequence = torch.cat((as_f64(initial_history).transpose(0, 1), projected), dim=0)
    windowed = sequence.unfold(0, width, 1)
    convolved = torch.sum(as_f64(windowed) * as_f64(weight)[None, :, :], dim=-1)
    activated = F.silu(convolved)
    next_history = sequence[-(width - 1) :].transpose(0, 1).contiguous().to(initial_history.dtype)
    if width == 1:
        next_history = initial_history.detach().to(device="cpu").clone()
    if activated.shape != (tokens, channels):
        raise AssertionError("depthwise convolution produced an invalid shape")
    return activated, next_history


def source_control_gates(
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return controls from source ``A_log`` (the fold occurs in this formula)."""

    if a.ndim != 2 or b.shape != a.shape:
        raise ValueError("A and B projections must have matching shape [T,value_heads]")
    value_heads = a.shape[1]
    if a_log.numel() != value_heads or dt_bias.numel() != value_heads:
        raise ValueError("A_log and dt_bias must contain one value per value head")
    shifted = as_f64(a) + as_f64(dt_bias).reshape(1, value_heads)
    log_decay = -torch.exp(as_f64(a_log).reshape(1, value_heads)) * F.softplus(shifted)
    return log_decay, sigmoid(b)


def actual_gguf_control_gates(
    a: torch.Tensor,
    b: torch.Tensor,
    ssm_a: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return controls from converted GGUF ``ssm_a=-exp(A_log)`` directly."""

    if a.ndim != 2 or b.shape != a.shape:
        raise ValueError("A and B projections must have matching shape [T,value_heads]")
    value_heads = a.shape[1]
    if ssm_a.numel() != value_heads or dt_bias.numel() != value_heads:
        raise ValueError("ssm_a and dt_bias must contain one value per value head")
    shifted = as_f64(a) + as_f64(dt_bias).reshape(1, value_heads)
    return as_f64(ssm_a).reshape(1, value_heads) * F.softplus(shifted), sigmoid(b)


def repeat_grouped_query_key_heads(value: torch.Tensor, value_heads: int) -> torch.Tensor:
    """Expand source Q/K heads contiguously into grouped V-head order."""

    if value.ndim != 3 or value_heads % value.shape[1]:
        raise ValueError("value-head count must be a multiple of the Q/K head count")
    return as_f64(value).repeat_interleave(value_heads // value.shape[1], dim=1)


def tile_actual_gguf_query_key_heads(value: torch.Tensor, value_heads: int) -> torch.Tensor:
    """Expand converted-GGUF Q/K heads so represented V head ``h`` uses ``h % Hq``."""

    if value.ndim != 3 or value_heads % value.shape[1]:
        raise ValueError("value-head count must be a multiple of Q/K head count")
    return as_f64(value).repeat(1, value_heads // value.shape[1], 1)


def _l2norm(value: torch.Tensor, eps: float) -> torch.Tensor:
    value = as_f64(value)
    return value / torch.sqrt(torch.sum(value * value, dim=-1, keepdim=True) + eps)


def recurrence(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> GDNResult:
    """Apply decay, delta correction, state update, then scaled state read.

    Q/K are L2-normalized and queries receive the checkpoint's ``1/sqrt(K)``
    scale.  Each token transition is recorded to FP32 because that recurrent
    matrix is persistent request state, while each formula is evaluated
    independently in float64 from its represented operands.
    """

    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("Q/K/V must have shape [T,H,D]")
    tokens, key_heads, key_width = query.shape
    value_heads, value_width = value.shape[1:]
    if key.shape != query.shape or value_heads % key_heads:
        raise ValueError("invalid GDN head geometry")
    if log_decay.shape != (tokens, value_heads) or beta.shape != (tokens, value_heads):
        raise ValueError("log_decay and beta must have shape [T,value_heads]")
    expected_state = (value_heads, key_width, value_width)
    if tuple(initial_state.shape) != expected_state or initial_state.dtype != torch.float32:
        raise ValueError("GDN persistent state must be FP32 [value_heads,K,V]")
    q = _l2norm(query, eps) / math.sqrt(key_width)
    k = _l2norm(key, eps)
    v = as_f64(value)
    decay = as_f64(log_decay)
    learning_rate = as_f64(beta)
    state = initial_state.detach().to(device="cpu").clone()
    head_map = torch.arange(value_heads, dtype=torch.int64) // (value_heads // key_heads)
    output = torch.empty((tokens, value_heads, value_width), dtype=torch.float64)
    records: list[torch.Tensor] = []
    for token in range(tokens):
        token_key = k[token].index_select(0, head_map)
        token_query = q[token].index_select(0, head_map)
        decayed = as_f64(state) * torch.exp(decay[token])[:, None, None]
        prediction = torch.sum(decayed * token_key[:, :, None], dim=1)
        delta = learning_rate[token, :, None] * (v[token] - prediction)
        next_state = decayed + token_key[:, :, None] * delta[:, None, :]
        state = next_state.to(torch.float32)
        records.append(state.clone())
        output[token] = torch.sum(as_f64(state) * token_query[:, :, None], dim=1)
    return GDNResult(output=output, final_state=state, state_records=tuple(records))


def output_projection(
    core: torch.Tensor,
    gate: torch.Tensor,
    norm_weight: torch.Tensor,
    output_weight: torch.Tensor,
    *,
    eps: float = 1e-6,
    activation_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Apply gated norm, its represented boundary, and the final projection.

    The gated RMSNorm is a closed Op whose output is represented BF16 in the
    preview.  The final projection's ideal result remains float64; its own final
    storage rounding belongs to that Op's production comparison criterion.
    """

    if core.ndim != 3 or gate.shape != core.shape:
        raise ValueError("core and gate must have matching shape [T,value_heads,width]")
    normalized = ordinary_rmsnorm(core, norm_weight, eps=eps)
    represented = (normalized * sigmoid(gate)).to(activation_dtype)
    return linear(represented.flatten(-2), output_weight)


def _sublayer(
    hidden: torch.Tensor,
    weights: SourceGDNWeights | ActualGgufGDNWeights,
    initial_state: GDNState,
    *,
    key_heads: int,
    value_heads: int,
    key_width: int,
    value_width: int,
    actual_gguf: bool,
    eps: float = 1e-6,
) -> GDNSublayerResult:
    if hidden.ndim != 2 or hidden.shape[0] < 1:
        raise ValueError("hidden must have shape [T,hidden_width] with T >= 1")
    if min(key_heads, value_heads, key_width, value_width) < 1:
        raise ValueError("GDN head counts and widths must be positive")
    if value_heads % key_heads:
        raise ValueError("value_heads must be a multiple of key_heads")
    hidden_width = hidden.shape[1]
    qkv_width = 2 * key_heads * key_width + value_heads * value_width
    gate_width = value_heads * value_width
    expected_shapes = {
        "qkv": (qkv_width, hidden_width),
        "z": (gate_width, hidden_width),
        "a": (value_heads, hidden_width),
        "b": (value_heads, hidden_width),
        "conv": (qkv_width, 4),
        "norm": (value_width,),
    }
    for name, expected in expected_shapes.items():
        if tuple(getattr(weights, name).shape) != expected:
            raise ValueError(f"{name} weight must have shape {expected}")
    if weights.output.ndim != 2 or weights.output.shape[1] != gate_width:
        raise ValueError(f"output weight must have shape [output_width,{gate_width}]")
    decay_weight = weights.ssm_a if actual_gguf else weights.a_log
    if decay_weight.numel() != value_heads or weights.dt_bias.numel() != value_heads:
        name = "ssm_a" if actual_gguf else "A_log"
        raise ValueError(f"{name} and dt_bias must contain one value per value head")
    if tuple(initial_state.conv_history.shape) != (qkv_width, 3):
        raise ValueError(f"convolution history must have shape [{qkv_width},3]")
    if initial_state.conv_history.dtype != torch.bfloat16:
        raise ValueError("preview convolution history must be BF16")
    if initial_state.recurrent.dtype != torch.float32:
        raise ValueError("GDN recurrent state must be FP32")

    projected_qkv = linear(hidden, weights.qkv)
    convolved_qkv, next_history = causal_depthwise_convolution(
        projected_qkv,
        weights.conv,
        initial_state.conv_history,
    )
    # These values are public outputs of gdn_input_proj_conv_snapshot.  The
    # projection and convolution above remain one ideal fused formula, but its
    # consumers begin from the represented BF16 outputs.
    convolved_qkv = convolved_qkv.to(torch.bfloat16)
    z = linear(hidden, weights.z).to(torch.bfloat16).reshape(-1, value_heads, value_width)
    a = linear(hidden, weights.a)
    b = linear(hidden, weights.b)
    if actual_gguf:
        log_decay, beta = actual_gguf_control_gates(a, b, decay_weight, weights.dt_bias)
    else:
        log_decay, beta = source_control_gates(a, b, decay_weight, weights.dt_bias)
    # gdn_gating_proj publishes FP32 controls.
    log_decay = log_decay.to(torch.float32)
    beta = beta.to(torch.float32)

    key_span = key_heads * key_width
    query = convolved_qkv[:, :key_span].reshape(-1, key_heads, key_width)
    key = convolved_qkv[:, key_span : 2 * key_span].reshape(-1, key_heads, key_width)
    value = convolved_qkv[:, 2 * key_span :].reshape(-1, value_heads, value_width)
    expand_heads = (
        tile_actual_gguf_query_key_heads if actual_gguf else repeat_grouped_query_key_heads
    )
    repeated_query = expand_heads(query, value_heads)
    repeated_key = expand_heads(key, value_heads)
    core_result = recurrence(
        repeated_query,
        repeated_key,
        value,
        log_decay,
        beta,
        initial_state.recurrent,
        eps=eps,
    )
    # gated_delta_net publishes BF16 before gated_rmsnorm consumes it.
    core = core_result.output.to(torch.bfloat16)
    output = output_projection(core, z, weights.norm, weights.output, eps=eps)
    return GDNSublayerResult(
        output=output,
        final_state=GDNState(conv_history=next_history, recurrent=core_result.final_state),
        convolved_qkv=convolved_qkv,
        repeated_query=repeated_query,
        repeated_key=repeated_key,
        log_decay=log_decay,
        beta=beta,
        core=core,
    )


def source_sublayer(
    hidden: torch.Tensor,
    weights: SourceGDNWeights,
    initial_state: GDNState,
    *,
    key_heads: int,
    value_heads: int,
    key_width: int,
    value_width: int,
    eps: float = 1e-6,
) -> GDNSublayerResult:
    """Evaluate the pinned source-checkpoint GDN formula for one request."""

    return _sublayer(
        hidden,
        weights,
        initial_state,
        key_heads=key_heads,
        value_heads=value_heads,
        key_width=key_width,
        value_width=value_width,
        actual_gguf=False,
        eps=eps,
    )


def actual_gguf_sublayer(
    hidden: torch.Tensor,
    weights: ActualGgufGDNWeights,
    initial_state: GDNState,
    *,
    key_heads: int,
    value_heads: int,
    key_width: int,
    value_width: int,
    eps: float = 1e-6,
) -> GDNSublayerResult:
    """Evaluate converted-GGUF direct-decay/tiled-head semantics for one request.

    The preview artifact uses ``key_heads=16``, ``value_heads=48`` and
    ``key_width=value_width=128``.  ``weights.conv`` is the mathematical
    ``[QKV,4]`` view after adapting the GGUF K-fastest bytes.
    """

    return _sublayer(
        hidden,
        weights,
        initial_state,
        key_heads=key_heads,
        value_heads=value_heads,
        key_width=key_width,
        value_width=value_width,
        actual_gguf=True,
        eps=eps,
    )


__all__ = [
    "GDNResult",
    "GDNState",
    "GDNSublayerResult",
    "ActualGgufGDNWeights",
    "SourceGDNWeights",
    "actual_gguf_control_gates",
    "actual_gguf_sublayer",
    "causal_depthwise_convolution",
    "output_projection",
    "recurrence",
    "repeat_grouped_query_key_heads",
    "source_control_gates",
    "source_sublayer",
    "tile_actual_gguf_query_key_heads",
]
