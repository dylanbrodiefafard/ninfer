"""Per-Layer Embedding injection oracle over caller-supplied table rows."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .common import grouped_zero_centered_rmsnorm, linear, sigmoid, silu


@dataclass(frozen=True, slots=True)
class PLEResult:
    output: torch.Tensor
    injection: torch.Tensor
    next_conv_state: torch.Tensor
    gate: torch.Tensor
    gated_value: torch.Tensor


def dilated_depthwise_conv(
    value: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    *,
    dilation: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = value.detach().to(dtype=torch.float64, device="cpu")
    w = weight.detach().to(dtype=torch.float64, device="cpu")
    previous = state.detach().to(dtype=torch.float64, device="cpu")
    if x.ndim != 2 or w.ndim != 2 or previous.ndim != 2:
        raise ValueError("convolution tensors must be [T,C], [C,K], and [C,state]")
    channels, kernel = w.shape
    state_len = (kernel - 1) * dilation
    if x.shape[1] != channels or previous.shape != (channels, state_len):
        raise ValueError("PLE convolution state or channel shape mismatch")
    sequence = torch.cat((previous.transpose(0, 1), x), dim=0)
    output = torch.empty_like(x)
    for token in range(x.shape[0]):
        samples = sequence[token : token + state_len + 1 : dilation]
        output[token] = torch.sum(samples * w.transpose(0, 1), dim=0)
    next_state = sequence[-state_len:].transpose(0, 1).contiguous() if state_len else previous[:, :0]
    return output, next_state


def inject(
    residual_branches: torch.Tensor,
    table_rows: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    key_norm_weight: torch.Tensor,
    query_norm_weight: torch.Tensor,
    conv_norm_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    *,
    dilation: int = 3,
    eps: float = 1e-6,
) -> PLEResult:
    """Evaluate lookup projections, contextual gate, and continuation convolution.

    ``table_rows`` is the already gathered represented tensor ``[T,heads,row_width]``;
    the oracle never reaches into a live embedding module or artifact.
    """

    if residual_branches.ndim != 3:
        raise ValueError("residual_branches must have shape [T,branches,hidden]")
    tokens, branch_count, hidden = residual_branches.shape
    embedding = table_rows.detach().to(dtype=torch.float64, device="cpu").reshape(tokens, -1)
    flat_residual = residual_branches.detach().to(dtype=torch.float64, device="cpu").flatten(-2)
    keys = grouped_zero_centered_rmsnorm(
        linear(embedding, key_weight), key_norm_weight, group_size=hidden, eps=eps
    ).reshape(tokens, branch_count, hidden)
    queries = grouped_zero_centered_rmsnorm(
        flat_residual, query_norm_weight, group_size=hidden, eps=eps
    ).reshape(tokens, branch_count, hidden)
    raw_gate = torch.sum(keys * queries, dim=-1, keepdim=True) / (hidden**0.5)
    gate = torch.sign(raw_gate) * torch.sqrt(torch.clamp(torch.abs(raw_gate), min=1e-6))
    value = linear(embedding, value_weight)
    gated = sigmoid(gate) * value.unsqueeze(-2)
    gated_flat = gated.flatten(-2)
    ideal_conv_input = grouped_zero_centered_rmsnorm(
        gated_flat, conv_norm_weight, group_size=hidden, eps=eps
    )
    # The PLE convolution history is persistent represented state.  Current
    # rows cross that same boundary before the convolution can observe them;
    # otherwise a one-shot FP64 evaluation would not describe chunked BF16
    # continuation.
    conv_input = ideal_conv_input.to(conv_state.dtype).to(torch.float64)
    convolved, next_state = dilated_depthwise_conv(
        conv_input, conv_weight, conv_state, dilation=dilation
    )
    injection = (gated_flat + silu(convolved)).reshape(tokens, branch_count, hidden)
    return PLEResult(
        output=residual_branches.detach().to(dtype=torch.float64, device="cpu") + injection,
        injection=injection,
        next_conv_state=next_state.to(conv_state.dtype),
        gate=gate,
        gated_value=gated,
    )


__all__ = ["PLEResult", "dilated_depthwise_conv", "inject"]
