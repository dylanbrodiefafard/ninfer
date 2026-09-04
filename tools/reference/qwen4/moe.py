"""Small-shape complete Sparse-MoE mathematical oracle."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .common import as_f64, ideal_softmax, linear, sigmoid, silu


@dataclass(frozen=True, slots=True)
class SparseMoeResult:
    output: torch.Tensor
    router_logits: torch.Tensor
    expert_ids: torch.Tensor
    route_weights: torch.Tensor


def route(router_logits: torch.Tensor, experts_per_token: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Ideal full-router softmax followed by stable top-k and selected renormalization."""

    probabilities = ideal_softmax(router_logits)
    ids = torch.argsort(probabilities, dim=-1, descending=True, stable=True)[
        :, :experts_per_token
    ]
    selected = torch.gather(probabilities, -1, ids)
    return selected / selected.sum(dim=-1, keepdim=True), ids


def _swiglu(
    value: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    return linear(silu(linear(value, gate_weight)) * linear(value, up_weight), down_weight)


def sparse_moe(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    expert_gate_weight: torch.Tensor,
    expert_up_weight: torch.Tensor,
    expert_down_weight: torch.Tensor,
    shared_gate_weight: torch.Tensor,
    shared_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_scale_weight: torch.Tensor,
    *,
    experts_per_token: int = 10,
) -> SparseMoeResult:
    """Evaluate all selected routed experts and the independently gated shared expert."""

    x = as_f64(hidden)
    logits = linear(x, router_weight)
    route_weights, expert_ids = route(logits, experts_per_token)
    tokens, hidden_width = x.shape
    if expert_gate_weight.shape[0] != router_weight.shape[0]:
        raise ValueError("expert bank and router sizes differ")
    routed = torch.zeros((tokens, hidden_width), dtype=torch.float64)
    for token in range(tokens):
        for route_slot in range(experts_per_token):
            expert = int(expert_ids[token, route_slot])
            expert_value = _swiglu(
                x[token],
                expert_gate_weight[expert],
                expert_up_weight[expert],
                expert_down_weight[expert],
            )
            routed[token] += route_weights[token, route_slot] * expert_value
    shared = _swiglu(x, shared_gate_weight, shared_up_weight, shared_down_weight)
    shared_scale = sigmoid(linear(x, shared_scale_weight))
    return SparseMoeResult(
        output=routed + shared_scale * shared,
        router_logits=logits,
        expert_ids=expert_ids,
        route_weights=route_weights,
    )


__all__ = ["SparseMoeResult", "route", "sparse_moe"]
