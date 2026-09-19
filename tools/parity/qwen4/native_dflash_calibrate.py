"""Bounded, offline activation-weighted NVFP4 calibration for the exact DFlash.

Scale selection minimizes a diagonal second-moment approximation to projection
error. Complete held-out FP64 forwards, never this fitting objective, evaluate
propagated behavior. No norm, activation policy, global divisor or codebook changes.
"""
from __future__ import annotations

import torch

from tools.convert.common.nvfp4_quantize import quantize_nvfp4_matrix

SCALE_FACTORS = (.5, .625, .75, .875, 1., 1.125, 1.25, 1.375, 1.5)
PROFILE = "QWEN4_DFLASH_NVFP4_INPUT_SECOND_MOMENT_SCALE_SEARCH_V1"


def decode_e4m3(codes):
    words = codes.long()
    exponent, fraction = words >> 3, words & 7
    if bool((words >= 127).any()):
        raise ValueError("nonfinite or negative block scale")
    return torch.where(exponent == 0, fraction.double() * 2.**-9,
                       (1. + fraction.double()/8.) * 2.**(exponent.double()-7))


def decode_e2m1(codes):
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float64)
    return torch.copysign(table[(codes & 7).long()], torch.where(codes < 8, 1., -1.).double())


def nearest_e2m1(values):
    """Nearest representable signed code, even low bit wins exact midpoint ties."""
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float64)
    magnitude = values.abs()
    hi = torch.searchsorted(table, magnitude).clamp(max=7)
    lo = (hi-1).clamp(min=0)
    dl, dh = (magnitude-table[lo]).abs(), (table[hi]-magnitude).abs()
    code = torch.where((dl < dh) | ((dl == dh) & ((lo & 1) == 0)), lo, hi).to(torch.uint8)
    return code | (torch.signbit(values).to(torch.uint8) << 3)


def calibrate_matrix(weight, second_moment):
    """Select each K16 scale independently, preserving the old encoder as a candidate.

    Returned stored words use the existing NVFP4 layout. Moments contain only
    calibration-input E[x_k**2]; ignored cross-column covariance is explicit.
    No held-out inputs, labels or results enter this function.
    """
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError("calibration requires represented source BF16 weights")
    rows, width = weight.shape
    moment = second_moment.detach().double().reshape(-1)
    if moment.numel() != width or not torch.isfinite(moment).all() or (moment < 0).any():
        raise ValueError("invalid calibration second moment")
    packed, scales, divisor = quantize_nvfp4_matrix(weight)
    original_scales = scales.clone()
    original_codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    result_codes = original_codes.clone()
    baseline_loss = 0.; fitted_loss = 0.; changed = 0
    for start in range(0, rows, 128):
        end = min(start+128, rows)
        source = weight[start:end].double().reshape(end-start, width//16, 16)
        original_scale = decode_e4m3(original_scales[start:end])
        codes = result_codes[start:end].reshape_as(source)
        scale_words = scales[start:end]
        def loss(candidate_codes, candidate_scale):
            decoded = decode_e2m1(candidate_codes) * candidate_scale[..., None] / float(divisor)
            return ((decoded-source).square() * moment.reshape(1, width//16, 16)).sum(-1)
        best = loss(codes, original_scale)
        baseline_loss += float(best.sum())
        for factor in SCALE_FACTORS:
            candidate_words = (original_scale*factor).clamp(0,448).float().to(torch.float8_e4m3fn).view(torch.uint8)
            candidate_scale = decode_e4m3(candidate_words)
            denominator = torch.where(candidate_scale == 0, 1., candidate_scale)
            candidate_codes = nearest_e2m1(source*float(divisor)/denominator[..., None])
            score = loss(candidate_codes, candidate_scale)
            better = score < best
            codes.copy_(torch.where(better[..., None], candidate_codes, codes))
            scale_words.copy_(torch.where(better, candidate_words, scale_words))
            best = torch.minimum(best, score)
        fitted_loss += float(best.sum())
        changed += int((scale_words != original_scales[start:end]).sum())
    packed = result_codes[:, 0::2] | (result_codes[:, 1::2] << 4)
    if fitted_loss > baseline_loss:
        raise AssertionError("calibration objective worsened despite retaining baseline")
    return packed, scales, divisor, dict(baseline_diagonal_objective=baseline_loss,
        fitted_diagonal_objective=fitted_loss, changed_scale_groups=changed,
        scale_groups=scales.numel(), zero_moment_columns=int((moment == 0).sum()))


class Moments:
    def __init__(self):
        self.squares = {}
        self.counts = {}

    def add(self, role, values):
        values = values.double().reshape(-1, values.shape[-1])
        if not torch.isfinite(values).all():
            raise ValueError("nonfinite calibration activation")
        square = values.square().sum(0)
        self.squares[role] = self.squares.get(role, torch.zeros_like(square)) + square
        self.counts[role] = self.counts.get(role, 0) + len(values)

    def observe(self, features, result):
        self.add("fc.weight", features)
        for index, layer in enumerate(result.layers):
            p = f"layers.{index}."
            self.add(p+"self_attn.q_proj.weight", layer["input_norm"])
            for role in ("k", "v"):
                self.add(p+f"self_attn.{role}_proj.weight", result.context)
                self.add(p+f"self_attn.{role}_proj.weight", layer["input_norm"])
            self.add(p+"self_attn.o_proj.weight", layer["attention"].flatten(-2))
            for role in ("gate", "up"):
                self.add(p+f"mlp.{role}_proj.weight", layer["mlp_input"])
            self.add(p+"mlp.down_proj.weight", layer["activated"])

    def mean(self, role):
        return self.squares[role] / self.counts[role]
