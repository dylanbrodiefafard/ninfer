"""PPL scheme registry.

Each scheme is one Engine configuration scored against the BF16-KV baseline.
Add a later attention implementation here; do not invent a fourth kv-dtype until
the Engine actually stores one. MTP / graphs-off / mid-page skip are runner extras
in `run.py`, not extra scheme names.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Scheme:
    name: str
    kv_dtype: str
    # Extra flags forwarded to ninfer-ppl after the standard ones. Use this when a
    # future attention path grows an Engine switch, e.g. ("--attention", "flash").
    extra_args: tuple[str, ...] = field(default_factory=tuple)


BASELINE = "kv-bf16"

SCHEMES: dict[str, Scheme] = {
    "kv-bf16": Scheme("kv-bf16", "bf16"),
    "kv-int8": Scheme("kv-int8", "int8"),
    "kv-nvfp4": Scheme("kv-nvfp4", "nvfp4"),
    # Approx-attention treatments (NVFP4 KV):
    #   attn-sage: the sage_attn FP4-PV recipe, exact (all key tiles kept);
    #   attn-topk: Sparge tile-skip on exact NVFP4 (keep_frac of the key tiles);
    #   attn-xattn: XAttention mass-threshold skip on exact NVFP4.
    "attn-sage": Scheme("attn-sage", "nvfp4", ("--sage",)),
    "attn-topk": Scheme("attn-topk", "nvfp4", ("--keep-frac", "0.5")),
    "attn-xattn": Scheme("attn-xattn", "nvfp4", ("--xattn-tau", "0.9")),
    # attn-tma: same numerics as attn-sage, but the S3 (nvfp4s3) prefill kernel runs
    # through the TMA + mbarrier pipeline (NINFER_S3_TMA=1, exact attention only).
    "attn-tma": Scheme("attn-tma", "nvfp4", ("--sage", "--s3-tma")),
}

# Baseline first. Additional attention schemes append after the KV codecs.
ORDER: tuple[str, ...] = (
    "kv-bf16", "kv-int8", "kv-nvfp4", "attn-sage", "attn-topk", "attn-xattn", "attn-tma")
