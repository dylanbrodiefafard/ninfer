"""Closed tensor inventory for the UD-IQ1_S architecture verifier.

The verifier is intentionally not a registered target.  Names follow the pinned
GGUF source because the conversion is byte-preserving; the C++ verifier binds
this exact closed inventory and never parses GGUF.
"""

from __future__ import annotations

from dataclasses import dataclass


MODEL_ID = "qwen4/verification"
WEIGHTS_ID = "unsloth-ud-iq1-s-host-staged"
LAYOUT = "ggml-block-row-v1"
CONTIGUOUS = "contiguous-le-v1"

TEXT_LAYERS = tuple(range(48))
QSA_LAYERS = tuple(range(3, 48, 4))
GDN_LAYERS = tuple(layer for layer in TEXT_LAYERS if layer not in QSA_LAYERS)
IQ2_EXPERT_LAYERS = (1, 2, 4, 14, 16, 25, 30, 32, 37, 39, 42, 45, 46, 47)
Q6_GDN_INPUT_LAYERS = (2,)
Q6_SHARED_GATE_UP_LAYERS = (2,)


@dataclass(frozen=True, slots=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    format: str

    @property
    def layout(self) -> str:
        return CONTIGUOUS if self.format in ("BF16", "FP32") else LAYOUT

    @property
    def mapped_host(self) -> bool:
        return self.name == "per_layer_token_embd.weight" or self.name.endswith(
            (".ffn_gate_exps.weight", ".ffn_up_exps.weight")
        )


def _tensor(name: str, shape: tuple[int, ...], format: str) -> TensorSpec:
    return TensorSpec(name, shape, format)


def _common_layer(layer: int) -> list[TensorSpec]:
    prefix = f"blk.{layer}."
    expert_format = "IQ2_XXS" if layer in IQ2_EXPERT_LAYERS else "IQ1_S"
    shared_gate_up_format = "Q6_K" if layer in Q6_SHARED_GATE_UP_LAYERS else "Q5_K"
    return [
        _tensor(prefix + "ffn_down_exps.weight", (512, 2560, 640), "IQ4_NL"),
        _tensor(prefix + "ffn_down_shexp.weight", (2560, 640), "Q8_0"),
        _tensor(prefix + "ffn_gate_exps.weight", (512, 640, 2560), expert_format),
        _tensor(prefix + "ffn_gate_inp.weight", (512, 2560), "FP32"),
        _tensor(prefix + "ffn_gate_inp_shexp.weight", (2560,), "FP32"),
        _tensor(prefix + "ffn_gate_shexp.weight", (640, 2560), shared_gate_up_format),
        _tensor(prefix + "ffn_up_exps.weight", (512, 640, 2560), expert_format),
        _tensor(prefix + "ffn_up_shexp.weight", (640, 2560), shared_gate_up_format),
        _tensor(prefix + "hc_attn_down.weight", (320, 10240), "Q8_0"),
        _tensor(prefix + "hc_attn_inject.weight", (4, 10240), "FP32"),
        _tensor(prefix + "hc_attn_norm.weight", (10240,), "FP32"),
        _tensor(prefix + "hc_attn_up.weight", (10240, 320), "Q8_0"),
        _tensor(prefix + "hc_ffn_down.weight", (320, 10240), "Q8_0"),
        _tensor(prefix + "hc_ffn_inject.weight", (4, 10240), "FP32"),
        _tensor(prefix + "hc_ffn_norm.weight", (10240,), "FP32"),
        _tensor(prefix + "hc_ffn_up.weight", (10240, 320), "Q8_0"),
    ]


def _gdn_layer(layer: int) -> list[TensorSpec]:
    prefix = f"blk.{layer}."
    input_format = "Q6_K" if layer in Q6_GDN_INPUT_LAYERS else "Q5_K"
    return [
        _tensor(prefix + "attn_gate.weight", (6144, 2560), input_format),
        _tensor(prefix + "attn_qkv.weight", (10240, 2560), input_format),
        _tensor(prefix + "ssm_a", (48,), "FP32"),
        _tensor(prefix + "ssm_alpha.weight", (48, 2560), "FP32"),
        _tensor(prefix + "ssm_beta.weight", (48, 2560), "FP32"),
        _tensor(prefix + "ssm_conv1d.weight", (10240, 4), "FP32"),
        _tensor(prefix + "ssm_dt.bias", (48,), "FP32"),
        _tensor(prefix + "ssm_norm.weight", (128,), "FP32"),
        _tensor(prefix + "ssm_out.weight", (2560, 6144), "Q6_K"),
    ]


def _qsa_layer(layer: int) -> list[TensorSpec]:
    prefix = f"blk.{layer}."
    return [
        _tensor(prefix + "attn_k.weight", (512, 2560), "Q5_K"),
        _tensor(prefix + "attn_k_norm.weight", (256,), "FP32"),
        _tensor(prefix + "attn_output.weight", (2560, 6144), "Q5_K"),
        _tensor(prefix + "attn_q.weight", (12288, 2560), "Q5_K"),
        _tensor(prefix + "attn_q_norm.weight", (256,), "FP32"),
        _tensor(prefix + "attn_v.weight", (512, 2560), "Q5_K"),
        _tensor(prefix + "indexer.k_norm.weight", (128,), "FP32"),
        _tensor(prefix + "indexer.k_proj.weight", (128, 2560), "BF16"),
        _tensor(prefix + "indexer.q_norm.weight", (128,), "FP32"),
        _tensor(prefix + "indexer.q_proj.weight", (512, 2560), "BF16"),
    ]


def _build() -> tuple[TensorSpec, ...]:
    specs = [
        _tensor("output.weight", (248320, 2560), "Q4_K"),
        _tensor("output_hc_down.weight", (320, 10240), "Q8_0"),
        _tensor("output_hc_norm.weight", (10240,), "FP32"),
        _tensor("output_hc_up.weight", (10240, 320), "Q8_0"),
        _tensor("per_layer_token_embd.weight", (320001536, 160), "IQ4_NL"),
        _tensor("token_embd.weight", (248320, 2560), "Q4_K"),
    ]
    for layer in TEXT_LAYERS:
        specs.extend(_common_layer(layer))
        specs.extend(_qsa_layer(layer) if layer in QSA_LAYERS else _gdn_layer(layer))
        if layer == 1:
            prefix = "blk.1."
            specs.extend(
                (
                    _tensor(prefix + "ple_conv1d.weight", (10240, 4), "FP32"),
                    _tensor(prefix + "ple_key.weight", (10240, 2560), "Q8_0"),
                    _tensor(prefix + "ple_norm_conv.weight", (10240,), "FP32"),
                    _tensor(prefix + "ple_norm_key.weight", (10240,), "FP32"),
                    _tensor(prefix + "ple_norm_query.weight", (10240,), "FP32"),
                    _tensor(prefix + "ple_value.weight", (2560, 2560), "Q8_0"),
                )
            )
    return tuple(specs)


TENSOR_SPECS = _build()
TENSORS_BY_NAME = {spec.name: spec for spec in TENSOR_SPECS}

if len(TENSOR_SPECS) != 1224 or len(TENSORS_BY_NAME) != 1224:
    raise RuntimeError("Qwen4 verifier inventory is not closed")


__all__ = [
    "CONTIGUOUS",
    "GDN_LAYERS",
    "IQ2_EXPERT_LAYERS",
    "LAYOUT",
    "MODEL_ID",
    "QSA_LAYERS",
    "Q6_GDN_INPUT_LAYERS",
    "Q6_SHARED_GATE_UP_LAYERS",
    "TENSORS_BY_NAME",
    "TENSOR_SPECS",
    "TEXT_LAYERS",
    "TensorSpec",
    "WEIGHTS_ID",
]
