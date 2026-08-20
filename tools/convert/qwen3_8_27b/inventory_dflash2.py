"""Append-only DFlash2 inventory for the Qwen3.8-27B NVFP4 artifact."""

from __future__ import annotations

from tools.convert.qwen3_6.common.inventory import BF16, Q4, W8, TensorSpec, tensor_spec


DFLASH2_LAYERS = tuple(range(5))
DFLASH2_HIDDEN = 5120
DFLASH2_INTERMEDIATE = 17408
DFLASH2_FEATURE_ROWS = 5 * DFLASH2_HIDDEN
DFLASH2_QUERY_SIZE = 32 * 128
DFLASH2_KV_SIZE = 8 * 128
DFLASH2_QKV_ROWS = DFLASH2_QUERY_SIZE + 2 * DFLASH2_KV_SIZE
DFLASH2_GATE_UP_ROWS = 2 * DFLASH2_INTERMEDIATE
DFLASH2_CONV_PROJ_ROWS = 1280
DFLASH2_SELECTOR_RANK = 256
DFLASH2_VOCAB = 248320
DFLASH2_MASK_TOKEN = 248070

DFLASH2_REPOSITORY = "z-lab/Qwen3.8-27B-DFlash2"
DFLASH2_REVISION = "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
DFLASH2_MODEL_PY_COMMIT = "95c8aeca5e4b4c4f9c0c967c05ab89fa3ed24f4c"

_FORMATS = {"w8": W8, "q4": Q4}


def dflash2_matrix_format(dflash_format: str) -> str:
    try:
        return _FORMATS[dflash_format]
    except KeyError as exc:
        raise ValueError("dflash format must be w8 or q4") from exc


def build_dflash2_specs(dflash_format: str = "w8") -> tuple[TensorSpec, ...]:
    matrix = dflash2_matrix_format(dflash_format)
    specs: list[TensorSpec] = [
        tensor_spec(
            "dflash/feature_projection",
            (DFLASH2_HIDDEN, DFLASH2_FEATURE_ROWS),
            matrix,
        ),
        tensor_spec("dflash/context_norm", (DFLASH2_HIDDEN,), BF16),
    ]
    for layer in DFLASH2_LAYERS:
        prefix = f"dflash/layers/{layer}/"
        specs.extend(
            (
                tensor_spec(prefix + "input_norm", (DFLASH2_HIDDEN,), BF16),
                tensor_spec(
                    prefix + "attention/query_key_value",
                    (DFLASH2_QKV_ROWS, DFLASH2_HIDDEN),
                    matrix,
                ),
                tensor_spec(prefix + "attention/query_norm", (128,), BF16),
                tensor_spec(prefix + "attention/key_norm", (128,), BF16),
                tensor_spec(
                    prefix + "attention/output",
                    (DFLASH2_HIDDEN, DFLASH2_QUERY_SIZE),
                    matrix,
                ),
                tensor_spec(
                    prefix + "attention_conv/base_kernel",
                    (DFLASH2_HIDDEN, 2, 2),
                    BF16,
                ),
                tensor_spec(
                    prefix + "attention_conv/kernel_projection",
                    (DFLASH2_CONV_PROJ_ROWS, DFLASH2_HIDDEN),
                    matrix,
                ),
                tensor_spec(prefix + "post_attention_norm", (DFLASH2_HIDDEN,), BF16),
                tensor_spec(
                    prefix + "mlp/gate_up",
                    (DFLASH2_GATE_UP_ROWS, DFLASH2_HIDDEN),
                    matrix,
                ),
                tensor_spec(
                    prefix + "mlp/down",
                    (DFLASH2_HIDDEN, DFLASH2_INTERMEDIATE),
                    matrix,
                ),
                tensor_spec(
                    prefix + "mlp_conv/base_kernel",
                    (DFLASH2_HIDDEN, 2, 2),
                    BF16,
                ),
                tensor_spec(
                    prefix + "mlp_conv/kernel_projection",
                    (DFLASH2_CONV_PROJ_ROWS, DFLASH2_HIDDEN),
                    matrix,
                ),
            )
        )
    specs.extend(
        (
            tensor_spec("dflash/final_norm", (DFLASH2_HIDDEN,), BF16),
            tensor_spec(
                "dflash/selector/hidden_projection",
                (DFLASH2_SELECTOR_RANK, DFLASH2_HIDDEN),
                matrix,
            ),
            tensor_spec(
                "dflash/selector/predecessor_codebook",
                (DFLASH2_SELECTOR_RANK, DFLASH2_VOCAB),
                BF16,
            ),
            tensor_spec(
                "dflash/selector/successor_codebook",
                (DFLASH2_SELECTOR_RANK, DFLASH2_VOCAB),
                BF16,
            ),
        )
    )
    return tuple(specs)
