"""The `qwen3.8-flash-next/f3-fp8-nvfp4` object inventory and its source mapping.

`docs/maintainer/qwen3.8-flash-next-artifact.md` is the authority for every name, shape, format,
row order, and source transform defined here. The dense projections, head, hyper-connection
down/up and shared expert are the dense-format study's quantized payloads (recipe `fallback_mid`
with the attention-input parents of `ATTN_PARENT.md` and the hyper-connection projections of
`HC.md`), ingested word for word: FP8 GDN projections, attention output, head and
hyper-connection down/up, an NVFP4 attention-input parent, and the shared expert as one NVFP4
routed-expert record per layer. Every other tensor is an exact copy of its BF16 checkpoint words.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from tools.artifact.container import ResourceSpec, TensorSpec

from .study_store import EXPERTS, HIDDEN, INTERMEDIATE, LAYERS, MATRICES

MODEL_ID = "qwen3.8-flash-next"
WEIGHTS_ID = "f3-fp8-nvfp4"

VOCAB = 248320
QSA_LAYERS = frozenset(range(3, LAYERS, 4))
PLE_LAYER = 1
HC_FLAT = 4 * HIDDEN
HC_RANK = 320
GDN_KEY = 2048
GDN_VALUE = 6144
GDN_HEADS = 48
GDN_CONV_CHANNELS = 2 * GDN_KEY + GDN_VALUE
QSA_HEADS = 24
QSA_HEAD_DIM = 256
QSA_KV = 512
QSA_QUERY = QSA_HEADS * QSA_HEAD_DIM
INDEX_QUERY = 512
INDEX_KEY = 128
MOE_CONTROL_ROWS = 544
NGRAM_HEADS = 16

FRONTEND_RESOURCES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)

BF16 = "BF16"
FP32 = "FP32"
I32 = "I32"
NVFP4 = "NVFP4"
FP8 = "FP8_E4M3FN_ROW_BF16S"
F3 = "F3G32_E4M3FN"
E2M1 = "E2M1G16_E4M3FN"
_LAYOUTS = {
    BF16: "contiguous-le-v1",
    FP32: "contiguous-le-v1",
    I32: "contiguous-le-v1",
    NVFP4: "blockscale-k16-m128x4-v1",
    FP8: "row-scale-v1",
    F3: "moe-expert-tiles-v1",
    E2M1: "moe-expert-tiles-v1",
}


class Family(enum.Enum):
    """Byte-accounting family of an object."""

    FRONTEND = "frontend"
    VOCABULARY = "vocabulary"
    HYPER_CONNECTION = "hyper_connection"
    PLE = "ple"
    GDN_INPUT = "gdn_input"
    GDN_OUTPUT = "gdn_output"
    GDN_CONTROL = "gdn_control"
    ATTENTION_INPUT = "attention_input"
    ATTENTION_OUTPUT = "attention_output"
    ATTENTION_CONTROL = "attention_control"
    MOE_CONTROL = "moe_control"
    SHARED_EXPERT = "shared_expert"
    ROUTED_EXPERTS = "routed_experts"
    ROUTED_EXPERT_TABLES = "routed_expert_tables"


class Transform(enum.Enum):
    """Closed set of source-to-object transforms (exact word moves unless quantized)."""

    ROWS = "rows"  # concatenate the parts' rows in order
    QSA_QUERY_KEY_GATE_VALUE = "qsa_query_key_gate_value"  # de-interleave q_proj per head
    INDEX_QUERY = "index_query"  # rows [0, 512) of index_qk_proj
    INDEX_KEY = "index_key"  # rows [512, 640) of index_qk_proj
    MOE_CONTROL = "moe_control"  # router rows, shared-expert gate row, zero rows
    CONV_TAP_MAJOR = "conv_tap_major"  # [C,1,4] -> [4,C]
    CONV_TAPS_FASTEST = "conv_taps_fastest"  # [C,1,4] -> [C,4]
    WIDEN_FP32 = "widen_fp32"  # exact BF16 -> FP32
    INT64_WORDS = "int64_words"  # int64 [n] -> I32 [n,2] (low word, high word)
    RESOURCE = "resource"
    ROUTED_EXPERTS = "routed_experts"  # study store records, tables
    STUDY_FP8_ROWS = "study_fp8_rows"  # dense-study FP8 records, rows concatenated in order
    STUDY_ATTENTION_PARENT = "study_attention_parent"  # dense-study NVFP4 attention-input parent
    STUDY_SHARED_EXPERT = "study_shared_expert"  # dense-study NVFP4 gate, up, down -> one record
    STUDY_SHARED_SCALES = "study_shared_scales"  # their FP32 matrix multipliers


STUDY_TRANSFORMS = frozenset(
    (
        Transform.STUDY_FP8_ROWS,
        Transform.STUDY_ATTENTION_PARENT,
        Transform.STUDY_SHARED_EXPERT,
        Transform.STUDY_SHARED_SCALES,
    )
)


@dataclass(frozen=True, slots=True)
class ObjectPlan:
    """One stored object: its spec, its source names, and how they become its words.

    ``sources`` are checkpoint tensor names, frontend file names, or dense-study record paths
    relative to the record directory. A study object also names the BF16 checkpoint tensors it
    quantizes (``reference``, combined by ``reference_transform``) for the validator's error
    report.
    """

    spec: TensorSpec | ResourceSpec
    family: Family
    transform: Transform
    sources: tuple[str, ...]
    reference: tuple[str, ...] = ()
    reference_transform: Transform | None = None


def _tensor(
    name: str,
    shape: tuple[int, ...],
    numeric_format: str,
    family: Family,
    transform: Transform,
    *sources: str,
) -> ObjectPlan:
    return ObjectPlan(
        TensorSpec(name, shape, numeric_format, _LAYOUTS[numeric_format]),
        family,
        transform,
        sources,
    )


# Dense-study methods of the recipe (study recipe `fallback_mid`).
STUDY_FP8_METHOD = "fp8_gptq"
STUDY_NVFP4_METHOD = "nv_gptq_mse"
STUDY_RECIPE = "fallback_mid"


def study_record(layer: int | str, source_key: str, method: str) -> str:
    """A dense-study record path relative to the study's `quant/dense` directory."""

    return f"{layer}/{source_key}/{method}.npz"


def study_attention_parent(layer: int) -> str:
    """The study's NVFP4 attention-input parent record of one QSA layer (`ATTN_PARENT.md`)."""

    return f"attn_parent/{layer}.npz"


def _projection(
    name: str,
    shape: tuple[int, int],
    family: Family,
    transform: Transform,
    layer: int,
    checkpoint_prefix: str,
    keys: tuple[str, ...],
) -> ObjectPlan:
    """A dense projection parent from its study payload.

    ``transform`` combines the checkpoint tensors ``keys`` into the parent's row order; it is the
    validator's reference for the quantization error report.
    """

    if family is Family.ATTENTION_INPUT:
        numeric_format, study_transform = NVFP4, Transform.STUDY_ATTENTION_PARENT
        records = (study_attention_parent(layer),)
    else:
        numeric_format, study_transform = FP8, Transform.STUDY_FP8_ROWS
        records = tuple(study_record(layer, key, STUDY_FP8_METHOD) for key in keys)
    return ObjectPlan(
        TensorSpec(name, shape, numeric_format, _LAYOUTS[numeric_format]),
        family,
        study_transform,
        records,
        tuple(checkpoint_prefix + key for key in keys),
        transform,
    )


def _hyper_connection(prefix: str, source: str, records: str, *, write: bool) -> list[ObjectPlan]:
    """One hyper-connection read: BF16 norm and write, FP8 study records for down and up.

    ``records`` is the dense-study record prefix of the down/up projections (`hc/{layer}/{module}.`
    or `hc/final/hyper_connection_mixer.`).
    """

    def projection(role: str, shape: tuple[int, int], key: str) -> ObjectPlan:
        return ObjectPlan(
            TensorSpec(prefix + "/" + role, shape, FP8, _LAYOUTS[FP8]),
            Family.HYPER_CONNECTION,
            Transform.STUDY_FP8_ROWS,
            (records + key + "/" + STUDY_FP8_METHOD + ".npz",),
            (source + key,),
            Transform.ROWS,
        )

    plans = [
        _tensor(
            prefix + "/norm",
            (HC_FLAT,),
            BF16,
            Family.HYPER_CONNECTION,
            Transform.ROWS,
            source + "hc_norm.weight",
        ),
        projection("down", (HC_RANK, HC_FLAT), "input_mix_weight_down.weight"),
        projection("up", (HC_FLAT, HC_RANK), "input_mix_weight_up.weight"),
    ]
    if write:
        plans.append(
            _tensor(
                prefix + "/write",
                (4, HC_FLAT),
                BF16,
                Family.HYPER_CONNECTION,
                Transform.ROWS,
                source + "block_inject_weight.weight",
            )
        )
    return plans


def _ple(prefix: str, source: str) -> list[ObjectPlan]:
    embedding = source + "ple_embedding."
    return [
        _tensor(
            prefix + "/key_projection",
            (HC_FLAT, HIDDEN),
            BF16,
            Family.PLE,
            Transform.ROWS,
            source + "key_proj.weight",
        ),
        _tensor(
            prefix + "/value_projection",
            (HIDDEN, HIDDEN),
            BF16,
            Family.PLE,
            Transform.ROWS,
            source + "value_proj.weight",
        ),
        _tensor(
            prefix + "/key_norm",
            (HC_FLAT,),
            BF16,
            Family.PLE,
            Transform.ROWS,
            source + "norm_key.weight",
        ),
        _tensor(
            prefix + "/query_norm",
            (HC_FLAT,),
            BF16,
            Family.PLE,
            Transform.ROWS,
            source + "norm_query.weight",
        ),
        _tensor(
            prefix + "/conv_norm",
            (HC_FLAT,),
            BF16,
            Family.PLE,
            Transform.ROWS,
            source + "norm_conv.weight",
        ),
        _tensor(
            prefix + "/convolution",
            (HC_FLAT, 4),
            BF16,
            Family.PLE,
            Transform.CONV_TAPS_FASTEST,
            source + "conv1d.weight",
        ),
        _tensor(
            prefix + "/ngram_hash_multipliers",
            (3, 2),
            I32,
            Family.PLE,
            Transform.INT64_WORDS,
            embedding + "layer_multipliers",
        ),
        _tensor(
            prefix + "/ngram_head_offsets",
            (NGRAM_HEADS, 2),
            I32,
            Family.PLE,
            Transform.INT64_WORDS,
            embedding + "ngram_heads_offsets",
        ),
        _tensor(
            prefix + "/ngram_head_sizes",
            (NGRAM_HEADS, 2),
            I32,
            Family.PLE,
            Transform.INT64_WORDS,
            embedding + "ngram_heads_vocab_sizes",
        ),
    ]


def _gdn(prefix: str, source: str, layer: int) -> list[ObjectPlan]:
    checkpoint = source
    source += "linear_attn."
    return [
        _projection(
            prefix + "/query_key_value_z",
            (2 * GDN_KEY + 2 * GDN_VALUE, HIDDEN),
            Family.GDN_INPUT,
            Transform.ROWS,
            layer,
            checkpoint,
            ("linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight"),
        ),
        _tensor(
            prefix + "/convolution",
            (4, GDN_CONV_CHANNELS),
            BF16,
            Family.GDN_CONTROL,
            Transform.CONV_TAP_MAJOR,
            source + "conv1d.weight",
        ),
        _tensor(
            prefix + "/a_b_projection",
            (2 * GDN_HEADS, HIDDEN),
            BF16,
            Family.GDN_CONTROL,
            Transform.ROWS,
            source + "in_proj_a.weight",
            source + "in_proj_b.weight",
        ),
        _tensor(
            prefix + "/a_log",
            (GDN_HEADS,),
            FP32,
            Family.GDN_CONTROL,
            Transform.WIDEN_FP32,
            source + "A_log",
        ),
        _tensor(
            prefix + "/dt_bias",
            (GDN_HEADS,),
            FP32,
            Family.GDN_CONTROL,
            Transform.WIDEN_FP32,
            source + "dt_bias",
        ),
        _tensor(
            prefix + "/norm",
            (128,),
            BF16,
            Family.GDN_CONTROL,
            Transform.ROWS,
            source + "norm.weight",
        ),
        _projection(
            prefix + "/output",
            (HIDDEN, GDN_VALUE),
            Family.GDN_OUTPUT,
            Transform.ROWS,
            layer,
            checkpoint,
            ("linear_attn.out_proj.weight",),
        ),
    ]


def _attention(prefix: str, source: str, layer: int) -> list[ObjectPlan]:
    checkpoint = source
    source += "self_attn."
    control = Family.ATTENTION_CONTROL
    return [
        _projection(
            prefix + "/query_key_gate_value",
            (2 * QSA_QUERY + 2 * QSA_KV, HIDDEN),
            Family.ATTENTION_INPUT,
            Transform.QSA_QUERY_KEY_GATE_VALUE,
            layer,
            checkpoint,
            ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"),
        ),
        _tensor(
            prefix + "/query_norm",
            (QSA_HEAD_DIM,),
            BF16,
            control,
            Transform.ROWS,
            source + "q_norm.weight",
        ),
        _tensor(
            prefix + "/key_norm",
            (QSA_HEAD_DIM,),
            BF16,
            control,
            Transform.ROWS,
            source + "k_norm.weight",
        ),
        _tensor(
            prefix + "/index_query",
            (INDEX_QUERY, HIDDEN),
            BF16,
            control,
            Transform.INDEX_QUERY,
            source + "indexer.index_qk_proj.weight",
        ),
        _tensor(
            prefix + "/index_key",
            (INDEX_KEY, HIDDEN),
            BF16,
            control,
            Transform.INDEX_KEY,
            source + "indexer.index_qk_proj.weight",
        ),
        _tensor(
            prefix + "/index_query_norm",
            (INDEX_KEY,),
            BF16,
            control,
            Transform.ROWS,
            source + "indexer.q_layernorm.weight",
        ),
        _tensor(
            prefix + "/index_key_norm",
            (INDEX_KEY,),
            BF16,
            control,
            Transform.ROWS,
            source + "indexer.k_layernorm.weight",
        ),
        _projection(
            prefix + "/output",
            (HIDDEN, QSA_QUERY),
            Family.ATTENTION_OUTPUT,
            Transform.ROWS,
            layer,
            checkpoint,
            ("self_attn.o_proj.weight",),
        ),
    ]


SHARED_KEYS = tuple(f"mlp.shared_expert.{matrix}_proj.weight" for matrix in ("gate", "up", "down"))


def _moe(prefix: str, source: str, layer: int) -> list[ObjectPlan]:
    checkpoint = source
    source += "mlp."
    control = _tensor(
        prefix + "/control",
        (MOE_CONTROL_ROWS, HIDDEN),
        BF16,
        Family.MOE_CONTROL,
        Transform.MOE_CONTROL,
        source + "gate.weight",
        source + "shared_expert_gate.weight",
    )
    record = TensorSpec(prefix + "/shared_expert", (1, HIDDEN, INTERMEDIATE), E2M1, _LAYOUTS[E2M1])
    scales = TensorSpec(prefix + "/shared_matrix_scales", (len(MATRICES),), FP32, _LAYOUTS[FP32])
    checkpoint_keys = tuple(checkpoint + key for key in SHARED_KEYS)
    records = tuple(study_record(layer, key, STUDY_NVFP4_METHOD) for key in SHARED_KEYS)
    return [
        control,
        ObjectPlan(
            record, Family.SHARED_EXPERT, Transform.STUDY_SHARED_EXPERT, records, checkpoint_keys
        ),
        ObjectPlan(scales, Family.SHARED_EXPERT, Transform.STUDY_SHARED_SCALES, records),
    ]


ROUTED_PREFIX = "text/routed_experts"
E2M1_RECORDS = ROUTED_PREFIX + "/e2m1_records"
F3_RECORDS = ROUTED_PREFIX + "/f3_records"
EXPERT_CODECS = ROUTED_PREFIX + "/codecs"
F3_CODEBOOKS = ROUTED_PREFIX + "/f3_codebooks"
MATRIX_SCALES = ROUTED_PREFIX + "/matrix_scales"


def build_inventory(
    nvfp4_experts: int, f3_experts: int, frontend_bytes: dict[str, int]
) -> tuple[ObjectPlan, ...]:
    """Return the complete ordered inventory for the store's expert counts."""

    if nvfp4_experts <= 0 or f3_experts <= 0 or nvfp4_experts + f3_experts != LAYERS * EXPERTS:
        raise ValueError("routed expert counts must be positive and cover 48 x 512 experts")
    plans: list[ObjectPlan] = [
        ObjectPlan(
            ResourceSpec("frontend/" + name, "raw-bytes-v1", frontend_bytes[name]),
            Family.FRONTEND,
            Transform.RESOURCE,
            (name,),
        )
        for name in FRONTEND_RESOURCES
    ]
    text = "model.language_model."
    plans.append(
        _tensor(
            "text/token_embedding",
            (VOCAB, HIDDEN),
            BF16,
            Family.VOCABULARY,
            Transform.ROWS,
            text + "embed_tokens.weight",
        )
    )
    for layer in range(LAYERS):
        prefix = f"text/layers/{layer}"
        source = f"{text}layers.{layer}."
        if layer == PLE_LAYER:
            plans.extend(_ple(prefix + "/ple", source + "ple."))
        plans.extend(
            _hyper_connection(
                prefix + "/mixer_hyper_connection",
                source + "attn_hyper_connection.",
                f"hc/{layer}/attn_hyper_connection.",
                write=True,
            )
        )
        if layer in QSA_LAYERS:
            plans.extend(_attention(prefix + "/attention", source, layer))
        else:
            plans.extend(_gdn(prefix + "/gdn", source, layer))
        plans.extend(
            _hyper_connection(
                prefix + "/moe_hyper_connection",
                source + "mlp_hyper_connection.",
                f"hc/{layer}/mlp_hyper_connection.",
                write=True,
            )
        )
        plans.extend(_moe(prefix + "/moe", source, layer))
    plans.extend(
        _hyper_connection(
            "text/final_hyper_connection",
            text + "hyper_connection_mixer.",
            "hc/final/hyper_connection_mixer.",
            write=False,
        )
    )
    plans.append(
        ObjectPlan(
            TensorSpec("text/output_head", (VOCAB, HIDDEN), FP8, _LAYOUTS[FP8]),
            Family.VOCABULARY,
            Transform.STUDY_FP8_ROWS,
            (study_record("head", "lm_head", STUDY_FP8_METHOD),),
            ("lm_head.weight",),
            Transform.ROWS,
        )
    )
    experts = Family.ROUTED_EXPERTS
    tables = Family.ROUTED_EXPERT_TABLES
    plans.extend(
        (
            _tensor(
                E2M1_RECORDS,
                (nvfp4_experts, HIDDEN, INTERMEDIATE),
                E2M1,
                experts,
                Transform.ROUTED_EXPERTS,
            ),
            _tensor(
                F3_RECORDS,
                (f3_experts, HIDDEN, INTERMEDIATE),
                F3,
                experts,
                Transform.ROUTED_EXPERTS,
            ),
            _tensor(EXPERT_CODECS, (LAYERS, EXPERTS), I32, tables, Transform.ROUTED_EXPERTS),
            _tensor(F3_CODEBOOKS, (LAYERS, len(MATRICES)), I32, tables, Transform.ROUTED_EXPERTS),
            _tensor(
                MATRIX_SCALES,
                (LAYERS, EXPERTS, len(MATRICES)),
                FP32,
                tables,
                Transform.ROUTED_EXPERTS,
            ),
        )
    )
    return tuple(plans)
