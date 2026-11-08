"""BF16 Hugging Face source recipes for the 27B Qwen3.6-family tensor roles.

Each recipe maps the BF16 source checkpoint onto one named tensor role of the shared 27B geometry:
Text, the optimized draft head, MTP, and Vision. Text input projections are recipes for their
source row groups: ``attention/query_key`` and ``attention/gate_value`` for full attention, and
``gdn/query_key`` and ``gdn/value_z`` for GDN. A converter that stores a fused parent such as
``attention/query_key_gate_value`` concatenates its row groups in that order. The MTP attention
parent is already the fused ``[query, key, output_gate, value]`` row order.
"""

from __future__ import annotations

from tools.convert.qwen3_6.common.inventory import FP32
from tools.convert.qwen3_6.common.recipe import (
    Cast,
    Concat,
    DraftHeadTokenIds,
    Expression,
    GatherRows,
    Reshape,
    Slice,
    TensorRecipe,
    Transpose,
    attention_qproj_part,
    build_vision_recipes,
    source,
)

from .convert import FULL_ATTENTION_LAYERS

DRAFT_ROWS = 131072

_source = source


def _attention_qproj_part(source_name: str, gate: bool) -> Expression:
    return attention_qproj_part(
        source_name,
        gate,
        num_heads=24,
        hidden_size=5120,
    )


def _build_text_recipes() -> tuple[TensorRecipe, ...]:
    recipes: list[TensorRecipe] = [
        TensorRecipe(
            "text/token_embedding",
            _source("model.language_model.embed_tokens.weight", (248320, 5120)),
        )
    ]

    for layer in range(64):
        source_prefix = f"model.language_model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"
        recipes.append(
            TensorRecipe(
                object_prefix + "input_norm",
                _source(source_prefix + "input_layernorm.weight", (5120,)),
            )
        )

        if layer in FULL_ATTENTION_LAYERS:
            q_proj = source_prefix + "self_attn.q_proj.weight"
            query = _attention_qproj_part(q_proj, gate=False)
            gate = _attention_qproj_part(q_proj, gate=True)
            recipes.extend(
                (
                    TensorRecipe(
                        object_prefix + "attention/query_key",
                        Concat(
                            (
                                query,
                                _source(source_prefix + "self_attn.k_proj.weight", (1024, 5120)),
                            ),
                            0,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/gate_value",
                        Concat(
                            (
                                gate,
                                _source(source_prefix + "self_attn.v_proj.weight", (1024, 5120)),
                            ),
                            0,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/query_norm",
                        _source(source_prefix + "self_attn.q_norm.weight", (256,)),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/key_norm",
                        _source(source_prefix + "self_attn.k_norm.weight", (256,)),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/output",
                        _source(source_prefix + "self_attn.o_proj.weight", (5120, 6144)),
                    ),
                )
            )
        else:
            qkv_source = _source(
                source_prefix + "linear_attn.in_proj_qkv.weight",
                (10240, 5120),
            )
            convolution = _source(
                source_prefix + "linear_attn.conv1d.weight",
                (10240, 1, 4),
            )
            recipes.extend(
                (
                    TensorRecipe(
                        object_prefix + "gdn/a_log",
                        Cast(_source(source_prefix + "linear_attn.A_log", (48,)), FP32),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/dt_bias",
                        Cast(_source(source_prefix + "linear_attn.dt_bias", (48,)), FP32),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/convolution",
                        Transpose(
                            Reshape(Slice(convolution, 1, 0, 1), (10240, 4)),
                            (1, 0),
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/a_projection",
                        _source(source_prefix + "linear_attn.in_proj_a.weight", (48, 5120)),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/b_projection",
                        _source(source_prefix + "linear_attn.in_proj_b.weight", (48, 5120)),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/query_key",
                        Slice(qkv_source, 0, 0, 4096),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/value_z",
                        Concat(
                            (
                                Slice(qkv_source, 0, 4096, 10240),
                                _source(
                                    source_prefix + "linear_attn.in_proj_z.weight",
                                    (6144, 5120),
                                ),
                            ),
                            0,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/norm",
                        _source(source_prefix + "linear_attn.norm.weight", (128,)),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/output",
                        _source(source_prefix + "linear_attn.out_proj.weight", (5120, 6144)),
                    ),
                )
            )

        recipes.extend(
            (
                TensorRecipe(
                    object_prefix + "post_attention_norm",
                    _source(source_prefix + "post_attention_layernorm.weight", (5120,)),
                ),
                TensorRecipe(
                    object_prefix + "mlp/gate_up",
                    Concat(
                        (
                            _source(source_prefix + "mlp.gate_proj.weight", (17408, 5120)),
                            _source(source_prefix + "mlp.up_proj.weight", (17408, 5120)),
                        ),
                        0,
                    ),
                ),
                TensorRecipe(
                    object_prefix + "mlp/down",
                    _source(source_prefix + "mlp.down_proj.weight", (5120, 17408)),
                ),
            )
        )

    recipes.extend(
        (
            TensorRecipe(
                "text/final_norm",
                _source("model.language_model.norm.weight", (5120,)),
            ),
            TensorRecipe(
                "text/output_head",
                _source("lm_head.weight", (248320, 5120)),
            ),
        )
    )
    return tuple(recipes)


def _build_draft_head_recipes() -> tuple[TensorRecipe, ...]:
    return (
        TensorRecipe(
            "text/draft_head",
            GatherRows(
                _source("lm_head.weight", (248320, 5120)),
                token_ids_object="text/draft_head_token_ids",
                rows=DRAFT_ROWS,
            ),
        ),
        TensorRecipe(
            "text/draft_head_token_ids",
            DraftHeadTokenIds(
                ranking_path="tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64",
                tokenizer_resource="frontend/tokenizer_config.json",
                vocab_rows=248320,
                tokenizer_id_count=248077,
                rows=DRAFT_ROWS,
            ),
        ),
    )


def _build_mtp_recipes() -> tuple[TensorRecipe, ...]:
    source_prefix = "mtp.layers.0."
    q_proj = source_prefix + "self_attn.q_proj.weight"
    return (
        TensorRecipe("mtp/input_projection", _source("mtp.fc.weight", (5120, 10240))),
        TensorRecipe(
            "mtp/embedding_norm",
            _source("mtp.pre_fc_norm_embedding.weight", (5120,)),
        ),
        TensorRecipe(
            "mtp/hidden_norm",
            _source("mtp.pre_fc_norm_hidden.weight", (5120,)),
        ),
        TensorRecipe(
            "mtp/layer/input_norm",
            _source(source_prefix + "input_layernorm.weight", (5120,)),
        ),
        TensorRecipe(
            "mtp/layer/attention/query_key_gate_value",
            Concat(
                (
                    _attention_qproj_part(q_proj, gate=False),
                    _source(source_prefix + "self_attn.k_proj.weight", (1024, 5120)),
                    _attention_qproj_part(q_proj, gate=True),
                    _source(source_prefix + "self_attn.v_proj.weight", (1024, 5120)),
                ),
                0,
            ),
        ),
        TensorRecipe(
            "mtp/layer/attention/query_norm",
            _source(source_prefix + "self_attn.q_norm.weight", (256,)),
        ),
        TensorRecipe(
            "mtp/layer/attention/key_norm",
            _source(source_prefix + "self_attn.k_norm.weight", (256,)),
        ),
        TensorRecipe(
            "mtp/layer/attention/output",
            _source(source_prefix + "self_attn.o_proj.weight", (5120, 6144)),
        ),
        TensorRecipe(
            "mtp/layer/post_attention_norm",
            _source(source_prefix + "post_attention_layernorm.weight", (5120,)),
        ),
        TensorRecipe(
            "mtp/layer/mlp/gate_up",
            Concat(
                (
                    _source(source_prefix + "mlp.gate_proj.weight", (17408, 5120)),
                    _source(source_prefix + "mlp.up_proj.weight", (17408, 5120)),
                ),
                0,
            ),
        ),
        TensorRecipe(
            "mtp/layer/mlp/down",
            _source(source_prefix + "mlp.down_proj.weight", (5120, 17408)),
        ),
        TensorRecipe("mtp/final_norm", _source("mtp.norm.weight", (5120,))),
    )


def _build_vision_recipes() -> tuple[TensorRecipe, ...]:
    return build_vision_recipes(5120)


RECIPE_SPECS = (
    _build_text_recipes()
    + _build_draft_head_recipes()
    + _build_mtp_recipes()
    + _build_vision_recipes()
)
RECIPES_BY_NAME = {recipe.object_name: recipe for recipe in RECIPE_SPECS}
if len(RECIPES_BY_NAME) != len(RECIPE_SPECS):
    raise RuntimeError("more than one 27B source recipe targets the same tensor role")
