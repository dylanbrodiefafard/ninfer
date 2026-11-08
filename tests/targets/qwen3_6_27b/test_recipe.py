import torch

from tools.convert.qwen3_6.common.recipe import (
    Concat,
    Reshape,
    Slice,
    SourceTensor,
    Transpose,
    expression_shape,
    materialize_expression,
    materialize_recipe,
    source_requirements,
)
from tools.convert.qwen3_6_27b import recipe


class TensorReader:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def get(self, name: str) -> torch.Tensor:
        return self.tensors[name]


def test_recipe_roles_and_source_requirements() -> None:
    assert len(recipe.RECIPE_SPECS) == 1118
    assert len(recipe.RECIPES_BY_NAME) == 1118
    assert expression_shape(recipe.RECIPES_BY_NAME["text/token_embedding"].expression) == (
        248320,
        5120,
    )
    assert expression_shape(
        recipe.RECIPES_BY_NAME["text/layers/3/attention/query_key"].expression
    ) == (7168, 5120)
    assert expression_shape(
        recipe.RECIPES_BY_NAME["text/layers/3/attention/gate_value"].expression
    ) == (7168, 5120)
    assert expression_shape(recipe.RECIPES_BY_NAME["mtp/layer/mlp/down"].expression) == (
        5120,
        17408,
    )
    assert expression_shape(recipe.RECIPES_BY_NAME["vision/merger/fc2"].expression) == (
        5120,
        4608,
    )

    requirements = source_requirements(recipe.RECIPE_SPECS)
    assert len(requirements) == 1199
    assert {requirement.dtype for requirement in requirements.values()} == {"BF16"}
    assert requirements["model.language_model.embed_tokens.weight"].shape == (248320, 5120)
    assert requirements["mtp.layers.0.self_attn.q_proj.weight"].shape == (12288, 5120)
    assert requirements["model.visual.patch_embed.proj.weight"].shape == (
        1152,
        3,
        2,
        16,
        16,
    )


def test_fused_mtp_attention_recipe_materializes_runtime_row_order() -> None:
    q_name = "mtp.layers.0.self_attn.q_proj.weight"
    k_name = "mtp.layers.0.self_attn.k_proj.weight"
    v_name = "mtp.layers.0.self_attn.v_proj.weight"

    # The transform is dtype-independent. Byte markers keep this real-shape fixture compact.
    q_source = (
        torch.arange(12288, dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .view(-1, 1)
        .expand(-1, 5120)
    )
    k_source = torch.full((1024, 1), 251, dtype=torch.uint8).expand(-1, 5120)
    v_source = torch.full((1024, 1), 252, dtype=torch.uint8).expand(-1, 5120)
    fused = materialize_recipe(
        recipe.RECIPES_BY_NAME["mtp/layer/attention/query_key_gate_value"],
        TensorReader({q_name: q_source, k_name: k_source, v_name: v_source}),
    )

    query_rows = (
        torch.cat([torch.arange(head * 512, head * 512 + 256) for head in range(24)])
        .remainder(251)
        .to(torch.uint8)
    )
    gate_rows = (
        torch.cat([torch.arange(head * 512 + 256, head * 512 + 512) for head in range(24)])
        .remainder(251)
        .to(torch.uint8)
    )

    assert fused.shape == (14336, 5120)
    assert torch.equal(fused[:6144, 0], query_rows)
    assert torch.equal(fused[:6144, -1], query_rows)
    assert torch.all(fused[6144:7168, 0] == 251)
    assert torch.equal(fused[7168:13312, 0], gate_rows)
    assert torch.equal(fused[7168:13312, -1], gate_rows)
    assert torch.all(fused[13312:, 0] == 252)


def test_fused_gdn_value_z_recipe_materializes_runtime_row_order() -> None:
    qkv_name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    z_name = "model.language_model.layers.0.linear_attn.in_proj_z.weight"
    qkv = (
        torch.arange(10240, dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .view(-1, 1)
        .expand(-1, 5120)
    )
    z = (
        torch.arange(6144, dtype=torch.int64)
        .add(37)
        .remainder(251)
        .to(torch.uint8)
        .view(-1, 1)
        .expand(-1, 5120)
    )
    fused = materialize_recipe(
        recipe.RECIPES_BY_NAME["text/layers/0/gdn/value_z"],
        TensorReader({qkv_name: qkv, z_name: z}),
    )

    assert fused.shape == (12288, 5120)
    assert torch.equal(fused[:6144, 0], qkv[4096:, 0])
    assert torch.equal(fused[:6144, -1], qkv[4096:, -1])
    assert torch.equal(fused[6144:, 0], z[:, 0])
    assert torch.equal(fused[6144:, -1], z[:, -1])


def test_representative_transforms_materialize_without_full_weights() -> None:
    convolution_source = torch.arange(10240 * 4, dtype=torch.float32).to(torch.bfloat16)
    convolution_source = convolution_source.reshape(10240, 1, 4)
    convolution_name = "model.language_model.layers.0.linear_attn.conv1d.weight"
    convolution = materialize_recipe(
        recipe.RECIPES_BY_NAME["text/layers/0/gdn/convolution"],
        TensorReader({convolution_name: convolution_source}),
    )
    assert convolution.shape == (4, 10240)
    assert torch.equal(convolution, convolution_source[:, 0, :].transpose(0, 1).contiguous())

    a_log_name = "model.language_model.layers.0.linear_attn.A_log"
    a_log_source = torch.linspace(-2, 2, 48, dtype=torch.bfloat16)
    a_log = materialize_recipe(
        recipe.RECIPES_BY_NAME["text/layers/0/gdn/a_log"],
        TensorReader({a_log_name: a_log_source}),
    )
    assert a_log.dtype == torch.float32
    assert torch.equal(a_log, a_log_source.float())

    small_source = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    expression = Concat(
        (
            Reshape(Slice(SourceTensor("small", (2, 3, 4)), 1, 0, 2), (4, 4)),
            Transpose(
                Reshape(
                    Slice(SourceTensor("small", (2, 3, 4)), 1, 2, 3),
                    (2, 4),
                ),
                (0, 1),
            ),
        ),
        0,
    )
    materialized = materialize_expression(expression, TensorReader({"small": small_source}))
    assert materialized.shape == (6, 4)
    assert torch.equal(materialized[:4], small_source[:, :2, :].reshape(4, 4))

    token_ids = torch.arange(131072, dtype=torch.int64)
    materialized_ids = materialize_recipe(
        recipe.RECIPES_BY_NAME["text/draft_head_token_ids"],
        TensorReader({}),
        {"text/draft_head_token_ids": token_ids},
    )
    assert materialized_ids.dtype == torch.int32
    assert materialized_ids.shape == (131072,)
