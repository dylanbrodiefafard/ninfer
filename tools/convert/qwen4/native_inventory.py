"""Canonical exact native preview inventory; this is not a configurable model graph."""
from tools.artifact.container import ArtifactIdentity, TensorSpec

IDENTITY = ArtifactIdentity("qwen4/native-preview", "nvfp4-a16")
MAIN = "model.language_model."
FRONTEND_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                  "generation_config.json", "preprocessor_config.json", "video_preprocessor_config.json")
FP8_ROLES = frozenset(
    [f"{MAIN}layers.{layer}.mlp.shared_expert.{role}_proj.weight"
     for layer in (0, 3) for role in ("gate", "up", "down")]
    + [f"{MAIN}layers.0.linear_attn.{role}.weight" for role in ("in_proj_qkv", "in_proj_z", "out_proj")]
    + [f"{MAIN}layers.3.self_attn.{role}_proj.weight" for role in ("q", "k", "v", "o")])
LAYOUTS = {"BF16": "contiguous-le-v1", "FP32": "contiguous-le-v1",
           "NVFP4_EXPERT_F32M": "expert-blockscale-k16-m128x4-v1",
           "FP8_E4M3FN_TENSOR_F32M": "tensor-calibrated-v1",
           "NVFP4": "blockscale-k16-m128x4-v1",
           "NVFP4_PARTITION_F32M": "partitioned-row-blockscale-k16-v1",
           "FP8_E4M3FN_TENSOR_BF16S": "tensor-scale-v1"}


def tensor_specs(ple_format="NVFP4_PARTITION_F32M", dflash_format=None, fp8_roles=()):
    if ple_format not in ("NVFP4_PARTITION_F32M", "FP8_E4M3FN_TENSOR_BF16S"):
        raise ValueError("native PLE must use an actual NVFP4 or FP8 source codec")
    if dflash_format not in (None, "BF16", "NVFP4") or not set(fp8_roles) <= FP8_ROLES:
        raise ValueError("unqualified native projection profile")
    specs = []
    def add(name, shape, fmt="BF16"):
        if name in fp8_roles:
            if fmt != "BF16": raise ValueError("protected control cannot become FP8")
            fmt = "FP8_E4M3FN_TENSOR_F32M"
        specs.append(TensorSpec(name, tuple(shape), fmt, LAYOUTS[fmt]))
    def gr(p, inject=True):
        add(p+"hc_norm.weight", (10240,), "FP32")
        add(p+"input_mix_weight_down.weight", (320,10240))
        add(p+"input_mix_weight_up.weight", (10240,320))
        if inject: add(p+"block_inject_weight.weight", (4,10240), "FP32")
    def moe(p):
        add(p+"gate.weight", (512,2560)); add(p+"shared_expert_gate.weight", (1,2560))
        for role in ("gate", "up", "down"):
            shape = (2560,640) if role == "down" else (640,2560)
            add(p+f"shared_expert.{role}_proj.weight", shape)
            add(p+f"experts.{role}_proj.weight", (512,)+shape, "NVFP4_EXPERT_F32M")
    def qsa(p):
        add(p+"indexer.index_qk_proj.weight", (640,2560))
        for role in ("indexer.q_layernorm", "indexer.k_layernorm", "q_norm", "k_norm"):
            add(p+role+".weight", (128 if role.startswith("indexer") else 256,), "FP32")
        for role, shape in (("q",(12288,2560)), ("k",(512,2560)), ("v",(512,2560)), ("o",(2560,6144))):
            add(p+role+"_proj.weight", shape)
    for layer in range(48):
        p = MAIN+f"layers.{layer}."
        gr(p+"attn_hyper_connection."); gr(p+"mlp_hyper_connection."); moe(p+"mlp.")
        if layer % 4 == 3:
            qsa(p+"self_attn.")
        else:
            p += "linear_attn."
            for role, shape in (("in_proj_qkv.weight",(10240,2560)), ("in_proj_z.weight",(6144,2560)),
                                ("out_proj.weight",(2560,6144))): add(p+role,shape)
            for role, shape in (("in_proj_a.weight",(48,2560)), ("in_proj_b.weight",(48,2560)),
                                ("conv1d.weight",(10240,1,4)), ("ssm_a",(48,)), ("dt_bias",(48,)), ("norm.weight",(128,))):
                add(p+role,shape,"FP32")
    add(MAIN+"embed_tokens.weight",(248320,2560)); add("lm_head.weight",(248320,2560))
    gr(MAIN+"hyper_connection_mixer.",False)
    p=MAIN+"layers.1.ple."
    add(p+"key_proj.weight",(10240,2560)); add(p+"value_proj.weight",(2560,2560))
    for role in ("norm_key", "norm_query", "norm_conv"): add(p+role+".weight",(10240,))
    add(p+"conv1d.weight",(10240,1,4))
    add("ple.table",(128,2500012,160) if ple_format=="NVFP4_PARTITION_F32M" else (320001536,160),ple_format)
    p="model.visual."
    add(p+"patch_embed.proj.weight",(1152,3,2,16,16));add(p+"patch_embed.proj.bias",(1152,))
    add(p+"pos_embed.weight",(2304,1152))
    for layer in range(27):
        b=p+f"blocks.{layer}."
        for role in ("norm1.weight","norm1.bias","norm2.weight","norm2.bias"): add(b+role,(1152,))
        for role,n,k in (("attn.qkv",3456,1152),("attn.proj",1152,1152),("mlp.linear_fc1",4304,1152),("mlp.linear_fc2",1152,4304)):
            add(b+role+".weight",(n,k));add(b+role+".bias",(n,))
    add(p+"merger.norm.weight",(1152,));add(p+"merger.norm.bias",(1152,))
    add(p+"merger.linear_fc1.weight",(4608,4608));add(p+"merger.linear_fc1.bias",(4608,))
    add(p+"merger.linear_fc2.weight",(2560,4608));add(p+"merger.linear_fc2.bias",(2560,))
    add("mtp.fc_embedding.weight",(2560,2560));add("mtp.fc_hidden.weight",(2560,2560))
    add("mtp.pre_fc_norm_embedding.weight",(2560,));add("mtp.pre_fc_norm_hidden.weight",(10240,))
    gr("mtp.layers.0.attn_hyper_connection.");gr("mtp.layers.0.mlp_hyper_connection.")
    gr("mtp.hyper_connection_mixer.",False);qsa("mtp.layers.0.self_attn.");moe("mtp.layers.0.mlp.")
    if dflash_format:
        add("dflash.fc.weight",(2560,12800),dflash_format)
        add("dflash.hidden_norm.weight",(2560,));add("dflash.norm.weight",(2560,))
        for layer in range(5):
            p=f"dflash.layers.{layer}."
            for role in ("input_layernorm","post_attention_layernorm","self_attn.q_norm","self_attn.k_norm"):
                add(p+role+".weight",(256 if role.startswith("self_attn") else 2560,))
            for role,n,k in (("self_attn.q_proj",6144,2560),("self_attn.k_proj",512,2560),
                ("self_attn.v_proj",512,2560),("self_attn.o_proj",2560,6144),("mlp.gate_proj",7680,2560),
                ("mlp.up_proj",7680,2560),("mlp.down_proj",2560,7680)):
                add(p+role+".weight",(n,k),dflash_format)
    return specs
