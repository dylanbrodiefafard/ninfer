"""Canonical exact native preview inventory; this is not a configurable model graph."""
from tools.artifact.container import ArtifactIdentity, TensorSpec

IDENTITY = ArtifactIdentity("qwen4/native-preview", "nvfp4-native")
MAIN = "model.language_model."
NVFP4_SHARED_UP = MAIN+"layers.0.mlp.shared_expert.up_proj.weight"
FRONTEND_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                  "generation_config.json", "preprocessor_config.json", "video_preprocessor_config.json")
FP8_ORIGINAL_ROLES = frozenset(
    [f"{MAIN}layers.{layer}.mlp.shared_expert.{role}_proj.weight"
     for layer in (0, 3) for role in ("gate", "up", "down")]
    + [f"{MAIN}layers.0.linear_attn.{role}.weight" for role in ("in_proj_qkv", "in_proj_z", "out_proj")]
    + [f"{MAIN}layers.3.self_attn.{role}_proj.weight" for role in ("q", "k", "v", "o")])
FP8_ADDITIONAL_ROLES = frozenset(
    [f"{MAIN}layers.{layer}.linear_attn.in_proj_z.weight" for layer in (1, 2)] +
    [f"{MAIN}layers.{layer}.mlp.shared_expert.{role}_proj.weight"
     for layer in (1, 2) for role in ("gate", "up", "down")])
# Immutable source inventories are not product admission. The additional source bundle
# includes unqualified candidates; only independently qualified selected roles may be emitted.
FP8_SOURCE_BUNDLES = {"original13": FP8_ORIGINAL_ROLES, "additional8": FP8_ADDITIONAL_ROLES}
FP8_SHARED_MASKS = ({0,1,2,3,4,6,7}, {0,1,2,4,5}, {0,1,2,3,4,5}, {0,1,2,3})
FP8_ROLES = frozenset(
    [f"{MAIN}layers.{layer}.linear_attn.in_proj_z.weight" for layer in (0,1)] +
    [f"{MAIN}layers.{layer}.mlp.shared_expert.{role}_proj.weight"
     for layer in range(4) for bit,role in ((1,"gate"),(2,"up"),(4,"down"))
     if bit in FP8_SHARED_MASKS[layer]])
PREFILL_POLICIES = {"a16": 0, "selective-a8": 1, "routed-a4": 2, "routed-a4-selective-a8": 3}
A8_ROLES = frozenset(
    [f"{MAIN}layers.0.linear_attn.in_proj_z.weight"] +
    [f"{MAIN}layers.{layer}.mlp.shared_expert.{role}_proj.weight"
     for layer in range(4) for role in (("gate","up") if layer==0 else ("gate",))])
A8_ROW_ROLES = frozenset([MAIN+"layers.2.linear_attn.in_proj_z.weight"])
# Storage roles are independent of activation policy; only row-Z2 participates in
# the explicitly selected complete A8 prefill recipe above.
ROW_FP8_ROLES = frozenset(
    [MAIN+"embed_tokens.weight", "lm_head.weight", MAIN+"layers.2.linear_attn.in_proj_z.weight"] +
    [MAIN+"hyper_connection_mixer.input_mix_weight_"+role+".weight" for role in ("down", "up")] +
    [MAIN+"layers.1.ple."+role+"_proj.weight" for role in ("key", "value")])


def validate_fp8_selection(fp8_roles):
    roles=set(fp8_roles)
    if not roles<=FP8_ROLES:
        raise ValueError("unqualified native tensor-FP8 role")
    masks=[]
    for layer in range(4):
        mask=sum(bit for bit,role in ((1,"gate"),(2,"up"),(4,"down"))
                 if f"{MAIN}layers.{layer}.mlp.shared_expert.{role}_proj.weight" in roles)
        if mask not in FP8_SHARED_MASKS[layer]:
            raise ValueError(f"unqualified layer-{layer} shared tensor-FP8 weight combination")
        masks.append(mask)
    return masks


def prefill_policy_bytes(policy, fp8_roles, row_fp8_roles=()):
    if policy not in PREFILL_POLICIES:
        raise ValueError("unknown exact native prefill recipe")
    value = PREFILL_POLICIES[policy]
    masks=validate_fp8_selection(fp8_roles)
    if not set(row_fp8_roles)<=ROW_FP8_ROLES:
        raise ValueError("unqualified native row-FP8 role")
    if value & 1 and (not A8_ROLES <= set(fp8_roles) or masks!=[3,1,1,1] or
                     not A8_ROW_ROLES <= set(row_fp8_roles)):
        raise ValueError("selective A8 requires tensor-Z0, row-Z2 and shared FP8 weight masks 3,1,1,1")
    return bytes([value])


LAYOUTS = {"BF16": "contiguous-le-v1", "FP32": "contiguous-le-v1",
           "NVFP4_EXPERT_F32M": "expert-blockscale-k16-m128x4-v1",
           "FP8_E4M3FN_TENSOR_F32M": "tensor-calibrated-v1",
           "FP8_E4M3FN_ROW_BF16S": "row-scale-v1",
           "NVFP4": "blockscale-k16-m128x4-v1",
           "NVFP4_PARTITION_F32M": "partitioned-row-blockscale-k16-v1",
           "FP8_E4M3FN_TENSOR_BF16S": "tensor-scale-v1"}


def tensor_specs(ple_format="NVFP4_PARTITION_F32M", dflash_format=None, fp8_roles=(), row_fp8_roles=(), nvfp4_shared_up=False):
    if ple_format not in ("NVFP4_PARTITION_F32M", "FP8_E4M3FN_TENSOR_BF16S"):
        raise ValueError("native PLE must use an actual NVFP4 or FP8 source codec")
    if dflash_format not in (None, "BF16", "NVFP4"):
        raise ValueError("unqualified native projection profile")
    validate_fp8_selection(fp8_roles)
    if nvfp4_shared_up and any(name.startswith(MAIN+"layers.0.mlp.shared_expert.")
                              for name in set(fp8_roles)|set(row_fp8_roles)):
        raise ValueError("NVFP4 shared-up cannot mix with unqualified layer-0 shared FP8 weights")
    if not set(row_fp8_roles) <= ROW_FP8_ROLES:
        raise ValueError("unqualified native weight-only FP8 role")
    specs = []
    def add(name, shape, fmt="BF16"):
        if nvfp4_shared_up and name==NVFP4_SHARED_UP:
            fmt="NVFP4"
        if name in fp8_roles:
            if fmt != "BF16": raise ValueError("protected control cannot become FP8")
            fmt = "FP8_E4M3FN_TENSOR_F32M"
        if name in row_fp8_roles:
            if fmt != "BF16": raise ValueError("weight-only FP8 requires original BF16 matrix")
            fmt = "FP8_E4M3FN_ROW_BF16S"
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
