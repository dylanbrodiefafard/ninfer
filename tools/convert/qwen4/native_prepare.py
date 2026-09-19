"""Offline native execution preparation. Packed-code permutations never reinterpret values."""
import math
import torch


def transformed(role, source):
    if ".hc_norm.weight" in role:
        return source.float() + 1
    if role.endswith("block_inject_weight.weight"):
        return source.float()
    if role.startswith("self_attn.") and (role.endswith("_norm.weight") or role.endswith("_layernorm.weight")):
        return source.float() + 1
    if not role.startswith("linear_attn."):
        return source
    name = role[len("linear_attn."):]
    heads = torch.arange(48).reshape(16, 3).T.flatten()
    channels = (heads[:, None] * 128 + torch.arange(128)).flatten()
    if name == "in_proj_qkv.weight":
        return torch.cat((source[:4096], source[4096:].index_select(0, channels)))
    if name == "in_proj_z.weight": return source.index_select(0, channels)
    if name == "out_proj.weight": return source.index_select(1, channels)
    if name in ("in_proj_a.weight", "in_proj_b.weight", "dt_bias"):
        return source.index_select(0, heads).float()
    if name == "conv1d.weight":
        return torch.cat((source[:4096], source[4096:].index_select(0, channels))).float()
    if name == "A_log":
        return torch.tensor([-math.exp(float(source[h])) for h in heads], dtype=torch.float32)
    if name == "norm.weight": return source.float()
    raise ValueError(f"unexpected native GDN role {role}")


def output_role(role):
    return "linear_attn.ssm_a" if role == "linear_attn.A_log" else role
