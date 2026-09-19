"""Independent exact head-address witnesses for offline native GDN preparation."""

import math

import torch

from tools.convert.qwen4.native_prepare import transformed


def test_native_gdn_permutation_preserves_every_represented_word_and_inverse_role():
    qkv = torch.arange(10240*2, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16).reshape(10240, 2)
    z = qkv[4096:].clone()
    output = z.T.contiguous()
    conv = qkv[:, :1, None].expand(-1, 1, 4).contiguous()
    actual_qkv = transformed("linear_attn.in_proj_qkv.weight", qkv)
    actual_z = transformed("linear_attn.in_proj_z.weight", z)
    actual_output = transformed("linear_attn.out_proj.weight", output)
    actual_conv = transformed("linear_attn.conv1d.weight", conv)
    assert torch.equal(actual_qkv[:4096].view(torch.uint16), qkv[:4096].view(torch.uint16))
    for h in range(48):
        src = 3*(h % 16) + h//16
        a, b = slice(h*128, (h+1)*128), slice(src*128, (src+1)*128)
        assert torch.equal(actual_qkv[4096+h*128:4096+(h+1)*128].view(torch.uint16),
                           qkv[4096+src*128:4096+(src+1)*128].view(torch.uint16))
        assert torch.equal(actual_z[a].view(torch.uint16), z[b].view(torch.uint16))
        assert torch.equal(actual_output[:, a].view(torch.uint16), output[:, b].view(torch.uint16))
        assert torch.equal(actual_conv[4096+h*128:4096+(h+1)*128], conv[4096+src*128:4096+(src+1)*128].float())


def test_native_effective_controls_preserve_source_semantics():
    norm = torch.tensor([-.75, -.5, 0., .5], dtype=torch.bfloat16)
    assert torch.equal(transformed("attn_hyper_connection.hc_norm.weight", norm), norm.float()+1)
    assert torch.equal(transformed("self_attn.indexer.k_layernorm.weight", norm), norm.float()+1)
    assert torch.equal(transformed("linear_attn.norm.weight", norm), norm.float())
    heads = torch.arange(48, dtype=torch.float32).bfloat16()
    a_log = (heads.float()/16-2).bfloat16()
    actual_dt = transformed("linear_attn.dt_bias", heads)
    actual_decay = transformed("linear_attn.A_log", a_log)
    for h in range(48):
        src = 3*(h % 16) + h//16
        assert float(actual_dt[h]) == float(heads[src])
        assert actual_decay[h] == torch.tensor(-math.exp(float(a_log[src])), dtype=torch.float32)
