import os
import json
from pathlib import Path

import pytest
import torch

from tools.artifact.container import Artifact, plan_objects
from tools.artifact.layouts import decode_fp8_calibrated_words
from tools.convert.qwen4.native import source_requirements, prepared, fp8_payload, validate_fp8, validate_component, artifact_bf16
from tools.convert.qwen4.native_inventory import MAIN, FP8_ROLES, tensor_specs
from tools.parity.qwen4.native_inventory import layer_grammar


def test_complete_canonical_inventory_matches_independent_source_grammar():
    source=source_requirements()
    for layer in range(48):
        prefix=MAIN+f"layers.{layer}."
        actual={name:(dtype,list(shape)) for name,(dtype,shape) in source.items()
                if name.startswith(prefix) and ".ple." not in name}
        assert actual==layer_grammar(layer)
    specs=tensor_specs()
    assert len(specs)==1577
    assert len({s.name for s in specs})==1577
    objects=plan_objects(specs)
    assert sum(o.bytes for o in objects)==109169195488
    assert len(tensor_specs(dflash_format="NVFP4"))==1635
    assert sum(o.bytes for o in objects if o.name!="ple.table")>32*1024**3
    with pytest.raises(ValueError): tensor_specs(fp8_roles=[MAIN+"layers.1.mlp.gate.weight"])


def test_main_and_mtp_source_controls_keep_their_distinct_norm_boundaries():
    x=torch.tensor([-.75,0,.5,1],dtype=torch.bfloat16)
    for name in (MAIN+"hyper_connection_mixer.hc_norm.weight","mtp.hyper_connection_mixer.hc_norm.weight",
                 "mtp.layers.0.self_attn.q_norm.weight"):
        assert torch.equal(prepared(name,x),x.float()+1)
    assert prepared("mtp.pre_fc_norm_hidden.weight",x).dtype==torch.bfloat16
    assert torch.equal(prepared(MAIN+"layers.1.ple.norm_key.weight",x),x)


def test_actual_full_source_headers_cover_every_converted_role():
    root=Path(os.environ.get("NINFER_QWEN4_NATIVE_HEADERS","out/qwen4-native-inventory"))
    if not (root/"model.safetensors.index.json").exists(): pytest.skip("pinned full source header audit unavailable")
    index=json.loads((root/"model.safetensors.index.json").read_text())["weight_map"]
    expected=source_requirements()
    for shard in sorted({index[name] for name in expected}):
        header=json.loads((root/(shard+".header.json")).read_text())["header"]
        for name in [name for name in expected if index[name]==shard]:
            dtype,shape=expected.pop(name)
            assert header[name]["dtype"]==dtype
            assert header[name]["shape"]==list(shape)
    assert not expected


def test_actual_fp8_preparation_exact_codes_and_scalar_words():
    # Optional real source witness, independent scalar head-address formula. No full checkpoint.
    root=Path(os.environ.get("NINFER_QWEN4_NATIVE_LAYERS","models/qwen4-native-layers"))
    path=root/"qwen4-fp8-projections.ninfer"
    if not path.exists(): pytest.skip("actual source FP8 projection component unavailable")
    with Artifact(path) as source:
        validate_fp8(source,root/"qwen4-fp8-projection-controls.json",FP8_ROLES)
        specs={s.name:s for s in tensor_specs(fp8_roles=FP8_ROLES) if s.name in FP8_ROLES}
        for name,spec in specs.items():
            view=source.payload(name)
            original,wm,im=decode_fp8_calibrated_words(view,spec.shape)
            view.release()
            result,rwm,rim=decode_fp8_calibrated_words(fp8_payload(source,spec),spec.shape)
            expected=original.clone()
            for h in range(48):
                source_head=3*(h%16)+h//16
                dst,src=slice(h*128,(h+1)*128),slice(source_head*128,(source_head+1)*128)
                if name.endswith("linear_attn.in_proj_qkv.weight"):
                    expected[4096+h*128:4096+(h+1)*128]=original[4096+source_head*128:4096+(source_head+1)*128]
                elif name.endswith("linear_attn.in_proj_z.weight"): expected[dst]=original[src]
                elif name.endswith("linear_attn.out_proj.weight"): expected[:,dst]=original[:,src]
            assert torch.equal(result,expected)
            assert torch.equal(rwm.reshape(1).view(torch.uint8),wm.reshape(1).view(torch.uint8))
            assert torch.equal(rim.reshape(1).view(torch.uint8),im.reshape(1).view(torch.uint8))


def test_actual_mtp_component_all_prepared_controls():
    path=Path(os.environ.get("NINFER_QWEN4_NATIVE_MTP","models/qwen4-mtp/qwen4-mtp-nvfp4.ninfer"))
    if not path.exists(): pytest.skip("actual source MTP component unavailable")
    specs=[s for s in tensor_specs() if s.name.startswith("mtp.")]
    with Artifact(path) as source:
        validate_component(source,specs,source_controls=True)
        for spec in specs:
            if spec.format!="FP32": continue
            represented=artifact_bf16(source,spec.name).float()
            if spec.name.endswith(("hc_norm.weight","q_norm.weight","k_norm.weight","q_layernorm.weight","k_layernorm.weight")):
                represented=represented+1
            assert torch.equal(prepared(spec.name,artifact_bf16(source,spec.name)),represented)
