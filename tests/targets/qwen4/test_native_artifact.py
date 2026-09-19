import os
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from tools.artifact.container import Artifact, ArtifactWriter, ArtifactIdentity, plan_objects
from tools.artifact.layouts import decode_fp8_calibrated_words
from tools.convert.common.fp8_quantize import encode_source_fp8
from tools.convert.qwen4.native import source_requirements, prepared, fp8_payload, validate_fp8_bundle, bind_fp8_sources, validate_component, validate_nvfp4_shared_up, copy_payload, artifact_bf16
from tools.convert.qwen4.native_inventory import MAIN, NVFP4_SHARED_UP, FP8_ROLES, FP8_ORIGINAL_ROLES, FP8_ADDITIONAL_ROLES, ROW_FP8_ROLES, A8_ROLES, A8_ROW_ROLES, prefill_policy_bytes, tensor_specs
from tools.parity.qwen4.native_inventory import layer_grammar


def test_prefill_candidate_recipe_encoding_and_required_weights():
    assert prefill_policy_bytes("a16", []) == b"\x00"
    assert prefill_policy_bytes("routed-a4", []) == b"\x02"
    assert prefill_policy_bytes("selective-a8", A8_ROLES,A8_ROW_ROLES) == b"\x01"
    assert prefill_policy_bytes("routed-a4-selective-a8", A8_ROLES,A8_ROW_ROLES) == b"\x03"
    assert prefill_policy_bytes("selective-a8", A8_ROLES,ROW_FP8_ROLES) == b"\x01"
    with pytest.raises(ValueError): prefill_policy_bytes("selective-a8",A8_ROLES)
    for missing in A8_ROLES:
        with pytest.raises(ValueError):
            prefill_policy_bytes("selective-a8", A8_ROLES - {missing},A8_ROW_ROLES)
    with pytest.raises(ValueError): prefill_policy_bytes("qsa-a8", FP8_ROLES)
    for extra in ("0.mlp.shared_expert.down_proj.weight","1.mlp.shared_expert.down_proj.weight",
                  "2.mlp.shared_expert.up_proj.weight","3.mlp.shared_expert.up_proj.weight"):
        with pytest.raises(ValueError): prefill_policy_bytes("selective-a8",A8_ROLES|{MAIN+"layers."+extra},A8_ROW_ROLES)


def test_tensor_fp8_shared_combinations_and_rejected_projection_roles():
    allowed=({0,1,2,3,4,6,7},{0,1,2,4,5},{0,1,2,3,4,5},{0,1,2,3})
    for layer in range(4):
        for mask in range(8):
            roles={f"{MAIN}layers.{layer}.mlp.shared_expert.{role}_proj.weight"
                   for bit,role in ((1,"gate"),(2,"up"),(4,"down")) if mask&bit}
            if mask in allowed[layer]:
                assert {s.name for s in tensor_specs(fp8_roles=roles) if s.format=="FP8_E4M3FN_TENSOR_F32M"}==roles
            else:
                with pytest.raises(ValueError): tensor_specs(fp8_roles=roles)
    for role in ("0.linear_attn.in_proj_qkv.weight","0.linear_attn.out_proj.weight",
                 "2.linear_attn.in_proj_z.weight","3.self_attn.q_proj.weight",
                 "3.self_attn.k_proj.weight","3.self_attn.v_proj.weight","3.self_attn.o_proj.weight"):
        with pytest.raises(ValueError): tensor_specs(fp8_roles=[MAIN+"layers."+role])


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


def test_weight_only_fp8_storage_is_independent_of_activations_and_controls():
    specs=tensor_specs(row_fp8_roles=ROW_FP8_ROLES)
    assert {s.name for s in specs if s.format=="FP8_E4M3FN_ROW_BF16S"}==ROW_FP8_ROLES
    assert all(s.layout=="row-scale-v1" for s in specs if s.name in ROW_FP8_ROLES)
    assert prefill_policy_bytes("a16",[])==b"\x00"
    with pytest.raises(ValueError): prefill_policy_bytes("selective-a8",ROW_FP8_ROLES)
    for name in (MAIN+"layers.0.attn_hyper_connection.input_mix_weight_down.weight",
                 MAIN+"layers.1.ple.norm_key.weight",MAIN+"layers.0.mlp.gate.weight"):
        with pytest.raises(ValueError): tensor_specs(row_fp8_roles=[name])


def test_finite_nvfp4_shared_up_storage_and_combination_rejection():
    specs=tensor_specs(nvfp4_shared_up=True)
    assert {s.name for s in specs if s.format=="NVFP4"}=={NVFP4_SHARED_UP}
    for role in ("gate","up","down"):
        with pytest.raises(ValueError):
            tensor_specs(fp8_roles=[MAIN+f"layers.0.mlp.shared_expert.{role}_proj.weight"],nvfp4_shared_up=True)
    with pytest.raises(ValueError): tensor_specs(fp8_roles=A8_ROLES,nvfp4_shared_up=True)
    assert prefill_policy_bytes("routed-a4",[])==b"\x02"  # Shared matrix remains A16.
    assert next(s for s in specs if s.name==NVFP4_SHARED_UP).shape==(640,2560)


def test_actual_row_fp8_z2_exact_frozen_codes_and_scale_permutation():
    root=Path(os.environ.get("NINFER_QWEN4_NATIVE_LAYERS","models/qwen4-native-layers"))
    fit_path=Path(os.environ.get("NINFER_QWEN4_ROW_Z2_FIT","out/qwen4-projection-study/layer2.fit.json"))
    source_path=root/"qwen4-layer-2.safetensors"
    if not fit_path.exists() or not source_path.exists(): pytest.skip("actual source/frozen row-Z2 candidate unavailable")
    fit=json.loads(fit_path.read_text());role=MAIN+"layers.2.linear_attn.in_proj_z.weight"
    candidate=Path(fit["roles"]["gdn_z"]["variants"]["row_fp8"]["artifact"])
    with safe_open(str(source_path),framework="pt") as source:
        actual=encode_source_fp8(prepared(role,source.get_tensor(role)))
    with Artifact(candidate) as artifact:
        obj=artifact.find("weight")
        assert artifact.identity==ArtifactIdentity("qwen4/native-projection-candidate","source-bf16-row-fp8")
        assert obj.format=="FP8_E4M3FN_ROW_BF16S" and obj.shape==(6144,2560)
        raw=bytes(artifact.payload("weight"));expected=bytearray(raw)
    # Independent source-head address arithmetic; row code and BF16 scale planes
    # must both follow the V-side permutation without changing any represented word.
    scale_offset=6144*2560
    for head in range(48):
        source_head=3*(head%16)+head//16
        for lane in range(128):
            dst,src=head*128+lane,source_head*128+lane
            expected[dst*2560:(dst+1)*2560]=raw[src*2560:(src+1)*2560]
            expected[scale_offset+2*dst:scale_offset+2*dst+2]=raw[scale_offset+2*src:scale_offset+2*src+2]
    assert actual==bytes(expected)
    assert role not in FP8_ROLES and role not in A8_ROLES
    assert role in A8_ROW_ROLES


def test_actual_frozen_nvfp4_shared_up_binding_and_exact_copy(tmp_path):
    fit_path=Path(os.environ.get("NINFER_QWEN4_SHARED_UP_FIT","out/qwen4-projection-study/layer0.fit.json"))
    if not fit_path.exists(): pytest.skip("frozen source shared-up calibration unavailable")
    fit=json.loads(fit_path.read_text())
    component=Path(fit["roles"]["shared_up"]["variants"]["nvfp4_diagonal_calibrated"]["artifact"])
    with Artifact(component) as source:
        provenance=validate_nvfp4_shared_up(source,component,fit_path)
        assert provenance["calibration_ids"]==fit["calibration_ids"]
        spec=next(s for s in tensor_specs(nvfp4_shared_up=True) if s.name==NVFP4_SHARED_UP)
        target=tmp_path/"shared-up.ninfer"
        with ArtifactWriter(target,ArtifactIdentity("qwen4/native-component-qualification","shared-up-copy"),[spec]) as writer:
            writer.write(spec.name,copy_payload(source,"weight"))
        with Artifact(target) as copied:
            assert bytes(copied.payload(spec.name))==bytes(source.payload("weight"))
        # These reports bind the otherwise role-less weight component to one source matrix.
        bad=tmp_path/"bad-fit.json"
        for key,value in (("layer",1),("stage","frozen_holdout"),("complete",False),
                          ("evaluated_panels",fit["calibration_ids"]+["heldout_algorithms"]),
                          ("recipe","maxabs only")):
            bad.write_text(json.dumps({**fit,key:value}))
            with pytest.raises(ValueError): validate_nvfp4_shared_up(source,component,bad)
        changed=json.loads(json.dumps(fit));changed["roles"]["shared_up"]["source_tensor"]=MAIN+"layers.1.mlp.shared_expert.up_proj.weight"
        bad.write_text(json.dumps(changed))
        with pytest.raises(ValueError): validate_nvfp4_shared_up(source,component,bad)
        changed=json.loads(json.dumps(fit));changed["roles"]["shared_up"]["variants"]["nvfp4_diagonal_calibrated"]["documents"][0]["split"]="heldout"
        bad.write_text(json.dumps(changed))
        with pytest.raises(ValueError): validate_nvfp4_shared_up(source,component,bad)
        manifest=json.loads(Path(fit["manifest"]).read_text());manifest["revision"]="wrong"
        manifest_path=tmp_path/"manifest.json";manifest_path.write_text(json.dumps(manifest))
        bad.write_text(json.dumps({**fit,"manifest":str(manifest_path)}))
        with pytest.raises(ValueError): validate_nvfp4_shared_up(source,component,bad)
        with pytest.raises(ValueError): validate_nvfp4_shared_up(source,tmp_path/"other.ninfer",fit_path)
    original=Path(fit["roles"]["shared_up"]["variants"]["nvfp4_maxabs"]["artifact"])
    with Artifact(original) as source:
        with pytest.raises(ValueError): validate_nvfp4_shared_up(source,original,fit_path)


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


@pytest.mark.parametrize("component,audit,bundle",[
    ("qwen4-fp8-projections.ninfer","qwen4-fp8-projection-controls.json","original13"),
    ("qwen4-fp8-projection-additional.ninfer","qwen4-fp8-projection-additional-controls.json","additional8")])
def test_actual_fp8_preparation_exact_codes_and_scalar_words(component,audit,bundle):
    # Optional real source witness, independent scalar head-address formula. No full checkpoint.
    root=Path(os.environ.get("NINFER_QWEN4_NATIVE_LAYERS","models/qwen4-native-layers"))
    path=root/component
    if not path.exists(): pytest.skip("actual source FP8 projection component unavailable")
    with Artifact(path) as source:
        assert validate_fp8_bundle(source,root/audit)==bundle
        roles=FP8_ORIGINAL_ROLES if bundle=="original13" else FP8_ADDITIONAL_ROLES
        # Fixture preparation validates source bytes even for not-yet-admitted roles.
        specs={s.name:s for s in tensor_specs() if s.name in roles}
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


def test_audited_fp8_bundle_selection_and_storage_activation_separation(tmp_path):
    root=Path(os.environ.get("NINFER_QWEN4_NATIVE_LAYERS","models/qwen4-native-layers"))
    original=root/"qwen4-fp8-projections.ninfer";additional=root/"qwen4-fp8-projection-additional.ninfer"
    if not original.exists() or not additional.exists(): pytest.skip("both exact FP8 source bundles unavailable")
    z1=MAIN+"layers.1.linear_attn.in_proj_z.weight"
    with Artifact(original) as first,Artifact(additional) as second:
        a=(first,root/"qwen4-fp8-projection-controls.json")
        b=(second,root/"qwen4-fp8-projection-additional-controls.json")
        selected=A8_ROLES|{z1}
        owners,provenance=bind_fp8_sources([a,b],sorted(selected))
        assert owners[z1] is second and all(owners[role] is first for role in selected&FP8_ORIGINAL_ROLES)
        assert {p["bundle"] for p in provenance}=={"original13","additional8"}
        original_selected=A8_ROLES&FP8_ORIGINAL_ROLES
        assert set(bind_fp8_sources([a],sorted(original_selected))[0])==original_selected
        assert bind_fp8_sources([b],[z1])[0][z1] is second
        for sources,roles in (([a],[z1]),([a,a],sorted(original_selected)),([a,b],[z1]),
                              ([b],[z1,z1]),([b],[MAIN+"layers.2.linear_attn.in_proj_z.weight"])):
            with pytest.raises(ValueError): bind_fp8_sources(sources,roles)
        with pytest.raises(ValueError): validate_fp8_bundle(second,a[1])
        report=json.loads(b[1].read_text());report["exact_equal"][next(iter(report["exact_equal"]))]=False
        failed=tmp_path/"failed-controls.json";failed.write_text(json.dumps(report))
        with pytest.raises(ValueError): validate_fp8_bundle(second,failed)
        report=json.loads(b[1].read_text());report["excluded_noncontrol_projections"].pop()
        failed.write_text(json.dumps(report))
        with pytest.raises(ValueError): validate_fp8_bundle(second,failed)
    specs={s.name:s for s in tensor_specs(fp8_roles=[z1])}
    assert specs[z1].format=="FP8_E4M3FN_TENSOR_F32M" and z1 not in A8_ROLES
    assert prefill_policy_bytes("a16",[z1])==b"\x00"
    with pytest.raises(ValueError): prefill_policy_bytes("selective-a8",[z1])
    for role in FP8_ADDITIONAL_ROLES-FP8_ROLES:
        with pytest.raises(ValueError): tensor_specs(fp8_roles=[role])


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
