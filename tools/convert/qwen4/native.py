"""Convert the complete exact native Qwen4 preview into one .ninfer artifact.

Requires local NVIDIA main/Vision sources, the audited native MTP component, and
the six official frontend files. No downloads, ordinary-weight streaming at runtime,
runtime repacking, or invented smaller checkpoint are involved. Default protected
matrices stay BF16. Explicit prefill candidates do not change A16 decode/verification
or establish full-model quality admission.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import re
import struct

import torch

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter, ResourceSpec, TensorSpec
from tools.artifact.layouts import encode_direct, encode_nvfp4_experts, encode_fp8_calibrated, decode_fp8_calibrated_words
from tools.convert.common.safetensors import ShardReader
from tools.convert.common.fp8_quantize import encode_source_fp8
from tools.convert.qwen4.native_inventory import IDENTITY, MAIN, NVFP4_SHARED_UP, FRONTEND_FILES, FP8_ROLES, FP8_SOURCE_BUNDLES, ROW_FP8_ROLES, PREFILL_POLICIES, prefill_policy_bytes, tensor_specs, validate_fp8_selection
from tools.convert.qwen4.native_prepare import transformed
from tools.reference.qwen4.ngram import NGramConfig, layer_multipliers, layout

NVIDIA_REVISION="fc694b54fb0174e0913e6adf86691ef85a4ead47"
MTP_REVISION="39d7d235eb4748cd90d3ae575a2a2e54b49018c9"
FP8_REVISION="5d37b3b3711d8406174b96ff950c0aa16324b266"
PLE_PREFIX=MAIN+"layers.1.ple.ple_embedding."


def source_name(name):
    return name[:-len("ssm_a")]+"A_log" if name.endswith(".linear_attn.ssm_a") else name


def source_requirements():
    """Expand the exact complete canonical compute inventory into NVIDIA source fields."""
    expected={}
    for spec in tensor_specs():
        if spec.name.startswith("mtp.") or spec.name=="ple.table": continue
        if spec.format=="NVFP4_EXPERT_F32M":
            stem, role = spec.name.rsplit(".",2)[0],spec.name.rsplit(".",2)[1]
            _, n, k = spec.shape
            for expert in range(512):
                p=f"{stem}.{expert}.{role}."
                expected[p+"weight"]=("U8",(n,k//2))
                expected[p+"weight_scale"]=("F8_E4M3",(n,k//16))
                expected[p+"weight_scale_2"]=("F32",())
                expected[p+"input_scale"]=("F32",())
        else: expected[source_name(spec.name)]=("BF16",spec.shape)
    for role, shape in (("layer_multipliers",(3,)),("ngram_heads_vocab_sizes",(16,)),("ngram_heads_offsets",(16,))):
        expected[PLE_PREFIX+role]=("I64",shape)
    return expected


def validate_source(reader):
    expected=source_requirements()
    if not set(expected)<=set(reader.names): raise ValueError("incomplete exact NVIDIA native source")
    actual=reader.metadata(expected)
    for name,(dtype,shape) in expected.items():
        item=actual[name]
        if item.dtype!=dtype or item.shape!=shape: raise ValueError(f"native source shape/dtype differs: {name}")
    # Additional Text layers/roles are not silently omitted. Native NVIDIA MTP and PLE
    # table tensors are intentionally replaced/selected by the explicit component recipe.
    extras={name for name in reader.names if name.startswith(MAIN+"layers.") and ".ple." not in name}-expected.keys()
    if extras: raise ValueError(f"unexpected main source role: {sorted(extras)[0]}")
    config=NGramConfig(); table=layout(config)
    for role,values in (("layer_multipliers",layer_multipliers(config)),
                        ("ngram_heads_vocab_sizes",table.head_vocab_sizes),("ngram_heads_offsets",table.head_offsets)):
        if reader.get(PLE_PREFIX+role).tolist()!=list(values):
            raise ValueError(f"source PLE constants disagree with exact compiled addressing: {role}")


def prepared(name, value):
    match=re.match(r"model\.language_model\.layers\.\d+\.(.*)",source_name(name))
    if match: return transformed(match[1],value)
    if ".hc_norm.weight" in name: return value.float()+1
    if name.startswith("mtp."):
        role=name.removeprefix("mtp.layers.0.")
        return transformed(role,value)
    return value


def expert_payload(reader,spec):
    prefix,role=spec.name.rsplit(".",2)[0],spec.name.rsplit(".",2)[1]
    codes=[]; scales=[]; weights=[]; inputs=[]
    for expert in range(512):
        p=f"{prefix}.{expert}.{role}."
        codes.append(reader.get(p+"weight"));scales.append(reader.get(p+"weight_scale").view(torch.uint8))
        weights.append(reader.get(p+"weight_scale_2").reshape(()));inputs.append(reader.get(p+"input_scale").reshape(()))
    return encode_nvfp4_experts(torch.stack(codes),torch.stack(scales),torch.stack(weights),torch.stack(inputs),spec.shape)


def artifact_bf16(artifact,name):
    obj=artifact.find(name)
    if obj.format!="BF16": raise ValueError(f"source protected role must be BF16: {name}")
    view=artifact.payload(obj)
    try: return torch.frombuffer(bytearray(view),dtype=torch.bfloat16).reshape(obj.shape)
    finally: view.release()


def copy_payload(artifact,name):
    view=artifact.payload(name)
    try:
        for begin in range(0,len(view),8*1024*1024): yield view[begin:begin+8*1024*1024]
    finally: view.release()


def validate_component(artifact,specs,prefix="",source_controls=False):
    expected={s.name.removeprefix(prefix):s for s in specs}
    if {x.name for x in artifact.objects}!=expected.keys(): raise ValueError("native component exact inventory differs")
    for obj in artifact.objects:
        spec=expected[obj.name]
        fmt="BF16" if source_controls and spec.format=="FP32" else spec.format
        if obj.shape!=spec.shape or obj.format!=fmt:
            raise ValueError(f"native component representation differs: {obj.name}")


def validate_fp8_bundle(artifact,audit_path):
    """Validate one immutable finite source bundle, independently of selected product roles."""
    if artifact.identity!=ArtifactIdentity("qwen4/native-fp8-projection-qualification","senfu-fp8-source"):
        raise ValueError("requires audited source-calibrated FP8 projection component")
    names={obj.name for obj in artifact.objects}
    bundle=next((name for name,roles in FP8_SOURCE_BUNDLES.items() if names==roles),None)
    if bundle is None: raise ValueError("requires exact original13 or additional8 FP8 source inventory")
    roles=FP8_SOURCE_BUNDLES[bundle]
    report=json.loads(audit_path.read_text())
    if (report.get("repository")!="senfu/Qwen3.8-Flash-Next-NVFP4" or
        report.get("compared_with_repository")!="nvidia/Qwen3.8-Flash-Next-NVFP4" or
        report.get("revision")!=FP8_REVISION or report.get("compared_with_revision")!=NVIDIA_REVISION):
        raise ValueError("FP8 protected-control provenance mismatch")
    layers=(0,3) if bundle=="original13" else (1,2)
    excluded={f"{MAIN}layers.{layer}.linear_attn.{role}.weight"
              for layer in (1,2) for role in ("in_proj_qkv","out_proj")} if bundle=="additional8" else set()
    if set(report.get("excluded_noncontrol_projections",[]))!=excluded:
        raise ValueError("FP8 control audit exclusion scope differs")
    control_names={name for name,(dtype,_) in source_requirements().items()
                   if dtype=="BF16" and any(name.startswith(f"{MAIN}layers.{layer}.") for layer in layers)
                   and ".ple." not in name and name not in roles and name not in excluded}
    if set(report.get("exact_equal",{}))!=control_names or not all(report["exact_equal"].values()):
        raise ValueError("FP8 source protected controls were not compared exactly")
    expected=[TensorSpec(s.name,s.shape,"FP8_E4M3FN_TENSOR_F32M","tensor-calibrated-v1")
              for s in tensor_specs() if s.name in roles]
    validate_component(artifact,expected)
    return bundle


def bind_fp8_sources(sources,roles):
    """One/two audited finite bundles -> unique source owner for each selected matrix."""
    selected=set(roles)
    if len(selected)!=len(roles) or not selected<=FP8_ROLES:
        raise ValueError("FP8 roles must be unique explicit qualified source projections")
    validate_fp8_selection(selected)
    if bool(sources)!=bool(selected) or len(sources)>2:
        raise ValueError("FP8 selection requires one or two audited source bundles and explicit roles")
    owners={};provenance=[];seen=set()
    for artifact,audit in sources:
        bundle=validate_fp8_bundle(artifact,audit)
        if bundle in seen: raise ValueError("duplicate FP8 source bundle")
        seen.add(bundle)
        supplied=selected & FP8_SOURCE_BUNDLES[bundle]
        if not supplied: raise ValueError("FP8 source bundle has no selected role")
        owners.update({name:artifact for name in supplied})
        provenance.append({"bundle":bundle,"roles":sorted(supplied),"revision":FP8_REVISION,
                           "protected_controls_compared_with_revision":NVIDIA_REVISION})
    if owners.keys()!=selected: raise ValueError("selected FP8 role has no audited source bundle")
    return owners,provenance


def fp8_payload(artifact,spec):
    view=artifact.payload(spec.name)
    try: codes,wm,im=decode_fp8_calibrated_words(view,spec.shape)
    finally: view.release()
    role=spec.name.split(".",4)[-1]  # main prefix and exact layer index are removed.
    if role.startswith("linear_attn."): codes=transformed(role,codes)
    return encode_fp8_calibrated(codes,wm,im)


def validate_nvfp4_shared_up(artifact,component_path,fit_path):
    """Admit only the frozen, role-bound layer-0 shared-up A16 weight candidate."""
    profile="nvfp4_diagonal_calibrated"
    if artifact.identity!=ArtifactIdentity("qwen4/native-projection-candidate",profile):
        raise ValueError("requires frozen diagonal-calibrated NVFP4 shared-up candidate")
    validate_component(artifact,[TensorSpec("weight",(640,2560),"NVFP4","blockscale-k16-m128x4-v1"),
                                 TensorSpec("input_scale_divisor",(),"FP32","contiguous-le-v1")])
    fit=json.loads(fit_path.read_text())
    if fit.get("layer")!=0 or fit.get("stage")!="calibration_fit" or fit.get("complete") is not True:
        raise ValueError("requires completed layer-0 calibration-only fit")
    role=fit.get("roles",{}).get("shared_up",{})
    if role.get("source_tensor")!=NVFP4_SHARED_UP or role.get("shape")!=[640,2560]:
        raise ValueError("frozen NVFP4 candidate is not exact layer-0 shared-up")
    candidate=role.get("variants",{}).get(profile,{})
    if Path(candidate.get("artifact","")).resolve()!=component_path.resolve():
        raise ValueError("explicit NVFP4 component differs from frozen fit artifact")
    manifest=json.loads(Path(fit["manifest"]).read_text())
    if manifest.get("repository")!="nvidia/Qwen3.8-Flash-Next-NVFP4" or manifest.get("revision")!=NVIDIA_REVISION:
        raise ValueError("NVFP4 shared-up calibration source provenance differs")
    calibration=[p["id"] for p in manifest["panels"] if p["split"]=="calibration"]
    documents=candidate.get("documents",[])
    if (not calibration or len(set(calibration))!=len(calibration) or
        fit.get("calibration_ids")!=calibration or fit.get("evaluated_panels")!=calibration or
        [p.get("id") for p in documents]!=calibration or any(p.get("split")!="calibration" for p in documents)):
        raise ValueError("NVFP4 frozen fit must contain only its declared calibration panels")
    recipe="K16 E4M3 scale factor search minimizing diagonal input-energy weighted error on calibration only; original global FP32 divisor; independent FP64 code decode and dots."
    if fit.get("recipe")!=recipe:
        raise ValueError("unqualified NVFP4 shared-up fitting recipe")
    return {"role":NVFP4_SHARED_UP,"profile":profile,"recipe":recipe,
            "source_revision":NVIDIA_REVISION,"calibration_ids":calibration,
            "activation_policy":"A16Only; fitted activation divisor not imported"}


def native_fp8_ple(reader):
    for part in range(128):
        name=PLE_PREFIX+f"ngram_embedding.shard_{part}.weight"
        value=reader.get(name)
        if value.dtype!=torch.float8_e4m3fn or tuple(value.shape)!=(2500012,160):
            raise ValueError("native FP8 PLE source shape/dtype differs")
        codes=value.view(torch.uint8)
        for begin in range(0,2500012,16384):
            chunk=codes[begin:begin+16384]
            if bool(((chunk&127)==127).any()): raise ValueError("nonfinite native FP8 PLE code")
            yield chunk.numpy().tobytes()
    scalar=reader.get(PLE_PREFIX+"ngram_embedding.weight_scale")
    if scalar.dtype!=torch.bfloat16 or scalar.numel()!=1 or not bool(torch.isfinite(scalar).all() & (scalar>0).all()):
        raise ValueError("native FP8 PLE scalar must be finite positive BF16")
    yield encode_direct(scalar,"BF16")


def convert(args):
    if args.out.exists() or args.out.with_suffix(".ninfer.partial").exists(): raise FileExistsError(args.out)
    with ExitStack() as stack:
        source=stack.enter_context(ShardReader(args.model))
        validate_source(source)
        mtp=stack.enter_context(Artifact(args.mtp))
        if mtp.identity!=ArtifactIdentity("qwen4/native-mtp-qualification","limpincat-nvfp4-w4a16-source"):
            raise ValueError("requires exact audited native NVFP4 MTP source component")
        ple=stack.enter_context(Artifact(args.ple)) if args.ple else None
        ple_format="FP8_E4M3FN_TENSOR_BF16S" if ple is None else ple.find("ple.table").format
        if ple and ple.identity!=ArtifactIdentity("qwen4/native-ple-qualification","primitive-nvfp4-complete-table"):
            raise ValueError("external PLE component must be the complete audited native NVFP4 table")
        draft=stack.enter_context(Artifact(args.dflash)) if args.dflash else None
        draft_format=None
        if draft:
            ids={ArtifactIdentity("qwen4/native-dflash-qualification","pixelml-bf16-a16"):"BF16",
                 ArtifactIdentity("qwen4/native-dflash-qualification","pixelml-nvfp4-a16"):"NVFP4"}
            if draft.identity not in ids: raise ValueError("requires the exact PixelML DFlash companion")
            draft_format=ids[draft.identity]
        fp8,fp8_sources=bind_fp8_sources(
            [(stack.enter_context(Artifact(component)),audit) for component,audit in args.fp8_source],args.fp8_role)
        shared_up=stack.enter_context(Artifact(args.nvfp4_shared_up[0])) if args.nvfp4_shared_up else None
        shared_up_profile=validate_nvfp4_shared_up(shared_up,*args.nvfp4_shared_up) if shared_up else None
        specs=tensor_specs(ple_format,draft_format,args.fp8_role,args.row_fp8_role,shared_up is not None)
        validate_component(mtp,[s for s in specs if s.name.startswith("mtp.")],source_controls=True)
        if ple: validate_component(ple,[s for s in specs if s.name=="ple.table"])
        if draft: validate_component(draft,[s for s in specs if s.name.startswith("dflash.")],prefix="dflash.")
        resources={"frontend/"+name:(args.model/name).read_bytes() for name in FRONTEND_FILES}
        # The shared frontend performs the authoritative tokenizer/template/processor validation.
        # Preserve original source bytes; do not rewrite a template into another checkpoint's form.
        profile={"source":"nvidia/Qwen3.8-Flash-Next-NVFP4","revision":NVIDIA_REVISION,
                 "mtp_source":"limpincat/flashnext-drafters","mtp_revision":MTP_REVISION,
                 "ple_format":ple_format,"dflash_format":draft_format,"fp8_roles":sorted(args.fp8_role),
                 "row_fp8_roles":sorted(args.row_fp8_role),
                 "nvfp4_shared_up":shared_up_profile,
                 "row_fp8_recipe":"source-BF16 per-row maxabs/448, BF16 scale, E4M3FN nearest-even",
                 "fp8_sources":fp8_sources,"prefill_policy":args.prefill_policy,
                 "decode_verify_policy":"A16Only","model_quality":"requires full-model qualification",
                 "dflash_quality":"not admitted as a default" if draft_format=="NVFP4" else "checkpoint-specific qualification required"}
        resources["native-profile.json"]=(json.dumps(profile,sort_keys=True)+"\n").encode()
        resources["native-prefill-policy"]=prefill_policy_bytes(args.prefill_policy,args.fp8_role,args.row_fp8_role)
        objects=specs+[ResourceSpec(name,"raw-bytes-v1",len(data)) for name,data in resources.items()]
        args.out.parent.mkdir(parents=True,exist_ok=True)
        temporary=args.out.with_suffix(".ninfer.partial")
        with ArtifactWriter(temporary,IDENTITY,objects) as writer:
            for spec in specs:
                name=spec.name
                if name=="ple.table": payload=copy_payload(ple,name) if ple else native_fp8_ple(source)
                elif name.startswith("dflash."): payload=copy_payload(draft,name.removeprefix("dflash."))
                elif name.startswith("mtp."):
                    payload=(encode_direct(prepared(name,artifact_bf16(mtp,name)),"FP32")
                             if spec.format=="FP32" else copy_payload(mtp,name))
                elif spec.format=="FP8_E4M3FN_TENSOR_F32M": payload=fp8_payload(fp8[name],spec)
                elif spec.format=="FP8_E4M3FN_ROW_BF16S":
                    payload=encode_source_fp8(prepared(name,source.get(source_name(name))))
                elif spec.format=="NVFP4_EXPERT_F32M": payload=expert_payload(source,spec)
                elif name==NVFP4_SHARED_UP and shared_up: payload=copy_payload(shared_up,"weight")
                else: payload=encode_direct(prepared(name,source.get(source_name(name))),spec.format)
                writer.write(name,payload)
                print(name,flush=True)
            for name,data in resources.items(): writer.write(name,data)
        temporary.rename(args.out)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True,help="complete local pinned NVIDIA HF source directory")
    parser.add_argument("--mtp",type=Path,required=True,help="audited native NVFP4 MTP component")
    parser.add_argument("--ple",type=Path,help="complete audited native NVFP4 PLE; omitted uses source NVIDIA FP8 table")
    parser.add_argument("--dflash",type=Path,help="optional audited PixelML BF16 or NVFP4 companion")
    parser.add_argument("--fp8-source",type=Path,nargs=2,action="append",default=[],metavar=("COMPONENT","AUDIT"),
                        help="one exact original13/additional8 component and its control audit; repeat at most twice")
    parser.add_argument("--fp8-role",action="append",default=[],choices=sorted(FP8_ROLES))
    parser.add_argument("--row-fp8-role",action="append",default=[],choices=sorted(ROW_FP8_ROLES),
                        help="explicit original-BF16 row weight; A16 default, Z2 A8 only in selective prefill; not quality admission")
    parser.add_argument("--nvfp4-shared-up",type=Path,nargs=2,metavar=("COMPONENT","FIT_JSON"),
                        help="frozen diagonal-calibrated layer-0 shared-up weight; A16 only, no layer-0 shared FP8 mix")
    parser.add_argument("--prefill-policy",default="a16",choices=tuple(PREFILL_POLICIES),
                        help="explicit full C1 prefill candidate; decode/verify/head remain A16; not quality admission")
    parser.add_argument("--out",type=Path,required=True)
    torch.set_num_threads(2)
    convert(parser.parse_args())
