"""Bounded original-BF16 row-FP8 candidates; never a model quality admission.

Encode real head/GR/PLE matrices and emit independent FP64-dot golden values for
GPU A16 qualification. Source-weight loss is reported separately from GPU error.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter, TensorSpec
from tools.artifact.layouts import encode_direct
from tools.convert.common.fp8_quantize import encode_source_fp8
from tools.reference.qwen4.common import linear, silu, source_grouped_rmsnorm
from tools.reference.qwen4.gated_residual import source_read
from tools.reference.qwen4.mtp import represented
from tools.reference.qwen4.ple import source_inject


def lut():
    values = []
    for word in range(256):
        exponent, fraction = (word >> 3) & 15, word & 7
        value = fraction / 512 if exponent == 0 else math.ldexp(1 + fraction / 8, exponent - 7)
        values.append(math.nan if exponent == 15 and fraction == 7 else (-value if word & 128 else value))
    return torch.tensor(values, dtype=torch.float64)


def read_bf16(artifact, name):
    obj = artifact.find(name)
    if obj.format != "BF16":
        raise ValueError(f"original BF16 required: {name}")
    return torch.frombuffer(bytearray(artifact.payload(name)), dtype=torch.bfloat16).reshape(obj.shape)


def decode_rows(payload, n, k):
    # Independent row/scale addressing and mathematical E4M3 decode; no production decoder.
    codes = torch.frombuffer(bytearray(payload[:n*k]), dtype=torch.uint8).reshape(n, k)
    offset = (n*k + 255) // 256 * 256
    scales = torch.frombuffer(bytearray(payload[offset:offset+2*n]), dtype=torch.bfloat16).double()
    return lut()[codes.long()] * scales[:, None]


def error(actual, reference):
    delta = actual.double() - reference.double()
    return {"relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference.double())),
            "max_absolute": float(delta.abs().max())}


def run(root: Path, trace: Path, output: Path):
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    report = json.loads(trace.read_text())
    if report["profile"] != "qwen4-mtp-vllm-tokenspeed-frozen-domain-w4a16":
        raise ValueError("expected authentic diagnostic-target MTP input trace")
    record = report["records"][0]
    hidden = represented(record["target_hidden"]).reshape(-1, 4, 2560)[:3].bfloat16()
    head_input = represented(record["stages"]["final_read"]).reshape(-1, 2560)[:3].bfloat16()
    main = "model.language_model.hyper_connection_mixer."
    pp = "model.language_model.layers.1.ple."
    with Artifact(root / "qwen4-endpoint.ninfer") as endpoint, Artifact(root / "qwen4-ple-component.ninfer") as ple, Artifact(root / "qwen4-text-panel.ninfer") as panel:
        if endpoint.identity != ArtifactIdentity("qwen4/native-endpoint-qualification", "nvidia-bf16-source"):
            raise ValueError("wrong native endpoint source identity")
        norm = read_bf16(endpoint, main + "hc_norm.weight")
        gd = read_bf16(endpoint, main + "input_mix_weight_down.weight")
        gu = read_bf16(endpoint, main + "input_mix_weight_up.weight")
        pk = read_bf16(ple, pp + "key_proj.weight")
        pv = read_bf16(ple, pp + "value_proj.weight")
        norms = [read_bf16(ple, pp + x) for x in ("norm_key.weight", "norm_query.weight", "norm_conv.weight")]
        conv = read_bf16(ple, pp + "conv1d.weight").reshape(10240, 4)
        embedding = read_bf16(panel, "token.embeddings")
        rows = panel.find("ple.rows")
        raw = bytearray(panel.payload("ple.rows"))
        codes = torch.frombuffer(raw, dtype=torch.uint8, count=rows.shape[0]*160).reshape(-1,160)
        scale = torch.frombuffer(raw, dtype=torch.bfloat16, count=1, offset=rows.shape[0]*160).double()
        ids = torch.frombuffer(bytearray(panel.payload("ple.local_rows")), dtype=torch.int32).long()[:48]
        gathered = (lut()[codes[ids].long()] * scale).bfloat16().reshape(3,16,160)
        normalized = source_grouped_rmsnorm(hidden.flatten(-2), norm, group_size=2560).bfloat16()
        low = silu(linear(normalized, gd)/4).bfloat16()
        inputs = {"head": head_input, "gr_down": normalized, "gr_up": low,
                  "ple_key": gathered.reshape(3,2560), "ple_value": gathered.reshape(3,2560)}
        shapes = {"head": (248320,2560), "gr_down": (320,10240), "gr_up": (10240,320),
                  "ple_key": (10240,2560), "ple_value": (2560,2560)}
        specs = []
        for name, (n,k) in shapes.items():
            specs.extend((TensorSpec(name+".weight",(n,k),"FP8_E4M3FN_ROW_BF16S","row-scale-v1"),
                          TensorSpec(name+".input",(3,k),"BF16","contiguous-le-v1"),
                          TensorSpec(name+".oracle",(3,n),"FP32","contiguous-le-v1")))
        results, decoded = {}, {}
        output.parent.mkdir(parents=True, exist_ok=True)
        with ArtifactWriter(output, ArtifactIdentity("qwen4/native-weight-candidates", "source-bf16-row-fp8-a16"), specs) as writer:
            for name in shapes:
                source = read_bf16(endpoint,"lm_head.weight") if name=="head" else {"gr_down":gd,"gr_up":gu,"ple_key":pk,"ple_value":pv}[name]
                payload = encode_source_fp8(source)
                n,k = shapes[name]
                # Chunk the full head's independent decode/dots, avoiding a multi-GiB FP64 copy.
                if name=="head":
                    oracle, baseline = [], []
                    offset=(n*k+255)//256*256
                    scales=torch.frombuffer(bytearray(payload[offset:offset+2*n]),dtype=torch.bfloat16).double()
                    codes=torch.frombuffer(bytearray(payload[:n*k]),dtype=torch.uint8).reshape(n,k)
                    squared, source_squared = 0., 0.
                    for start in range(0,n,1024):
                        w=lut()[codes[start:start+1024].long()]*scales[start:start+1024,None]
                        original=source[start:start+1024].double()
                        squared+=float(((w-original)**2).sum());source_squared+=float((original**2).sum())
                        oracle.append(linear(inputs[name],w));baseline.append(linear(inputs[name],original))
                    ideal=torch.cat(oracle,dim=1);original_output=torch.cat(baseline,dim=1)
                    weight_error={"relative_l2":math.sqrt(squared/source_squared)}
                else:
                    w=decode_rows(payload,n,k);decoded[name]=w
                    ideal=linear(inputs[name],w);original_output=linear(inputs[name],source)
                    weight_error=error(w,source)
                results[name]={"source_weight_loss":weight_error,"source_projection_loss":error(ideal,original_output),
                               "source_bytes":source.numel()*2,"candidate_bytes":len(payload)}
                if name=="head":
                    p=torch.log_softmax(original_output,dim=-1);q=torch.log_softmax(ideal,dim=-1)
                    results[name]["distribution_domain"]="Full physical 248320 rows; diagnostic softmax, not native masked 248077-token domain or product p-less sampling"
                    results[name]["softmax_kl_per_input"]=(p.exp()*(p-q)).sum(-1).tolist()
                    results[name]["argmax_agreement"]=(ideal.argmax(-1)==original_output.argmax(-1)).tolist()
                writer.write(name+".weight",payload)
                writer.write(name+".input",encode_direct(inputs[name],"BF16"))
                writer.write(name+".oracle",encode_direct(ideal.float(),"FP32"))
                print(name,results[name],flush=True)
                del source,payload
        gr_reference=source_read(hidden,norm,gd,gu,None).mixed
        gr_results={}
        for name,down,up in (("down",decoded["gr_down"],gu),("up",gd,decoded["gr_up"]),("both",decoded["gr_down"],decoded["gr_up"])):
            gr_results[name]=error(source_read(hidden,norm,down,up,None).mixed,gr_reference)
        state=torch.zeros((10240,9),dtype=torch.bfloat16)
        def inject(key,value):return source_inject(hidden,gathered,key,value,*norms,conv,state)
        ple_reference=inject(pk,pv);ple_results={}
        for name,key,value in (("key",decoded["ple_key"],pv),("value",pk,decoded["ple_value"]),("both",decoded["ple_key"],decoded["ple_value"])):
            candidate=inject(key,value)
            ple_results[name]={field:error(getattr(candidate,field),getattr(ple_reference,field)) for field in ("output","injection","next_conv_state","gate")}
        emb_payload=encode_source_fp8(embedding)
        embedding_error=error(decode_rows(emb_payload,*embedding.shape).bfloat16(),embedding)
        component_values={"hidden":hidden,"gr.norm":norm.float()+1,
                          "ple.embedding":gathered.reshape(3,2560),"ple.key_norm":norms[0],
                          "ple.query_norm":norms[1],"ple.conv_norm":norms[2],"ple.conv":conv,
                          "ple.state":state.T.contiguous()}
        component_weights={"gr_down":gd,"gr_up":gu,"ple_key":pk,"ple_value":pv}
        component_values["gr.oracle"]=source_read(hidden,norm,decoded["gr_down"],decoded["gr_up"],None).mixed.float()
        for name,key,value in (("key",decoded["ple_key"],pv),("both",decoded["ple_key"],decoded["ple_value"])):
            candidate=inject(key,value)
            component_values[f"ple.{name}.oracle"]=candidate.output.float()
            component_values[f"ple.{name}.state_oracle"]=candidate.next_conv_state.T.contiguous().float()
        component_specs=[TensorSpec(name,tuple(value.shape),"FP32" if value.dtype==torch.float32 else "BF16","contiguous-le-v1")
                         for name,value in component_values.items()]
        for name,value in component_weights.items():
            component_specs.append(TensorSpec(name+".weight",tuple(value.shape),"FP8_E4M3FN_ROW_BF16S","row-scale-v1"))
        component_specs.append(TensorSpec("ple_value.source",tuple(pv.shape),"BF16","contiguous-le-v1"))
        with ArtifactWriter(output.with_suffix(".components.ninfer"),ArtifactIdentity("qwen4/native-weight-candidate-components","source-bf16-row-fp8-a16"),component_specs) as writer:
            for name,value in component_values.items():
                writer.write(name,encode_direct(value,"FP32" if value.dtype==torch.float32 else "BF16"))
            for name,value in component_weights.items():writer.write(name+".weight",encode_source_fp8(value))
            writer.write("ple_value.source",encode_direct(pv,"BF16"))
    summary={"source_root":str(root),"trace":str(trace),"matrices":results,"final_gr_source_loss":gr_results,
             "ple_source_loss":ple_results,"selected_embedding_rows_source_loss":embedding_error,
             "oracle":"Independent E4M3 scalar-formula lookup, explicit row-scale addresses, FP64 dots; GPU golden is FP32-rounded FP64 result, represented BF16 input",
             "scope":"Original source BF16 matrices; three authentic diagnostic-target carried-state/MTP rows; selected authentic PLE table rows paired as a bounded component input, not a full model rollout. No calibration, PPL, acceptance, throughput or full embedding-table admission."}
    output.with_suffix(".json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,required=True)
    parser.add_argument("--trace",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    run(args.source_root,args.trace,args.out)
