"""Original BF16 head W4A16 assessment using the named existing max-abs recipe.

This is weight-loss evidence, not calibration, CUDA qualification or admission.
Row chunks carry a discarded source-amax sentinel tile so the existing encoder
uses the exact full-matrix divisor without allocating its full 8-way distance array.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tools.artifact.container import Artifact, ArtifactIdentity
from tools.convert.common.nvfp4_quantize import ENCODER_PROFILE, quantize_nvfp4_matrix
from tools.parity.qwen4.native_weight_candidates import error, lut, read_bf16
from tools.reference.qwen4.common import linear
from tools.reference.qwen4.mtp import represented


def run(root: Path, trace: Path, output: Path):
    if output.exists():raise FileExistsError(output)
    report=json.loads(trace.read_text())
    if report["profile"]!="qwen4-mtp-vllm-tokenspeed-frozen-domain-w4a16":
        raise ValueError("wrong authentic MTP trace")
    x=represented(report["records"][0]["stages"]["final_read"]).reshape(-1,2560)[:3].bfloat16()
    with Artifact(root/"qwen4-endpoint.ninfer") as source:
        if source.identity!=ArtifactIdentity("qwen4/native-endpoint-qualification","nvidia-bf16-source"):
            raise ValueError("wrong native endpoint")
        weight=read_bf16(source,"lm_head.weight")
    amax=weight.abs().amax()
    sentinel=torch.zeros((128,2560),dtype=torch.bfloat16);sentinel[0,0]=amax
    signed=torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],dtype=torch.float64)
    candidate,reference=[],[]
    squared,source_squared=0.,0.
    divisor_word=None
    for first in range(0,248320,1024):
        chunk=weight[first:first+1024];n=len(chunk)
        packed,scales,divisor=quantize_nvfp4_matrix(torch.cat((chunk,sentinel)))
        word=int(divisor.view(torch.int32))
        if divisor_word is None:divisor_word=word
        if word!=divisor_word:raise AssertionError("chunk divisor differs from full-matrix max")
        packed=packed[:n];scales=scales[:n]
        codes=torch.empty((n,2560),dtype=torch.uint8)
        codes[:,0::2]=packed&15;codes[:,1::2]=packed>>4
        decoded=signed[codes.long()]*lut()[scales.long()].repeat_interleave(16,dim=1)/float(divisor)
        original=chunk.double()
        squared+=float(((decoded-original)**2).sum());source_squared+=float((original**2).sum())
        candidate.append(linear(x,decoded));reference.append(linear(x,original))
    y=torch.cat(candidate,dim=1);baseline=torch.cat(reference,dim=1)
    p=torch.log_softmax(baseline,dim=-1);q=torch.log_softmax(y,dim=-1)
    result={"source":str(root/"qwen4-endpoint.ninfer"),"trace":str(trace),"encoder_profile":ENCODER_PROFILE,
            "activation_policy":"A16 only; no invented calibration",
            "weight_relative_l2":(squared/source_squared)**.5,"logit_source_loss":error(y,baseline),
            "logit_source_loss_per_input":[error(a,b) for a,b in zip(y,baseline)],
            "softmax_kl_per_input":(p.exp()*(p-q)).sum(-1).tolist(),
            "distribution_domain":"Full physical 248320 rows; diagnostic softmax, not native masked 248077-token domain or product p-less sampling",
            "argmax_agreement":(y.argmax(-1)==baseline.argmax(-1)).tolist(),
            "weight_divisor_fp32_word":divisor_word,
            "oracle":"FP64 dot of independently decoded signed E2M1 times exact stored E4M3 scale divided by exact FP32 divisor; same represented BF16 input as source head",
            "scope":"Full original head, three authentic private-MTP output rows from diagnostic target inputs. Weight-only compression loss, not GPU error, independent quality holdout, calibration, PPL or runtime admission."}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,required=True)
    parser.add_argument("--trace",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    run(args.source_root,args.trace,args.out)
