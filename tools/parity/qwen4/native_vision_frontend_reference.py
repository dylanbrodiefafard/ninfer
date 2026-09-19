"""Pinned Qwen2VL/Qwen3VL pixel-layout oracle, using PyTorch CPU RGB8 bicubic.

The transformer consumer pin is c119ec3cc37ab69642f39cca2de4187714002b08;
native NVIDIA processor settings are fc694b54fb0174e0913e6adf86691ef85a4ead47.
No tokenizer, video decoding or model weights are needed for these pixel goldens.
"""
import argparse
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from tools.artifact.container import ArtifactIdentity, ArtifactWriter, TensorSpec

CASES={"image":(1,83,121),"video":(3,63,95),"small_video":(2,17,25)}


def build(path):
    if path.exists(): raise FileExistsError(path)
    values={}
    for name,(frames,height,width) in CASES.items():
        source=torch.stack([((torch.arange(height*width*3)*29+frame*37)%256).to(torch.uint8)
                            .reshape(height,width,3).permute(2,0,1) for frame in range(frames)])
        h,w=height,width
        video=name!="image"
        if video and (h<32 or w<32):
            scale=max(32/h,32/w); h,w=int(h*scale),int(w*scale)
        hh,ww=round(h/32)*32,round(w/32)*32
        tt=round(frames/2)*2 if video else 1
        minimum,maximum=(4096,25165824) if video else (65536,16777216)
        if tt*hh*ww>maximum:
            beta=math.sqrt(frames*h*w/maximum)
            hh,ww=max(32,math.floor(h/beta/32)*32),max(32,math.floor(w/beta/32)*32)
        elif tt*hh*ww<minimum:
            beta=math.sqrt(minimum/(frames*h*w))
            hh,ww=math.ceil(h*beta/32)*32,math.ceil(w*beta/32)*32
        resized=F.interpolate(source,size=(hh,ww),mode="bicubic",align_corners=False,antialias=True)
        normalized=(resized.to(torch.float32)-127.5)/127.5
        if frames%2: normalized=torch.cat((normalized,normalized[-1:]),dim=0)
        t,gh,gw=normalized.shape[0]//2,hh//16,ww//16
        patches=normalized.reshape(t,2,3,gh//2,2,16,gw//2,2,16)
        patches=patches.permute(0,3,6,4,7,2,1,5,8).reshape(-1,1536).contiguous()
        values[name+".patches"]=patches
        values[name+".grid"]=torch.tensor([t,gh,gw],dtype=torch.int32)
    with ArtifactWriter(path,ArtifactIdentity("qwen4/vision-frontend-reference","source-rgb8-bicubic"),
        [TensorSpec(name,tuple(v.shape),"FP32" if v.dtype==torch.float32 else "I32","contiguous-le-v1")
         for name,v in values.items()]) as writer:
        for name,v in values.items(): writer.write(name,v.view(torch.uint8).numpy().tobytes())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",type=Path,required=True)
    build(parser.parse_args().out)
