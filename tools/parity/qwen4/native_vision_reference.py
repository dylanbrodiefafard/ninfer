"""Independent FP64 native Vision schedule goldens at public Op BF16 boundaries.

This is a numerical oracle, not an inference backend. Norm, projection, bias,
RoPE, attention, activation and residual are separately public Ops in the native
schedule. Their outputs become represented BF16 inputs to the next Op. The closed
merger is evaluated as one complete ideal formula, without its private casts.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import torch

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter, TensorSpec
from tools.reference.qwen4.vision import learned_bias_layer_norm, patch_merger


def represented(x):
    return x.to(torch.bfloat16).to(torch.float64)


def controls(grids):
    positions, indices, weights, segments = [], [], [], [0]
    # Enumerate raster coordinates and sort by logical merge group, independently of
    # the production nested block traversal. Float32 grid coefficients are public inputs.
    for t, h, w in grids:
        coords = sorted(((y, x) for y in range(h) for x in range(w)),
                        key=lambda p: (p[0]//2, p[1]//2, p[0]%2, p[1]%2))
        for _ in range(t):
            for y, x in coords:
                positions.append((y, x))
                fy = float(torch.tensor(y*47., dtype=torch.float32) / (h-1))
                fx = float(torch.tensor(x*47., dtype=torch.float32) / (w-1))
                iy, ix = math.floor(fy), math.floor(fx)
                wy, wx = fy-iy, fx-ix
                indices.append([min(47, iy+dy)*48+min(47, ix+dx) for dy in range(2) for dx in range(2)])
                weights.append([(wy if dy else 1-wy)*(wx if dx else 1-wx) for dy in range(2) for dx in range(2)])
            segments.append(len(positions))
    return (torch.tensor(positions, dtype=torch.int32), torch.tensor(indices, dtype=torch.int32),
            torch.tensor(weights, dtype=torch.float32), segments)


def rotary(x, positions):
    frequency = 10000. ** (-torch.arange(18, dtype=torch.float64)/18.)
    angles = (positions.to(torch.float64)[:, :, None]*frequency).reshape(-1, 36)[:, None, :]
    lo, hi = x[..., :36], x[..., 36:]
    return torch.cat((lo*angles.cos()-hi*angles.sin(), hi*angles.cos()+lo*angles.sin()), dim=-1)


def evaluate(artifact: Artifact, patches, positions, indices, weights, segments, trace=None):
    def source(name):
        item = artifact.find("model.visual."+name)
        return torch.frombuffer(bytearray(artifact.payload(item)), dtype=torch.bfloat16).reshape(item.shape).to(torch.float64)
    def linear(x, name):
        w=source(name+".weight").flatten(1)
        return represented(x @ w.T+source(name+".bias"))
    def norm(x, name):
        return represented(learned_bias_layer_norm(x,source(name+".weight"),source(name+".bias")))
    def block(x,layer):
        p=f"blocks.{layer}."
        qkv=linear(norm(x,p+"norm1"),p+"attn.qkv").reshape(-1,3,16,72)
        q=represented(rotary(qkv[:,0],positions)); k=represented(rotary(qkv[:,1],positions)); v=qkv[:,2]
        attention=torch.empty_like(q)
        for begin,end in zip(segments,segments[1:]):
            for head in range(16):
                scores=q[begin:end,head] @ k[begin:end,head].T / math.sqrt(72.)
                exp=(scores-scores.max(dim=-1,keepdim=True).values).exp()
                attention[begin:end,head]=(exp/exp.sum(dim=-1,keepdim=True)) @ v[begin:end,head]
        x=represented(x+linear(represented(attention).flatten(1),p+"attn.proj"))
        up=linear(norm(x,p+"norm2"),p+"mlp.linear_fc1")
        up=represented(.5*up*(1+(math.sqrt(2/math.pi)*(up+.044715*up**3)).tanh()))
        x=represented(x+linear(up,p+"mlp.linear_fc2"))
        return x
    x=linear(patches,"patch_embed.proj")
    table=source("pos_embed.weight")
    position=represented((table[indices.long()]*weights.to(torch.float64)[...,None]).sum(1))
    x=represented(x+position)
    def report(label, got, expected):
        print(label,"rel_l2",float(torch.linalg.vector_norm(got-expected)/torch.linalg.vector_norm(expected)),
              "max_abs",float((got-expected).abs().max()),"max_ref",float(expected.abs().max()),flush=True)
    if trace is not None: report("patch",trace[0],x)
    for layer in range(27):
        x=block(x,layer)
        if trace is not None:
            report(f"block{layer} accumulated",trace[layer+1],x)
            report(f"block{layer} same_input",trace[layer+1],block(trace[layer],layer))
        else: print(f"oracle block {layer}",flush=True)
    merged=patch_merger(x.reshape(-1,2,2,1152),source("merger.norm.weight"),source("merger.norm.bias"),
        source("merger.linear_fc1.weight"),source("merger.linear_fc1.bias"),
        source("merger.linear_fc2.weight"),source("merger.linear_fc2.bias")).output
    return x,merged


def build(source: Path, output: Path | None, trace_path: Path | None = None):
    if output is not None and output.exists(): raise FileExistsError(output)
    torch.set_num_threads(8)
    grids=[(1,2,2),(1,2,4)]
    positions,indices,weights,segments=controls(grids)
    patches=represented(((torch.arange(12*1536,dtype=torch.float64)*13 % 251)-125).reshape(12,1536)/128.)
    with Artifact(source) as artifact:
        if artifact.identity!=ArtifactIdentity("qwen4/native-vision-qualification","nvidia-bf16-source"):
            raise ValueError("unexpected native Vision source")
        trace=None
        if trace_path is not None:
            trace=torch.frombuffer(bytearray(trace_path.read_bytes()),dtype=torch.bfloat16).reshape(28,12,1152).double()
        encoder,output_values=evaluate(artifact,patches,positions,indices,weights,segments,trace)
    if output is None: return
    values={"patches":patches.to(torch.bfloat16),"encoder":encoder.to(torch.float32),
        "output":output_values.to(torch.float32),"positions":positions.T.contiguous(),
        "indices":indices,"weights":weights,"segments":torch.tensor(segments,dtype=torch.int32)}
    formats={torch.bfloat16:"BF16",torch.float32:"FP32",torch.int32:"I32"}
    with ArtifactWriter(output,ArtifactIdentity("qwen4/native-vision-reference","fp64-public-op-boundaries"),
        [TensorSpec(name,tuple(value.shape),formats[value.dtype],"contiguous-le-v1") for name,value in values.items()]) as writer:
        for name,value in values.items(): writer.write(name,value.contiguous().view(torch.uint8).numpy().tobytes())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True); parser.add_argument("--out",type=Path)
    parser.add_argument("--trace",type=Path)
    args=parser.parse_args()
    if args.out is None and args.trace is None: parser.error("--out or --trace required")
    build(args.source,args.out,args.trace)
