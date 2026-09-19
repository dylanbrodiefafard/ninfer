"""Acquire only the complete pinned FP8 PLE payload and retain resumable source ranges.

No ordinary model weights, cache tuning, or inference-time disk fallback. Source
parts and converted output are separate; reads and conversion have bounded buffers.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
from pathlib import Path
import shutil
import struct
import time
import urllib.error

import numpy as np

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter, TensorSpec
from tools.parity.qwen4.native_source import BASE, REPOSITORY, REVISION, read_header, read_range
from tools.parity.qwen4.native_ple_fixture import SHARD_ROWS
from tools.parity.qwen4.native_nvfp4_ple_fixture import bf16_word

PARTITIONS=128
PART_BYTES=SHARD_ROWS*160
PAYLOAD_BYTES=PARTITIONS*PART_BYTES+2
SHARD="model-fp8-mtp-ple.safetensors"
PREFIX="model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
CHUNK=16*1024*1024


def capacity(root: Path):
    memory={line.split()[0].rstrip(":"):int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines()}
    available=memory["MemAvailable"]
    arc_path=Path("/proc/spl/kstat/zfs/arcstats")
    arc={};reclaim=0
    if arc_path.exists():
        if Path("/sys/module/zfs/parameters/zfs_arc_pc_percent").read_text().strip()!="0":
            raise RuntimeError("documented ARC admission requires zfs_arc_pc_percent=0")
        for line in arc_path.read_text().splitlines():
            words=line.split()
            if len(words)==3 and words[0] in ("mru_evictable_data","mfu_evictable_data","size","c_min"):
                arc[words[0]]=int(words[2])
        reclaim=min((arc["mru_evictable_data"]+arc["mfu_evictable_data"])//2,max(arc["size"]-arc["c_min"],0))
    required=PAYLOAD_BYTES+16*1024**3
    result={"mem_available":available,"arc":arc,"discounted_clean_arc":reclaim,
            "available_with_documented_arc":available+reclaim,"required_with_16gib_reserve":required,
            "disk_free":shutil.disk_usage(root).free}
    print("Capacity "+json.dumps(result),flush=True)
    if available+reclaim<required:raise RuntimeError("insufficient conservative host capacity for complete FP8 table")
    if result["disk_free"]<2*PAYLOAD_BYTES+1024**3:
        raise RuntimeError("insufficient disk for retained source plus artifact")
    return result


def audit():
    base,header=read_header(SHARD)
    ranges=[]
    for p in range(PARTITIONS):
        name=PREFIX+f"shard_{p}.weight";t=header[name];begin,end=t["data_offsets"]
        if t["dtype"]!="F8_E4M3" or t["shape"]!=[SHARD_ROWS,160] or end-begin!=PART_BYTES or begin<0:
            raise ValueError("pinned FP8 PLE partition inventory differs")
        ranges.append([base+begin,base+end])
    ordered=sorted(ranges)
    if any(a[1]>b[0] for a,b in zip(ordered,ordered[1:])):raise ValueError("overlapping PLE ranges")
    t=header[PREFIX+"weight_scale"];begin,end=t["data_offsets"]
    if t["dtype"]!="BF16" or t["shape"]!=[1] or end-begin!=2:raise ValueError("invalid PLE scale")
    scale=read_range(BASE+SHARD,base+begin,base+end-1)
    if scale!=b"\x51\x39":raise ValueError("pinned original BF16 PLE multiplier differs")
    return {"repository":REPOSITORY,"revision":REVISION,"source_shard":SHARD,
            "partitions":PARTITIONS,"rows_per_partition":SHARD_ROWS,"ranges_exclusive":ranges,
            "scale_range_exclusive":[base+begin,base+end],"scale_hex":scale.hex(),"payload_bytes":PAYLOAD_BYTES}


def acquire(root: Path, manifest, workers: int):
    def one(partition):
        output=root/f"part-{partition:03d}.fp8";temporary=output.with_suffix(".fp8.partial")
        if output.exists():
            if output.stat().st_size!=PART_BYTES:raise ValueError("incomplete published source part")
            return
        begin,end=manifest["ranges_exclusive"][partition]
        mode="ab" if temporary.exists() else "xb"
        done=temporary.stat().st_size if temporary.exists() else 0
        if done>PART_BYTES:raise ValueError("partial source exceeds exact interval")
        with temporary.open(mode) as stream:
            while done<PART_BYTES:
                count=min(CHUNK,PART_BYTES-done)
                for attempt in range(4):
                    try:
                        data=read_range(BASE+SHARD,begin+done,begin+done+count-1)
                        break
                    except (urllib.error.URLError,TimeoutError,ConnectionError):
                        if attempt==3:raise
                        time.sleep(2**attempt)
                if np.any((np.frombuffer(data,dtype=np.uint8)&127)==127):raise ValueError("nonfinite source FP8 code")
                # Entire exact HTTP range was validated before append; an interrupted local
                # write leaves a valid prefix whose byte length is the next resume offset.
                stream.write(data);stream.flush();done+=len(data)
        if temporary.stat().st_size!=end-begin:raise ValueError("downloaded part extent differs")
        temporary.rename(output)
        print(f"FP8 PLE source part {partition+1}/128 complete ({PART_BYTES} bytes)",flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(one,range(PARTITIONS)))


def decode_source_row(codes,scale):
    multiplier=struct.unpack("<f",b"\x00\x00"+scale)[0]
    result=[]
    for word in codes:
        exponent,fraction=(word>>3)&15,word&7
        if exponent==15 and fraction==7:raise ValueError("nonfinite source code")
        value=math.ldexp(fraction,-9) if exponent==0 else math.ldexp(8+fraction,exponent-10)
        value=math.copysign(value,-1. if word&128 else 1.)
        product=struct.unpack("<f",struct.pack("<f",value*multiplier))[0]
        result.append(bf16_word(product))
    return struct.pack("<160H",*result)


def convert(root: Path,output: Path,manifest,admission):
    reference=output.with_name(output.stem+"-boundary-reference.ninfer")
    report=output.with_suffix(".json");temporary=output.with_suffix(".ninfer.partial")
    for path in (output,reference,report,temporary):
        if path.exists():raise FileExistsError(path)
    scale=bytes.fromhex(manifest["scale_hex"])
    def chunks():
        for p in range(PARTITIONS):
            path=root/f"part-{p:03d}.fp8"
            if path.stat().st_size!=PART_BYTES:raise ValueError("source part extent differs")
            with path.open("rb") as stream:
                while data:=stream.read(CHUNK):
                    if np.any((np.frombuffer(data,dtype=np.uint8)&127)==127):raise ValueError("nonfinite FP8 source")
                    yield data
            print(f"FP8 PLE converted part {p+1}/128",flush=True)
        yield scale
    spec=TensorSpec("ple.table",(PARTITIONS*SHARD_ROWS,160),"FP8_E4M3FN_TENSOR_BF16S","tensor-scale-v1")
    with ArtifactWriter(temporary,ArtifactIdentity("qwen4/native-ple-qualification","nvidia-fp8-complete-table"),[spec]) as writer:
        writer.write("ple.table",chunks())
    ids=[];expected=bytearray()
    # Scalar source oracle is independent of converted payload and GPU decoder.
    with Artifact(temporary) as artifact:
        obj=artifact.find("ple.table");payload=artifact.payload(obj)
        try:
            if bytes(payload[-2:])!=scale:raise AssertionError("converted scale differs")
            for p in range(PARTITIONS):
                with (root/f"part-{p:03d}.fp8").open("rb") as stream:
                    for row in (0,SHARD_ROWS-1):
                        stream.seek(row*160);codes=stream.read(160);index=p*SHARD_ROWS+row
                        if bytes(payload[index*160:(index+1)*160])!=codes:raise AssertionError("converted source boundary differs")
                        ids.append(index);expected.extend(decode_source_row(codes,scale))
        finally:payload.release()
    temporary.rename(output)
    with ArtifactWriter(reference,ArtifactIdentity("qwen4/native-ple-reference","source-scalar-bf16"),[
            TensorSpec("row_ids",(256,),"I32","contiguous-le-v1"),
            TensorSpec("embedding",(256,160),"BF16","contiguous-le-v1")]) as writer:
        writer.write("row_ids",struct.pack("<256i",*ids));writer.write("embedding",expected)
    with report.open("x") as stream:
        json.dump({**manifest,"admission":admission,"conversion":"Exact FP8 source bytes partition-major, exact BF16 multiplier footer",
                   "reference":"256 independently scalar-decoded original source boundary rows; converted boundary code bytes checked exactly",
                   "scope":"Complete source/conversion; GPU eager-lock residency qualification is separate"},stream,indent=2)
        stream.write("\n")


def run(root: Path,output: Path,workers: int,download: bool):
    root.mkdir(parents=True,exist_ok=True);output.parent.mkdir(parents=True,exist_ok=True)
    with (root/"acquisition.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        admission=capacity(root);manifest=audit()
        path=root/"source-ranges.json"
        if path.exists():
            if json.loads(path.read_text())!=manifest:raise ValueError("resume manifest differs from exact remote source")
        else:
            with path.open("x") as stream:json.dump(manifest,stream,indent=2)
        print("Remote pinned source validated: only128 PLE partitions + original BF16 scale",flush=True)
        if download:acquire(root,manifest,workers)
        convert(root,output,manifest,admission)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--download",action="store_true")
    parser.add_argument("--workers",type=int,choices=range(1,5),default=4)
    args=parser.parse_args();run(args.sources,args.out,args.workers,args.download)
