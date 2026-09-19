"""Encode only authentic, explicitly addressed native draft embedding fixture rows.

This is not a full token table or product artifact. Unavailable token rows must
remain poisoned in the bounded runtime witness rather than synthesized.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter, TensorSpec
from tools.artifact.layouts import encode_direct
from tools.convert.common.fp8_quantize import encode_source_fp8
from tools.parity.qwen4.native_weight_candidates import decode_rows, error, read_bf16


def run(native: Path, draft: Path, output: Path):
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    with Artifact(native / "qwen4-text-panel.ninfer") as panel, Artifact(draft / "qwen4-dflash-inputs.ninfer") as inputs:
        if panel.identity != ArtifactIdentity("qwen4/native-text-qualification", "nvidia-source-33"):
            raise ValueError("wrong original native text panel")
        ids = torch.frombuffer(bytearray(panel.payload("token.ids")), dtype=torch.int32).clone()
        rows = read_bf16(panel, "token.embeddings")
        mask = read_bf16(inputs, "mask.embedding").reshape(1,2560)
        if tuple(rows.shape) != (33,2560) or ids.numel() != 33:
            raise ValueError("expected exact 33-row native token panel")
        ids = torch.cat((ids, torch.tensor([248077],dtype=torch.int32)))
        rows = torch.cat((rows,mask))
        for i, token in enumerate(ids.tolist()):
            if token < 0 or token >= 248320:
                raise ValueError("embedding token outside exact native vocabulary")
            for j in range(i):
                if ids[j] == token and not torch.equal(rows[j],rows[i]):
                    raise ValueError("duplicate token source rows disagree")
        payload = encode_source_fp8(rows)
        decoded = decode_rows(payload,34,2560).bfloat16()
        specs=[TensorSpec("token.ids",(34,),"I32","contiguous-le-v1"),
               TensorSpec("embedding.rows",(34,2560),"FP8_E4M3FN_ROW_BF16S","row-scale-v1"),
               TensorSpec("embedding.decoded",(34,2560),"BF16","contiguous-le-v1")]
        output.parent.mkdir(parents=True,exist_ok=True)
        with ArtifactWriter(output,ArtifactIdentity("qwen4/native-embedding-row-candidate","original-bf16-row-fp8"),specs) as writer:
            writer.write("token.ids",encode_direct(ids,"I32"))
            writer.write("embedding.rows",payload)
            writer.write("embedding.decoded",encode_direct(decoded,"BF16"))
        report={"native_source":str(native / "qwen4-text-panel.ninfer"),
                "mask_source":str(draft / "qwen4-dflash-inputs.ninfer"),
                "token_ids":ids.tolist(),"source_loss":error(decoded,rows),
                "scope":"34 addressed fixture rows including duplicates and authentic mask; no complete embedding table, calibration or model quality admission. Decoded gather oracle uses independent E4M3 formula and stored BF16 row scales."}
        output.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root",type=Path,required=True)
    parser.add_argument("--draft-root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    run(args.native_root,args.draft_root,args.out)
