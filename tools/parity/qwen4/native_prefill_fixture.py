"""Prepare seven selective-A8 weights: six audited tensor-FP8 roles plus source row-Z2."""
from __future__ import annotations

import argparse
from pathlib import Path
from safetensors import safe_open

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter
from tools.convert.common.fp8_quantize import encode_source_fp8
from tools.convert.qwen4.native import fp8_payload, bind_fp8_sources, prepared
from tools.convert.qwen4.native_inventory import A8_ROLES, A8_ROW_ROLES, prefill_policy_bytes, tensor_specs


def prepare(root: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    with Artifact(root / "qwen4-fp8-projections.ninfer") as original, \
         Artifact(root / "qwen4-fp8-projection-additional.ninfer") as additional, \
         safe_open(str(root / "qwen4-layer-2.safetensors"),framework="pt") as source:
        sources,_=bind_fp8_sources([(original,root/"qwen4-fp8-projection-controls.json"),
            (additional,root/"qwen4-fp8-projection-additional-controls.json")],A8_ROLES)
        prefill_policy_bytes("selective-a8",A8_ROLES,A8_ROW_ROLES)
        specs = [s for s in tensor_specs(fp8_roles=A8_ROLES,row_fp8_roles=A8_ROW_ROLES)
                 if s.name in A8_ROLES|A8_ROW_ROLES]
        with ArtifactWriter(output, ArtifactIdentity("qwen4/native-prefill-qualification",
                                                     "mixed-fp8-tiled"), specs) as writer:
            for spec in specs:
                writer.write(spec.name,fp8_payload(sources[spec.name],spec) if spec.name in A8_ROLES else
                             encode_source_fp8(prepared(spec.name,source.get_tensor(spec.name))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source_root, args.out)
