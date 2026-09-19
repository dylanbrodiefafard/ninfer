"""Prepare only the thirteen audited FP8 overrides for native decoder qualification."""
from __future__ import annotations

import argparse
from pathlib import Path

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter
from tools.convert.qwen4.native import fp8_payload, validate_fp8
from tools.convert.qwen4.native_inventory import FP8_ROLES, tensor_specs


def prepare(root: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    with Artifact(root / "qwen4-fp8-projections.ninfer") as source:
        validate_fp8(source, root / "qwen4-fp8-projection-controls.json", FP8_ROLES)
        specs = [s for s in tensor_specs(fp8_roles=FP8_ROLES) if s.name in FP8_ROLES]
        with ArtifactWriter(output, ArtifactIdentity("qwen4/native-prefill-qualification",
                                                     "senfu-fp8-tiled"), specs) as writer:
            for spec in specs:
                writer.write(spec.name, fp8_payload(source, spec))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source_root, args.out)
