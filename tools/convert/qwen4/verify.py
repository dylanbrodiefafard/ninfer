"""Verify a converted Qwen4 verifier artifact byte-for-byte."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Sequence

from tools.artifact.container import Artifact, TensorObject

from .convert import _COPY_CHUNK, preflight
from . import inventory


def verify(
    model_dir: str | Path,
    artifact_path: str | Path,
    *,
    verify_source_hashes: bool = True,
) -> None:
    source = preflight(model_dir, verify_hashes=verify_source_hashes)
    with Artifact.open(artifact_path) as artifact:
        if artifact.identity.model_id != inventory.MODEL_ID:
            raise ValueError("artifact model id does not name the verifier")
        if artifact.identity.weights_id != inventory.WEIGHTS_ID:
            raise ValueError("artifact weights id does not name UD-IQ1_S host staging")
        if len(artifact.objects) != len(inventory.TENSOR_SPECS):
            raise ValueError("artifact tensor count differs from the closed inventory")

        for index, spec in enumerate(inventory.TENSOR_SPECS, 1):
            obj = artifact.find(spec.name)
            if not isinstance(obj, TensorObject):
                raise ValueError(f"{spec.name}: artifact object is not a tensor")
            expected = source.source_by_name[spec.name]
            if (
                obj.shape != spec.shape
                or obj.format != spec.format
                or obj.layout != spec.layout
                or obj.bytes != expected.tensor.bytes
            ):
                raise ValueError(f"{spec.name}: artifact descriptor mismatch")
            payload = artifact.payload(obj)
            begin, _ = expected.shard.absolute_span(expected.tensor)
            try:
                source_digest = hashlib.sha256()
                artifact_digest = hashlib.sha256()
                with expected.shard.path.open("rb") as file:
                    file.seek(begin)
                    for offset in range(0, obj.bytes, _COPY_CHUNK):
                        size = min(_COPY_CHUNK, obj.bytes - offset)
                        source_bytes = file.read(size)
                        if len(source_bytes) != size:
                            raise IOError(f"{spec.name}: source tensor short read")
                        source_digest.update(source_bytes)
                        artifact_digest.update(payload[offset : offset + size])
                if source_digest.digest() != artifact_digest.digest():
                    raise ValueError(f"{spec.name}: encoded payload SHA-256 differs")
            finally:
                payload.release()
            print(f"[{index}/{len(artifact.objects)}] {spec.name}", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--skip-source-hash", action="store_true")
    args = parser.parse_args(argv)
    verify(
        args.model,
        args.artifact,
        verify_source_hashes=not args.skip_source_hash,
    )


if __name__ == "__main__":
    main()


__all__ = ["main", "verify"]
