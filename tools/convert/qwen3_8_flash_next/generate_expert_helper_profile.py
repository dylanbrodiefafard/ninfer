"""Compile calibration NPY, server route JSON, or their normalized blend (stdlib only)."""

import argparse
import ast
import hashlib
import json
import math
import shlex
import struct
import sys
from fractions import Fraction
from pathlib import Path


def read_counts(path: Path) -> tuple[list[int], str]:
    raw = path.read_bytes()
    if raw[:6] != b"\x93NUMPY" or raw[6:8] not in (b"\x01\x00", b"\x02\x00"):
        raise ValueError("expected an NPY v1/v2 array")
    width = 2 if raw[6] == 1 else 4
    start = 8 + width
    length = int.from_bytes(raw[8:start], "little")
    header = ast.literal_eval(raw[start : start + length].decode("latin1"))
    if header["shape"] != (48, 512) or header["fortran_order"]:
        raise ValueError("CALIB must be C-order [48,512]")
    dtype = header["descr"]
    if dtype not in ("<i8", "<u8", "<i4", "<u4", "|i1", "|u1", "<i2", "<u2"):
        raise ValueError(f"CALIB must contain little-endian integers, got {dtype}")
    code = {"i8": "q", "u8": "Q", "i4": "i", "u4": "I", "i2": "h", "u2": "H", "i1": "b", "u1": "B"}[
        dtype[1:]
    ]
    payload = raw[start + length :]
    if len(payload) != 48 * 512 * int(dtype[2:]):
        raise ValueError("CALIB payload has the wrong byte length")
    counts = [entry[0] for entry in struct.iter_unpack("<" + code, payload)]
    if any(value < 0 or value > (1 << 64) - 1 for value in counts):
        raise ValueError("CALIB count outside uint64 range")
    return counts, hashlib.sha256(raw).hexdigest()


def read_server_profile(path: Path) -> tuple[list[int], dict, str]:
    raw = path.read_bytes()
    profile = json.loads(raw)
    if profile["version"] != 1 or profile["identity"] != {
        "model_id": "qwen3.8-flash-next",
        "weights_id": "f3-fp8-nvfp4",
    }:
        raise ValueError("server profile version or identity mismatch")
    rows = profile["decode_lookups"]
    if (
        not isinstance(rows, list)
        or len(rows) != 48
        or any(not isinstance(row, list) or len(row) != 512 for row in rows)
    ):
        raise ValueError("server decode_lookups must have shape [48,512]")
    counts = [v for row in rows for v in row]
    codecs = ("nvfp4", "f3g32")
    lookups = [profile["codecs"][c]["decode_lookups"] for c in codecs]
    misses = [profile["codecs"][c]["decode_misses"] for c in codecs]
    integers = counts + lookups + misses + [profile["rounds"], profile["miss_bytes"]]
    if any(type(v) is not int or not 0 <= v < 1 << 64 for v in integers):
        raise ValueError("server counters must be uint64 integers")
    if sum(counts) != sum(lookups) or any(m > n for m, n in zip(misses, lookups, strict=True)):
        raise ValueError("server route/codec counters disagree")
    if any(v > profile["rounds"] for v in counts):
        raise ValueError("server count exceeds decode rounds")
    if profile["miss_bytes"] != misses[0] * 2764800 + misses[1] * 1996800:
        raise ValueError("server miss_bytes disagrees with codec misses")
    hit_rate = 1 - sum(misses) / sum(lookups) if sum(lookups) else 1.0
    if not math.isfinite(profile["hit_rate"]) or abs(profile["hit_rate"] - hit_rate) > 1e-12:
        raise ValueError("server hit_rate disagrees with codec counters")
    if not math.isfinite(profile["uptime_s"]) or profile["uptime_s"] < 0:
        raise ValueError("server uptime_s must be finite and nonnegative")
    return counts, profile, hashlib.sha256(raw).hexdigest()


def blend_counts(workload: list[int], calibration: list[int], weight: Fraction) -> list[int]:
    """Normalize each entire [48,512] distribution before weighting; quantize to 2^48 units."""
    if not 0 <= weight <= 1:
        raise ValueError("--workload-weight must lie in [0,1]")
    work_total, calib_total = sum(workload), sum(calibration)
    if not work_total or not calib_total:
        raise ValueError("blending requires nonempty workload and calibration distributions")
    return [
        round(
            (1 << 48) * (weight * Fraction(w, work_total) + (1 - weight) * Fraction(c, calib_total))
        )
        for w, c in zip(workload, calibration, strict=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="calibration .npy or server .json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--calibration", type=Path, help="blend this calibration NPY with server JSON"
    )
    parser.add_argument(
        "--workload-weight", type=Fraction, help="normalized workload share in [0,1]"
    )
    args = parser.parse_args()
    if (args.calibration is None) != (args.workload_weight is None):
        parser.error("--calibration and --workload-weight must be supplied together")
    is_npy = args.input.read_bytes()[:6] == b"\x93NUMPY"
    if is_npy:
        if args.calibration is not None:
            parser.error("blending requires server JSON as --input")
        counts, digest = read_counts(args.input)
        source = f"calibration NPY: {args.input} SHA256={digest}"
    else:
        counts, _, digest = read_server_profile(args.input)
        source = f"decode workload JSON: {args.input} SHA256={digest}"
        if args.calibration is not None:
            calibration, calib_digest = read_counts(args.calibration)
            counts = blend_counts(counts, calibration, args.workload_weight)
            source += (
                f"\n// Calibration NPY: {args.calibration} SHA256={calib_digest}"
                f"\n// Globally normalized blend: workload={args.workload_weight}, "
                f"calibration={1 - args.workload_weight}; rounded to 2^48 units."
            )
    rows = [
        "    " + ", ".join(f"{v}ULL" for v in counts[i : i + 8]) + ","
        for i in range(0, len(counts), 8)
    ]
    args.output.write_text(
        "#pragma once\n\n"
        "// Compiled routing counts; performance policy only, not model weights.\n"
        f"// Source: {source}\n"
        "// Generator: tools/convert/qwen3_8_flash_next/generate_expert_helper_profile.py\n"
        f"// Recipe (join wrapped lines with spaces): {shlex.join(['python3.11', '-I', *sys.argv])}\n"
        "#include <array>\n#include <cstdint>\n\n"
        "namespace ninfer::targets::qwen3_8_flash_next {\n"
        "inline constexpr std::array<std::uint64_t, 48 * 512> kExpertHelperCalibCounts = {\n"
        + "\n".join(rows)
        + "\n};\n} // namespace ninfer::targets::qwen3_8_flash_next\n"
    )
    print(
        f"counts={len(counts)} total={sum(counts)} nonzero={sum(v != 0 for v in counts)} sha256={digest}"
    )


if __name__ == "__main__":
    main()
