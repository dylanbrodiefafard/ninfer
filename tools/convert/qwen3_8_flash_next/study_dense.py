"""Reader for the dense-format study's quantized payloads (recipe `fallback_mid`).

The study (`/home/battlefront/ai/study`, `results/dense/REPORT.md` and `fnstudy/dense.py`
`save_record`) stores one non-pickle NPZ record per quantized matrix and method at
`quant/dense/{layer}/{hf_key}/{method}.npz` (the head at `head/lm_head/`), with members

- `q`: uint8 codes, row-major. FP8 records hold one E4M3FN word per weight `[M,N]`; NVFP4 records
  two E2M1 nibbles per byte `[M,N/2]`, column `2j` in the low nibble of byte `j`;
- `scale`: FP8 records the BF16 row multiplier words as uint16 `[M,1]`; NVFP4 records one E4M3FN
  word per 16 consecutive columns `[M,N/16]`;
- `global_scale`: FP32; 1.0 for FP8, the NVFP4 matrix multiplier (`amax / 2688`) otherwise;
- `method`, `damping`: the encoding method and its GPTQ damping.

A weight decodes to `E4M3FN(q) * BF16(scale)` (FP8) or `E2M1(q) * E4M3FN(scale) * global_scale`
(NVFP4). `results/dense/artifact_hashes.json` fingerprints every record and `summary.json`
names the recommended recipe and each matrix's damping; both are checked before a record's
words are used. The attention-input parents (`quant/dense/attn_parent/{layer}.npz`, specified in
`results/dense/ATTN_PARENT.md` and fingerprinted in `attn_parent_validation.json`) are complete
NVFP4 parents under the ninfer convention: packed codes `q`, natural E4M3FN block scales `scale`,
and one FP32 `weight_divisor` (`value = E2M1 * E4M3FN / d_w`).
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tools.artifact.layouts import ExpertMatrixWords

from .inventory import STUDY_FP8_METHOD, STUDY_NVFP4_METHOD, STUDY_RECIPE

# Study family -> method of recipe `fallback_mid`, as `summary.json` records it.
_RECIPE_METHODS = {
    "gdn_in": STUDY_FP8_METHOD,
    "gdn_out": STUDY_FP8_METHOD,
    "attn_in": STUDY_NVFP4_METHOD,
    "attn_out": STUDY_FP8_METHOD,
    "shared": STUDY_NVFP4_METHOD,
    "head": STUDY_FP8_METHOD,
    "embed": "bf16",
    "ple": "bf16",
}
_FP32_ONE = 0x3F800000


@dataclass(frozen=True, slots=True)
class StudyRecord:
    path: str
    method: str
    codes: np.ndarray
    scales: np.ndarray
    global_word: int
    damping: float


class DenseStudy:
    """Validated access to the study's dense records under `quant/dense`."""

    def __init__(self, records_dir: str | Path, results_dir: str | Path) -> None:
        self.records_dir = Path(records_dir)
        self.results_dir = Path(results_dir)
        hashes = json.loads((self.results_dir / "artifact_hashes.json").read_text())
        if hashes.get("status") != "complete":
            raise ValueError("artifact_hashes.json is not complete")
        self._hashes = {
            path.removeprefix("quant/dense/"): item for path, item in hashes["records"].items()
        }
        summary = json.loads((self.results_dir / "summary.json").read_text())
        recipe = summary["recommended_recipe"]
        for family, method in _RECIPE_METHODS.items():
            if recipe[family]["method"] != method:
                raise ValueError(
                    f"summary.json recommends {recipe[family]['method']} for {family}, "
                    f"not the {STUDY_RECIPE} method {method}"
                )
        self._damping: dict[str, float] = {}
        for family in _RECIPE_METHODS:
            self._damping.update(recipe[family].get("damping_by_tensor", {}))
        # Hyper-connection down/up records (`HC.md`): fingerprints and the selected damping in
        # hc_validation.json, which must agree with the calibration's selection.
        hc = json.loads((self.results_dir / "hc_validation.json").read_text())
        hc_calibration = json.loads((self.results_dir / "hc_calibration.json").read_text())
        if hc.get("status") != "complete" or hc_calibration.get("status") != "complete":
            raise ValueError("hc_validation.json or hc_calibration.json is not complete")
        self._hc: dict[str, dict[str, object]] = {}
        for key, item in hc["records"].items():
            selected = hc_calibration["matrices"][key]
            if selected["selected_damping"] != item["damping"] or selected["path"] != item["path"]:
                raise ValueError(f"{key}: HC validation and calibration disagree")
            self._hc[item["path"].removeprefix("quant/dense/")] = item
        parents = json.loads((self.results_dir / "attn_parent_validation.json").read_text())
        if parents.get("status") != "complete":
            raise ValueError("attn_parent_validation.json is not complete")
        self.parent_hashes: dict[str, dict[str, object]] = parents["records"]

    def damping(self, key: str) -> float:
        """The recommended GPTQ damping of `{layer}:{hf_key}` from `summary.json`."""

        if key not in self._damping:
            raise ValueError(f"no recommended damping for {key}")
        return self._damping[key]

    def load(self, relative: str) -> StudyRecord:
        """Read, fingerprint-check, and parse one per-matrix record.

        Paths are `{layer}/{key}/{method}.npz` (fingerprints in artifact_hashes.json, damping in
        summary.json) or `hc/{layer}/{key}/{method}.npz` (both in hc_validation.json).
        """

        if relative.startswith("hc/"):
            _, layer, key, file_name = relative.split("/")
            expected = self._hc.get(relative)
            recommended = None if expected is None else float(expected["damping"])
            manifest = "hc_validation.json"
        else:
            layer, key, file_name = relative.split("/")
            expected = self._hashes.get(relative)
            recommended = None if expected is None else self.damping(f"{layer}:{key}")
            manifest = "artifact_hashes.json"
        method = file_name.removesuffix(".npz")
        if expected is None:
            raise ValueError(f"{relative}: not fingerprinted by {manifest}")
        data = (self.records_dir / relative).read_bytes()
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError(f"{relative}: bytes differ from {manifest}")
        with np.load(io.BytesIO(data), allow_pickle=False) as record:
            if str(record["method"]) != method:
                raise ValueError(f"{relative}: stores method {record['method']}")
            damping = float(record["damping"])
            if damping != recommended:
                raise ValueError(f"{relative}: damping {damping} is not the recommended one")
            global_scale = record["global_scale"]
            if global_scale.dtype != np.float32 or global_scale.shape != ():
                raise ValueError(f"{relative}: global_scale is not one FP32 word")
            return StudyRecord(
                relative,
                method,
                record["q"].copy(),
                record["scale"].copy(),
                int(global_scale.view(np.uint32)),
                damping,
            )


def fp8_words(record: StudyRecord, rows: int, columns: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact E4M3FN codes uint8 `[rows, columns]` and BF16 row scales `[rows]` of an FP8 record."""

    if record.method != STUDY_FP8_METHOD or record.global_word != _FP32_ONE:
        raise ValueError(f"{record.path}: not an FP8 row record with unit global scale")
    if record.codes.dtype != np.uint8 or record.codes.shape != (rows, columns):
        raise ValueError(f"{record.path}: codes are not uint8 {(rows, columns)}")
    if record.scales.dtype != np.uint16 or record.scales.shape != (rows, 1):
        raise ValueError(f"{record.path}: scales are not uint16 {(rows, 1)}")
    scales = torch.from_numpy(record.scales.reshape(rows).view(np.int16).copy())
    return torch.from_numpy(record.codes), scales.view(torch.bfloat16)


def nvfp4_words(record: StudyRecord, rows: int, columns: int) -> tuple[ExpertMatrixWords, int]:
    """Natural-order E2M1 codes, E4M3FN scales and the FP32 multiplier word of an NVFP4 record."""

    if record.method != STUDY_NVFP4_METHOD:
        raise ValueError(f"{record.path}: not an NVFP4 record")
    if record.codes.dtype != np.uint8 or record.codes.shape != (rows, columns // 2):
        raise ValueError(f"{record.path}: codes are not uint8 {(rows, columns // 2)}")
    if record.scales.dtype != np.uint8 or record.scales.shape != (rows, columns // 16):
        raise ValueError(f"{record.path}: scales are not uint8 {(rows, columns // 16)}")
    multiplier = struct.unpack("<f", struct.pack("<I", record.global_word))[0]
    if not (np.isfinite(multiplier) and multiplier > 0):
        raise ValueError(f"{record.path}: matrix multiplier must be finite and positive")
    codes = np.empty((rows, columns), dtype=np.uint8)
    codes[:, 0::2] = record.codes & 0x0F
    codes[:, 1::2] = record.codes >> 4
    words = ExpertMatrixWords(torch.from_numpy(codes), torch.from_numpy(record.scales))
    return words, record.global_word


# The attention-input parent records (`results/dense/ATTN_PARENT.md`): one NVFP4 `[13312,2560]`
# parent per QSA layer in checkpoint row order per part, with one FP32 weight divisor.
_PARENT_PARTS = ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight")
_PARENT_ROW_OFFSETS = (0, 12288, 12800, 13312)
_PARENT_SHAPE = (13312, 2560)


@dataclass(frozen=True, slots=True)
class AttentionParentRecord:
    path: str
    # Packed E2M1 codes uint8 [13312, 1280] (even K in the low nibble) and natural E4M3FN block
    # scales uint8 [13312, 160], rows in checkpoint order: q_proj (per-head query/gate
    # interleave), k_proj, v_proj.
    codes: np.ndarray
    scales: np.ndarray
    divisor_word: int


def load_attention_parent(study: DenseStudy, relative: str) -> AttentionParentRecord:
    """Read, fingerprint-check, and parse one attention-input parent record."""

    layer = relative.removeprefix("attn_parent/").removesuffix(".npz")
    data = (study.records_dir / relative).read_bytes()
    expected = study.parent_hashes.get(layer)
    if (
        expected is None
        or expected["path"] != "quant/dense/" + relative
        or len(data) != expected["bytes"]
        or hashlib.sha256(data).hexdigest() != expected["sha256"]
    ):
        raise ValueError(f"{relative}: bytes differ from attn_parent_validation.json")
    with np.load(io.BytesIO(data), allow_pickle=False) as record:
        metadata = (
            str(record["format"]),
            int(record["version"]),
            tuple(int(item) for item in record["shape"]),
            tuple(str(item) for item in record["parts"]),
            tuple(int(item) for item in record["row_offsets"]),
            tuple(str(item) for item in record["rounding"]),
        )
        if metadata != (
            "NVFP4",
            1,
            _PARENT_SHAPE,
            _PARENT_PARTS,
            _PARENT_ROW_OFFSETS,
            ("RTN", "RTN", "GPTQ"),
        ):
            raise ValueError(f"{relative}: unexpected parent metadata {metadata}")
        damping = record["damping"]
        value_damping = study.damping(f"{layer}:self_attn.v_proj.weight")
        if damping.tolist() != [0.0, 0.0, np.float32(value_damping)]:
            raise ValueError(f"{relative}: damping {damping.tolist()} is not the recommended one")
        divisor = record["weight_divisor"]
        if divisor.dtype != np.float32 or divisor.shape != ():
            raise ValueError(f"{relative}: weight_divisor is not one FP32 word")
        codes, scales = record["q"].copy(), record["scale"].copy()
    rows, columns = _PARENT_SHAPE
    if codes.dtype != np.uint8 or codes.shape != (rows, columns // 2):
        raise ValueError(f"{relative}: codes are not uint8 {(rows, columns // 2)}")
    if scales.dtype != np.uint8 or scales.shape != (rows, columns // 16):
        raise ValueError(f"{relative}: scales are not uint8 {(rows, columns // 16)}")
    return AttentionParentRecord(relative, codes, scales, int(divisor.view(np.uint32)))


def check_record(study: DenseStudy, relative: str) -> None:
    """Fingerprint- and metadata-check one record of either kind before any output exists."""

    if relative.startswith("attn_parent/"):
        load_attention_parent(study, relative)
    else:
        study.load(relative)
