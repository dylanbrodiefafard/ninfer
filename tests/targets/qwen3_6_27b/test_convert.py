from __future__ import annotations

import json
import os
from pathlib import Path

from tools.convert.qwen3_6_27b import convert

SOURCE_MODEL = Path(os.environ.get("NINFER_QWEN3_8_27B_BF16_DIR", "/models/qwen3.8-27b-bf16"))


def test_source_config_passes_the_registered_geometry():
    config = json.loads((SOURCE_MODEL / "config.json").read_text())

    summary = convert.validate_config(config)
    assert summary["mtp_num_hidden_layers"] == 1
    assert summary["text"]["mtp_num_hidden_layers"] == 1
    assert summary["layer_types"]["full_attention_layers"] == list(range(3, 64, 4))
