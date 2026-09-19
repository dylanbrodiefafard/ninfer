"""Header grammar qualification, without manufacturing checkpoint weight payloads."""
import copy

import pytest

from tools.parity.qwen4.native_inventory import ITEM_BYTES, layer_grammar, validate_header, validate_layer
import math


def fixture(layer):
    offset = 0
    result = {}
    for name, (dtype, shape) in layer_grammar(layer).items():
        size = math.prod(shape) * ITEM_BYTES[dtype]
        result[name] = dict(dtype=dtype, shape=shape, data_offsets=[offset, offset + size])
        offset += size
    return result


@pytest.mark.parametrize("layer,expected", [(0, 1570383296), (47, 1557359104)])
def test_native_layer_inventory(layer, expected):
    header = fixture(layer)
    validate_header(header)
    assert validate_layer(layer, header)["payload_bytes"] == expected
    missing = copy.deepcopy(header)
    del missing[next(iter(missing))]
    with pytest.raises(ValueError, match="names differ"):
        validate_layer(layer, missing)
    wrong = copy.deepcopy(header)
    wrong[next(iter(wrong))]["dtype"] = "F32"
    with pytest.raises(ValueError, match="dtype/shape"):
        validate_layer(layer, wrong)


def test_source_spans_cannot_overlap_or_lie_about_dtype_bytes():
    header = fixture(3)
    names = list(header)
    header[names[1]]["data_offsets"][0] -= 2
    with pytest.raises(ValueError, match="invalid tensor span"):
        validate_header(header)
    header = fixture(3)
    header[names[1]]["data_offsets"] = [0, 81920]
    with pytest.raises(ValueError, match="overlapping"):
        validate_header(header)
