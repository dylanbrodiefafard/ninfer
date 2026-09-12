"""Check the source quantizer against an independent FP64 nearest-level oracle."""
import numpy as np
import pytest
import torch

from tools.convert.common.fp8_quantize import encode_source_fp8
from tools.artifact.layouts import decode_fp8_row_scaled_words


def test_source_quantization_rounding_and_zero_rows():
    words=np.arange(127,dtype=np.uint8)
    exponent=(words>>3).astype(int)
    levels=np.where(exponent==0,(words&7)/512.,
                    np.ldexp(1.+(words&7)/8.,exponent-7))
    # Exactly represented E4M3 values and halfway BF16 values exercise ties.
    row=np.concatenate((levels,(levels[:-1]+levels[1:])/2.,[-448.,-0.,448.]))
    x=torch.tensor(np.stack((row,-row,np.zeros_like(row))),dtype=torch.bfloat16)
    codes,scales=decode_fp8_row_scaled_words(encode_source_fp8(x),tuple(x.shape))
    assert scales.float().tolist()==[1.,1.,0.]
    represented=x.float().numpy().astype(np.float64)
    magnitude=np.abs(represented)
    hi=np.clip(np.searchsorted(levels,magnitude),1,126)
    lo=hi-1
    high_error=np.abs(levels[hi]-magnitude)
    low_error=np.abs(levels[lo]-magnitude)
    nearest=np.where((high_error<low_error)|((high_error==low_error)&((hi&1)==0)),hi,lo)
    expected=nearest.astype(np.uint8)|(np.signbit(represented).astype(np.uint8)<<7)
    assert np.array_equal(codes.numpy(),expected)


def test_source_quantization_rejects_nonfinite_and_unrepresentable_scale():
    for value in (float('nan'),float('inf')):
        with pytest.raises(ValueError,match='nonfinite'):
            encode_source_fp8(torch.tensor([[value]],dtype=torch.bfloat16))
    tiny=torch.tensor([[1]],dtype=torch.int16).view(torch.bfloat16)
    with pytest.raises(ValueError,match='underflow'):
        encode_source_fp8(tiny)
