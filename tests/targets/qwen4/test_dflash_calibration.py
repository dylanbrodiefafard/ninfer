"""Exact-format and fitting-objective checks for the offline DFlash candidate."""
import torch

from tools.artifact.layouts import encode_nvfp4
from tools.convert.common.nvfp4_quantize import quantize_nvfp4_matrix
from tools.parity.qwen4.native_dflash_calibrate import calibrate_matrix, nearest_e2m1
from tools.reference.qwen4.dflash import decode_nvfp4_weight, decode_nvfp4_words_exact


def test_signed_nearest_even_midpoints():
    values = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], dtype=torch.float64)
    expected = torch.tensor([0,2,2,4,4,6,6], dtype=torch.uint8)
    assert torch.equal(nearest_e2m1(values), expected)
    assert torch.equal(nearest_e2m1(-values), expected | 8)
    assert nearest_e2m1(torch.tensor([-0.])).item() == 8


def test_fitting_preserves_divisor_and_improves_independent_objective():
    generator = torch.Generator().manual_seed(918)
    source = torch.randn(128,64,generator=generator).bfloat16()
    baseline = quantize_nvfp4_matrix(source)
    old = decode_nvfp4_weight(encode_nvfp4(*baseline, source.shape), source.shape)
    fitted_codes = []
    for moment in (torch.ones(64), torch.logspace(-3,3,64)):
        codes, scales, divisor, stats = calibrate_matrix(source, moment)
        assert torch.equal(divisor.view(torch.int32), baseline[2].view(torch.int32))
        payload = encode_nvfp4(codes, scales, divisor, source.shape)
        exact = decode_nvfp4_words_exact(payload, source.shape)
        assert torch.equal(exact[0], codes) and torch.equal(exact[1], scales)
        assert torch.equal(exact[2].view(torch.int32), divisor.view(torch.int32))
        represented = decode_nvfp4_weight(payload, source.shape)
        new_loss = ((represented-source.double()).square()*moment.double()).sum()
        old_loss = ((old-source.double()).square()*moment.double()).sum()
        assert new_loss <= old_loss
        assert abs(float(new_loss)-stats["fitted_diagonal_objective"]) < 1e-9
        fitted_codes.append(codes)
    assert not torch.equal(*fitted_codes), "actual calibration moments must affect the candidate"


def test_zero_source_and_unobserved_columns_preserve_baseline():
    source = torch.zeros(128,64,dtype=torch.bfloat16)
    result = calibrate_matrix(source, torch.zeros(64))
    baseline = quantize_nvfp4_matrix(source)
    assert all(torch.equal(a,b) for a,b in zip(result[:3], baseline))
    assert result[3]["fitted_diagonal_objective"] == 0
