"""Exact codec and calibration-objective checks for bounded source experiments."""
import unittest
import tempfile
from pathlib import Path

import torch

from tools.convert.common.nvfp4_quantize import quantize_nvfp4_matrix
from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter, TensorSpec
from tools.artifact.layouts import encode_nvfp4, decode_nvfp4_words, encode_direct
from tools.parity.qwen4.native_projection_candidates import codes, decode, encode_weight, screen, SOURCE_SCREEN


class ProjectionCandidates(unittest.TestCase):
    def test_source_screen_cannot_hide_one_token_failure(self):
        reference=torch.ones(100,4,dtype=torch.float64)
        actual=reference.clone();actual[0]*=1.03
        result=screen(actual,reference,SOURCE_SCREEN)
        self.assertLess(result['relative_l2'],SOURCE_SCREEN[0])
        self.assertFalse(result['passed'])
        self.assertEqual(result['gross_token_failures'],1)

    def test_activation_midpoint_rounding(self):
        x=torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5.,-.75])
        self.assertEqual(codes(x,ties_even=True).tolist(),[0,2,2,4,4,6,6,10])
        self.assertEqual(codes(x).tolist(),[0,1,2,3,4,5,6,9])

    def test_baseline_exact_and_calibration_objective(self):
        generator=torch.Generator().manual_seed(109)
        weight=torch.randn(128,64,generator=generator).bfloat16()
        energy=torch.linspace(.01,4,64,dtype=torch.float64)
        actual=encode_weight(weight,energy,False)
        expected=quantize_nvfp4_matrix(weight)
        for a,b in zip(actual,expected):self.assertTrue(torch.equal(a,b))
        fitted=encode_weight(weight,energy,True)
        baseline=decode(*actual);candidate=decode(*fitted)
        a=((baseline-weight.double()).square()*energy).reshape(128,4,16).sum(-1)
        b=((candidate-weight.double()).square()*energy).reshape(128,4,16).sum(-1)
        self.assertTrue(bool((b<=a).all()))
        self.assertLess(float(b.sum()),float(a.sum()))

    def test_independent_signed_decode(self):
        packed=torch.tensor([[0x10,0x32,0x54,0x76,0x98,0xba,0xdc,0xfe]],dtype=torch.uint8)
        # E4M3 word 56 is exactly 1; FP32 divisor 2 halves all codes.
        actual=decode(packed,torch.tensor([[56]],dtype=torch.uint8),torch.tensor(2.))
        expected=torch.tensor([[0,.25,.5,.75,1,1.5,2,3,0,-.25,-.5,-.75,-1,-1.5,-2,-3]],dtype=torch.float64)
        self.assertTrue(torch.equal(actual,expected))

    def test_candidate_existing_artifact_format(self):
        weight=torch.linspace(-2,3,128*64).reshape(128,64).bfloat16()
        packed,scales,divisor=encode_weight(weight,torch.ones(64),True)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'candidate.ninfer'
            spec=TensorSpec('weight',(128,64),'NVFP4','blockscale-k16-m128x4-v1')
            activation_divisor=torch.tensor(12.375,dtype=torch.float32)
            with ArtifactWriter(path,ArtifactIdentity('qwen4/native-projection-candidate','nvfp4_diagonal_calibrated'),
                [spec,TensorSpec('input_scale_divisor',(),'FP32','contiguous-le-v1')]) as writer:
                writer.write('weight',encode_nvfp4(packed,scales,divisor,(128,64)))
                writer.write('input_scale_divisor',encode_direct(activation_divisor,'FP32'))
            with Artifact(path) as artifact:
                actual=decode_nvfp4_words(artifact.payload('weight'),(128,64))
                for a,b in zip(actual,(packed,scales,divisor)):
                    self.assertTrue(torch.equal(a.reshape(-1),b.reshape(-1)))
                self.assertEqual(bytes(artifact.payload('input_scale_divisor')),b'\x00\x00FA')


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
