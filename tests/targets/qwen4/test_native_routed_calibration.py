"""Exact native activation codec and train-only projection-fit checks."""
import unittest

import torch

from tools.parity.qwen4.native_routed_calibration import activation, select
from tools.reference.qwen4.common import linear


class RoutedCalibration(unittest.TestCase):
    def test_source_multiplier_rounding_and_signed_zero(self):
        x=torch.tensor([[6.,.25,.75,1.25,1.75,2.5,3.5,5.,-6.,-.25,-.75,-1.25,-1.75,-2.5,-3.5,-5.]],dtype=torch.bfloat16)
        expected=torch.tensor([[6.,0.,1.,1.,2.,2.,4.,4.,-6.,-0.,-1.,-1.,-2.,-2.,-4.,-4.]],dtype=torch.float64)
        self.assertTrue(torch.equal(activation(x,1.),expected))
        self.assertTrue(torch.equal(torch.signbit(activation(x,1.)),torch.signbit(expected)))
        self.assertTrue(torch.equal(activation(x*2,2.),expected*2))
        self.assertTrue(torch.equal(activation(torch.zeros_like(x),1.),torch.zeros_like(expected)))

    def test_fit_cannot_worsen_training_objective(self):
        generator=torch.Generator().manual_seed(311)
        x=torch.randn(12,32,generator=generator).bfloat16()
        weight=torch.randn(16,32,generator=generator).double()
        source=.002
        fitted=select(x,weight,source)
        ideal=linear(x,weight)
        loss=lambda multiplier: float(((linear(activation(x,multiplier),weight)-ideal)**2).sum())
        self.assertLessEqual(loss(fitted),loss(source))


if __name__=="__main__":
    torch.set_num_threads(2)
    unittest.main()
