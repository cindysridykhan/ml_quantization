import pytest
import unittest
import torch
import torch.nn as nn
from src.quantization.calibration import *

class TestCalibration(unittest.TestCase):
    def setUp(self):
        batch_size = 4
        in_dim = 2
        hid_dim = 3
        out_dim = 2
        X = torch.tensor([
            [-3., -3.],
            [ 2.,  2.],
            [ 2.,  2.],
            [-3.,  2.]]).float()

        model = nn.ModuleDict({
            'layer0':nn.Linear(in_dim, hid_dim),
            'layer1': nn.Linear(hid_dim, out_dim)
            })
        model.forward = lambda x: model['layer1'](nn.ReLU()(model['layer0'](x)))

        w1 = torch.tensor(
            [[ 0.,  1.],
            [-2.,  0.],
            [ 1.,  0.]], dtype=torch.float)
        b1 = torch.tensor([ 0., -2., -1.], dtype=torch.float)
        w2 = torch.tensor(
            [[ 1.,  0.,  1.],
            [-2.,  0.,  0.]], dtype=torch.float)
        b2 = torch.tensor([2., 2.], dtype=torch.float)

        model.layer0.weight = nn.Parameter(w1)
        model.layer0.bias = nn.Parameter(b1)
        model.layer1.weight = nn.Parameter(w2)
        model.layer1.bias = nn.Parameter(b2)
        self.model = model
        self.X = X

    def test_get_input_output_calibration_per_tensor_min_max(self):
        input_calibration, output_calibration = get_input_output_calibration_per_tensor(
            model=self.model,
            calibration_dataset=self.X
        )
        expected_input_calibration = {
            'layer0': {
                'fp_min': -3.0,
                'fp_max': 2.0
            },
            'layer1': {
                'fp_min': 0.,
                'fp_max': 4.0
            },
        }

        expected_output_calibration = {
            'layer0': {
                'fp_min': -6.0,
                'fp_max': 4.0
            },
            'layer1': {
                'fp_min': -2.0,
                'fp_max': 5.0
            },
        }
        self.assertDictEqual(input_calibration, expected_input_calibration)
        self.assertDictEqual(output_calibration, expected_output_calibration)
