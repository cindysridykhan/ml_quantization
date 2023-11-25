import torch
from linear_quantization_utils import *
import pytest
from unittest import mock
import unittest

class TestLinearQuantizationUtils(unittest.TestCase):
    def test_get_quantized_range(self):
        bitwidth = 8
        qmin, qmax = get_quantized_range(bitwidth)
        assert qmin == -128
        assert qmax == 127

    @mock.patch('linear_quantization_utils.get_quantized_range', return_value=(-4, 3))
    def test_get_scale_and_zero_per_tensor(self, m_get_quantized_range):
        fp_tensor = torch.tensor([
            [-2, 4],
            [1, 0.5]
        ])
        scale, zero_point = get_scale_and_zero_per_tensor(fp_tensor, bitwidth=None)
        assert round(scale, 3) == 0.857
        assert zero_point == -2
        assert type(zero_point) == int

        fp_tensor = torch.tensor([
            [-2, -1],
        ])
        scale, zero_point = get_scale_and_zero_per_tensor(fp_tensor, bitwidth=None)
        assert round(scale, 3) == 0.143
        assert zero_point == 3

        fp_tensor = torch.tensor([
            [2, 1],
        ])
        scale, zero_point = get_scale_and_zero_per_tensor(fp_tensor, bitwidth=None)
        assert round(scale, 3) == 0.143
        assert zero_point == -4

    @mock.patch('linear_quantization_utils.get_quantized_range', return_value=(-4, 3))
    def test_linear_quantize_tensor_with_float_scale(self, m_get_quantized_range):
        fp_tensor = torch.tensor([-1.5, 4.2, 150, 800, -800])
        scale = 42.857
        zero_point = -2
        expected = torch.tensor([-2, -2, 2, 3, -4], dtype=torch.int8)
        output = linear_quantize_tensor(fp_tensor, None, scale, zero_point)
        assert torch.equal(expected, output)

    @mock.patch('linear_quantization_utils.get_quantized_range', return_value=(-8, 7))
    def test_linear_quantize_tensor_per_channel_symmetric_last_dim(self, m_get_quantized_range):
        fp_tensor = torch.tensor([
            [-1.5, 4.2, 150, 800, -800],
            [1, 4.2, 1, 1, 1]
            ])
        scale = torch.tensor([1, 2, 3, 4, 5])
        expected = torch.tensor([
            [-2, 2, 7, 7, -8],
            [1, 2, 0, 0, 0]
            ], dtype=torch.int8)
        output = linear_quantize_tensor(fp_tensor, None, scale, zero_point=0)
        assert torch.equal(expected, output)

    @mock.patch('linear_quantization_utils.get_quantized_range', return_value=(-8, 7))
    def test_linear_quantize_tensor_per_channel_symmetric_first_dim(self, m_get_quantized_range):
        fp_tensor = torch.tensor([
            [-1.8, 4.2, 8, -8],
            [-1.8, 4.2, 8, -8]
            ])
        scale = torch.tensor([[1], [2]])
        expected = torch.tensor([
            [-2, 4, 7, -8],
            [-1, 2, 4, -4]
            ], dtype=torch.int8)
        output = linear_quantize_tensor(fp_tensor, None, scale, zero_point=0)
        assert torch.equal(expected, output)

    @mock.patch('linear_quantization_utils.get_quantized_range', return_value=(-4, 3))
    def test_get_absmax_quantization_scale(self, m_get_quantized_range):
        symmetric_fp_tensor = torch.tensor([-10, 8, 0])
        assert get_absmax_quantization_scale_per_tensor(symmetric_fp_tensor, None) == 10/3

    @mock.patch('linear_quantization_utils.linear_quantize_tensor', return_value=None)
    @mock.patch(
            'linear_quantization_utils.get_absmax_quantization_scale_per_tensor',
            side_effect=[2, 2]
            )
    def test_linear_quantize_per_channel_symmetric(
        self,
        m_get_absmax_quantization_scale_per_tensor,
        m_linear_quantize_tensor
        ):
        fp_tensor = torch.tensor([
            [-1.5, 4.2, 150, 800, -800],
            [1, 4.2, 1, 1, 1]
            ])
        _, scale, zero_point = linear_quantize_per_channel_symmetric(fp_tensor, bitwidth=None, channel_dim=0)
        assert torch.equal(scale, torch.tensor([[2], [2]]))

