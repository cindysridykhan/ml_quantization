import pickle
import pytest
import unittest
from unittest import mock
import torch
from fc_linear_quantization import *
from linear_quantization_utils import *

@pytest.fixture()
def sample_W_X_b():
    # bsz = 100
    # in_dim = 500
    # out_dim = 10
    # bitwidth= 8
    # X = torch.randn((bsz, in_dim))
    # W = torch.randn((out_dim, in_dim))
    # b = torch.randn((out_dim,))
    # data = {'W': W, 'X':X, 'b':b}
    # with open('test_fc_linear_quantization_data.p', 'wb') as f:
    #     pickle.dump(data, f)

    with open('test_fc_linear_quantization_data.p', 'rb') as f:
        data = pickle.load(f)
    X = data['X']
    W = data['W']
    b = data['b']
    return W, X, b



class TestMatMulQuantization:
    @mock.patch('fc_linear_quantization.get_quantized_range', return_value=(-128, 127))
    def test_get_matmul_quantized_output(self, m_get_quantized_range):
        q_input = torch.tensor([
            [1, 2],
            [-2, 7],
            [0, 0],
            ], dtype=torch.int8)
        q_weight = torch.tensor([
            [1, -1],
            [-8, 7],
            [-2, 7],
            [-4, 3]
        ], dtype=torch.int8)

        weight_scale = torch.tensor([
            [10],
            [1.5],
            [2],
            [4]
        ])
        shifted_q_bias = torch.tensor([1, 1, 1, 1], dtype=torch.int8)

        input_scale = 2.3
        output_scale = 2.2
        output_zero_point = 3
        output = get_matmul_quantized_output(
            q_input=q_input, q_weight=q_weight, shifted_q_bias=shifted_q_bias,
            weight_scale=weight_scale, input_scale=input_scale, output_scale=output_scale, output_zero_point=output_zero_point, 
            bitwidth=8
        )
        expected = torch.tensor([
            [  3,  14,  30,  16],
            [-81, 106, 116, 127],
            [ 13,   5,   5,   7]], dtype=torch.int8)
        assert torch.equal(output, expected)


    def test_restore(self):
        q_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        zero_point = 1
        scale = torch.tensor([[0.5], [1]])
        output = dequantize(q_tensor, scale, zero_point)
        assert torch.equal(output, torch.tensor([[0, 0.5, 1], [3, 4, 5]]))


class TestFCIntegration:
    def test_qmatmul_close_to_matmul(self, sample_W_X_b):
        bitwidth = 8
        W, X, b = sample_W_X_b

        y = F.linear(X, W, b)
        output_scale, output_zero_point = get_scale_and_zero_per_tensor(y, bitwidth)

        q_X, input_scale, input_zero_point = linear_quantize_per_tensor_asymmetric(fp_tensor=X, bitwidth=bitwidth)
        q_W, weight_scale, _ = linear_quantize_per_channel_symmetric(symmetric_fp_tensor=W, bitwidth=bitwidth, channel_dim=0)
        shifted_q_bias, _ = compute_shifted_q_bias(
            bias=b, weight_scale=weight_scale, 
            input_scale=input_scale, quantized_weight=q_W, input_zero_point=input_zero_point
        )

        quantized_y = get_matmul_quantized_output(
            q_input=q_X, q_weight=q_W,
            shifted_q_bias=shifted_q_bias,
            weight_scale=weight_scale, input_scale=input_scale,
            output_scale=output_scale, output_zero_point=output_zero_point,
            bitwidth=bitwidth
        )
        restored_y = dequantize(quantized_y, output_scale, output_zero_point)
        rel_error = abs(y - restored_y)/abs(y)
        assert rel_error.quantile(0.9) <= 0.5


class TestQuantizedLinearIntegration:
    def test_forward(self, sample_W_X_b):
        bitwidth = 8
        W, X, b = sample_W_X_b
        y = F.linear(X, W, b)
        output_scale, output_zero_point = get_scale_and_zero_per_tensor(y, bitwidth)

        out_dim, in_dim = W.shape
        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data = W
        linear.bias.data = b
        q_X, input_scale, input_zero_point = linear_quantize_per_tensor_asymmetric(X, bitwidth=bitwidth, dtype=torch.int8)

        q_linear = QuantizedLinear(
            linear=linear,
            input_scale=input_scale, input_zero_point=input_zero_point,
            output_scale=output_scale, output_zero_point=output_zero_point,
            bitwidth=bitwidth
        )
        assert q_linear.weight.dtype == torch.int8
        assert q_linear.bias.dtype == torch.int32

        q_y = q_linear(q_X)
        
        restored_y = dequantize(q_y, output_scale, output_zero_point)
        rel_error = abs(y - restored_y)/abs(y)
        assert rel_error.quantile(0.9) <= 0.5


