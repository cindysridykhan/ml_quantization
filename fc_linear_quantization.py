# %%

from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from linear_quantization_utils import get_quantized_range, linear_quantize_tensor, linear_quantize_per_tensor_asymmetric, linear_quantize_per_channel_symmetric

def compute_shifted_q_bias(
        bias: torch.Tensor, 
        weight_scale: Union[torch.Tensor, float], 
        input_scale: float,
        quantized_weight: torch.Tensor,
        input_zero_point: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    shifted_q_bias = q_bias - Z_X * q_W@1 (where 1 = torch.ones_like(q_X))
    '''
    q_bias, _, _ = compute_q_bias(bias=bias, weight_scale=weight_scale, input_scale=input_scale)
    offset = input_zero_point * quantized_weight.sum(dim=1).to(torch.int32)
    shifted_q_bias = q_bias - offset
    return shifted_q_bias, q_bias

def compute_q_bias(
        bias: torch.Tensor, 
        weight_scale: Union[torch.Tensor, float], 
        input_scale: float,
        ) -> tuple[torch.Tensor, float, int]:
    if isinstance(weight_scale, torch.Tensor):
        weight_scale = weight_scale.view(-1)
    bias_scale = weight_scale * input_scale
    q_bias = linear_quantize_tensor(
        fp_tensor=bias, bitwidth=32, scale=bias_scale, zero_point=0
    ) 
    return q_bias, bias_scale, 0

def get_matmul_quantized_output(
        q_input: torch.Tensor, q_weight: torch.Tensor,
        shifted_q_bias: torch.Tensor,
        weight_scale: torch.Tensor, 
        input_scale: float,
        output_scale: float, output_zero_point: float,
        bitwidth: int
):
    '''
    weight: per output channel granularity
    input and output: per tensor granularity
    y = W @ X + b
    q_y = (S_W * S_X / S_y) * (q_W @ q_X + shifted_q_b) + Z_y 
    '''
    if 'cpu' in q_input.device.type:
        # use 32-b MAC for simplicity
        quantized_output = F.linear(q_input.to(torch.int32), q_weight.to(torch.int32), shifted_q_bias.to(torch.int32))

    else:
        # current version pytorch does not yet support integer-based linear() on GPUs
        quantized_output = F.linear(q_input.float(), q_weight.float(), shifted_q_bias.float())
    
    scale = (weight_scale.view(-1) * input_scale / output_scale)
    quantized_output = scale * quantized_output + output_zero_point
    qmin, qmax = get_quantized_range(bitwidth)
    quantized_output = quantized_output.round().clamp(qmin, qmax).to(torch.int8)
    return quantized_output


def dequantize(q_tensor: torch.Tensor, scale: Union[torch.Tensor, float], zero_point: int) -> torch.Tensor:
    return (q_tensor.float() - zero_point) * scale


class QuantizedLinear(nn.Module):
    def __init__(
            self,
            linear: nn.Module,
            input_scale: float,
            input_zero_point: int,
            output_scale: float,
            output_zero_point: int,
            bitwidth: int=8
            ) -> None:
        super().__init__()
        self.bitwidth = bitwidth
        self.input_scale = input_scale
        self.input_zero_point = input_zero_point
        self.output_scale = output_scale
        self.output_zero_point = output_zero_point

        self.weight, self.weight_scale, _ = linear_quantize_per_channel_symmetric(
            symmetric_fp_tensor=linear.weight, bitwidth=self.bitwidth, channel_dim=0
            )
        self.bias, _ = compute_shifted_q_bias(
            bias=linear.bias, weight_scale=self.weight_scale, input_scale=self.input_scale,
            quantized_weight=self.weight, input_zero_point=self.input_zero_point
        )

    def forward(self, q_x):
        quantized_output = get_matmul_quantized_output(
            q_input=q_x,
            q_weight=self.weight, shifted_q_bias=self.bias,
            weight_scale=self.weight_scale, input_scale=self.input_scale,
            output_scale=self.output_scale, output_zero_point=self.output_zero_point,
            bitwidth=self.bitwidth
        )
        return quantized_output
