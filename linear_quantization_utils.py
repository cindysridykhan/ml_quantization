from typing import Union
import torch

def get_quantized_range(bitwidth: int) -> tuple[float]:
    qmin = -(1 << (bitwidth - 1))
    qmax = (1 << (bitwidth - 1)) - 1
    return qmin, qmax


def get_scale_and_zero_per_tensor(fp_tensor: torch.Tensor, bitwidth: int) -> tuple[float, int]:
    quantized_min, quantized_max = get_quantized_range(bitwidth)
    fp_max = fp_tensor.max().item()
    fp_min = fp_tensor.min().item()

    scale = (fp_max - fp_min) / (quantized_max - quantized_min)
    zero_point = int(round(quantized_min - fp_min/scale))

    zero_point = max(zero_point, quantized_min)
    zero_point = min(zero_point, quantized_max)
    return scale, zero_point

def linear_quantize_tensor(
        fp_tensor: torch.Tensor,
        bitwidth: int,
        scale: Union[torch.Tensor, float],
        zero_point: int,
        dtype=torch.int8
        ) -> torch.Tensor:
    # TODO: handle asymmetric per channel? (zero_point per channel)
    quantized_min, quantized_max = get_quantized_range(bitwidth)
    quantized_tensor = (fp_tensor / scale).round() + zero_point
    quantized_tensor = quantized_tensor.clamp(quantized_min, quantized_max)
    return quantized_tensor.to(dtype)


def get_absmax_quantization_scale_per_tensor(symmetric_fp_tensor: torch.Tensor, bitwidth: int) -> float:
    fp_max = max(symmetric_fp_tensor.abs().max().item(), 5e-7)
    _, quantized_max = get_quantized_range(bitwidth)
    return fp_max / quantized_max

def linear_quantize_per_channel_symmetric(
        symmetric_fp_tensor: torch.Tensor, 
        bitwidth: int, 
        channel_dim: int=0
        ) -> tuple[torch.Tensor, torch.Tensor, int]:
    # TODO: handle asymmetric as well?
    num_output_channel = symmetric_fp_tensor.shape[channel_dim]
    scale = torch.zeros((num_output_channel,), device=symmetric_fp_tensor.device)
    for channel in range(num_output_channel):
        scale[channel] = get_absmax_quantization_scale_per_tensor(
            symmetric_fp_tensor=torch.select(symmetric_fp_tensor, channel_dim, channel),
            bitwidth=bitwidth
            )
    scale_shape = [1] * symmetric_fp_tensor.dim()
    scale_shape[channel_dim] = -1
    scale = scale.view(scale_shape)
    quantized_tensor = linear_quantize_tensor(symmetric_fp_tensor, bitwidth, scale, zero_point=0)
    return quantized_tensor, scale, 0

def linear_quantize_per_tensor_asymmetric(
        fp_tensor: torch.Tensor, bitwidth: int, dtype=torch.int8
        ) -> torch.Tensor:
    scale, zero = get_scale_and_zero_per_tensor(fp_tensor=fp_tensor, bitwidth=bitwidth)
    q_tensor = linear_quantize_tensor(
        fp_tensor=fp_tensor, bitwidth=bitwidth, scale=scale, zero_point=zero, dtype=dtype
        )
    return q_tensor, scale, zero