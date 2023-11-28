import torch.nn as nn
import torch

def fuse_batchnorm_conv(conv: nn.Conv2d, batchnorm: nn.BatchNorm2d) -> nn.Conv2d:

    output_channel = conv.weight.shape[0]
    conv_dims = conv.weight.dim()
    bn_redim = (output_channel,) + (1,) * (conv_dims - 1)

    bn_mean = batchnorm.running_mean.data
    bn_std = torch.sqrt(batchnorm.running_var.data + batchnorm.eps)
    bn_gamma = batchnorm.weight.data
 
    conv.weight.data = conv.weight.data * (bn_gamma / bn_std).reshape(bn_redim)
    conv_bias = conv.bias.data if conv.bias is not None else 0
    conv.bias = nn.Parameter(conv_bias + batchnorm.bias.data - bn_mean * bn_gamma / bn_std)

    return conv