import torch
import torch.nn as nn

from src.quantization.fuse_layer import fuse_batchnorm_conv

class TestFuseLayers:
    def test_fuse_batchnorm_conv(self,):
        in_channel, out_channel, k, k = 1, 2, 3, 3
        bsz, w, h = 1, 6, 6
        X = torch.arange(bsz*in_channel*w*h).view((bsz, in_channel, w, h))*.2
        conv2d = nn.Conv2d(in_channel, out_channel, (k, k))
        batchnorm2d = nn.BatchNorm2d(out_channel)
        # batchnorm2d.weight.data = torch.arange(out_channel)*.5
        # batchnorm2d.bias.data = -torch.arange(out_channel)*.5
        batchnorm2d.training = False
        expected_output = batchnorm2d(conv2d(X))
        fused = fuse_batchnorm_conv(conv2d, batchnorm2d)
        output = fused(X)
        assert torch.allclose(output, expected_output)
