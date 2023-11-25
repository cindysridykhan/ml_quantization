import pytest
from unittest import mock
import torch
import torch.nn as nn
from calibration import *

class TestCalibration:
    def test_get_input_output_calibration_per_tensor(self):
        model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 2))
        calibration_dataset = torch.ones((4, 2))
        expected_input_calibration = {
            'layer0': {
                'scale': 1.4,
                'zero_point': 1
            },
            'layer1': {
                'scale': 1.4,
                'zero_point': 1
            },
        }

        expected_output_calibration = {
            'layer0': {
                'scale': 1.4,
                'zero_point': 1
            },
            'layer1': {
                'scale': 1.4,
                'zero_point': 1
            },
        }
        # TODO