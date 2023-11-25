from functools import partial
import torch
import torch.nn as nn
from enum import Enum


def _min_max_hook(
        model: nn.Module,
        x: torch.Tensor, 
        y: torch.Tensor, 
        input_calibration: dict,
        output_calibration: dict,
        module_name: str
        ) -> None:
    if isinstance(x, tuple):
        x = x[0]
    input_calibration[module_name] = {
        'fp_min': x.min().item(),
        'fp_max': x.max().item()
    }
    output_calibration[module_name] = {
        'fp_min': y.min().item(),
        'fp_max': y.max().item()
    }

class CalibrationMethod(Enum):
    min_max = _min_max_hook

def _register_calibration_hooks(
        model: nn.Module,
        input_calibration: dict,
        output_calibration: dict,
        hook_fn: callable
) -> list[torch.utils.hooks.RemovableHandle]:
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            hook = layer.register_forward_hook(
                partial(
                    hook_fn, 
                    input_calibration=input_calibration,
                    output_calibration=output_calibration,
                    module_name=name
                    )
            )
            hooks.append(hook)
    return hooks


def _remove_hooks(hooks: list[torch.utils.hooks.RemovableHandle]) -> None:
    for hook in hooks:
        hook.remove()

@torch.inference_mode()
def get_input_output_calibration_per_tensor(
        model: nn.Module,
        calibration_dataset: torch.Tensor,
        calibration_method: CalibrationMethod=CalibrationMethod.min_max
        ) -> tuple[dict, dict]:
    input_calibration = {}
    output_calibration = {}
    hooks = _register_calibration_hooks(
        model,
        input_calibration=input_calibration,
        output_calibration=output_calibration,
        hook_fn=calibration_method
        )
    model(calibration_dataset)
    _remove_hooks(hooks)
    return input_calibration, output_calibration
