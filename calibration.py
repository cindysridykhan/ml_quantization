
def _register_calibration_hooks():
    pass

def _remove_hooks(hooks):
    pass

def get_input_output_calibration_per_tensor(
        model: nn.Module,
        calibration_dataset,
        #calibration_method: enum min_max, KL, percentile
        ) -> tuple[dict, dict]:
    input_calibration = {}
    output_calibration = {}
    hooks = _register_calibration_hooks(model)
    model(calibration_dataset)
    _remove_hooks(hooks)
    return input_calibration, output_calibration
