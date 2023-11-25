import torch
import torch. nn as nn
from tqdm import tqdm

@torch.inference_mode()
def evaluate(
  model: nn.Module,
  dataloader: torch.utils.data.DataLoader,
  extra_preprocess = None
) -> float:
  model.eval()

  num_samples = 0
  num_correct = 0

  for inputs, targets in tqdm(dataloader, desc="eval", leave=False):
    inputs = inputs.cuda()
    if extra_preprocess is not None:
        for preprocess in extra_preprocess:
            inputs = preprocess(inputs)

    targets = targets.cuda()
    outputs = model(inputs)
    outputs = outputs.argmax(dim=1)
    num_samples += targets.size(0)
    num_correct += (outputs == targets).sum()

  return (num_correct / num_samples * 100).item()

def get_model_size_gb(model):
    param_size = 0
    for n, p in model.named_parameters():
        param_size += p.nelement() * p.element_size() 

    for n, p in model.named_buffers():
        param_size += p.nelement() * p.element_size() 
    
    return param_size*1e-9

# def get_model_flops(model, inputs):
#     num_macs = profile_macs(model, inputs)
#     return num_macs