import torch
from typing import Dict, Union


def inference(model, data, output_key, normalizer=None, device="cpu"):
    """Inference

    Args:
            model (Torch.nn.Module): torch model
            data (Torch.nn.Data): torch Data
            output_key (str): output key
            normalizer (Dict): Information for normalization

    Returns:
            out (torch.Tensor): inference tensor
    """
    model.to(device)
    out = model(data)[output_key].detach()
    if normalizer is None:
        return out
    else:
        normalizer = Normalizer(normalizer)
        out = normalizer.denorm(out)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


TESNOR = torch.Tensor


class Normalizer:
    """Normalize a Tensor and restore it later."""

    def __init__(self, inputs: Union[TESNOR, Dict]):
        """tensor is taken as a sample to calculate the mean and std"""
        if isinstance(inputs, TESNOR):
            self.mean = torch.mean(inputs)
            self.std = torch.std(inputs)
            self.max = torch.max(inputs)
            self.min = torch.min(inputs)
            self.sum = torch.sum(inputs)

        elif isinstance(inputs, Dict):
            self.load_state_dict(inputs)

        else:
            TypeError

    def norm(self, tensor):
        return (tensor - self.mean) / self.std
        # return (tensor - self.mean) / self.std

    def norm_to_unity(self, tensor):
        return tensor / self.sum

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean
        # return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {
            "mean": self.mean,
            "std": self.std,
            "max": self.max,
            "min": self.min,
            "sum": self.sum,
        }

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]
        self.max = state_dict["max"]
        self.min = state_dict["min"]
        self.sum = state_dict["sum"]
