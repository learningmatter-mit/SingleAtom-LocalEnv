import torch
from typing import Dict, Union

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
        elif isinstance(inputs, Dict):
            self.load_state_dict(inputs)
        else:
            TypeError

    def norm(self, tensor):
        return (tensor - self.mean) / self.std
        # return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean
        # return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std, "max": self.max, "min": self.min}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]
        self.max = state_dict["max"]
        self.min = state_dict["min"]
