from typing import Dict, Union

import numpy as np
import torch

from persite_painn.utils.cuda import batch_to


def inference(model, data, normalizer=None, output_key="target", device="cpu"):
    """Inference

    Args:
            model (Torch.nn.Module): torch model
            data (Torch.nn.Data): torch Data
            normalizer (Dict): Information for normalization
            output_key (str): output key

    Returns:
            out (torch.Tensor): inference tensor
    """
    model.to(device)
    model.eval()
    out = model(data, inference=True)[output_key]
    if normalizer is None:
        return out
    else:
        out = normalizer[output_key].denorm(out)
        return out


def ensemble_inference(
    model_list, data, normalizer=None, output_key="target", device="cpu", var=False
):
    """Inference

    Args:
            model_list (List[Torch.nn.Module)]: torch model list
            data (Torch.nn.Data): torch Data
            output_key (str): output key
            normalizer (Dict): Information for normalization

    Returns:
            out (torch.Tensor): inference tensor
    """
    output_bin = []
    for model in model_list:
        model.to(device)
        model.eval()
        out = model(data, inference=True)[output_key]
        if normalizer is None:
            output_bin.append(out.unsqueeze(1))
        else:
            out = normalizer[output_key].denorm(out).unsqueeze(1)
            output_bin.append(out)

    output_tensor = torch.mean(torch.stack(output_bin, dim=1), dim=1).squeeze(1)
    if var:
        output_tensor_var = torch.var(torch.stack(output_bin, dim=1), dim=1).squeeze(1)
        return output_tensor, output_tensor_var
    else:
        return output_tensor


TESNOR = torch.Tensor


class Normalizer:
    """Normalize a Tensor and restore it later."""

    def __init__(self, inputs: Union[TESNOR, Dict], key: str):
        """tensor is taken as a sample to calculate the mean and std"""
        if isinstance(inputs, TESNOR):
            self.mean, self.std, self.max, self.min, self.sum = self.preprocess(inputs, key)

        elif isinstance(inputs, Dict):
            self.load_state_dict(inputs)

        else:
            TypeError

    def norm(self, tensor):
        mean = self.mean.to(tensor.device)
        std = self.std.to(tensor.device)

        return (tensor - mean) / std
        # return (tensor - self.mean) / self.std

    def norm_to_unity(self, tensor):
        _sum = self.sum.to(tensor.device)

        return tensor / _sum

    def denorm(self, normed_tensor):
        std = self.std.to(normed_tensor.device)
        mean = self.mean.to(normed_tensor.device)

        return normed_tensor * std + mean
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

    def preprocess(self, tensor, key):
        """
        Preprocess the tensor:
        (1) filter nan
        (2) calculate mean, std, max, min, sum depending on the dimension
        """
        if tensor.dim() == 1:
            valid_index = torch.bitwise_not(torch.isnan(tensor))
            filtered_targs = tensor[valid_index]
            print(f"Number of {key} for normlization: {filtered_targs.shape[0]}")
            mean = torch.mean(filtered_targs)
            std = torch.std(filtered_targs)
            _max = torch.max(filtered_targs)
            _min = torch.min(filtered_targs)
            _sum = torch.sum(filtered_targs)
        elif tensor.dim() == 2:
            mean_bin = []
            std_bin = []
            _max_bin = []
            _min_bin = []
            _sum_bin = []
            transposed = torch.transpose(tensor, dim0=0, dim1=1)
            for i, values in enumerate(transposed):
                valid_index = torch.bitwise_not(torch.isnan(values))
                filtered_targs = values[valid_index]
                print(f"Number of {key}_{i} for normlization: {filtered_targs.shape[0]}")
                mean_temp = torch.mean(filtered_targs)
                mean_bin.append(mean_temp)
                std_temp = torch.std(filtered_targs)
                std_bin.append(std_temp)
                max_temp = torch.max(filtered_targs)
                _max_bin.append(max_temp)
                min_temp = torch.min(filtered_targs)
                _min_bin.append(min_temp)
                sum_temp = torch.sum(filtered_targs)
                _sum_bin.append(sum_temp)

            mean = torch.tensor(mean_bin)
            std = torch.tensor(std_bin)
            _max = torch.tensor(_max_bin)
            _min = torch.tensor(_min_bin)
            _sum = torch.tensor(_sum_bin)
        else:
            ValueError("input dimension is not right.")

        return mean, std, _max, _min, _sum


def test_model_list(
    model_list,
    data_loader,
    device,
    normalizer=None,
    multifidelity=False,
):
    """
    test the model performances
    Args:
        model: Model,
        data_loader: DataLoader,
        metric_fn: metric function,
        device: "cpu" or "cuda",
    Return:
        Lists of prediction, targets, ids, and metric
    """
    test_targets = []
    test_preds = []
    test_ids = []
    test_targets_fidelity = []
    test_preds_fidelity = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch_to(batch, device)
            target = batch["target"]
            # Compute output
            
            output = ensemble_inference(model_list, batch, normalizer, "target", device)
            # Rearrange the outputs
            test_pred = output.data.cpu()
            test_target = target.detach().cpu()

            batch_ids = []
            count = 0
            num_bin = []
            for i, val in enumerate(batch["num_atoms"].detach().cpu().numpy()):
                count += val
                num_bin.append(count)
                if i == 0:
                    change = list(np.arange(val))
                else:
                    adding_val = num_bin[i - 1]
                    change = list(np.arange(val) + adding_val)
                batch_ids.append(change)

            test_preds += [test_pred[i].tolist() for i in batch_ids]
            test_targets += [test_target[i].tolist() for i in batch_ids]

            if multifidelity:
                target_fidelity = batch["fidelity"].detach().cpu()
                # Compute output
                output_fidelity = ensemble_inference(model_list, batch, normalizer, "fidelity", device).data.cpu()

                test_preds_fidelity += [output_fidelity[i].tolist() for i in batch_ids]
                test_targets_fidelity += [
                    target_fidelity[i].tolist() for i in batch_ids
                ]
            if isinstance(batch["name"], list):
                test_ids += batch["name"]
            else:
                test_ids += batch["name"].detach().tolist()
    return (
        test_preds,
        test_targets,
        test_ids,
        test_preds_fidelity,
        test_targets_fidelity,
    )
