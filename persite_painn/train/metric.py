from scipy.ndimage import gaussian_filter1d
import torch
from .loss import sid_operation


def sis_operation(pred_spectra, target_spectra, torch_device: str = "cpu"):

    filtered_target = torch.tensor(gaussian_filter1d(target_spectra, 1))
    filtered_target = filtered_target.to(torch_device)
    filtered_pred = torch.tensor(gaussian_filter1d(pred_spectra, 1))
    filtered_pred = filtered_pred.to(torch_device)
    sid = sid_operation(filtered_pred, filtered_target)
    sis = 1 / (1 + sid)

    return sis


def mae_operation(
    prediction,
    target,
    torch_device: str = "cpu",
):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    flattened_pred = prediction.view(1, -1)
    flattened_targ = target.view(1, -1)
    flattened_pred = flattened_pred.to(torch_device)
    flattened_targ = flattened_targ.to(torch_device)
    return torch.mean(torch.abs(flattened_pred - flattened_targ), dim=1)
