import torch
from .loss import sid_operation
from persite_painn.utils.tools import gaussian_smoothing


def sis_operation(pred_spectra, target_spectra, sigma=10):

    filtered_target = torch.tensor(gaussian_smoothing(target_spectra, sigma))
    filtered_pred = torch.tensor(gaussian_smoothing(pred_spectra, sigma))
    sid = sid_operation(filtered_pred, filtered_target)
    sis = 1 / (1 + sid)

    return sis


def mae_operation(
    prediction,
    target,
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

    return torch.mean(torch.abs(flattened_pred - flattened_targ), dim=1)
