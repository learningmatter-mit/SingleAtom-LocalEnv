import torch
from .loss import sid_operation
from persite_painn.utils.tools import gaussian_smoothing


def sis_operation(pred_spectra, target_spectra, sigma=5):

    filtered_target = gaussian_smoothing(target_spectra, sigma)
    filtered_pred = gaussian_smoothing(pred_spectra, sigma)
    sid = sid_operation(filtered_pred, filtered_target)
    sis = 1 / (1 + sid)

    return sis


def sis_loss(pred_spectra, target_spectra):
    loss = torch.mean(
        sis_operation(pred_spectra=pred_spectra, target_spectra=target_spectra)
    )
    return loss


def mae_loss(
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
    assert flattened_pred.shape[0] == flattened_targ.shape[0]
    nan_mask = torch.isnan(flattened_targ)
    nan_mask = nan_mask.to(target.device)

    return torch.mean(torch.abs(flattened_pred[~nan_mask] - flattened_targ[~nan_mask]))
