import torch


def sid_operation(
    model_spectra: torch.Tensor,
    target_spectra: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    # normalize the model spectra before comparison
    nan_mask = torch.isnan(target_spectra) + torch.isnan(model_spectra)
    nan_mask = nan_mask.to(target_spectra.device)
    zero_sub = torch.zeros_like(target_spectra, device=target_spectra.device)
    one_sub = torch.ones_like(model_spectra, device=target_spectra.device)

    sum_model_spectra = torch.sum(
        torch.where(nan_mask, zero_sub, model_spectra), dim=1, keepdim=True
    )

    model_spectra = torch.div(model_spectra, sum_model_spectra)
    model_spectra = torch.where(nan_mask, one_sub, model_spectra) + eps
    target_spectra = torch.where(nan_mask, one_sub, target_spectra) + eps

    loss = torch.sum(
        torch.mul(torch.log(torch.div(model_spectra, target_spectra)), model_spectra)
        + torch.mul(
            torch.log(torch.div(target_spectra, model_spectra)), target_spectra
        ),
        dim=1,
        keepdim=True,
    )
    return loss


def stmse_operation(
    model_spectra: torch.Tensor,
    target_spectra: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    # normalize the model spectra before comparison
    nan_mask = torch.isnan(target_spectra) + torch.isnan(model_spectra)
    nan_mask = nan_mask.to(target_spectra.device)
    zero_sub = torch.zeros_like(target_spectra, device=target_spectra.device)
    one_sub = torch.ones_like(model_spectra, device=target_spectra.device)

    sum_model_spectra = torch.sum(
        torch.where(nan_mask, zero_sub, model_spectra), dim=1, keepdim=True
    )
    model_spectra = torch.div(model_spectra, sum_model_spectra)
    target_spectra = torch.where(nan_mask, one_sub, target_spectra) + eps
    model_spectra = torch.where(nan_mask, one_sub, model_spectra) + eps

    loss = torch.mean(
        torch.div((model_spectra - target_spectra) ** 2, target_spectra), dim=1
    )
    return loss


def mse_operation(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    flattened_pred = prediction.view(1, -1)
    flattened_targ = target.view(1, -1)
    assert flattened_pred.shape[0] == flattened_targ.shape[0]
    loss = torch.mean((flattened_pred - flattened_targ) ** 2, dim=1)
    return loss
