from persite_painn.train import Normalizer


def inference(model, data, output_key, normalizer=None):
    """Inference

    Args:
            model (Torch.nn.Module): torch model
            data (Torch.nn.Data): torch Data
            output_key (str): output key
            normalizer (Dict): Information for normalization

    Returns:
            out (torch.Tensor): inference tensor
    """
    out = model(data)[output_key].detach()
    if normalizer is None:
        return out
    else:
        normalizer = Normalizer(normalizer)
        out = normalizer.denorm(out)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
