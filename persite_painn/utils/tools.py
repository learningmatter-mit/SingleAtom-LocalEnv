import torch
import json
import numpy as np
from nff.nn.activations import shifted_softplus, Swish, LearnableSwish

from nff.nn.layers import Dense

layer_types = {
    "linear": torch.nn.Linear,
    "Tanh": torch.nn.Tanh,
    "ReLU": torch.nn.ReLU,
    "Dense": Dense,
    "shifted_softplus": shifted_softplus,
    "sigmoid": torch.nn.Sigmoid,
    "Dropout": torch.nn.Dropout,
    "LeakyReLU": torch.nn.LeakyReLU,
    "ELU": torch.nn.ELU,
    "swish": Swish,
    "learnable_swish": LearnableSwish,
    "softplus": torch.nn.Softplus,
}


def make_directed(nbr_list):

    gtr_ij = (nbr_list[:, 0] > nbr_list[:, 1]).any().item()
    gtr_ji = (nbr_list[:, 1] > nbr_list[:, 0]).any().item()
    directed = gtr_ij and gtr_ji

    if directed:
        return nbr_list, directed

    new_nbrs = torch.cat([nbr_list, nbr_list.flip(1)], dim=0)
    return new_nbrs, directed


def to_json(jsonpath, argparse_dict):
    """
    This function creates a .json file as a copy of argparse_dict

    Args:
        jsonpath (str): path to the .json file
        argparse_dict (dict): dictionary containing arguments from argument parser
    """
    with open(jsonpath, "w") as fp:
        json.dump(argparse_dict, fp, sort_keys=True, indent=4)


def compute_params(model):
    """
    This function gets a model as an input and computes its trainable parameters

    Args:
        model (AtomisticModel): model for which you want to compute the trainable parameters

    Returns:
        params (int): number of trainable parameters for the model
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def gaussian_filter_1d(size, sigma):
    double_pi_sqrt = (torch.pi * 2)**0.5
    filter_range = torch.linspace(-int(size / 2), int(size / 2), size)

    _filter = 1
    _filter = _filter / (sigma * double_pi_sqrt)
    _filter = _filter * torch.exp(-(filter_range**2) / (2 * sigma**2))
    return _filter


def gaussian_smoothing(signal, sigma):
    filter_size = signal.shape[-1]
    gaussian_filter = gaussian_filter_1d(filter_size, sigma).to(signal.device)
    if signal.dim() == 1:
        signal = signal.unsqueeze(0)
    signal = signal[:, None, :]
    gaussian_filter = gaussian_filter[None, None, :]
    smoothed_signal = torch.conv1d(signal,
                                   gaussian_filter,
                                   padding=filter_size // 2)
    return smoothed_signal.squeeze()


def get_offsets(batch, key):
    nxyz = batch["nxyz"]
    zero = torch.Tensor([0]).to(nxyz.device)
    offsets = batch.get(key, zero)
    if isinstance(offsets, torch.Tensor) and offsets.is_sparse:
        offsets = offsets.to_dense()
    return offsets


def get_rij(xyz, batch, nbrs, cutoff):

    offsets = get_offsets(batch, "offsets")
    # + offsets not - offsets because it's r_j - r_i,
    # whereas for schnet we've coded it as r_i - r_j
    r_ij = xyz[nbrs[:, 1]] - xyz[nbrs[:, 0]] + offsets

    # remove nbr skin (extra distance added to cutoff
    # to catch atoms that become neighbors between nbr
    # list updates)
    dist = (r_ij.detach()**2).sum(-1)**0.5

    if type(cutoff) == torch.Tensor:
        dist = dist.to(cutoff.device)
    use_nbrs = dist <= cutoff

    r_ij = r_ij[use_nbrs]
    nbrs = nbrs[use_nbrs]

    return r_ij, nbrs
