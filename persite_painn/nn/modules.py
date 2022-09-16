import torch
from nff.nn.layers import Dense
from nff.nn.modules.schnet import ScaleShift
from nff.utils.tools import layer_types
from torch import nn

from nff.utils.scatter import compute_grad
from torch_scatter import scatter
from itertools import repeat


def to_module(activation):
    return layer_types[activation]()


class nn_exp(nn.Module):
    def __init__(self):
        super(nn_exp, self).__init__()

    def forward(self, x):
        return torch.exp(x)


class ReadoutBlock_(nn.Module):
    def __init__(
        self,
        feat_dim,
        output_atom_fea,
        output_keys,
        activation,
        dropout,
        means=None,
        stddevs=None,
    ):
        super().__init__()

        self.readoutdict = nn.ModuleDict(
            {
                key: Dense(
                    in_features=feat_dim,
                    out_features=output_atom_fea,
                    bias=True,
                    dropout_rate=dropout,
                    activation=to_module(activation),
                )
                for key in output_keys
            }
        )

        self.scale_shift = ScaleShift(means=means, stddevs=stddevs)

    def forward(self, s_i):
        """
        Note: no atomwise summation. That's done in the model itself
        """

        results = {}

        for key, readoutdict in self.readoutdict.items():
            output = readoutdict(s_i)
            output = self.scale_shift(output, key)
            results[key] = output

        return results


class AtomwiseReadoutBlock_orig(nn.Module):
    def __init__(
        self,
        feat_dim,
        output_atom_fea,
        output_keys,
        activation,
        dropout,
        means=None,
        stddevs=None,
    ):
        super().__init__()

        self.readoutdict = nn.ModuleDict(
            {
                key: nn.Sequential(
                    Dense(
                        in_features=feat_dim,
                        out_features=feat_dim,
                        bias=True,
                        dropout_rate=dropout,
                        activation=to_module(activation),
                    ),
                    Dense(
                        in_features=feat_dim,
                        out_features=output_atom_fea,
                        bias=True,
                        dropout_rate=dropout,
                    ),
                )
                for key in output_keys
            }
        )

        self.scale_shift = ScaleShift(means=means, stddevs=stddevs)

    def forward(self, s_i):
        """
        Note: no atomwise summation. That's done in the model itself
        """

        results = {}

        for key, readoutdict in self.readoutdict.items():
            output = readoutdict(s_i)
            output = self.scale_shift(output, key)
            results[key] = output

        return results


class AtomwiseReadoutBlock(nn.Module):
    def __init__(
        self,
        feat_dim,
        output_atom_fea,
        output_keys,
        activation,
        dropout,
    ):
        super().__init__()

        self.readoutdict = nn.ModuleDict(
            {
                key: Dense(
                    in_features=feat_dim,
                    out_features=output_atom_fea,
                    bias=True,
                    dropout_rate=dropout,
                    activation=to_module(activation),
                )
                for key in output_keys
            }
        )

        # self.scale_shift = ScaleShift(means=means, stddevs=stddevs)

    def forward(self, s_i):
        """
        Note: no atomwise summation. That's done in the model itself
        """

        results = {}

        for key, readoutdict in self.readoutdict.items():
            output = readoutdict(s_i)
            # output = self.scale_shift(output, key)
            results[key] = output

        return results


class SumPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch, xyz, atomwise_output, grad_keys, out_keys=None):
        results = sum_and_grad(
            batch=batch,
            xyz=xyz,
            atomwise_output=atomwise_output,
            grad_keys=grad_keys,
            out_keys=out_keys,
        )
        return results


def sum_and_grad(batch, xyz, atomwise_output, grad_keys, out_keys=None, mean=False):

    N = batch["num_atoms"].detach().cpu().tolist()
    results = {}
    if out_keys is None:
        out_keys = list(atomwise_output.keys())

    for key, val in atomwise_output.items():
        if key not in out_keys:
            continue

        mol_idx = (
            torch.arange(len(N)).repeat_interleave(torch.LongTensor(N)).to(val.device)
        )
        dim_size = mol_idx.max() + 1

        if val.reshape(-1).shape[0] == mol_idx.shape[0]:
            use_val = val.reshape(-1)

        # summed atom features
        elif val.shape[0] == mol_idx.shape[0]:
            use_val = val

        else:
            raise Exception(
                (
                    "Don't know how to handle val shape "
                    "{} for key {}".format(val.shape, key)
                )
            )
        pooled_result = global_sum_pool(use_val, mol_idx.reshape(-1, 1), size=dim_size)
        if mean:
            pooled_result = pooled_result / torch.Tensor(N).to(val.device)

        results[key] = pooled_result

    # compute gradients

    for key in grad_keys:
        output = results[key.replace("_grad", "")]
        grad = compute_grad(output=output, inputs=xyz)
        results[key] = grad

    return results


def gen(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    dim = range(src.dim())[dim]  # Get real dim value.

    # Automatically expand index tensor to the right dimensions.
    if index.dim() == 1:
        index_size = list(repeat(1, src.dim()))
        index_size[dim] = src.size(dim)
        index = index.view(index_size).expand_as(src)

    # Generate output tensor if not given.
    if out is None:
        dim_size = index.max().item() + 1 if dim_size is None else dim_size
        out_size = list(src.size())
        out_size[dim] = dim_size
        out = src.new_full(out_size, fill_value)

    return src, out, index, dim


def scatter_add(src, index, dim=-1, out=None, dim_size=None, fill_value=0):

    src, out, index, dim = gen(
        src=src, index=index, dim=dim, out=out, dim_size=dim_size, fill_value=fill_value
    )
    output = out.scatter_add_(dim, index, src)

    return output


def global_sum_pool(x, batch, size=None):
    """
    Globally pool node embeddings into graph embeddings, via elementwise mean.
    Pooling function takes in node embedding [num_nodes x emb_dim] and
    batch (indices) and outputs graph embedding [num_graphs x emb_dim].

    Args:
        x (torch.tensor): Input node embeddings
        batch (torch.tensor): Batch tensor that indicates which node
        belongs to which graph
        size (optional): Total number of graphs. Can be auto-inferred.

    Returns: Pooled graph embeddings

    """
    size = batch.max().item() + 1 if size is None else size
    return scatter(x, batch, dim=0, dim_size=size, reduce="sum")
