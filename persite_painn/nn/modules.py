import torch
from nff.nn.layers import Dense
from nff.nn.modules.schnet import ScaleShift
from nff.utils.tools import layer_types
from torch import nn


def to_module(activation):
    return layer_types[activation]()


class nn_exp(nn.Module):
    def __init__(self):
        super(nn_exp, self).__init__()

    def forward(self, x):
        return torch.exp(x)


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
