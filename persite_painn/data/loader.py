# Source: https://github.com/learningmatter-mit/NeuralForceField/blob/master/nff/data/loader.py

import numpy as np
import torch

from torch.utils.data.sampler import Sampler

REINDEX_KEYS = [
    "atoms_nbr_list",
    "nbr_list",
    "bonded_nbr_list",
    "angle_list",
    "mol_nbrs",
]
NBR_LIST_KEYS = ["bond_idx", "kj_idx", "ji_idx"]
MOL_IDX_KEYS = ["atomwise_mol_list", "directed_nbr_mol_list", "undirected_nbr_mol_list"]
IGNORE_KEYS = ["rd_mols"]

TYPE_KEYS = {
    "atoms_nbr_list": torch.long,
    "nbr_list": torch.long,
    "num_atoms": torch.long,
    "bond_idx": torch.long,
    "bonded_nbr_list": torch.long,
    "angle_list": torch.long,
    "ji_idx": torch.long,
    "kj_idx": torch.long,
}


def collate_dicts(dicts):
    """Collates dictionaries within a single batch. Automatically reindexes
        neighbor lists and periodic boundary conditions to deal with the batch.
    Args:
        dicts (list of dict): each element of the dataset
    Returns:
        batch (dict)
    """

    # new indices for the batch: the first one is zero and the
    # last does not matter

    cumulative_atoms = np.cumsum([0] + [d["num_atoms"] for d in dicts])[:-1]

    for n, d in zip(cumulative_atoms, dicts):
        for key in REINDEX_KEYS:
            if key in d:
                d[key] = d[key] + int(n)

    if all(["nbr_list" in d for d in dicts]):
        # same idea, but for quantities whose maximum value is the length of
        # the nbr list in each batch
        cumulative_nbrs = np.cumsum([0] + [len(d["nbr_list"]) for d in dicts])[:-1]
        for n, d in zip(cumulative_nbrs, dicts):
            for key in NBR_LIST_KEYS:
                if key in d:
                    d[key] = d[key] + int(n)

    for key in MOL_IDX_KEYS:
        if key not in dicts[0]:
            continue
        for i, d in enumerate(dicts):
            d[key] += i

    # batching the data
    batch = {}
    for key, val in dicts[0].items():
        if key in IGNORE_KEYS:
            continue
        if type(val) == str:
            batch[key] = [data[key] for data in dicts]
        elif hasattr(val, "shape") and len(val.shape) > 0:
            batch[key] = torch.cat([data[key] for data in dicts], dim=0)
        else:
            batch[key] = torch.stack([data[key] for data in dicts], dim=0)

    # adjusting the data types:
    for key, dtype in TYPE_KEYS.items():
        if key in batch:
            batch[key] = batch[key].to(dtype)

    return batch


