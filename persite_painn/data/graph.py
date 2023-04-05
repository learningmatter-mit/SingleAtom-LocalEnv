import torch
from .utils import tqdm_enum
import numpy as np

DISTANCETHRESHOLDICT_Z = {
    (1., 1.): 1.00,
    (1., 3.): 1.30,
    (1., 5.): 1.50,
    (1., 6.): 1.30,
    (1., 7.): 1.30,
    (1., 8.): 1.30,
    (1., 9.): 1.30,
    (1., 11.): 1.65,
    (1., 14.): 1.65,
    (1., 12.): 1.40,
    (1., 16.): 1.50,
    (1., 17.): 1.60,
    (1., 35.): 1.60,
    (3., 6.): 0.0,
    (3., 7.): 0.0,
    (3., 8.): 0.0,
    (3., 9.): 0.0,
    (3., 12.): 0.0,
    (5., 6.): 1.70,
    (5., 7.): 1.70,
    (5., 8.): 1.70,
    (5., 9.): 1.70,
    (5., 11.): 1.8,
    (5., 12.): 1.8,
    (5., 17.): 2.1,
    (5., 35.): 2.1,
    (6., 6.): 1.70,
    (6., 8.): 1.70,
    (6., 7.): 1.8,
    (6., 9.): 1.65,
    (6., 11.): 1.80,
    (6., 12.): 1.70,
    (6., 14.): 2.10,
    (6., 16.): 2.20,
    (7., 8.): 1.55,
    (7., 11.): 1.70,
    (7., 16.): 2.0,
    (8., 11.): 1.70,
    (8., 12.): 1.35,
    (8., 16.): 2.00,
    (8., 17.): 1.80,
    (8., 8.): 1.70,
    (8., 9.): 1.50,
    (8., 14.): 1.85,
    (8., 35.): 1.70,
    (9., 12.): 1.35}


def m_idx_of_angles(angle_list,
                    nbr_list,
                    angle_start,
                    angle_end):
    """
    Get the array index of elements of an angle list.
    Args:
        angle_list (torch.LongTensor): directed indices
            of sets of three atoms that are all in each
            other's neighborhood.
        nbr_list (torch.LongTensor): directed indices
            of pairs of atoms that are in each other's
            neighborhood.
        angle_start (int): the first index in the angle
            list you want.
        angle_end (int): the last index in the angle list
            you want.
    Returns:
        idx (torch.LongTensor): `m` indices.
    Example:
        angle_list = torch.LongTensor([[0, 1, 2],
                                       [0, 1, 3]])
        nbr_list = torch.LongTensor([[0, 1],
                                    [0, 2],
                                    [0, 3],
                                    [1, 0],
                                    [1, 2],
                                    [1, 3],
                                    [2, 0],
                                    [2, 1],
                                    [2, 3],
                                    [3, 0],
                                    [3, 1],
                                    [3, 2]])

        # This means that message vectors m_ij are ordered
        # according to m = {m_01, m_01, m_03, m_10,
        # m_12, m_13, m_30, m_31, m_32}. Say we are interested
        # in indices 2 and 1 for each element in the angle list.
        # If we want to know what the corresponding indices
        # in m (or the nbr list) are, we would call `m_idx_of_angles`
        # with angle_start = 2, angle_end = 1 (if we want the
        # {2,1} and {3,1} indices), or angle_start = 1,
        # angle_end = 0 (if we want the {1,2} and {1,3} indices).
        # Say we choose angle_start = 2 and angle_end = 1. Then
        # we get the indices of {m_21, m_31}, which we can see
        # from the nbr list are [7, 10].


    """

    # expand nbr_list[:, 0] so it's repeated once
    # for every element of `angle_list`.
    repeated_nbr = nbr_list[:, 0].repeat(angle_list.shape[0], 1)
    reshaped_angle = angle_list[:, angle_start].reshape(-1, 1)
    # gives you a matrix that shows you where each angle is equal
    # to nbr_list[:, 0]
    mask = repeated_nbr == reshaped_angle

    # same idea, but with nbr_list[:, 1] and angle_list[:, angle_end]

    repeated_nbr = nbr_list[:, 1].repeat(angle_list.shape[0], 1)
    reshaped_angle = angle_list[:, angle_end].reshape(-1, 1)

    # the full mask is the product of both
    mask *= (repeated_nbr == reshaped_angle)

    # get the indices where everything is true
    idx = mask.nonzero(as_tuple=False)[:, 1]

    return idx


def add_ji_kj(angle_lists, nbr_lists):
    """
    Get ji and kj idx (explained more below):
    Args:
        angle_list (list[torch.LongTensor]): list of angle
            lists
        nbr_list (list[torch.LongTensor]): list of directed neighbor
            lists
    Returns:
        ji_idx_list (list[torch.LongTensor]): ji_idx for each geom
        kj_idx_list (list[torch.LongTensor]): kj_idx for each geom

    """

    # given an angle a_{ijk}, we want
    # ji_idx, which is the array index of m_ji.
    # We also want kj_idx, which is the array index
    # of m_kj. For example, if i,j,k = 0,1,2,
    # and our neighbor list is [[0, 1], [0, 2],
    # [1, 0], [1, 2], [2, 0], [2, 1]], then m_10 occurs
    # at index 2, and m_21 occurs at index 5. So
    # ji_idx = 2 and kj_idx = 5.

    ji_idx_list = []
    kj_idx_list = []
    for i, nbr_list in tqdm_enum(nbr_lists):
        angle_list = angle_lists[i]
        ji_idx = m_idx_of_angles(angle_list=angle_list,
                                 nbr_list=nbr_list,
                                 angle_start=1,
                                 angle_end=0)

        kj_idx = m_idx_of_angles(angle_list=angle_list,
                                 nbr_list=nbr_list,
                                 angle_start=2,
                                 angle_end=1)
        ji_idx_list.append(ji_idx)
        kj_idx_list.append(kj_idx)

    return ji_idx_list, kj_idx_list


def make_directed(nbr_list):
    """
    Check if a neighbor list is directed, and make it
    directed if it isn't.
    Args:
        nbr_list (torch.LongTensor): neighbor list
    Returns:
        new_nbrs (torch.LongTensor): directed neighbor
            list
        directed (bool): whether the old one was directed
            or not  
    """

    gtr_ij = (nbr_list[:, 0] > nbr_list[:, 1]).any().item()
    gtr_ji = (nbr_list[:, 1] > nbr_list[:, 0]).any().item()
    directed = gtr_ij and gtr_ji

    if directed:
        return nbr_list, directed

    new_nbrs = torch.cat([nbr_list, nbr_list.flip(1)], dim=0)
    return new_nbrs, directed


def make_dset_directed(dset):
    """
    Make everything in the dataset correspond to a directed 
    neighbor list.
    Args:
        dset (nff.data.Dataset): nff dataset
    Returns:
        None
    """

    # make the neighbor list directed
    for i, batch in enumerate(dset):
        nbr_list, nbr_was_directed = make_directed(batch['nbr_list'])
        dset.props['nbr_list'][i] = nbr_list

        # fix bond_idx
        bond_idx = batch.get("bond_idx")
        has_bond_idx = (bond_idx is not None)
        if (not nbr_was_directed) and has_bond_idx:
            nbr_dim = nbr_list.shape[0]
            bond_idx = torch.cat([bond_idx,
                                  bond_idx + nbr_dim // 2])
            dset.props['bond_idx'][i] = bond_idx

        # make the bonded nbr list directed
        bond_nbrs = batch.get('bonded_nbr_list')
        has_bonds = (bond_nbrs is not None)
        if has_bonds:
            bond_nbrs, bonds_were_directed = make_directed(bond_nbrs)
            dset.props['bonded_nbr_list'][i] = bond_nbrs

        # fix the corresponding bond features
        bond_feats = batch.get('bond_features')
        has_bond_feats = (bond_feats is not None)
        if (has_bonds and has_bond_feats) and (not bonds_were_directed):
            bond_feats = torch.cat([bond_feats] * 2, dim=0)
            dset.props['bond_features'][i] = bond_feats


def get_neighbor_list(xyz, cutoff=5, undirected=True):
    """Get neighbor list from xyz positions of atoms.
    Args:
        xyz (torch.Tensor or np.array): (N, 3) array with positions
            of the atoms.
        cutoff (float): maximum distance to consider atoms as
            connected.
    Returns:
        nbr_list (torch.Tensor): (num_edges, 2) array with the
            indices of connected atoms.
    """

    if torch.is_tensor(xyz) == False:
        xyz = torch.Tensor(xyz)
    n = xyz.size(0)

    # calculating distances
    dist = (xyz.expand(n, n, 3) - xyz.expand(n, n, 3).transpose(0, 1)
            ).pow(2).sum(dim=2).sqrt()

    # neighbor list
    mask = (dist <= cutoff)
    mask[np.diag_indices(n)] = 0
    nbr_list = mask.nonzero(as_tuple=False)

    if undirected:
        nbr_list = nbr_list[nbr_list[:, 1] > nbr_list[:, 0]]

    return nbr_list
