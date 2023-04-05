import numpy as np
import torch


def clean_matrix(matrix, eps=1e-12):
    """ clean from small values"""
    matrix = np.array(matrix)
    for ij in np.ndindex(matrix.shape):
        if abs(matrix[ij]) < eps:
            matrix[ij] = 0
    return matrix


def lattice_points_in_supercell(supercell_matrix):
    """Adapted from ASE to find all lattice points contained in a supercell.

    Adapted from pymatgen, which is available under MIT license:
    The MIT License (MIT) Copyright (c) 2011-2012 MIT & The Regents of the
    University of California, through Lawrence Berkeley National Laboratory
    """

    diagonals = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
        [0, -1, 1],
        [1, 0, -1],
        [1, -1, 0],
        [1, -1, 1],
        [-1, -1, 1],
        [1, 1, -1],
        [1, -1, -1],
    ])

    d_points = np.dot(diagonals, supercell_matrix)

    mins = np.min(d_points, axis=0)
    maxes = np.max(d_points, axis=0) + 1

    ar = np.arange(mins[0], maxes[0])[:, None] * np.array([1, 0, 0])[None, :]
    br = np.arange(mins[1], maxes[1])[:, None] * np.array([0, 1, 0])[None, :]
    cr = np.arange(mins[2], maxes[2])[:, None] * np.array([0, 0, 1])[None, :]

    all_points = ar[:, None, None] + br[None, :, None] + cr[None, None, :]
    all_points = all_points.reshape((-1, 3))

    frac_points = np.dot(all_points, np.linalg.inv(supercell_matrix))

    tvects = frac_points[np.all(abs(frac_points) < 1 - 1e-10, axis=1)]
    
    return tvects


def torch_nbr_list(atomsobject,
                   cutoff,
                   device='cuda:0',
                   directed=True,
                   requires_large_offsets=True):
    """Pytorch implementations of nbr_list for minimum image convention, the offsets are only limited to 0, 1, -1:
    it means that no pair interactions is allowed for more than 1 periodic box length. It is so much faster than
    neighbor_list algorithm in ase.
    It is similar to the output of neighbor_list("ijS", atomsobject, cutoff) but a lot faster
    Args:
        atomsobject (TYPE): Description
        cutoff (float): cutoff for
        device (str, optional): Description
        requires_large_offsets: to get offsets beyond -1,0,1
    Returns:
        i, j, cutoff: just like ase.neighborlist.neighbor_list
    """
    
    if any(atomsobject.pbc):
        # check if sufficiently large to run the "fast" nbr_list function
        # also check if orthorhombic
        # otherwise, default to the "robust" nbr_list function below for small cells
        if ( np.all(2*cutoff < atomsobject.cell.cellpar()[:3]) and 
            not np.count_nonzero(atomsobject.cell.T-np.diag(np.diagonal(atomsobject.cell.T)))!=0 ):
            # "fast" nbr_list function for large cells (pbc)
            xyz = torch.Tensor(atomsobject.get_positions(wrap=False)).to(device)
            dis_mat = xyz[None, :, :] - xyz[:, None, :]
            cell_dim = torch.Tensor(atomsobject.get_cell()[:]).diag().to(device)
            if requires_large_offsets:
                shift = torch.round(torch.divide(dis_mat,cell_dim))
                offsets = -shift
            else:
                offsets = -dis_mat.ge(0.5 * cell_dim).to(torch.float) + \
                dis_mat.lt(-0.5 * cell_dim).to(torch.float)
                
            dis_mat=dis_mat+offsets*cell_dim
            dis_sq = dis_mat.pow(2).sum(-1)
            mask = (dis_sq < cutoff ** 2) & (dis_sq != 0)
            nbr_list = mask.nonzero(as_tuple=False)
            offsets = offsets[nbr_list[:, 0], nbr_list[:, 1], :].detach().to("cpu").numpy()

        else:
            # "robust" nbr_list function for all cells (pbc)
            xyz = torch.Tensor(atomsobject.get_positions(wrap=True)).to(device)

            # since we are not wrapping
            # retrieve the shift vectors that would be equivalent to wrapping
            positions = atomsobject.get_positions(wrap=True)
            unwrapped_positions = atomsobject.get_positions(wrap=False)
            shift = positions - unwrapped_positions
            cell = atomsobject.cell
            cell = np.broadcast_to(cell.T, (shift.shape[0],cell.shape[0],cell.shape[1]))
            shift = np.linalg.solve(cell, shift).round().astype(int)

            # estimate getting close to the cutoff with supercell expansion
            cell = atomsobject.cell
            a_mul = int(np.ceil( cutoff / np.linalg.norm(cell[0]) ))+1
            b_mul = int(np.ceil( cutoff / np.linalg.norm(cell[1]) ))+1
            c_mul = int(np.ceil( cutoff / np.linalg.norm(cell[2]) ))+1
            supercell_matrix = np.array([[a_mul, 0, 0], [0, b_mul, 0], [0, 0, c_mul]])
            supercell = clean_matrix(supercell_matrix @ cell)

            # cartesian lattice points
            lattice_points_frac = lattice_points_in_supercell(supercell_matrix)
            lattice_points = np.dot(lattice_points_frac, supercell)
            # need to get all negative lattice translation vectors but remove duplicate 0 vector
            zero_idx = np.where(np.all(lattice_points.__eq__(np.array([0,0,0])), axis=1))[0][0]
            lattice_points = np.concatenate([lattice_points[zero_idx:, :], lattice_points[:zero_idx, :]])

            N = len(lattice_points)
            # perform lattice translation vectors on positions
            lattice_points_T = torch.tile(torch.from_numpy(lattice_points), 
                                            (len(xyz), ) + (1, ) * (len(lattice_points.shape) - 1) 
                                            ).to(device)
            xyz_T = torch.repeat_interleave(xyz.view(-1,1,3), N, dim=1)
            xyz_T = xyz_T + lattice_points_T.view(xyz_T.shape)
            diss=xyz_T[None,:,None,:,:]-xyz_T[:,None,:,None,:]
            diss=diss[:,:,0,:,:]
            dis_sq = diss.pow(2).sum(-1)
            mask = (dis_sq < cutoff ** 2) & (dis_sq != 0)
            nbr_list = mask.nonzero(as_tuple=False)[:,:2]
            offsets=(lattice_points_T.view(xyz_T.shape)
                        [mask.nonzero(as_tuple=False)[:,1],mask.nonzero(as_tuple=False)[:,2]])

            # get offsets as original integer multiples of lattice vectors
            cell = np.broadcast_to(cell.T, (offsets.shape[0],cell.shape[0],cell.shape[1]))
            offsets = offsets.detach().to("cpu").numpy()
            offsets = np.linalg.solve(cell, offsets).round().astype(int)

            # add shift to offsets with the right indices according to pairwise nbr_list
            offsets = torch.from_numpy(offsets).int().to(device)
            shift = torch.from_numpy(shift).int().to(device)

            # index shifts by atom but then apply shifts to pairwise interactions
            # get shifts for each atom i and j that would be equivalent to wrapping
            # convention is j - i for get_rij with NNs
            shift_i = shift[nbr_list[:,0]]
            shift_j = shift[nbr_list[:,1]]
            offsets = (shift_j - shift_i + offsets).detach().to("cpu").numpy()

    else:
        xyz = torch.Tensor(atomsobject.get_positions(wrap=False)).to(device)
        dis_mat = xyz[None, :, :] - xyz[:, None, :]        
        dis_sq = dis_mat.pow(2).sum(-1)
        mask = (dis_sq < cutoff ** 2) & (dis_sq != 0)
        nbr_list = mask.nonzero(as_tuple=False)
        
    if not directed:
        nbr_list = nbr_list[nbr_list[:, 1] > nbr_list[:, 0]]

    i, j = nbr_list[:, 0].detach().to("cpu").numpy(
    ), nbr_list[:, 1].detach().to("cpu").numpy()

    if any(atomsobject.pbc):
        offsets = offsets
    else:
        offsets = np.zeros((nbr_list.shape[0], 3))

    return i, j, offsets
