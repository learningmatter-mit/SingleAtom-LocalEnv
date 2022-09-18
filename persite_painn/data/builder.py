from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

from pymatgen.io.ase import AseAtomsAdaptor as AA
import numpy as np
from .dataset import Dataset


def build_dataset(
    raw_data,
    prop_to_predict,
    cutoff=5.0,
    site_prediction=True,
    seed=1234,
) -> Dataset:
    if site_prediction:
        samples = [[id_, struct] for id_, struct in raw_data.items()]
    else:
        samples = [
            [id_, data["structure"], data["target"]] for id_, data in raw_data.items()
        ]
    props = gen_props_from_file(
        samples=samples,
        prop_to_predict=prop_to_predict,
        site_prediction=site_prediction,
        seed=seed,
    )
    dataset = Dataset(props=props)
    dataset.generate_neighbor_list(cutoff=cutoff, undirected=False)

    return dataset


def compute_prop(id_, crystal, target_val, prop_to_predict, site_prediction):
    if site_prediction:
        target = crystal.site_properties[prop_to_predict]
        site_prop = None
    else:
        site_prop = crystal.site_properties["site_prop"]
        target = target_val
    structure = AA.get_atoms(crystal)
    n = np.asarray(structure.numbers).reshape(-1, 1)
    xyz = np.asarray(structure.positions)
    nxyz = np.concatenate((n, xyz), axis=1)
    lattice = structure.cell[:]

    return id_, nxyz, lattice, site_prop, target


def gen_props_from_file(
    samples,
    prop_to_predict,
    site_prediction=True,
    seed=1234,
):
    """Summary

    Args:
        path (TYPE): Description

    Returns:
        TYPE: Description

    Raises:
        TypeError: Description
    """
    print("Creating props...")
    random.seed(seed)
    random.shuffle(samples)

    props = {}
    name_list = []
    nxyz_list = []
    lattice_list = []
    site_prop_list = []
    target = []
    for idx in tqdm(range(len(samples)), position=0, leave=True):
        if site_prediction:
            id_, nxyz, lattice, site_prop, target_val = compute_prop(
                samples[idx][0],
                samples[idx][1],
                None,
                prop_to_predict,
                site_prediction,
            )
        else:
            id_, nxyz, lattice, site_prop, target_val = compute_prop(
                samples[idx][0],
                samples[idx][1],
                samples[idx][2],
                prop_to_predict,
                site_prediction,
            )
        name_list.append(id_)
        nxyz_list.append(nxyz)
        lattice_list.append(lattice)
        site_prop_list.append(site_prop)
        target.append(target_val)

    props["nxyz"] = nxyz_list
    props["lattice"] = lattice_list
    props["site_prop"] = site_prop_list
    props["name"] = name_list
    props["target"] = target

    return props


def binary_split(dataset, targ_name, test_size, seed):
    """
    Split the dataset with proportional amounts of a binary label in each.
    Args:
        dataset (nff.data.dataset): NFF dataset
        targ_name (str, optional): name of the binary label to use
            in splitting.
        test_size (float, optional): fraction of dataset for test
    Returns:
        idx_train (list[int]): indices of species in the training set
        idx_test (list[int]): indices of species in the test set
    """

    # get indices of positive and negative values
    pos_idx = [i for i, targ in enumerate(dataset.props[targ_name]) if targ]
    neg_idx = [i for i in range(len(dataset)) if i not in pos_idx]

    # split the positive and negative indices separately
    pos_idx_train, pos_idx_test = train_test_split(
        pos_idx, test_size=test_size, random_state=seed
    )
    neg_idx_train, neg_idx_test = train_test_split(
        neg_idx, test_size=test_size, random_state=seed
    )

    # combine the negative and positive test idx to get the test idx
    # do the same for train

    idx_train = pos_idx_train + neg_idx_train
    idx_test = pos_idx_test + neg_idx_test

    return idx_train, idx_test


def split_train_test(dataset, test_size=0.2, binary=False, targ_name=None, seed=None):
    """Splits the current dataset in two, one for training and
    another for testing.

    Args:
        dataset (nff.data.dataset): NFF dataset
        test_size (float, optional): fraction of dataset for test
        binary (bool, optional): whether to split the dataset with
            proportional amounts of a binary label in each.
        targ_name (str, optional): name of the binary label to use
            in splitting.
    Returns:
        TYPE: Description
    """

    if binary:
        idx_train, idx_test = binary_split(
            dataset=dataset, targ_name=targ_name, test_size=test_size, seed=seed
        )
    else:
        idx = list(range(len(dataset)))
        idx_train, idx_test = train_test_split(
            idx, test_size=test_size, random_state=seed
        )

    train = Dataset(
        props={key: [val[i] for i in idx_train] for key, val in dataset.props.items()},
    )
    test = Dataset(
        props={key: [val[i] for i in idx_test] for key, val in dataset.props.items()},
    )

    return train, test


def split_train_validation_test(
    dataset, val_size=0.2, test_size=0.2, seed=None, **kwargs
):
    """Summary
    Args:
        dataset (TYPE): Description
        val_size (float, optional): Description
        test_size (float, optional): Description
    Returns:
        TYPE: Description
    """
    train, test = split_train_test(dataset, test_size=test_size, seed=seed, **kwargs)
    train, validation = split_train_test(
        train, test_size=val_size / (1 - test_size), seed=seed, **kwargs
    )

    return train, validation, test
