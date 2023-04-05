import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def add_morgan(dataset, vec_length):
    """
    Add Morgan fingerprints to the dataset. Note that this uses
    the smiles of each species to get one fingerprint per species, 
    as opposed to getting the graph of each conformer and its 
    fingerprint.

    Args:
        dataset (nff.data.dataset): NFF dataset
        vec_length (int): how long the fingerprint should be.
    Returns:
        None

    """

    dataset.props["morgan"] = []
    for smiles in dataset.props['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if vec_length != 0:
            morgan = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=vec_length)
        else:
            morgan = []

        arr_morgan = np.array(list(morgan)).astype('float32')
        morgan_tens = torch.tensor(arr_morgan)
        dataset.props["morgan"].append(morgan_tens)
