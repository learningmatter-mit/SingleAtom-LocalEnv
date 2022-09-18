"""
Helper functions to create models, functions and other classes
while checking for the validity of hyperparameters.
"""

import json

import torch
from persite_painn.nn.models import Painn

PARAMS_TYPE = {
    "Painn": {
        "feat_dim": int,
        "activation": str,
        "activation_f": str,
        "n_rbf": int,
        "cutoff": float,
        "num_conv": int,
        "atom_fea_len": int,
        "n_h": int,
        "h_fea_len": int,
        "n_outputs": int,
        "output_keys": list,
        "site_prediction": bool,
    },
}

MODEL_DICT = {"Painn": Painn}


class ParameterError(Exception):
    """Raised when a hyperparameter is of incorrect type"""

    pass


def check_parameters(params_type, params):
    """Check whether the parameters correspond to the specified types

    Args:
            params (dict)
    """
    for key, val in params.items():
        if val is None:
            continue
        if key in params_type and not isinstance(val, params_type[key]):
            raise ParameterError("%s is not %s" % (str(key), params_type[key]))

        for model in PARAMS_TYPE.keys():
            if key == "{}_params".format(model.lower()):
                check_parameters(PARAMS_TYPE[model], val)


def get_model(params, model_type="Painn", **kwargs):
    """Create new model with the given parameters.

    Args:
            params (dict): parameters used to construct the model
            model_type (str): name of the model to be used

    Returns:
            model (nff.nn.models)
    """

    check_parameters(PARAMS_TYPE[model_type], params)
    model = MODEL_DICT[model_type](params, **kwargs)

    return model


def load_params(param_path):
    with open(param_path, "r") as f:
        info = json.load(f)

    details = info["details"]
    params = info["modelparams"]
    model_type = info["modeltype"]

    return details, params, model_type


def load_model(params_path, model_path, model_type="Painn"):
    """Load pretrained model from the path.

    Args:
            path (str): path where the model was trained.
            params (dict, optional): Any parameters you need to instantiate
                    a model before loading its state dict. This is required for DimeNet,
                    in which you can't pickle the model directly.
            model_type (str, optional): name of the model to be used
    Returns:
            model, best_checkoint
    """
    details, params, model_type = load_params(params_path)
    model = get_model(params,
                      model_type=model_type,
                      site_prediction=details["site_prediction"],
                      spectra=details["spectra"],
                      multifidelity=details["multifidelity"])
    best_checkpoint = torch.load(model_path)
    model.load_state_dict(best_checkpoint["state_dict"])
    model.eval()
    return model, best_checkpoint
