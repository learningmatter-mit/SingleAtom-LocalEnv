"""
Helper functions to create models, functions and other classes
while checking for the validity of hyperparameters.
"""

import json
import os

import torch
from persite_painn.nn.models import PainnAtomwise

PARAMS_TYPE = {
    "PainnAtomwise": {
        "feat_dim": int,
        "activation": str,
        "n_rbf": int,
        "cutoff": float,
        "num_conv": int,
        "output_keys": list,
        "grad_keys": list,
        "excl_vol": bool,
        "V_ex_power": int,
        "V_ex_sigma": float,
    },
}

MODEL_DICT = {
    "PainnAtomwise": PainnAtomwise,
}


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


def get_model(params, model_type="PainnAtomwise", **kwargs):
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
    keys = ["details", "modelparams"]
    params = None
    for key in keys:
        if key in info:
            params = info[key]
            break
    if params is None:
        params = info

    model_type = params["model_type"]

    return params, model_type


def load_model(path, params=None, model_type=None, **kwargs):
    """Load pretrained model from the path. If no epoch is specified,
            load the best model.

    Args:
            path (str): path where the model was trained.
            params (dict, optional): Any parameters you need to instantiate
                    a model before loading its state dict. This is required for DimeNet,
                    in which you can't pickle the model directly.
            model_type (str, optional): name of the model to be used
    Returns:
            model
    """

    try:
        if os.path.isdir(path):
            return torch.load(os.path.join(path, "best_model"), map_location="cpu")
        elif os.path.exists(path):
            return torch.load(path, map_location="cpu")
        else:
            raise FileNotFoundError("{} was not found".format(path))
    except (FileNotFoundError, EOFError, RuntimeError):

        param_path = os.path.join(path, "params.json")
        if os.path.isfile(param_path):
            params, model_type = load_params(param_path)

        assert (
            params is not None
        ), "Must specify params if you want to load the state dict"
        assert (
            model_type is not None
        ), "Must specify the model type if you want to load the state dict"

        model = get_model(params, model_type=model_type, **kwargs)

        if os.path.isdir(path):
            state_dict = torch.load(
                os.path.join(path, "best_model.pth.tar"), map_location="cpu"
            )
        elif os.path.exists(path):
            state_dict = torch.load(path, map_location="cpu")
        else:
            raise FileNotFoundError("{} was not found".format(path))

        model.load_state_dict(state_dict["model"], strict=False)
        return model
