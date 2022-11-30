def convert_to_sigopt_params(details, modelparams, sigoptparams, fidelity=False):
    converted_params = {}
    converted_details = details
    if fidelity:
        sigoptparams.setdefault(
            "atom_fea_len_fidelity", modelparams["atom_fea_len"]["fidelity"]
        )
        sigoptparams.setdefault("h_fea_len_fidelity", modelparams["h_fea_len"]["fidelity"])
        sigoptparams.setdefault("n_h_fidelity", modelparams["n_h"]["fidelity"])
        sigoptparams.setdefault(
            "readout_dropout_fidelity", modelparams["readout_dropout"]["fidelity"]
        )
        sigoptparams.setdefault(
            "fc_dropout_fidelity", modelparams["fc_dropout"]["fidelity"]
        )
        sigoptparams.setdefault(
            "loss_coeff_fidelity", modelparams["loss_coeff"]["fidelity"]
        )

    sigoptparams.setdefault("feat_dim", modelparams["feat_dim"])
    sigoptparams.setdefault("n_rbf", modelparams["n_rbf"])
    sigoptparams.setdefault("cutoff", modelparams["cutoff"])
    sigoptparams.setdefault("num_conv", modelparams["num_conv"])
    sigoptparams.setdefault("conv_dropout", modelparams["conv_dropout"])
    sigoptparams.setdefault(
        "atom_fea_len_target", modelparams["atom_fea_len"]["target"]
    )
    sigoptparams.setdefault("h_fea_len_target", modelparams["h_fea_len"]["target"])
    sigoptparams.setdefault("n_h_target", modelparams["n_h"]["target"])
    sigoptparams.setdefault(
        "readout_dropout_target", modelparams["readout_dropout"]["target"]
    )
    sigoptparams.setdefault("fc_dropout_target", modelparams["fc_dropout"]["target"])
    sigoptparams.setdefault("loss_coeff_target", modelparams["loss_coeff"]["target"])

    # details
    sigoptparams.setdefault("lr", details["lr"])
    sigoptparams.setdefault("weight_decay", details["weight_decay"])
    sigoptparams.setdefault("epochs", details["epochs"])

    converted_params["activation"] = modelparams["activation"]
    converted_params["activation_f"] = modelparams["activation_f"]
    converted_params["learnable_k"] = modelparams["learnable_k"]
    converted_params["n_outputs"] = modelparams["n_outputs"]
    converted_params["cutoff"] = sigoptparams.cutoff
    converted_params["feat_dim"] = sigoptparams.feat_dim
    converted_params["n_rbf"] = sigoptparams.n_rbf
    converted_params["num_conv"] = sigoptparams.num_conv
    converted_params["conv_dropout"] = sigoptparams.conv_dropout
    if fidelity:
        converted_params["atom_fea_len"] = {
            "target": sigoptparams.atom_fea_len_target,
            "fidelity": sigoptparams.atom_fea_len_fidelity,
        }
        converted_params["h_fea_len"] = {
            "target": sigoptparams.h_fea_len_target,
            "fidelity": sigoptparams.h_fea_len_fidelity,
        }
        converted_params["n_h"] = {
            "target": sigoptparams.n_h_target,
            "fidelity": sigoptparams.n_h_fidelity,
        }
        converted_params["readout_dropout"] = {
            "target": sigoptparams.readout_dropout_target,
            "fidelity": sigoptparams.readout_dropout_fidelity,
        }
        converted_params["fc_dropout"] = {
            "target": sigoptparams.fc_dropout_target,
            "fidelity": sigoptparams.fc_dropout_fidelity,
        }
        converted_params["loss_coeff"] = {
            "target": sigoptparams.loss_coeff_target,
            "fidelity": sigoptparams.loss_coeff_fidelity,
        }
    else:
        converted_params["atom_fea_len"] = {
            "target": sigoptparams.atom_fea_len_target,
        }
        converted_params["h_fea_len"] = {
            "target": sigoptparams.h_fea_len_target,
        }
        converted_params["n_h"] = {
            "target": sigoptparams.n_h_target,
        }
        converted_params["readout_dropout"] = {
            "target": sigoptparams.readout_dropout_target,
        }
        converted_params["fc_dropout"] = {
            "target": sigoptparams.fc_dropout_target,
        }

    converted_details["lr"] = {
        "lr": sigoptparams.lr,
    }
    converted_details["weight_decay"] = {
        "weight_decay": sigoptparams.weight_decay,
    }
    return converted_details, converted_params
