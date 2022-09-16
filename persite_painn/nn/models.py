import torch
from nff.nn.layers import Dense
from nff.nn.modules.painn import EmbeddingBlock, MessageBlock, UpdateBlock
from nff.nn.modules.painn import ReadoutBlock as org_ReadoutBlock
from nff.nn.modules.schnet import (
    AttentionPool,
    MeanPool,
    MolFpPool,
    add_stress,
    get_rij,
)
from nff.utils.scatter import scatter_add
from nff.utils.tools import make_directed
from persite_painn.nn.modules import (
    AtomwiseReadoutBlock,
    nn_exp,
    to_module,
    ReadoutBlock_,
    SumPool,
)
from torch import nn

POOL_DIC = {
    "sum": SumPool,
    "mean": MeanPool,
    "attention": AttentionPool,
    "mol_fp": MolFpPool,
}


class Painn(nn.Module):
    def __init__(self, modelparams):
        """
        Args:
            modelparams (dict): dictionary of model parameters



        """

        super().__init__()

        feat_dim = modelparams["feat_dim"]
        activation = modelparams["activation"]
        n_rbf = modelparams["n_rbf"]
        cutoff = modelparams["cutoff"]
        num_conv = modelparams["num_conv"]
        output_keys = modelparams["output_keys"]
        learnable_k = modelparams.get("learnable_k", False)
        conv_dropout = modelparams.get("conv_dropout", 0)
        readout_dropout = modelparams.get("readout_dropout", 0)
        means = modelparams.get("means")
        stddevs = modelparams.get("stddevs")
        pool_dic = modelparams.get("pool_dic")

        self.embed_block = EmbeddingBlock(feat_dim=feat_dim)
        self.message_blocks = nn.ModuleList(
            [
                MessageBlock(
                    feat_dim=feat_dim,
                    activation=activation,
                    n_rbf=n_rbf,
                    cutoff=cutoff,
                    learnable_k=learnable_k,
                    dropout=conv_dropout,
                )
                for _ in range(num_conv)
            ]
        )
        self.update_blocks = nn.ModuleList(
            [
                UpdateBlock(
                    feat_dim=feat_dim, activation=activation, dropout=conv_dropout
                )
                for _ in range(num_conv)
            ]
        )

        self.output_keys = output_keys
        # no skip connection in original paper
        self.skip = modelparams.get(
            "skip_connection", {key: False for key in self.output_keys}
        )

        num_readouts = num_conv if any(self.skip.values()) else 1
        self.readout_blocks = nn.ModuleList(
            [
                org_ReadoutBlock(
                    feat_dim=feat_dim,
                    output_keys=output_keys,
                    activation=activation,
                    dropout=readout_dropout,
                    means=means,
                    stddevs=stddevs,
                )
                for _ in range(num_readouts)
            ]
        )

        if pool_dic is None:
            self.pool_dic = {key: SumPool() for key in self.output_keys}
        else:
            self.pool_dic = nn.ModuleDict({})
            for out_key, sub_dic in pool_dic.items():
                if out_key not in self.output_keys:
                    continue
                pool_name = sub_dic["name"].lower()
                kwargs = sub_dic["param"]
                pool_class = POOL_DIC[pool_name]
                self.pool_dic[out_key] = pool_class(**kwargs)

        self.compute_delta = modelparams.get("compute_delta", False)
        self.cutoff = cutoff

    def set_cutoff(self):
        if hasattr(self, "cutoff"):
            return
        msg = self.message_blocks[0]
        dist_embed = msg.inv_message.dist_embed
        self.cutoff = dist_embed.f_cut.cutoff

    def atomwise(self, batch, xyz=None):

        # for backwards compatability
        if isinstance(self.skip, bool):
            self.skip = {key: self.skip for key in self.output_keys}

        nbrs, _ = make_directed(batch["nbr_list"])
        nxyz = batch["nxyz"]

        if xyz is None:
            xyz = nxyz[:, 1:]
            xyz.requires_grad = True

        z_numbers = nxyz[:, 0].long()

        # get r_ij including offsets and excluding
        # anything in the neighbor skin
        self.set_cutoff()
        r_ij, nbrs = get_rij(xyz=xyz, batch=batch, nbrs=nbrs, cutoff=self.cutoff)

        s_i, v_i = self.embed_block(z_numbers, nbrs=nbrs, r_ij=r_ij)
        results = {}

        for i, message_block in enumerate(self.message_blocks):
            update_block = self.update_blocks[i]
            ds_message, dv_message = message_block(
                s_j=s_i, v_j=v_i, r_ij=r_ij, nbrs=nbrs
            )

            s_i = s_i + ds_message
            v_i = v_i + dv_message

            ds_update, dv_update = update_block(s_i=s_i, v_i=v_i)

            s_i = s_i + ds_update
            v_i = v_i + dv_update

            if not any(self.skip.values()):
                continue

            readout_block = self.readout_blocks[i]
            new_results = readout_block(s_i=s_i)
            for key, skip in self.skip.items():
                if not skip:
                    continue
                if key not in new_results:
                    continue
                if key in results:
                    results[key] += new_results[key]
                else:
                    results[key] = new_results[key]

        if not all(self.skip.values()):
            first_readout = self.readout_blocks[0]
            new_results = first_readout(s_i=s_i)
            for key, skip in self.skip.items():
                if key not in new_results:
                    continue
                if not skip:
                    results[key] = new_results[key]

        results["features"] = s_i

        return results, xyz, r_ij, nbrs

    def pool(self, batch, atomwise_out, xyz, r_ij, nbrs, inference=False):

        # import here to avoid circular imports
        from nff.train import batch_detach

        if not hasattr(self, "output_keys"):
            self.output_keys = list(self.readout_blocks[0].readoutdict.keys())

        if not hasattr(self, "pool_dic"):
            self.pool_dic = {key: SumPool() for key in self.output_keys}

        all_results = {}

        for key, pool_obj in self.pool_dic.items():
            grad_key = f"{key}_grad"
            grad_keys = [grad_key] if (grad_key in self.grad_keys) else []
            if "stress" in self.grad_keys and not "stress" in all_results:
                grad_keys.append("stress")
            results = pool_obj(
                batch=batch,
                xyz=xyz,
                r_ij=r_ij,
                nbrs=nbrs,
                atomwise_output=atomwise_out,
                grad_keys=grad_keys,
                out_keys=[key],
            )

            if inference:
                results = batch_detach(results)
            all_results.update(results)

        return all_results, xyz

    def add_delta(self, all_results):
        for i, e_i in enumerate(self.output_keys):
            if i == 0:
                continue
            e_j = self.output_keys[i - 1]
            key = f"{e_i}_{e_j}_delta"
            all_results[key] = all_results[e_i] - all_results[e_j]
        return all_results

    def V_ex(self, r_ij, nbr_list, xyz):

        dist = (r_ij).pow(2).sum(1).sqrt()
        potential = (dist.reciprocal() * self.sigma).pow(self.power)

        return scatter_add(potential, nbr_list[:, 0], dim_size=xyz.shape[0])[:, None]

    def run(self, batch, xyz=None, requires_stress=False, inference=False):

        atomwise_out, xyz, r_ij, nbrs = self.atomwise(batch=batch, xyz=xyz)

        if getattr(self, "excl_vol", None):
            # Excluded Volume interactions
            r_ex = self.V_ex(r_ij, nbrs, xyz)
            atomwise_out["energy"] += r_ex

        all_results, xyz = self.pool(
            batch=batch,
            atomwise_out=atomwise_out,
            xyz=xyz,
            r_ij=r_ij,
            nbrs=nbrs,
            inference=inference,
        )

        if requires_stress:
            all_results = add_stress(
                batch=batch, all_results=all_results, nbrs=nbrs, r_ij=r_ij
            )

        if getattr(self, "compute_delta", False):
            all_results = self.add_delta(all_results)

        return all_results, xyz

    def forward(
        self, batch, xyz=None, requires_stress=False, inference=False, **kwargs
    ):
        """
        Call the model
        Args:
            batch (dict): batch dictionary
        Returns:
            results (dict): dictionary of predictions
        """

        results, _ = self.run(
            batch=batch, xyz=xyz, requires_stress=requires_stress, inference=inference
        )

        return results


class PainnAtomwise(Painn):
    def __init__(self, modelparams):
        super().__init__(modelparams)
        feat_dim = modelparams["feat_dim"]
        activation = modelparams["activation"]
        self.output_keys = modelparams["output_keys"]
        output_atom_fea_dim = modelparams["atom_fea_len"]
        num_conv = modelparams["num_conv"]
        readout_dropout = modelparams.get("readout_dropout", 0)
        fc_dropout = modelparams.get("fc_dropout", 0)
        # means = modelparams.get("means")
        # stddevs = modelparams.get("stddevs")
        num_readouts = num_conv if any(self.skip.values()) else 1
        self.readout_blocks = nn.ModuleList(
            [
                AtomwiseReadoutBlock(
                    feat_dim=feat_dim,
                    output_atom_fea=output_atom_fea_dim,
                    output_keys=self.output_keys,
                    activation=activation,
                    dropout=readout_dropout,
                )
                for _ in range(num_readouts)
            ]
        )
        n_h = modelparams["n_h"]
        h_fea_len = modelparams["h_fea_len"]
        n_density = modelparams["n_density"]
        self.conv_to_fc = nn.Linear(output_atom_fea_dim, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()

        if n_h > 1:
            self.fcs = nn.ModuleList(
                [
                    Dense(
                        in_features=h_fea_len,
                        out_features=h_fea_len,
                        bias=True,
                        dropout_rate=fc_dropout,
                        activation=to_module("softplus"),
                    )
                    for _ in range(n_h - 1)
                ]
            )
        self.force_positive = modelparams.get("force_positive", True)
        if self.force_positive:
            self.fc_out = nn.Linear(h_fea_len, n_density)
            self.fc_out_act = nn_exp()
        else:
            self.fc_out = nn.Linear(h_fea_len, n_density)

    def atomwise(self, batch, xyz=None):

        # for backwards compatability
        if isinstance(self.skip, bool):
            self.skip = {key: self.skip for key in self.output_keys}

        nbrs, _ = make_directed(batch["nbr_list"])
        nxyz = batch["nxyz"]

        if xyz is None:
            xyz = nxyz[:, 1:]
            xyz.requires_grad = True

        z_numbers = nxyz[:, 0].long()

        # get r_ij including offsets and excluding
        # anything in the neighbor skin
        self.set_cutoff()
        r_ij, nbrs = get_rij(xyz=xyz, batch=batch, nbrs=nbrs, cutoff=self.cutoff)

        s_i, v_i = self.embed_block(z_numbers, nbrs=nbrs, r_ij=r_ij)
        results = {}

        for i, message_block in enumerate(self.message_blocks):
            update_block = self.update_blocks[i]
            ds_message, dv_message = message_block(
                s_j=s_i, v_j=v_i, r_ij=r_ij, nbrs=nbrs
            )

            s_i = s_i + ds_message
            v_i = v_i + dv_message

            ds_update, dv_update = update_block(s_i=s_i, v_i=v_i)

            s_i = s_i + ds_update
            v_i = v_i + dv_update

            if not any(self.skip.values()):
                continue
            readout_block = self.readout_blocks[i]
            new_results = readout_block(s_i=s_i)
            for key, skip in self.skip.items():
                if not skip:
                    continue
                if key not in new_results:
                    continue
                if key in results:
                    results[key] += new_results[key]
                else:
                    results[key] = new_results[key]

        if not all(self.skip.values()):
            first_readout = self.readout_blocks[0]
            new_results = first_readout(s_i=s_i)
            for key, skip in self.skip.items():
                if key not in new_results:
                    continue
                if not skip:
                    results[key] = new_results[key]

        results["features"] = s_i

        return results, xyz, r_ij, nbrs

    def run(self, batch, xyz=None, requires_stress=False, inference=False):

        atomwise_out, xyz, r_ij, nbrs = self.atomwise(batch=batch, xyz=xyz)
        results = {}
        for key, val in atomwise_out.items():
            if key in self.output_keys:
                # Atom_fea to fc layers
                val = self.conv_to_fc_softplus(self.conv_to_fc(val))
                if hasattr(self, "fcs"):
                    for fc in self.fcs:
                        val = fc(val)
                out = self.fc_out(val)
                if self.force_positive:
                    out = self.fc_out_act(out)
                    results[key] = out
                else:
                    results[key] = out

        return results, xyz


class PainnMultifidelity(Painn):
    def __init__(self, modelparams):
        super().__init__(modelparams)
        feat_dim = modelparams["feat_dim"]
        activation = modelparams["activation"]
        self.output_keys = modelparams["output_keys"]
        output_atom_fea_dim = modelparams["atom_fea_len"]
        num_conv = modelparams["num_conv"]
        readout_dropout = modelparams.get("readout_dropout", 0)
        fc_dropout = modelparams.get("fc_dropout", 0)
        means = modelparams.get("means")
        stddevs = modelparams.get("stddevs")
        num_readouts = num_conv if any(self.skip.values()) else 1
        self.readout_blocks = nn.ModuleList(
            [
                ReadoutBlock_(
                    feat_dim=feat_dim,
                    output_atom_fea=output_atom_fea_dim,
                    output_keys=self.output_keys,
                    activation=activation,
                    dropout=readout_dropout,
                    means=means,
                    stddevs=stddevs,
                )
                for _ in range(num_readouts)
            ]
        )
        n_h = modelparams["n_h"]
        h_fea_len = modelparams["h_fea_len"]
        n_density = modelparams["n_density"]
        self.conv_to_fc = nn.Linear(output_atom_fea_dim + 3, h_fea_len)
        # self.conv_to_fc = nn.Linear(output_atom_fea_dim, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()

        self.pool_dic = {key: SumPool() for key in self.output_keys}
        if n_h > 1:
            self.fcs = nn.ModuleList(
                [
                    Dense(
                        in_features=h_fea_len,
                        out_features=h_fea_len,
                        bias=True,
                        dropout_rate=fc_dropout,
                        activation=to_module("softplus"),
                    )
                    for _ in range(n_h - 1)
                ]
            )
        self.force_positive = modelparams.get("force_positive", True)
        if self.force_positive:
            self.fc_out = nn.Linear(h_fea_len, n_density)
            self.fc_out_act = nn_exp()
        else:
            self.fc_out = nn.Linear(h_fea_len, n_density)

    def atomwise(self, batch, xyz=None):

        # for backwards compatability
        if isinstance(self.skip, bool):
            self.skip = {key: self.skip for key in self.output_keys}

        nbrs, _ = make_directed(batch["nbr_list"])
        nxyz = batch["nxyz"]

        if xyz is None:
            xyz = nxyz[:, 1:]
            xyz.requires_grad = True

        z_numbers = nxyz[:, 0].long()

        # get r_ij including offsets and excluding
        # anything in the neighbor skin
        self.set_cutoff()
        r_ij, nbrs = get_rij(xyz=xyz, batch=batch, nbrs=nbrs, cutoff=self.cutoff)

        s_i, v_i = self.embed_block(z_numbers, nbrs=nbrs, r_ij=r_ij)
        results = {}

        for i, message_block in enumerate(self.message_blocks):
            update_block = self.update_blocks[i]
            ds_message, dv_message = message_block(
                s_j=s_i, v_j=v_i, r_ij=r_ij, nbrs=nbrs
            )

            s_i = s_i + ds_message
            v_i = v_i + dv_message

            ds_update, dv_update = update_block(s_i=s_i, v_i=v_i)

            s_i = s_i + ds_update
            v_i = v_i + dv_update

            if not any(self.skip.values()):
                continue

            readout_block = self.readout_blocks[i]
            new_results = readout_block(s_i=s_i)
            for key, skip in self.skip.items():
                if not skip:
                    continue
                if key not in new_results:
                    continue
                if key in results:
                    results[key] += new_results[key]
                else:
                    results[key] = new_results[key]

        if not all(self.skip.values()):
            first_readout = self.readout_blocks[0]
            new_results = first_readout(s_i=s_i)
            for key, skip in self.skip.items():
                if key not in new_results:
                    continue
                if not skip:
                    results[key] = new_results[key]

        results["features"] = s_i

        return results, xyz, r_ij, nbrs

    def pool(self, batch, atomwise_out, xyz, r_ij, nbrs, inference=False):

        # import here to avoid circular imports
        from nff.train import batch_detach

        if not hasattr(self, "output_keys"):
            self.output_keys = list(self.readout_blocks[0].readoutdict.keys())

        if not hasattr(self, "pool_dic"):
            self.pool_dic = {key: SumPool() for key in self.output_keys}

        all_results = {}

        for key, pool_obj in self.pool_dic.items():
            grad_key = f"{key}_grad"
            grad_keys = [grad_key] if (grad_key in self.grad_keys) else []
            if "stress" in self.grad_keys and not "stress" in all_results:
                grad_keys.append("stress")
            results = pool_obj(
                batch=batch,
                xyz=xyz,
                atomwise_output=atomwise_out,
                grad_keys=grad_keys,
                out_keys=[key],
            )

            if inference:
                results = batch_detach(results)
            all_results.update(results)

        return all_results, xyz

    def run(self, batch, xyz=None, requires_stress=False, inference=False):

        atomwise_out, xyz, r_ij, nbrs = self.atomwise(batch=batch, xyz=xyz)

        persite_props = batch["site_prop"]

        new_atomwise_out = {}
        for key, val in atomwise_out.items():
            new_val = torch.cat((persite_props, val), dim=1)
            new_atomwise_out[key] = new_val
        all_results, xyz = self.pool(
            batch=batch,
            xyz=xyz,
            r_ij=r_ij,
            atomwise_out=new_atomwise_out,
            nbrs=nbrs,
            inference=inference,
        )
        for key, val in all_results.items():
            if key in self.output_keys:
                val = self.conv_to_fc_softplus(self.conv_to_fc(val))
                if hasattr(self, "fcs"):
                    for fc in self.fcs:
                        val = fc(val)
                out = self.fc_out(val)
                if self.force_positive:
                    out = self.fc_out_act(out)
                    new_atomwise_out[key] = out
                else:
                    new_atomwise_out[key] = out
            else:
                pass

        return new_atomwise_out, xyz
