import torch
from persite_painn.utils.tools import make_directed, get_rij
from persite_painn.nn.modules import (
    SumPool,
    EmbeddingBlock,
    MessageBlock,
    UpdateBlock,
    ReadoutBlock,
    FullyConnected,
)
from torch import nn

POOL_DIC = {
    "sum": SumPool,
}


class Painn(nn.Module):

    def __init__(self, modelparams, **kwargs):
        """
        Args:
            modelparams (dict): dictionary of model parameters

        """

        super().__init__()

        feat_dim = modelparams["feat_dim"]
        activation = modelparams["activation"]
        activation_f = modelparams["activation_f"]
        n_rbf = modelparams["n_rbf"]
        cutoff = modelparams["cutoff"]
        num_conv = modelparams["num_conv"]
        self.output_keys = modelparams["output_keys"]
        learnable_k = modelparams.get("learnable_k", False)
        conv_dropout = modelparams.get("conv_dropout", 0)
        readout_dropout = modelparams.get("readout_dropout", 0)
        fc_dropout = modelparams.get("fc_dropout", 0)
        means = modelparams.get("means")
        stddevs = modelparams.get("stddevs")
        pool_dic = modelparams.get("pool_dic")
        self.site_prediction = kwargs["site_prediction"]
        self.multifideltiy = kwargs["multifidelity"]
        output_atom_fea_dim = modelparams["atom_fea_len"]
        num_conv = modelparams["num_conv"]

        self.embed_block = EmbeddingBlock(feat_dim=feat_dim)
        self.message_blocks = nn.ModuleList([
            MessageBlock(
                feat_dim=feat_dim,
                activation=activation,
                n_rbf=n_rbf,
                cutoff=cutoff,
                learnable_k=learnable_k,
                dropout=conv_dropout,
            ) for _ in range(num_conv)
        ])
        self.update_blocks = nn.ModuleList([
            UpdateBlock(feat_dim=feat_dim,
                        activation=activation,
                        dropout=conv_dropout) for _ in range(num_conv)
        ])
        if self.multifideltiy:
            self.readout_block = ReadoutBlock(
                feat_dim=feat_dim,
                output_atom_fea=output_atom_fea_dim,
                output_keys=self.output_keys,
                activation=activation,
                dropout=readout_dropout,
                means=means,
                stddevs=stddevs,
                scale=True,
            )
        else:
            self.readout_block = ReadoutBlock(
                feat_dim=feat_dim,
                output_atom_fea=output_atom_fea_dim,
                output_keys=self.output_keys,
                activation=activation,
                dropout=readout_dropout,
                means=means,
                stddevs=stddevs,
                scale=False,
            )

        # Fully connected layers
        n_h = modelparams["n_h"]
        h_fea_len = modelparams["h_fea_len"]
        n_outputs = modelparams["n_outputs"]
        self.force_positive = kwargs["spectra"]
        if self.multifideltiy:
            n_fidelity = modelparams["n_fidelity"]
            self.fullyconnected = FullyConnected(
                output_atom_fea_dim=output_atom_fea_dim + n_fidelity,
                h_fea_len=h_fea_len,
                n_h=n_h,
                activation=activation_f,
                n_outputs=n_outputs,
                dropout=fc_dropout,
                force_positive=self.force_positive,
            )
        else:
            self.fullyconnected = FullyConnected(
                output_atom_fea_dim=output_atom_fea_dim,
                h_fea_len=h_fea_len,
                n_h=n_h,
                activation=activation_f,
                n_outputs=n_outputs,
                dropout=fc_dropout,
                force_positive=self.force_positive,
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

        self.cutoff = cutoff

    def set_cutoff(self):
        if hasattr(self, "cutoff"):
            return
        msg = self.message_blocks[0]
        dist_embed = msg.inv_message.dist_embed
        self.cutoff = dist_embed.f_cut.cutoff

    def atomwise(self, batch, xyz=None):
        nbrs, _ = make_directed(batch["nbr_list"])
        nxyz = batch["nxyz"]

        if xyz is None:
            xyz = nxyz[:, 1:]
            xyz.requires_grad = True

        z_numbers = nxyz[:, 0].long()

        # get r_ij including offsets and excluding
        # anything in the neighbor skin
        self.set_cutoff()
        r_ij, nbrs = get_rij(xyz=xyz,
                             batch=batch,
                             nbrs=nbrs,
                             cutoff=self.cutoff)

        s_i, v_i = self.embed_block(z_numbers, nbrs=nbrs, r_ij=r_ij)
        results = {}

        for i, message_block in enumerate(self.message_blocks):
            update_block = self.update_blocks[i]
            ds_message, dv_message = message_block(s_j=s_i,
                                                   v_j=v_i,
                                                   r_ij=r_ij,
                                                   nbrs=nbrs)

            s_i = s_i + ds_message
            v_i = v_i + dv_message

            ds_update, dv_update = update_block(s_i=s_i, v_i=v_i)

            s_i = s_i + ds_update
            v_i = v_i + dv_update

        new_results = self.readout_block(s_i=s_i)

        results.update(new_results)

        return results, xyz, r_ij, nbrs

    def pool(self, batch, atomwise_out):
        if not hasattr(self, "output_keys"):
            self.output_keys = list(self.readout_block.readoutdict.keys())

        if not hasattr(self, "pool_dic"):
            self.pool_dic = {key: SumPool() for key in self.output_keys}

        all_results = {}

        for key, pool_obj in self.pool_dic.items():
            results = pool_obj(
                batch=batch,
                atomwise_output=atomwise_out,
                out_keys=[key],
            )

            all_results.update(results)

        return all_results

    def run(self, batch, xyz=None, inference=False):

        atomwise_out, xyz, _, _ = self.atomwise(batch=batch, xyz=xyz)
        if self.multifideltiy:
            persite_props = batch["site_prop"]

            new_atomwise_out = {}
            for key, val in atomwise_out.items():
                new_val = torch.cat((persite_props, val), dim=1)
                new_atomwise_out[key] = new_val
            atomwise_out = new_atomwise_out
        if not self.site_prediction:
            all_results = self.pool(
                batch=batch,
                atomwise_out=atomwise_out,
            )
        else:
            all_results = atomwise_out
        results = {}
        for key, val in all_results.items():
            out = self.fullyconnected(val)
            results[key] = out

        if inference:
            # import here to avoid circular imports
            from persite_painn.utils.cuda import batch_detach

            results = batch_detach(results)

        return results

    def forward(self, batch, xyz=None, inference=False, **kwargs):
        """
        Call the model
        Args:
            batch (dict): batch dictionary
        Returns:
            results (dict): dictionary of predictions
        """

        results = self.run(batch=batch, xyz=xyz, inference=inference)

        return results
