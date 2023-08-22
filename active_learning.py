import argparse
import os
import pickle as pkl
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from persite_painn.data import collate_dicts
from persite_painn.nn.builder import load_model
from persite_painn.utils.cuda import batch_to
from persite_painn.utils.postprocess import (get_best_target,
                                             get_expectedImprovement, get_ensemble_uncertainty)
from persite_painn.utils.tools import get_metal_filter

parser = argparse.ArgumentParser(description="Active Learning")
parser.add_argument("--model_path", default="./", type=str, help="path to raw data")
parser.add_argument(
    "--dataset",
    default="dataset_cache/dataset",
    type=str,
    help="dataset path",
)
parser.add_argument(
    "--total_dataset",
    default="dataset_cache/total_dataset",
    type=str,
    help="total dataset path",
)
parser.add_argument("--device", default="cuda", type=str, help="DEVICE")
parser.add_argument("--cuda", default=3, type=int, help="GPU setting")
parser.add_argument("--save", action='store_true',
    default=False,
    help="Whether to save in pkl",)
parser.add_argument("--plot", action='store_true',
    default=False,
    help="Whether to plot",)
parser.add_argument("--uncertainty_type", default="bayesian", type=str, help="Types of uncertainty sampling bayesian or variance")
parser.add_argument("--get_data", action='store_true',
    default=False,
    help="Whether to get next data",)
parser.add_argument("--uncertainty", action='store_true',
    default=False,
    help="Whether to calculate uncertatinty",)
parser.add_argument("--multifidelity", action='store_true',
    default=False,
    help="Whether to consider multifidelity",)
parser.add_argument("--num_u", default=60, type=int, help='number of uncertainty datapoints')
parser.add_argument("--num_d", default=20, type=int, help='number of diversity datapoints')

def get_num_metals_batch(batch, metal_filter):
    count = 0
    num_metals = []
    for num_atoms in batch["num_atoms"]:
        previous = count
        count += num_atoms.item()
        num_metals.append(sum(metal_filter[previous:count]))
    return num_metals


class ActiveLearning:
    def __init__(self, models_path, dataset, total_dataset, num_model=10, model_type="PainnMultifidelity", batch_size=64, device="cuda"):
        self.dataset = dataset
        self.dataloader = self.make_dataloader(dataset, batch_size=batch_size)
        self.total_dataset = total_dataset
        self.unlabelled_dataset = self.get_unlabelled_data()
        self.unlabelled_dataloader = self.make_dataloader(self.unlabelled_dataset, batch_size=batch_size)
        self.total_dataloader = self.make_dataloader(total_dataset, batch_size=batch_size)
        self.model_list = self.make_model_list(models_path, num_model, model_type)
        self.device = device

    def get_unlabelled_data(self):
        indices_to_keep = []
        key_bin = []
        for data in self.dataset:
            key_bin.append(data['name'].item())
        for i, data in enumerate(self.total_dataset):
            if data['name'].item() not in key_bin:
                indices_to_keep.append(i)
        subset_dataset = Subset(self.total_dataset, indices_to_keep)
        
        return subset_dataset
        
    def make_model_list(self, models_path, num_model, model_type="PainnMultifidelity"):
        model_list = []
        for i in range(num_model):
            path = Path(models_path) / str(i) / "best_model.pth.tar"
            model, _ = load_model(model_path=path, model_type=model_type)
            model_list.append(model)

        return model_list
    
    def make_dataloader(self, dataset, batch_size):
        dataloader =DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=0,
                    collate_fn=collate_dicts,
                )
        return dataloader

    def uncertainty_sampling(self, num_sample):
        new_struc_name = []
        var_bin = []
        total_var_dict = {}
        print("Uncertainty Sampling...")
        for batch in tqdm(self.unlabelled_dataloader):
            metal_filter = get_metal_filter(batch)
            num_metals = get_num_metals_batch(batch, metal_filter)
            var = get_ensemble_uncertainty(batch, self.model_list, self.device)[1][metal_filter]
            for value in var:
                var_bin.append(value[0].item()) 
            batch_var = []
            count = 0
            for val in num_metals:
                var_temp = []
                for item in range(count, count+val):
                    var_temp.append(var[item])
                batch_var.append(var_temp)
                count += val
            for i, val in enumerate(num_metals):
                total_var_dict[batch["name"][i].item()] = batch_var[i]
                for _ in range(val):
                    new_struc_name.append(batch["name"][i].item())
        print(np.array(var_bin).mean())
        uncertainty_sample = {}
        sorted_expI = np.sort(var_bin)
        criterion = sorted_expI[-num_sample:][0]
        print(f"Criterion: {criterion}")
        total_key_bin = []
        for data in self.dataset:
            total_key_bin.append(data['name'].item())
            
        for i, value in enumerate(var_bin):
            if value >= criterion:
                if new_struc_name[i] in total_key_bin:
                    print(new_struc_name[i], value)
                else:
                    uncertainty_sample[new_struc_name[i]] = value
        var_mask = []
        for val in var_bin:
            if val > criterion:
                var_mask.append(True)
            else:
                var_mask.append(False)
        return uncertainty_sample, criterion, var_bin, var_mask, total_var_dict

    def random_sampling(self, num_sample):

        print("Random Sampling...")
        total_key_bin = []
        for data in self.unlabelled_dataset:
            total_key_bin.append(data['name'].item())

        sampled_key_bin = np.random.choice(total_key_bin, num_sample, replace=False)
        return sampled_key_bin


    def bayesian_sampling(self, num_sample, epsilon):
        new_struc_name = []
        expI_bin = []
        total_expI_dict = {}
        print("Bayesian Sampling...")
        best_objective = get_best_target(self.dataloader, self.model_list, self.device)
        for batch in tqdm(self.unlabelled_dataloader):
            metal_filter = get_metal_filter(batch)
            num_metals = get_num_metals_batch(batch, metal_filter)
            expI = get_expectedImprovement(batch, self.model_list, best_objective, epsilon, self.device)
            for value in expI:
                expI_bin.append(value) 
            batch_expI = []
            count = 0
            for val in num_metals:
                expI_temp = []
                for item in range(count, count+val):
                    expI_temp.append(expI[item])
                batch_expI.append(expI_temp)
                count += val
            for i, val in enumerate(num_metals):
                total_expI_dict[batch["name"][i].item()] = batch_expI[i]
                for _ in range(val):
                    new_struc_name.append(batch["name"][i].item())

        bayesian_sample = {}
        sorted_expI = np.sort(expI_bin)
        criterion = sorted_expI[-num_sample:][0]
        print(f"Criterion: {criterion}")
        total_key_bin = []
        for data in self.dataset:
            total_key_bin.append(data['name'].item())
            
        for i, value in enumerate(expI_bin):
            if value >= criterion:
                if new_struc_name[i] in total_key_bin:
                    print(new_struc_name[i], value)
                else:
                    bayesian_sample[new_struc_name[i]] = value
        expI_mask = []
        for val in expI_bin:
            if val > criterion:
                expI_mask.append(True)
            else:
                expI_mask.append(False)
        return bayesian_sample, criterion, expI_bin, expI_mask, total_expI_dict


    def diversity_sampling(self, bayesian_sample, total_expI_dict, criterion, num_sample):
        features = {}
        def get_atom_features(name):
            def hook(model, input, output):
                features[name] = output.detach()
            return hook
        
        print(f"Diversity Sampling...")
        hook_emb_bin = []
        for i in range(len(self.model_list)):
            hook_emb_bin.append(self.model_list[i].readout_block.readoutdict["atom_emb"][-1].register_forward_hook(get_atom_features(f"atom_emb{i}")))

        embedding_cos_bin= []
        embedding_name_bin = []
        for data in tqdm(self.unlabelled_dataset):
            if data['name'].item() in list(bayesian_sample.keys()):
                batch = batch_to(data, self.device)
                metal_filter = get_metal_filter(batch)
                embedding_tmp = []
                for i, model in enumerate(self.model_list):
                    model.eval()
                    model.device = self.device
                    model.to(self.device)
                    _ = model(batch, inference=True)["target"]
                    embedding = features[f"atom_emb{i}"]
                    new_embedding = embedding[metal_filter].unsqueeze(-1)

                    embedding_tmp.append(new_embedding)

                val = torch.mean(torch.cat(embedding_tmp, dim=2), dim=2).squeeze(-1)

                for j, embedding in enumerate(val):
                    expI = total_expI_dict[batch['name'].item()][j]
                    if expI >= criterion:
                        embedding_name_bin.append(batch['name'].item())
                        embedding_cos_bin.append(embedding.cpu().numpy())
        # assume your data is stored in a PyTorch tensor called 'data'
        cosine_sim = torch.tensor(np.array(embedding_cos_bin))

        # compute cosine similarity matrix
        similarity_matrix = torch.matmul(cosine_sim, cosine_sim.T)
        similarity_matrix /= torch.outer(torch.norm(cosine_sim, dim=1), torch.norm(cosine_sim, dim=1))

        # compute average similarity for each data point
        average_similarity = torch.mean(similarity_matrix, dim=1)

        # sort data points by average similarity
        sorted_indices = torch.argsort(average_similarity)
        # select the 20 least similar data points
        sample_indices = sorted_indices[:num_sample]
        sampled_data_name=np.array(embedding_name_bin)[sample_indices]
        return sampled_data_name, similarity_matrix

    def get_atom_embedding(self, dataset):
        features = {}
        def get_atom_features(name):
            def hook(model, input, output):
                features[name] = output.detach()
            return hook

        hook_emb_bin = []
        for i in range(len(self.model_list)):
            hook_emb_bin.append(self.model_list[i].readout_block.readoutdict["atom_emb"][-1].register_forward_hook(get_atom_features(f"atom_emb{i}")))

        embedding_bin= []
        device = "cuda"
        for batch in tqdm(dataset):
            batch = batch_to(batch, device)
            metal_filter = get_metal_filter(batch)
            embedding_tmp = []
            for i, model in enumerate(self.model_list):
                model.eval()
                model.device = device
                model.to(device)
                _ = model(batch, inference=True)["target"]
                embedding = features[f"atom_emb{i}"]
                new_embedding = embedding[metal_filter].unsqueeze(-1)
                # print(new_embedding.shape, oer_output.unsqueeze(-1).shape, orr_output.unsqueeze(-1).shape)
                embedding_tmp.append(new_embedding)

            val = torch.mean(torch.cat(embedding_tmp, dim=2), dim=2).squeeze(-1)

            for embedding in val:
                embedding_bin.append(embedding.cpu().numpy())

        for hook_emb in hook_emb_bin:
            hook_emb.remove()
        embedding_bin = np.asarray(embedding_bin)

        return embedding_bin
    
    
    def get_next_data(self, num_uncertainty, num_diversity, epsilon=0.01, save=False, plot=False, uncertainty_type="bayesian"):
        if uncertainty_type == "bayesian":
            bayesian_sample, criterion, expI_bin, expI_mask, total_expI_dict = self.bayesian_sampling(num_sample=num_uncertainty, epsilon=epsilon)
            label_name = "$EI$"
            sampled_data_name, similarity_matrix = self.diversity_sampling(bayesian_sample=bayesian_sample,total_expI_dict=total_expI_dict, criterion=criterion, num_sample=num_diversity)
            print(len(bayesian_sample), len(set(bayesian_sample)))
        elif uncertainty_type == "variance":
            bayesian_sample, criterion, expI_bin, expI_mask, total_expI_dict = self.uncertainty_sampling(num_sample=num_uncertainty)
            label_name = "Var"
            sampled_data_name, similarity_matrix = self.diversity_sampling(bayesian_sample=bayesian_sample,total_expI_dict=total_expI_dict, criterion=criterion, num_sample=num_diversity)
            print(len(bayesian_sample), len(set(bayesian_sample)))
        elif uncertainty_type == "random":
            sampled_data_name = self.random_sampling(num_sample=num_uncertainty)
            print(len(sampled_data_name), len(set(sampled_data_name)))
        if save and uncertainty_type != "random":
            pkl.dump(list(bayesian_sample.keys()), open("uncertainty_keys.pkl", 'wb'))
            pkl.dump(list(bayesian_sample.values()), open("uncertainty_values.pkl", 'wb'))
            pkl.dump(list(sampled_data_name), open("new_data_keys.pkl", 'wb'))
        elif save and uncertainty_type == "random":
            pkl.dump(list(sampled_data_name), open("new_data_keys_random.pkl", 'wb'))
        if plot:
            print("Plot Results...")
            embedding_bin = self.get_atom_embedding(self.unlabelled_dataloader)
            self.plot_uncertainty(embedding_bin, expI_bin, expI_mask, label_name=label_name)
            self.plot_diversity(similarity_matrix)

        return sampled_data_name

    def get_uncertainty(self, multifidelity):
        uncertainty_target_bin = []
        uncertainty_fidelity_bin = []
        for batch in tqdm(self.unlabelled_dataloader):
            metal_filter = get_metal_filter(batch)
            target_var_bin, fidelity_var_bin = get_ensemble_uncertainty(batch, self.model_list, self.device, multifidelity)
            for val in target_var_bin.view(-1):
                uncertainty_target_bin.append(val.item())
            if multifidelity:
                for val in fidelity_var_bin[metal_filter].view(-1):
                    uncertainty_fidelity_bin.append(val.item())
        if multifidelity:
            return np.array(uncertainty_target_bin).mean(), np.array(uncertainty_fidelity_bin).mean()
        else:
            return np.array(uncertainty_target_bin).mean(), None
        
    def plot_uncertainty(self, embedding_bin, expI_bin, expI_mask, label_name="$EI$", plot_name="uncertainty.png"):
        X_multifidelity = np.array(embedding_bin)

        pca = PCA(n_components=8).fit(X_multifidelity)
        df_pca = pca.transform(X_multifidelity)
        with plt.style.context("scifig"):
            fig = plt.figure()
            cm = plt.cm.get_cmap("YlGnBu")
            ax = fig.add_subplot()
            ax.scatter(df_pca[:,0], df_pca[:,1], s=1.0, alpha=0.8, c = expI_bin, cmap=cm)
            im = ax.scatter(np.array(df_pca[:,0])[expI_mask], np.array(df_pca[:,1])[expI_mask], s=2.0, c = np.array(expI_bin)[expI_mask], cmap=cm)
            # im.set_clim(0.00,0.10)
            fig.colorbar(im, ax=ax, label=label_name)
            ax.set_xlabel("Principal component 1")
            ax.set_ylabel("Principal component 2")
            ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(plot_name, dpi=500)

    def plot_diversity(self, similarity_matrix, plot_name="diversity.png"):
        with plt.style.context("scifig"):
            fig = plt.figure()
            ax = fig.add_subplot()
            im = ax.imshow(similarity_matrix, cmap='YlGnBu_r', interpolation='nearest', vmax=1)
            fig.colorbar(im, ax=ax, label="Cosine Similarity")
            ax.set_xticklabels("")
            ax.set_yticklabels("")
        fig.tight_layout()
        fig.savefig(plot_name, dpi=500)


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
        assert torch.cuda.is_available(), "cuda is not available"

    model_path = args.model_path
    print("Loading dataset...")
    dataset = torch.load(args.dataset)
    total_dataset = torch.load(args.total_dataset)
    if args.multifidelity:
        model_type = "PainnMultifidelity"
    else:
        model_type = "Painn"
    print(model_type)
    activelearning = ActiveLearning(models_path=model_path, dataset=dataset, total_dataset=total_dataset, model_type=model_type)
    if args.get_data:
        # new_data = activelearning.get_next_data(60, 20, save=args.save, plot=args.plot, uncertainty_type=args.uncertainty_type)
        # new_data = activelearning.get_next_data(30, 15, save=args.save, plot=args.plot, uncertainty_type=args.uncertainty_type)
        new_data = activelearning.get_next_data(args.num_u, args.num_d, save=args.save, plot=args.plot, uncertainty_type=args.uncertainty_type)
        print(f"Next Samples: {new_data}")
        print(f"Next Samples in list: {list(new_data)}")
    if args.uncertainty:
        uncertainty_targ, uncertainty_fidel = activelearning.get_uncertainty(multifidelity=args.multifidelity)
        print(f"Uncertainty {uncertainty_targ}, {uncertainty_fidel}")