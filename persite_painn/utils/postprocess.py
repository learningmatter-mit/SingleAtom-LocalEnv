import torch
import numpy as np
from scipy.stats import norm
from persite_painn.utils.cuda import batch_to
from persite_painn.utils.tools import get_metal_filter

def getdGs_tensor(output):
        #Constants
        #kbt = 0.0256
        #const = kbt*log(10)
        #Contributions to Gibbs Energies for gas molecules (VASP-PBE calculated by Max; T= 300K)
        zpeh2o=0.560    #exp. NIST 0.558
        zpeh2=0.268     #exp. NIST 0.273
        cvh2o=0.103     #From Colin at P = 0.035 bar
        cvh2=0.0905
        tsh2o=0.675     #From Colin at P = 0.035 bar
        tsh2=0.408      #at P = 1bar
        #Contributions to Gibbs Energies for adsorbates (VASP-PBE calculated by Max using Michal data for NiCe; T= 300K)
        #if(PBE_VASP):
        zpeo=0.065 #0.061
        zpeoh=0.344 #0.360
        zpeooh=0.443 #0.468 #0.459 old
        cvo=0.038 #0.0325
        cvoh=0.051 #0.049
        cvooh=0.068 #0.077
        tso=0.080 #0.051        #0.060 From Colin
        tsoh=0.080 #0.085 From Colin
        tsooh=0.116  #0.135     #0.215 From Colin
        #Gibbs Energies for the gas molecules
        dgh2o=zpeh2o +cvh2o -tsh2o
        dgh2=zpeh2 +cvh2 -tsh2
        #Gibbs Energy corrections for adsorbates
        dgo=zpeo +cvo -tso -(dgh2o -dgh2)
        dgoh=zpeoh +cvoh -tsoh -(dgh2o -0.5*dgh2)
        # dgooh=zpeooh +cvooh -tsooh -(2*dgh2o -1.5*dgh2)
        output_dgo = output[:,0]+dgo
        output_dgoh = output[:,1]+dgoh
        output_calculated = torch.stack((output_dgo, output_dgoh), dim= -1)
        return  output_calculated

def getOverpotential_tensor(tensor):
        dgooh = tensor[:,1]*0.899579759974096 + 3.3834055406273777 # Linear fitted for 82.3 % data
        # dgooh = tensor[:,1]+3.20
        #We calculate the OER steps and the overpotential
        dgde_oer = torch.stack((tensor[:,1], tensor[:,0]-tensor[:,1], dgooh-tensor[:,0], 4.92-dgooh), dim=1)
        dgde_orr = torch.stack((dgooh-4.92, tensor[:,0]-dgooh, tensor[:,1]-tensor[:,0], -tensor[:,1]), dim=1)

        max_values_oer = torch.max(dgde_oer, dim=1).values
        max_values_orr = torch.max(dgde_orr, dim=1).values
        maxoer =  max_values_oer - 1.23
        maxorr = 1.23 + max_values_orr

        return maxoer, maxorr


def get_best_target(dataset, s_models, device):
    objective_bin_measured = []

    for data in dataset:
        data =batch_to(data, device)
        metal_filter = get_metal_filter(data)
        out_measured = data["target"][metal_filter]
        dGs_tensor_measured = getdGs_tensor(out_measured)
        oer_output_measured, orr_output_measured = getOverpotential_tensor(dGs_tensor_measured)
        objective_measured = 1 / torch.sqrt(torch.pow(oer_output_measured.unsqueeze(1), 2) + torch.pow(orr_output_measured.unsqueeze(1), 2))
        objective_bin_measured.append(objective_measured)
        
    flattened_measured = torch.cat(objective_bin_measured, dim=0).view(-1)
    nan_filter = torch.isnan(flattened_measured)
    optimal_measured = flattened_measured[~nan_filter].detach()

    return torch.max(optimal_measured).item()

def get_expectedImprovement(unlabelled_data, s_models, ybest, epsilon, device):
    """Calculate the expected improvement
    objective function: 1/sqrt(oer_overpotential^2+orr_overpotential^2)
    Args:
        unlabelled_data (_type_): _description_
        s_models (_type_): _description_
        ybest (_type_): _description_
        epsilon (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """
    objective_bin = []
    for model in s_models:
        model.eval()
        model.device = device
        model.to(device)
        unlabelled_data = batch_to(unlabelled_data, device)
        out = model(unlabelled_data, inference=True)["target"]
        metal_filter = get_metal_filter(unlabelled_data)
        dGs_tensor = getdGs_tensor(out)[metal_filter]
        oer_output, orr_output = getOverpotential_tensor(dGs_tensor)
        objective = 1 / torch.sqrt(torch.pow(oer_output.unsqueeze(1), 2) + torch.pow(orr_output.unsqueeze(1), 2))

        objective_bin.append(objective)


    output_tensor_mean = torch.mean(torch.stack(objective_bin, dim=1), dim=1).squeeze(1)
    output_tensor_var = torch.pow(torch.var(torch.stack(objective_bin, dim=1), dim=1).squeeze(1), 0.5)
    expI = np.empty(output_tensor_mean.size(), dtype=float)
    for ii in range(len(output_tensor_mean)):
        if output_tensor_var[ii] > 0:
            zzval=(output_tensor_mean[ii]-ybest-epsilon)/float(output_tensor_var[ii])
            expI[ii]=(output_tensor_mean[ii]-ybest-epsilon)*norm.cdf(zzval)+output_tensor_var[ii]*norm.pdf(zzval)
        else:
            expI[ii]=0.0


    return expI

def get_ensemble_uncertainty(unlabelled_data, s_models, device, multifidelity=True):
    """Calculate the expected improvement
    objective function: 1/sqrt(oer_overpotential^2+orr_overpotential^2)
    Args:
        unlabelled_data (_type_): _description_
        s_models (_type_): _description_
        ybest (_type_): _description_
        epsilon (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """
    output_bin_targ = []
    output_bin_fildelity = []
    for model in s_models:
        model.eval()
        model.device = device
        model.to(device)
        unlabelled_data = batch_to(unlabelled_data, device)
        out_targ = model(unlabelled_data, inference=True)["target"]
        metal_filter = get_metal_filter(unlabelled_data)
        dGs_tensor = getdGs_tensor(out_targ)[metal_filter]
        output_bin_targ.append(dGs_tensor)
        if multifidelity:
            out_fidelity = model(unlabelled_data, inference=True)["fidelity"]
            output_bin_fildelity.append(out_fidelity)
        # print(out_fidelity.shape)
    
    output_targ_var = torch.max(torch.var(torch.stack(output_bin_targ, dim=1), dim=1, keepdim=True).squeeze(1), dim=-1, keepdim=True)
    if multifidelity:
        output_fidelity_var = torch.max(torch.var(torch.stack(output_bin_fildelity, dim=1), dim=1, keepdim=True).squeeze(1), dim=-1, keepdim=True)
    # print(output_bin_targ, output_bin_fildelity)
        return output_targ_var[0], output_fidelity_var[0]
    else:
         return output_targ_var[0], None