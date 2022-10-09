import numpy as np
from tqdm import tqdm


def convert_site_prop(data, output_keys, fidelity_keys=None):
    data_converted = {}
    print("Preprocessing...")
    for key, val in tqdm(data.items()):
        fidelity = []
        target = []
        site_prop = val.site_properties
        for i, _ in enumerate(val):
            o_val = []
            for key_o in output_keys:
                if key_o == "magmom":
                    if key_o in list(site_prop.keys()):
                        o_val += [np.abs(site_prop[key_o][i])]
                    else:
                        o_val += [np.nan]
                elif key_o == "deltaE_O" or key_o == "deltaE_OH":
                    if key_o in list(site_prop.keys()):
                        E_val = site_prop[key_o][i]
                        if E_val > -5 and E_val < 5:
                            o_val += [E_val]
                        else:
                            o_val += [E_val]
                    else:
                        o_val += [np.nan]
                else:
                    if key_o in list(site_prop.keys()):
                        o_val += [site_prop[key_o][i]]
                    else:
                        o_val += [np.nan]
                target.append(o_val)
            if fidelity_keys is not None:
                f_val = []
                for key_f in fidelity_keys:
                    if key_f == "magmom":
                        if key_f in list(site_prop.keys()):
                            f_val += [np.abs(site_prop[key_f][i])]
                        else:
                            f_val += [np.nan]
                    elif key_f == "deltaE_O" or key_f == "deltaE_OH":
                        if key_f in list(site_prop.keys()):
                            E_val = site_prop[key_f][i]
                            if E_val > -5 and E_val < 5:
                                f_val += [E_val]
                            else:
                                f_val += [E_val]
                        else:
                            f_val += [E_val]
                    else:
                        if key_f in list(site_prop.keys()):
                            f_val += [site_prop[key_f][i]]
                        else:
                            f_val += [np.nan]
                    fidelity.append(f_val)
        if fidelity_keys is not None:
            converted_site_prop = {"target": target, "fidelity": fidelity}
        else:
            converted_site_prop = {"target": target}

        new_structure = val.copy(site_properties=converted_site_prop)
        data_converted[key] = new_structure

    return data_converted
