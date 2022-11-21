import numpy as np
from tqdm import tqdm


def convert_site_prop(data, output_keys, fidelity_keys=None):
    data_converted = {}
    print("Preprocessing...")
    for key, val in tqdm(data.items()):
        target_key_bin = []
        fidelity_key_bin = []
        fidelity = []
        target = []
        site_prop = val.site_properties
        for i, _ in enumerate(val):
            o_val = []
            for key_o in output_keys:
                target_key_bin.append(key_o)
                if key_o in list(site_prop.keys()) and key_o == "magmom":
                    o_val += [np.abs(site_prop[key_o][i])]
                elif key_o in list(site_prop.keys()) and key_o in ["deltaE", "deltaO", "deltaOH", "deltaOOH"]:
                    E_val = site_prop[key_o][i]
                    if E_val > -5 and E_val < 5:
                        o_val += [E_val]
                    else:
                        o_val += [np.nan]
                elif key_o in list(site_prop.keys()) and key_o not in ["magmom", "deltaE", "deltaO", "deltaOH","deltaOOH"]:
                    o_val += [site_prop[key_o][i]]
                else:
                    ValueError("output key not in the site props")
            target.append(o_val)
            if fidelity_keys is not None:
                f_val = []
                for key_f in fidelity_keys:
                    fidelity_key_bin.append(key_f)
                    if key_f in list(site_prop.keys()) and key_f == "magmom":
                        f_val += [np.abs(site_prop[key_f][i])]
                    elif key_o in list(site_prop.keys()) and key_o in ["deltaE", "deltaO", "deltaOH", "deltaOOH"]:
                        E_val = site_prop[key_f][i]
                        if E_val > -5 and E_val < 5:
                            f_val += [E_val]
                        else:
                            f_val += [np.nan]
                    elif key_f in list(site_prop.keys()) and key_f not in ["magmom", "deltaE", "deltaO", "deltaOH","deltaOOH"]:
                        f_val += [site_prop[key_f][i]]
                    else:
                        ValueError("fidelity key not in the site props")

                fidelity.append(f_val)
        if fidelity_keys is not None:
            converted_site_prop = {"target": target, "fidelity": fidelity}
        else:
            converted_site_prop = {"target": target}

        new_structure = val.copy(site_properties=converted_site_prop)
        data_converted[key] = new_structure

    return data_converted
