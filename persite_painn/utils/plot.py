import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import numpy as np
import matplotlib


def plot_hexbin(
    targ,
    pred,
    key,
    title="",
    scale="linear",
    inc_factor=1.1,
    dec_factor=0.9,
    bins=None,
    plot_helper_lines=False,
    cmap="viridis",
):

    props = {
        "center_diff": "B 3d $-$ O 2p difference",
        "op": "O 2p $-$ $E_v$",
        "form_e": "formation energy",
        "e_hull": "energy above hull",
        "tot_e": "energy per atom",
        "time": "runtime",
        "magmom": "magnetic moment",
        "magmom_abs": "magnetic moment",
        "ads_e": "adsorption energy",
        "acid_stab": "electrochemical stability",
        "bandcenter": "DOS band center",
        "phonon": "atomic vibration frequency",
        "bader": "Bader charge",
    }

    units = {
        "center_diff": "eV",
        "op": "eV",
        "form_e": "eV",
        "e_hull": "eV/atom",
        "tot_e": "eV/atom",
        "time": "s",
        "magmom": "$\mu_B$",
        "magmom_abs": "|$\mu_B$|",
        "ads_e": "eV",
        "acid_stab": "eV/atom",
        "bandcenter": "eV",
        "phonon": "THz",
        "bader": "$q_e$",
    }

    fig, ax = plt.subplots(figsize=(6, 6))

    mae = mean_absolute_error(targ, pred)
    r, _ = pearsonr(targ, pred)

    if scale == "log":
        pred = np.abs(pred) + 1e-8
        targ = np.abs(targ) + 1e-8

    lim_min = min(np.min(pred), np.min(targ))
    if lim_min < 0:
        if lim_min > -0.1:
            lim_min = -0.1
        lim_min *= inc_factor
    else:
        if lim_min < 0.1:
            lim_min = -0.1
        lim_min *= dec_factor
    lim_max = max(np.max(pred), np.max(targ))
    if lim_max <= 0:
        if lim_max > -0.1:
            lim_max = 0.2
        lim_max *= dec_factor
    else:
        if lim_max < 0.1:
            lim_max = 0.25
        lim_max *= inc_factor

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal")

    # ax.plot((lim_min, lim_max),
    #        (lim_min, lim_max),
    #        color='#000000',
    #        zorder=-1,
    #        linewidth=0.5)
    ax.axline((0, 0), (1, 1), color="#000000", zorder=-1, linewidth=0.5)

    hb = ax.hexbin(
        targ,
        pred,
        cmap=cmap,
        gridsize=60,
        bins=bins,
        mincnt=1,
        edgecolors=None,
        linewidths=(0.1,),
        xscale=scale,
        yscale=scale,
        extent=(lim_min, lim_max, lim_min, lim_max),
        norm=matplotlib.colors.LogNorm(),
    )

    cb = fig.colorbar(hb, shrink=0.822)
    cb.set_label("Count")

    if plot_helper_lines:

        if scale == "linear":
            x = np.linspace(lim_min, lim_max, 50)
            y_up = x + mae
            y_down = x - mae

        elif scale == "log":
            x = np.logspace(np.log10(lim_min), np.log10(lim_max), 50)

            # one order of magnitude
            y_up = np.maximum(x + 1e-2, x * 10)
            y_down = np.minimum(np.maximum(1e-8, x - 1e-2), x / 10)

            # one kcal/mol/Angs
            y_up = x + 1
            y_down = np.maximum(1e-8, x - 1)

        for y in [y_up, y_down]:
            ax.plot(x, y, color="#000000", zorder=2, linewidth=0.5, linestyle="--")

    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Predicted %s [%s]" % (props[key], units[key]), fontsize=12)
    ax.set_xlabel("Calculated %s [%s]" % (props[key], units[key]), fontsize=12)

    ax.annotate(
        "Pearson's r: %.3f \nMAE: %.3f %s " % (r, mae, units[key]),
        (0.03, 0.88),
        xycoords="axes fraction",
        fontsize=12,
    )

    return r, mae, ax, hb
