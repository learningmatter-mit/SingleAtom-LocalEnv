# Per-site PaiNN

Description

## Installation

1. Clone the repository

```bash
git clone git@github.mit.edu:hojechun/per-site_PaiNN.git
```

2. Conda environments

```bash
conda upgrade conda
conda env create -f environment.yml
conda activate persitePainn
```

3. Install NeuralForceField (nff)

```bash
git clone https://github.com/learningmatter-mit/NeuralForceField.git
# Go to the nff directory
pip install .
# Copy nff/utils/table_data to the installed directory in conda envs python packages
```

4. Install

```bash
# Go to per-site_painn directory
pip install .
```

## Usage

run `main.py` with settings (e.g., `details.json` below)\

- example command line

```bash
python main.py --data data_raw/data.pkl --cache data_cache/data_cache --details details.json --savedir results
```

- example `details.json`

```json
{
  "modeltype": "Painn",
  "details": {
    "spectra": false,
    "multifidelity": false
  },
  "modelparams": {
    "output_keys": ["d_p"],
    "feat_dim": 128,
    "activation": "swish",
    "activation_f": "softplus",
    "n_rbf": 20,
    "cutoff": 6.5,
    "num_conv": 4,
    "atom_fea_len": 128,
    "h_fea_len": 256,
    "n_h": 4,
    "n_outputs": 1,
    "conv_dropout": 0.0,
    "readout_dropout": 0.05,
    "fc_dropout": 0.05
  }
}
```

- Modeltype: Type of model
- Details: \
  `spectra`: whether you predict spectra (To impose the ouptuts to be positive if true)\
  `multifidelity`: whether you use multi-fidelity stretagy (To add atomwise properties)
- Modelparams:\
  `feat_dim`: number of features in PaiNN\
  `activation`: activation function in PaiNN\
  `activation_f`: activation function in FullyConnected NN (FFNN)\
  `n_rbf`: number of rbf\
  `cutoff`: cutoff radius\
  `num_conv`: number of convolution (message & update blocks)\
  `output_keys`: target property to predict\
  `atom_fea_len`: output_atomwise feature lenghts after the readoutblock\
  `h_fea_len`: number of nodes in FFNN\
  `n_h`: number of hidden layers\
  `n_outputs`: number of predictions (for spectra number of mesh)\
  `conv_dropout`: dropout ratio in PaiNN\
  `readout_droupout`: dropout ratio in readoutblock\
  `fc_dropout`: dropout ratio in FFNN\
  `n_fidelity`: number of adding values for node properties
