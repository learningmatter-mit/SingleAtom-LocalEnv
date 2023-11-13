# High-throughput screening of the local environments of Single Atom Catalysts

This repository includes necessary scripts and models for reproducing "Active learning accelerated exploration of the single atom local environments in multimetallic systems for oxygen electrocatalysis". More information about the data, models and results can be found [here](https://hojechun.github.io/). Dataset and trained model can be found in our Zenodo dataset [here](https://zenodo.org/records/10119944)

## Installation

1. Clone the repository

```bash
git clone https://github.com/learningmatter-mit/SingleAtom-LocalEnv.git
```

2. Conda environments setting and necessary package

- Conda environment

```bash
conda upgrade conda
conda env create -f environment.yml
conda activate SingeAtom
```

- Install NeuralForceField (nff)

Used for the dataset (graph) structure. Tested up to commit `72d1f32f43f202c1a466116beeed15845a6456e7` on the `master` branch.

```bash
git clone https://github.com/learningmatter-mit/NeuralForceField.git
```

Add the following to `~/.bashrc` or equivalent with appropriate paths and then `source ~/.bashrc`.

```
export NFFDIR="/path/to/NeuralForceField"
export PYTHONPATH=$NFFDIR:$PYTHONPATH
```

4. Install Wandb

To monitor the training log more efficiently you can use `wandb`.

Create an account [here](https://wandb.ai/home) and install the Python package:

```bash
pip install wandb
wandb login
```

5. Install `persite_painn`

```bash
# Go to SingleAtom-LocalEnv directory
pip install .
# For editable mode
pip install -e .
```

## Usage

- Train the model:

  run `main.py` with config (example config files in `examples/configs`)

```bash

## In case you have converted dataset
python main.py --data_cache "path/to/dataset" --details "path/to/details" --savedir "path/to/savedir"
## In case you have raw dataset
python main.py --data_raw "path/to/dataset_raw" --data_cache "path/to/dataset" --details "path/to/details" --savedir "path/to/savedir"
```

- Active Learning Sampling

```bash
python active_learning.py --total_dataset "path/to/search_space" --model_path "path/to/trained_ensemble_models" --dataset "path/to/dataset" --multifidelity --uncertainty_type bayesian --save --plot
```

## Examples

Some examples of ensemble inference and node embedding analysis are described in `examples`.
