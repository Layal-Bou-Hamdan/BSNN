

# Bayesian Sheaf Neural Networks

This repository contains the code for the paper  
**Bayesian Sheaf Neural Networks** *(currently available on arXiv)*.

<img src="figures/bsnn.png" alt="Bayesian Sheaf Neural Networks" width="500"/>

## Getting Started

To set up the environment, run the following:

```bash
conda env create --file=environment.yml
conda activate bsnn
```

## Run Hyperparameter Sweep

To run a hyperparameter sweep, first create a [Weights & Biases account](https://wandb.ai/site). Then, authenticate:

```bash
wandb online
wandb login
```

After logging in, you can launch an example sweep with:

```bash
export ENTITY=<WANDB_ACCOUNT_ID>
wandb sweep --project bsnn config/bayesbundle_cora_sweep.yml
```

This launches a sweep for Bayesian sheaf neural networks with orthogonal restriction maps on the Cora dataset.

## Experimental Setup

All experiments were conducted on a high-performance cluster.  
Each run used an NVIDIA H100 GPU (80GB HBM3), dual Intel Xeon Platinum 8462Y+ CPUs, and 1.0 TiB of RAM.  
CUDA version: 12.4  
Driver version: 550.54.15

## Reproducibility

We ensure reproducibility by fixing random seeds across all experiments.  
Each reported metric is averaged over 30 independent seeds, with standard deviation reported.  
All training logs and configurations are saved using `wandb` and `results/`.

To reproduce key results from the paper, use the following scripts:

```bash
bash scripts/run_cora.sh
bash scripts/run_pubmed.sh
```

## Acknowledgment

This repository builds upon the open-source codebase from [Neural Sheaf Diffusion](https://github.com/twitter-research/neural-sheaf-diffusion), which is licensed under Apache 2.0.
