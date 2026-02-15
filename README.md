# Federated Learning for Medical Image Segmentation

This thesis project investigates and compares federated learning algorithms ,  **FedAvg** and **FedProx** ,  for brain tumor segmentation, enabling collaborative model training across simulated hospital sites without sharing raw patient data.

## Motivation

In medical imaging, data is distributed across hospitals and cannot be centralized due to privacy regulations. Federated learning allows multiple institutions to collaboratively train a shared model while keeping patient data local. However, real-world medical data is inherently **non-IID** (non-identically distributed): hospitals differ in patient populations, scanner hardware, and imaging protocols. This project evaluates how well FedAvg and FedProx handle these heterogeneity challenges.

## Research Focus

- **Task**: 2D brain tumor segmentation on the [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/) dataset with three target regions: Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET)
- **Model**: 2D U-Net with GroupNorm, trained on 4 MRI modalities (FLAIR, T1, T1ce, T2)
- **Algorithms**: FedAvg (McMahan et al., 2017) and FedProx (Li et al., 2020) implemented via the [Flower](https://flower.ai/) framework
- **Heterogeneity scenarios**: label distribution skew (Dirichlet allocation, ET-skewed partitions), feature distribution shift (simulated scanner variability via bias fields, noise, blur), and data quantity imbalance

## Key Findings

- FedProx outperforms FedAvg by **+6.0 percentage points** in mean Dice score under extreme label heterogeneity (ET-skewed, 2-client setting)
- FedAvg is susceptible to catastrophic single-round performance collapses with high data heterogeneity, while FedProx maintains stable convergence
- The proximal term (mu) provides a tunable **accuracy-fairness tradeoff**: lower values (mu=0.2) maximize global performance, higher values (mu>=0.3) ensure fairness across clients
- In larger consortia (8 clients), ensemble averaging masks the FedProx advantage, but it remains critical in small-client settings

## Tech Stack

- **Python** 3.9+
- **PyTorch** for model training
- **Flower (flwr)** for federated learning orchestration
- **NVFlare** for alternative FL experiments
- **NumPy** and **Matplotlib** for analysis and visualization

## Getting Started

### Prerequisites

Create a conda environment from the provided specification:

```bash
conda env create -f environment.yml
```

Or install dependencies via pip:

```bash
pip install -r requirements.txt
```

### Data

This project uses the [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/) dataset. Download the data and preprocess it into 2D NPZ slices. Partition scripts in `data/partitions/` generate the client splits used in the experiments.

### Running Experiments

**Federated training (BraTS):**

```bash
python experiments/brats/brats_n_clients.py
```

**Federated training (MNIST benchmark):**

```bash
python experiments/mnist/mnist_n_clients.py
```

**Centralized baseline:**

```bash
python federated_unet/unet/unet_non_federated.py
```

## Results

Experimental results, figures, and tables are stored in `results/`. The main thesis analysis comparing FedProx and FedAvg on ET-skewed data can be found in `results/70_30_skewed_analysis/`.

## License

This project was developed as part of a master's thesis.
