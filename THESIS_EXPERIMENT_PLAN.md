# Thesis Experiment Plan: FedProx vs FedAvg on Heterogeneous BraTS Data

## Overview

This document outlines a systematic experimental design to demonstrate FedProx advantages over FedAvg in federated brain tumor segmentation using BraTS 2D data with 4 modalities (T1, T1ce, T2, FLAIR).

**Key Thesis Question**: Under what heterogeneity conditions does FedProx outperform FedAvg, and by how much?

---

## Experimental Framework

### Baselines (Required for All Experiments)

| Baseline | Description | Purpose |
|----------|-------------|---------|
| **Centralized** | All data pooled, single model | Upper bound (ideal scenario) |
| **Local-Only** | Each client trains independently | Lower bound (no federation) |
| **FedAvg** | Standard federated averaging | Primary comparison |
| **FedProx** | FedAvg + proximal term | Method under investigation |

---

## Experiment 1: Label Distribution Skew (ET Ratio Heterogeneity)

**Hypothesis**: Clients specializing in different tumor types (low vs high ET ratio) will cause gradient divergence that FedProx can mitigate.

### 1.1 Partition Creation

```bash
# Create 3 heterogeneity levels using Dirichlet distribution
cd /Users/basiakoch/Federated/federated-thesis

# Extreme heterogeneity (α=0.1)
python data/partitions/make_dirichlet_partition.py --alpha 0.1 --num_clients 4

# Moderate heterogeneity (α=0.5)
python data/partitions/make_dirichlet_partition.py --alpha 0.5 --num_clients 4

# Mild heterogeneity (α=1.0)
python data/partitions/make_dirichlet_partition.py --alpha 1.0 --num_clients 4
```

### 1.2 Training Matrix

| Partition | Strategy | μ | Local Epochs | Rounds |
|-----------|----------|---|--------------|--------|
| α=0.1 | FedAvg | 0 | 5 | 50 |
| α=0.1 | FedProx | 0.1 | 5 | 50 |
| α=0.1 | FedProx | 0.4 | 5 | 50 |
| α=0.1 | FedProx | 1.0 | 5 | 50 |
| α=0.5 | FedAvg | 0 | 5 | 50 |
| α=0.5 | FedProx | 0.1 | 5 | 50 |
| α=0.5 | FedProx | 0.4 | 5 | 50 |
| α=1.0 | FedAvg | 0 | 5 | 50 |
| α=1.0 | FedProx | 0.01 | 5 | 50 |
| α=1.0 | FedProx | 0.1 | 5 | 50 |

### 1.3 Expected Results

- **α=0.1 (extreme)**: FedProx with μ=0.4-1.0 should outperform FedAvg by 5-15%
- **α=0.5 (moderate)**: FedProx with μ=0.1-0.4 should show 2-8% improvement
- **α=1.0 (mild)**: FedAvg and FedProx should perform similarly

### 1.4 Metrics to Report

- Mean Dice (WT, TC, ET) per round
- Convergence speed (rounds to 80%, 85%, 90% Dice)
- Stability (variance in last 10 rounds)
- Client drift (||w_local - w_global||²)

---

## Experiment 2: Modality Heterogeneity (Feature Skew)

**Hypothesis**: Different hospitals having different MRI modalities creates feature heterogeneity that severely impacts FedAvg but can be handled by FedProx.

**This is the most realistic scenario for medical imaging!**

### 2.1 Partition Creation

```bash
python data/partitions/make_modality_heterogeneous_partition.py --num_clients 4
```

### 2.2 Client Configuration

| Client | Modalities | Channels | Clinical Scenario |
|--------|-----------|----------|-------------------|
| 0 | T1, T1ce, T2, FLAIR | 4 | Academic center (full protocol) |
| 1 | T1, T2, FLAIR | 3 | No contrast (contraindicated) |
| 2 | T1ce, T2, FLAIR | 3 | Protocol variation |
| 3 | T2, FLAIR | 2 | Rural hospital (limited scanner) |

### 2.3 Implementation Note

**Critical**: The model must handle variable input channels. Options:

1. **Zero-padding**: Pad missing channels with zeros
2. **Channel-specific encoders**: Separate initial convolutions per modality
3. **Modality dropout during training**: Train all clients with random modality dropout

Recommended approach for thesis: **Zero-padding** (simplest, fair comparison)

```python
# In data loading, pad to 4 channels:
def pad_to_4_channels(img, original_channels):
    """Pad image to 4 channels, filling missing with zeros."""
    if img.shape[0] == 4:
        return img
    padded = np.zeros((4, img.shape[1], img.shape[2]), dtype=img.dtype)
    # Map channels based on which modalities are present
    padded[:img.shape[0]] = img
    return padded
```

### 2.4 Training Matrix

| Strategy | μ | Local Epochs | Expected Outcome |
|----------|---|--------------|------------------|
| FedAvg | 0 | 5 | Poor - feature mismatch causes divergence |
| FedProx | 0.1 | 5 | Moderate improvement |
| FedProx | 0.3 | 5 | Best expected performance |
| FedProx | 0.5 | 5 | May over-regularize |

---

## Experiment 3: Quantity Skew (Data Imbalance)

**Hypothesis**: When one client has much more data, FedAvg's weighted averaging over-represents that client. FedProx keeps smaller clients from drifting.

### 3.1 Partition Creation

```bash
# Extreme imbalance: 70/15/10/5 split
python data/partitions/make_dirichlet_partition.py \
    --alpha 0.5 --num_clients 4 \
    --quantity_skew 0.7,0.15,0.10,0.05
```

(You may need to modify the script to support quantity skew)

### 3.2 Expected Results

With imbalanced data:
- FedAvg: Model biased toward large client's distribution
- FedProx: Smaller clients contribute more meaningfully

---

## Experiment 4: Local Epochs Analysis

**Hypothesis**: More local epochs increase client drift. FedProx benefit grows with more local epochs.

### 4.1 Training Matrix (Fixed: α=0.5, 4 clients)

| Local Epochs | FedAvg Dice | FedProx (μ=0.1) Dice | Δ |
|--------------|-------------|---------------------|---|
| 1 | Expected: similar | Expected: similar | ~0% |
| 3 | Expected: slight advantage | Moderate improvement | 2-5% |
| 5 | Expected: degradation | Better stability | 5-10% |
| 10 | Expected: significant drift | Clear advantage | 10-15% |

### 4.2 Commands

```bash
for E in 1 3 5 10; do
    # FedAvg
    python federated_unet/unet/unet_flower_any_Nclients.py \
        --partition_dir data/partitions/brats2d_4client_dirichlet_a0p5/client_data \
        --strategy fedavg --local_epochs $E --rounds 50

    # FedProx
    python federated_unet/unet/unet_flower_any_Nclients.py \
        --partition_dir data/partitions/brats2d_4client_dirichlet_a0p5/client_data \
        --strategy fedprox --mu 0.1 --local_epochs $E --rounds 50
done
```

---

## Experiment 5: Partial Client Participation

**Hypothesis**: When only a subset of clients participate each round, FedProx provides more stable convergence.

### 5.1 Configuration

| fraction_fit | Clients/Round | Expected FedProx Advantage |
|--------------|---------------|---------------------------|
| 1.0 | 4/4 | Baseline |
| 0.5 | 2/4 | Moderate |
| 0.25 | 1/4 | Significant |

### 5.2 Commands

```bash
for FRAC in 1.0 0.5 0.25; do
    python federated_unet/unet/unet_flower_any_Nclients.py \
        --partition_dir data/partitions/brats2d_4client_dirichlet_a0p5/client_data \
        --strategy fedprox --mu 0.1 \
        --fraction_fit $FRAC --rounds 100
done
```

---

## Recommended Thesis Structure

### Chapter: Experiments

1. **Baseline Establishment**
   - Centralized training results
   - Local-only training results
   - IID federated baseline (α=10.0)

2. **Label Heterogeneity Analysis**
   - Dirichlet α sweep (0.1, 0.5, 1.0)
   - FedProx μ sweep per α level
   - Convergence curves and stability analysis

3. **Feature Heterogeneity Analysis** (Most Novel!)
   - Modality-heterogeneous partition
   - Show FedAvg failure mode
   - FedProx recovery

4. **Hyperparameter Sensitivity**
   - Local epochs analysis
   - Participation rate analysis
   - μ sensitivity curves

5. **Ablation Studies**
   - Effect of number of clients (2, 4, 8)
   - Effect of learning rate
   - Comparison with other methods (FedProx vs SCAFFOLD vs FedNova)

---

## Key Metrics for Thesis Tables

### Per-Experiment Metrics

| Metric | Description |
|--------|-------------|
| **Final Dice** | WT, TC, ET Dice at last round |
| **Best Dice** | Peak performance during training |
| **Convergence Round** | First round to reach 80%/85%/90% Dice |
| **Stability** | Std dev of Dice in last 10 rounds |
| **Client Drift** | Mean ||w_k - w_global||² |

### Heterogeneity Metrics (from partition scripts)

| Metric | Interpretation |
|--------|---------------|
| **Mean JSD** | > 0.4 = substantial, 0.2-0.4 = moderate, < 0.2 = mild |
| **Max JSD** | Worst-case client pair divergence |
| **ET Gap** | Ratio of max/min mean ET ratio |
| **KS Statistic** | Distribution difference test |

---

## Visualization Recommendations

1. **Convergence Curves**: Dice vs Round (FedAvg vs FedProx, multiple μ values)
2. **Heatmap**: Final Dice across (α, μ) grid
3. **Box Plots**: Per-client performance distribution
4. **Drift Plots**: Client drift over rounds
5. **Stability Bands**: Mean ± std Dice curves

---

## Sample Results Table (Template)

### Table: Label Heterogeneity Results (4 Clients, E=5, 50 Rounds)

| α | Strategy | μ | WT Dice | TC Dice | ET Dice | Mean Dice | Convergence |
|---|----------|---|---------|---------|---------|-----------|-------------|
| 0.1 | FedAvg | - | 0.XX | 0.XX | 0.XX | 0.XX | Round XX |
| 0.1 | FedProx | 0.1 | 0.XX | 0.XX | 0.XX | 0.XX | Round XX |
| 0.1 | FedProx | 0.4 | 0.XX | 0.XX | 0.XX | 0.XX | Round XX |
| 0.5 | FedAvg | - | 0.XX | 0.XX | 0.XX | 0.XX | Round XX |
| 0.5 | FedProx | 0.1 | 0.XX | 0.XX | 0.XX | 0.XX | Round XX |
| 1.0 | FedAvg | - | 0.XX | 0.XX | 0.XX | 0.XX | Round XX |
| 1.0 | FedProx | 0.01 | 0.XX | 0.XX | 0.XX | 0.XX | Round XX |

---

## Quick Start: Run Full Experiment Suite

```bash
#!/bin/bash
# Run complete thesis experiments

BASE_DIR="/Users/basiakoch/Federated/federated-thesis"
PARTITION_BASE="$BASE_DIR/data/partitions"
RESULTS_BASE="$BASE_DIR/results/thesis_experiments"

# 1. Create partitions
echo "Creating partitions..."
python $PARTITION_BASE/make_dirichlet_partition.py --alpha 0.1 --num_clients 4
python $PARTITION_BASE/make_dirichlet_partition.py --alpha 0.5 --num_clients 4
python $PARTITION_BASE/make_dirichlet_partition.py --alpha 1.0 --num_clients 4
python $PARTITION_BASE/make_modality_heterogeneous_partition.py --num_clients 4

# 2. Run experiments (example for α=0.5)
PARTITION="$PARTITION_BASE/brats2d_4client_dirichlet_a0p5/client_data"

# FedAvg baseline
python $BASE_DIR/federated_unet/unet/unet_flower_any_Nclients.py \
    --partition_dir $PARTITION \
    --strategy fedavg --rounds 50 --local_epochs 5 \
    --out_dir $RESULTS_BASE/dirichlet_a0p5_fedavg

# FedProx μ=0.1
python $BASE_DIR/federated_unet/unet/unet_flower_any_Nclients.py \
    --partition_dir $PARTITION \
    --strategy fedprox --mu 0.1 --rounds 50 --local_epochs 5 \
    --out_dir $RESULTS_BASE/dirichlet_a0p5_fedprox_mu0p1

# FedProx μ=0.4
python $BASE_DIR/federated_unet/unet/unet_flower_any_Nclients.py \
    --partition_dir $PARTITION \
    --strategy fedprox --mu 0.4 --rounds 50 --local_epochs 5 \
    --out_dir $RESULTS_BASE/dirichlet_a0p5_fedprox_mu0p4

echo "Experiments complete!"
```

---

## Summary: When Will FedProx Outperform FedAvg?

| Condition | FedProx Advantage |
|-----------|-------------------|
| High label skew (α ≤ 0.5) | Strong |
| Feature heterogeneity (missing modalities) | Very Strong |
| Many local epochs (E ≥ 5) | Strong |
| Partial participation (fraction < 0.5) | Moderate |
| Quantity imbalance | Moderate |
| IID or near-IID data | None (may hurt) |
| Single local epoch (E=1) | None |

**Key Takeaway**: Design experiments with sufficient heterogeneity and local computation to demonstrate FedProx benefits. Your current 70/30 split with 2 clients and low local epochs is too mild.
