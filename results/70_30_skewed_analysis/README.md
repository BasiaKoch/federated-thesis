# ET-Skewed Federated Learning Analysis

## FedProx vs FedAvg on Label-Heterogeneous Brain Tumor Segmentation

This directory contains the comprehensive analysis of federated learning experiments comparing FedProx and FedAvg on ET-skewed (label-heterogeneous) BraTS 2D data.

---

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | SGD |
| **Learning Rate** | 0.01 |
| **Batch Size** | 10 |
| **Rounds** | 30 |
| **Local Epochs** | 5 |
| **FedProx μ** | 0.01 |
| **Clients** | 2 (70/30 split) |
| **Partition Type** | ET-skewed (label heterogeneity) |

### Data Heterogeneity

- **Client 0 (Low-ET)**: 169 cases (57.3%), ET ratio mean = 0.119
- **Client 1 (High-ET)**: 126 cases (42.7%), ET ratio mean = 0.356
- **ET Ratio Gap**: 3× difference between clients

---

## Key Results

### Final Performance (Round 30)

| Metric | FedProx (μ=0.01) | FedAvg | Improvement |
|--------|------------------|--------|-------------|
| **Global Mean Dice** | **0.657** | 0.597 | **+6.0 pp** |
| Global WT Dice | 0.729 | 0.613 | +11.6 pp |
| Global TC Dice | 0.634 | 0.618 | +1.6 pp |
| **Global ET Dice** | **0.608** | 0.560 | **+4.8 pp** |
| Client 0 Mean | 0.567 | 0.508 | +5.9 pp |
| Client 1 Mean | 0.732 | 0.671 | +6.1 pp |

### Stability

- **FedProx**: Final = Best (0.657) ✓
- **FedAvg**: Final (0.597) < Best (0.655) - drops 5.8 pp

---

## Generated Files

### Figures

| File | Description |
|------|-------------|
| `fig1_et_distribution.png/pdf` | ET ratio distribution by client (histogram, box plot, CDF) |
| `fig2_global_convergence.png/pdf` | Global convergence curves (Mean, WT, TC, ET) |
| `fig3_perclient_performance.png/pdf` | Per-client performance comparison |
| `fig4_et_convergence.png/pdf` | **ET-specific convergence by client (KEY FIGURE)** |
| `fig5_stability.png/pdf` | Training stability analysis |
| `fig6_heatmap.png/pdf` | Per-client, per-class performance heatmap |
| `fig7_summary_comparison.png/pdf` | Summary bar chart comparison |

### Tables

| File | Description |
|------|-------------|
| `table1_data_distribution.csv/tex` | Client data distribution statistics |
| `table2_global_performance.csv/tex` | Global segmentation performance |
| `table3_perclient_performance.csv/tex` | Per-client, per-class Dice scores |
| `table4_stability.csv/tex` | Training stability metrics |
| `table5_et_progression.csv/tex` | ET Dice progression over rounds |
| `full_results_per_round.csv` | Complete per-round metrics for both strategies |

### Scripts

| File | Description |
|------|-------------|
| `analyze_federated_results.py` | Main analysis script |

---

## How to Regenerate

```bash
cd /Users/basiakoch/Federated/federated-thesis
python3 results/70_30_skewed_analysis/analyze_federated_results.py
```

---

## Thesis Integration

### Recommended Figure Order

1. **Figure 1**: Data heterogeneity characterization (establishes the experimental setup)
2. **Figure 2**: Global convergence (main algorithm comparison)
3. **Figure 4**: ET convergence by client (**most important** - shows heterogeneity effect)
4. **Figure 3**: Per-client performance (quantifies client gap)
5. **Figure 5**: Stability analysis (explains FedProx advantage)

### Key Narratives

1. **FedProx outperforms FedAvg** under label heterogeneity (+6 pp global Mean Dice)
2. **ET segmentation most affected** by heterogeneity (3× ET ratio difference → ~12 pp client gap)
3. **FedProx stabilizes training** (no end-of-training degradation)
4. **Both clients benefit** from FedProx despite unequal ET contributions

---

## Source Data

- FedProx results: `results/federated/brats2d_7030_c0ET10/fedprox_mu0.01_R30_E5_20260128_002555/results.json`
- FedAvg results: `results/federated/brats2d_7030_c0ET10/fedavg_mu0.0_R30_E5_20260128_103516/results.json`
- Client map: `data/partitions/brats2d_7030_et_skewed/client_map.json`
