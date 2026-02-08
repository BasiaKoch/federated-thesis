# Federated Learning Results Analysis

**Comparison:** FedAvg vs FedProx (μ=0.1)

**Results A:** `results/federated/data/fedavg_mu0.0_R30_E5_20260202_173756/results.json`

**Results B:** `results/federated/data/fedprox_mu0.1_R30_E5_20260202_191106/results.json`


## Comparison: FedAvg vs FedProx (μ=0.1)

### Configuration

| Parameter | FedAvg | FedProx (μ=0.1) |
|-----------|---------|---------|
| Strategy | fedavg | fedprox |
| μ (mu) | 0.0 | 0.1 |
| Rounds | 30 | 30 |
| Local Epochs | 5 | 5 |
| Clients | 2 | 2 |

### Final Metrics

| Metric | FedAvg | FedProx (μ=0.1) | Δ | Winner |
|--------|---------|---------|---|--------|
| Client0 Final | **0.8278** | 0.7987 | -3.5% | FedAvg |
| Client1 Final | 0.5935 | **0.7234** | +21.9% | FedProx (μ=0.1) |
| **Global Final** | 0.7826 | **0.7880** | +0.7% | FedProx (μ=0.1) |
| Global Best | 0.7872 | 0.7880 | +0.1% | Tie |

### Per-Class Breakdown (Final)

| Class | FedAvg | FedProx (μ=0.1) | Δ |
|-------|---------|---------|---|
| WT | 0.7927 | 0.8369 | +5.6% |
| TC | 0.7367 | 0.7301 | -0.9% |
| ET | 0.8185 | 0.7971 | -2.6% |

### Stability Analysis (Last 5 Rounds)

| Metric | FedAvg Std | FedProx (μ=0.1) Std | More Stable |
|--------|-------------|-------------|-------------|
| Client0 | 0.0457 | 0.0210 | FedProx (μ=0.1) |
| Client1 | 0.0502 | 0.0557 | FedAvg |
| **Global** | 0.0339 | 0.0158 | FedProx (μ=0.1) |

### Convergence Summary

**Client0:**
- FedAvg: Reached 90% of best (0.828) at round 11
- FedProx (μ=0.1): Reached 90% of best (0.800) at round 14

**Client1:**
- FedAvg: Reached 90% of best (0.724) at round 25
- FedProx (μ=0.1): Reached 90% of best (0.723) at round 20
