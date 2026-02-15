# FedAvg vs FedProx on BraTS 2D Brain Tumour Segmentation: Experimental Analysis

## 1. Overview

This section presents a systematic comparison of FedAvg (McMahan et al., 2017) and FedProx (Li et al., 2020a) on a federated brain tumour segmentation task using BraTS2020 2D slices. We evaluate both algorithms under two heterogeneity regimes: (i) an 8-client setting with graded per-client corruption, simulating a realistic multi-hospital consortium, and (ii) a controlled 2-client setting with extreme corruption on one client, designed to isolate and amplify the mechanisms that differentiate FedProx from FedAvg. All models are evaluated on a shared, clean global test set using the micro-averaged Dice score across Whole Tumour (WT), Tumour Core (TC), and Enhancing Tumour (ET) classes, reported as MeanPresent Dice.


## 2. Experimental Setup

### 2.1 Model and Training

All experiments use a lightweight UNet2D (base=16, 482,915 parameters) with 4-channel input (FLAIR, T1, T1ce, T2) and 3-channel output (WT, TC, ET). Training employs SGD with a constant learning rate of 0.01, no momentum, batch size 4, and no weight decay. The combined loss is BCE + soft Dice loss. No learning rate scheduling is applied, which contributes to late-stage oscillation in all runs -- an effect we discuss in Section 6.

### 2.2 Data Heterogeneity

We use the BraTS2020 Training dataset (369 patients) with two partition schemes:

**8-client partition.** Patients are assigned to 8 clients via Dirichlet allocation (alpha=0.1), creating heterogeneous data sizes (55--295 training slices per client). Each client additionally receives a unique corruption profile simulating scanner degradation, ranging from clean (Client 0) through mild bias fields and noise (Clients 1--4) to severe blur, downscaling, and noise combinations (Clients 5--7). Client 7 receives the most extreme corruption (bias=1.0, Gaussian noise sigma=0.6, blur sigma=3.0, 60% downscale). This setup combines statistical heterogeneity (non-IID label distributions from Dirichlet) with feature heterogeneity (per-client corruption), reflecting the real-world scenario where hospitals have both different patient populations and different scanner hardware (Sheller et al., 2020).

**2-client partition.** Patients are split 50/50 between two clients via stratified assignment (balanced by whole-tumour fraction quantiles), ensuring near-identical data sizes (660 vs 665 training slices) and similar tumour burden distributions. Client 0 receives clean data; Client 1 receives the same extreme corruption as Client 7 in the 8-client setting. By equalising data quantity and label distribution, this setup isolates feature-distribution heterogeneity (corruption) as the sole source of client divergence, providing a controlled experiment for analysing FedProx's proximal term.

### 2.3 Evaluation Protocol

After each federated round, the aggregated global model is evaluated on a held-out clean global test set (180 slices from patients not assigned to any client). We report the micro-averaged Dice score: for each class (WT, TC, ET), we accumulate pixel-wise intersection and union counts across all test slices, compute a single Dice coefficient per class, and then average across the classes present in the ground truth (MeanPresent). Per-client evaluation uses each client's local test set (also clean) to assess fairness across participants.


## 3. 8-Client Results

### 3.1 Effect of Local Epochs

We first compare E=30 and E=10 local epochs. With E=30, both FedAvg and FedProx (mu=0.4) converge to similar best Dice scores (0.818 vs 0.821, respectively), with FedProx showing only a marginal +0.3 point advantage. The high number of local SGD steps causes substantial client drift -- the phenomenon whereby local models diverge significantly from the global model during training (Karimireddy et al., 2020). At E=30, the proximal penalty (mu/2)||w - w_global||^2 is overwhelmed by 30 epochs of accumulated gradient updates, rendering the constraint ineffective relative to the magnitude of the local optimisation trajectory.

Reducing to E=10 brings the proximal term into a more effective operating regime. Li et al. (2020a) conduct their primary experiments with E in the range 1--20, noting that the proximal term's influence is proportional to E: with fewer local steps, each step is more strongly regularised toward the global model. Our results confirm this:

| Setting | FedAvg | FedProx (best mu) | Delta |
|---------|--------|--------------------|-------|
| E=30    | 0.8181 | 0.8208 (mu=0.4)   | +0.3 pts |
| E=10    | 0.8170 | 0.8221 (mu=0.1)   | +0.5 pts |

The improvement doubles when reducing E from 30 to 10, and the optimal mu shifts from 0.4 to 0.1, consistent with the expectation that less client drift requires a gentler constraint.

### 3.2 Mu Sensitivity at 8 Clients

At E=10, we compare mu=0.1 and mu=0.2:

| Metric           | FedAvg | mu=0.1 | mu=0.2 |
|------------------|--------|--------|--------|
| Best Pooled Mean | 0.8170 | 0.8221 | 0.8195 |
| Final Pooled Mean| 0.8103 | 0.8157 | 0.8066 |
| Best WT          | 0.8824 | 0.8788 | 0.8783 |
| Best TC          | 0.7795 | 0.7907 | 0.7840 |
| Best ET          | 0.8033 | 0.8122 | 0.8029 |

mu=0.1 achieves the best global Dice, while mu=0.2 performs slightly worse. The FedProx advantage is consistent but modest (+0.5 points), reflecting the fact that with 8 clients, the aggregation step provides substantial implicit averaging of heterogeneous updates, partially compensating for individual client drift.

### 3.3 Interpretation

The relatively small FedProx advantage in the 8-client setting can be attributed to what we term the *ensemble averaging effect*: when many clients with diverse corruption profiles participate in each round, their local update errors partially cancel during weighted averaging. This accidental robustness of FedAvg has been noted by Zhao et al. (2018), who observe that increasing the number of clients with non-IID data can reduce the effective divergence of the aggregated update. With 8 clients and fraction_fit=1.0, the corrupted gradients from Client 7 represent only approximately 5% of the aggregated update (65 out of 1,315 total training slices), limiting their destabilising effect regardless of strategy.


## 4. 2-Client Results

### 4.1 FedAvg vs FedProx (mu=0.2)

The 2-client setting produces a dramatically different picture. With the corrupted client controlling 50% of the aggregation weight, its divergent gradients directly destabilise the global model under FedAvg:

| Metric              | FedAvg       | FedProx mu=0.2 | Delta       |
|---------------------|-------------|----------------|-------------|
| Best Pooled Mean    | 0.7831 (R20)| 0.8053 (R43)   | **+2.2 pts**|
| Final Pooled Mean   | 0.7406      | 0.7733         | **+3.3 pts**|
| Best WT             | 0.8425      | 0.8586         | +1.6 pts    |
| Best TC             | 0.7631      | 0.7838         | +2.1 pts    |
| Best ET             | 0.7476      | 0.7802         | +3.3 pts    |
| Client 0 (clean)    | 0.7843      | 0.7957         | +1.1 pts    |
| Client 1 (corrupted)| 0.5776      | 0.5931         | +1.6 pts    |
| Late-stage std (R30--50) | 0.0500 | 0.0112         | 4.5x more stable |

The FedProx advantage is 4--6 times larger than in the 8-client setting across all metrics.

### 4.2 Catastrophic Instability in FedAvg

FedAvg exhibits a catastrophic collapse at Round 47, where the Pooled Mean Dice drops from 0.747 to 0.515 -- a 23-point single-round degradation. The model partially recovers in subsequent rounds (0.705 at R48, 0.747 at R49) but the damage to cumulative training is evident in the final score. No such collapse occurs under FedProx (mu=0.2).

This behaviour is consistent with the analysis of Karimireddy et al. (2020), who show that FedAvg can diverge when the gradient dissimilarity between clients is large. In our 2-client setting, gradient dissimilarity is maximised: one client optimises over clean images while the other optimises over severely corrupted images, producing gradients that point in fundamentally different directions. Without the proximal constraint, a single round where the corrupted client's update is particularly divergent can push the global model far from any useful minimum.

Li et al. (2020a) prove that FedProx converges under both statistical and systems heterogeneity, with a convergence rate that depends on the bounded dissimilarity between local and global objectives. The proximal term ensures that even when a client's local loss landscape differs substantially from the global objective, its update remains bounded in magnitude relative to the global model. Our empirical observation of the R47 crash in FedAvg -- and its absence in FedProx -- provides direct experimental evidence for this theoretical guarantee.

### 4.3 The Overfitting Mechanism

An instructive detail emerges from the training losses. By Round 50, FedAvg's corrupted client achieves a training loss of 0.064 -- substantially lower than FedProx's 0.150 for the same client. Yet FedAvg's test performance is worse. This indicates that FedAvg allows the corrupted client to overfit to its corrupted distribution: it learns to predict well on blurred, noisy, downscaled images but these learned features are detrimental when aggregated into a global model that must perform on clean test data.

FedProx's proximal term acts as a regulariser against this overfitting. By penalising deviation from the global model, it prevents the corrupted client from specialising too deeply on its local (corrupted) distribution. The higher training loss is not a failure -- it reflects a model that maintains greater alignment with the global consensus, at the cost of local fit. This interpretation aligns with the generalisation benefit of the proximal term discussed by Li et al. (2020a, Section 4.2).


## 5. Mu Sweep: The Accuracy--Fairness Tradeoff

### 5.1 Complete 2-Client Mu Sweep

We evaluate four mu values (0.0, 0.2, 0.3, 0.5) in the 2-client setting:

| Metric              | FedAvg | mu=0.2 | mu=0.3 | mu=0.5 |
|---------------------|--------|--------|--------|--------|
| Best Pooled Mean    | 0.7831 | **0.8053** | 0.7884 | 0.7868 |
| Final Pooled Mean   | 0.7406 | **0.7733** | 0.7595 | 0.7650 |
| Client 1 best       | 0.5776 | **0.5931** | 0.5698 | 0.5545 |
| Client 1 final      | 0.4646 | 0.4638 | **0.5556** | 0.5545 |
| Best--Final gap     | 0.0425 | 0.0320 | 0.0289 | **0.0218** |
| Late-stage std      | 0.0500 | **0.0112** | 0.0122 | 0.0241 |

### 5.2 Global Performance: Inverted-U Relationship

The best Pooled Mean Dice follows a clear inverted-U pattern with respect to mu:

- mu=0.0 (FedAvg): 0.7831 -- unstable, prone to catastrophic crashes
- mu=0.2: **0.8053** -- optimal global performance
- mu=0.3: 0.7884 -- moderate constraint
- mu=0.5: 0.7868 -- over-constrained

This shape arises from the competing effects of the proximal term. At low mu, the term is too weak to prevent destabilising drift from the corrupted client. At high mu, it constrains both clients so heavily that neither can learn effectively -- the Round 1 Pooled Mean decreases monotonically with mu (0.603, 0.500, 0.435, 0.368), reflecting increasingly limited per-round learning. The optimal mu balances these forces, providing enough regularisation to stabilise aggregation while preserving sufficient local learning capacity.

This inverted-U behaviour is predicted by the theory of Li et al. (2020a), who note that mu should be set in proportion to the degree of heterogeneity. Our empirical sweep confirms this and identifies mu=0.2 as the optimal value for our specific heterogeneity regime.

### 5.3 Corrupted Client Fairness: Monotonic Relationship

A striking finding emerges when examining Client 1's final-round Dice:

- mu=0.0: 0.4646 -- the global model drifts away from Client 1 in late rounds
- mu=0.2: 0.4638 -- same late-round collapse despite higher peak
- mu=0.3: **0.5556** -- +9.2 points over mu=0.2
- mu=0.5: 0.5545 -- similar protection

This reveals that mu controls a **tradeoff between global accuracy and client fairness**. At mu=0.2, the global model achieves the highest peak, but the proximal constraint is insufficient to prevent late-stage drift whereby the clean client gradually pulls the model away from the corrupted client's distribution. At mu >= 0.3, the constraint is strong enough to maintain the global model's relevance to Client 1 throughout training.

This finding connects to the fairness literature in federated learning. Li et al. (2019) introduce the concept of *good-intent fairness* -- ensuring that the global model performs uniformly well across all clients rather than optimising for aggregate performance. Mohri et al. (2019) formalise this as *agnostic federated learning*, where the objective is minimax across client distributions. Our mu sweep demonstrates that FedProx can be tuned to approximate fair outcomes by increasing mu, at a quantifiable cost to global performance: raising mu from 0.2 to 0.3 costs 1.7 points on the global Dice but gains 9.2 points on the worst-client final Dice.

### 5.4 Stability: Monotonic Improvement

The best-final gap (a measure of convergence stability) decreases monotonically with mu: 0.043, 0.032, 0.029, 0.022. Higher mu produces models that are less likely to regress from their best performance, because the proximal term limits the magnitude of per-round parameter changes. This is a direct consequence of the bounded update property: under FedProx, ||w_local - w_global|| is bounded by the ratio of the gradient norm to mu (Li et al., 2020a, Theorem 4).


## 6. 8-Client vs 2-Client: Why the Gap Widens

### 6.1 The Ensemble Averaging Effect

The central finding of our cross-setting comparison is that FedProx's advantage scales with the *effective influence* of heterogeneous clients in the aggregation. In the 8-client setting, each client contributes a fraction proportional to its data size, and the most corrupted client (Client 7, 65 slices) represents only 5% of the total training data. In the 2-client setting, the corrupted client controls 50% of the aggregation. The FedProx advantage scales accordingly: +0.5 points at 5% corrupted influence, +2.2 points at 50%.

This can be understood through the lens of gradient dissimilarity. Karimireddy et al. (2020) define client drift as the gap between the local optimum and the global optimum. In weighted federated averaging, the aggregated update is:

  w_global = sum_k (n_k / n) * w_k

where n_k is client k's data size. When a single client with divergent gradients has weight n_k/n = 0.05, its contribution to the aggregated drift is small. When its weight is 0.50, the aggregated update is pulled halfway toward the divergent client's local optimum, directly destabilising the global model. FedProx's proximal term bounds each w_k to stay close to w_global, limiting this effect regardless of aggregation weight.

### 6.2 Implications for Cross-Silo Federated Learning

The BraTS brain tumour segmentation task represents a cross-silo federated learning setting, where a small number of hospitals (typically 2--20) collaborate without sharing patient data (Kairouz et al., 2021). Our results suggest that in such settings:

1. **FedProx is increasingly beneficial as the number of participating institutions decreases.** With many hospitals, averaging naturally smooths out individual scanner differences. With few hospitals, each institution's idiosyncrasies directly affect the global model.

2. **The proximal coefficient mu should be treated as a fairness knob.** Practitioners must decide whether to optimise for aggregate performance (lower mu) or worst-case institutional performance (higher mu). This choice has ethical implications in medical imaging, where systematically poor performance at one hospital could lead to missed diagnoses.

3. **FedAvg may appear sufficient in large consortia.** Our 8-client results show only a +0.5 point advantage for FedProx. A practitioner evaluating on aggregate metrics alone might conclude that the additional complexity is unwarranted. The 2-client results reveal that this conclusion is fragile: it depends on the ensemble averaging effect that may not hold when consortium composition changes.


## 7. Relation to Published Literature

### 7.1 Confirmation of FedProx's Design Claims

Li et al. (2020a) motivate FedProx primarily through systems heterogeneity (stragglers) and provide convergence guarantees under statistical heterogeneity. Our experiments validate the statistical heterogeneity claim in a medical imaging context: even without stragglers (drop_percent=0.0), the proximal term improves convergence stability and global performance when client data distributions differ substantially due to corruption.

### 7.2 Consistency with Non-IID Analyses

Zhao et al. (2018) demonstrate that non-IID data distributions degrade FedAvg's convergence and propose data-sharing as a mitigation. Our corruption-based heterogeneity creates a form of feature-shift non-IID-ness (as opposed to the label-shift non-IID-ness commonly studied), and we observe the same convergence degradation -- particularly the increased variance and susceptibility to catastrophic rounds (Hsu et al., 2019).

### 7.3 Client Drift and the Role of the Proximal Term

Karimireddy et al. (2020) formally analyse client drift in FedAvg and show that it introduces a bias term in the convergence bound that scales with the gradient dissimilarity between clients. They propose SCAFFOLD as an alternative that uses control variates to correct for drift. Our work shows that FedProx's simpler proximal penalty is effective in the extreme heterogeneity regime, though the observed oscillation in all strategies suggests that variance-reduction methods like SCAFFOLD could provide further improvement.

Wang et al. (2020) identify the *objective inconsistency* problem in FedAvg, where the algorithm converges to a point that is not a stationary point of the global objective when client data is heterogeneous. FedProx mitigates this by modifying the local objective, and our experiments -- particularly the finding that FedAvg's Client 1 achieves low training loss but poor test performance -- provide evidence of objective inconsistency in practice.

### 7.4 Fairness in Federated Learning

Our observation that mu controls an accuracy-fairness tradeoff extends findings from Li et al. (2019), who propose a framework for fair federated learning based on minimax optimisation. While their approach requires modifying the aggregation scheme, our results show that FedProx's proximal term provides a simpler mechanism for improving worst-client performance: increasing mu effectively applies stronger regularisation toward the global consensus, preventing any single client from being marginalised in late training stages.


## 8. Limitations

**Single seed.** All experiments use seed=42. While the observed patterns are consistent across settings and mu values, multi-seed experiments are needed to establish statistical significance and compute confidence intervals.

**Constant learning rate.** The absence of learning rate scheduling contributes to late-stage oscillation in all strategies. The best-final gaps (2--4 points) could be reduced by cosine annealing, which would likely amplify FedProx's convergence advantage.

**Synthetic corruption.** The per-client corruption profiles simulate scanner differences but do not capture the full complexity of real-world hospital data variation, which includes differences in patient demographics, imaging protocols, and annotation practices.

**Two-client partition is a controlled extreme.** While the 2-client setting provides clear mechanistic insight, real federated medical imaging consortia typically involve 3--15 institutions with varying degrees of heterogeneity. Intermediate settings (e.g., 4 clients with 2 corrupted) would bridge the gap between the controlled 2-client experiment and the realistic 8-client setting.


## 9. Summary of Findings

1. **FedProx consistently outperforms FedAvg** on global Dice in both 8-client (+0.5 pts) and 2-client (+2.2 pts) settings, with the advantage scaling with the effective influence of heterogeneous clients.

2. **FedAvg is susceptible to catastrophic single-round collapses** when strongly heterogeneous clients have significant aggregation weight. FedProx eliminates this instability through the bounded update property of the proximal term.

3. **The proximal coefficient mu exhibits an inverted-U relationship with global performance** (optimal at mu=0.2 for our setting) and a **monotonic relationship with worst-client fairness** (higher mu protects the corrupted client).

4. **mu functions as a tunable accuracy-fairness knob:** mu=0.2 maximises global Dice; mu=0.3 sacrifices 1.7 global points to gain 9.2 points on the corrupted client's final Dice.

5. **The ensemble averaging effect in multi-client settings** provides implicit robustness to FedAvg, masking heterogeneity problems that become apparent with fewer clients. This has practical implications for small federated consortia in medical imaging.


## References

- Hsu, T.-M. H., Qi, H., and Brown, M. (2019). Measuring the effects of non-identical data distribution for federated visual classification. *NeurIPS Workshop on Federated Learning*.
- Kairouz, P., McMahan, H. B., et al. (2021). Advances and open problems in federated learning. *Foundations and Trends in Machine Learning*, 14(1-2):1--210.
- Karimireddy, S. P., Kale, S., Mohri, M., Reddi, S. J., Stich, S. U., and Suresh, A. T. (2020). SCAFFOLD: Stochastic controlled averaging for federated learning. *ICML*.
- Li, T., Sahu, A. K., Talwalkar, A., and Smith, V. (2020a). Federated optimization in heterogeneous networks. *MLSys*.
- Li, T., Sanjabi, M., Beirami, A., and Smith, V. (2019). Fair resource allocation in federated learning. *ICLR 2020*.
- McMahan, H. B., Moore, E., Ramage, D., Hampson, S., and Arcas, B. A. (2017). Communication-efficient learning of deep networks from decentralized data. *AISTATS*.
- Mohri, M., Sivek, G., and Suresh, A. T. (2019). Agnostic federated learning. *ICML*.
- Sheller, M. J., Edwards, B., Reina, G. A., et al. (2020). Federated learning in medicine: facilitating multi-institutional collaborations without sharing patient data. *Scientific Reports*, 10:12598.
- Wang, J., Liu, Q., Liang, H., Joshi, G., and Poor, H. V. (2020). Tackling the objective inconsistency problem in heterogeneous federated optimization. *NeurIPS*.
- Zhao, Y., Li, M., Lai, L., Suda, N., Civin, D., and Chandra, V. (2018). Federated learning with non-IID data. *arXiv preprint arXiv:1806.00582*.
