# Table 1: Cross-Checkpoint Longitudinal Evaluation & Retention Matrix

> **Caption:** Cross-checkpoint evaluation matrix across streaming cohorts ($N=100$ per cohort). $R_t(C_1)$ denotes backward retention relative to accepted adapter $C_1$. Values falling below the $0.9800$ SLA boundary demonstrate contract-violating retention degradation across successive streaming updates.

| Adapter / Checkpoint | Cohort C1 | Cohort C2 | Cohort C3 | Cohort C4 | Held-Out Cohort H | Backward Retention $R_t(C_1)$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base Model (Untuned)** | 0.4330 | 0.4095 | 0.4648 | 0.3808 | 0.4507 | 0.7535 |
| **Checkpoint 1 ($C_1$, Incumbent)** | **0.5746** | 0.5035 | 0.5318 | 0.4767 | 0.5233 | **1.0000** |
| **Checkpoint 2 ($C_2$, Candidate)** | 0.5590$^{\dagger}$ | 0.5082 | 0.5325 | 0.4701 | 0.5336 | 0.9729$^{\dagger}$ *(Violates SLA)* |
| **Checkpoint 3 ($C_3$)** | 0.5602 | 0.5129 | 0.5339 | 0.4679 | 0.5311 | 0.9749 *(Violates SLA)* |
| **Checkpoint 4 ($C_4$)** | 0.5559$^*$ | 0.5024 | 0.5306 | 0.4870 | 0.5317 | 0.9674$^*$ *(Violates SLA)* |
| **Contract SLA Floor** | --- | --- | --- | --- | --- | **$\ge 0.9800$** |

### Statistical Significance Notes
- $^{\dagger}p < 0.05$ on lexical precision (ROUGE-1 $p = 0.0372$, BLEU $p = 0.0231$, paired $t$-test).
- $^*p = 0.0224$ across long-horizon stream degradation ($C_1 \to C_4$, Wilcoxon signed-rank test).
- $N = 100$ evaluated queries per stream cohort.
