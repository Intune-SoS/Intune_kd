# Table 2: Granular 9-Metric Canary Benchmark Across Model Versions

> **Caption:** Deterministic nine-metric benchmark on pinned C1 canary ($N=100$) across four model iterations: untuned base student, accepted incumbent ($C_1$), ungated candidate ($C_2$-Original), and contract-remediated patch ($C_2$-Replay). $\Delta_{\rm heal}$ denotes recovery delta ($C_2$-Replay $-$ $C_2$-Original). Reported $p$-values reflect paired tests comparing $C_2$-Replay against $C_2$-Original.

| Metric | Base Model | Accepted Incumbent ($C_1$) | Ungated Candidate ($C_2$-Orig) | Remediated Patch ($C_2$-Replay) | Recovery Delta ($\Delta_{\rm heal}$) | Paired $p$-value (Unadjusted) | FDR Adjusted $q$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Instruction Following (T1) | 0.9925 | 0.9838 | 0.9862 | 0.9838 | $-0.0024$ | $p = 0.3197$ | $q = 0.3197$ |
| Structured Correctness (T1) | 0.6200 | 0.5185 | 0.5200 | 0.5170 | $-0.0030$ | $p = 0.1583$ | $q = 0.1781$ |
| Coverage | 0.3426 | 0.4475 | 0.4237 | 0.4451 | $+0.0214$ | $p = 0.0482^*$ | $q = 0.0620$ |
| **Hallucination Rate ($\downarrow$, T2)** | 0.3404 | 0.1800 | 0.1997 | **0.1717** | $\mathbf{-0.0280}$ | $\mathbf{p = 0.0084^{**}}$ | $\mathbf{q = 0.0189^*}$ |
| Context Grounding | 0.8245 | 0.8203 | 0.8030 | 0.8258 | $+0.0228$ | $p = 0.0416^*$ | $q = 0.0624$ |
| **Conciseness** | 0.6420 | 0.8240 | 0.8060 | 0.8360 | $+0.0300$ | $\mathbf{p = 0.0092^{**}}$ | $\mathbf{q = 0.0166^*}$ |
| **ROUGE-1** | 0.3104 | 0.5534 | 0.5206 | 0.5587 | $+0.0381$ | $\mathbf{p = 0.0041^{**}}$ | $\mathbf{q = 0.0185^*}$ |
| **ROUGE-L** | 0.2540 | 0.4800 | 0.4517 | 0.4844 | $+0.0327$ | $\mathbf{p = 0.0068^{**}}$ | $\mathbf{q = 0.0204^*}$ |
| **BLEU** | 0.0587 | 0.2309 | 0.2034 | 0.2290 | $+0.0256$ | $\mathbf{p = 0.0035^{**}}$ | $\mathbf{q = 0.0315^*}$ |
| **Overall Macro Score** | 0.4330 | 0.5693 | 0.5554 | **0.5869** | $\mathbf{+0.0315}$ | $\mathbf{p < 0.001^{***}}$ | $\mathbf{q < 0.001^{***}}$ |
| **Backward Retention ($R$, T2)** | --- | 1.0000 | **0.9666** *(Breach)* | **1.0214** *(Pass)* | $\mathbf{+0.0548}$ | $\mathbf{p = 0.0182^*}$ | $\mathbf{q = 0.0273^*}$ |

### Statistical Notes
- Bold rows denote contract-critical clauses (Tier 1 invariants, Tier 2 retention/hallucination SLAs) or improvements meeting false discovery rate (FDR) control at $q < 0.05$ (Benjamini-Hochberg adjustment).
- Unadjusted $p$-values computed via two-sided paired $t$-test ($N=100$) comparing $C_2$-Replay directly with $C_2$-Original.
- Tier 1 functional properties (Instruction Following $\ge 0.9500$, Structured Correctness $\ge 0.5000$) satisfied invariant thresholds under both candidates.
