# Table IV: Release Decisions (N = 100)

> **Caption:** RELEASE DECISIONS ($N = 100$). NEW-DATA CHECK: SCORE CHANGE ON THE NEWEST TEST SET. FIXED: THRESHOLD WITHOUT UNCERTAINTY. PROPOSED: EQ. 2. \*AGAINST $C_1$. $^\ddagger$HALLUCINATION CHECK FAILED. G3 NOT MEASURED.

| New | Deployed | New-Data Check | $R$ | 95% CI | $P(R^* < 0.98)$ | $\Delta$ Halluc. | Share Down | Wilcoxon $p$ | Fixed Decision | Proposed Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C2** | **C1** | $+0.0047$ (release) | 0.9729 | [0.9336, 1.0093] | 0.632 | $+0.0059$ | 0.30 | 0.5683 | fail | **inconclusive** |
| **C3** | **C2** | $+0.0014$ (release) | 1.0022 | [0.9724, 1.0326] | 0.076 | $+0.0100$ | 0.28 | 0.8385 | pass | **inconclusive** |
| **C4** | **C3** | $+0.0191$ (release) | 0.9922 | [0.9701, 1.0169] | 0.145 | $-0.0006$ | 0.36 | 0.1543 | pass | **inconclusive** |
| **C3** | **C1\*** | --- | 0.9751 | [0.9411, 1.0018] | 0.600 | $+0.0159$ | 0.31 | 0.4706 | fail | **fail**$^\ddagger$ |
| **C4** | **C1\*** | --- | 0.9674 | [0.9303, 1.0012] | 0.752 | $+0.0153$ | 0.42 | 0.0224 | fail | **fail**$^\ddagger$ |

\* *Comparison evaluated against deployed baseline $C_1$.*  
$^\ddagger$ *Failed due to regression check: hallucination rate increase exceeds the $+0.0150$ tolerance boundary.*

