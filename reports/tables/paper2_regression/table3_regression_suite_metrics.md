# Table III: Regression-Suite Metrics (N = 100) and Paired Test of C2 − C1

> **Caption:** REGRESSION-SUITE METRICS ($N = 100$) AND PAIRED TEST OF $C_2 - C_1$ (BH OVER THE NINE METRIC ROWS).

| Metric | Base | C1 | C2 | C3 | C4 | $\Delta(C_2 - C_1)$ | Wilcoxon $p$ | $t$ $p$ | BH $q$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Instruction following** | 0.9925 | 0.9838 | 0.9850 | 0.9812 | 0.9862 | $+0.0013$ | 0.3173 | 0.3197 | 0.4760 |
| **Structured correctness** | 0.6200 | 0.5185 | 0.5215 | 0.5170 | 0.5215 | $+0.0030$ | 0.1573 | 0.1583 | 0.2914 |
| **Coverage** | 0.3426 | 0.4475 | 0.4270 | 0.4297 | 0.4357 | $-0.0205$ | 0.4666 | 0.1437 | 0.5999 |
| **Hallucination rate $h$ ($\downarrow$)** | 0.3404 | 0.1895 | 0.1954 | 0.2054 | 0.2048 | $+0.0059$ | 0.7333 | 0.6609 | 0.7333 |
| **Context grounding** | 0.8245 | 0.8203 | 0.8153 | 0.8069 | 0.8047 | $-0.0050$ | 0.5738 | 0.7140 | 0.6455 |
| **Conciseness** | 0.6420 | 0.8240 | 0.8040 | 0.8360 | 0.8120 | $-0.0200$ | 0.1463 | 0.0770 | 0.2914 |
| **ROUGE-1** *(not in $s$)* | 0.3104 | 0.5534 | 0.5313 | 0.5536 | 0.5421 | $-0.0221$ | 0.0860 | 0.0372 | 0.2914 |
| **ROUGE-L** *(not in $s$)* | 0.2539 | 0.4800 | 0.4603 | 0.4691 | 0.4720 | $-0.0197$ | 0.1619 | 0.0600 | 0.2914 |
| **BLEU** *(not in $s$)* | 0.0587 | 0.2309 | 0.1978 | 0.2215 | 0.2208 | $-0.0331$ | 0.0935 | 0.0231 | 0.2914 |
| **Task success**$^\dagger$ | 0.6392 | 0.6248 | 0.6249 | 0.6324 | 0.6307 | $+0.0001$ | 0.3591 | 0.9918 | --- |
| **Faithfulness**$^\dagger$ | 0.3726 | 0.5505 | 0.5299 | 0.5348 | 0.5271 | $-0.0206$ | 0.2482 | 0.1291 | --- |
| **Overall score $S$** | 0.4330 | 0.5746 | 0.5590 | 0.5602 | 0.5559 | $-0.0156$ | 0.5683 | 0.1682 | --- |

$^\dagger$ *Part of $s$ but not stored by the evaluation run; recomputed from the stored outputs with the same metric code. Not in the BH family.*

