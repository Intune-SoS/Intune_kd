# Table V: Monolithic Generalist Baseline vs. Incremental CP4 Behavioral Divergence

> **Caption:** THE SPECIALIZATION PROOF: MONOLITHIC GENERALIST BASELINE VS. INCREMENTAL CP4 BEHAVIORAL DIVERGENCE. BOLD VALUES INDICATE WHERE CP4 OUTPERFORMS THE BATCH BASELINE. HAL: LOWER IS BETTER. \* WEIGHTED MEAN ACROSS CONTEXT TYPES.

| Evaluation Metric | Monolithic Baseline | CP4 With Context | CP4 Without Context |
| :--- | :---: | :---: | :---: |
| **Structured Correctness (SC)** | 0.5000 | **0.6201** | **0.6663** |
| **Instruction Following (IF)** | 0.5000 | **0.9865** | **0.9907** |
| **Coverage (COV)** | 0.5000 | **0.5297** | 0.3288 |
| **Hallucination (HAL) $\downarrow$** | 0.5000 | 0.7524 | **0.0000** |
| **Context Grounding (CG)** | 0.5000 | **0.8316** | **1.0000** |
| **Conciseness (CONC)** | 0.5000 | 0.2760 | 0.3205 |
| **ROUGE-1** | 0.1321\* | 0.1143 | **0.1653** |
| **ROUGE-L** | 0.1034\* | 0.0926 | **0.1201** |
| **BLEU** | 0.0205\* | **0.0209** | **0.0229** |

\* *Weighted mean of Batch ROUGE/BLEU scores across all context types.*

