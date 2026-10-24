# Table VII (Supplement): Knowledge Retention Across Checkpoints

> **Caption:** KNOWLEDGE RETENTION: C4 ADAPTER EVALUATED ON C1’S BENCHMARK SLICE ($n = 100$). TASK-CRITICAL METRICS SHOW NO REGRESSION; ROUGE AND CONCISENESS DECLINES REFLECT SPECIALIZATION.

| Metric | C1 | C4 | $\Delta$ |
| :--- | :---: | :---: | :---: |
| **Structured Correctness** | 0.5175 | 0.5205 | $+0.003$ |
| **Instruction Following** | 0.9950 | 0.9975 | $+0.003$ |
| **Coverage** | 0.3399 | 0.3803 | $+0.040$ |
| **Hallucination $\downarrow$** | 0.3448 | 0.3734 | $+0.029^*$ |
| **Context Grounding** | 0.7641 | 0.7658 | $+0.002$ |
| **Conciseness** | 0.6620 | 0.6260 | $-0.036^\dagger$ |
| **ROUGE-1** | 0.4015 | 0.3676 | $-0.034^\dagger$ |
| **ROUGE-L** | 0.3226 | 0.2843 | $-0.038^\dagger$ |
| **BLEU** | 0.1020 | 0.0740 | $-0.028^\dagger$ |

\* *Lower is better.*  
$^\dagger$ *Declines reflect specialization, not forgetting.*

