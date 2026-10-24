# Table IV: Teacher Selection Metrics

> **Caption:** TEACHER SELECTION METRICS, WHICH SHOWS THE ALPACA HAS SIGNIFICANTLY LOWER HALLUCINATION SCORES (0.53 VS 0.88) IN CONTEXT-BASED TASKS, CONFIRMING SUPERIOR ALIGNMENT AND FACTUAL GROUNDING. THIS ADVANTAGE DISAPPEARS IN ZERO-CONTEXT SETTINGS WHERE BOTH MODELS ACHIEVE PERFECT GROUNDING.

| Evaluation Metric | Context-Based Prompts: Alpaca | Context-Based Prompts: OSS | Context-Based Prompts: $p$-value | Context-Based Prompts: Winner | Non-Context Prompts: Alpaca | Non-Context Prompts: OSS | Non-Context Prompts: $p$-value | Non-Context Prompts: Winner |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sample Size ($N$)** | 1,732 | 1,732 | --- | --- | 2,268 | 2,268 | --- | --- |
| **Structural Correctness** | 0.49 | 0.56 | $< 0.001$ | OSS | 0.51 | 0.58 | $< 0.001$ | OSS |
| **Task Success** | 0.49 | 0.67 | $< 0.001$ | OSS | 0.64 | 0.67 | $< 0.05$ | OSS |
| **Instruction Following** | 0.99 | 0.99 | n.s. | Tie | 0.99 | 1.00 | n.s. | Tie |
| **Coverage** | 0.22 | 0.29 | $< 0.001$ | OSS | 0.18 | 0.19 | n.s. | Tie |
| **Faithfulness** | 0.34 | 0.25 | $< 0.001$ | Alpaca | 0.35 | 0.29 | $< 0.001$ | Alpaca |
| **Hallucination\*** | 0.53 | 0.88 | $< 0.001$ | Alpaca | 0.00 | 0.00 | n.s. | Tie |
| **Context Grounding** | 0.54 | 0.58 | $< 0.05$ | OSS | 1.00 | 1.00 | n.s. | Tie |
| **Overall Score** | 0.27 | 0.07 | $< 0.001$ | Alpaca | 0.62 | 0.62 | n.s. | Tie |

\* *Lower score indicates better performance.*  
*n.s. = Not statistically significant ($p \ge 0.05$).*

