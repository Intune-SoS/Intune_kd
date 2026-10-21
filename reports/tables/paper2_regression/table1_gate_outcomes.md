# Table I: The Three Outcomes of the Proposed Gate

> **Caption:** THE THREE OUTCOMES OF THE PROPOSED GATE.

| Outcome | Condition | Action |
| :--- | :--- | :--- |
| **Pass** | Whole interval of $R$ at or above $0.98$; G1, G3, and the hallucination check hold | Release the new version |
| **Fail** | Whole interval below $0.98$, or G1, G3, or the hallucination check fails | Reject; keep the deployed version |
| **Inconclusive** | Interval contains $0.98$ | Run more tests or ask a reviewer |

### Gate Checks Specification
* **G1 (Functional check):** Mean instruction-following score $\ge 0.95$ and mean structured-correctness score $\ge 0.50$.
* **G2 (Regression check):** $R \ge 0.98$, and the hallucination rate does not rise by more than $0.015$ over the deployed version.
* **G3 (Resource check):** Mean latency per prompt $\le 3.0\text{ s}$ and peak GPU memory $\le 4.0\text{ GB}$.

