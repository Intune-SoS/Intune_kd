# Table 3: Software Engineering Delivery System Comparison

> **Caption:** Software engineering delivery comparison: ungated continuous fine-tuning (status quo), monolithic batch retraining, and contract-gated delivery (C3B-Adapter). Values for streaming workflows reflect empirical measurements from our prototype; batch retraining values reflect operational estimates based on retraining throughput across 20,000 samples.

| Operational / SE Dimension | Ungated Continuous Fine-Tuning | Monolithic Batch Retraining (Estimated) | C3B-Adapter Delivery Gate (Ours) |
| :--- | :---: | :---: | :---: |
| **Pre-Flight Verification** | ❌ None (promoted on converge) | ❌ Offline post-hoc evaluation | ✅ **Automated 3-Tier Behavioral Contract** |
| **Verification Latency** | 0 ms (unverified release) | Hours (full dataset re-evaluation) | ✅ **47.02 ms (measured on canary)** |
| **Regression Handling (C2 Candidate)** | ❌ Undetected (released silently) | ❌ Detected post-hoc | ✅ **Detected & Blocked ($R = 0.9666 < 0.98$)** |
| **User Exposure on Regression** | ❌ **Faulty weights served to live queries** | ❌ Cold deployment / offline switch | ✅ **0 queries exposed (traffic locked on $C_1$)** |
| **Cumulative Query Regressions** | ❌ **23.2% (4,588 / 19,810 queries)** | ❌ 23.2% localized regressions | ✅ **Zero-regression canary gate** |
| **Autonomous Self-Healing** | ❌ None (manual rollback / intervention) | ❌ None (requires full retrain run) | ✅ **Automated 10% Selective Replay** |
| **Post-Remediation Retention ($R$)** | --- | --- | ✅ **$R = 1.0214$ ($+0.0548$ on $R$ scale; $+5.67\%$ relative)** |
| **Artifact Durability & Integrity** | ❌ In-memory pointer swap risk | ❌ Manual script / server reboot | ✅ **SHA-256 Digest & Atomic `os.replace`** |

### Implementation & Measurement Notes
- **Ungated:** Evaluates streaming checkpoints without pre-flight gating, releasing weights immediately upon convergence.
- **Batch Retraining (Estimated):** Re-training all accumulated data (20,000 samples) requires an estimated $\sim 3.5$ hours of GPU compute based on observed fine-tuning throughput (8.2 samples/s), precluding sub-minute deployment cycles.
- **C3B-Adapter:** Measured values obtained from verified execution of `c3b_circuit_breaker.py` on the 100-example C1 canary slice under NVIDIA RTX 4070 (12GB) hardware.
