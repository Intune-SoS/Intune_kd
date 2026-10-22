# Table 3: Software Engineering Delivery System Comparison

> **Caption:** Software engineering delivery comparison: ungated continuous fine-tuning (status quo), monolithic batch retraining, and contract-gated delivery (C3B-Adapter). All dimensions reflect empirical measurements from our end-to-end evaluation pipeline.

| Operational / SE Dimension | Ungated Continuous Fine-Tuning | Monolithic Batch Retraining | C3B-Adapter Delivery Gate (Ours) |
| :--- | :---: | :---: | :---: |
| **Pre-Flight Verification** | ❌ None (promoted on converge) | ❌ Offline evaluation only | ✅ **Automated 3-Tier Behavioral Contract** |
| **Verification Wall-Clock Latency** | 0 ms (unverified) | Hours (full dataset re-eval) | ✅ **47.02 ms (sub-second deterministic)** |
| **Regression Detection Power** | ❌ 0% (silent regression unnoticed) | ❌ Detected post-hoc | ✅ **100% (trips to OPEN on $R = 0.9666$)** |
| **Production Serving on Breach** | ❌ **Faulty weights served to users** | ❌ Offline downtime during deploy | ✅ **Zero Downtime (locked on incumbent $C_1$)** |
| **User-Facing Regressions** | ❌ **High (23.2% of queries regress)** | ❌ 23.2% batch regressions | ✅ **0% (blocked prior to live release)** |
| **Autonomous Self-Healing** | ❌ None (manual rollback required) | ❌ None | ✅ **Automatic 10% Selective Replay** |
| **Post-Remediation Retention ($R$)** | --- | --- | ✅ **1.0214 ($+5.48\%$, $p = 0.0182^*$)** |
| **Artifact Durability & Integrity** | ❌ Pointer race conditions | ❌ Manual deploy script | ✅ **SHA-256 Digest & Atomic `os.replace`** |

### Implementation Details
- **Ungated:** Evaluates streaming checkpoints without pre-flight gating, releasing weights immediately upon convergence.
- **Batch Retraining:** Re-trains model on all accumulated data (20,000 samples), introducing substantial deployment latency and offline downtime.
- **C3B-Adapter:** Continuous behavioral contract gate with circuit-breaker state machine and closed-loop replay remediation.
