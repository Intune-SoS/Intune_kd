# Table VI: Orchestration Latency Comparison Across Polling Strategies

> **Caption:** ORCHESTRATION LATENCY COMPARISON ACROSS POLLING STRATEGIES. AGAINST A NAIVE POLLING BASELINE, GPU RESOURCES REMAIN IDLE FOR THE FULL POLL INTERVAL. AGAINST A REALISTIC ASYNC POLLING BASELINE THAT OVERLAPS DATA PREFETCHING WITH PRIOR TRAINING, INTUNE’S EVENT-DRIVEN ARCHITECTURE ACHIEVES COMPARABLE RAW LATENCY WHILE ELIMINATING SPECULATIVE COMPUTATION AND PROVIDING FAULT-TOLERANT EXACTLY-ONCE CHECKPOINT DELIVERY.

| System | Mean Transition Latency | Notes |
| :--- | :---: | :--- |
| **Naive Polling Baseline** | $\sim 150\text{ s}$ | GPU idle during full 300s poll interval; no data prefetching |
| **Async Polling Baseline** | $6.41\text{ s} \pm 0.18\text{ s}$ | Data prefetched during prior training step; standard production optimization |
| **INTUNE Event-Driven** | $6.50\text{ s}$ | CDC + Kafka push; no speculative computation |

