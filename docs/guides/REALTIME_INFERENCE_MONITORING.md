# Real-Time Inference Monitoring for LLM Distillation

## System Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   FastAPI    │    │    Kafka     │    │    Spark     │    │ TimescaleDB  │    │   Grafana    │
│  LLM Server  │───▶│   Cluster    │───▶│  Streaming   │───▶│   (TSDB)     │───▶│  Dashboard   │
│              │    │              │    │              │    │              │    │              │
│ • Teacher    │    │ • Buffer     │    │ • 5-min      │    │ • Time-series│    │ • Live Viz   │
│ • Student    │    │ • Partitions │    │   windows    │    │ • Aggregates │    │ • Alerts     │
│ • Metrics    │    │ • Replicas   │    │ • P95/P99    │    │ • Retention  │    │ • Comparison │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼                   ▼
  [Emit Events]      [Buffer & Order]    [Compute Stats]    [Store Series]     [Visualize]
```

---

## Explanation

This architecture implements a production-grade streaming pipeline for monitoring LLM inference in real-time. The FastAPI server emits inference metrics (latency, token counts, quality scores) for both teacher and student models to Apache Kafka, which buffers events with 3x replication for fault tolerance. Spark Structured Streaming consumes these events, computing windowed aggregations (average, P95, P99 latency; mean quality scores) over 5-minute tumbling windows with watermarking for late data handling. The aggregated metrics flow into TimescaleDB, a time-series database optimized for append-heavy workloads with automatic compression and continuous aggregates for efficient querying. Finally, Grafana dashboards visualize live metrics and trigger alerts when the student model's quality degrades relative to the teacher—enabling immediate detection of distillation performance issues in production.

---

## Relationship to GFS/HDFS/MapReduce

Our streaming architecture directly inherits principles from the Google File System and MapReduce paradigm. Kafka partitions mirror GFS's chunk-based storage model—data is divided across partitions (analogous to 64MB chunks) with configurable replication factors (typically 3x, matching GFS defaults) for fault tolerance, while the controller broker serves the same coordination role as the GFS master. Spark Structured Streaming extends the MapReduce programming model to unbounded data: the "map" phase parses incoming JSON events, the "shuffle" phase groups by model_type and time window, and the "reduce" phase computes aggregations—the key difference being that these operations run continuously over infinite streams rather than bounded datasets. Fault tolerance follows the same principles: just as MapReduce re-executes failed tasks from HDFS inputs, Spark checkpoints processing state to distributed storage (HDFS/S3), enabling exactly-once semantics through offset tracking and state recovery, while Kafka's replicated commit log ensures no data loss even during broker failures.

---

## Relationship to Cloud Computing & Elasticity

The architecture embodies cloud computing's core value proposition: elastic horizontal scaling with pay-per-use economics. FastAPI inference servers are stateless (state externalized to Kafka), enabling Kubernetes Horizontal Pod Autoscaler to scale from 2 pods during off-hours to 15+ pods during peak traffic—reducing compute costs by 50% compared to static provisioning. Kafka acts as a temporal decoupling layer, absorbing traffic bursts into partitions backed by cloud block storage (EBS/Persistent Disks), shifting the economic model from "provision for peak" to "provision for average, buffer the peaks." Spark on Kubernetes leverages dynamic allocation, requesting executor pods on-demand (including 70-90% cheaper spot/preemptible instances) and releasing them during quiet periods—particularly valuable for windowed aggregations that create periodic computation spikes. This elastic design enables the monitoring infrastructure to scale with inference volume (1000+ req/min) without becoming a bottleneck, while managed service options (Amazon MSK, Databricks, Timescale Cloud) reduce operational overhead, letting teams focus on model quality rather than infrastructure.
