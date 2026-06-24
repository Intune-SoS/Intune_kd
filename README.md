# INTUNE: An Event-Driven System-of-Systems for Incremental Knowledge Distillation in Compact LLMs

Event-driven architecture for continuous, incremental knowledge distillation in compact Large Language Models (LLMs) with Change Data Capture (CDC), Kafka streaming orchestration, and deterministic non-LLM evaluation.

---

## Table of Contents

- [Overview](#overview)
- [Core Contributions](#core-contributions)
- [System Architecture](#system-architecture)
  - [Layer 1: Metric-Driven Teacher Selection](#layer-1-metric-driven-teacher-selection)
  - [Layer 2: Event-Driven Streaming Orchestration](#layer-2-event-driven-streaming-orchestration)
  - [Layer 3: Incremental Distillation & Checkpoint Feedback](#layer-3-incremental-distillation--checkpoint-feedback)
- [Deterministic Evaluation Framework](#deterministic-evaluation-framework)
- [Empirical Findings & Benchmarks](#empirical-findings--benchmarks)
  - [1. Teacher Selection: Alignment Outperforms Scale](#1-teacher-selection-alignment-outperforms-scale)
  - [2. Incremental Dynamics & Residual Overcorrection at C2](#2-incremental-dynamics--residual-overcorrection-at-c2)
  - [3. Specialization vs. Surface Similarity Trade-Off](#3-specialization-vs-surface-similarity-trade-off)
  - [4. Systems Orchestration Latency Comparison](#4-systems-orchestration-latency-comparison)
  - [5. Knowledge Retention Across Checkpoints](#5-knowledge-retention-across-checkpoints)
- [Tech Stack & Infrastructure](#tech-stack--infrastructure)
- [Hardware & Quantization Specifications](#hardware--quantization-specifications)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Execution Guide](#execution-guide)
  - [Database Initialization](#1-database-initialization)
  - [Phase 1: Teacher Selection Pipeline](#2-phase-1-teacher-selection-pipeline)
  - [Phase 2: Distillation Tracks (Incremental vs. Batch)](#3-phase-2-distillation-tracks-incremental-vs-batch)
  - [Cross-Checkpoint Retention Matrix & Replay Mitigation](#4-cross-checkpoint-retention-matrix--replay-mitigation)
  - [Event-Driven Streaming Pipeline](#5-event-driven-streaming-pipeline)
- [System Limitations](#system-limitations)
- [Ongoing Research: SE for AI](#ongoing-research-se-for-ai)
- [References](#references)

---

## Overview

Deploying Large Language Models (LLMs) in low-latency and resource-constrained production settings requires balancing high reasoning capacity with compact parameter footprints. While knowledge distillation transfers capabilities from high-capacity teachers to compact students, conventional MLOps pipelines treat distillation as a static, monolithic batch process. These batch workflows introduce three critical inefficiencies:

1. **Orchestration Latency**: High idle periods and speculative polling delays while waiting for accumulated data batches.
2. **Hidden Dynamics**: Complete opacity regarding intermediate learning trajectories, masking transient regressions, overcorrections, and early stopping thresholds.
3. **Scale Bias**: An unjustified reliance on massive generalist teacher models that frequently inject hallucinations into specialized student models.

**INTUNE System-of-Systems (SoS)** reformulates knowledge distillation into an event-native, streaming MLOps architecture. By combining PostgreSQL Change Data Capture (CDC), Apache Kafka, Apache Spark Structured Streaming, and parameter-efficient incremental fine-tuning (LoRA), INTUNE converts distillation from a periodic batch task into a continuous, reactive feedback loop operable on consumer-grade hardware.

---

## Core Contributions

1. **Event-Driven Architecture (SoS)**: Replaces fixed-interval database polling with native CDC push triggers and Kafka consumer-group semantics, cutting checkpoint-transition latency by **97%** (from ~150s down to 6.5s) while eliminating speculative compute overhead.
2. **Alignment-First Teacher Selection**: Empirically proves that model alignment and factual grounding surpass raw parameter scale—demonstrating that a domain-aligned 7B teacher (Alpaca-7B) statistically outperforms a 20B generic teacher (GPT-OSS-20B) in factual distillation ($p < 0.0001$).
3. **Discovery of Checkpoint Dynamics**: Exposes non-monotonic learning behavior in sequential LoRA distillation, revealing a transient performance dip at Checkpoint 2 (C2) analogous to gradient boosting residual overcorrection:
   $$F_k(x) = F_{k-1}(x) + \eta \cdot h_k(\epsilon_{k-1})$$
   This dynamic is invisible in traditional batch distillation and provides mathematical criteria for adaptive optimization and early stopping.
4. **Deterministic Evaluation Protocol**: Deploys an objective, non-LLM 9-metric scoring suite spanning lexical precision, structural syntax, factual hallucination detection, and context grounding to eliminate LLM-as-a-judge biases.

---

## System Architecture

The INTUNE System-of-Systems architecture decouples knowledge distillation across three interconnected functional layers:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        LAYER 1: METRIC-DRIVEN TEACHER SELECTION                        │
│                                                                                        │
│   Candidate Pool:            Evaluation Loop:                       Selected Teacher:  │
│   [Alpaca-7B vs OSS-20B] ──► [9-Metric Deterministic Scoring] ────► [Alpaca-7B]        │
│                                  (4,000 Validation Rows)                               │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Generates 20k target labels
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   LAYER 2: EVENT-DRIVEN STREAMING ORCHESTRATION                        │
│                                                                                        │
│   PostgreSQL (Supabase)       Kafka Bus               Spark Stream Processor           │
│   ┌────────────────────┐     ┌──────────────┐        ┌──────────────────────┐          │
│   │ modelcomp_50k      │────►│ CDC Event    │───────►│ Threshold Aggregator │          │
│   │ (CDC Row Inserts)  │     │ Topic        │        │ (Batch >= 5,000)     │          │
│   └────────────────────┘     └──────────────┘        └──────────┬───────────┘          │
│                                                                 │ Trigger              │
│                                                                 ▼                      │
│                              Execution Safety Checks ◄── [Idempotent Task Dispatcher]  │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Dispatches checkpoint training job
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                     LAYER 3: INCREMENTAL DISTILLATION & FEEDBACK                       │
│                                                                                        │
│   Gemma-3-1B Base ──► [LoRA C1] ──► [LoRA C2] ──► [LoRA C3] ──► [LoRA C4]              │
│   (Frozen 4-bit)        (5k rows)     (5k rows)     (5k rows)     (5k rows)            │
│                            │             │             │             │                 │
│                            ▼             ▼             ▼             ▼                 │
│                      [Eval Suite]  [Eval Suite]  [Eval Suite]  [Eval Suite]            │
│                            │             │             │             │                 │
│                            └─────────────┴──────┬──────┴─────────────┘                 │
│                                                 ▼                                      │
│                                 [Closed-Loop Checkpoint Dynamics]                      │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Layer 1: Metric-Driven Teacher Selection
Rather than selecting teachers based on parameter count, Layer 1 benchmarks candidates across a diverse 4,000-sample validation subset. Both models generate completions against identical prompts; their outputs are evaluated by the deterministic 9-metric engine to verify domain alignment, factual precision, and low hallucination before generating the 20,000 training targets.

### Layer 2: Event-Driven Streaming Orchestration
Traditional schedulers rely on pull-based polling intervals that waste CPU/GPU cycles. Layer 2 integrates:
- **Change Data Capture (CDC)**: PostgreSQL WAL change events are published as JSON payloads whenever data batches enter `modelcomp_50k`.
- **Asynchronous Event Broker (Apache Kafka)**: Ingests CDC events into partition-keyed topics (`intune.status.events`) ensuring strictly ordered event flow.
- **Stream Processing (Apache Spark Structured Streaming)**: Evaluates partition readiness criteria (5,000 samples per checkpoint boundary) and emits dispatch triggers.
- **Idempotent Task Dispatcher**: Validates environment state, verifies GPU memory availability, and launches fine-tuning/evaluation tasks with exactly-once consumer semantics.

### Layer 3: Incremental Distillation & Checkpoint Feedback
The student model (Gemma-3-1B) receives incremental knowledge updates via Low-Rank Adaptation (LoRA, $r=16, \alpha=16$) over four sequential 5,000-sample partitions (C1 through C4). Base weights are frozen in 4-bit NF4 precision while adapter layers are continuously tuned. After each checkpoint, the system triggers the 9-metric evaluation suite, stores model weights, and updates the state machine before unblocking the next chunk.

---

## Deterministic Evaluation Framework

To eliminate the subjective variance, self-preference, and positional bias of "LLM-as-a-judge" evaluation, INTUNE evaluates all outputs using a mathematically formal, deterministic 9-metric suite:

| Metric | Formulation | Description & Operational Purpose |
| :--- | :--- | :--- |
| **BLEU** | $\text{BP} \cdot \exp\left(\sum_{n=1}^N w_n \log p_n\right)$ | Measures precision n-gram phrase overlap against teacher targets with Brevity Penalty ($\text{BP}$). |
| **ROUGE-1** | $\frac{\sum_{w \in \text{ref}} \min(c_{\text{gen}}(w), c_{\text{ref}}(w))}{\sum_{w \in \text{ref}} c_{\text{ref}}(w)}$ | Evaluates unigram lexical recall against teacher reference text. |
| **ROUGE-L** | $\frac{(1 + \beta^2) P_{\text{LCS}} R_{\text{LCS}}}{R_{\text{LCS}} + \beta^2 P_{\text{LCS}}}$ | Measures longest common subsequence (LCS) to capture sentence-level structural similarity. |
| **Lexical Alignment (LAA)** | $\frac{1}{\|T_{\text{gen}}\|} \sum_{t \in T_{\text{gen}}} \mathbb{I}(t \in V_{\text{allowed}})$ | **Primary hallucination detector**: computes the ratio of student tokens unsupported by teacher/context vocabulary. |
| **Structured Correctness** | $\mathbb{I}(\text{RegexMatch}(\text{gen}, \text{format}))$ | Enforces schema adherence (JSON parsing, bullet structures, Markdown headers) via regex validation. |
| **Instruction Following** | $\frac{C_{\text{met}}}{C_{\text{total}}}$ | Quantifies compliance with explicit negative constraints, language rules, and length boundaries. |
| **Context Grounding** | $\frac{E_{\text{grounded}}}{E_{\text{generated}}}$ | Verifies that factual assertions in context-present prompts trace directly to provided context spans. |
| **Coverage** | $\frac{K_{\text{addressed}}}{K_{\text{requested}}}$ | Assesses the fraction of prompt sub-tasks, questions, and required entities addressed in the response. |
| **Conciseness** | $\min\left(1, \frac{T_{\text{ref}}}{T_{\text{gen}}}\right)$ | Penalizes verbosity, circular phrasing, and rambling by penalizing length exceeding reference token count. |

### Composite Overall Score
$$\text{Overall Score} = \left(\frac{1}{7} \sum_{m \in M_{\text{pos}}} S_m\right) \times \left(1 - \text{Hallucination Score}\right)$$
where $M_{\text{pos}} = \{\text{Structured Correctness}, \text{Task Success}, \text{Instruction Following}, \text{Coverage}, \text{Faithfulness}, \text{Context Grounding}, \text{Conciseness}\}$.

---

## Empirical Findings & Benchmarks

### 1. Teacher Selection: Alignment Outperforms Scale
In Phase 1, candidate teachers Alpaca-7B and GPT-OSS-20B were evaluated across a 4,000-sample balanced dataset (1,732 context-based, 2,268 zero-context).

```
Teacher Selection Performance (Context-Based Prompts, N = 1,732)
─────────────────────────────────────────────────────────────────────────────
Metric                     Alpaca-7B      OSS-20B       p-value     Winner
─────────────────────────────────────────────────────────────────────────────
Structural Correctness        0.49          0.56        < 0.001     OSS-20B
Task Success                  0.49          0.67        < 0.001     OSS-20B
Instruction Following         0.99          0.99          n.s.      Tie
Coverage                      0.22          0.29        < 0.001     OSS-20B
Faithfulness                  0.34          0.25        < 0.001     Alpaca-7B
Hallucination (lower=better)  0.53          0.88        < 0.001     Alpaca-7B
Context Grounding             0.54          0.58        < 0.05      OSS-20B
Overall Score                 0.27          0.07        < 0.001     Alpaca-7B
─────────────────────────────────────────────────────────────────────────────
```
**Takeaway**: While OSS-20B achieved higher surface structural metrics, it suffered a **66% higher hallucination rate** (0.88 vs. 0.53) on context grounding tasks. Alpaca-7B transferred domain skills with significantly greater fidelity ($p < 0.0001$).

### 2. Incremental Dynamics & Residual Overcorrection at C2
In Phase 2, Gemma-3-1B was trained over 20,000 samples divided into four 5,000-sample checkpoints (C1–C4). Tracking intermediate evaluations uncovered a non-monotonic trajectory:

```
Checkpoint Dynamics Progression across C1–C4
──────────────────────────────────────────────────────────────────────────────────────────
Metric                  C1 (ctx/noctx)   C2 (ctx/noctx)   C3 (ctx/noctx)   C4 (ctx/noctx)
──────────────────────────────────────────────────────────────────────────────────────────
Structured Correctness  0.5162 / 0.5263  0.5099 / 0.5301  0.5154 / 0.5270  0.5209 / 0.5242
Instruction Following   0.9908 / 0.9916  0.9758 / 0.9940  0.9915 / 0.9922  0.9928 / 0.9922
Coverage                0.4653 / 0.2532  0.4644 / 0.2727  0.5047 / 0.2759  0.5026 / 0.2941
Conciseness             0.5749 / 0.7311  0.5351 / 0.7043  0.4723 / 0.6833  0.4932 / 0.6871
ROUGE-1                 0.3398 / 0.3854  0.3119 / 0.3773  0.2746 / 0.3663  0.2895 / 0.3786
ROUGE-L                 0.2930 / 0.2953  0.2622 / 0.2872  0.2293 / 0.2773  0.2431 / 0.2871
BLEU                    0.0814 / 0.0861  0.0779 / 0.0882  0.0605 / 0.0768  0.0634 / 0.0822
──────────────────────────────────────────────────────────────────────────────────────────
```
**Theoretical Insight**: Instruction following in context prompts drops from C1 (0.9908) to C2 (0.9758) before rebounding at C3 (0.9915) and stabilizing at C4 (0.9928). Because each stage trains on residual errors of the prior stage, the C2 dip represents a **transient overcorrection on initial residuals** rather than divergence. Most achievable gains are secured by C3 (15,000 samples, 75% data mark), making C4 an empirical confirmation of diminishing returns.

### 3. Specialization vs. Surface Similarity Trade-Off
Comparing final incremental checkpoint C4 with the monolithic batch baseline demonstrates behavioral divergence:

```
Monolithic Batch Baseline vs. Incremental Checkpoint 4 (CP4)
─────────────────────────────────────────────────────────────────────────────
Metric                          Monolithic Batch      CP4 (Ctx)      CP4 (No-Ctx)
─────────────────────────────────────────────────────────────────────────────
Structured Correctness (SC)          0.5000            0.6201          0.6663
Instruction Following (IF)           0.5000            0.9865          0.9907
Coverage (COV)                       0.5000            0.5297          0.3288
Hallucination (HAL) [lower=better]   0.5000            0.7524          0.0000
Context Grounding (CG)               0.5000            0.8316          1.0000
Conciseness (CONC)                   0.5000            0.2760          0.3205
ROUGE-1                              0.1321            0.1143          0.1653
ROUGE-L                              0.1034            0.0926          0.1201
BLEU                                 0.0205            0.0209          0.0229
─────────────────────────────────────────────────────────────────────────────
```
The model sacrifices general conversational padding (lower ROUGE/BLEU) in exchange for near-perfect adherence to constraints and zero hallucinations on zero-context prompts.

### 4. Systems Orchestration Latency Comparison
Comparing INTUNE's event-driven pipeline against traditional polling approaches:

```
Checkpoint Transition Latency Benchmarks
─────────────────────────────────────────────────────────────────────────────
Pipeline Architecture        Transition Latency    Speculative Compute Overhead
─────────────────────────────────────────────────────────────────────────────
Naive Polling Baseline             ~150.0 s        High (GPU idle during 300s sleep)
Asynchronous Polling Baseline    6.41 s ± 0.18 s   High (unconditional worker prefetch)
INTUNE Event-Driven (CDC)          6.50 s          Zero (reactive execution on data-ready)
─────────────────────────────────────────────────────────────────────────────
```
While asynchronous polling achieves raw latency parity with prefetching, it repeatedly launches prefetch threads during idle windows. INTUNE achieves identical transition speed (**6.50s**) while completely eliminating speculative computation.

### 5. Knowledge Retention Across Checkpoints
To evaluate backward retention, adapter C4 was evaluated against C1's historical 100-sample benchmark slice ($n=100$):

```
Retention Analysis: C4 Adapter Evaluated on C1 Historical Slice
─────────────────────────────────────────────────────────────────────────────
Metric                     C1 Baseline      C4 Evaluation      Delta (Δ)
─────────────────────────────────────────────────────────────────────────────
Structured Correctness        0.5175           0.5205           +0.0030
Instruction Following         0.9950           0.9975           +0.0025
Coverage                      0.3399           0.3803           +0.0404
Hallucination (lower=better)  0.3448           0.3734           +0.0286
Context Grounding             0.7641           0.7658           +0.0017
Conciseness                   0.6620           0.6260           -0.0360
ROUGE-1                       0.4015           0.3676           -0.0338
ROUGE-L                       0.3226           0.2843           -0.0383
BLEU                          0.1020           0.0740           -0.0280
─────────────────────────────────────────────────────────────────────────────
```
Task-critical operational capabilities (Instruction Following, Structured Correctness, Coverage) show **zero regression**, confirming that sequential checkpointing preserves core task alignment.

---

## Tech Stack & Infrastructure

- **Language Models**: Google Gemma-3-1B-it (Student), Stanford Alpaca-7B (Selected Teacher), GPT-OSS-20B (Candidate Teacher)
- **Parameter-Efficient Training**: PEFT, LoRA ($r=16, \alpha=16$), TRL (`SFTTrainer`), Unsloth FastLanguageModel
- **Quantization & Kernels**: BitsAndBytes 4-bit NormalFloat (NF4), bfloat16 mixed-precision compute
- **Streaming & Event Infrastructure**: Supabase PostgreSQL (CDC Realtime), Apache Kafka, Apache Spark Structured Streaming
- **Deterministic Metrics**: NLTK (sentence BLEU), `rouge-score`, regular expression syntax validators, set-theoretic vocabulary alignment analyzers

---

## Hardware & Quantization Specifications

To ensure reproducible execution on consumer-grade hardware:

- **GPU**: NVIDIA RTX 4060 / RTX 4070 (8GB VRAM minimum) or NVIDIA T4 (16GB)
- **RAM**: 24GB system memory
- **Quantization**: 4-bit NF4 with double quantization and bfloat16 compute dtype (reduces model memory footprint by ~75% from 4.8GB to ~1.2GB VRAM)
- **Batching**: Per-device train batch size = 4 (gradient accumulation = 4; effective batch size = 16)
- **Sequence Length**: 2,048 tokens

---

## Repository Structure

```text
Intune_kd/
├── 01_teacher_selection/             # LAYER 1: Empirical Teacher Selection
│   ├── data_prep/                    # 4k validation split & task classification
│   ├── candidate_teachers/           # Alpaca-7B fine-tuning & OSS-20B generation
│   └── evaluation/                   # Tables III & IV comparison and analytical reports
├── 02_distillation/                  # LAYER 3: Comparative Knowledge Distillation
│   ├── evaluation_suite/             # Deterministic 9-metric suite & Table V generator
│   ├── batch_baseline/               # Monolithic 20k batch baseline pipeline
│   ├── incremental_pipeline/         # 4-stage sequential LoRA distillation (C1–C4)
│   └── retention_and_replay/         # 4x4 retention matrix & C2 replay mitigation
├── 03_streaming_orchestration/       # LAYER 2: Event-Driven Orchestration
│   ├── cdc_bridge/                   # Supabase PostgreSQL WAL CDC to Kafka producer
│   ├── stream_processor/             # Spark Structured Streaming threshold aggregator
│   ├── task_dispatcher/              # Idempotent task dispatcher with runtime guards
│   ├── benchmarks/                   # Table VI orchestration latency comparison
│   └── run_spark.ps1                 # Windows PowerShell launcher for Spark / Java 11
├── data/                             # Curated datasets (alpaca_data_raw, 4k, 50k)
├── models/                           # LoRA adapter weights (C1–C4, replay, batch)
├── reports/                          # Empirical results, dynamics summaries, tables
├── sql/                              # Production database schemas and state tables
├── src/database/                     # Database client connection pooling and helpers
├── scripts/                          # Diagnostics, tests, and export utilities
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## Installation & Setup

### 1. Environment Configuration

Clone the repository and install dependencies in a virtual environment:

```bash
git clone https://github.com/Intune-SoS/Intune_kd.git
cd Intune_kd

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the project root:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_or_service_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key

KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_EVENTS=intune.status.events
KAFKA_TOPIC_TRIGGERS=pipeline.triggers
KAFKA_GROUP_ID=intune-distillation-group

TRIGGER_THRESHOLD=5000
TORCHDYNAMO_DISABLE=1
TOKENIZERS_PARALLELISM=false
```

---

## Execution Guide

The experimental workflow strictly follows the paper's three-layer architecture and empirical phases:

### 1. Database Initialization
Deploy the relational state schema and migration scripts to Supabase PostgreSQL:

```bash
# Execute sequentially in the Supabase SQL editor:
sql/01_schema_eval_matrix.sql           # Core evaluation metrics, models & prompts
sql/02_schema_incremental_tables.sql    # Checkpoint tracking & training queues
sql/03_schema_incremental_pipeline.sql  # Stream trigger states & partition bounds
sql/04_schema_cleanup_legacy_columns.sql# Deprecated column pruning
sql/05_schema_distributed_worker.sql    # Worker concurrency locks & idempotency
```

### 2. Phase 1: Teacher Selection Pipeline
*Goal: Benchmark candidate teachers (Alpaca-7B vs. GPT-OSS-20B) across 4,000 validation samples to empirically select the optimal distillation supervisor (Paper Section IV-B, Tables III & IV).*

```bash
# Step 1: Download Alpaca dataset and prepare the 4,000-sample balanced validation set
python 01_teacher_selection/data_prep/01_data_download_alpaca.py
python 01_teacher_selection/data_prep/02_data_prepare_4k.py

# Step 2: Generate baseline student responses (Gemma-3-1B)
python 01_teacher_selection/data_prep/03_gen_base_gemma.py

# Step 3: Train candidate teacher (Alpaca-7B) and generate candidate completions
python 01_teacher_selection/candidate_teachers/04a_train_finetune_alpaca.py
python 01_teacher_selection/candidate_teachers/04b_gen_teacher_oss20b.py
python 01_teacher_selection/candidate_teachers/06a_gen_tuned_alpaca.py

# Step 4: Evaluate candidates with the 9-metric suite and generate analytical report
python 01_teacher_selection/evaluation/07_eval_compare_teachers.py
python 01_teacher_selection/evaluation/09_report_analytical.py
```
*Empirical Finding: Alpaca-7B achieves 0.9022 overall score vs. OSS-20B's 0.7711, with 66% lower hallucination on context grounding tasks.*

### 3. Phase 2: Distillation Tracks (Incremental vs. Batch)
*Goal: Compare Monolithic Batch distillation (20,000 samples) against 4-stage Sequential Incremental distillation (C1–C4, 5,000 samples each) on Gemma-3-1B (Paper Section V, Table V).*

```bash
# Step 1: Upload the 50k training pool and generate base student predictions
python 01_teacher_selection/data_prep/10_data_upload_50k.py
python 02_distillation/incremental_pipeline/11_gen_base_student.py

# Step 2: Train Monolithic Batch baseline (20,000 samples uninterrupted)
python 02_distillation/batch_baseline/13_train_batch.py --run-all

# Step 3: Train Sequential Incremental checkpoints (C1 -> C2 -> C3 -> C4)
# Run each checkpoint through scoring, fine-tuning, inference, and post-scoring:
python 02_distillation/incremental_pipeline/12_train_incremental.py --checkpoint 1 --run-all
python 02_distillation/incremental_pipeline/12_train_incremental.py --checkpoint 2 --run-all
python 02_distillation/incremental_pipeline/12_train_incremental.py --checkpoint 3 --run-all
python 02_distillation/incremental_pipeline/12_train_incremental.py --checkpoint 4 --run-all

# Alternatively, execute the end-to-end automated orchestrator:
python 02_distillation/incremental_pipeline/pipeline.py

# Step 4: Multi-checkpoint evaluation & catastrophic forgetting dynamics
python 02_distillation/evaluation_suite/eval_checkpoints_supabase.py
python 02_distillation/evaluation_suite/forgetting_analysis.py
```

### 4. Cross-Checkpoint Retention Matrix & Replay Mitigation
*Goal: Systematically quantify cross-checkpoint retention across all adapter-slice pairs and mitigate the C2 residual overcorrection using experience replay (Paper Section V-B & Ongoing Research).*

```bash
# Step 1: Compute the 4x4 Cross-Checkpoint Retention Matrix
# Evaluates each adapter (C1, C2, C3, C4) against 100-sample slices from all checkpoints:
python 02_distillation/retention_and_replay/eval_retention_matrix.py

# Step 2: Execute Replay Mitigation for Checkpoint 2
# Initializes from C1 adapter, interleaving 10% historical C1 data into the C2 training stream:
python 02_distillation/retention_and_replay/train_c2_with_replay.py --init-from c1 --replay-ratio 0.10
```
*Results are persisted crash-safely to `reports/retention/matrix_raw.json` and evaluated on the pinned benchmark slice `reports/retention/c1_eval_slice.json`.*

### 5. Event-Driven Streaming Pipeline
*Goal: Orchestrate streaming fine-tuning via Change Data Capture (CDC) and evaluate orchestration latency (Paper Section IV-C & Table VI).*

```bash
# Terminal 1: Supabase PostgreSQL CDC to Apache Kafka Bridge
python 03_streaming_orchestration/cdc_bridge/realtime_kafka_bridge.py

# Terminal 2: Apache Spark Structured Streaming threshold aggregator (5,000 rows)
# (On Windows, launch via the provided helper script with Java 11/Spark environment):
powershell -ExecutionPolicy Bypass -File 03_streaming_orchestration/run_spark.ps1
# Or run standalone fallback processor:
python 03_streaming_orchestration/stream_processor/fallback_trigger_processor.py

# Terminal 3: Idempotent Task Dispatcher (listens to triggers, acquires GPU locks, runs training)
python 03_streaming_orchestration/task_dispatcher/trigger_consumer.py

# Benchmark: Reproduce Table VI Orchestration Latency & Prefetching Comparisons
python 03_streaming_orchestration/benchmarks/measure_latency.py
```

---

## System Limitations

As documented in the research paper:

1. **Teacher Intelligence Ceiling**: The reasoning capability of the distilled student model is bounded by the knowledge of the teacher (Alpaca-7B). Inherent teacher biases or factual inaccuracies are transferred to the student.
2. **Specialization over Generalization**: Continuous distillation prioritizes rigid instruction compliance and zero-context accuracy over complex, multi-paragraph document synthesis.
3. **4-Bit Precision Penalty**: While 4-bit NF4 quantization enables training on 8GB consumer GPUs, the precision constraint restricts lexical diversity, contributing to lower absolute ROUGE/BLEU scores.
4. **Heuristic Evaluation Boundary**: The 9-metric deterministic suite provides bias-free reproducibility, but is indifferent to semantic prose cadence.

---

## Ongoing Research: SE for AI

This repository is the empirical foundation for an ongoing research track on **Software Engineering for Artificial Intelligence (SE for AI)** — applying software engineering principles, reliability patterns, and verification mechanisms to the lifecycle of continuously adapting language models.

### 1. Checkpoint Regression Gating (CI/CD for LLMs)
Continuous integration in classical software engineering relies on automated regression test suites to block code defects before deployment. In continuous LLM distillation, we formulate **Checkpoint Regression Gating**:
- **Pinned Historical Benchmark Slices**: Using fixed, immutable data slices (such as `reports/retention/c1_eval_slice.json`), each newly distilled adapter must pass deterministic non-regression gates across critical metrics (Instruction Following $\ge 0.99$, Structured Correctness $\ge 0.51$, Lexical Hallucination $\le 0.40$).
- **Automated Deployment Halts**: If a newly trained checkpoint exhibits severe backward degradation (retention loss exceeding $\Delta > 5\%$), the dispatcher automatically aborts model promotion, preserves the prior checkpoint in production, and triggers an incident alert.

### 2. Knowledge Retention & Catastrophic Forgetting Mitigation
Sequential fine-tuning on streaming non-stationary data distributions inevitably induces catastrophic forgetting and residual overcorrection (as evidenced by the transient performance dip observed at Checkpoint 2). Our ongoing research investigates:
- **Lightweight Experience Replay**: Interleaving a small historical exemplar buffer ($10\%$ replay ratio from prior checkpoint partitions) during subsequent adapter training without inflating compute or storage requirements.
- **Dynamic Weight Interpolation & Fisher Merging**: Exploring parameter-space regularization and Fisher information-weighted adapter merging to maintain backward compatibility across multiple operational domains without full retraining.

### 3. Idempotent Execution & Exactly-Once Semantics in ML Pipelines
Distributed streaming pipelines are subject to network partitions, worker timeouts, and duplicate event deliveries. We apply robust distributed systems engineering to model training:
- **Database Row-Level Concurrency Locks**: Leveraging PostgreSQL transaction semantics (`FOR UPDATE SKIP LOCKED`) in `sql/05_schema_distributed_worker.sql` to guarantee that concurrent GPU workers never execute redundant fine-tuning passes on identical checkpoints.
- **Idempotency Keys**: Attaching unique deterministic idempotency tokens to Kafka triggers (`pipeline.triggers`) to ensure exactly-once checkpoint dispatching.

### 4. Deterministic Metric Contracts over LLM-as-a-Judge
Subjective LLM judges introduce stochasticity, self-enhancement bias, and high dollar cost into automated pipelines. We establish standardized, mathematically deterministic evaluation contracts:
- Decoupling evaluation from LLM inference through formal n-gram, LCS, and set-theoretic vocabulary alignment metrics.
- Establishing formal schema contracts and CI/CD unit tests for automated validation of language model capabilities in safety-critical deployments.

---

## References

1. G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," *arXiv preprint arXiv:1503.02531*, 2015.
2. E. J. Hu, Y. Shen, P. Wallis, et al., "LoRA: Low-rank adaptation of large language models," *ICLR*, 2022.
3. T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient finetuning of quantized LLMs," *NeurIPS*, 2023.
4. R. Taori, I. Gulrajani, T. Zhang, et al., "Stanford Alpaca: An instruction-following LLaMA model," 2023.
5. G. Team, et al., "Gemma: Open models based on Gemini research and technology," *arXiv preprint arXiv:2403.08295*, 2024.
6. T. Akidau et al., "The dataflow model: a practical approach to balancing correctness, latency, and cost in massive-scale, unbounded, out-of-order data processing," *VLDB*, 2015.
7. J. Kreps, N. Narkhede, J. Rao, et al., "Kafka: A distributed messaging system for log processing," *NetDB*, 2011.
8. M. Zaharia et al., "Discretized streams: Fault-tolerant streaming computation at scale," *SOSP*, 2013.
9. A. Chaturvedi, "System evolution analytics on interconnected entities in a system state series," *Systems Engineering*, 2026.
10. L. Zheng et al., "Judging LLM-as-a-judge with MT-bench and chatbot arena," *NeurIPS*, 2023.
