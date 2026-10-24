# INTUNE System-of-Systems: Event-Driven Incremental Knowledge Distillation for Compact LLMs

Code, data, and reports for the INTUNE paper (IEEE DSAA 2026). INTUNE triggers LoRA fine-tuning of a compact student model (Gemma-3-1B) when new data arrives, using change data capture (CDC), Kafka, and Spark Structured Streaming, and trains in checkpoints instead of one batch job.

## Repository layout
| Folder | Purpose |
|---|---|
| `01_teacher_selection/` | Data preparation and teacher comparison (Alpaca-7B vs. GPT-OSS-20B) |
| `02_distillation/batch_baseline/` | Monolithic batch training baseline |
| `02_distillation/incremental_pipeline/` | Checkpoint-based incremental training (adapters C1–C4) |
| `02_distillation/evaluation_suite/` | Deterministic rule-based metric suite (`06_eval_metrics.py`) and checkpoint evaluation |
| `03_streaming_orchestration/` | CDC bridge, Kafka trigger consumer, Spark stream processor, latency benchmark |
| `app/` | Inference service |
| `src/` | Shared database, metric, and training helpers |
| `sql/` | Database schemas, triggers, and worker queue |
| `models/` | LoRA adapter configurations (weights are not stored in git) |
| `reports/` | Evaluation reports; paper tables in `reports/tables/paper1_dsaa/` |
| `data/experiment/` | Prepared datasets (derived from Stanford Alpaca) |

## Setup
```bash
python -m venv venv
venv\Scripts\activate            # Windows  (source venv/bin/activate on Linux/macOS)
pip install -r requirements.txt  # requirements-mac.txt on Apple Silicon
copy config\.env.example .env    # then fill in your own database URL and key
```

## Running
- Incremental training of checkpoint k: `python 02_distillation/incremental_pipeline/12_train_incremental.py --checkpoint k --run-all`
- Batch baseline: `python 02_distillation/batch_baseline/13_train_batch.py`
- Streaming pipeline: start Kafka, then `run_spark.ps1`, then `python 03_streaming_orchestration/task_dispatcher/trigger_consumer.py`
- Latency benchmark: `python 03_streaming_orchestration/benchmarks/measure_latency.py`

## Notes
- `.env` holds credentials and is never committed.
- Model weights and GGUF exports are excluded from git; only adapter configurations are tracked.
