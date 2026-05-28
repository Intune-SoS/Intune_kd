"""
train_c2_with_replay.py — Replay Mitigation Training & Evaluation for Checkpoint 2
==================================================================================

Mitigates catastrophic forgetting during Checkpoint 2 fine-tuning by mixing
10% of prior Checkpoint 1 training records into the Checkpoint 2 dataset (experience replay).

Setup:
------
- Base architecture: Gemma 3-1B (4-bit QLoRA, r=16, alpha=16)
- Init-from default: "c1" (initializes replay training on top of models/gemma-ckpt1-lora)
- C2 Data: All training records from modelcomp_50k where checkpoint=2 (~4,969 records)
- C1 Replay Data: 10% random sample from checkpoint=1 (~483 records)
  * Data Isolation: Strictly excludes the 100 records in reports/retention/c1_eval_slice.json
- Prompt format:
    ### Instruction:\n{input}\n\n### Context:\n{context}\n\n### Response:\n{sevenb}
    (Context omitted if empty)
- Training constraints: RTX 4070 8GB, batch size 4, gradient accumulation 4, 1 epoch
- Output adapter: models/gemma-ckpt2-replay-lora
- Evaluation: Evaluates C2-replay adapter on the pinned 100-record C1 slice,
  evaluates original C2 adapter on the same C1 slice, computes deltas for 9 metrics + overall,
  prints comparison table, and saves to reports/retention/replay_comparison.json.

Usage:
------
    python experiment/training/train_c2_with_replay.py
    python experiment/training/train_c2_with_replay.py --init-from c1  # default
    python experiment/training/train_c2_with_replay.py --dry-run      # test data prep without training
    python experiment/training/train_c2_with_replay.py --max-steps 2  # smoke test
"""

import os
import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple

# ── Windows / PyTorch optimizations & stability ──────────────────────────────
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── Path configuration ───────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

eval_dir = str(project_root / "02_distillation" / "evaluation_suite")
if eval_dir not in sys.path:
    sys.path.insert(0, eval_dir)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_c2_with_replay")

# ── Metrics imports ──────────────────────────────────────────────────────────
from importlib import import_module
_eval_metrics_mod = import_module("06_eval_metrics")
evaluate_single_output = _eval_metrics_mod.evaluate_single_output

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from src.database.supabase_client import get_supabase_client

# ── Constants & Paths ────────────────────────────────────────────────────────
BASE_MODEL = "unsloth/gemma-3-1b-it-bnb-4bit"
C1_ADAPTER_PATH = str(project_root / "models" / "gemma-ckpt1-lora")
ORIGINAL_C2_ADAPTER_PATH = str(project_root / "models" / "gemma-ckpt2-lora")
REPLAY_ADAPTER_PATH = str(project_root / "models" / "gemma-ckpt2-replay-lora")

REPORTS_DIR = project_root / "reports" / "retention"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

C1_EVAL_SLICE_FILE = REPORTS_DIR / "c1_eval_slice.json"
COMPARISON_JSON_FILE = REPORTS_DIR / "replay_comparison.json"
COMPARISON_TABLE_FILE = REPORTS_DIR / "replay_comparison_table.txt"

# Constraints: RTX 4070 8GB
BATCH_SIZE = 4
GRAD_ACCUM = 4
MAX_SEQ_LENGTH = 2048
MAX_NEW_TOKENS = 64
EVAL_BATCH_SIZE = 4
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0

METRICS = [
    "structured_correctness",
    "instruction_following",
    "coverage",
    "hallucination",
    "context_grounding",
    "conciseness",
    "rouge1",
    "rougel",
    "bleu",
]

METRIC_LABELS = {
    "structured_correctness": "Structured Correctness",
    "instruction_following":  "Instruction Following",
    "coverage":               "Coverage",
    "hallucination":          "Hallucination",
    "context_grounding":      "Context Grounding",
    "conciseness":            "Conciseness",
    "rouge1":                 "ROUGE-1",
    "rougel":                 "ROUGE-L",
    "bleu":                   "BLEU",
    "overall":                "Overall Score",
}


# ─────────────────────────────────────────────────────────────────────────────
# C1 Evaluation Slice (Pinned)
# ─────────────────────────────────────────────────────────────────────────────

def get_c1_eval_slice(client, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Returns the pinned C1 evaluation slice.
    Reads from reports/retention/c1_eval_slice.json if it exists.
    Otherwise fetches from Supabase, writes to file, and returns it.
    """
    if C1_EVAL_SLICE_FILE.exists():
        log.info(f"Loading pinned C1 evaluation slice from: {C1_EVAL_SLICE_FILE}")
        with open(C1_EVAL_SLICE_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
        log.info(f"  Loaded {len(records)} records from pinned C1 eval file")
        return records[:limit]

    log.info(f"Pinning C1 evaluation slice to: {C1_EVAL_SLICE_FILE}")
    id_resp = (
        client.table("modelcomp_50k")
        .select("id")
        .eq("checkpoint", 1)
        .order("id", desc=False)
        .limit(limit)
        .execute()
    )
    ids = [row["id"] for row in (id_resp.data or [])]
    if not ids:
        raise RuntimeError("No records found for checkpoint 1")

    hydrate_resp = (
        client.table("modelcomp_50k")
        .select("id, input, context, sevenb, checkpoint")
        .in_("id", ids)
        .execute()
    )
    records = list(hydrate_resp.data or [])
    records.sort(key=lambda r: r.get("id") or 0)

    with open(C1_EVAL_SLICE_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    log.info(f"  Successfully pinned {len(records)} records to {C1_EVAL_SLICE_FILE}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Data Fetching for Replay Training
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_records_by_checkpoint(client, ckpt_num: int) -> List[Dict[str, Any]]:
    """
    Fetch all rows for a checkpoint with pagination (1,000 per page).
    """
    log.info(f"Fetching all training records for checkpoint {ckpt_num}...")
    records: List[Dict[str, Any]] = []
    offset = 0
    page_size = 1000

    while True:
        resp = (
            client.table("modelcomp_50k")
            .select("id, input, context, sevenb, checkpoint")
            .eq("checkpoint", ckpt_num)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = resp.data or []
        if not batch:
            break
        records.extend(batch)
        offset += page_size
        if len(batch) < page_size:
            break

    log.info(f"  Checkpoint {ckpt_num}: total fetched {len(records)} records")
    return records


def prepare_replay_dataset(
    c1_eval_ids: Set[int],
    c1_sample_pct: float = 0.10,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Prepares the mixed dataset:
    1. All C2 records with valid teacher output (checkpoint=2)
    2. 10% random sample of C1 records (checkpoint=1), strictly excluding c1_eval_ids
    3. Shuffled combination
    """
    client = get_supabase_client()

    # 1. Fetch C2 records
    c2_all = fetch_all_records_by_checkpoint(client, ckpt_num=2)
    c2_valid = [r for r in c2_all if (r.get("sevenb") or "").strip() and (r.get("input") or "").strip()]
    log.info(f"C2 valid training records: {len(c2_valid)} / {len(c2_all)}")

    # 2. Fetch C1 records & isolate evaluation slice
    c1_all = fetch_all_records_by_checkpoint(client, ckpt_num=1)
    c1_train_pool = [
        r for r in c1_all
        if r["id"] not in c1_eval_ids
        and (r.get("sevenb") or "").strip()
        and (r.get("input") or "").strip()
    ]
    log.info(f"C1 available for replay (excluding {len(c1_eval_ids)} eval IDs): {len(c1_train_pool)} / {len(c1_all)}")

    # Draw 10% random sample
    rng = random.Random(seed)
    sample_size = max(1, int(len(c1_train_pool) * c1_sample_pct))
    c1_replay_sample = rng.sample(c1_train_pool, sample_size)
    log.info(f"C1 replay sample drawn: {len(c1_replay_sample)} records ({c1_sample_pct*100:.1f}%)")

    # 3. Mix & Shuffle
    mixed_data = c2_valid + c1_replay_sample
    rng.shuffle(mixed_data)
    log.info(f"Total mixed dataset ready: {len(mixed_data)} records ({len(c2_valid)} C2 + {len(c1_replay_sample)} C1 replay)")

    return mixed_data


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Formatting & Dataset Creation
# ─────────────────────────────────────────────────────────────────────────────

def format_training_sample(record: Dict[str, Any]) -> str:
    """
    Format training sample:
    ### Instruction:\n{input}\n\n### Context:\n{context}\n\n### Response:\n{sevenb}
    (Context block omitted if empty)
    """
    instruction = (record.get("input") or "").strip()
    context = (record.get("context") or "").strip()
    response = (record.get("sevenb") or "").strip()

    if context:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Context:\n{context}\n\n"
            f"### Response:\n{response}"
        )
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{response}"
    )


def build_prompt_eval(record: Dict[str, Any]) -> str:
    """Evaluation prompt template (instruction + optional context + response prefix)."""
    instruction = (record.get("input") or "").strip()
    context = (record.get("context") or "").strip()

    if context:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Context:\n{context}\n\n"
            f"### Response:\n"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


# ─────────────────────────────────────────────────────────────────────────────
# Training Routine (Unsloth + SFTTrainer)
# ─────────────────────────────────────────────────────────────────────────────

def train_c2_replay(
    mixed_records: List[Dict[str, Any]],
    init_from: str = "c1",
    output_dir: str = REPLAY_ADAPTER_PATH,
    batch_size: int = BATCH_SIZE,
    grad_accum: int = GRAD_ACCUM,
    epochs: int = 1,
    learning_rate: float = 2e-4,
    max_steps: int = None,
    seed: int = 42,
) -> str:
    """
    Fine-tunes Gemma 3-1B on mixed data with LoRA r=16 alpha=16.
    By default (--init-from c1), loads C1 adapter weights as starting point.
    """
    import torch
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    try:
        import torch._dynamo as _dynamo
        _dynamo.config.disable = True
        _dynamo.config.suppress_errors = True
    except Exception:
        pass

    log.info("=" * 65)
    log.info(f"STARTING REPLAY TRAINING: Checkpoint 2 + 10% C1 Replay")
    log.info(f"Initialization:  {init_from}")
    log.info(f"Training samples: {len(mixed_records)}")
    log.info(f"Batch size:      {batch_size} (effective: {batch_size * grad_accum})")
    log.info(f"Output adapter:  {output_dir}")
    log.info("=" * 65)

    _lora_kwargs = dict(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    # 1. Model Initialization
    if init_from.lower() == "c1":
        if not Path(C1_ADAPTER_PATH).exists():
            raise FileNotFoundError(f"C1 adapter not found at {C1_ADAPTER_PATH}")
        log.info(f"Loading C1 adapter from {C1_ADAPTER_PATH}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=C1_ADAPTER_PATH,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        # Ensure PEFT LoRA parameters are trainable
        model = FastLanguageModel.get_peft_model(model, **_lora_kwargs)
    else:
        log.info(f"Loading base model {BASE_MODEL}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(model, **_lora_kwargs)

    # 2. Format & Pre-tokenize Dataset (single worker to avoid Windows multiprocessing crash)
    log.info("Formatting dataset...")
    formatted_texts = [{"text": format_training_sample(r)} for r in mixed_records]
    dataset = Dataset.from_list(formatted_texts)

    log.info("Pre-tokenizing dataset (num_proc=1)...")
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, num_proc=1, remove_columns=["text"])
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # 3. Trainer Configuration
    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps else -1,
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=25,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=seed,
        save_strategy="no",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=1,
        packing=False,
        args=training_args,
    )

    # 4. Train
    t0 = time.time()
    trainer.train()
    elapsed = (time.time() - t0) / 60
    log.info(f"Training completed in {elapsed:.2f} minutes")

    # 5. Save model and tokenizer
    log.info(f"Saving fine-tuned adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info("Adapter and tokenizer saved successfully")

    # 6. Purge GPU memory
    del model, tokenizer, trainer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    log.info("Training GPU memory cleared")

    return output_dir


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation on C1 Slice
# ─────────────────────────────────────────────────────────────────────────────

def calculate_metrics_dict(
    prediction: str,
    reference: str,
    instruction: str = "",
    context: str = "",
) -> Dict[str, float]:
    """Calculate 9 paper metrics + overall score."""
    _zero = {m: 0.0 for m in METRICS}
    _zero["overall"] = 0.0
    if not prediction or not reference:
        return _zero

    eval_result = evaluate_single_output(
        instruction=instruction,
        student_output=prediction,
        teacher_output=reference,
        context=context,
        task_label="general_qa",
    )

    rouge_obj = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = rouge_obj.score(reference, prediction)
    smooth = SmoothingFunction().method1
    try:
        bleu = sentence_bleu([reference.split()], prediction.split(), smoothing_function=smooth)
    except Exception:
        bleu = 0.0

    return {
        "overall":                eval_result.get("overall_score", 0.0),
        "structured_correctness": eval_result.get("structured_correctness", 0.0),
        "instruction_following":  eval_result.get("instruction_following", 0.0),
        "coverage":               eval_result.get("coverage", 0.0),
        "hallucination":          eval_result.get("hallucination", 0.0),
        "context_grounding":      eval_result.get("context_grounding", 0.0),
        "conciseness":            eval_result.get("conciseness", 0.0),
        "rouge1":                 rouge_scores["rouge1"].fmeasure,
        "rougel":                 rouge_scores["rougeL"].fmeasure,
        "bleu":                   bleu,
    }


def evaluate_adapter_on_c1_slice(
    adapter_name: str,
    adapter_path: str,
    c1_records: List[Dict[str, Any]],
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Evaluates an adapter on the 100-record C1 slice using plain transformers + PEFT.
    Returns (mean_metrics, record_evaluations).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    log.info(f"Evaluating {adapter_name} on C1 slice ({len(c1_records)} records)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True, fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.generation_config.do_sample = False
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Batched inference
    saved_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device

    generated_texts: List[str] = []
    try:
        for i in range(0, len(c1_records), EVAL_BATCH_SIZE):
            batch = c1_records[i : i + EVAL_BATCH_SIZE]
            prompts = [build_prompt_eval(r) for r in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            input_len = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            for j in range(len(batch)):
                new_tokens = out[j][input_len:]
                reply = tokenizer.decode(new_tokens, skip_special_tokens=True)
                reply = reply.split("### Instruction:")[0].split("### Response:")[-1].strip()
                generated_texts.append(reply)
    finally:
        tokenizer.padding_side = saved_side

    # Score each record
    eval_list: List[Dict[str, Any]] = []
    for rec, gen in zip(c1_records, generated_texts):
        scores = calculate_metrics_dict(
            prediction=gen,
            reference=(rec.get("sevenb") or "").strip(),
            instruction=(rec.get("input") or "").strip(),
            context=(rec.get("context") or "").strip(),
        )
        eval_list.append({
            "id": rec["id"],
            "generated": gen[:300],
            "metrics": scores,
        })

    # Mean scores
    all_keys = METRICS + ["overall"]
    mean_scores = {
        m: round(sum(item["metrics"][m] for item in eval_list) / len(eval_list), 4)
        for m in all_keys
    }

    # Free memory
    del model, base_model, tokenizer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return mean_scores, eval_list


# ─────────────────────────────────────────────────────────────────────────────
# Comparison & Reporting
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_table(
    c2_orig_scores: Dict[str, float],
    c2_replay_scores: Dict[str, float],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Builds the comparison table between C2-original and C2-replay on the C1 slice.
    """
    lines = [
        "=" * 85,
        "REPLAY MITIGATION COMPARISON (C1 Evaluation Slice, N=100)",
        "=" * 85,
        f"{'Metric':<28}{'C2-Original':>14}{'C2-Replay':>14}{'Delta':>14}{'Status':>14}",
        "-" * 85,
    ]

    all_keys = METRICS + ["overall"]
    rows_data: List[Dict[str, Any]] = []

    for m in all_keys:
        orig_val = c2_orig_scores.get(m, 0.0)
        replay_val = c2_replay_scores.get(m, 0.0)
        delta = replay_val - orig_val

        # Status determination: for hallucination, lower score is better!
        if m == "hallucination":
            if delta < -0.0005:
                status = "improved"
            elif delta > 0.0005:
                status = "regressed"
            else:
                status = "unchanged"
        else:
            if delta > 0.0005:
                status = "improved"
            elif delta < -0.0005:
                status = "regressed"
            else:
                status = "unchanged"

        label = METRIC_LABELS.get(m, m)
        if m == "overall":
            lines.append("-" * 85)
            label = "Overall Score"

        lines.append(f"{label:<28}{orig_val:>14.4f}{replay_val:>14.4f}{delta:>+14.4f}{status:>14}")
        rows_data.append({
            "metric": m,
            "metric_label": label,
            "c2_original": orig_val,
            "c2_replay": replay_val,
            "delta": round(delta, 4),
            "status": status,
        })

    lines.append("=" * 85)
    lines.append("")
    lines.append("Note: For Hallucination, lower is better (negative delta indicates improvement).")
    table_str = "\n".join(lines)
    return table_str, rows_data


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train C2 with Experience Replay & Evaluate Retention")
    parser.add_argument("--init-from", type=str, default="c1", choices=["c1", "base"],
                        help="Model initialization: 'c1' (default: train on top of C1 adapter) or 'base' (base Gemma)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs (default: 1)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Per device batch size (default: 4)")
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM, help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--replay-pct", type=float, default=0.10, help="Fraction of C1 records for replay (default: 0.10)")
    parser.add_argument("--dry-run", action="store_true", help="Prepare dataset and pin eval slice without running training")
    parser.add_argument("--max-steps", type=int, default=None, help="Max training steps (useful for smoke test)")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and run evaluation only (requires existing replay adapter)")
    args = parser.parse_args()

    log.info("=" * 65)
    log.info("EXPERIMENT 2: Replay Mitigation (C2 + 10% C1)")
    log.info(f"Init from:       {args.init_from}")
    log.info(f"Replay fraction: {args.replay_pct * 100:.1f}%")
    log.info("=" * 65)

    client = get_supabase_client()

    # Step 1: Ensure C1 evaluation slice is pinned
    c1_eval_records = get_c1_eval_slice(client)
    eval_ids = set(r["id"] for r in c1_eval_records)
    log.info(f"Pinned C1 evaluation slice: {len(c1_eval_records)} records (IDs: {min(eval_ids)}..{max(eval_ids)})")

    # Step 2: Prepare mixed dataset (strictly excluding eval_ids)
    mixed_data = prepare_replay_dataset(c1_eval_ids=eval_ids, c1_sample_pct=args.replay_pct, seed=42)

    if args.dry_run:
        log.info("[DRY RUN] Dataset prepared and eval slice verified. Exiting without training.")
        return

    # Step 3: Train C2 with Replay
    if not args.skip_train:
        train_c2_replay(
            mixed_records=mixed_data,
            init_from=args.init_from,
            output_dir=REPLAY_ADAPTER_PATH,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            epochs=args.epochs,
            max_steps=args.max_steps,
        )
    else:
        log.info(f"--skip-train specified: using existing adapter at {REPLAY_ADAPTER_PATH}")

    # Step 4: Evaluate C2-replay on C1 slice
    log.info("")
    log.info("Step 4: Evaluating C2-Replay on C1 evaluation slice...")
    c2_replay_scores, replay_record_evals = evaluate_adapter_on_c1_slice(
        adapter_name="C2-Replay",
        adapter_path=REPLAY_ADAPTER_PATH,
        c1_records=c1_eval_records,
    )

    # Step 5: Evaluate Original C2 on same C1 slice
    log.info("")
    log.info("Step 5: Evaluating Original C2 on C1 evaluation slice...")
    c2_orig_scores, orig_record_evals = evaluate_adapter_on_c1_slice(
        adapter_name="C2-Original",
        adapter_path=ORIGINAL_C2_ADAPTER_PATH,
        c1_records=c1_eval_records,
    )

    # Step 6: Comparison Table & Report
    table_str, comparison_rows = build_comparison_table(c2_orig_scores, c2_replay_scores)
    print("\n" + table_str + "\n")

    report_payload = {
        "experiment": "Replay Mitigation (Checkpoint 2 + 10% Checkpoint 1)",
        "init_from": args.init_from,
        "replay_percentage": args.replay_pct,
        "c1_eval_slice_file": str(C1_EVAL_SLICE_FILE),
        "c1_eval_record_count": len(c1_eval_records),
        "comparison_table": comparison_rows,
        "c2_original_mean_scores": c2_orig_scores,
        "c2_replay_mean_scores": c2_replay_scores,
    }

    with open(COMPARISON_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    with open(COMPARISON_TABLE_FILE, "w", encoding="utf-8") as f:
        f.write(table_str + "\n")

    log.info(f"Comparison report saved to: {COMPARISON_JSON_FILE}")
    log.info(f"Comparison table saved to:  {COMPARISON_TABLE_FILE}")


if __name__ == "__main__":
    main()
