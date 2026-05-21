"""
eval_retention_matrix.py — Cross-Checkpoint Retention Matrix Evaluation
========================================================================

Evaluates catastrophic forgetting across sequential checkpoints by measuring
the performance of each adapter (C1, C2, C3, C4) on data slices from ALL
checkpoints (C1, C2, C3, C4).

Setup:
------
- Base model: unsloth/gemma-3-1b-it-bnb-4bit (4-bit BitsAndBytes, bfloat16, eager attention)
- Adapters evaluated:
    * C1: models/gemma-ckpt1-lora
    * C2: models/gemma-ckpt2-lora
    * C3: models/gemma-ckpt3-lora
    * C4: models/gemma-ckpt4-lora
- Data: 100 records per checkpoint slice (ordered by id ascending).
  The C1 slice is pinned to reports/retention/c1_eval_slice.json so it is
  strictly identical across both this experiment and train_c2_with_replay.py.
- 9 Paper Metrics + Overall from 06_eval_metrics.py:
    structured_correctness, instruction_following, coverage,
    hallucination, context_grounding, conciseness, rouge1, rougel, bleu.
- Crash-safe: results saved to reports/retention/matrix_raw.json after each adapter.
- Output: 4x4 table printed at the end (rows = adapter used, columns = data slice).

Usage:
------
    python experiment/evaluation/eval_retention_matrix.py
    python experiment/evaluation/eval_retention_matrix.py --limit 2  # smoke test
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# ── Windows / PyTorch optimizations & stability ──────────────────────────────
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
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
log = logging.getLogger("eval_retention_matrix")

# ── Metrics imports ──────────────────────────────────────────────────────────
from importlib import import_module
_eval_metrics_mod = import_module("06_eval_metrics")
evaluate_single_output = _eval_metrics_mod.evaluate_single_output

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from src.database.supabase_client import get_supabase_client

# ── Constants ────────────────────────────────────────────────────────────────
BASE_MODEL = "unsloth/gemma-3-1b-it-bnb-4bit"

CHECKPOINTS = {
    "C1": str(project_root / "models" / "gemma-ckpt1-lora"),
    "C2": str(project_root / "models" / "gemma-ckpt2-lora"),
    "C3": str(project_root / "models" / "gemma-ckpt3-lora"),
    "C4": str(project_root / "models" / "gemma-ckpt4-lora"),
}

REPORTS_DIR = project_root / "reports" / "retention"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

C1_EVAL_SLICE_FILE = REPORTS_DIR / "c1_eval_slice.json"
RAW_RESULTS_FILE = REPORTS_DIR / "matrix_raw.json"
SUMMARY_RESULTS_FILE = REPORTS_DIR / "matrix_summary.json"
TABLE_OUTPUT_FILE = REPORTS_DIR / "matrix_table.txt"

# Evaluation constraints: RTX 4070 8GB
BATCH_SIZE = 4
MAX_NEW_TOKENS = 64
SLICE_SIZE = 100

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
# Metric Calculation
# ─────────────────────────────────────────────────────────────────────────────

def calculate_metrics(
    prediction: str,
    reference: str,
    instruction: str = "",
    context: str = "",
    task_label: str = "general_qa",
) -> Dict[str, Any]:
    """
    Run 06_eval_metrics evaluation + ROUGE-1/L + BLEU.
    Returns dictionary containing all 9 paper metrics plus 'overall'.
    """
    _zero = {
        "overall": 0.0,
        "structured_correctness": 0.0,
        "instruction_following": 0.0,
        "coverage": 0.0,
        "hallucination": 0.0,
        "context_grounding": 0.0,
        "conciseness": 0.0,
        "rouge1": 0.0,
        "rougel": 0.0,
        "bleu": 0.0,
        "details": {},
    }
    if not prediction or not reference:
        return _zero

    eval_result = evaluate_single_output(
        instruction=instruction,
        student_output=prediction,
        teacher_output=reference,
        context=context,
        task_label=task_label,
    )

    rouge_obj = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = rouge_obj.score(reference, prediction)
    smooth = SmoothingFunction().method1
    try:
        bleu = sentence_bleu(
            [reference.split()], prediction.split(), smoothing_function=smooth
        )
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
        "details":                eval_result.get("details", {}),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data Fetching & Slice Pinning
# ─────────────────────────────────────────────────────────────────────────────

def fetch_checkpoint_slice(client, ckpt_num: int, limit: int = SLICE_SIZE) -> List[Dict[str, Any]]:
    """
    Fetch slice for a checkpoint from modelcomp_50k ordered by id ascending.
    Uses two-step query (cheap index pull then hydration) to avoid statement timeouts.
    """
    log.info(f"Fetching slice for checkpoint {ckpt_num} (limit={limit})...")
    # Step 1: cheap id-only query using index
    id_resp = (
        client.table("modelcomp_50k")
        .select("id")
        .eq("checkpoint", ckpt_num)
        .order("id", desc=False)
        .limit(limit)
        .execute()
    )
    ids = [row["id"] for row in (id_resp.data or [])]
    if not ids:
        raise RuntimeError(f"No records found for checkpoint {ckpt_num}")

    # Step 2: hydrate full columns by id in (...)
    hydrate_resp = (
        client.table("modelcomp_50k")
        .select("id, input, context, sevenb, checkpoint")
        .in_("id", ids)
        .execute()
    )
    rows = list(hydrate_resp.data or [])
    # Strictly sort ascending by id
    rows.sort(key=lambda r: r.get("id") or 0)
    log.info(f"  Checkpoint {ckpt_num}: fetched {len(rows)} rows (id range {rows[0]['id']}..{rows[-1]['id']})")
    return rows


def get_c1_eval_slice(client, limit: int = SLICE_SIZE) -> List[Dict[str, Any]]:
    """
    Returns the pinned C1 evaluation slice.
    If reports/retention/c1_eval_slice.json exists, reads from disk.
    Otherwise fetches it from Supabase, writes to disk, and returns it.
    """
    if C1_EVAL_SLICE_FILE.exists():
        log.info(f"Loading pinned C1 evaluation slice from: {C1_EVAL_SLICE_FILE}")
        with open(C1_EVAL_SLICE_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
        log.info(f"  Loaded {len(records)} records from pinned C1 eval file")
        return records[:limit]

    log.info(f"Pinning C1 evaluation slice to: {C1_EVAL_SLICE_FILE}")
    records = fetch_checkpoint_slice(client, ckpt_num=1, limit=limit)
    with open(C1_EVAL_SLICE_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    log.info(f"  Successfully pinned {len(records)} records to {C1_EVAL_SLICE_FILE}")
    return records


def load_all_eval_slices(limit: int = SLICE_SIZE) -> Dict[str, List[Dict[str, Any]]]:
    """
    Loads all four evaluation slices (C1, C2, C3, C4), 100 records each.
    """
    client = get_supabase_client()
    slices: Dict[str, List[Dict[str, Any]]] = {}

    # Slice C1 (pinned)
    slices["C1"] = get_c1_eval_slice(client, limit=limit)

    # Slices C2, C3, C4
    for ckpt in [2, 3, 4]:
        name = f"C{ckpt}"
        slices[name] = fetch_checkpoint_slice(client, ckpt_num=ckpt, limit=limit)

    total_records = sum(len(s) for s in slices.values())
    log.info(f"All eval slices loaded: {total_records} records total across {list(slices.keys())}")
    return slices


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Formatting & Batched Inference
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt_training_format(record: Dict[str, Any]) -> str:
    """
    Format prompt matching training:
    ### Instruction:\n{input}\n\n### Context:\n{context}\n\n### Response:\n
    (Context omitted if empty)
    """
    instruction = (record.get("input") or "").strip()
    context = (record.get("context") or "").strip()

    if context:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Context:\n{context}\n\n"
            f"### Response:\n"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def generate_batch(
    model, tokenizer, records: List[Dict[str, Any]], batch_size: int = BATCH_SIZE
) -> List[str]:
    """
    Run greedy batched generation with left padding.
    """
    import torch

    all_outputs: List[str] = []
    device = next(model.parameters()).device
    saved_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            prompts = [build_prompt_training_format(r) for r in batch]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(device)

            input_len = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            for j in range(len(batch)):
                new_tokens = output_ids[j][input_len:]
                reply = tokenizer.decode(new_tokens, skip_special_tokens=True)
                # Strip instruction/response prefixes
                reply = reply.split("### Instruction:")[0]
                reply = reply.split("### Response:")[-1]
                all_outputs.append(reply.strip())
    finally:
        tokenizer.padding_side = saved_side

    return all_outputs


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading & GPU Memory Management
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(adapter_path: str):
    """
    Load base model in 4-bit and attach PEFT LoRA adapter.
    Uses eager attention to avoid Windows CUDA SDPA stalls.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    if torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except AttributeError:
            pass

    log.info(f"Loading base model: {BASE_MODEL}")
    log.info(f"Attaching LoRA adapter: {adapter_path}")

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
        tokenizer = AutoTokenizer.from_pretrained(
            adapter_path, trust_remote_code=True, fix_mistral_regex=True
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    log.info("Model and tokenizer successfully loaded")
    return model, tokenizer


def unload_model(model, tokenizer) -> None:
    """Free GPU memory completely between adapter evaluations."""
    import gc
    import torch

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    log.info("GPU memory cleared successfully")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Runner
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_adapter_on_slices(
    adapter_name: str,
    adapter_path: str,
    slices: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Evaluates one adapter across all four checkpoint slices.
    """
    log.info("=" * 65)
    log.info(f"STARTING EVALUATION: Adapter {adapter_name}")
    log.info(f"Path: {adapter_path}")
    log.info("=" * 65)

    model, tokenizer = load_model_and_tokenizer(adapter_path)
    adapter_results: List[Dict[str, Any]] = []

    # Warmup
    import torch
    warmup_inputs = tokenizer(["Hello"], return_tensors="pt").to(next(model.parameters()).device)
    with torch.inference_mode():
        model.generate(**warmup_inputs, max_new_tokens=4, do_sample=False)
    del warmup_inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for slice_name, slice_records in slices.items():
        log.info(f"Evaluating {adapter_name} on slice {slice_name} ({len(slice_records)} records)...")
        t0 = time.perf_counter()
        generated_texts = generate_batch(model, tokenizer, slice_records, batch_size=BATCH_SIZE)
        elapsed = time.perf_counter() - t0
        log.info(f"  Inference done in {elapsed:.2f}s ({elapsed / max(1, len(slice_records)):.2f}s/rec)")

        for record, gen_text in zip(slice_records, generated_texts):
            reference = (record.get("sevenb") or "").strip()
            instruction = (record.get("input") or "").strip()
            context = (record.get("context") or "").strip()

            scores = calculate_metrics(
                prediction=gen_text,
                reference=reference,
                instruction=instruction,
                context=context,
                task_label="general_qa",
            )

            adapter_results.append({
                "adapter":      adapter_name,
                "eval_slice":   slice_name,
                "id":           record.get("id"),
                "checkpoint":   record.get("checkpoint"),
                "has_context":  bool(context),
                "input":        instruction[:200],
                "generated":    gen_text[:400],
                "reference":    reference[:200],
                "metrics":      {k: round(scores[k], 4) for k in METRICS},
                "overall":      round(scores["overall"], 4),
            })

    unload_model(model, tokenizer)
    return adapter_results


def build_and_save_matrix(raw_results: List[Dict[str, Any]]) -> str:
    """
    Computes the 4x4 matrix of overall scores and builds a formatted table string.
    """
    adapters = ["C1", "C2", "C3", "C4"]
    slices = ["C1", "C2", "C3", "C4"]

    # Compute means: matrix[adapter][slice] = overall_score
    matrix: Dict[str, Dict[str, float]] = {a: {} for a in adapters}
    matrix_per_metric: Dict[str, Dict[str, Dict[str, float]]] = {a: {s: {} for s in slices} for a in adapters}

    for a in adapters:
        for s in slices:
            recs = [r for r in raw_results if r["adapter"] == a and r["eval_slice"] == s]
            if recs:
                mean_overall = sum(r["overall"] for r in recs) / len(recs)
                matrix[a][s] = round(mean_overall, 4)
                for m in METRICS:
                    mean_m = sum(r["metrics"][m] for r in recs) / len(recs)
                    matrix_per_metric[a][s][m] = round(mean_m, 4)
            else:
                matrix[a][s] = 0.0

    # Build ASCII table
    col_w = 14
    header = f"{'Adapter':<12}" + "".join(f"{('Slice ' + s):>{col_w}}" for s in slices)
    sep = "=" * len(header)
    lines = [
        sep,
        "CROSS-CHECKPOINT RETENTION MATRIX (Overall Score)",
        sep,
        header,
        "-" * len(header),
    ]

    for a in adapters:
        row = f"{a:<12}" + "".join(f"{matrix[a].get(s, 0.0):>{col_w}.4f}" for s in slices)
        lines.append(row)

    lines.append(sep)
    lines.append("")
    lines.append("Retention Summary Notes:")
    lines.append("  - Diagonal (Ci on Slice Ci): Performance on the native training distribution.")
    lines.append("  - Below diagonal (C(i+k) on Slice Ci): Backward retention / catastrophic forgetting.")
    lines.append("  - Above diagonal (Ci on Slice C(i+k)): Forward transfer / zero-shot generalisation.")
    table_str = "\n".join(lines)

    # Save summary JSON
    summary_data = {
        "matrix_overall": matrix,
        "matrix_per_metric": matrix_per_metric,
        "adapters": adapters,
        "slices": slices,
    }
    with open(SUMMARY_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    with open(TABLE_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(table_str + "\n")

    return table_str


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cross-Checkpoint Retention Matrix Evaluation")
    parser.add_argument("--limit", type=int, default=SLICE_SIZE, help="Number of records per checkpoint slice (default: 100)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing matrix_raw.json if present")
    args = parser.parse_args()

    log.info("=" * 65)
    log.info("EXPERIMENT 1: Cross-Checkpoint Retention Matrix")
    log.info(f"Limit per slice: {args.limit} records")
    log.info(f"Reports directory: {REPORTS_DIR}")
    log.info("=" * 65)

    # Step 1: Load all 4 evaluation slices (pins C1 to c1_eval_slice.json)
    slices = load_all_eval_slices(limit=args.limit)

    # Step 2: Crash-safe results accumulator
    raw_results: List[Dict[str, Any]] = []
    if args.resume and RAW_RESULTS_FILE.exists():
        try:
            with open(RAW_RESULTS_FILE, "r", encoding="utf-8") as f:
                raw_results = json.load(f)
            log.info(f"Resumed from {RAW_RESULTS_FILE} with {len(raw_results)} prior evaluations")
        except Exception as e:
            log.warning(f"Failed to read existing raw results: {e}. Starting fresh.")
            raw_results = []

    completed_adapters = set(r["adapter"] for r in raw_results)

    # Step 3: Loop over adapters C1..C4
    for adapter_name, adapter_path in CHECKPOINTS.items():
        if adapter_name in completed_adapters and args.resume:
            log.info(f"Adapter {adapter_name} already evaluated — skipping")
            continue

        if not Path(adapter_path).exists():
            log.error(f"Adapter directory not found: {adapter_path} — skipping")
            continue

        results = evaluate_adapter_on_slices(adapter_name, adapter_path, slices)
        raw_results.extend(results)

        # Crash-safe save after each adapter
        with open(RAW_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(raw_results, f, indent=2, ensure_ascii=False)
        log.info(f"Crash-safe checkpoint saved: {RAW_RESULTS_FILE} ({len(raw_results)} total evaluations)")

    # Step 4: Build 4x4 matrix and print table
    log.info("")
    table_str = build_and_save_matrix(raw_results)
    print("\n" + table_str + "\n")
    log.info(f"Raw results saved to:     {RAW_RESULTS_FILE}")
    log.info(f"Summary results saved to: {SUMMARY_RESULTS_FILE}")
    log.info(f"Table saved to:           {TABLE_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
