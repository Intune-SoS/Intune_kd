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
- Crash-safe & Auto-Resumable: results saved per slice (every 100 records).
- Dual Live Stopwatch: tracks current slice time, speed (s/rec), and total overall time left.
- Dual Table Outputs:
    1. Cross-Checkpoint Retention Matrix (Overall Score)
    2. Cross-Checkpoint Stopwatch Timing Matrix (Inference Latency in seconds & s/rec)

Usage:
------
    python 02_distillation/retention_and_replay/eval_retention_matrix.py
    python 02_distillation/retention_and_replay/eval_retention_matrix.py --fresh   # start from scratch
    python 02_distillation/retention_and_replay/eval_retention_matrix.py --limit 5 # quick smoke test
"""

import os
import sys
import json
import time
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional

# ── Suppress noisy warnings ──────────────────────────────────────────────────
warnings.filterwarnings("ignore", message=".*max_new_tokens.*")
warnings.filterwarnings("ignore", message=".*quantization_config.*")
warnings.filterwarnings("ignore", message=".*Triton.*")

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
TIMING_OUTPUT_FILE = REPORTS_DIR / "matrix_timing.txt"

# Evaluation constraints: batch size 4 for memory safety
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
# Stopwatch & Progress Tracker
# ─────────────────────────────────────────────────────────────────────────────

def format_duration(seconds: float) -> str:
    """Format duration in seconds to mm:ss or hh:mm:ss."""
    s = max(0, int(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h:02d}h {m:02d}m {sec:02d}s"
    return f"{m:02d}m {sec:02d}s"


class RetentionStopwatch:
    """
    Dual live stopwatch tracking:
    1. Active slice timer (elapsed, ETA, speed in s/record)
    2. Overall experiment job timer (slices completed, total elapsed, total time left)
    """

    def __init__(self, total_slices: int = 16, slice_size: int = SLICE_SIZE):
        self.total_slices = total_slices
        self.slice_size = slice_size
        self.global_start_time = time.perf_counter()
        self.slice_start_time = time.perf_counter()
        self.completed_slices = 0
        self.current_adapter = "C1"
        self.current_slice = "C1"
        self.slice_timings: Dict[str, Dict[str, float]] = {}

    def start_slice(self, adapter: str, slice_name: str, already_done_slices: int):
        self.current_adapter = adapter
        self.current_slice = slice_name
        self.completed_slices = already_done_slices
        self.slice_start_time = time.perf_counter()

    def update_batch(self, batch_idx: int, total_batches: int, processed_records: int):
        now = time.perf_counter()
        slice_elapsed = now - self.slice_start_time
        sec_per_rec = slice_elapsed / max(1, processed_records)
        remaining_in_slice = max(0, self.slice_size - processed_records)
        slice_eta = remaining_in_slice * sec_per_rec

        # Overall estimation
        current_fraction = processed_records / max(1, self.slice_size)
        effective_done = self.completed_slices + current_fraction
        global_elapsed = now - self.global_start_time

        if effective_done > 0:
            avg_sec_per_slice = global_elapsed / effective_done
            remaining_slices = max(0.0, self.total_slices - effective_done)
            overall_eta = remaining_slices * avg_sec_per_slice
        else:
            overall_eta = (self.total_slices - 1) * (self.slice_size * sec_per_rec)

        pct_slice = (processed_records / max(1, self.slice_size)) * 100
        overall_idx = min(self.total_slices, self.completed_slices + 1)
        pct_overall = (effective_done / max(1, self.total_slices)) * 100

        ticker = (
            f"\r[{self.current_adapter} | Slice {self.current_slice}] "
            f"Batch {batch_idx:02d}/{total_batches:02d} ({processed_records}/{self.slice_size}) [{pct_slice:.0f}%] | "
            f"Slice: {format_duration(slice_elapsed)} (ETA: {format_duration(slice_eta)}) | "
            f"{sec_per_rec:.2f}s/rec | "
            f"Overall: {overall_idx}/{self.total_slices} [{pct_overall:.0f}%] | "
            f"Total Left: {format_duration(overall_eta)}"
        )
        sys.stdout.write(ticker)
        sys.stdout.flush()

    def finish_slice(self) -> float:
        sys.stdout.write("\n")
        sys.stdout.flush()
        elapsed = time.perf_counter() - self.slice_start_time
        if self.current_adapter not in self.slice_timings:
            self.slice_timings[self.current_adapter] = {}
        self.slice_timings[self.current_adapter][self.current_slice] = round(elapsed, 2)
        return elapsed


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

    hydrate_resp = (
        client.table("modelcomp_50k")
        .select("id, input, context, sevenb, checkpoint")
        .in_("id", ids)
        .execute()
    )
    rows = list(hydrate_resp.data or [])
    rows.sort(key=lambda r: r.get("id") or 0)
    log.info(f"  Checkpoint {ckpt_num}: fetched {len(rows)} rows (id range {rows[0]['id']}..{rows[-1]['id']})")
    return rows


def get_c1_eval_slice(client, limit: int = SLICE_SIZE) -> List[Dict[str, Any]]:
    """
    Returns the pinned C1 evaluation slice.
    Reads from disk if reports/retention/c1_eval_slice.json exists.
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
    """Loads all four evaluation slices (C1, C2, C3, C4), limit records each."""
    client = get_supabase_client()
    slices: Dict[str, List[Dict[str, Any]]] = {}

    slices["C1"] = get_c1_eval_slice(client, limit=limit)
    for ckpt in [2, 3, 4]:
        name = f"C{ckpt}"
        slices[name] = fetch_checkpoint_slice(client, ckpt_num=ckpt, limit=limit)

    total_records = sum(len(s) for s in slices.values())
    log.info(f"All eval slices loaded: {total_records} records total across {list(slices.keys())}")
    return slices


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Formatting & Batched Inference with Live Stopwatch
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt_training_format(record: Dict[str, Any]) -> str:
    instruction = (record.get("input") or "").strip()
    context = (record.get("context") or "").strip()

    if context:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Context:\n{context}\n\n"
            f"### Response:\n"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def generate_batch_with_stopwatch(
    model,
    tokenizer,
    records: List[Dict[str, Any]],
    batch_size: int = BATCH_SIZE,
    tracker: Optional[RetentionStopwatch] = None,
) -> List[str]:
    """
    Run greedy batched generation with left padding and live stopwatch updates.
    """
    import torch

    all_outputs: List[str] = []
    device = next(model.parameters()).device
    saved_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    total_records = len(records)
    total_batches = (total_records + batch_size - 1) // batch_size

    try:
        for batch_idx, start in enumerate(range(0, total_records, batch_size), 1):
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
                reply = reply.split("### Instruction:")[0]
                reply = reply.split("### Response:")[-1]
                all_outputs.append(reply.strip())

            # Update live stopwatch after every batch
            if tracker:
                tracker.update_batch(
                    batch_idx=batch_idx,
                    total_batches=total_batches,
                    processed_records=len(all_outputs),
                )
    finally:
        tokenizer.padding_side = saved_side

    return all_outputs


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading & GPU Memory Management
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(adapter_path: str):
    """
    Load base model in 4-bit and attach PEFT LoRA adapter.
    Silences max_length warnings and uses eager attention.
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

    # Configure generation to be clean and warning-free
    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.max_length = None  # Silences warning completely

    log.info("Model and tokenizer successfully loaded")
    return model, tokenizer


def unload_model(model, tokenizer) -> None:
    """Free GPU / CPU memory completely between adapter evaluations."""
    import gc
    import torch

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    log.info("Model unloaded & memory cleared successfully")


# ─────────────────────────────────────────────────────────────────────────────
# Matrix & Stopwatch Timing Table Builders
# ─────────────────────────────────────────────────────────────────────────────

def build_and_save_matrices(
    raw_results: List[Dict[str, Any]],
    timing_data: Optional[Dict[str, Dict[str, float]]] = None,
) -> str:
    """
    Computes both:
    1. 4x4 Retention Matrix (Overall Score)
    2. 4x4 Stopwatch Timing Matrix (Seconds per slice & s/rec)
    Saves outputs to summary JSON and text tables.
    """
    adapters = ["C1", "C2", "C3", "C4"]
    slices = ["C1", "C2", "C3", "C4"]

    # Compute overall score and metric means
    matrix: Dict[str, Dict[str, float]] = {a: {} for a in adapters}
    matrix_per_metric: Dict[str, Dict[str, Dict[str, float]]] = {a: {s: {} for s in slices} for a in adapters}

    for a in adapters:
        for s in slices:
            recs = [r for r in raw_results if r.get("adapter") == a and r.get("eval_slice") == s]
            if recs:
                mean_overall = sum(r.get("overall", 0.0) for r in recs) / len(recs)
                matrix[a][s] = round(mean_overall, 4)
                for m in METRICS:
                    mean_m = sum(r.get("metrics", {}).get(m, 0.0) for r in recs) / len(recs)
                    matrix_per_metric[a][s][m] = round(mean_m, 4)
            else:
                matrix[a][s] = 0.0

    # 1. Build Score Table
    col_w = 14
    header = f"{'Adapter':<12}" + "".join(f"{('Slice ' + s):>{col_w}}" for s in slices)
    sep = "=" * len(header)
    score_lines = [
        sep,
        "CROSS-CHECKPOINT RETENTION MATRIX (Overall Score)",
        sep,
        header,
        "-" * len(header),
    ]
    for a in adapters:
        row = f"{a:<12}" + "".join(f"{matrix[a].get(s, 0.0):>{col_w}.4f}" for s in slices)
        score_lines.append(row)
    score_lines.append(sep)
    score_lines.append("")
    score_lines.append("Retention Summary Notes:")
    score_lines.append("  - Diagonal (Ci on Slice Ci): Performance on native training distribution.")
    score_lines.append("  - Below diagonal (C(i+k) on Slice Ci): Backward retention / catastrophic forgetting.")
    score_lines.append("  - Above diagonal (Ci on Slice C(i+k)): Forward transfer / zero-shot generalisation.")
    score_table_str = "\n".join(score_lines)

    # 2. Build Timing Table (if available)
    timing_lines = []
    if timing_data:
        timing_lines = [
            "",
            sep,
            "CROSS-CHECKPOINT STOPWATCH INFERENCE TIMING (Seconds per slice)",
            sep,
            header,
            "-" * len(header),
        ]
        for a in adapters:
            row_times = []
            for s in slices:
                t = timing_data.get(a, {}).get(s, 0.0)
                if t > 0:
                    row_times.append(f"{t:>{col_w-1}.1f}s")
                else:
                    row_times.append(f"{'--':>{col_w}}")
            timing_lines.append(f"{a:<12}" + "".join(row_times))
        timing_lines.append(sep)

    timing_table_str = "\n".join(timing_lines)
    full_output_str = score_table_str + "\n" + timing_table_str

    # Save to disk
    summary_data = {
        "matrix_overall": matrix,
        "matrix_per_metric": matrix_per_metric,
        "timing_data": timing_data or {},
        "adapters": adapters,
        "slices": slices,
    }
    with open(SUMMARY_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    with open(TABLE_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(full_output_str + "\n")

    return full_output_str


# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluation Loop with Granular Per-Slice Resume
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cross-Checkpoint Retention Matrix Evaluation")
    parser.add_argument("--limit", type=int, default=SLICE_SIZE, help="Number of records per checkpoint slice (default: 100)")
    parser.add_argument("--fresh", action="store_true", help="Start fresh and overwrite existing progress in matrix_raw.json")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("EXPERIMENT 1: Cross-Checkpoint Retention Matrix (with Dual Stopwatch)")
    log.info(f"Slice limit: {args.limit} records | Auto-resume: {'DISABLED (--fresh)' if args.fresh else 'ENABLED'}")
    log.info(f"Reports directory: {REPORTS_DIR}")
    log.info("=" * 70)

    # 1. Load evaluation slices (pins C1 to c1_eval_slice.json)
    slices = load_all_eval_slices(limit=args.limit)
    total_matrix_cells = len(CHECKPOINTS) * len(slices)  # 4 x 4 = 16

    # 2. Load existing results for auto-resume
    raw_results: List[Dict[str, Any]] = []
    completed_cells = set()
    timing_data: Dict[str, Dict[str, float]] = {a: {} for a in CHECKPOINTS.keys()}

    if not args.fresh and RAW_RESULTS_FILE.exists():
        try:
            with open(RAW_RESULTS_FILE, "r", encoding="utf-8") as f:
                raw_results = json.load(f)

            # A cell is only considered complete if it has at least `args.limit` evaluations
            from collections import Counter
            cell_counts = Counter((r.get("adapter"), r.get("eval_slice")) for r in raw_results)
            completed_cells = set(cell for cell, count in cell_counts.items() if count >= args.limit)

            log.info(f"Auto-resumed: found {len(completed_cells)}/{total_matrix_cells} slices already completed")
        except Exception as e:
            log.warning(f"Could not parse existing raw results ({e}). Starting fresh.")
            raw_results = []
            completed_cells = set()

    # Load existing timings if available
    if SUMMARY_RESULTS_FILE.exists():
        try:
            with open(SUMMARY_RESULTS_FILE, "r", encoding="utf-8") as f:
                saved_summary = json.load(f)
                timing_data = saved_summary.get("timing_data", timing_data)
        except Exception:
            pass

    # 3. Initialize Global Stopwatch
    tracker = RetentionStopwatch(total_slices=total_matrix_cells, slice_size=args.limit)
    completed_slices_count = len(completed_cells)

    # 4. Loop through each adapter
    for adapter_name, adapter_path in CHECKPOINTS.items():
        # Identify which slices still need to be evaluated for this adapter
        needed_slices = [
            s_name for s_name in slices.keys()
            if (adapter_name, s_name) not in completed_cells
        ]

        if not needed_slices:
            log.info(f"Adapter {adapter_name}: all slices already finished ({len(slices)}/{len(slices)}) — skipping model load")
            continue

        if not Path(adapter_path).exists():
            log.error(f"Adapter directory not found: {adapter_path} — skipping {adapter_name}")
            continue

        log.info("=" * 70)
        log.info(f"LOADING ADAPTER {adapter_name} ({len(needed_slices)} slices remaining)")
        log.info("=" * 70)

        model, tokenizer = load_model_and_tokenizer(adapter_path)

        # Warmup pass
        import torch
        warmup_inputs = tokenizer(["Hello"], return_tensors="pt").to(next(model.parameters()).device)
        with torch.inference_mode():
            model.generate(**warmup_inputs, max_new_tokens=4, do_sample=False)
        del warmup_inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Evaluate each needed slice
        for slice_name in needed_slices:
            slice_records = slices[slice_name]
            log.info(f"Starting {adapter_name} on slice {slice_name} ({len(slice_records)} records)...")

            tracker.start_slice(adapter_name, slice_name, completed_slices_count)
            generated_texts = generate_batch_with_stopwatch(
                model=model,
                tokenizer=tokenizer,
                records=slice_records,
                batch_size=BATCH_SIZE,
                tracker=tracker,
            )
            elapsed_sec = tracker.finish_slice()
            timing_data[adapter_name][slice_name] = round(elapsed_sec, 2)

            log.info(f"  Finished {adapter_name} on slice {slice_name} in {elapsed_sec:.1f}s ({elapsed_sec/len(slice_records):.2f}s/rec)")

            # Compute metrics for each record
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

                raw_results.append({
                    "adapter":     adapter_name,
                    "eval_slice":  slice_name,
                    "id":          record.get("id"),
                    "checkpoint":  record.get("checkpoint"),
                    "has_context": bool(context),
                    "input":       instruction[:200],
                    "generated":   gen_text[:400],
                    "reference":   reference[:200],
                    "metrics":     {k: round(scores[k], 4) for k in METRICS},
                    "overall":     round(scores["overall"], 4),
                })

            completed_cells.add((adapter_name, slice_name))
            completed_slices_count += 1

            # Crash-safe save immediately after EACH slice
            with open(RAW_RESULTS_FILE, "w", encoding="utf-8") as f:
                json.dump(raw_results, f, indent=2, ensure_ascii=False)

            # Update live summary table on disk after each slice
            build_and_save_matrices(raw_results, timing_data)
            log.info(f"  [Checkpoint Saved] Slice {completed_slices_count}/{total_matrix_cells} persisted to {RAW_RESULTS_FILE}")

        unload_model(model, tokenizer)

    # 5. Final Output
    log.info("")
    final_output = build_and_save_matrices(raw_results, timing_data)
    print("\n" + final_output + "\n")
    log.info(f"Raw results saved to:     {RAW_RESULTS_FILE}")
    log.info(f"Summary results saved to: {SUMMARY_RESULTS_FILE}")
    log.info(f"Table output saved to:    {TABLE_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
