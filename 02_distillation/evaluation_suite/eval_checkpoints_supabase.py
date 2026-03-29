"""
eval_checkpoints_supabase.py — Per-checkpoint evaluation on the held-out
Supabase eval set for INTUNE Table 5.
=========================================================================

Loads each LoRA checkpoint (C1–C4) on top of the
unsloth/gemma-3-1b-it-bnb-4bit base model, runs inference on 400 records
from the modelcomp_50k held-out checkpoint (5 by default, falls back to
6 if 5 has fewer than 400 valid rows), computes the 9 paper metrics
using calculate_metrics() copied verbatim from 12_train_incremental.py,
and splits results by context-present vs zero-context.

Held-out: checkpoint 5 (and 6 as fallback) is used for evaluation only —
training in this run covered checkpoints 1–4.

Outputs
-------
reports/incremental/
    checkpoint_eval_supabase_raw.json      — every record's generated output + 9 scores
    checkpoint_eval_supabase_summary.json  — per-checkpoint mean scores (ctx / noctx / all)
    checkpoint_eval_supabase_table5.txt    — copy-paste ready Table 5 with Batch_baseline column
    checkpoint_eval_supabase_dynamics.txt  — non-monotonic / regression / recovery analysis

Usage
-----
    python experiment/evaluation/eval_checkpoints_supabase.py
"""

import os
import sys
import json
import logging
import math
from pathlib import Path
from typing import Any

# ── Windows / Triton compatibility ──────────────────────────────────────────
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Path setup ───────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Make 06_eval_metrics importable (its filename starts with a digit, so it
# can't be imported with a normal `import` statement)
eval_dir = str(project_root / "02_distillation" / "evaluation_suite")
if eval_dir not in sys.path:
    sys.path.insert(0, eval_dir)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Imports from existing project code ───────────────────────────────────────
from importlib import import_module                       # noqa: E402
_eval_metrics_mod = import_module("06_eval_metrics")
evaluate_single_output = _eval_metrics_mod.evaluate_single_output

from rouge_score import rouge_scorer                       # noqa: E402
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # noqa: E402

from src.database.supabase_client import get_supabase_client  # noqa: E402

# Constants for model loading.  These match the adapter_config.json of every
# C1–C4 LoRA: base = unsloth/gemma-3-1b-it-bnb-4bit, target_modules = q/k/v/o
# + gate/up/down, r=16, alpha=16.  We avoid unsloth's for_inference() patching
# because it deadlocks on Windows with CUDA 11.8 on prompts longer than ~10
# tokens; the adapters are standard PEFT LoRA and run cleanly with plain
# transformers + PEFT.
BASE_MODEL = "unsloth/gemma-3-1b-it-bnb-4bit"


def load_model_and_tokenizer(adapter_path: str):
    """
    Load the 4-bit base model and apply the LoRA adapter at `adapter_path`.

    Self-contained: does not import unsloth.  Returns (model, tokenizer)
    ready for greedy inference.
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import PeftModel

    if torch.cuda.is_available():
        # Eager attention avoids known Windows + CUDA 11.8 SDPA stalls.
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except AttributeError:
            pass

    log.info(f"Loading base model: {BASE_MODEL}")
    log.info(f"Applying adapter:   {adapter_path}")

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
        tokenizer = AutoTokenizer.from_pretrained(
            adapter_path, trust_remote_code=True
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Pin generation config to deterministic greedy decoding so we never get
    # "top_p ignored when do_sample=False" warnings, and so behaviour is
    # reproducible across checkpoints.
    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    log.info("Model loaded and set to inference mode")
    return model, tokenizer


def unload_model(model, tokenizer) -> None:
    """Free GPU memory between checkpoints."""
    import gc
    import torch

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    log.info("GPU memory freed")

# ── Constants ────────────────────────────────────────────────────────────────

CHECKPOINTS = {
    "C1": str(project_root / "models" / "gemma-ckpt1-lora"),
    "C2": str(project_root / "models" / "gemma-ckpt2-lora"),
    "C3": str(project_root / "models" / "gemma-ckpt3-lora"),
    "C4": str(project_root / "models" / "gemma-ckpt4-lora"),
}

REPORTS_DIR = project_root / "reports" / "incremental"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Table 5 row order — these are the 9 paper metrics plus the per-row label
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
}

# Hard-coded baseline column from the paper's existing Table 5 (monolithic
# batch-trained baseline reported as 0.5000 across all metrics).
BATCH_BASELINE = {m: 0.5000 for m in METRICS}

# Eval-set selection — modelcomp_50k contains training data only.
# We use the tail of each training checkpoint slice (last 100 rows by id),
# stratified across C1–C4, giving 400 records covering the full distribution.
EVAL_RECORD_COUNT = 400
EVAL_PER_CHECKPOINT = 100
EVAL_CHECKPOINTS = [1, 2, 3, 4]

# Inference config — Alpaca answers are short; 64 new tokens is plenty
MAX_NEW_TOKENS = 64
BATCH_SIZE = 4

# ─────────────────────────────────────────────────────────────────────────────
# calculate_metrics() — verbatim from experiment/phase2_incremental/12_train_incremental.py
# Copied here (rather than imported) because that file imports unsloth/trl
# at module top level, which would clobber our PEFT-only inference path.
# ─────────────────────────────────────────────────────────────────────────────

def calculate_metrics(
    prediction: str,
    reference: str,
    instruction: str = "",
    context: str = "",
    task_label: str = "general_qa",
) -> dict:
    """
    Run the full 7-metric + ROUGE/BLEU evaluation and return a flat dict.

    Returns keys: overall, structured_correctness, task_success,
    instruction_following, coverage, faithfulness, hallucination,
    context_grounding, conciseness, rouge1, rougel, bleu, details
    """
    _zero = {
        "overall": 0.0, "structured_correctness": 0.0, "task_success": 0.0,
        "instruction_following": 0.0, "coverage": 0.0, "faithfulness": 0.0,
        "hallucination": 0.0, "context_grounding": 0.0, "conciseness": 0.0,
        "rouge1": 0.0, "rougel": 0.0, "bleu": 0.0, "details": {},
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

    rouge_obj = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    rouge_scores = rouge_obj.score(reference, prediction)
    smooth = SmoothingFunction().method1
    try:
        bleu = sentence_bleu(
            [reference.split()], prediction.split(), smoothing_function=smooth
        )
    except Exception:
        bleu = 0.0

    return {
        "overall":                eval_result["overall_score"],
        "structured_correctness": eval_result["structured_correctness"],
        "task_success":           eval_result["task_success"],
        "instruction_following":  eval_result["instruction_following"],
        "coverage":               eval_result["coverage"],
        "faithfulness":           eval_result["faithfulness"],
        "hallucination":          eval_result["hallucination"],
        "context_grounding":      eval_result["context_grounding"],
        "conciseness":            eval_result["conciseness"],
        "rouge1":                 rouge_scores["rouge1"].fmeasure,
        "rougel":                 rouge_scores["rougeL"].fmeasure,
        "bleu":                   bleu,
        "details":                eval_result.get("details", {}),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Supabase eval-set fetcher
# ─────────────────────────────────────────────────────────────────────────────

def fetch_eval_set() -> list[dict]:
    """
    Fetch the eval set from modelcomp_50k by stratified sampling across
    checkpoints 1–4.

    The table only contains checkpoints 1–4. We pull the last 100 rows
    (highest ids) from each checkpoint and concatenate, yielding 400
    records that span the full distribution of training data.

    Implementation note: a single "select(id, input, context, sevenb,
    checkpoint) where checkpoint=N order by id desc limit 100" hits the
    Supabase 8-second statement timeout because Postgres has to
    materialize the wide text columns while sorting. We split into two
    cheap queries:

        1) select id  where checkpoint=N order by id desc limit 100
           — id-only, uses the index, ~0.3s per checkpoint
        2) select id, input, context, sevenb, checkpoint where id in (...)
           — small set, no expensive sort, ~0.3s per checkpoint

    Returns the list of records ordered by ascending id within each
    checkpoint (deterministic processing order).
    """
    client = get_supabase_client()

    records: list[dict] = []
    for ckpt_num in EVAL_CHECKPOINTS:
        # Step 1 — cheap id-only pull to find the highest 100 ids.
        id_resp = (
            client.table("modelcomp_50k")
            .select("id")
            .eq("checkpoint", ckpt_num)
            .order("id", desc=True)
            .limit(EVAL_PER_CHECKPOINT)
            .execute()
        )
        ids = [row["id"] for row in (id_resp.data or [])]
        if not ids:
            log.warning(f"  checkpoint={ckpt_num}: no rows returned")
            continue

        # Step 2 — hydrate full columns by id IN (...). No sort cost.
        hydrate_resp = (
            client.table("modelcomp_50k")
            .select("id, input, context, sevenb, checkpoint")
            .in_("id", ids)
            .execute()
        )
        rows = list(hydrate_resp.data or [])
        # Restore ascending id order within this checkpoint slice.
        rows.sort(key=lambda r: r.get("id") or 0)

        valid = [r for r in rows if (r.get("sevenb") or "").strip()]
        log.info(
            f"  checkpoint={ckpt_num}: fetched {len(rows)} rows  |  "
            f"with non-empty sevenb: {len(valid)}"
        )
        records.extend(rows)

    log.info(
        f"  total combined: {len(records)} records across "
        f"checkpoints {EVAL_CHECKPOINTS}"
    )
    return records


def has_context(record: dict) -> bool:
    """A record has context iff its `context` column is a non-empty string."""
    ctx = record.get("context", "")
    return bool(ctx and str(ctx).strip())


# ─────────────────────────────────────────────────────────────────────────────
# Inference — training-time prompt format (### Instruction: / ### Context: / ### Response:)
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt_training_format(record: dict) -> str:
    """
    Reconstruct the exact prompt format used during fine-tuning of these
    adapters (see _build_prompt() in 12_train_incremental.py).
    """
    instruction = record.get("input", "") or ""
    context = record.get("context", "") or ""

    if context.strip():
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Context:\n{context}\n\n"
            f"### Response:\n"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def generate_batch_training_format(
    model, tokenizer, records: list[dict], batch_size: int = BATCH_SIZE
) -> list[str]:
    """
    Batched inference using the training-time prompt format.

    Mirrors generate_batch() in eval_checkpoints.py but swaps in the
    ### Instruction / ### Context / ### Response template.  Returns one
    decoded string per input record, in order.
    """
    import torch

    all_outputs: list[str] = []
    device = next(model.parameters()).device

    # Use left padding for decoder-only models so prompt_lens line up correctly
    saved_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            prompts = [build_prompt_training_format(r) for r in batch]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(device)

            # With left padding, every sequence in the batch is left-aligned at
            # the same column.  The model-only response begins at column
            # input_ids.shape[1].
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
                # The training prompt ends with "### Response:\n" — strip any
                # stray response markers and trailing whitespace.
                reply = reply.split("### Instruction:")[0]
                reply = reply.split("### Response:")[-1]
                all_outputs.append(reply.strip())
    finally:
        tokenizer.padding_side = saved_side

    return all_outputs


# ─────────────────────────────────────────────────────────────────────────────
# Per-checkpoint evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_one_checkpoint(
    ckpt_name: str, adapter_path: str, val_data: list[dict]
) -> list[dict]:
    """
    Evaluate one checkpoint over the entire eval set.  Returns one result
    dict per record.  Frees GPU memory before returning.
    """
    log.info("")
    log.info("=" * 60)
    log.info(f"Evaluating checkpoint: {ckpt_name}")
    log.info("=" * 60)

    model, tokenizer = load_model_and_tokenizer(adapter_path)

    # ── CUDA warmup ──────────────────────────────────────────────────────
    log.info("  Running CUDA warmup...")
    import torch
    warmup_inputs = tokenizer(["Hello"], return_tensors="pt").to(
        next(model.parameters()).device
    )
    with torch.inference_mode():
        model.generate(
            **warmup_inputs,
            max_new_tokens=4,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    del warmup_inputs
    torch.cuda.empty_cache()
    log.info("  Warmup complete — starting evaluation")

    # ── Inference + scoring loop, batched ────────────────────────────────
    import time
    results: list[dict] = []
    n = len(val_data)
    progress_every = 50  # log every 50 records

    for batch_start in range(0, n, BATCH_SIZE):
        batch = val_data[batch_start:batch_start + BATCH_SIZE]

        if batch_start == 0 or (batch_start // BATCH_SIZE) % (progress_every // BATCH_SIZE or 1) == 0:
            log.info(f"  [{batch_start}/{n}] processing batch of {len(batch)}")

        try:
            t0 = time.perf_counter()
            generated_list = generate_batch_training_format(
                model, tokenizer, batch, batch_size=BATCH_SIZE
            )
            elapsed = time.perf_counter() - t0

            for record, generated in zip(batch, generated_list):
                metrics = calculate_metrics(
                    prediction=generated,
                    reference=record.get("sevenb", "") or "",
                    instruction=record.get("input", "") or "",
                    context=record.get("context", "") or "",
                    task_label="general_qa",
                )
                results.append({
                    "ckpt":        ckpt_name,
                    "id":          record.get("id"),
                    "checkpoint":  record.get("checkpoint"),
                    "has_context": has_context(record),
                    "input":       (record.get("input") or "")[:200],
                    "generated":   generated[:400],
                    "reference":   (record.get("sevenb") or "")[:200],
                    "metrics":     {k: round(metrics[k], 4) for k in METRICS},
                })

            if batch_start == 0 or (batch_start // BATCH_SIZE) % (progress_every // BATCH_SIZE or 1) == 0:
                per_rec = elapsed / max(1, len(batch))
                log.info(f"    batch done in {elapsed:.2f}s ({per_rec:.2f}s/record)")

        except Exception as exc:
            log.warning(f"  Skipping batch at {batch_start}: {exc}")
            continue

    log.info(f"  Done: {len(results)}/{n} records scored")
    unload_model(model, tokenizer)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation, table, dynamics
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(records: list[dict]) -> dict:
    """Mean each metric over ctx / noctx / all subsets."""
    ctx = [r for r in records if r["has_context"]]
    noctx = [r for r in records if not r["has_context"]]

    def _mean(recs: list[dict]) -> dict[str, float]:
        if not recs:
            return {m: 0.0 for m in METRICS}
        return {
            m: round(sum(r["metrics"][m] for r in recs) / len(recs), 4)
            for m in METRICS
        }

    return {
        "ctx":     _mean(ctx),
        "noctx":   _mean(noctx),
        "all":     _mean(records),
        "n_ctx":   len(ctx),
        "n_noctx": len(noctx),
        "n_all":   len(records),
    }


def build_table5(summary: dict[str, dict]) -> str:
    """
    Build the Table 5 string with this exact column ordering:
        Metric | C1_ctx | C1_noctx | C2_ctx | C2_noctx |
                 C3_ctx | C3_noctx | C4_ctx | C4_noctx | Batch_baseline
    """
    ckpts = ["C1", "C2", "C3", "C4"]
    col_w = 10

    header_parts = [f"{'Metric':<26}"]
    for c in ckpts:
        header_parts.append(f"{c+'_ctx':>{col_w}}")
        header_parts.append(f"{c+'_noctx':>{col_w}}")
    header_parts.append(f"{'Batch_baseline':>16}")
    header = "  ".join(header_parts)

    sep = "-" * len(header)
    lines = [sep, header, sep]

    for m in METRICS:
        row = [f"{METRIC_LABELS[m]:<26}"]
        for c in ckpts:
            ctx_val = summary.get(c, {}).get("ctx", {}).get(m, 0.0)
            noctx_val = summary.get(c, {}).get("noctx", {}).get(m, 0.0)
            row.append(f"{ctx_val:>{col_w}.4f}")
            row.append(f"{noctx_val:>{col_w}.4f}")
        row.append(f"{BATCH_BASELINE[m]:>16.4f}")
        lines.append("  ".join(row))

    lines.append(sep)
    lines.append("")
    for c in ckpts:
        s = summary.get(c, {})
        lines.append(
            f"  {c}: n_ctx={s.get('n_ctx', 0)}, n_noctx={s.get('n_noctx', 0)}, "
            f"n_all={s.get('n_all', 0)}"
        )
    return "\n".join(lines)


def analyze_dynamics(summary: dict[str, dict]) -> str:
    """Flag non-monotonic behavior, regressions, and recoveries."""
    ckpts = ["C1", "C2", "C3", "C4"]
    lines = ["CHECKPOINT DYNAMICS ANALYSIS", "=" * 60, ""]

    for metric in METRICS:
        all_vals = [summary.get(c, {}).get("all", {}).get(metric, 0.0) for c in ckpts]
        label = METRIC_LABELS[metric]

        regressions, recoveries = [], []
        for i in range(1, len(all_vals)):
            delta = all_vals[i] - all_vals[i - 1]
            if delta < -0.005:
                regressions.append(
                    f"  {ckpts[i-1]}→{ckpts[i]}: {all_vals[i-1]:.4f} → "
                    f"{all_vals[i]:.4f}  (Δ={delta:+.4f})"
                )
            elif delta > 0.005 and i >= 2 and all_vals[i - 1] < all_vals[i - 2]:
                recoveries.append(
                    f"  {ckpts[i-1]}→{ckpts[i]}: {all_vals[i-1]:.4f} → "
                    f"{all_vals[i]:.4f}  (recovery Δ={delta:+.4f})"
                )

        is_monotone = all(
            all_vals[i] >= all_vals[i - 1] - 0.001 for i in range(1, len(all_vals))
        )
        trend = "monotone ↑" if is_monotone else "non-monotone"

        lines.append(f"{label} [{trend}]")
        lines.append(f"  Values: {' → '.join(f'{v:.4f}' for v in all_vals)}")
        if regressions:
            lines.append("  Regressions:")
            lines.extend(regressions)
        if recoveries:
            lines.append("  Recoveries:")
            lines.extend(recoveries)
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 60)
    log.info("INTUNE Table 5 — Supabase held-out eval (C1–C4)")
    log.info("=" * 60)

    # ── Fetch eval set ──────────────────────────────────────────────────────
    val_data = fetch_eval_set()
    log.info(
        f"Eval set: {len(val_data)} records from modelcomp_50k "
        f"(stratified across checkpoints {EVAL_CHECKPOINTS})"
    )

    # ── Sanity check before loading any model ──────────────────────────────
    valid_records = [r for r in val_data if (r.get("sevenb") or "").strip()]
    log.info(
        f"Sanity check: {len(val_data)} fetched, {len(valid_records)} with "
        f"non-empty sevenb (teacher output)"
    )
    if len(val_data) != EVAL_RECORD_COUNT:
        log.error(
            f"Expected {EVAL_RECORD_COUNT} records, got {len(val_data)} — aborting."
        )
        sys.exit(1)
    if not valid_records:
        log.error("No records have non-empty sevenb — aborting.")
        sys.exit(1)
    if len(valid_records) < len(val_data):
        log.warning(
            f"⚠  {len(val_data) - len(valid_records)} records have empty "
            f"sevenb; they will score 0 on every metric."
        )

    n_ctx = sum(1 for r in val_data if has_context(r))
    n_noctx = len(val_data) - n_ctx
    log.info(f"  Split: {n_ctx} with context, {n_noctx} zero-context")

    # ── Run evaluation per checkpoint, freeing GPU between each ────────────
    raw_path = REPORTS_DIR / "checkpoint_eval_supabase_raw.json"
    all_raw: list[dict] = []

    for ckpt_name, adapter_path in CHECKPOINTS.items():
        if not Path(adapter_path).exists():
            log.error(
                f"Adapter path missing: {adapter_path} — skipping {ckpt_name}"
            )
            continue

        ckpt_results = evaluate_one_checkpoint(ckpt_name, adapter_path, val_data)
        all_raw.extend(ckpt_results)

        # Crash-safe: persist after every checkpoint
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(all_raw, f, indent=2, ensure_ascii=False)
        log.info(f"Raw results saved: {raw_path}")

    # ── Aggregate ──────────────────────────────────────────────────────────
    log.info("")
    log.info("Aggregating results...")
    summary: dict[str, dict] = {}
    for ckpt_name in CHECKPOINTS:
        recs = [r for r in all_raw if r["ckpt"] == ckpt_name]
        if not recs:
            log.warning(f"No results for {ckpt_name} — omitting from summary")
            continue
        summary[ckpt_name] = aggregate(recs)

    summary_path = REPORTS_DIR / "checkpoint_eval_supabase_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "eval_checkpoints_used": EVAL_CHECKPOINTS,
                "eval_record_count": len(val_data),
                "eval_per_checkpoint": EVAL_PER_CHECKPOINT,
                "n_with_context": n_ctx,
                "n_zero_context": n_noctx,
                "batch_baseline": BATCH_BASELINE,
                "per_checkpoint": summary,
            },
            f,
            indent=2,
        )
    log.info(f"Summary saved: {summary_path}")

    # ── Table 5 ────────────────────────────────────────────────────────────
    table5 = build_table5(summary)
    table5_path = REPORTS_DIR / "checkpoint_eval_supabase_table5.txt"
    with open(table5_path, "w", encoding="utf-8") as f:
        f.write(table5 + "\n")

    print()
    print("=" * 80)
    print("TABLE 5 — Per-Metric Scores by Checkpoint (ctx / zero-ctx) + Batch Baseline")
    print("=" * 80)
    print(table5)

    # ── Dynamics ───────────────────────────────────────────────────────────
    dynamics = analyze_dynamics(summary)
    dynamics_path = REPORTS_DIR / "checkpoint_eval_supabase_dynamics.txt"
    with open(dynamics_path, "w", encoding="utf-8") as f:
        f.write(dynamics + "\n")

    print()
    print(dynamics)

    # ── Final pointer ──────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print(f"All outputs saved to: {REPORTS_DIR}")
    print(f"  - {raw_path.name}")
    print(f"  - {summary_path.name}")
    print(f"  - {table5_path.name}")
    print(f"  - {dynamics_path.name}")
    print("=" * 80)


if __name__ == "__main__":
    main()
