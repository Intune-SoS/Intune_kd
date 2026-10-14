"""
Incremental Learning Pipeline with Status-Based Workflow
=========================================================

Status Flow for each checkpoint (~5000 records):
  score        -> Score base student_output vs teacher (all metrics)
  finetune     -> Finetune model on this checkpoint's data
  output_tuned -> Generate student_output_tuned with finetuned model
  score_tuned  -> Score tuned output vs teacher (all metrics, _tuned suffix)
  completed    -> Calculate improvement = score_tuned - score

─────────────────────────────────────────────────────────────────────────────
EVALUATION METRIC FRAMEWORK  (7-metric LLM-judge system)
─────────────────────────────────────────────────────────────────────────────

  Overall = 0.60 × Quality  +  0.40 × Fidelity

  WHY 0.60 / 0.40?
  Quality captures whether the answer is useful to an end-user (the dominant
  goal). Fidelity captures faithfulness to source material, which matters in
  RAG/grounding tasks but is secondary when context is absent.
  60/40 keeps quality primary without ignoring faithfulness.

── GROUP 1 · QUALITY (weight 0.60) ─────────────────────────────────────────
  These ask: "Is this a good answer, independent of the teacher text?"

  structured_correctness  Does the output follow the expected format/schema
                          (JSON, bullets, steps)?  Good = format matches spec.

  task_success            Did the model actually do the task (answer, extract,
                          write code)?  Good = task objectively achieved.

  instruction_following   Did the model obey all explicit prompt constraints
                          (length, language, tone)?  Good = zero violations.

── GROUP 2 · FIDELITY (weight 0.40) ────────────────────────────────────────
  These ask: "How well does the output match the teacher and context?"

  coverage                Fraction of teacher key-facts in student output.
                          Good ≥ 0.85.

  faithfulness            All claims supported by teacher/context.
                          Good = 1.0 (no unsupported claims).

  hallucination           Fraction of fabricated sentences. Good = 0.0.
                          (lower is better; stored raw, not inverted)

  context_grounding       Token overlap with provided context spans.
                          Good = high when context is present.

  conciseness             Stored for diagnostics ONLY — not in Overall.
                          Excluded because optimal length is task-dependent.

─────────────────────────────────────────────────────────────────────────────

Columns:
  input, context          — prompt data
  sevenb                  — teacher output
  student_output          — base model output
  student_output_tuned    — finetuned model output
  latency, latency_tuned  — inference times (ms)
  checkpoint              — 1–10
  status                  — workflow status

  Base metrics:  score, structured_correctness, task_success,
                 instruction_following, coverage, faithfulness,
                 hallucination, context_grounding, conciseness,
                 rouge1, rougel, bleu

  Tuned metrics: score_tuned + all above with _tuned suffix
  improvement   = score_tuned - score

Usage:
    python experiment/12_train_incremental.py --checkpoint 1 --step status
    python experiment/12_train_incremental.py --checkpoint 1 --step score
    python experiment/12_train_incremental.py --checkpoint 1 --step finetune
    python experiment/12_train_incremental.py --checkpoint 1 --step output_tuned
    python experiment/12_train_incremental.py --checkpoint 1 --step score_tuned
    python experiment/12_train_incremental.py --checkpoint 1 --step completed
    python experiment/12_train_incremental.py --checkpoint 1 --run-all
    python experiment/12_train_incremental.py --checkpoint 1 --init
"""

import os
import sys
import json
import time
import argparse

# Fix Unicode/Emoji encoding on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


load_dotenv(project_path('.env'))

# CPU thread limits
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '2')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '2')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
# Disable torch.compile (Windows/Triton incompatibility)
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
os.environ.setdefault('INDUCTOR_NOBUILD', '1')
os.environ.setdefault('TORCHINDUCTOR_CPP_WRAPPER', '0')

import torch
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from supabase import create_client

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True
torch._inductor.config.cpp_wrapper = False

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'evaluation_suite'))
from importlib import import_module
eval_metrics = import_module('06_eval_metrics')
evaluate_single_output = eval_metrics.evaluate_single_output
compare_teacher_student = eval_metrics.compare_teacher_student

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TABLE_NAME = os.getenv('SUPABASE_TABLE_INCREMENTAL', 'modelcomp_50k')
MODEL_NAME = "unsloth/gemma-3-1b-it-bnb-4bit"
MAX_SEQ_LENGTH = 512          # Dataset P99 is 328 tokens, max is 440 tokens
RECORDS_PER_CHECKPOINT = 5000
MAX_NEW_TOKENS = 256          # Target response median is 81 tokens
TUNED_INFER_BATCH_SIZE = 128  # Buffer before GPU chunking
TUNED_SCAN_PAGE_SIZE = 500
TUNED_GPU_BATCH_SIZE = int(os.getenv('TUNED_GPU_BATCH_SIZE', '32'))  # RTX 4060 fits batch 32 in <2.5GB VRAM


LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0

STATUS_FLOW = ['score', 'finetune', 'output_tuned', 'score_tuned', 'completed']
MIN_RECORDS_PER_CHECKPOINT = 100

# How many rows to UPDATE per Supabase client session.
# Each batch opens ONE fresh HTTP/2 connection and executes rows sequentially
# on that connection. On disconnect, the whole batch retries with a new client.
UPDATE_BATCH_SIZE = 100

# Network error signatures that warrant retry + fresh connection
_RETRYABLE_SIGS = (
    'Server disconnected', 'RemoteProtocolError', 'ConnectError',
    'ReadError', 'TimeoutException', 'Connection', 'ConnectionError',
    'RemoteDisconnected', 'BrokenPipe',
    'statement timeout', '57014',
    'ReadTimeout', 'timeout', 'timed out', 'Timeout', 'ConnectTimeout',
    'PoolTimeout', 'NetworkError', 'TransportError', 'httpcore', 'httpx'
)


# ─────────────────────────────────────────────────────────────────────────────
# SUPABASE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_supabase():
    """
    Always returns a *fresh* Supabase client (new HTTP/2 connection).

    Calling this on every batch — not caching the result — is intentional.
    Long-running loops exhaust HTTP/2 stream IDs on a single connection,
    causing 'Server disconnected'. A fresh client resets the connection.
    """
    key = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_KEY')
    return create_client(os.getenv('SUPABASE_URL'), key)


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc)
    exc_type = type(exc).__name__
    return any(sig in msg or sig in exc_type for sig in _RETRYABLE_SIGS)


def supabase_batch_update_with_retry(
    table: str,
    rows: list,          # list of dicts, each MUST contain 'id' key
    id_col: str = 'id',
    max_retries: int = 5,
    concurrency: int = 16,
) -> None:
    """
    Update multiple rows concurrently using a ThreadPoolExecutor.
    Each thread gets its own client connection, speeding up database
    updates by up to 16x compared to sequential HTTP round-trips.
    """
    if not rows:
        return

    def _update_single(row):
        row_id = row[id_col]
        payload = {k: v for k, v in row.items() if k != id_col}
        last_exc = None
        for attempt in range(max_retries):
            try:
                client = get_supabase()
                client.table(table).update(payload).eq(id_col, row_id).execute()
                return
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    time.sleep(1.0 * (1.5 ** attempt))
                else:
                    print(f"⚠️  [DB WARN] Row {row_id} update failed after {max_retries} attempts: {exc}")
                    return

    for i in range(0, len(rows), UPDATE_BATCH_SIZE):
        batch = rows[i: i + UPDATE_BATCH_SIZE]
        workers = min(concurrency, len(batch))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_update_single, batch))


def supabase_single_update_with_retry(
    table: str,
    data: dict,
    match_col: str,
    match_val,
    max_retries: int = 6,
) -> None:
    """
    Update a single row with exponential-backoff retry + fresh client.
    Used for status transitions and individual record writes (e.g. inference).
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            get_supabase().table(table).update(data).eq(match_col, match_val).execute()
            return
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1 and _is_retryable(exc):
                wait = min(2.0 * (2 ** attempt), 60)
                print(
                    f"\n⚠️  DB write error (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {wait:.0f}s\n   {exc}"
                )
                time.sleep(wait)
            else:
                raise
    raise last_exc


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def get_checkpoint_size(checkpoint: int) -> int:
    result = get_supabase().table(TABLE_NAME) \
        .select('id', count='exact').eq('checkpoint', checkpoint).execute()
    return result.count or 0


def fetch_records_by_status(checkpoint: int, status: str) -> list:
    """Paginated fetch — Supabase caps at 1000 rows per response."""
    print(f"\nFetching checkpoint {checkpoint} records with status='{status}' from '{TABLE_NAME}'...")
    all_records, offset, batch = [], 0, 1000
    while True:
        result = get_supabase().table(TABLE_NAME) \
            .select('*') \
            .eq('checkpoint', checkpoint) \
            .eq('status', status) \
            .range(offset, offset + batch - 1) \
            .execute()
        if not result.data:
            break
        all_records.extend(result.data)
        offset += batch
        if len(result.data) < batch:
            break
    print(f"Fetched {len(all_records)} records")
    return all_records


def fetch_checkpoint_records(checkpoint: int) -> list:
    """Fetch ALL records for a checkpoint regardless of status."""
    print(f"\nFetching all records for checkpoint {checkpoint} from '{TABLE_NAME}'...")
    all_records, offset, batch = [], 0, 1000
    while True:
        result = get_supabase().table(TABLE_NAME) \
            .select('*').eq('checkpoint', checkpoint) \
            .range(offset, offset + batch - 1).execute()
        if not result.data:
            break
        all_records.extend(result.data)
        offset += batch
        if len(result.data) < batch:
            break
    print(f"Fetched {len(all_records)} records")
    return all_records


def update_status_bulk(record_ids: list, new_status: str) -> None:
    """Bulk-update the status column using batched UPDATE with retry."""
    print(f"\nUpdating {len(record_ids)} records to status='{new_status}' in '{TABLE_NAME}'...")
    rows = [{'id': rid, 'status': new_status} for rid in record_ids]
    # Show tqdm over batches, not individual rows
    batch_count = (len(rows) + UPDATE_BATCH_SIZE - 1) // UPDATE_BATCH_SIZE
    for i in tqdm(range(0, len(rows), UPDATE_BATCH_SIZE),
                  total=batch_count, desc="Updating status"):
        batch = rows[i: i + UPDATE_BATCH_SIZE]
        supabase_batch_update_with_retry(TABLE_NAME, batch)


def validate_records_count(
    records,
    step_name: str,
    min_required: int = MIN_RECORDS_PER_CHECKPOINT,
) -> bool:
    """Print a readiness dashboard and return True only if enough records exist."""
    count = len(records) if isinstance(records, list) else int(records)
    remaining = max(0, min_required - count)
    pct = (count / min_required * 100) if min_required else 0
    bar_len = 40
    filled = min(int(bar_len * count / min_required), bar_len) if min_required else 0
    bar = '█' * filled + '░' * (bar_len - filled)

    print(f"\n{'─' * 50}")
    print(f"📊 DATA READINESS CHECK: {step_name}")
    print(f"{'─' * 50}")
    print(f"   Records ready:     {count:,}")
    print(f"   Records required:  {min_required:,}")
    print(f"   Records remaining: {remaining:,}")
    print(f"   Progress:          {pct:.1f}%")
    print(f"   [{bar}]")
    print(f"{'─' * 50}")

    if count < min_required:
        print(f"\n❌ VALIDATION FAILED: found {count:,} records for '{step_name}', but need at least {min_required:,}.")
        print(f"   Table queried: '{TABLE_NAME}'")
        print(f"   💡 Likely cause: Row Level Security (RLS) is enabled in Supabase, blocking the API from reading rows.")
        print(f"   Fix: Run this in your Supabase SQL Editor:")
        print(f"        ALTER TABLE {TABLE_NAME} DISABLE ROW LEVEL SECURITY;")
        return False
    print(f"✅ Validation passed: {count:,} records ready for {step_name}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_metrics(
    prediction: str,
    reference: str,
    instruction: str = '',
    context: str = '',
    task_label: str = 'general_qa',
) -> dict:
    """
    Run the full 7-metric + ROUGE/BLEU evaluation and return a flat dict.

    Overall = 0.60 × Quality + 0.40 × Fidelity
      Quality  = mean(structured_correctness, task_success, instruction_following)
      Fidelity = weighted blend: coverage(0.375) + faithfulness(0.25)
                 + (1-hallucination)(0.25) + context_grounding(0.125)

    Returns keys: overall, structured_correctness, task_success,
    instruction_following, coverage, faithfulness, hallucination,
    context_grounding, conciseness, rouge1, rougel, bleu, details
    """
    _zero = {
        'overall': 0.0, 'structured_correctness': 0.0, 'task_success': 0.0,
        'instruction_following': 0.0, 'coverage': 0.0, 'faithfulness': 0.0,
        'hallucination': 0.0, 'context_grounding': 0.0, 'conciseness': 0.0,
        'rouge1': 0.0, 'rougel': 0.0, 'bleu': 0.0, 'details': {},
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

    rouge_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge_obj.score(reference, prediction)
    smooth = SmoothingFunction().method1
    try:
        bleu = sentence_bleu([reference.split()], prediction.split(), smoothing_function=smooth)
    except Exception:
        bleu = 0.0

    return {
        'overall':                eval_result['overall_score'],
        'structured_correctness': eval_result['structured_correctness'],
        'task_success':           eval_result['task_success'],
        'instruction_following':  eval_result['instruction_following'],
        'coverage':               eval_result['coverage'],
        'faithfulness':           eval_result['faithfulness'],
        'hallucination':          eval_result['hallucination'],
        'context_grounding':      eval_result['context_grounding'],
        'conciseness':            eval_result['conciseness'],
        'rouge1':                 rouge_scores['rouge1'].fmeasure,
        'rougel':                 rouge_scores['rougeL'].fmeasure,
        'bleu':                   bleu,
        'details':                eval_result.get('details', {}),
    }


def _print_metrics_table(label: str, checkpoint: int, all_metrics: list) -> None:
    """Pretty-print a two-group evaluation summary."""
    if not all_metrics:
        return
    n = len(all_metrics)
    avg = lambda k: sum(m[k] for m in all_metrics) / n  # noqa: E731

    print(f"\n{'=' * 62}")
    print(f"📊 {label}  |  Checkpoint {checkpoint}")
    print(f"{'=' * 62}")
    print(f"   Records evaluated:         {n:,}")
    print(f"   {'─' * 44}")
    print(f"   ★ Overall Score:           {avg('overall'):.4f}")
    print(f"   {'─' * 44}")
    print(f"   [QUALITY  · weight 0.60 of Overall]")
    print(f"   Structured Correctness:    {avg('structured_correctness'):.4f}"
          f"  ← format/schema match")
    print(f"   Task Success:              {avg('task_success'):.4f}"
          f"  ← task actually achieved")
    print(f"   Instruction Following:     {avg('instruction_following'):.4f}"
          f"  ← prompt constraints obeyed")
    print(f"   {'─' * 44}")
    print(f"   [FIDELITY · weight 0.40 of Overall]")
    print(f"   Coverage:                  {avg('coverage'):.4f}"
          f"  ← teacher key-facts reproduced  (↑ good)")
    print(f"   Faithfulness:              {avg('faithfulness'):.4f}"
          f"  ← no unsupported claims         (↑ good)")
    print(f"   Hallucination:             {avg('hallucination'):.4f}"
          f"  ← fabricated sentences          (↓ good)")
    print(f"   Context Grounding:         {avg('context_grounding'):.4f}"
          f"  ← anchored to provided context")
    print(f"   {'─' * 44}")
    print(f"   [DIAGNOSTICS — not in Overall]")
    print(f"   Conciseness:               {avg('conciseness'):.4f}")
    print(f"   ROUGE-1:                   {avg('rouge1'):.4f}")
    print(f"   ROUGE-L:                   {avg('rougel'):.4f}")
    print(f"   BLEU:                      {avg('bleu'):.4f}")
    print(f"{'=' * 62}")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt(instruction: str, context: str = '') -> str:
    return (
        f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n"
        if context else
        f"### Instruction:\n{instruction}\n\n### Response:\n"
    )


def _extract_response(decoded: str) -> str:
    if '### Response:' in decoded:
        return decoded.split('### Response:')[-1].strip()
    return decoded.strip()


def _is_cuda_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return 'out of memory' in msg and ('cuda' in msg or 'cublas' in msg)


def _prompt_len_proxy(item: dict) -> int:
    return len(item.get('input', '') or '') + len(item.get('context', '') or '')


def generate_output(model, tokenizer, instruction: str, context: str = '') -> tuple:
    """Run single-item inference and return (response_text, latency_ms)."""
    prompt = _build_prompt(instruction, context)
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs['input_ids'].shape[1]

    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True, temperature=0.7, top_p=0.9,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    latency_ms = (time.time() - t0) * 1000

    response_tokens = out[0][input_len:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
    if not response:
        response = _extract_response(tokenizer.decode(out[0], skip_special_tokens=True))
    return response, latency_ms


def generate_output_batch(model, tokenizer, records: list) -> tuple[list, list]:
    """
    Run batched inference for a list of records with left-padding.
    Returns (outputs, per_row_latency_ms).
    """
    prompts = [_build_prompt(r['input'], r.get('context', '')) for r in records]
    saved_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    try:
        inputs = tokenizer(
            prompts,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=MAX_SEQ_LENGTH,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs['input_ids'].shape[1]

        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        batch_latency_ms = (time.time() - t0) * 1000

        outputs = []
        for seq in out:
            response_tokens = seq[input_len:]
            text = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            if not text:
                text = _extract_response(tokenizer.decode(seq, skip_special_tokens=True))
            outputs.append(text)

        per_row_latency = [batch_latency_ms / max(1, len(records))] * len(records)
        return outputs, per_row_latency
    finally:
        tokenizer.padding_side = saved_side


def generate_output_batch_adaptive(
    model,
    tokenizer,
    records: list,
    max_gpu_batch: int,
    desc: str = 'Generating tuned outputs',
) -> tuple[list, list]:
    """
    Adaptive micro-batching: shrinks batch size on CUDA OOM and resumes.
    """
    outputs, latencies = [], []
    idx = 0
    curr_bs = max(1, max_gpu_batch)

    pbar = tqdm(total=len(records), desc=desc, unit='records')
    try:
        while idx < len(records):
            take = min(curr_bs, len(records) - idx)
            chunk = records[idx: idx + take]
            try:
                chunk_out, chunk_lat = generate_output_batch(model, tokenizer, chunk)
                outputs.extend(chunk_out)
                latencies.extend(chunk_lat)
                idx += take
                pbar.update(take)
            except RuntimeError as exc:
                if _is_cuda_oom(exc) and take > 1:
                    curr_bs = max(1, take // 2)
                    print(f"⚠️  CUDA OOM at batch={take}; retrying with batch={curr_bs}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise
    finally:
        pbar.close()

    return outputs, latencies


# ─────────────────────────────────────────────────────────────────────────────
# STEP: status
# ─────────────────────────────────────────────────────────────────────────────

def step_status(checkpoint: int) -> dict:
    """Print a full status dashboard for a checkpoint."""
    print(f"\n{'=' * 60}")
    print(f"📊 STATUS SUMMARY — Checkpoint {checkpoint}")
    print(f"{'=' * 60}")

    all_records = fetch_checkpoint_records(checkpoint)
    counts: dict = {}
    for r in all_records:
        s = r.get('status') or 'null/empty'
        counts[s] = counts.get(s, 0) + 1

    print(f"\n   Total records: {len(all_records)}")
    print(f"   Required per step: {MIN_RECORDS_PER_CHECKPOINT}")
    print(f"\n   {'─' * 52}")

    next_step, ready_count = None, 0
    for status in ['null/empty'] + STATUS_FLOW:
        count = counts.get(status, 0)
        pct = (count / MIN_RECORDS_PER_CHECKPOINT * 100) if MIN_RECORDS_PER_CHECKPOINT else 0
        filled = min(int(30 * count / MIN_RECORDS_PER_CHECKPOINT), 30) if MIN_RECORDS_PER_CHECKPOINT else 0
        bar = '█' * filled + '░' * (30 - filled)
        icon = '✅' if count >= MIN_RECORDS_PER_CHECKPOINT else ('⏳' if count > 0 else '❌')
        print(f"   {icon} {status:15} : {count:5} [{bar}] {pct:.0f}%")
        if status in STATUS_FLOW and count > 0 and next_step is None:
            next_step, ready_count = status, count

    print(f"\n   {'─' * 52}")
    if next_step:
        remaining = max(0, MIN_RECORDS_PER_CHECKPOINT - ready_count)
        pct = ready_count / MIN_RECORDS_PER_CHECKPOINT * 100
        print(f"   Next step:    --step {next_step}")
        print(f"   Ready:        {ready_count:,}  ({pct:.1f}%)")
        print(f"   Remaining:    {remaining:,}")
        if ready_count >= MIN_RECORDS_PER_CHECKPOINT:
            print(f"   ✅ READY TO RUN: --checkpoint {checkpoint} --step {next_step}")
        else:
            print(f"   ⏳ Need {remaining:,} more records before '{next_step}'")
    else:
        null_c = counts.get('null/empty', 0)
        done_c = counts.get('completed', 0)
        if null_c > 0:
            print(f"   ⚠️  Run --init to initialize {null_c:,} records")
        elif done_c >= MIN_RECORDS_PER_CHECKPOINT:
            print(f"   ✅ Checkpoint {checkpoint} COMPLETED!")
            if checkpoint < 10:
                print(f"      Ready for checkpoint {checkpoint + 1}")

    print(f"{'=' * 60}")
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# STEP: init
# ─────────────────────────────────────────────────────────────────────────────

def init_checkpoint(checkpoint: int) -> int:
    """Set status='score' for records that have no status yet."""
    print(f"\n{'=' * 60}")
    print(f"🔧 INIT — Checkpoint {checkpoint}")
    print(f"{'=' * 60}")

    all_records, offset, batch = [], 0, 1000
    while True:
        result = get_supabase().table(TABLE_NAME) \
            .select('id, sevenb, student_output') \
            .eq('checkpoint', checkpoint) \
            .or_('status.is.null,status.eq.') \
            .range(offset, offset + batch - 1) \
            .execute()
        if not result.data:
            break
        all_records.extend(result.data)
        offset += batch
        if len(result.data) < batch:
            break

    print(f"Found {len(all_records)} records without status")

    valid_ids, missing_sevenb, missing_student = [], 0, 0
    for r in all_records:
        if not r.get('sevenb'):
            missing_sevenb += 1
        elif not r.get('student_output'):
            missing_student += 1
        else:
            valid_ids.append(r['id'])

    if missing_sevenb:
        print(f"⚠️  {missing_sevenb} records missing teacher output (sevenb)")
    if missing_student:
        print(f"⚠️  {missing_student} records missing base student output")

    total = get_checkpoint_size(checkpoint)
    required = max(MIN_RECORDS_PER_CHECKPOINT, total - missing_sevenb)
    count = len(valid_ids)
    pct = count / required * 100 if required else 0
    bar = '█' * min(int(40 * pct / 100), 40) + '░' * max(0, 40 - int(40 * pct / 100))

    print(f"\n{'─' * 50}")
    print(f"   Total in checkpoint: {total:,}")
    print(f"   Records ready:       {count:,} / {required:,}  ({pct:.1f}%)")
    print(f"   [{bar}]")
    print(f"{'─' * 50}")

    if valid_ids:
        update_status_bulk(valid_ids, 'score')
        print(f"✅ {len(valid_ids)} records initialized → status 'score'")
    else:
        print("❌ No valid records to initialize")

    return len(valid_ids)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: score  —  evaluate BASE student output
# ─────────────────────────────────────────────────────────────────────────────

def step_score(checkpoint: int) -> float:
    """
    Score the base student_output against the teacher (sevenb).

    Writes `score` (overall) + all 11 individual metric columns.
    Uses batched UPDATE — never upsert — so NOT NULL constraints are safe.
    Status transitions: 'score' → 'finetune'.

    Returns average overall score (0–1).
    """
    print(f"\n{'=' * 60}")
    print(f"📊 STEP: score — Base Student Evaluation | Checkpoint {checkpoint}")
    print(f"   Weights: Quality 0.60 · Fidelity 0.40")
    print(f"{'=' * 60}")

    records = fetch_records_by_status(checkpoint, 'score')
    if not records:
        downstream = get_supabase().table(TABLE_NAME) \
            .select('id', count='exact') \
            .eq('checkpoint', checkpoint) \
            .in_('status', ['finetune', 'output_tuned', 'score_tuned', 'completed']) \
            .limit(1).execute()
        if downstream.count and downstream.count > 0:
            print(f"⏩ Checkpoint {checkpoint}: {downstream.count:,} records already past 'score' (status='finetune' or later). Skipping to next step.")
            return 1.0

    if not validate_records_count(records, 'score'):
        return 0.0

    all_metrics: list = []
    pending: list = []

    def _flush(force: bool = False) -> None:
        if pending and (force or len(pending) >= UPDATE_BATCH_SIZE):
            supabase_batch_update_with_retry(TABLE_NAME, list(pending))
            pending.clear()

    for item in tqdm(records, desc="Scoring base student"):
        m = None
        if item.get('student_output') and item.get('sevenb'):
            m = calculate_metrics(
                prediction=item['student_output'],
                reference=item['sevenb'],
                instruction=item.get('input', ''),
                context=item.get('context', ''),
                task_label=item.get('task_label', 'general_qa'),
            )

        row = {
            'id':     item['id'],
            'status': 'finetune',
            # ── score (overall) explicitly set ───────────────────────────────
            'score':  round(m['overall'], 4) if m else None,
        }
        if m:
            row.update({
                # Quality group
                'structured_correctness': round(m['structured_correctness'], 4),
                'task_success':           round(m['task_success'], 4),
                'instruction_following':  round(m['instruction_following'], 4),
                # Fidelity group
                'coverage':               round(m['coverage'], 4),
                'faithfulness':           round(m['faithfulness'], 4),
                'hallucination':          round(m['hallucination'], 4),
                'context_grounding':      round(m['context_grounding'], 4),
                # Diagnostics
                'conciseness':            round(m['conciseness'], 4),
                'rouge1':                 round(m['rouge1'], 4),
                'rougel':                 round(m['rougel'], 4),
                'bleu':                   round(m['bleu'], 4),
            })
            all_metrics.append(m)

        pending.append(row)
        _flush()

    _flush(force=True)

    _print_metrics_table('BASE STUDENT EVALUATION SUMMARY', checkpoint, all_metrics)
    avg = sum(m['overall'] for m in all_metrics) / len(all_metrics) if all_metrics else 0.0
    print(f"\n✅ {len(records)} records → 'finetune' | avg score {avg:.4f}")
    return avg


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: finetune  —  LoRA fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def step_finetune(checkpoint: int):
    """
    Fine-tune the model on this checkpoint's data using QLoRA (4-bit + LoRA).
    Status transitions: 'finetune' → 'output_tuned'.
    Returns output directory path, or None on validation failure.
    """
    print(f"\n{'=' * 60}")
    print(f"🔧 STEP: finetune — Train LoRA | Checkpoint {checkpoint}")
    print(f"{'=' * 60}")

    records = fetch_records_by_status(checkpoint, 'finetune')
    if not records:
        downstream = get_supabase().table(TABLE_NAME) \
            .select('id', count='exact') \
            .eq('checkpoint', checkpoint) \
            .in_('status', ['output_tuned', 'score_tuned', 'completed']) \
            .limit(1).execute()
        if downstream.count and downstream.count > 0:
            print(f"⏩ Checkpoint {checkpoint}: {downstream.count:,} records already past 'finetune' (status='output_tuned' or later). Skipping to next step.")
            return project_path('models', f'gemma-ckpt{checkpoint}-lora')

    if not validate_records_count(records, 'finetune'):
        return None

    _lora_kwargs = dict(
        r=LORA_R,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
        lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias='none', use_gradient_checkpointing='unsloth', random_state=42,
    )

    prev_path = project_path('models', f'gemma-ckpt{checkpoint - 1}-lora')
    if checkpoint > 1 and os.path.exists(prev_path):
        print(f"\nLoading previous adapter from {prev_path}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=prev_path, max_seq_length=MAX_SEQ_LENGTH,
            dtype=None, load_in_4bit=True,
        )
    else:
        print(f"\nLoading base model {MODEL_NAME}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH,
            dtype=None, load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(model, **_lora_kwargs)

    eos = tokenizer.eos_token or "<end_of_turn>"
    print(f"Using EOS token: {repr(eos)}")

    formatted = []
    for item in records:
        ctx = (item.get('context') or '').strip()
        ans = (item.get('sevenb') or '').strip()
        text = (
            f"### Instruction:\n{item['input']}\n\n"
            f"### Context:\n{ctx}\n\n### Response:\n{ans}{eos}"
            if ctx else
            f"### Instruction:\n{item['input']}\n\n### Response:\n{ans}{eos}"
        )
        formatted.append({'text': text})

    dataset = Dataset.from_list(formatted)
    print("Pre-tokenizing dataset (single worker, Windows-safe)...")
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )
    dataset = dataset.map(tokenize_fn, batched=True, num_proc=1, remove_columns=["text"])
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    print(f"Training on {len(dataset)} records")

    output_dir = project_path('models', f'gemma-ckpt{checkpoint}-lora')

    training_args = TrainingArguments(
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        warmup_steps=30, num_train_epochs=1, learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
        logging_steps=25, optim='adamw_8bit', weight_decay=0.01,
        lr_scheduler_type='linear', seed=42, output_dir=output_dir,
        save_strategy='epoch', gradient_checkpointing=True,
        remove_unused_columns=False, ddp_find_unused_parameters=False,
        torch_compile=False, torch_compile_backend='inductor',
    )
    training_args.dataset_num_proc = 1

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=dataset,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=1, packing=False,
        args=training_args,
    )

    torch._dynamo.config.disable = True
    torch.compiler.disable()
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Adapter saved → {output_dir}")

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    update_status_bulk([r['id'] for r in records], 'output_tuned')
    print(f"✅ Status → 'output_tuned'")
    return output_dir


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: output_tuned  —  inference with fine-tuned model
# ─────────────────────────────────────────────────────────────────────────────

def step_output_tuned(checkpoint: int) -> int:
    """
    Generate student_output_tuned using the fine-tuned LoRA adapter.
    Status transitions: 'output_tuned' → 'score_tuned'.
    Returns number of records processed.
    """
    print(f"\n{'=' * 60}")
    print(f"🤖 STEP: output_tuned — Tuned Inference | Checkpoint {checkpoint}")
    print(f"{'=' * 60}")

    pending_count = get_supabase().table(TABLE_NAME) \
        .select('id', count='exact') \
        .eq('checkpoint', checkpoint) \
        .eq('status', 'output_tuned') \
        .execute().count or 0

    if pending_count == 0:
        downstream = get_supabase().table(TABLE_NAME) \
            .select('id', count='exact') \
            .eq('checkpoint', checkpoint) \
            .in_('status', ['score_tuned', 'completed']) \
            .limit(1).execute()
        if downstream.count and downstream.count > 0:
            print(f"⏩ Checkpoint {checkpoint}: {downstream.count:,} records already past 'output_tuned' (status='score_tuned' or later). Skipping to next step.")
            return downstream.count

    if not validate_records_count(pending_count, 'output_tuned'):
        return 0

    model_path = project_path('models', f'gemma-ckpt{checkpoint}-lora')
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}  — run 'finetune' first.")
        return 0

    print(f"\nLoading fine-tuned model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path, max_seq_length=MAX_SEQ_LENGTH,
        dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    processed = 0
    last_seen_id = 0
    pending_buffer = []

    while True:
        while len(pending_buffer) < TUNED_INFER_BATCH_SIZE:
            page = get_supabase().table(TABLE_NAME) \
                .select('id, input, context') \
                .eq('checkpoint', checkpoint) \
                .eq('status', 'output_tuned') \
                .gt('id', last_seen_id) \
                .order('id') \
                .limit(TUNED_SCAN_PAGE_SIZE) \
                .execute()

            if not page.data:
                break

            last_seen_id = page.data[-1]['id']
            pending_buffer.extend(page.data)

            if len(page.data) < TUNED_SCAN_PAGE_SIZE:
                break

        if not pending_buffer:
            break

        current_batch = pending_buffer[:TUNED_INFER_BATCH_SIZE]
        pending_buffer = pending_buffer[TUNED_INFER_BATCH_SIZE:]

        rows_to_update = []
        current_batch.sort(key=_prompt_len_proxy)
        try:
            outputs, latencies = generate_output_batch_adaptive(
                model=model,
                tokenizer=tokenizer,
                records=current_batch,
                max_gpu_batch=max(1, TUNED_GPU_BATCH_SIZE),
                desc=f"Generating tuned outputs (ckpt {checkpoint})",
            )

            for item, output, latency in zip(current_batch, outputs, latencies):
                rows_to_update.append(
                    {
                        'id': item['id'],
                        'student_output_tuned': output[:5000],
                        'latency_tuned': round(latency, 3),
                        'status': 'score_tuned',
                    }
                )
        except Exception as e:
            print(f"Error while processing checkpoint batch at id>{last_seen_id}: {e}")

        if rows_to_update:
            supabase_batch_update_with_retry(TABLE_NAME, rows_to_update)
            processed += len(rows_to_update)

        print(f"Checkpoint {checkpoint} output_tuned progress: {processed}/{pending_count}")

    print(f"✅ Tuned outputs generated for {processed} records → 'score_tuned'")
    return processed


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: score_tuned  —  evaluate TUNED student output
# ─────────────────────────────────────────────────────────────────────────────

def step_score_tuned(checkpoint: int) -> float:
    """
    Score student_output_tuned against the teacher using the full metric framework.

    FIX 1: Uses batched UPDATE (not upsert) — safe for NOT NULL constraints.
    FIX 2: `score_tuned` (overall) explicitly written in every row.
    FIX 3: Fresh Supabase client per batch prevents 'Server disconnected'.
    Status transitions: 'score_tuned' → 'completed'.

    Returns average tuned overall score (0–1).
    """
    print(f"\n{'=' * 60}")
    print(f"📊 STEP: score_tuned — Tuned Evaluation | Checkpoint {checkpoint}")
    print(f"   Weights: Quality 0.60 · Fidelity 0.40")
    print(f"{'=' * 60}")

    records = fetch_records_by_status(checkpoint, 'score_tuned')
    if not records:
        downstream = get_supabase().table(TABLE_NAME) \
            .select('id', count='exact') \
            .eq('checkpoint', checkpoint) \
            .eq('status', 'completed') \
            .limit(1).execute()
        if downstream.count and downstream.count > 0:
            print(f"⏩ Checkpoint {checkpoint}: {downstream.count:,} records already completed. Skipping to next step.")
            return 1.0

    if not validate_records_count(records, 'score_tuned'):
        return 0.0

    all_metrics: list = []
    pending: list = []

    def _flush(force: bool = False) -> None:
        if pending and (force or len(pending) >= UPDATE_BATCH_SIZE):
            supabase_batch_update_with_retry(TABLE_NAME, list(pending))
            pending.clear()

    for item in tqdm(records, desc="Scoring tuned student"):
        m = None
        if item.get('student_output_tuned') and item.get('sevenb'):
            m = calculate_metrics(
                prediction=item['student_output_tuned'],
                reference=item['sevenb'],
                instruction=item.get('input', ''),
                context=item.get('context', ''),
                task_label=item.get('task_label', 'general_qa'),
            )

        row = {
            'id':          item['id'],
            'status':      'completed',
            # ── score_tuned (overall) explicitly set ─────────────────────────
            'score_tuned': round(m['overall'], 4) if m else None,
        }
        if m:
            row.update({
                # Quality group with _tuned suffix
                'structured_correctness_tuned': round(m['structured_correctness'], 4),
                'task_success_tuned':           round(m['task_success'], 4),
                'instruction_following_tuned':  round(m['instruction_following'], 4),
                # Fidelity group with _tuned suffix
                'coverage_tuned':               round(m['coverage'], 4),
                'faithfulness_tuned':           round(m['faithfulness'], 4),
                'hallucination_tuned':          round(m['hallucination'], 4),
                'context_grounding_tuned':      round(m['context_grounding'], 4),
                # Diagnostics
                'conciseness_tuned':            round(m['conciseness'], 4),
                'rouge1_tuned':                 round(m['rouge1'], 4),
                'rougel_tuned':                 round(m['rougel'], 4),
                'bleu_tuned':                   round(m['bleu'], 4),
            })
            all_metrics.append(m)

        pending.append(row)
        _flush()

    _flush(force=True)

    _print_metrics_table('TUNED STUDENT EVALUATION SUMMARY', checkpoint, all_metrics)
    avg = sum(m['overall'] for m in all_metrics) / len(all_metrics) if all_metrics else 0.0
    print(f"\n✅ {len(records)} records → 'completed' | avg tuned score {avg:.4f}")
    return avg


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: completed  —  improvement calculation
# ─────────────────────────────────────────────────────────────────────────────

def step_completed(checkpoint: int):
    """
    Compute improvement = score_tuned - score and write to the `improvement` column.

    FIX: Uses batched UPDATE — never upsert — so existing rows are safely updated
    without touching NOT NULL columns that aren't in the payload.

    Also saves a JSON report to reports/incremental/.
    """
    print(f"\n{'=' * 60}")
    print(f"✅ STEP: completed — Improvement | Checkpoint {checkpoint}")
    print(f"{'=' * 60}")

    records = fetch_records_by_status(checkpoint, 'completed')
    if not validate_records_count(records, 'completed'):
        return None

    improvements = []
    pending: list = []

    def _flush(force: bool = False) -> None:
        if pending and (force or len(pending) >= UPDATE_BATCH_SIZE):
            supabase_batch_update_with_retry(TABLE_NAME, list(pending))
            pending.clear()

    for item in tqdm(records, desc="Writing improvement scores"):
        score_b = float(item.get('score') or 0.0)
        score_a = float(item.get('score_tuned') or 0.0)
        imp = round(score_a - score_b, 4)

        # Only 'id' + 'improvement' — safe because we UPDATE, not upsert
        pending.append({'id': item['id'], 'improvement': imp})
        _flush()

        if item.get('score') is not None and item.get('score_tuned') is not None:
            improvements.append({'before': score_b, 'after': score_a, 'improvement': imp})

    _flush(force=True)

    if not improvements:
        print("⚠️  No records with both score and score_tuned — nothing to report.")
        return None

    n = len(improvements)
    avg_b    = sum(i['before']      for i in improvements) / n
    avg_a    = sum(i['after']       for i in improvements) / n
    avg_imp  = sum(i['improvement'] for i in improvements) / n
    pct_imp  = avg_imp / avg_b * 100 if avg_b else 0.0
    positive = sum(1 for i in improvements if i['improvement'] > 0)
    negative = sum(1 for i in improvements if i['improvement'] < 0)
    neutral  = n - positive - negative

    print(f"\n{'=' * 62}")
    print(f"📈 CHECKPOINT {checkpoint} IMPROVEMENT REPORT")
    print(f"{'=' * 62}")
    print(f"   Records processed:   {n:,}")
    print(f"   {'─' * 44}")
    print(f"   Avg Score Before:    {avg_b:.4f}")
    print(f"   Avg Score After:     {avg_a:.4f}")
    print(f"   Avg Improvement:     {avg_imp:+.4f}  ({pct_imp:+.1f}%)")
    print(f"   {'─' * 44}")
    print(f"   Improved records:    {positive:,}  ({positive / n * 100:.1f}%)")
    print(f"   Regressed records:   {negative:,}  ({negative / n * 100:.1f}%)")
    print(f"   Neutral records:     {neutral:,}")
    print(f"{'=' * 62}")

    metrics_list = [
        'structured_correctness', 'task_success', 'instruction_following',
        'coverage', 'faithfulness', 'hallucination', 'context_grounding',
        'conciseness', 'rouge1', 'rougel', 'bleu'
    ]

    def _safe_avg(field):
        vals = [float(r[field]) for r in records if r.get(field) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    metric_details = {}
    for m in metrics_list:
        m_b = _safe_avg(m)
        m_a = _safe_avg(f"{m}_tuned")
        delta = round(m_a - m_b, 4) if (m_a is not None and m_b is not None) else None
        metric_details[m] = {
            'before': m_b,
            'after': m_a,
            'improvement': delta
        }

    report = {
        'checkpoint': checkpoint,
        'timestamp': datetime.now().isoformat(),
        'records_evaluated': n,
        'summary': {
            'overall_score_before': round(avg_b, 4),
            'overall_score_after': round(avg_a, 4),
            'improvement': round(avg_imp, 4),
            'improvement_pct': round(pct_imp, 2),
            'win_rate_improved_pct': round(positive / n * 100, 1),
            'regression_pct': round(negative / n * 100, 1),
            'neutral_pct': round(neutral / n * 100, 1),
            'counts': {
                'total': n,
                'improved': positive,
                'regressed': negative,
                'neutral': neutral
            }
        },
        'quality_metrics (weight 0.60)': {
            'structured_correctness': metric_details['structured_correctness'],
            'task_success': metric_details['task_success'],
            'instruction_following': metric_details['instruction_following'],
        },
        'fidelity_metrics (weight 0.40)': {
            'coverage': metric_details['coverage'],
            'faithfulness': metric_details['faithfulness'],
            'hallucination': metric_details['hallucination'],
            'context_grounding': metric_details['context_grounding'],
        },
        'diagnostic_metrics': {
            'conciseness': metric_details['conciseness'],
            'rouge1': metric_details['rouge1'],
            'rougel': metric_details['rougel'],
            'bleu': metric_details['bleu'],
        },
        'all_metrics_breakdown': metric_details
    }

    reports_dir = project_path('reports', 'incremental')
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, f'checkpoint_{checkpoint}_report.json')
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"📄 Report saved → {path}")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────────────────────────────────────

def run_all_steps(checkpoint: int) -> dict:
    """Execute every step in STATUS_FLOW order for a checkpoint."""
    print(f"\n{'=' * 60}")
    print(f"🚀 RUNNING ALL STEPS — Checkpoint {checkpoint} | Table: {TABLE_NAME}")
    print(f"{'=' * 60}")

    # Auto-initialize if any records have null/empty status
    try:
        null_res = get_supabase().table(TABLE_NAME) \
            .select('id', count='exact') \
            .eq('checkpoint', checkpoint) \
            .or_('status.is.null,status.eq.') \
            .limit(1).execute()
        if null_res.count and null_res.count > 0:
            print(f"Found {null_res.count:,} uninitialized records in checkpoint {checkpoint}. Running auto-init to set status='score'...")
            init_checkpoint(checkpoint)
    except Exception as e:
        print(f"Note on status check: {e}")

    step_fns = {
        'score':        step_score,
        'finetune':     step_finetune,
        'output_tuned': step_output_tuned,
        'score_tuned':  step_score_tuned,
        'completed':    step_completed,
    }

    results = {}
    for step in STATUS_FLOW:
        print(f"\n{'─' * 60}\n▶️  {step}\n{'─' * 60}")
        result = step_fns[step](checkpoint)
        results[step] = result
        if result is None or (isinstance(result, (int, float)) and result == 0):
            print(f"\n❌ Step '{step}' did not complete. Stopping pipeline.")
            break

    print(f"\n{'=' * 60}\n📋 SUMMARY — Checkpoint {checkpoint}\n{'=' * 60}")
    for step, result in results.items():
        print(f"   {'✅' if result else '❌'} {step}: {result}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global TABLE_NAME
    parser = argparse.ArgumentParser(description='Incremental Learning Pipeline')
    parser.add_argument('--checkpoint', type=int, required=True, help='Checkpoint number (1–10)')
    parser.add_argument('--step', type=str, choices=STATUS_FLOW + ['status'])
    parser.add_argument('--run-all', action='store_true')
    parser.add_argument('--init', action='store_true')
    parser.add_argument('--table', type=str, default=None, help=f'Supabase table name (default: {TABLE_NAME})')
    args = parser.parse_args()

    if args.table:
        TABLE_NAME = args.table

    if not 1 <= args.checkpoint <= 10:
        print("Error: --checkpoint must be 1–10"); sys.exit(1)

    if not any([args.step, args.run_all, args.init]):
        print("Error: specify --step, --run-all, or --init")
        parser.print_help(); sys.exit(1)

    cp = args.checkpoint
    print(f"\n{'=' * 60}\n🚀 INCREMENTAL LEARNING — Checkpoint {cp} | Table: {TABLE_NAME}\n{'=' * 60}")

    if args.init:
        init_checkpoint(cp); return
    if args.run_all:
        run_all_steps(cp); return

    step = args.step
    dispatch = {
        'status':       lambda: step_status(cp),
        'score':        lambda: step_score(cp),
        'finetune':     lambda: step_finetune(cp),
        'output_tuned': lambda: step_output_tuned(cp),
        'score_tuned':  lambda: step_score_tuned(cp),
        'completed':    lambda: step_completed(cp),
    }
    dispatch[step]()

    if step != 'status' and step in STATUS_FLOW:
        idx = STATUS_FLOW.index(step)
        if idx < len(STATUS_FLOW) - 1:
            print(f"\n   ▶️  Next: --checkpoint {cp} --step {STATUS_FLOW[idx + 1]}")
        else:
            print(f"\n   ✅ Checkpoint {cp} complete!")
            if cp < 10:
                print(f"   ▶️  Next: --checkpoint {cp + 1} --init")


if __name__ == '__main__':
    main()