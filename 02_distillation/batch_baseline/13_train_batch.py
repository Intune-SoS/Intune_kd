#!/usr/bin/env python3
"""
Batch Training Pipeline - RTX 4060 Safe (8GB VRAM)
====================================================
Fixes:
  - Blank terminal: heartbeat thread + lazy imports (no top-level unsloth)
  - VRAM OOM: expandable allocator, batch=1, grad_accum=16, no packing,
               use_reentrant=False, decode only new tokens
  - API hang: bulk DB updates (.in_() batched), no row-by-row loops
  - Trainer: built ONCE, not recreated per chunk

Status flow:  score -> finetune -> output_tuned -> score_tuned -> completed
"""

# ── stdlib only at top level (fast, no CUDA init) ────────────────────────────
import os, sys, json, time, gc, argparse, threading
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
eval_dir = os.path.join(project_root, "02_distillation", "evaluation_suite")
if eval_dir not in sys.path:
    sys.path.insert(0, eval_dir)

# ── MUST be set before torch import ──────────────────────────────────────────
os.environ["TORCHDYNAMO_DISABLE"]     = "1"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
os.environ["CUDA_LAUNCH_BLOCKING"]    = "0"
os.environ["TORCHINDUCTOR_DISABLE"]   = "1"
os.environ["TORCH_COMPILE_DISABLE"]   = "1"
os.environ["TRITON_DISABLE"]          = "1"
os.environ["OMP_NUM_THREADS"]         = "1"
os.environ["MKL_NUM_THREADS"]         = "1"
# FIX: expandable_segments is the correct modern OOM fix (not max_split_size_mb)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME       = "unsloth/gemma-3-1b-it-bnb-4bit"
MAX_SEQ_LENGTH   = 2048
MAX_NEW_TOKENS   = 512
MIN_RECORDS      = 100
BATCH_SIZE       = 1       # RTX 4060 8GB — do not increase
GRAD_ACCUM       = 16      # effective batch = 16
STREAM_SIZE      = 1000    # Larger chunks for DB fetches (reduced round-trips)
DB_CHUNK         = 500     # Increased from 100 for faster bulk updates
INFERENCE_CHUNK  = 16      # records between GPU purges during inference
TUNED_INFER_BATCH_SIZE = 50
TUNED_SCAN_PAGE_SIZE   = 500   # Increased from 250 to reduce pagination
TUNED_GPU_BATCH_SIZE   = int(os.getenv('TUNED_GPU_BATCH_SIZE', '12'))
LORA_R           = 16
LORA_ALPHA       = 16
LORA_DROPOUT     = 0
BATCH_MODEL_PATH  = "models/gemma-batch-lora"
BATCH_REPORT_PATH = "reports/batch_learning"


# =============================================================================
# UTILITIES  (no torch yet — stays fast for --status)
# =============================================================================

def _flush(*a):
    print(*a, flush=True)

def _heartbeat(label: str, interval: int = 20):
    """Daemon thread: prints still-alive every interval seconds so terminal is never blank."""
    start = time.time()
    def _run():
        while True:
            time.sleep(interval)
            elapsed = (time.time() - start) / 60
            _flush(f"  [heartbeat:{label}] still running — {elapsed:.1f} min elapsed")
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t

def _import_torch():
    import torch
    return torch

def purge_gpu():
    torch = _import_torch()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def safe_delete(*objs):
    for o in objs:
        try: del o
        except Exception: pass
    purge_gpu()

def gpu_info():
    torch = _import_torch()
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    alloc = torch.cuda.memory_allocated() / 1e9
    res   = torch.cuda.memory_reserved()  / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free  = total - res
    return f"GPU {alloc:.1f}GB alloc | {res:.1f}GB reserved | {free:.1f}GB free / {total:.1f}GB"


# =============================================================================
# DATABASE
# =============================================================================

def get_supabase():
    from supabase import create_client
    return create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

def count_by_status(sb):
    counts = {}
    for s in ['score','finetune','output_tuned','score_tuned','completed']:
        counts[s] = sb.table('modelcomp_batch').select('id', count='exact').eq('status', s).execute().count
    counts['null'] = sb.table('modelcomp_batch').select('id', count='exact').is_('status','null').execute().count
    return counts

def fetch_all_by_status(sb, status):
    _flush(f"[DB] Fetching status='{status}'...")
    rows, offset, chunk = [], 0, STREAM_SIZE
    while True:
        r = sb.table('modelcomp_batch').select('*').eq('status', status)\
              .range(offset, offset + chunk - 1).execute()
        if not r.data: break
        rows.extend(r.data)
        offset += chunk
        if len(r.data) < chunk: break
    _flush(f"[DB] Got {len(rows)} rows")
    return rows

def bulk_update_status(sb, ids, new_status):
    """FIX: batch .in_() updates — not one HTTP round-trip per row."""
    _flush(f"[DB] Bulk-updating {len(ids)} rows to '{new_status}'...")
    for i in range(0, len(ids), DB_CHUNK):
        chunk = ids[i:i+DB_CHUNK]
        try:
            sb.table('modelcomp_batch').update({'status': new_status}).in_('id', chunk).execute()
        except Exception as e:
            _flush(f"[WARN] bulk chunk {i//DB_CHUNK} failed ({e}), retrying row-by-row")
            for rid in chunk:
                try:
                    sb.table('modelcomp_batch').update({'status': new_status}).eq('id', rid).execute()
                except Exception as e2:
                    _flush(f"[ERROR] row {rid}: {e2}")

def _default_metrics():
    return {k: 0.5 for k in ['structured_correctness','task_success','instruction_following',
                               'coverage','faithfulness','hallucination','context_grounding',
                               'overall_score','conciseness']}

def _get_eval():
    try:
        from importlib import import_module
        m = import_module('06_eval_metrics')
        return m.evaluate_single_output
    except Exception:
        return None

def _compute_metrics(fn, instruction, student, teacher, context, task_label):
    if fn:
        try:
            return fn(instruction=instruction, student_output=student,
                      teacher_output=teacher, context=context, task_label=task_label)
        except Exception:
            pass
    return _default_metrics()

def _rouge_bleu(rouge, smooth, teacher, student):
    from nltk.translate.bleu_score import sentence_bleu
    rs = rouge.score(teacher, student)
    try:
        bleu = sentence_bleu([teacher.split()], student.split(), smoothing_function=smooth)
    except Exception:
        bleu = 0.0
    return rs['rouge1'].fmeasure, rs['rougeL'].fmeasure, bleu


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


def generate_output_batch(model, tokenizer, records: list) -> tuple[list, list]:
    """
    Run batched inference and return (outputs, per-row-latency-ms).
    """
    torch = _import_torch()
    prompts = [_build_prompt(r['input'], r.get('context', '') or '') for r in records]

    inputs = tokenizer(
        prompts,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    batch_latency_ms = (time.time() - t0) * 1000

    input_lengths = inputs['attention_mask'].sum(dim=1).tolist()
    outputs = []
    for i, seq in enumerate(out):
        response_tokens = seq[int(input_lengths[i]):]
        text = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        if not text:
            text = _extract_response(tokenizer.decode(seq, skip_special_tokens=True))
        outputs.append(text)

    per_row_latency = [batch_latency_ms / max(1, len(records))] * len(records)
    return outputs, per_row_latency


def generate_output_batch_adaptive(model, tokenizer, records: list, max_gpu_batch: int, desc: str) -> tuple[list, list]:
    """
    Adaptive micro-batching: shrink on CUDA OOM and continue.
    """
    from tqdm import tqdm
    torch = _import_torch()

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
                    _flush(f"⚠️ CUDA OOM at batch={take}; retrying with batch={curr_bs}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise
    finally:
        pbar.close()

    return outputs, latencies


def bulk_update_rows(sb, rows: list) -> None:
    """
    Update rows in DB efficiently with automatic retry and progress tracking.
    Uses batch grouping to reduce total requests.
    """
    if not rows:
        return
    
    from tqdm import tqdm
    import time
    
    failed_rows = []
    
    with tqdm(total=len(rows), desc="DB update", unit='rows', disable=False) as pbar:
        for i in range(0, len(rows), DB_CHUNK):
            chunk = rows[i:i + DB_CHUNK]
            
            for attempt in range(2):  # Retry once on failure
                try:
                    for row in chunk:
                        rid = row['id']
                        payload = {k: v for k, v in row.items() if k != 'id'}
                        if payload:
                            sb.table('modelcomp_batch').update(payload).eq('id', rid).execute()
                        pbar.update(1)
                    break  # Success, move to next chunk
                except Exception as e:
                    if attempt == 0:
                        _flush(f"[RETRY] Batch {i//DB_CHUNK}: {str(e)[:60]}...")
                        time.sleep(0.5)  # Brief pause before retry
                    else:
                        _flush(f"[ERROR] Batch {i//DB_CHUNK} failed after 2 attempts")
                        failed_rows.extend([r['id'] for r in chunk])
                        pbar.update(len(chunk))
    
    if failed_rows:
        _flush(f"[WARN] {len(failed_rows)} rows failed to update: {failed_rows[:10]}...")


# =============================================================================
# STEP: STATUS
# =============================================================================

def step_status(sb):
    _flush("\n" + "="*60)
    _flush("STATUS OVERVIEW")
    _flush("="*60)
    c = count_by_status(sb)
    for k in ['null','score','finetune','output_tuned','score_tuned','completed']:
        _flush(f"  {k:<22} {c.get(k,0):,}")
    _flush("="*60)


# =============================================================================
# STEP: SCORE  (base outputs)
# =============================================================================

def step_score(sb):
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import SmoothingFunction
    from tqdm import tqdm
    _flush("\n[SCORE] Scoring base student outputs...")
    rows = []
    offset, chunk = 0, STREAM_SIZE
    while True:
        r = sb.table('modelcomp_batch').select('*').is_('status','null')\
              .neq('student_output','').not_.is_('student_output','null')\
              .range(offset, offset+chunk-1).execute()
        if not r.data: break
        rows.extend(r.data); offset += chunk
        if len(r.data) < chunk: break
    _flush(f"[SCORE] {len(rows)} records to score")
    if len(rows) < MIN_RECORDS:
        _flush("[SKIP] not enough"); return False

    eval_fn = _get_eval()
    rouge   = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)
    smooth  = SmoothingFunction().method1
    
    # Batch updates instead of per-row
    updates = []
    for rec in tqdm(rows, desc="Scoring"):
        try:
            ins     = rec.get('input','')
            teacher = rec.get('sevenb','')
            student = rec.get('student_output','')
            ctx     = rec.get('context','') or ''
            task    = rec.get('task_label','general_qa')
            if not student or not teacher: continue
            m = _compute_metrics(eval_fn, ins, student, teacher, ctx, task)
            r1, rl, bleu = _rouge_bleu(rouge, smooth, teacher, student)
            updates.append({
                'id': rec['id'],
                'score': m.get('overall_score',0.5),
                'structured_correctness': m.get('structured_correctness',0.5),
                'task_success': m.get('task_success',0.5),
                'instruction_following': m.get('instruction_following',0.5),
                'coverage': m.get('coverage',0.5),
                'faithfulness': m.get('faithfulness',0.5),
                'hallucination': m.get('hallucination',0.5),
                'context_grounding': m.get('context_grounding',0.5),
                'conciseness': m.get('conciseness',0.5),
                'rouge1': r1, 'rougel': rl, 'bleu': bleu, 'status': 'score',
            })
        except Exception as e:
            _flush(f"[ERROR] score {rec['id']}: {e}")
    
    # Bulk update all at once
    bulk_update_rows(sb, updates)
    _flush(f"[SCORE] Done: {len(updates)}/{len(rows)}")
    return True


# =============================================================================
# STEP: FINETUNE
# =============================================================================

def step_finetune(sb):
    # FIX: ALL heavy imports inside function.
    # Top-level unsloth import triggers CUDA init + cache checks which hang
    # for 40+ min before any print appears.
    import torch
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset
    from tqdm import tqdm

    try:
        import torch._dynamo as _dynamo
        _dynamo.config.disable = True
        _dynamo.config.suppress_errors = True
    except Exception:
        pass

    _flush("\n" + "="*60)
    _flush("FINETUNE — RTX 4060 safe mode")
    _flush("="*60)

    # Heartbeat keeps terminal alive during model load (biggest blank-screen cause)
    hb = _heartbeat("finetune", interval=20)

    _flush("[1] Fetching records...")
    records = fetch_all_by_status(sb, 'score')
    if len(records) < MIN_RECORDS:
        _flush(f"[SKIP] only {len(records)} records"); return False

    _flush(f"[2] Formatting {len(records):,} samples...")
    training_data = []
    for item in tqdm(records, desc="Format"):
        ctx = item.get('context','') or ''
        if ctx:
            text = (f"### Instruction:\n{item['input']}\n\n"
                    f"### Context:\n{ctx}\n\n"
                    f"### Response:\n{item['sevenb']}")
        else:
            text = (f"### Instruction:\n{item['input']}\n\n"
                    f"### Response:\n{item['sevenb']}")
        training_data.append({"text": text})
    dataset = Dataset.from_list(training_data)
    _flush(f"[2] Dataset ready: {len(dataset)} samples")

    _flush(f"[3] {gpu_info()}")
    purge_gpu()

    # ── Load model ────────────────────────────────────────────────────────────
    _flush(f"[4] Loading {MODEL_NAME}  <-- may take 3-6 min, heartbeat prints every 20s")
    model = tokenizer = None
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        _flush(f"[4] Model loaded  {gpu_info()}")
    except Exception as e:
        _flush(f"[ERROR] load failed: {e}")
        safe_delete(model, tokenizer)
        return False

    # ── LoRA ──────────────────────────────────────────────────────────────────
    _flush("[5] Adding LoRA...")
    try:
        model = FastLanguageModel.get_peft_model(
            model, r=LORA_R,
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        _flush("[5] LoRA ready")
    except Exception as e:
        _flush(f"[ERROR] LoRA: {e}")
        safe_delete(model, tokenizer)
        return False

    # ── Pre-tokenize BEFORE SFTTrainer (fixes UnslothSFTTrainer multiprocess crash) ──
    # Unsloth patches SFTTrainer to spawn up to 24 worker processes for tokenization.
    # Those workers can't unpickle UnslothSFTTrainer → ModuleNotFoundError crash.
    # Solution: tokenize the dataset ourselves with num_proc=1 first, then pass
    # it as already-tokenized so SFTTrainer skips its internal tokenization workers.
    _flush("[6a] Pre-tokenizing dataset (single process, avoids worker crash)...")
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )
    dataset = dataset.map(tokenize_fn, batched=True, num_proc=1, remove_columns=["text"])
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    _flush(f"[6a] Tokenized: {len(dataset)} samples")

    # ── Trainer built ONCE (not recreated per chunk) ──────────────────────────
    os.makedirs(BATCH_MODEL_PATH, exist_ok=True)
    _flush("[6] Building trainer...")
    try:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            # dataset_text_field omitted — dataset is already tokenized
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_num_proc=1,
            packing=False,          # FIX: packing causes unpredictable VRAM spikes
            args=TrainingArguments(
                per_device_train_batch_size=BATCH_SIZE,    # 1 = safe for 8GB VRAM
                gradient_accumulation_steps=GRAD_ACCUM,   # effective batch = 16
                warmup_steps=50,
                num_train_epochs=1,
                learning_rate=2e-4,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=10,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                seed=42,
                output_dir=BATCH_MODEL_PATH,
                save_strategy="epoch",
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},  # FIX: prevents backward hang
                max_grad_norm=1.0,
                dataloader_num_workers=0,     # FIX: workers deadlock on laptop/Windows
                dataloader_pin_memory=False,  # FIX: reduces host RAM pressure
                report_to=[],                 # FIX: no wandb/tb network call stall
                torch_compile=False,          # FIX: avoid inductor/triton on Windows
                # NOTE: optim_target_modules intentionally OMITTED
                # — causes silent hang with some trl/unsloth combos
            ),
        )
        _flush("[6] Trainer ready — starting training now...")
    except Exception as e:
        _flush(f"[ERROR] trainer init: {e}")
        safe_delete(model, tokenizer)
        return False

    t0 = time.time()
    try:
        trainer.train()
    except Exception as e:
        _flush(f"[ERROR] training: {e}")
        safe_delete(model, trainer, tokenizer)
        return False
    elapsed = time.time() - t0
    _flush(f"[6] Training done in {elapsed/60:.1f} min  {gpu_info()}")

    _flush("[7] Saving model...")
    model.save_pretrained(BATCH_MODEL_PATH)
    tokenizer.save_pretrained(BATCH_MODEL_PATH)
    _flush(f"[7] Saved to {BATCH_MODEL_PATH}")

    safe_delete(model, trainer, tokenizer)

    bulk_update_status(sb, [r['id'] for r in records], 'finetune')

    os.makedirs(BATCH_REPORT_PATH, exist_ok=True)
    with open(f"{BATCH_REPORT_PATH}/finetune_report.json",'w') as f:
        json.dump({
            "records": len(records),
            "train_min": elapsed/60,
            "model_path": BATCH_MODEL_PATH,
            "ts": datetime.now().isoformat()
        }, f, indent=2)
    _flush("[SUCCESS] Finetune complete!")
    return True


# =============================================================================
# STEP: OUTPUT_TUNED
# =============================================================================

def step_output_tuned(sb):
    from unsloth import FastLanguageModel
    torch = _import_torch()

    _flush("\n" + "="*60)
    _flush("OUTPUT_TUNED — generating with fine-tuned model")
    _flush("="*60)

    hb = _heartbeat("output_tuned", interval=20)

    pending_count = sb.table('modelcomp_batch') \
        .select('id', count='exact') \
        .eq('status', 'finetune') \
        .execute().count or 0
    if pending_count == 0:
        _flush("[SKIP] No rows with status='finetune'")
        return False

    _flush(f"[1] {gpu_info()}")
    purge_gpu()

    _flush(f"[2] Loading fine-tuned model from {BATCH_MODEL_PATH}  <-- heartbeat active")
    model = tokenizer = None
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BATCH_MODEL_PATH,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        _flush(f"[2] Model loaded  {gpu_info()}")
    except Exception as e:
        _flush(f"[ERROR] load: {e}")
        safe_delete(model, tokenizer)
        return False

    processed = 0
    last_seen_id = 0
    pending_buffer = []

    while True:
        while len(pending_buffer) < TUNED_INFER_BATCH_SIZE:
            page = sb.table('modelcomp_batch') \
                .select('id, input, context') \
                .eq('status', 'finetune') \
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
        current_batch.sort(key=_prompt_len_proxy)

        try:
            outputs, latencies = generate_output_batch_adaptive(
                model=model,
                tokenizer=tokenizer,
                records=current_batch,
                max_gpu_batch=max(1, TUNED_GPU_BATCH_SIZE),
                desc="Generating tuned outputs",
            )
        except Exception as e:
            _flush(f"[ERROR] batch generation at id>{last_seen_id}: {e}")
            purge_gpu()
            continue

        rows_to_update = []
        for item, output, latency in zip(current_batch, outputs, latencies):
            rows_to_update.append({
                'id': item['id'],
                'student_output_tuned': output[:5000],
                'latency_tuned': round(latency, 3),
                'status': 'output_tuned',
            })

        try:
            bulk_update_rows(sb, rows_to_update)
            processed += len(rows_to_update)
        except Exception as e:
            _flush(f"[ERROR] DB update failed: {e}")

        _flush(f"BATCH output_tuned progress: {processed}/{pending_count}")
        # Only purge every 3 batches to reduce overhead
        if processed % (TUNED_INFER_BATCH_SIZE * 3) == 0:
            purge_gpu()

    safe_delete(model, tokenizer)
    _flush(f"[SUCCESS] Generated {processed}/{pending_count}")
    return True


# =============================================================================
# STEP: SCORE_TUNED
# =============================================================================

def step_score_tuned(sb):
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import SmoothingFunction
    from tqdm import tqdm
    _flush("\n[SCORE_TUNED] Scoring tuned outputs...")
    records = fetch_all_by_status(sb, 'output_tuned')
    if len(records) == 0:
        _flush(f"[SKIP] no records to score"); return False

    eval_fn = _get_eval()
    rouge   = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)
    smooth  = SmoothingFunction().method1
    
    # Batch updates instead of per-row
    updates = []
    for rec in tqdm(records, desc="Score tuned"):
        try:
            ins     = rec.get('input','')
            teacher = rec.get('sevenb','')
            student = rec.get('student_output_tuned','')
            ctx     = rec.get('context','') or ''
            task    = rec.get('task_label','general_qa')
            if not student:
                # Missing output_tuned — mark with defaults and skip
                updates.append({
                    'id': rec['id'],
                    'score_tuned': 0.5,
                    'structured_correctness_tuned': 0.5,
                    'task_success_tuned': 0.5,
                    'instruction_following_tuned': 0.5,
                    'coverage_tuned': 0.5,
                    'faithfulness_tuned': 0.5,
                    'hallucination_tuned': 0.5,
                    'context_grounding_tuned': 0.5,
                    'conciseness_tuned': 0.5,
                    'rouge1_tuned': 0.5, 'rougel_tuned': 0.5, 'bleu_tuned': 0.5,
                    'status': 'score_tuned',
                })
                continue
            m = _compute_metrics(eval_fn, ins, student, teacher, ctx, task)
            r1, rl, bleu = _rouge_bleu(rouge, smooth, teacher, student)
            updates.append({
                'id': rec['id'],
                'score_tuned': m.get('overall_score',0.5),
                'structured_correctness_tuned': m.get('structured_correctness',0.5),
                'task_success_tuned': m.get('task_success',0.5),
                'instruction_following_tuned': m.get('instruction_following',0.5),
                'coverage_tuned': m.get('coverage',0.5),
                'faithfulness_tuned': m.get('faithfulness',0.5),
                'hallucination_tuned': m.get('hallucination',0.5),
                'context_grounding_tuned': m.get('context_grounding',0.5),
                'conciseness_tuned': m.get('conciseness',0.5),
                'rouge1_tuned': r1, 'rougel_tuned': rl, 'bleu_tuned': bleu,
                'status': 'score_tuned',
            })
        except Exception as e:
            _flush(f"[ERROR] score_tuned {rec['id']}: {e}")
    
    # Bulk update all at once
    bulk_update_rows(sb, updates)
    _flush(f"[SCORE_TUNED] Done: {len(updates)}/{len(records)}")
    return True


# =============================================================================
# STEP: COMPLETED
# =============================================================================

def step_completed(sb):
    from tqdm import tqdm
    _flush("\n[COMPLETED] Calculating improvements...")
    records = fetch_all_by_status(sb, 'score_tuned')
    if len(records) < MIN_RECORDS:
        _flush(f"[SKIP] only {len(records)}"); return False

    improvements = []
    updates = []
    for rec in tqdm(records, desc="Finalising"):
        try:
            imp = (rec.get('score_tuned',0) or 0) - (rec.get('score',0) or 0)
            updates.append({
                'id': rec['id'],
                'improvement': imp,
                'status': 'completed'
            })
            improvements.append(imp)
        except Exception as e:
            _flush(f"[ERROR] completed {rec['id']}: {e}")

    # Bulk update all at once
    bulk_update_rows(sb, updates)
    
    if improvements:
        avg = sum(improvements) / len(improvements)
        pos = sum(1 for i in improvements if i > 0)
        neg = sum(1 for i in improvements if i < 0)
        _flush(f"\nRESULTS: {len(improvements):,} records | avg improvement {avg:.4f}")
        _flush(f"  Improved: {pos:,}  Degraded: {neg:,}")
        os.makedirs(BATCH_REPORT_PATH, exist_ok=True)
        with open(f"{BATCH_REPORT_PATH}/results.json",'w') as f:
            json.dump({
                "total": len(improvements), "avg_improvement": avg,
                "improved": pos, "degraded": neg,
                "ts": datetime.now().isoformat()
            }, f, indent=2)
    _flush("[SUCCESS] Pipeline complete!")
    return True


# =============================================================================
# PREFLIGHT
# =============================================================================

def preflight(sb):
    import torch
    _flush("\n" + "="*60)
    _flush("PREFLIGHT")
    _flush("="*60)
    _flush(f"PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        _flush(f"  {p.name}  {p.total_memory/1e9:.1f}GB VRAM")
    try:
        r = sb.table('modelcomp_batch').select('id', count='exact').limit(1).execute()
        _flush(f"  Supabase OK — {r.count} total rows")
    except Exception as e:
        _flush(f"  Supabase FAIL: {e}"); return False
    counts = count_by_status(sb)
    for k, v in counts.items():
        _flush(f"  {k:<22} {v}")
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(MODEL_NAME)
        _flush("  Model accessible")
    except Exception as e:
        _flush(f"  Model FAIL: {e}"); return False
    _flush("[PREFLIGHT PASSED]")
    return True


# =============================================================================
# MAIN
# =============================================================================

STEPS = {
    'score':        step_score,
    'finetune':     step_finetune,
    'output_tuned': step_output_tuned,
    'score_tuned':  step_score_tuned,
    'completed':    step_completed,
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--status',    action='store_true')
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--step',      choices=list(STEPS.keys()))
    p.add_argument('--run-all',   action='store_true')
    args = p.parse_args()

    _flush("[INIT] Connecting to Supabase...")
    sb = get_supabase()
    _flush("[INIT] Connected!")

    if args.preflight:
        preflight(sb); return
    if args.status or not (args.step or args.run_all):
        step_status(sb); return
    if not preflight(sb):
        _flush("[ABORT] Fix preflight issues first."); return
    if args.run_all:
        for name, fn in STEPS.items():
            _flush(f"\n>>> {name}")
            if not fn(sb):
                _flush(f"Stopped at {name}"); break
        return
    STEPS[args.step](sb)

if __name__ == "__main__":
    main()