"""
Step 1 (LAPTOP-OPTIMIZED): Generate base student outputs — tuned for RTX 4060 (8GB VRAM)
- GPU batch 8: safe for 8GB VRAM with 4-bit 1B model (~0.7GB model + ~1–2GB activations)
- Thermal throttle: pauses if GPU hits 80°C to protect the laptop
- VRAM guard: drops batch size live if memory pressure spikes
- Inter-batch cooldown: 300ms breathing room between batches
- Threaded DB writes: overlaps Supabase IO with next GPU batch

Usage:
  python experiment/11_gen_base_student.py --checkpoint 2
  python experiment/11_gen_base_student.py                # auto-detect
  python experiment/11_gen_base_student.py --gpu-batch 4  # if you see OOM
"""

import os
import sys
import time
import argparse
import threading
from postgrest.exceptions import APIError

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# Slightly reduces GPU power draw with no accuracy cost on Ampere/Ada
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from tqdm import tqdm
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME          = "unsloth/gemma-3-1b-it-bnb-4bit"
SCAN_PAGE_SIZE      = 500
FETCH_BATCH_SIZE    = 200
GPU_BATCH_SIZE      = 8       # Safe default for 8GB VRAM — do NOT raise above 12
MAX_NEW_TOKENS      = 256
RECORDS_PER_CKPT    = 5000

# Laptop thermal / memory guards
TEMP_CEILING_C      = 80      # Pause generation above this GPU temp
TEMP_RESUME_C       = 72      # Resume once cooled back to this
VRAM_HEADROOM_MB    = 512     # Always keep this free; halve batch if breached
INTER_BATCH_SLEEP_S = 0.3     # Breathing room between batches (seconds)


# ── GPU health helpers ─────────────────────────────────────────────────────────

def gpu_temp_c() -> int | None:
    """Returns current GPU temp in °C, or None if pynvml unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    except Exception:
        return None


def vram_free_mb() -> int:
    """Returns free VRAM in MB."""
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        return free // (1024 * 1024)
    return 9999


def thermal_check(pbar: tqdm):
    """Block until GPU cools below TEMP_RESUME_C. Shows warning in progress bar."""
    temp = gpu_temp_c()
    if temp is None or temp < TEMP_CEILING_C:
        return
    pbar.write(f"🌡️  GPU at {temp}°C — pausing until ≤{TEMP_RESUME_C}°C to protect laptop...")
    while True:
        time.sleep(5)
        temp = gpu_temp_c()
        if temp is None or temp <= TEMP_RESUME_C:
            pbar.write(f"✅ GPU cooled to {temp}°C — resuming.")
            break
        pbar.write(f"   Still {temp}°C, waiting...")


# ── DB helpers ─────────────────────────────────────────────────────────────────

def is_timeout_error(exc):
    msg = str(exc).lower()
    return (
        "statement timeout" in msg
        or "canceling statement due to statement timeout" in msg
        or "'code': '57014'" in msg
    )


def execute_with_retry(query_fn, op_name, max_retries=4, base_delay=1.5):
    for attempt in range(1, max_retries + 1):
        try:
            return query_fn().execute()
        except APIError as e:
            if not is_timeout_error(e) or attempt == max_retries:
                raise
            sleep_s = base_delay * attempt
            print(f"⚠️  {op_name} timeout (attempt {attempt}/{max_retries}), retry in {sleep_s:.1f}s...")
            time.sleep(sleep_s)


def bulk_upsert(supabase, rows: list[dict]):
    """
    Use UPDATE (not upsert) — rows already exist in the table.
    Upsert on partial columns would null-out 'input' and other required fields
    on any INSERT path, causing not-null constraint violations.
    """
    for row in rows:
        row_id = row["id"]
        payload = {k: v for k, v in row.items() if k != "id"}
        execute_with_retry(
            lambda rid=row_id, p=payload: supabase.table('modelcomp_50k')
                .update(p)
                .eq('id', rid),
            f"update row {row_id}",
        )


# ── Model ──────────────────────────────────────────────────────────────────────

def load_model():
    print(f"Loading {MODEL_NAME} (4-bit) — RTX 4060 mode...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    tokenizer.padding_side = "left"           # Required for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    free_mb = vram_free_mb()
    print(f"✅ Model loaded on {model.device} | VRAM free: {free_mb} MB")
    return model, tokenizer


def build_prompt(instruction: str, context: str = "") -> str:
    if context:
        return f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


# ── Batched inference (VRAM-aware) ─────────────────────────────────────────────

def generate_batch(
    model, tokenizer, records: list[dict], pbar: tqdm
) -> tuple[list[dict], int]:
    """
    Run a single batched forward pass.
    Returns (results_for_upsert, actual_batch_size_used).
    Automatically halves the batch if VRAM headroom is low.
    """
    # VRAM guard: if free memory is tight, shrink this batch
    if vram_free_mb() < VRAM_HEADROOM_MB and len(records) > 1:
        half = max(1, len(records) // 2)
        pbar.write(f"⚠️  Low VRAM ({vram_free_mb()} MB free) — shrinking batch to {half}")
        records = records[:half]

    prompts = [build_prompt(r['input'], r.get('context', '')) for r in records]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,                       # Greedy — faster, no quality loss here
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.time() - t0) * 1000
    per_rec_latency = latency_ms / len(records)

    prompt_len = inputs["input_ids"].shape[1]
    results = []
    for i, record in enumerate(records):
        new_tokens = outputs[i][prompt_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append({
            "id": record["id"],
            "student_output": response[:5000],
            "generation_latency": round(per_rec_latency, 3),
        })

    # Free activations immediately — important on 8GB
    del inputs, outputs
    torch.cuda.empty_cache()

    return results, len(records)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=int, default=None)
    parser.add_argument('--gpu-batch', type=int, default=GPU_BATCH_SIZE,
                        help='Records per GPU pass (default 8, max recommended 12 for 8GB VRAM)')
    args = parser.parse_args()

    gpu_batch = min(args.gpu_batch, 12)        # Hard cap — protects 8GB VRAM
    if args.gpu_batch > 12:
        print(f"⚠️  --gpu-batch capped at 12 for RTX 4060 (8GB VRAM). Using 12.")

    supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

    # ── Checkpoint selection ───────────────────────────────────────────────────
    if args.checkpoint:
        target_ckpt = args.checkpoint
        if not 1 <= target_ckpt <= 10:
            print("Error: --checkpoint must be 1-10"); sys.exit(1)
    else:
        print("Auto-detecting checkpoint...")
        for ckpt in range(1, 11):
            r = execute_with_retry(
                lambda: supabase.table('modelcomp_50k')
                    .select('id', count='exact')
                    .eq('checkpoint', ckpt)
                    .is_('student_output', 'null'),
                f"count pending ckpt {ckpt}",
            )
            if r.count > 0:
                target_ckpt = ckpt
                print(f"  Checkpoint {ckpt}: {r.count} records pending")
                break
        else:
            print("✅ All checkpoints complete!"); return

    # ── Status ─────────────────────────────────────────────────────────────────
    pending_r = execute_with_retry(
        lambda: supabase.table('modelcomp_50k')
            .select('id', count='exact')
            .eq('checkpoint', target_ckpt)
            .is_('student_output', 'null'),
        "count pending",
    )
    done_r = execute_with_retry(
        lambda: supabase.table('modelcomp_50k')
            .select('id', count='exact')
            .eq('checkpoint', target_ckpt)
            .not_.is_('student_output', 'null'),
        "count done",
    )
    pending_count = pending_r.count
    done_count    = done_r.count

    est_minutes = (pending_count / gpu_batch) * (MAX_NEW_TOKENS * 0.005 + INTER_BATCH_SLEEP_S) / 60

    print(f"\n{'='*62}")
    print(f"📋 CHECKPOINT {target_ckpt}  |  RTX 4060 laptop mode")
    print(f"{'='*62}")
    print(f"   Already done   : {done_count:,} / {RECORDS_PER_CKPT:,}")
    print(f"   Pending        : {pending_count:,}")
    print(f"   Progress       : {done_count/RECORDS_PER_CKPT*100:.1f}%")
    print(f"   GPU batch size : {gpu_batch}  (max 12 for 8GB VRAM)")
    print(f"   VRAM free      : {vram_free_mb()} MB")
    print(f"   GPU temp       : {gpu_temp_c() or 'N/A'} °C  (ceiling: {TEMP_CEILING_C}°C)")
    print(f"   Est. time      : ~{est_minutes:.0f} min (rough)")
    print(f"{'='*62}")

    if pending_count == 0:
        print(f"✅ Checkpoint {target_ckpt} already complete!"); return

    # ── Load model ─────────────────────────────────────────────────────────────
    model, tokenizer = load_model()

    # ── Generation loop ────────────────────────────────────────────────────────
    total_processed = 0
    last_seen_id    = 0
    pending_buffer: list[dict] = []
    db_thread: threading.Thread | None = None

    def flush_to_db(rows):
        bulk_upsert(supabase, rows)

    pbar = tqdm(total=pending_count, desc=f"Ckpt {target_ckpt}", unit="rec",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    while True:
        # ── Refill fetch buffer ────────────────────────────────────────────────
        while len(pending_buffer) < FETCH_BATCH_SIZE:
            page = execute_with_retry(
                lambda: supabase.table('modelcomp_50k')
                    .select('id, input, context, student_output')
                    .eq('checkpoint', target_ckpt)
                    .gt('id', last_seen_id)
                    .order('id')
                    .limit(SCAN_PAGE_SIZE),
                f"fetch page after id {last_seen_id}",
            )
            if not page.data:
                break
            last_seen_id = page.data[-1]['id']
            for row in page.data:
                if row.get('student_output') in (None, ''):
                    pending_buffer.append({
                        'id': row['id'],
                        'input': row['input'],
                        'context': row.get('context', ''),
                    })
            if len(page.data) < SCAN_PAGE_SIZE:
                break

        if not pending_buffer:
            break

        # ── Thermal check before each batch ───────────────────────────────────
        thermal_check(pbar)

        # ── GPU inference ──────────────────────────────────────────────────────
        gpu_records     = pending_buffer[:gpu_batch]
        pending_buffer  = pending_buffer[gpu_batch:]

        try:
            results, used = generate_batch(model, tokenizer, gpu_records, pbar)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            pbar.write("🔴 OOM — halving batch and retrying...")
            gpu_batch = max(1, gpu_batch // 2)
            pbar.write(f"   New GPU batch size: {gpu_batch}")
            # Put records back and retry next loop
            pending_buffer = gpu_records + pending_buffer
            time.sleep(2)
            continue

        total_processed += used
        pbar.update(used)

        # ── Async DB write ─────────────────────────────────────────────────────
        if db_thread and db_thread.is_alive():
            db_thread.join()
        db_thread = threading.Thread(target=flush_to_db, args=(results,), daemon=True)
        db_thread.start()

        # ── Inter-batch cooldown (keeps laptop thermals stable) ────────────────
        time.sleep(INTER_BATCH_SLEEP_S)

    # Final flush
    if db_thread and db_thread.is_alive():
        db_thread.join()

    pbar.close()
    torch.cuda.empty_cache()

    print(f"\n✅ Checkpoint {target_ckpt} done! Generated {total_processed} records")
    print(f"   Total: {done_count + total_processed}/{RECORDS_PER_CKPT}")

    if done_count + total_processed >= RECORDS_PER_CKPT:
        print(f"\n   🎯 Checkpoint {target_ckpt} READY for training!")
        print(f"   Run: python experiment/12_train_incremental.py --checkpoint {target_ckpt} --init")

    if target_ckpt < 10:
        print(f"\n   Next: python experiment/11_gen_base_student.py --checkpoint {target_ckpt + 1}")


if __name__ == "__main__":
    main()