#!/usr/bin/env python3
"""
Quick diagnostic to find where step_finetune is hanging.
Run this to isolate the issue.
"""
import os
import sys
import time
import torch
from datetime import datetime

def log(msg):
    """Print with timestamp"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()

log("="*70)
log("DIAGNOSTIC: Finding hang in step_finetune")
log("="*70)

# 1. Check PyTorch/CUDA
log("\n[1/5] Testing PyTorch/CUDA...")
try:
    log(f"  PyTorch: {torch.__version__}")
    log(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"  GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        log(f"  GPU Memory: {mem_gb:.1f} GB")
    log("  ✓ PyTorch OK")
except Exception as e:
    log(f"  ✗ PyTorch ERROR: {e}")
    sys.exit(1)

# 2. Test import
log("\n[2/5] Testing imports...")
try:
    print("  - torch...", end=" ", flush=True)
    import torch
    print("✓")
    
    print("  - transformers...", end=" ", flush=True)
    from transformers import AutoTokenizer
    print("✓")
    
    print("  - unsloth...", end=" ", flush=True)
    from unsloth import FastLanguageModel
    print("✓")
    
    print("  - unsloth_zoo...", end=" ", flush=True)
    import unsloth_zoo
    print("✓")
    
    log("  ✓ All imports OK")
except Exception as e:
    log(f"  ✗ Import ERROR: {e}")
    log(f"    This is likely the culprit. Full traceback:")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. Test model availability
log("\n[3/5] Testing model accessibility...")
try:
    MODEL_NAME = "unsloth/gemma-3-1b-it-bnb-4bit"
    log(f"  Model: {MODEL_NAME}")
    print("  Downloading metadata...", end=" ", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("✓")
    log("  ✓ Model accessible")
    del tokenizer
except Exception as e:
    log(f"  ✗ Model ERROR: {e}")
    log("  Fix: Check internet, HuggingFace token, model exists")
    sys.exit(1)

# 4. Test database
log("\n[4/5] Testing Supabase connection...")
try:
    from dotenv import load_dotenv
    from supabase import create_client
    
    load_dotenv()
    print("  Connecting...", end=" ", flush=True)
    
    supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
    print("✓")
    
    print("  Querying records...", end=" ", flush=True)
    
    result = supabase.table('modelcomp_batch').select('id', count='exact').limit(1).execute()
    print(f"✓")
    log(f"  Total records in DB: {result.count}")
    log("  ✓ Database OK")
except Exception as e:
    log(f"  ✗ Database ERROR: {e}")
    log("  Fix: Check SUPABASE_URL and SUPABASE_KEY in .env")
    sys.exit(1)

# 5. Test model loading (THE REAL TEST)
log("\n[5/5] Testing actual model loading (THIS COULD HANG)...")
log("  ⚠️  If this hangs, press Ctrl+C")
log("  Starting model load...")
sys.stdout.flush()

import threading

load_done = False
load_error = None
model = tokenizer = None

def load_with_monitor():
    global model, tokenizer, load_done, load_error
    try:
        print("    [Loading...] Starting FastLanguageModel.from_pretrained()")
        sys.stdout.flush()
        start = time.time()
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/gemma-3-1b-it-bnb-4bit",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        elapsed = time.time() - start
        print(f"    [Loaded] took {elapsed:.1f}s")
        load_done = True
    except Exception as e:
        load_error = str(e)
        load_done = True

# Start in background thread
t = threading.Thread(target=load_with_monitor, daemon=True)
t.start()

# Monitor with timeout and heartbeat
elapsed = 0
timeout = 600  # 10 minute timeout
heartbeat_interval = 30

last_beat = time.time()
while not load_done and elapsed < timeout:
    time.sleep(5)
    elapsed += 5
    
    now = time.time()
    if (now - last_beat) > heartbeat_interval:
        print(f"    ... {elapsed}s elapsed, still loading", flush=True)
        last_beat = now

if load_error:
    log(f"  ✗ Model load ERROR: {load_error}")
    sys.exit(1)

if not load_done:
    log(f"  ✗ Model load TIMEOUT after {elapsed}s")
    log("  Your GPU may not have enough VRAM or quantization is very slow")
    sys.exit(1)

log(f"  ✓ Model loaded successfully")

log("\n" + "="*70)
log("✓ ALL DIAGNOSTICS PASSED - Your setup is working!")
log("="*70)
print("\nNow you can run:")
print("  python 13_train_batch.py --step finetune")
print("\nIf it still hangs, it's likely:")
print("  1. Database query taking forever (check status)")
print("  2. Dataset formatting is slow (check record count)")
print("  3. Trainer initialization (check memory)")
