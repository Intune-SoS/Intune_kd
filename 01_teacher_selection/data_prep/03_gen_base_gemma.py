"""
Step 3: Generate Gemma 3:1B outputs and save to Supabase
========================================================
- Generates Gemma outputs for all 4K samples
- Saves to Supabase table 'modelComp' with:
  id, input, context, actual_output (Gemma), sevenb (from Alpaca), twentyb (null for now)

Usage:
    python experiment/03_generate_gemma_save_supabase.py
    python experiment/03_generate_gemma_save_supabase.py --model gemma3:1b
"""

import json
import time
import argparse
import requests
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

DATA_DIR = PROJECT_ROOT / "data" / "experiment"

# Ollama config
OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:1b"
TEMPERATURE = 0.7
MAX_TOKENS = 512
TIMEOUT = 120
BATCH_SIZE = 10  # Save to Supabase every N records


def get_supabase_client():
    """Get Supabase client"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    return create_client(url, key)


def check_ollama():
    """Check if Ollama is running"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [m.get("name", "") for m in models]
    except requests.RequestException:
        return None


def generate_response(prompt: str, model: str) -> str:
    """Generate response using Ollama"""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": TEMPERATURE,
                    "num_predict": MAX_TOKENS
                }
            },
            timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.RequestException as e:
        print(f"\n  ✗ Error: {e}")
        return None


def get_existing_inputs(supabase) -> set:
    """Get inputs already in modelComp table (to avoid duplicates)"""
    try:
        response = supabase.table("modelComp").select("input").execute()
        return set(r["input"] for r in response.data)
    except Exception as e:
        print(f"  Warning: Could not fetch existing records: {e}")
        return set()


def insert_batch_to_supabase(supabase, records: list):
    """Insert batch of records to Supabase"""
    try:
        supabase.table("modelComp").insert(records).execute()
        return True
    except Exception as e:
        print(f"\n  ✗ Supabase error: {e}")
        return False


def main(model: str = DEFAULT_MODEL):
    """Generate Gemma outputs and save to Supabase"""
    
    input_path = DATA_DIR / "experiment_4k.json"
    
    print("=" * 60)
    print("STEP 3: GENERATE GEMMA → SAVE TO SUPABASE")
    print("=" * 60)
    
    # Check prerequisites
    if not input_path.exists():
        print(f"✗ Dataset not found: {input_path}")
        print("  Run Step 2 first: python experiment/02_prepare_4k_dataset.py")
        return
    
    # Check Ollama
    available_models = check_ollama()
    if available_models is None:
        print("✗ Ollama is not running!")
        print("  Start Ollama with: ollama serve")
        return
    
    print(f"✓ Ollama is running")
    
    # Check model
    model_found = any(model.split(":")[0] in m for m in available_models)
    if not model_found:
        print(f"\n✗ Model '{model}' not found!")
        print(f"  Pull it with: ollama pull {model}")
        return
    
    print(f"✓ Using model: {model}")
    
    # Connect to Supabase
    try:
        supabase = get_supabase_client()
        print("✓ Connected to Supabase")
    except Exception as e:
        print(f"✗ Supabase connection failed: {e}")
        return
    
    # Load data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} samples")
    
    # Get existing inputs to skip (avoid duplicates)
    existing_inputs = get_existing_inputs(supabase)
    print(f"✓ Found {len(existing_inputs)} existing records in modelComp")
    
    # Filter to process
    to_process = [d for d in data if d["instruction"] not in existing_inputs]
    print(f"✓ Will process {len(to_process)} new records")
    
    if not to_process:
        print("\n✓ All records already in Supabase!")
        return
    
    print(f"\nGenerating Gemma outputs...")
    print("-" * 60)
    
    start_time = time.time()
    batch = []
    success_count = 0
    fail_count = 0
    
    for sample in tqdm(to_process, desc="Gemma → Supabase"):
        # Generate Gemma output
        gemma_output = generate_response(sample["input"], model)
        
        if gemma_output:
            # Prepare record for Supabase (id is auto-generated UUID)
            record = {
                "input": sample["instruction"],  # Original instruction
                "context": sample["context"] if sample["context"] else None,
                "actual_output": gemma_output,  # Gemma output
                "sevenb": sample["alpaca_output"],  # LLaMA 7B from Alpaca
                "twentyb": None  # Will be filled by GPT-OSS step
            }
            batch.append(record)
            success_count += 1
        else:
            fail_count += 1
        
        # Save batch to Supabase
        if len(batch) >= BATCH_SIZE:
            if insert_batch_to_supabase(supabase, batch):
                tqdm.write(f"  💾 Saved batch of {len(batch)} to Supabase")
            batch = []
    
    # Save remaining batch
    if batch:
        insert_batch_to_supabase(supabase, batch)
        print(f"  💾 Saved final batch of {len(batch)}")
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"✓ Success: {success_count}")
    print(f"✗ Failed: {fail_count}")
    print(f"⏱  Time: {elapsed/60:.1f} minutes")
    print(f"⚡ Rate: {success_count/elapsed*60:.1f} samples/min")
    print("=" * 60)
    print("\n→ Next: Run 04_generate_gpt_oss_supabase.py on MacBook")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Gemma outputs → Supabase")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                       help=f"Ollama model name (default: {DEFAULT_MODEL})")
    args = parser.parse_args()
    main(model=args.model)
