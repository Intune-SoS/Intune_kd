"""
Step 4: Generate GPT-OSS 20B outputs and update Supabase
========================================================
- Fetches records from Supabase modelComp where twentyb is NULL
- Generates GPT-OSS 20B outputs via MacBook Pro
- Updates twentyb column in Supabase

⚠️ Runs on MacBook Pro (24GB) via network!

Usage:
    python experiment/04_generate_gpt_oss_supabase.py --ip 192.168.1.100
    python experiment/04_generate_gpt_oss_supabase.py --ip 192.168.1.100 --model gpt-neox:20b
"""

import time
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# MacBook Ollama config
OLLAMA_PORT = 11434
DEFAULT_MODEL = "gpt-neox:20b"  # Adjust to your 20B model name
TEMPERATURE = 0.7
MAX_TOKENS = 512
TIMEOUT = 300  # 20B model needs more time
BATCH_SIZE = 5  # Smaller batch for large model


def get_supabase_client():
    """Get Supabase client"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    return create_client(url, key)


def check_ollama(ollama_url: str):
    """Check if Ollama is running on MacBook"""
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=10)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [m.get("name", "") for m in models]
    except requests.RequestException as e:
        print(f"  Connection error: {e}")
        return None


def generate_response(prompt: str, model: str, ollama_url: str) -> str:
    """Generate response using Ollama on MacBook"""
    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
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
    except requests.exceptions.Timeout:
        print(f"\n  ✗ Timeout (>{TIMEOUT}s)")
        return None
    except requests.RequestException as e:
        print(f"\n  ✗ Error: {e}")
        return None


def build_prompt(input_text: str, context: str = None) -> str:
    """Build prompt for 20B model"""
    if context:
        return f"Answer the following question accurately and concisely.\n\nContext: {context}\n\nQuestion: {input_text}"
    return f"Answer the following question accurately and concisely.\n\nQuestion: {input_text}"


def fetch_pending_records(supabase) -> list:
    """Fetch ALL records where twentyb is NULL with pagination"""
    print("Fetching pending records from Supabase...")
    all_records = []
    page_size = 1000
    offset = 0
    
    while True:
        try:
            response = supabase.table("modelComp")\
                .select("id, input, context")\
                .is_("twentyb", "null")\
                .order("id")\
                .range(offset, offset + page_size - 1)\
                .execute()
            
            if not response.data:
                break
            
            all_records.extend(response.data)
            print(f"  Fetched page {offset // page_size + 1}: {len(response.data)} records")
            
            if len(response.data) < page_size:
                break
            
            offset += page_size
        except Exception as e:
            print(f"  Error fetching records: {e}")
            break
    
    print(f"✓ Total pending records: {len(all_records)}")
    return all_records


def update_twentyb(supabase, record_id: int, output: str):
    """Update twentyb column for a record"""
    try:
        supabase.table("modelComp")\
            .update({"twentyb": output})\
            .eq("id", record_id)\
            .execute()
        return True
    except Exception as e:
        print(f"\n  ✗ Update error: {e}")
        return False


def main(macbook_ip: str, model: str = DEFAULT_MODEL):
    """Generate GPT-OSS 20B outputs and update Supabase"""
    
    ollama_url = f"http://{macbook_ip}:{OLLAMA_PORT}"
    
    print("=" * 60)
    print("STEP 4: GENERATE GPT-OSS 20B → UPDATE SUPABASE")
    print("=" * 60)
    print(f"⚠️  Running on MacBook Pro via network")
    print(f"   MacBook IP: {macbook_ip}")
    print(f"   Ollama URL: {ollama_url}")
    
    # Check Ollama on MacBook
    print("\nChecking MacBook connection...")
    available_models = check_ollama(ollama_url)
    
    if available_models is None:
        print("✗ Cannot connect to MacBook Ollama!")
        print("\nOn MacBook, run:")
        print("  OLLAMA_HOST=0.0.0.0 ollama serve")
        print("\nThen get IP with:")
        print("  ipconfig getifaddr en0")
        return
    
    print(f"✓ Connected to MacBook Ollama")
    print(f"  Available models: {', '.join(available_models[:5])}")
    
    # Check model
    model_found = any(model.split(":")[0] in m for m in available_models)
    if not model_found:
        print(f"\n✗ Model '{model}' not found on MacBook!")
        print(f"  On MacBook, run: ollama pull {model}")
        return
    
    print(f"✓ Using model: {model}")
    
    # Connect to Supabase
    try:
        supabase = get_supabase_client()
        print("✓ Connected to Supabase")
    except Exception as e:
        print(f"✗ Supabase connection failed: {e}")
        return
    
    # Fetch pending records
    pending = fetch_pending_records(supabase)
    print(f"✓ Found {len(pending)} records needing twentyb")
    
    if not pending:
        print("\n✓ All records already have twentyb!")
        return
    
    print(f"\nGenerating GPT-OSS 20B outputs...")
    print("-" * 60)
    
    start_time = time.time()
    success_count = 0
    fail_count = 0
    
    for record in tqdm(pending, desc="GPT-OSS 20B → Supabase"):
        record_id = record["id"]
        input_text = record["input"]
        context = record.get("context")
        
        # Build prompt
        prompt = build_prompt(input_text, context)
        
        # Generate output
        output = generate_response(prompt, model, ollama_url)
        
        if output:
            if update_twentyb(supabase, record_id, output):
                success_count += 1
            else:
                fail_count += 1
        else:
            fail_count += 1
        
        # Progress update every batch
        if (success_count + fail_count) % BATCH_SIZE == 0:
            elapsed = time.time() - start_time
            rate = (success_count + fail_count) / elapsed * 60
            tqdm.write(f"  ⚡ Rate: {rate:.1f} samples/min")
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"✓ Success: {success_count}")
    print(f"✗ Failed: {fail_count}")
    print(f"⏱  Time: {elapsed/60:.1f} minutes")
    print(f"⚡ Rate: {success_count/elapsed*60:.1f} samples/min")
    print("=" * 60)
    print("\n→ Next: Evaluation (after meeting with sir)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GPT-OSS 20B → Supabase")
    parser.add_argument("--ip", type=str, required=True,
                       help="MacBook IP address (e.g., 192.168.1.100)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                       help=f"Ollama model name (default: {DEFAULT_MODEL})")
    args = parser.parse_args()
    main(macbook_ip=args.ip, model=args.model)
