"""
Properly merge LoRA adapters and dequantize the model for GGUF conversion
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHECKPOINT_PATH = os.path.join(project_root, 'models', 'gemma-finetuned', 'checkpoint-1233')
OUTPUT_PATH = os.path.join(project_root, 'models', 'gemma-finetuned-merged-fp16')

def merge_and_dequantize():
    """Load base model in FP16 + LoRA adapter and merge"""
    
    print("=" * 80)
    print("MERGING LORA + SAVING IN FP16 FOR GGUF")
    print("=" * 80)
    
    print(f"\n📦 Loading checkpoint: {CHECKPOINT_PATH}")
    
    # Load base model in FP16 (not quantized)
    print("\n1. Loading base model in FP16 (unquantized)...")
    print("   This may take a few minutes and use more memory...")
    
    # Use the same base as training but load without quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        "unsloth/gemma-3-1b-it",  # Unsloth version without bnb suffix
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("   ✓ Base model loaded in FP16")
    
    # Load LoRA adapters from checkpoint
    print("\n2. Loading LoRA adapters from checkpoint...")
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
    print("   ✓ LoRA adapters loaded")
    
    # Merge LoRA into base model
    print("\n3. Merging LoRA adapters into FP16 model...")
    merged_model = model.merge_and_unload()
    print("   ✓ LoRA merged")
    
    # Save to disk
    print(f"\n4. Saving merged FP16 model to: {OUTPUT_PATH}")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    merged_model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH, trust_remote_code=True)
    tokenizer.save_pretrained(OUTPUT_PATH)
    print("   ✓ Model and tokenizer saved")
    
    print("\n" + "=" * 80)
    print("✅ MERGE COMPLETE!")
    print("=" * 80)
    print(f"\n🎯 Output model ready at: {OUTPUT_PATH}")
    print("\nNext step: Run convert_to_gguf.py")
    print("=" * 80)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA not available")
        sys.exit(1)
    
    torch.cuda.empty_cache()
    merge_and_dequantize()
