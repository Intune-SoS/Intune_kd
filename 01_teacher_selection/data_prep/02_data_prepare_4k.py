"""
Step 2: Prepare 4K Comparison Dataset
======================================
Extracts first 4000 samples for teacher model comparison (LLaMA 7B vs GPT-OSS 20B)

Schema:
{
    "id": int,
    "instruction": str,           # From Alpaca
    "context": str,               # From Alpaca 'input' field  
    "input": str,                 # Formatted prompt
    "alpaca_output": str,         # Original reference (not used in training)
    "expected_output_llama7b": null,   # To be filled by Step 4
    "expected_output_gpt20b": null,    # To be filled by Step 5
    "actual_output_gemma": null,       # To be filled by Step 3
}

Usage:
    python experiment/02_prepare_4k_dataset.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "experiment"

NUM_SAMPLES = 50000  # For teacher comparison phase


def format_input_prompt(instruction: str, context: str) -> str:
    """
    Format instruction and context into a single input prompt
    """
    if context.strip():
        return f"""Answer the following question accurately and concisely based on the provided information.

Context:
{context}

Question: {instruction}"""
    else:
        return f"""Answer the following question accurately and concisely.

Question: {instruction}"""


def prepare_4k_dataset():
    """
    Prepare the 4K comparison dataset from Alpaca
    """
    alpaca_path = DATA_DIR / "alpaca_data_raw.json"
    output_path = DATA_DIR / "experiment_4k.json"
    
    print("=" * 60)
    print("STEP 2: PREPARE 4K COMPARISON DATASET")
    print("=" * 60)
    
    # Check if Alpaca dataset exists
    if not alpaca_path.exists():
        print(f"✗ Alpaca dataset not found: {alpaca_path}")
        print("  Run Step 1 first: python experiment/01_download_alpaca.py")
        return None
    
    # Load Alpaca dataset
    print(f"Loading: {alpaca_path}")
    with open(alpaca_path, 'r', encoding='utf-8') as f:
        alpaca_data = json.load(f)
    
    print(f"✓ Loaded {len(alpaca_data)} samples")
    print(f"Extracting first {NUM_SAMPLES} samples...")
    
    # Extract and format
    experiment_data = []
    
    for i, sample in enumerate(alpaca_data[:NUM_SAMPLES]):
        instruction = sample.get('instruction', '').strip()
        context = sample.get('input', '').strip()  # Alpaca uses 'input' for context
        alpaca_output = sample.get('output', '').strip()
        
        formatted_input = format_input_prompt(instruction, context)
        
        experiment_sample = {
            "id": i + 1,
            "instruction": instruction,
            "context": context,
            "input": formatted_input,
            "alpaca_output": alpaca_output,
            # Placeholders - to be filled by subsequent steps
            "expected_output_llama7b": None,
            "expected_output_gpt20b": None,
            "actual_output_gemma": None,
        }
        
        experiment_data.append(experiment_sample)
    
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(experiment_data)} samples to: {output_path}")
    
    # Show sample
    print(f"\n{'='*60}")
    print("Sample Entry (ID: 1):")
    print("=" * 60)
    sample = experiment_data[0]
    print(f"Instruction: {sample['instruction'][:100]}...")
    print(f"Context: {sample['context'][:100] if sample['context'] else '(none)'}...")
    print(f"Input prompt length: {len(sample['input'])} chars")
    print("=" * 60)
    
    return str(output_path)


if __name__ == "__main__":
    prepare_4k_dataset()
