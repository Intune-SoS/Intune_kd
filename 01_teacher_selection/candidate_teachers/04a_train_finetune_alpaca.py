"""
Step 5: Fine-tune Gemma 3:1B using Alpaca outputs as teacher (IMPROVED)
=======================================================================
Changes:
- Added validation split (90/10)
- Better hyperparameters for LoRA
- More logging and monitoring
- Data quality checks
- Evaluation during training
"""

import os
import sys
from pathlib import Path

# Windows compatibility - MUST be set before importing unsloth
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["UNSLOTH_NUM_PROC"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from supabase import create_client
import json

load_dotenv()

# OPTIMIZED Training config (Best of both approaches)
MODEL_NAME = "unsloth/gemma-3-1b-it-bnb-4bit"
OUTPUT_DIR = PROJECT_ROOT / "models" / "gemma-alpaca-teacher"
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 16  # REVERTED: Keep original for this dataset size
LORA_DROPOUT = 0  # REVERTED: No dropout - data is limited
DEFAULT_EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-4  # REVERTED: Original LR was better
WARMUP_RATIO = 0.05  # REDUCED: Less warmup needed


def get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    return create_client(url, key)


def fetch_training_data():
    """Fetch ALL data from Supabase and split train/val"""
    print("Fetching training data from Supabase...")
    supabase = get_supabase_client()
    
    all_records = []
    page_size = 1000
    offset = 0
    
    while True:
        response = supabase.table("modelComp")\
            .select("input, context, sevenb")\
            .not_.is_("sevenb", "null")\
            .range(offset, offset + page_size - 1)\
            .execute()
        
        if not response.data:
            break
        
        all_records.extend(response.data)
        print(f"  Fetched page {offset // page_size + 1}: {len(response.data)} records")
        
        if len(response.data) < page_size:
            break
        
        offset += page_size
    
    print(f"✓ Fetched {len(all_records)} total samples")
    
    # OPTIMIZED: Use 95% train, 5% validation (keep more training data)
    split_idx = int(len(all_records) * 0.95)
    train_data = all_records[:split_idx]
    val_data = all_records[split_idx:]
    
    print(f"✓ Split: {len(train_data)} train, {len(val_data)} validation")
    return train_data, val_data


def format_for_training(data):
    """Format data for Gemma fine-tuning"""
    formatted = []
    
    for item in data:
        instruction = item["input"]
        context = item.get("context") or ""
        output = item["sevenb"]
        
        if context:
            input_text = f"Context: {context}\n\nQuestion: {instruction}"
        else:
            input_text = f"Question: {instruction}"
        
        formatted.append({
            "instruction": "Answer the following question accurately and concisely.",
            "input": input_text,
            "output": output
        })
    
    return formatted


def analyze_data(formatted_data):
    """NEW: Analyze data quality"""
    print("\n" + "=" * 60)
    print("DATA QUALITY ANALYSIS")
    print("=" * 60)
    
    output_lengths = [len(item["output"]) for item in formatted_data]
    input_lengths = [len(item["input"]) for item in formatted_data]
    
    print(f"Output lengths: min={min(output_lengths)}, max={max(output_lengths)}, avg={sum(output_lengths)/len(output_lengths):.0f}")
    print(f"Input lengths: min={min(input_lengths)}, max={max(input_lengths)}, avg={sum(input_lengths)/len(input_lengths):.0f}")
    
    print(f"\nSample training example:")
    print(f"Input: {formatted_data[0]['input'][:100]}...")
    print(f"Output: {formatted_data[0]['output'][:100]}...")


def train(epochs: int = DEFAULT_EPOCHS):
    """Fine-tune Gemma with Alpaca outputs as teacher"""
    from unsloth import FastLanguageModel
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from datasets import Dataset
    
    print("=" * 60)
    print("STEP 5: FINE-TUNE GEMMA WITH ALPACA TEACHER (IMPROVED)")
    print("=" * 60)
    
    # Fetch and format data with validation split
    train_raw, val_raw = fetch_training_data()
    if not train_raw:
        print("✗ No training data found!")
        return
    
    train_formatted = format_for_training(train_raw)
    val_formatted = format_for_training(val_raw)
    
    # NEW: Analyze data quality
    analyze_data(train_formatted)
    
    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted)
    
    print(f"\n✓ Train: {len(train_dataset)} samples")
    print(f"✓ Validation: {len(val_dataset)} samples")
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Add LoRA adapters with improved config
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,  # Now 32 instead of 16
        lora_dropout=LORA_DROPOUT,  # Now 0.05 instead of 0
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    print("✓ Model loaded with LoRA adapters")
    print(f"  LoRA r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    
    # Format function
    def formatting_func(examples):
        texts = []
        for i in range(len(examples["instruction"])):
            text = f"""<bos><start_of_turn>user
{examples["instruction"][i]}

{examples["input"][i]}<end_of_turn>
<start_of_turn>model
{examples["output"][i]}<end_of_turn>"""
            texts.append(text)
        return {"text": texts}
    
    train_dataset = train_dataset.map(formatting_func, batched=True)
    val_dataset = val_dataset.map(formatting_func, batched=True)
    
    # Pre-tokenize
    print("\nTokenizing datasets...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )
    
    train_dataset = train_dataset.map(
        tokenize_function, 
        batched=True, 
        num_proc=1, 
        remove_columns=["instruction", "input", "output", "text"]
    )
    val_dataset = val_dataset.map(
        tokenize_function, 
        batched=True, 
        num_proc=1, 
        remove_columns=["instruction", "input", "output", "text"]
    )
    
    print(f"✓ Tokenized train: {len(train_dataset)} samples")
    print(f"✓ Tokenized val: {len(val_dataset)} samples")
    
    # Calculate warmup steps
    total_steps = (len(train_dataset) // (BATCH_SIZE * 4)) * epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    print(f"\nTraining plan:")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Learning rate: {LEARNING_RATE}")
    
    # IMPROVED Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        warmup_steps=warmup_steps,
        num_train_epochs=epochs,
        learning_rate=LEARNING_RATE,
        bf16=True,
        logging_steps=10,  # More frequent logging
        eval_strategy="steps",  # NEW: Evaluate during training
        eval_steps=50,  # NEW: Evaluate every 50 steps
        save_strategy="steps",  # NEW: Save based on steps
        save_steps=100,  # NEW: Save every 100 steps
        save_total_limit=3,  # NEW: Keep only best 3 checkpoints
        load_best_model_at_end=True,  # NEW: Load best at end
        metric_for_best_model="eval_loss",  # NEW: Use eval loss
        output_dir=str(OUTPUT_DIR),
        optim="adamw_8bit",
        dataloader_num_workers=0,
        report_to="none",  # Disable wandb/tensorboard
    )
    
    # Trainer with validation
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # NEW: Added validation
        args=training_args,
        data_collator=data_collator,
    )
    
    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 60)
    
    trainer.train()
    
    # NEW: Print final metrics
    print("\n" + "=" * 60)
    print("FINAL METRICS")
    print("=" * 60)
    final_metrics = trainer.evaluate()
    print(f"Final validation loss: {final_metrics['eval_loss']:.4f}")
    
    # Save model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"✓ Model saved to: {OUTPUT_DIR}")
    print("→ Next: Run 06_finetune_with_oss20b.py")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    args = parser.parse_args()
    train(epochs=args.epochs)