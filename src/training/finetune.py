"""
Fine-tune Gemma 3:1b using Unsloth + LoRA
Optimized for 8GB VRAM (RTX 4060)
"""

# Windows compatibility: MUST be set before any other imports
import os
import sys

# Fix UTF-8 encoding for Windows (to support emoji characters in print statements)
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Disable all multiprocessing to avoid Windows spawn issues
os.environ["HF_DATASETS_DISABLE_MP"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Prevent CUDA fork issues on Windows
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Force datasets library to use single process
os.environ["HF_DATASETS_NUM_PROC"] = "1"

# Disable Triton/torch.compile (Windows compatibility - Triton has issues on Windows)
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TRITON_DISABLE_LINE_INFO"] = "1"
os.environ["DISABLE_TRITON"] = "1"
os.environ["UNSLOTH_DISABLE_TRITON"] = "1"  # Unsloth-specific flag

import json
from datasets import load_dataset

# Disable Triton at import time
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# NOTE: No unsloth import here - we use transformers + PEFT for Python 3.9 compatibility

# Configuration
MAX_SEQ_LENGTH = 2048  # Max sequence length (can reduce if OOM)
DTYPE = None  # Auto-detect (will use float16 for training)
LOAD_IN_4BIT = True  # 4-bit quantization for 1B model (optimized for 8GB VRAM)

# LoRA Configuration (optimized for 8GB VRAM)
LORA_R = 16  # LoRA rank
LORA_ALPHA = 32  # LoRA alpha
LORA_DROPOUT = 0
TARGET_MODULES = ["c_attn", "c_proj", "c_fc"]  # DialoGPT specific modules

# Training Configuration (CPU optimized)
BATCH_SIZE = 1  # Small batch for CPU training
GRADIENT_ACCUMULATION_STEPS = 8  # Higher accumulation for effective batch = 1 * 8 = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1  # Single epoch for quick testing
WARMUP_STEPS = 5
LOGGING_STEPS = 1
SAVE_STEPS = 50

# Model name - using open model without access restrictions
MODEL_NAME = "microsoft/DialoGPT-small"  # Small model for CPU compatibility, no restrictions


def load_training_data():
    """Load training and validation datasets"""
    print("Loading datasets...")
    
    # Get project root (two levels up from src/training/)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    # Paths to data files
    train_file = os.path.join(project_root, 'data', 'processed', 'train_dataset.jsonl')
    val_file = os.path.join(project_root, 'data', 'processed', 'val_dataset.jsonl')
    
    # Check if files exist
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}\n"
                              f"Please run: python src/data_generation/prepare_data.py")
    if not os.path.exists(val_file):
        raise FileNotFoundError(f"Validation file not found: {val_file}\n"
                              f"Please run: python src/data_generation/prepare_data.py")
    
    print(f"  Loading from: {train_file}")
    print(f"  Loading from: {val_file}")
    
    # Load from JSONL files
    dataset = load_dataset('json', data_files={
        'train': train_file,
        'validation': val_file
    })
    
    print(f"✓ Training samples: {len(dataset['train'])}")
    print(f"✓ Validation samples: {len(dataset['validation'])}")
    
    return dataset


def format_prompt(sample):
    """Format sample into instruction prompt for DialoGPT"""
    instruction = sample['instruction']
    input_text = sample['input']
    output_text = sample['output']

    # Simple format for DialoGPT - just input and expected output
    prompt = f"Human: {instruction}\n\n{input_text}\nAssistant: {output_text}<|endoftext|>"

    return prompt


def formatting_prompts_func(examples):
    """Format multiple examples for training"""
    texts = []
    for i in range(len(examples['instruction'])):
        sample = {
            'instruction': examples['instruction'][i],
            'input': examples['input'][i],
            'output': examples['output'][i]
        }
        texts.append(format_prompt(sample))
    return {"text": texts}


def setup_model_and_tokenizer():
    """Load and prepare model with LoRA"""
    print("\n" + "=" * 80)
    print("LOADING MODEL AND APPLYING LORA")
    print("=" * 80)

    # Use standard transformers instead of Unsloth to avoid Triton requirements
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    # CPU-optimized model loading (no quantization for CPU)
    if torch.cuda.is_available():
        print("🎮 CUDA available - using GPU with quantization")
        from transformers import BitsAndBytesConfig

        # Configure 4-bit quantization for GPU
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Prepare model for k-bit training
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    else:
        print("💻 CUDA not available - using CPU (slower but functional)")
        # Load model for CPU without quantization
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float32,  # Use float32 for CPU
            device_map={"": "cpu"},
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("✓ Model loaded")

    # Apply LoRA using PEFT
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    print("✓ LoRA applied")
    print(f"  - Rank: {LORA_R}")
    print(f"  - Alpha: {LORA_ALPHA}")
    print(f"  - Dropout: {LORA_DROPOUT}")

    return model, tokenizer


def train_model(model, tokenizer, dataset):
    """Train the model"""
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    
    # Define checkpoint output directory relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    checkpoint_dir = os.path.join(project_root, 'models', 'gemma-finetuned')
    
    # Find the latest checkpoint to resume from
    resume_from_checkpoint = None
    if os.path.exists(checkpoint_dir):
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
        if checkpoints:
            # Sort by checkpoint number and get the latest
            checkpoints.sort(key=lambda x: int(x.split('-')[1]))
            latest_checkpoint = checkpoints[-1]
            resume_from_checkpoint = os.path.join(checkpoint_dir, latest_checkpoint)
            print(f"\n📍 Found existing checkpoint: {latest_checkpoint}")
            print(f"   Resuming training from: {resume_from_checkpoint}")
    
    # CRITICAL: Disable Unsloth's custom cross-entropy (requires Triton)
    # Patch model to use standard PyTorch cross-entropy instead
    if hasattr(model, 'config'):
        # Disable any cross-entropy optimizations that require Triton
        model.config.use_cache = False
    
    # Training arguments optimized for CPU/GPU compatibility
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),  # Only use fp16 on GPU
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),  # Only use bf16 on GPU
        logging_steps=LOGGING_STEPS,
        optim="adamw_torch" if not torch.cuda.is_available() else "adamw_8bit",  # Use standard optimizer for CPU
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        eval_strategy="steps",  # Updated from evaluation_strategy
        eval_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        report_to="none",  # Disable wandb/tensorboard
        dataloader_num_workers=0,  # Windows compatibility: disable multiprocessing
    )
    
    # Create data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )
    
    # Create trainer (use standard Trainer, not SFTTrainer to avoid Triton)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
    )
    
    # Resume from checkpoint if it exists
    if resume_from_checkpoint:
        print(f"\n🔄 Resuming from checkpoint: {resume_from_checkpoint}\n")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        train_result = trainer.train()
    
    return trainer, train_result

def show_training_info(dataset):
    """Show training configuration and hardware info"""
    if torch.cuda.is_available():
        # Show GPU memory stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

        print(f"\n🎮 GPU: {gpu_stats.name}")
        print(f"💾 Memory: {start_gpu_memory} GB / {max_memory} GB allocated")
    else:
        print(f"\n💻 Using CPU for training")

    print(f"\n📊 Training configuration:")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  - Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print(f"  - Training samples: {len(dataset['train'])}")
    print(f"  - Validation samples: {len(dataset['validation'])}")

    print("\n" + "=" * 80)
    print("🚀 TRAINING STARTED...")
    print("=" * 80 + "\n")


def save_model(model, tokenizer):
    """Save fine-tuned LoRA adapters and optionally merged model"""
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)
    
    # Define output paths relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    models_dir = os.path.join(project_root, 'models')
    merged_path = os.path.join(models_dir, 'gemma-finetuned-merged')
    lora_path = os.path.join(models_dir, 'gemma-finetuned-lora')
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Save LoRA adapters (lightweight - only trained weights)
    print("\n1. Saving LoRA adapters...")
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"   ✓ LoRA adapters saved to: {lora_path}")
    
    # Merge and save full model (standard PEFT approach)
    print("\n2. Merging LoRA weights and saving full model...")
    from peft import PeftModel
    
    # Get base model and merge
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    print(f"   ✓ Merged model saved to: {merged_path}")
    
    print("\n" + "=" * 80)


def test_model(model, tokenizer):
    """Test the fine-tuned model with a sample"""
    print("\n" + "=" * 80)
    print("TESTING FINE-TUNED MODEL")
    print("=" * 80)

    # Enable inference mode (standard PyTorch)
    model.eval()

    # Test sample
    test_instruction = "Answer the following question accurately and concisely based on the provided information."
    test_input = "Question: What is machine learning?"

    prompt = f"Human: {test_instruction}\n\n{test_input}\nAssistant:"

    # Use appropriate device (CPU or GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer([prompt], return_tensors="pt").to(device)

    print(f"\n🖥️  Using device: {device}")
    print("\n📝 Test input:")
    print(f"  {test_input}")
    print("\n🤖 Model output:")

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the model's response (after "Assistant:")
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1]

    response = response.strip()

    print(f"  {response}")
    print("\n" + "=" * 80)


def main():
    """Main fine-tuning pipeline"""
    print("=" * 80)
    print("FINE-TUNING DIALOGPT WITH LORA")
    print("=" * 80)
    print(f"\n🎯 Target: DialoGPT-small (CPU Compatible)")
    print(f"💻 CPU/GPU: Automatic detection")
    print(f"🔧 Method: LoRA (Low-Rank Adaptation)")
    
    # Load datasets
    dataset = load_training_data()
    
    # Format datasets
    print("\nFormatting prompts...")
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        num_proc=1,  # Force single process (Windows compatibility)
    )
    
    # Setup model and tokenizer FIRST
    model, tokenizer = setup_model_and_tokenizer()

    # Tokenize dataset for standard Trainer (after tokenizer is available)
    print("\nTokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
        )

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=["instruction", "input", "output", "text"]
    )

    # Show training info before starting
    show_training_info(dataset)

    # Train
    trainer, train_result = train_model(model, tokenizer, dataset)
    
    # Save
    save_model(model, tokenizer)
    
    # Test
    test_model(model, tokenizer)
    
    print("\n" + "=" * 80)
    print("🎉 FINE-TUNING PIPELINE COMPLETED!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Test the model: python test_finetuned_model.py")
    print("  2. Deploy to Ollama: python export_to_ollama.py")
    print("  3. Compare metrics: python compare_models.py")
    print("=" * 80)


if __name__ == "__main__":
    # Show hardware info
    if torch.cuda.is_available():
        print("🎮 CUDA is available - using GPU acceleration")
        gpu_stats = torch.cuda.get_device_properties(0)
        print(f"   GPU: {gpu_stats.name}")
        print(f"   Memory: {round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)} GB")
    else:
        print("💻 CUDA not available - using CPU (training will be slower but functional)")

    # Clear CUDA cache before starting (if available)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    main()
