"""
Convert fine-tuned HuggingFace model to GGUF format for Ollama
Requires llama.cpp converter
"""

import os
import subprocess
import sys

# Configuration - Use paths relative to project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HF_MODEL_PATH = os.path.join(project_root, "models", "gemma-finetuned-merged-fp16")  # USE FP16 VERSION
LLAMA_CPP_PATH = os.path.join(project_root, "llama.cpp")
QUANT_TYPE = "Q4_K_M"  # Good balance of size/quality (Q4_K_M, Q5_K_M, Q8_0, etc.)
OUTPUT_GGUF_F16 = os.path.join(project_root, "models", "gemma-finetuned-f16.gguf")  # Intermediate F16 GGUF
OUTPUT_GGUF = os.path.join(project_root, "models", f"gemma-finetuned.{QUANT_TYPE}.gguf")  # Final quantized GGUF


def check_llama_cpp():
    """Check if llama.cpp exists, if not clone it"""
    if not os.path.exists(LLAMA_CPP_PATH):
        print("=" * 80)
        print("CLONING LLAMA.CPP")
        print("=" * 80)
        print("\nllama.cpp not found. Cloning from GitHub...")
        
        try:
            subprocess.run([
                "git", "clone", 
                "https://github.com/ggerganov/llama.cpp",
                LLAMA_CPP_PATH
            ], check=True)
            print("✓ llama.cpp cloned successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error cloning llama.cpp: {e}")
            sys.exit(1)
    else:
        print("✓ llama.cpp found")
    
    return True


def check_converter_script():
    """Check if the converter script exists"""
    converter_path = os.path.join(LLAMA_CPP_PATH, "convert_hf_to_gguf.py")
    
    if not os.path.exists(converter_path):
        print(f"\n✗ Converter script not found: {converter_path}")
        print("Please ensure llama.cpp is up to date.")
        sys.exit(1)
    
    print("✓ Converter script found")
    return converter_path


def convert_to_gguf_f16(converter_path):
    """Convert HuggingFace model to F16 GGUF (unquantized)"""
    print("\n" + "=" * 80)
    print("STEP 1: CONVERTING TO F16 GGUF")
    print("=" * 80)
    
    print(f"\n📦 Source model: {HF_MODEL_PATH}")
    print(f"🎯 Output file: {OUTPUT_GGUF_F16}")
    print(f"🔧 Format: F16 (unquantized)")
    print("\nThis may take a few minutes...\n")
    
    try:
        # Run the converter (F16 format)
        result = subprocess.run([
            sys.executable,
            converter_path,
            HF_MODEL_PATH,  # Positional argument
            "--outtype", "f16",
            "--outfile", OUTPUT_GGUF_F16
        ], check=True, capture_output=True, text=True)
        
        print(result.stdout)
        print("\n✓ F16 GGUF created successfully!")
        
        # Show file size
        if os.path.exists(OUTPUT_GGUF_F16):
            size_mb = os.path.getsize(OUTPUT_GGUF_F16) / (1024 * 1024)
            print(f"📊 F16 GGUF size: {size_mb:.2f} MB")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Conversion failed!")
        print(f"Error: {e}")
        if e.stderr:
            print(f"Details: {e.stderr}")
        return False


def quantize_gguf():
    """Quantize F16 GGUF to specified quantization level"""
    print("\n" + "=" * 80)
    print("STEP 2: QUANTIZING TO " + QUANT_TYPE)
    print("=" * 80)
    
    print(f"\n📦 Source: {OUTPUT_GGUF_F16}")
    print(f"🎯 Output: {OUTPUT_GGUF}")
    print(f"🔧 Quantization: {QUANT_TYPE}")
    print("\nThis may take several minutes...\n")
    
    # Find quantize executable
    quantize_exe = os.path.join(LLAMA_CPP_PATH, "llama-quantize.exe")
    if not os.path.exists(quantize_exe):
        print(f"⚠️  Pre-built quantize.exe not found. Building llama.cpp...")
        print("This requires CMake and Visual Studio Build Tools.\n")
        print("Alternative: Download pre-built binaries from:")
        print("https://github.com/ggerganov/llama.cpp/releases\n")
        print("Or use F16 GGUF directly (unquantized but works):")
        print(f"  Rename: {OUTPUT_GGUF_F16} → gemma-finetuned.gguf")
        return False
    
    try:
        # Run quantization
        result = subprocess.run([
            quantize_exe,
            OUTPUT_GGUF_F16,
            OUTPUT_GGUF,
            QUANT_TYPE
        ], check=True, capture_output=True, text=True)
        
        print(result.stdout)
        print("\n✓ Quantization completed successfully!")
        
        # Show file sizes
        if os.path.exists(OUTPUT_GGUF):
            size_mb = os.path.getsize(OUTPUT_GGUF) / (1024 * 1024)
            f16_size_mb = os.path.getsize(OUTPUT_GGUF_F16) / (1024 * 1024)
            reduction = (1 - size_mb / f16_size_mb) * 100
            print(f"📊 Quantized size: {size_mb:.2f} MB ({reduction:.1f}% smaller)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Quantization failed!")
        print(f"Error: {e}")
        if e.stderr:
            print(f"Details: {e.stderr}")
        return False


def main():
    """Main conversion pipeline"""
    print("=" * 80)
    print("HUGGINGFACE → GGUF CONVERSION FOR OLLAMA")
    print("=" * 80)
    
    # Check if HF model exists
    if not os.path.exists(HF_MODEL_PATH):
        print(f"\n✗ HuggingFace model not found: {HF_MODEL_PATH}")
        print("Please run finetune_gemma.py first to create the fine-tuned model.")
        sys.exit(1)
    
    print(f"\n✓ HuggingFace model found: {HF_MODEL_PATH}")
    
    # Check/clone llama.cpp
    check_llama_cpp()
    
    # Check converter script
    converter_path = check_converter_script()
    
    # Step 1: Convert to F16 GGUF
    if not convert_to_gguf_f16(converter_path):
        print("\n❌ Failed to create F16 GGUF. Exiting.")
        sys.exit(1)
    
    # Step 2: Quantize (optional but recommended)
    print("\n" + "=" * 80)
    print("📝 NOTE: Quantization requires llama-quantize.exe")
    print("=" * 80)
    print("\nOptions:")
    print("  1. Use F16 GGUF directly (larger but works)")
    print("  2. Download pre-built llama.cpp from releases")
    print("  3. Build llama.cpp with CMake (requires Visual Studio)\n")
    
    quantize_success = quantize_gguf()
    
    print("\n" + "=" * 80)
    if quantize_success:
        print("✅ CONVERSION COMPLETED!")
        print("=" * 80)
        print(f"\n✓ Quantized GGUF ready: {OUTPUT_GGUF}")
        print(f"\nNext step:")
        print(f"  python create_ollama_model.py")
    else:
        print("⚠️  QUANTIZATION SKIPPED")
        print("=" * 80)
        print(f"\n✓ F16 GGUF ready: {OUTPUT_GGUF_F16}")
        print(f"\nYou can use F16 GGUF directly:")
        print(f"  1. Rename to: gemma-finetuned.gguf")
        print(f"  2. Update GGUF_FILE in create_ollama_model.py")
        print(f"  3. Run: python create_ollama_model.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
