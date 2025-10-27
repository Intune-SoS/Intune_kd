"""
Step 1: Download Stanford Alpaca Dataset
=========================================
Downloads the original 52K instruction-following dataset

Usage:
    python experiment/01_download_alpaca.py
"""

import os
import json
import requests
from pathlib import Path

# Stanford Alpaca dataset URL (raw JSON from GitHub)
ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "experiment"


def download_alpaca_dataset() -> str:
    """
    Download the Stanford Alpaca dataset
    
    Returns:
        Path to the downloaded JSON file
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = DATA_DIR / "alpaca_data_raw.json"
    
    print("=" * 60)
    print("STEP 1: DOWNLOAD STANFORD ALPACA DATASET")
    print("=" * 60)
    print(f"URL: {ALPACA_URL}")
    print(f"Output: {output_file}")
    print()
    
    # Check if already downloaded
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Already downloaded: {len(data)} samples")
        return str(output_file)
    
    print("Downloading...")
    
    try:
        response = requests.get(ALPACA_URL, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Downloaded {len(data)} samples")
        print(f"✓ Saved to: {output_file}")
        
        # Print sample structure
        print(f"\nSample structure:")
        sample = data[0]
        for key in sample.keys():
            value_preview = str(sample[key])[:60] + "..." if len(str(sample[key])) > 60 else sample[key]
            print(f"  - {key}: {value_preview}")
        
        # Statistics
        print(f"\n{'='*60}")
        print("Dataset Statistics:")
        with_input = sum(1 for d in data if d.get('input', '').strip())
        print(f"  Total samples: {len(data)}")
        print(f"  With context (input field): {with_input}")
        print(f"  Without context: {len(data) - with_input}")
        print("=" * 60)
        
        return str(output_file)
        
    except requests.RequestException as e:
        print(f"✗ Error downloading: {e}")
        raise


if __name__ == "__main__":
    download_alpaca_dataset()
