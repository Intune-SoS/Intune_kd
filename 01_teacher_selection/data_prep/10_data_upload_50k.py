"""
Upload 50K dataset to Supabase with correct field mapping:
- instruction -> input
- context -> context
- alpaca_output -> sevenb (teacher output)
"""

import json
import os
from tqdm import tqdm
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

def main():
    # Load the 50K data
    print('Loading experiment_4k.json (50K records)...')
    with open('data/experiment/experiment_4k.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f'Loaded {len(data)} records')
    
    # Show sample record
    print("\nSample record mapping:")
    sample = data[0]
    print(f"  instruction: {sample['instruction'][:100]}...")
    print(f"  context: {(sample.get('context') or '')[:100]}...")
    print(f"  alpaca_output: {sample['alpaca_output'][:100]}...")
    
    # Connect to Supabase
    supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
    
    # Clear existing data first
    print("\nClearing existing data from modelcomp_50k...")
    try:
        # Delete all existing records
        supabase.table('modelcomp_50k').delete().gte('id', 0).execute()
        print("Cleared existing records")
    except Exception as e:
        print(f"Clear failed (table might be empty): {e}")
    
    # Upload in batches with CORRECT mapping
    print("\nUploading with correct mapping:")
    print("  instruction -> input")
    print("  context -> context") 
    print("  alpaca_output -> sevenb")
    print()
    
    batch_size = 100
    success = 0
    
    for i in tqdm(range(0, len(data), batch_size), desc='Uploading'):
        batch = data[i:i+batch_size]
        records = [{
            'input': item['instruction'][:5000],           # instruction -> input
            'context': (item.get('context') or '')[:2000], # context -> context
            'sevenb': item['alpaca_output'][:5000],        # alpaca_output -> sevenb
            'checkpoint': ((item['id'] - 1) // 5000) + 1   # Stage 1-10
        } for item in batch]
        
        try:
            supabase.table('modelcomp_50k').insert(records).execute()
            success += len(batch)
        except Exception as e:
            print(f'\nError at batch {i//batch_size}: {e}')
            break
    
    print(f'\nUploaded {success}/{len(data)} records')
    
    # Verify upload
    result = supabase.table('modelcomp_50k').select('id', count='exact').limit(1).execute()
    print(f'Total records in table: {result.count}')
    
    # Show sample from database
    sample_db = supabase.table('modelcomp_50k').select('*').limit(1).execute()
    if sample_db.data:
        print("\nSample record from database:")
        rec = sample_db.data[0]
        print(f"  input: {rec['input'][:100]}...")
        print(f"  context: {(rec.get('context') or '')[:100]}...")
        print(f"  sevenb: {rec['sevenb'][:100]}...")

if __name__ == "__main__":
    main()
