#!/usr/bin/env python3
"""
Generate Context Derived from Teacher Output (No LLM)
=====================================================
Extracts background context from teacher (sevenb) output to help student learning.
Context provides supporting information derived from teacher's answer.

For research validity:
- Context is derived from teacher output (sevenb column)
- Provides domain knowledge extracted from the answer
- Student learns to use context + instruction to generate answers

Usage:
    python 08_gen_context_ollama.py --limit 100    # Test with 100 records
    python 08_gen_context_ollama.py                # Process all NULL records
    python 08_gen_context_ollama.py --dry-run      # Preview without DB updates
"""

import os
import sys
import re
import time
import argparse
from dotenv import load_dotenv
from supabase import create_client
from tqdm import tqdm

load_dotenv()

BATCH_SIZE = 100


def get_supabase():
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


def fetch_null_context_records(supabase, limit=None, offset=0):
    """Fetch records where context IS NULL"""
    query = supabase.table("modelcomp_50k")\
        .select("id, input, sevenb, task_label")\
        .is_("context", "null")\
        .order("id")\
        .range(offset, offset + (limit or BATCH_SIZE) - 1)

    result = query.execute()
    return result.data


def count_null_context(supabase):
    """Count total NULL context records (estimated, faster than exact)"""
    try:
        result = supabase.table("modelcomp_50k")\
            .select("id", count="estimated")\
            .is_("context", "null")\
            .execute()
        return result.count if result.count else 0
    except Exception as e:
        print(f"Note: Count may be approximate due to large dataset")
        return 30000  # Approximate for progress display


def extract_key_concepts(text):
    """Extract key concepts/terms from text"""
    if not text:
        return []

    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]+`', '', text)

    # Extract sentences
    sentences = re.split(r'[.!?]+', text)

    # Get meaningful sentences (at least 3 words)
    concepts = []
    for s in sentences:
        s = s.strip()
        words = s.split()
        if len(words) >= 3:
            # Truncate very long sentences
            if len(words) > 40:
                s = " ".join(words[:40])
            concepts.append(s)

    return concepts[:4]  # Max 4 key sentences


def derive_context_from_teacher(instruction, teacher_output, task_label):
    """
    Derive supporting context from teacher output.
    Context provides background info extracted from teacher's answer.
    """

    if not teacher_output:
        return None

    teacher_clean = teacher_output.strip()

    # For very short outputs, just use them directly
    if len(teacher_clean) < 20:
        if len(teacher_clean) > 5:
            return f"Teacher answer: {teacher_clean}"
        return None

    label = (task_label or "general").lower()

    # Extract key sentences from teacher output
    key_concepts = extract_key_concepts(teacher_output)

    # Build context based on task type
    if "code" in label or "technical" in label:
        # For code tasks, extract the approach description (not the code)
        non_code = re.sub(r'```[\s\S]*?```', '', teacher_output)
        non_code = re.sub(r'def \w+\(.*?\):', '', non_code)
        non_code = re.sub(r'function \w+\(.*?\)', '', non_code)

        sentences = extract_key_concepts(non_code)
        if sentences:
            context = "Technical approach: " + " ".join(sentences[:2])
        else:
            context = f"Teacher guidance: {teacher_clean[:150]}"

    elif "math" in label or "logic" in label:
        if key_concepts:
            context = "Mathematical approach: " + key_concepts[0]
        else:
            context = f"Teacher reasoning: {teacher_clean[:150]}"

    elif "classification" in label or "analysis" in label:
        if key_concepts:
            context = "Analysis approach: " + " ".join(key_concepts[:2])
        else:
            context = f"Teacher analysis: {teacher_clean[:150]}"

    elif "creative" in label or "generative" in label:
        if key_concepts:
            context = "Creative guidance: " + key_concepts[0]
        else:
            context = f"Teacher example: {teacher_clean[:150]}"

    elif "language" in label or "editing" in label:
        if key_concepts:
            context = "Language guidance: " + key_concepts[0]
        else:
            context = f"Teacher example: {teacher_clean[:150]}"

    else:
        # General QA - use teacher output as background
        if key_concepts:
            context = "Teacher knowledge: " + " ".join(key_concepts[:2])
        else:
            # Use teacher output directly as context hint
            context = f"Teacher reference: {teacher_clean[:200]}"

    # Clean up context
    context = context.strip()
    context = re.sub(r'\s+', ' ', context)

    # Ensure reasonable length
    if len(context) < 15:
        return None
    if len(context) > 500:
        context = context[:497] + "..."

    return context


def update_context(supabase, record_id, context):
    """Update context column in database"""
    try:
        supabase.table("modelcomp_50k")\
            .update({"context": context})\
            .eq("id", record_id)\
            .execute()
        return True
    except Exception as e:
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate context derived from teacher output (no LLM)")
    parser.add_argument("--limit", type=int, help="Limit number of records to process")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for fetching")
    parser.add_argument("--dry-run", action="store_true", help="Preview without DB updates")
    parser.add_argument("--show-samples", action="store_true", help="Show sample outputs")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("GENERATE CONTEXT DERIVED FROM TEACHER OUTPUT")
    print("="*70)
    print("Context = background knowledge extracted from teacher answer")
    print("No LLM used - pure text extraction")
    print("="*70)

    # Connect to Supabase
    try:
        supabase = get_supabase()
        print("OK: Connected to Supabase")
    except Exception as e:
        print(f"ERROR: Supabase connection failed: {e}")
        return

    # Count NULL context
    total_null = count_null_context(supabase)
    print(f"\nRecords with NULL context: {total_null:,}")

    if total_null == 0:
        print("All records have context. Nothing to do!")
        return

    # Determine how many to process
    to_process = min(total_null, args.limit) if args.limit else total_null
    print(f"Will process: {to_process:,} records")

    if args.dry_run:
        print("\n*** DRY RUN - No database updates ***")

    print("-"*70)

    # Process records
    start_time = time.time()
    success_count = 0
    fail_count = 0
    offset = 0
    samples = []

    with tqdm(total=to_process, desc="Deriving context") as pbar:
        while offset < to_process:
            # Fetch batch
            batch_limit = min(args.batch_size, to_process - offset)
            records = fetch_null_context_records(supabase, limit=batch_limit, offset=offset)

            if not records:
                break

            for record in records:
                # Derive context from teacher output
                context = derive_context_from_teacher(
                    record.get("input", ""),
                    record.get("sevenb", ""),
                    record.get("task_label", "general")
                )

                if context:
                    if args.dry_run or args.show_samples:
                        if len(samples) < 5:
                            samples.append({
                                "id": record["id"],
                                "instruction": record.get("input", "")[:80],
                                "teacher": record.get("sevenb", "")[:100],
                                "label": record.get("task_label", ""),
                                "context": context[:150]
                            })
                        if not args.dry_run:
                            if update_context(supabase, record["id"], context):
                                success_count += 1
                            else:
                                fail_count += 1
                        else:
                            success_count += 1
                    elif update_context(supabase, record["id"], context):
                        success_count += 1
                    else:
                        fail_count += 1
                else:
                    fail_count += 1

                pbar.update(1)
                offset += 1

                if offset >= to_process:
                    break

    elapsed = time.time() - start_time

    # Show samples if requested
    if samples and (args.dry_run or args.show_samples):
        print("\n" + "="*70)
        print("SAMPLE OUTPUTS")
        print("="*70)
        for i, s in enumerate(samples, 1):
            print(f"\n--- Sample {i} (ID: {s['id']}, Label: {s['label']}) ---")
            print(f"Instruction: {s['instruction']}...")
            print(f"Teacher: {s['teacher']}...")
            print(f"Derived Context: {s['context']}...")

    print("\n" + "="*70)
    print("CONTEXT GENERATION COMPLETE")
    print("="*70)
    print(f"Success: {success_count:,}")
    print(f"Failed:  {fail_count:,}")
    print(f"Time:    {elapsed/60:.1f} minutes")
    if elapsed > 0:
        print(f"Rate:    {success_count/elapsed*60:.1f} records/min")
    print("="*70)

    remaining = total_null - success_count
    if remaining > 0:
        print(f"\nRemaining NULL context: {remaining:,}")
        print("Run again to continue processing.")


if __name__ == "__main__":
    main()
