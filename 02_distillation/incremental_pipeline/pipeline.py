#!/usr/bin/env python3
"""
UNIFIED PIPELINE - Incremental and Batch Learning for Research
===============================================================
Central controller for the INTUNE self-improving LLM framework.

Supports:
1. Context generation (derived from teacher output, no LLM)
2. Base student output generation
3. Scoring (7 metrics + ROUGE/BLEU)
4. Fine-tuning (Incremental by checkpoint OR Batch all at once)
5. Tuned output generation
6. Tuned output scoring
7. Learning curve analysis

Status Flow:
  NULL -> (generate outputs) -> ready -> score -> finetune -> output_tuned -> score_tuned -> completed

Usage:
    # Check overall status
    python experiment/pipeline.py --status

    # Step-by-step workflow:

    # 1. Generate context derived from teacher output (no LLM, fast)
    python experiment/pipeline.py --context

    # 2. Generate base student outputs (for records missing student_output)
    python experiment/pipeline.py --generate --checkpoint 1

    # 3. Score available student outputs
    python experiment/pipeline.py --score --checkpoint 1

    # 4a. INCREMENTAL: Fine-tune checkpoint by checkpoint
    python experiment/pipeline.py --mode incremental --checkpoint 1 --finetune
    python experiment/pipeline.py --mode incremental --checkpoint 1 --output-tuned
    python experiment/pipeline.py --mode incremental --checkpoint 1 --score-tuned

    # 4b. BATCH: Fine-tune on all scored data at once
    python experiment/pipeline.py --mode batch --finetune
    python experiment/pipeline.py --mode batch --output-tuned
    python experiment/pipeline.py --mode batch --score-tuned

    # Run all steps for a checkpoint
    python experiment/pipeline.py --mode incremental --checkpoint 1 --run-all
    python experiment/pipeline.py --mode batch --run-all

Research Paper Workflow:
    1. Context: Derived from teacher (reproducible, no LLM dependency)
    2. Metrics: Model-agnostic (no LLM judge)
    3. Comparison: Incremental vs Batch learning curves
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# Config
PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENT_DIR = Path(__file__).parent
RECORDS_PER_CHECKPOINT = 5000
MIN_RECORDS = 4900


def get_supabase():
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


def get_database_status():
    """Get comprehensive database status"""
    supabase = get_supabase()
    status = {}

    # Total records
    result = supabase.table("modelcomp_50k").select("id", count="exact").execute()
    status["total"] = result.count

    # NULL context
    result = supabase.table("modelcomp_50k").select("id", count="exact").is_("context", "null").execute()
    status["null_context"] = result.count

    # Has student_output (non-null and non-empty)
    result = supabase.table("modelcomp_50k").select("id", count="exact")\
        .neq("student_output", "").not_.is_("student_output", "null").execute()
    status["has_student_output"] = result.count

    # NULL student_output
    result = supabase.table("modelcomp_50k").select("id", count="exact").is_("student_output", "null").execute()
    status["null_student_output"] = result.count

    # Status breakdown
    for s in ['score', 'finetune', 'output_tuned', 'score_tuned', 'completed']:
        result = supabase.table("modelcomp_50k").select("id", count="exact").eq("status", s).execute()
        status[f"status_{s}"] = result.count

    # NULL status with student_output (ready to score)
    result = supabase.table("modelcomp_50k").select("id", count="exact")\
        .is_("status", "null").neq("student_output", "").not_.is_("student_output", "null").execute()
    status["ready_to_score"] = result.count

    # Checkpoint breakdown
    for cp in range(1, 11):
        result = supabase.table("modelcomp_50k").select("id", count="exact").eq("checkpoint", cp).execute()
        status[f"checkpoint_{cp}"] = result.count

    return status


def print_status():
    """Print formatted status report"""
    status = get_database_status()

    print("\n" + "="*70)
    print("INTUNE PIPELINE STATUS")
    print("="*70)

    print(f"\nDATABASE OVERVIEW:")
    print(f"  Total records:        {status['total']:,}")
    print(f"  NULL context:         {status['null_context']:,}")
    print(f"  Has student_output:   {status['has_student_output']:,}")
    print(f"  NULL student_output:  {status['null_student_output']:,}")

    print(f"\nWORKFLOW STATUS:")
    print(f"  Ready to score:       {status['ready_to_score']:,}")
    print(f"  status='score':       {status['status_score']:,}")
    print(f"  status='finetune':    {status['status_finetune']:,}")
    print(f"  status='output_tuned': {status['status_output_tuned']:,}")
    print(f"  status='score_tuned': {status['status_score_tuned']:,}")
    print(f"  status='completed':   {status['status_completed']:,}")

    print(f"\nCHECKPOINT BREAKDOWN:")
    for cp in range(1, 11):
        count = status.get(f"checkpoint_{cp}", 0)
        bar = "#" * (count // 500) if count > 0 else ""
        print(f"  Checkpoint {cp:2d}: {count:,} {bar}")

    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)

    if status['null_context'] > 0:
        print(f"\n1. GENERATE CONTEXT ({status['null_context']:,} records):")
        print(f"   python experiment/pipeline.py --context")

    if status['null_student_output'] > 0:
        print(f"\n2. GENERATE STUDENT OUTPUTS ({status['null_student_output']:,} records):")
        print(f"   python experiment/pipeline.py --generate --checkpoint <N>")

    if status['ready_to_score'] > 0:
        print(f"\n3. SCORE STUDENT OUTPUTS ({status['ready_to_score']:,} records):")
        print(f"   python experiment/pipeline.py --score")

    if status['status_score'] >= MIN_RECORDS:
        print(f"\n4. FINE-TUNE ({status['status_score']:,} scored records ready):")
        print(f"   INCREMENTAL: python experiment/pipeline.py --mode incremental --checkpoint 1 --run-all")
        print(f"   BATCH:       python experiment/pipeline.py --mode batch --run-all")

    print("\n" + "="*70)


def run_context_generation(limit=None):
    """Run context generation (derived from teacher output, no LLM)"""
    script = EXPERIMENT_DIR / "data_processing" / "08_gen_context_ollama.py"
    cmd = [sys.executable, str(script)]
    if limit:
        cmd.extend(["--limit", str(limit)])

    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_student_generation(checkpoint):
    """Generate base student outputs"""
    script = EXPERIMENT_DIR / "phase2_incremental" / "11_gen_base_student.py"
    cmd = [sys.executable, str(script), "--checkpoint", str(checkpoint)]

    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_scoring(checkpoint=None):
    """Score student outputs"""
    script = EXPERIMENT_DIR / "phase2_incremental" / "12_train_incremental.py"

    if checkpoint:
        cmd = [sys.executable, str(script), "--checkpoint", str(checkpoint), "--step", "score"]
    else:
        # Score all available
        cmd = [sys.executable, str(script), "--checkpoint", "1", "--step", "score"]

    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_incremental(checkpoint, step=None, run_all=False):
    """Run incremental training"""
    script = EXPERIMENT_DIR / "phase2_incremental" / "12_train_incremental.py"

    if run_all:
        cmd = [sys.executable, str(script), "--checkpoint", str(checkpoint), "--run-all"]
    elif step:
        cmd = [sys.executable, str(script), "--checkpoint", str(checkpoint), "--step", step]
    else:
        cmd = [sys.executable, str(script), "--checkpoint", str(checkpoint), "--step", "status"]

    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_batch(step=None, run_all=False):
    """Run batch training (all scored data at once)"""
    # For batch mode, we use checkpoint=0 or a special flag
    # This needs modification in 12_train_incremental.py to support batch mode

    status = get_database_status()
    total_scored = status['status_score']

    print(f"\nBATCH MODE: Training on {total_scored:,} scored records")
    print("Note: Batch mode uses all scored data regardless of checkpoint")

    script = EXPERIMENT_DIR / "phase2_incremental" / "12_train_incremental.py"

    # For now, batch mode is implemented as training on checkpoint 0 (all data)
    # The actual implementation would need to modify 12_train_incremental.py
    if run_all:
        cmd = [sys.executable, str(script), "--checkpoint", "0", "--run-all"]
    elif step:
        cmd = [sys.executable, str(script), "--checkpoint", "0", "--step", step]
    else:
        cmd = [sys.executable, str(script), "--checkpoint", "0", "--step", "status"]

    print(f"\nWould run: {' '.join(cmd)}")
    print("TODO: Implement batch mode in 12_train_incremental.py")


def main():
    parser = argparse.ArgumentParser(
        description='INTUNE Unified Pipeline - Incremental and Batch Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check status
  python experiment/pipeline.py --status

  # Context generation (derived from teacher, no LLM)
  python experiment/pipeline.py --context

  # Generate student outputs for checkpoint 1
  python experiment/pipeline.py --generate --checkpoint 1

  # Score student outputs
  python experiment/pipeline.py --score --checkpoint 1

  # Incremental training
  python experiment/pipeline.py --mode incremental --checkpoint 1 --run-all

  # Batch training (all data)
  python experiment/pipeline.py --mode batch --run-all
"""
    )

    # Status
    parser.add_argument('--status', action='store_true', help='Show pipeline status')

    # Context generation
    parser.add_argument('--context', action='store_true', help='Generate context from teacher output')

    # Student output generation
    parser.add_argument('--generate', action='store_true', help='Generate base student outputs')

    # Scoring
    parser.add_argument('--score', action='store_true', help='Score student outputs')

    # Training mode
    parser.add_argument('--mode', choices=['incremental', 'batch'], help='Training mode')
    parser.add_argument('--checkpoint', type=int, help='Checkpoint number (1-10)')

    # Training steps
    parser.add_argument('--finetune', action='store_true', help='Run fine-tuning step')
    parser.add_argument('--output-tuned', action='store_true', help='Generate tuned outputs')
    parser.add_argument('--score-tuned', action='store_true', help='Score tuned outputs')
    parser.add_argument('--run-all', action='store_true', help='Run all training steps')

    # Limits
    parser.add_argument('--limit', type=int, help='Limit records to process')

    args = parser.parse_args()

    # Default to status if no args
    if len(sys.argv) == 1:
        args.status = True

    if args.status:
        print_status()
        return

    if args.context:
        run_context_generation(limit=args.limit)
        return

    if args.generate:
        if not args.checkpoint:
            print("Error: --checkpoint required for --generate")
            return
        run_student_generation(args.checkpoint)
        return

    if args.score:
        run_scoring(checkpoint=args.checkpoint)
        return

    if args.mode == 'incremental':
        if not args.checkpoint:
            print("Error: --checkpoint required for incremental mode")
            return

        if args.run_all:
            run_incremental(args.checkpoint, run_all=True)
        elif args.finetune:
            run_incremental(args.checkpoint, step='finetune')
        elif args.output_tuned:
            run_incremental(args.checkpoint, step='output_tuned')
        elif args.score_tuned:
            run_incremental(args.checkpoint, step='score_tuned')
        else:
            run_incremental(args.checkpoint)
        return

    if args.mode == 'batch':
        if args.run_all:
            run_batch(run_all=True)
        elif args.finetune:
            run_batch(step='finetune')
        elif args.output_tuned:
            run_batch(step='output_tuned')
        elif args.score_tuned:
            run_batch(step='score_tuned')
        else:
            run_batch()
        return

    # Show help if nothing matched
    parser.print_help()


if __name__ == "__main__":
    main()
