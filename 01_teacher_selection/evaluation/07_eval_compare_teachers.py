"""
Step 08: Compare Alpaca vs OSS-20B Teachers for Selection
=========================================================
Compares tuned_alpaca and tuned_oss20b outputs to select the better teacher.

Workflow:
1. Fetch all labeled records from modelComp
2. Evaluate both tuned models using the evaluation matrix
3. Aggregate scores by category (6 labels)
4. Generate comparison report
5. Update database with new metric columns

Usage:
    python experiment/08_compare_teachers.py
    python experiment/08_compare_teachers.py --dry-run  # Preview without DB updates
    python experiment/08_compare_teachers.py --limit 100  # Test with subset
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from supabase import create_client, Client
from tqdm import tqdm

# Import our evaluation metrics
# Note: Python doesn't allow module names starting with numbers, so we import dynamically
import importlib.util
spec = importlib.util.spec_from_file_location(
    "eval_metrics", 
    PROJECT_ROOT / "02_distillation" / "evaluation_suite" / "06_eval_metrics.py"
)
eval_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_metrics)

evaluate_single_output = eval_metrics.evaluate_single_output
compare_teacher_student = eval_metrics.compare_teacher_student
ensure_string = eval_metrics.ensure_string

load_dotenv()

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def get_supabase_client() -> Client:
    """Create Supabase client"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    return create_client(url, key)


def fetch_labeled_records(
    supabase: Client,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Fetch records that have:
    - label (task category)
    - tuned_alpaca output
    - tuned_oss20b output
    """
    print("Fetching labeled records from modelComp...")
    
    all_records = []
    page_size = 1000
    offset = 0
    
    while True:
        query = supabase.table("modelComp")\
            .select("*")\
            .not_.is_("label", "null")\
            .not_.is_("tuned_alpaca", "null")\
            .not_.is_("tuned_oss20b", "null")\
            .order("id")\
            .range(offset, offset + page_size - 1)
        
        response = query.execute()
        
        if not response.data:
            break
        
        all_records.extend(response.data)
        print(f"  Fetched {len(all_records)} records...")
        
        if len(response.data) < page_size:
            break
        
        if limit and len(all_records) >= limit:
            break
        
        offset += page_size
    
    if limit:
        all_records = all_records[:limit]
    
    print(f"✓ Total: {len(all_records)} records with both teacher outputs\n")
    return all_records


def to_int8(value: float) -> int:
    """Convert 0-1 float to INT8 (multiply by 10000)"""
    if value is None:
        return 0
    return int(round(value * 10000))


def update_record_metrics(
    supabase: Client,
    record_id: int,
    alpaca_metrics: Dict[str, float],
    oss_metrics: Dict[str, float],
    winner: str = "tie",
    dry_run: bool = False
) -> bool:
    """Update a record with new evaluation metrics"""
    if dry_run:
        return True
    
    try:
        update_data = {
            # Alpaca metrics
            "alpaca_struct_correct": to_int8(alpaca_metrics.get("structured_correctness", 0)),
            "alpaca_task_success": to_int8(alpaca_metrics.get("task_success", 0)),
            "alpaca_instr_follow": to_int8(alpaca_metrics.get("instruction_following", 0)),
            "alpaca_coverage": to_int8(alpaca_metrics.get("coverage", 0)),
            "alpaca_faithfulness": to_int8(alpaca_metrics.get("faithfulness", 0)),
            "alpaca_hallucination": to_int8(alpaca_metrics.get("hallucination", 0)),
            "alpaca_ctx_grounding": to_int8(alpaca_metrics.get("context_grounding", 0)),
            "alpaca_overall": to_int8(alpaca_metrics.get("overall_score", 0)),
            
            # OSS-20B metrics
            "oss_struct_correct": to_int8(oss_metrics.get("structured_correctness", 0)),
            "oss_task_success": to_int8(oss_metrics.get("task_success", 0)),
            "oss_instr_follow": to_int8(oss_metrics.get("instruction_following", 0)),
            "oss_coverage": to_int8(oss_metrics.get("coverage", 0)),
            "oss_faithfulness": to_int8(oss_metrics.get("faithfulness", 0)),
            "oss_hallucination": to_int8(oss_metrics.get("hallucination", 0)),
            "oss_ctx_grounding": to_int8(oss_metrics.get("context_grounding", 0)),
            "oss_overall": to_int8(oss_metrics.get("overall_score", 0)),
            
            # Status and winner
            "eval_status": "completed",
            "eval_winner": winner,
        }
        
        supabase.table("modelComp").update(update_data).eq("id", record_id).execute()
        return True
        
    except Exception as e:
        print(f"  Error updating record {record_id}: {e}")
        return False


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate both tuned_alpaca and tuned_oss20b for a single record.
    Returns metrics for both.
    """
    # Extract fields
    instruction = ensure_string(record.get("input", ""))
    context_raw = record.get("context", [])
    context = ensure_string(context_raw) if context_raw else ""
    # Try both column names (actual_output or output)
    reference = ensure_string(record.get("actual_output", "") or record.get("output", ""))
    task_label = record.get("label", "general_qa")
    
    alpaca_output = ensure_string(record.get("tuned_alpaca", ""))
    oss_output = ensure_string(record.get("tuned_oss20b", ""))
    
    # Evaluate Alpaca output
    alpaca_metrics = evaluate_single_output(
        instruction=instruction,
        student_output=alpaca_output,
        teacher_output=reference,  # Compare to original reference
        context=context,
        task_label=task_label
    )
    
    # Evaluate OSS-20B output
    oss_metrics = evaluate_single_output(
        instruction=instruction,
        student_output=oss_output,
        teacher_output=reference,  # Compare to original reference
        context=context,
        task_label=task_label
    )
    
    return {
        "id": record.get("id"),
        "label": task_label,
        "has_context": bool(context and context.strip()),  # Track context availability
        "alpaca": alpaca_metrics,
        "oss": oss_metrics,
        "alpaca_wins": alpaca_metrics["overall_score"] > oss_metrics["overall_score"],
        "difference": alpaca_metrics["overall_score"] - oss_metrics["overall_score"],
    }


def aggregate_by_category(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate results by task category (label).
    """
    category_stats = defaultdict(lambda: {
        "count": 0,
        "alpaca_wins": 0,
        "oss_wins": 0,
        "ties": 0,
        "alpaca_scores": defaultdict(list),
        "oss_scores": defaultdict(list),
    })
    
    metrics = [
        "structured_correctness", "task_success", "instruction_following",
        "coverage", "faithfulness", "hallucination", "context_grounding", "overall_score"
    ]
    
    for result in results:
        cat = result["label"]
        category_stats[cat]["count"] += 1
        
        # Who wins?
        if result["alpaca_wins"]:
            category_stats[cat]["alpaca_wins"] += 1
        elif result["difference"] < 0:
            category_stats[cat]["oss_wins"] += 1
        else:
            category_stats[cat]["ties"] += 1
        
        # Collect scores
        for metric in metrics:
            category_stats[cat]["alpaca_scores"][metric].append(
                result["alpaca"].get(metric, 0)
            )
            category_stats[cat]["oss_scores"][metric].append(
                result["oss"].get(metric, 0)
            )
    
    # Calculate averages
    aggregated = {}
    for cat, stats in category_stats.items():
        aggregated[cat] = {
            "count": stats["count"],
            "alpaca_wins": stats["alpaca_wins"],
            "oss_wins": stats["oss_wins"],
            "ties": stats["ties"],
            "alpaca_win_rate": stats["alpaca_wins"] / stats["count"] if stats["count"] > 0 else 0,
            "oss_win_rate": stats["oss_wins"] / stats["count"] if stats["count"] > 0 else 0,
            "alpaca_avg": {},
            "oss_avg": {},
        }
        
        for metric in metrics:
            alpaca_vals = stats["alpaca_scores"][metric]
            oss_vals = stats["oss_scores"][metric]
            
            aggregated[cat]["alpaca_avg"][metric] = (
                sum(alpaca_vals) / len(alpaca_vals) if alpaca_vals else 0
            )
            aggregated[cat]["oss_avg"][metric] = (
                sum(oss_vals) / len(oss_vals) if oss_vals else 0
            )
    
    return aggregated


def calculate_overall_winner(aggregated: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Determine overall winner across all categories.
    """
    total_count = 0
    total_alpaca_wins = 0
    total_oss_wins = 0
    total_ties = 0
    
    alpaca_overall_sum = 0
    oss_overall_sum = 0
    
    for cat, stats in aggregated.items():
        total_count += stats["count"]
        total_alpaca_wins += stats["alpaca_wins"]
        total_oss_wins += stats["oss_wins"]
        total_ties += stats["ties"]
        
        alpaca_overall_sum += stats["alpaca_avg"]["overall_score"] * stats["count"]
        oss_overall_sum += stats["oss_avg"]["overall_score"] * stats["count"]
    
    alpaca_weighted_avg = alpaca_overall_sum / total_count if total_count > 0 else 0
    oss_weighted_avg = oss_overall_sum / total_count if total_count > 0 else 0
    
    if alpaca_weighted_avg > oss_weighted_avg:
        winner = "tuned_alpaca"
    elif oss_weighted_avg > alpaca_weighted_avg:
        winner = "tuned_oss20b"
    else:
        winner = "tie"
    
    return {
        "total_records": total_count,
        "alpaca_wins": total_alpaca_wins,
        "oss_wins": total_oss_wins,
        "ties": total_ties,
        "alpaca_weighted_overall": round(alpaca_weighted_avg, 4),
        "oss_weighted_overall": round(oss_weighted_avg, 4),
        "winner": winner,
        "margin": abs(alpaca_weighted_avg - oss_weighted_avg),
    }


def aggregate_by_context(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Split results by context availability (with_context vs without_context).
    Returns aggregated stats for each group.
    """
    with_context = [r for r in results if r.get("has_context", False)]
    without_context = [r for r in results if not r.get("has_context", False)]
    
    def compute_stats(subset: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
        if not subset:
            return {"name": name, "count": 0}
        
        alpaca_wins = sum(1 for r in subset if r["alpaca_wins"])
        oss_wins = sum(1 for r in subset if r["difference"] < 0)
        ties = len(subset) - alpaca_wins - oss_wins
        
        alpaca_overall = sum(r["alpaca"]["overall_score"] for r in subset) / len(subset)
        oss_overall = sum(r["oss"]["overall_score"] for r in subset) / len(subset)
        
        # Context-specific metrics
        alpaca_ctx_ground = sum(r["alpaca"].get("context_grounding", 0) for r in subset) / len(subset)
        oss_ctx_ground = sum(r["oss"].get("context_grounding", 0) for r in subset) / len(subset)
        alpaca_halluc = sum(r["alpaca"].get("hallucination", 0) for r in subset) / len(subset)
        oss_halluc = sum(r["oss"].get("hallucination", 0) for r in subset) / len(subset)
        alpaca_faith = sum(r["alpaca"].get("faithfulness", 0) for r in subset) / len(subset)
        oss_faith = sum(r["oss"].get("faithfulness", 0) for r in subset) / len(subset)
        
        winner = "alpaca" if alpaca_overall > oss_overall else ("oss" if oss_overall > alpaca_overall else "tie")
        
        return {
            "name": name,
            "count": len(subset),
            "alpaca_wins": alpaca_wins,
            "oss_wins": oss_wins,
            "ties": ties,
            "alpaca_overall": round(alpaca_overall, 4),
            "oss_overall": round(oss_overall, 4),
            "alpaca_ctx_grounding": round(alpaca_ctx_ground, 4),
            "oss_ctx_grounding": round(oss_ctx_ground, 4),
            "alpaca_hallucination": round(alpaca_halluc, 4),
            "oss_hallucination": round(oss_halluc, 4),
            "alpaca_faithfulness": round(alpaca_faith, 4),
            "oss_faithfulness": round(oss_faith, 4),
            "winner": winner,
        }
    
    return {
        "with_context": compute_stats(with_context, "With Context"),
        "without_context": compute_stats(without_context, "Without Context"),
    }


# ============================================================================
# REPORT GENERATION
# ============================================================================

def print_comparison_report(
    aggregated: Dict[str, Dict[str, Any]],
    overall: Dict[str, Any],
    context_split: Dict[str, Dict[str, Any]] = None
):
    """Print formatted comparison report"""
    print("\n" + "=" * 80)
    print("TEACHER COMPARISON REPORT: ALPACA vs OSS-20B")
    print("=" * 80)
    
    print(f"\nTotal Records Evaluated: {overall['total_records']}")
    print("-" * 40)
    
    # Overall results
    print("\n📊 OVERALL RESULTS")
    print("-" * 40)
    print(f"  Alpaca Wins:     {overall['alpaca_wins']:>6} ({overall['alpaca_wins']/overall['total_records']*100:.1f}%)")
    print(f"  OSS-20B Wins:    {overall['oss_wins']:>6} ({overall['oss_wins']/overall['total_records']*100:.1f}%)")
    print(f"  Ties:            {overall['ties']:>6} ({overall['ties']/overall['total_records']*100:.1f}%)")
    print(f"\n  Alpaca Weighted Overall: {overall['alpaca_weighted_overall']:.4f}")
    print(f"  OSS-20B Weighted Overall: {overall['oss_weighted_overall']:.4f}")
    print(f"\n  🏆 WINNER: {overall['winner'].upper()} (margin: {overall['margin']:.4f})")
    
    # Per-category breakdown
    print("\n\n📂 RESULTS BY CATEGORY")
    print("-" * 80)
    
    categories = sorted(aggregated.keys())
    
    # Header
    print(f"{'Category':<25} {'Count':>6} {'Alp Wins':>9} {'OSS Wins':>9} {'Alp Avg':>8} {'OSS Avg':>8} {'Better':>10}")
    print("-" * 80)
    
    for cat in categories:
        stats = aggregated[cat]
        alpaca_avg = stats["alpaca_avg"]["overall_score"]
        oss_avg = stats["oss_avg"]["overall_score"]
        better = "ALPACA" if alpaca_avg > oss_avg else ("OSS" if oss_avg > alpaca_avg else "TIE")
        
        print(f"{cat:<25} {stats['count']:>6} {stats['alpaca_wins']:>9} {stats['oss_wins']:>9} {alpaca_avg:>8.4f} {oss_avg:>8.4f} {better:>10}")
    
    # Detailed metrics comparison
    print("\n\n📈 DETAILED METRIC COMPARISON (WEIGHTED AVERAGES)")
    print("-" * 80)
    
    metrics = [
        ("structured_correctness", "Struct. Correct"),
        ("task_success", "Task Success"),
        ("instruction_following", "Instr. Follow"),
        ("coverage", "Coverage"),
        ("faithfulness", "Faithfulness"),
        ("hallucination", "Hallucination ↓"),
        ("context_grounding", "Ctx Grounding"),
        ("overall_score", "OVERALL"),
    ]
    
    print(f"{'Metric':<20} {'Alpaca':>10} {'OSS-20B':>10} {'Diff':>10} {'Better':>10}")
    print("-" * 60)
    
    for metric_key, metric_name in metrics:
        # Calculate weighted averages across all categories
        alpaca_sum = 0
        oss_sum = 0
        total = 0
        
        for cat, stats in aggregated.items():
            alpaca_sum += stats["alpaca_avg"][metric_key] * stats["count"]
            oss_sum += stats["oss_avg"][metric_key] * stats["count"]
            total += stats["count"]
        
        alpaca_avg = alpaca_sum / total if total > 0 else 0
        oss_avg = oss_sum / total if total > 0 else 0
        diff = alpaca_avg - oss_avg
        
        # For hallucination, lower is better
        if metric_key == "hallucination":
            better = "ALPACA" if diff < 0 else ("OSS" if diff > 0 else "TIE")
        else:
            better = "ALPACA" if diff > 0 else ("OSS" if diff < 0 else "TIE")
        
        print(f"{metric_name:<20} {alpaca_avg:>10.4f} {oss_avg:>10.4f} {diff:>+10.4f} {better:>10}")
    
    # Context split analysis
    if context_split:
        print("\n\n📋 CONTEXT ANALYSIS: WITH vs WITHOUT CONTEXT")
        print("=" * 80)
        
        for key in ["with_context", "without_context"]:
            stats = context_split.get(key, {})
            if stats.get("count", 0) == 0:
                continue
            
            print(f"\n🔹 {stats['name'].upper()} ({stats['count']} records)")
            print("-" * 50)
            print(f"  Alpaca Wins: {stats['alpaca_wins']} | OSS Wins: {stats['oss_wins']} | Ties: {stats['ties']}")
            print(f"  Alpaca Overall: {stats['alpaca_overall']:.4f}")
            print(f"  OSS-20B Overall: {stats['oss_overall']:.4f}")
            print(f"  🏆 Winner: {stats['winner'].upper()}")
            
            if key == "with_context":
                print(f"\n  Context-Specific Metrics:")
                print(f"    Context Grounding: Alpaca={stats['alpaca_ctx_grounding']:.4f} | OSS={stats['oss_ctx_grounding']:.4f}")
                print(f"    Hallucination:     Alpaca={stats['alpaca_hallucination']:.4f} | OSS={stats['oss_hallucination']:.4f}")
                print(f"    Faithfulness:      Alpaca={stats['alpaca_faithfulness']:.4f} | OSS={stats['oss_faithfulness']:.4f}")
    
    print("\n" + "=" * 80)


def save_report_json(
    results: List[Dict[str, Any]],
    aggregated: Dict[str, Dict[str, Any]],
    overall: Dict[str, Any],
    context_split: Dict[str, Dict[str, Any]],
    output_path: str
):
    """Save detailed report as JSON"""
    report = {
        "generated_at": datetime.now().isoformat(),
        "overall": overall,
        "context_split": context_split,
        "by_category": {
            cat: {
                "count": stats["count"],
                "alpaca_wins": stats["alpaca_wins"],
                "oss_wins": stats["oss_wins"],
                "ties": stats["ties"],
                "alpaca_avg_scores": {k: round(v, 4) for k, v in stats["alpaca_avg"].items()},
                "oss_avg_scores": {k: round(v, 4) for k, v in stats["oss_avg"].items()},
            }
            for cat, stats in aggregated.items()
        },
        "sample_results": results[:50],  # Save first 50 for inspection
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Detailed report saved to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare Alpaca vs OSS-20B teachers")
    parser.add_argument("--dry-run", action="store_true", help="Preview without DB updates")
    parser.add_argument("--limit", type=int, help="Limit number of records to process")
    parser.add_argument("--no-update", action="store_true", help="Skip database updates")
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("TEACHER COMPARISON: ALPACA vs OSS-20B")
    print("=" * 80)
    
    if args.dry_run:
        print("⚠️  DRY RUN MODE - No database updates will be made")
    
    # Connect to database
    supabase = get_supabase_client()
    
    # Fetch records
    records = fetch_labeled_records(supabase, limit=args.limit)
    
    if not records:
        print("❌ No records found with both teacher outputs")
        return
    
    # Evaluate all records
    print("Evaluating records...")
    results = []
    errors = 0
    
    for record in tqdm(records, desc="Evaluating"):
        try:
            result = evaluate_record(record)
            results.append(result)
            
            # Determine winner for this record
            if result["alpaca_wins"]:
                winner = "alpaca"
            elif result["difference"] < 0:
                winner = "oss"
            else:
                winner = "tie"
            
            # Update database (if not dry-run)
            if not args.dry_run and not args.no_update:
                update_record_metrics(
                    supabase,
                    record["id"],
                    result["alpaca"],
                    result["oss"],
                    winner=winner,
                    dry_run=False
                )
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\n  Error on record {record.get('id')}: {e}")
    
    if errors > 0:
        print(f"\n⚠️  {errors} errors during evaluation")
    
    # Aggregate results
    print("\nAggregating results by category...")
    aggregated = aggregate_by_category(results)
    overall = calculate_overall_winner(aggregated)
    
    # Aggregate by context
    print("Aggregating results by context availability...")
    context_split = aggregate_by_context(results)
    
    # Print report
    print_comparison_report(aggregated, overall, context_split)
    
    # Save JSON report
    report_path = PROJECT_ROOT / "reports" / "teacher_comparison_report.json"
    report_path.parent.mkdir(exist_ok=True)
    save_report_json(results, aggregated, overall, context_split, str(report_path))
    
    # Summary
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    
    if overall["winner"] == "tuned_alpaca":
        print("""
✅ SELECT: tuned_alpaca as the TEACHER for 50K Gemma 3 fine-tuning

Reasons:
- Higher weighted overall score across categories
- Use tuned_alpaca outputs to generate the full training dataset
""")
    elif overall["winner"] == "tuned_oss20b":
        print("""
✅ SELECT: tuned_oss20b as the TEACHER for 50K Gemma 3 fine-tuning

Reasons:
- Higher weighted overall score across categories
- Use tuned_oss20b outputs to generate the full training dataset
""")
    else:
        print("""
⚖️  TIE: Both teachers perform similarly

Recommendation:
- Review per-category performance
- Consider which performs better on your priority categories
""")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
